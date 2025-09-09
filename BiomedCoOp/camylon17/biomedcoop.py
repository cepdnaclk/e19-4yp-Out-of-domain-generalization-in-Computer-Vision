import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from trainers.prompt_templates import BIOMEDCOOP_TEMPLATES
from open_clip import create_model_from_pretrained, get_tokenizer


# ----------------------------
# Text encoder
# ----------------------------
class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dtype = model.text.transformer.dtype

    def forward(self, prompts, tokenized_prompts):
        return self.model.encode_text(prompts, True, tokenized_prompts)


# ----------------------------
# Prompt Learner
# ----------------------------
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDCOOP.N_CTX
        dtype = model.text.transformer.dtype
        ctx_dim = 768

        self.tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

        # random context init
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # prepare class prompts
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [("X " * n_ctx) + name + "." for name in classnames]
        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])

        self.tokenized_prompts = tokenized_prompts
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        # Pre-compute frozen embeddings
        teacher_model, _ = create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        teacher_model = teacher_model.eval().cuda().float()
        all_feats = []
        for i in range(cfg.TRAINER.BIOMEDCOOP.N_PROMPTS):
            x = torch.cat([self.tokenizer(BIOMEDCOOP_TEMPLATES[c][i]) for c in classnames])
            all_feats.append(teacher_model.encode_text(x.cuda()).unsqueeze(1))
        self.fixed_embeddings = torch.cat(all_feats, dim=1)
        self.ZS_image_encoder = teacher_model.visual

    def forward(self):
        return self.ctx


# ----------------------------
# Custom BiomedCoOp model
# ----------------------------
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, model)
        self.image_encoder = model.visual
        self.text_encoder = TextEncoder(model)
        self.logit_scale = model.logit_scale
        self.dtype = model.text.transformer.dtype

    def forward(self, image, label=None):
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = self.logit_scale.exp() * image_features @ text_features.t()

        if self.training:
            fixed = self.prompt_learner.fixed_embeddings
            fixed = fixed / fixed.norm(dim=-1, keepdim=True)

            loss_ce = F.cross_entropy(logits, label)
            loss_mse = F.mse_loss(text_features, fixed.mean(dim=1).cuda())
            return logits, loss_ce, loss_mse
        return logits


# ----------------------------
# Trainer
# ----------------------------
@TRAINER_REGISTRY.register()
class BiomedCoOpTrainer(TrainerX):
    def build_model(self):
        classnames = self.dm.dataset.classnames
        model, _ = create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        model.float()

        self.model = CustomCLIP(self.cfg, classnames, model.eval())
        self.model.to(self.device)

        self.optim = build_optimizer(self.model, self.cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if self.cfg.TRAINER.BIOMEDCOOP.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = batch["img"].to(self.device), batch["label"].to(self.device)
        logits, loss_ce, loss_mse = self.model(image, label)
        loss = loss_ce + self.cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA * loss_mse
        self.model_backward_and_update(loss)

        return {"loss": loss.item(), "acc": compute_accuracy(logits, label)[0].item()}
