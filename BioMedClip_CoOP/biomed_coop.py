import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score
from open_clip import create_model_from_pretrained, get_tokenizer

class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompts,tokenized_prompts):

        x = self.model.encode_text(prompts,True,tokenized_prompts)

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg,classnames, biomedclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = self.tokenizer(ctx_init)
            with torch.no_grad():
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:

            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])
        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg,classnames, biomedclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg,classnames, biomedclip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = biomedclip_model.visual
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts,tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class CoOpTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training parameters
        self.lr = args.get('lr', 0.002)
        self.epochs = args.get('train_epoch', 50)
        self.batch_size = args.get('batch_size', 32)
        
        # Precision settings
        self.precision = args.get('precision', 'amp')  # 'fp16', 'fp32', or 'amp'
        self.scaler = GradScaler() if self.precision == 'amp' else None
        
        # CoOp specific parameters
        self.n_ctx = args.get('n_ctx', 4)
        self.class_token_position = args.get('class_token_position', 'end')
        self.csc = args.get('csc', False)

    def build_model(self, classnames):
        """Initialize BiomedCLIP model with CoOp prompt learning"""
        print("Loading BiomedCLIP model...")
        model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
         # 1. Put BiomedCLIP in eval mode FIRST
        model = model.eval()  # <-- This freezes the base model


        print("Building custom CLIP with CoOp...")
        self.model = CustomCLIP(
            cfg={
                'TRAINER': {
                    'COOP': {
                        'N_CTX': self.n_ctx,
                        'CLASS_TOKEN_POSITION': self.class_token_position,
                        'CSC': self.csc
                    }
                },
                'classnames': classnames
            },
            classnames=classnames,
            biomedclip_model=model
        ).to(self.device)
        
        # 3. Explicitly freeze non-prompt-learner parameters
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # Only optimize prompt learner parameters
        self.optimizer = torch.optim.AdamW(
            [p for n, p in self.model.named_parameters() if 'prompt_learner' in n],
            lr=self.lr
        )

        # Alternatively
        # 4. Set up optimizer ONLY for prompt learner
        self.optimizer = torch.optim.AdamW(
            self.model.prompt_learner.parameters(),  # Explicitly only prompt learner
            lr=self.lr
        )

    def train(self, train_loader, val_loader, test_features, test_labels, text_weights):
        """Main training loop"""
        best_acc = 0.0
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            total_loss, total_correct, total_samples = 0, 0, 0
            
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass with automatic precision
                if self.precision == 'amp':
                    with autocast():
                        outputs = self.model(images)
                        loss = F.cross_entropy(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = F.cross_entropy(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                if self.precision == 'amp':
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                # Metrics
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
            
            train_acc = total_correct / total_samples * 100
            avg_loss = total_loss / len(train_loader)
            
            # Validation phase
            val_acc = self.evaluate(val_loader)
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_coop_model.pth')
        
        # Final evaluation
        self.model.load_state_dict(torch.load('best_coop_model.pth'))
        test_acc = self.evaluate_test(test_features, test_labels, text_weights)
        return best_acc, test_acc

    def evaluate(self, loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_correct, total_samples = 0, 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        
        return total_correct / total_samples * 100

    def evaluate_test(self, test_features, test_labels, text_weights):
        """Evaluate on test set with pre-extracted features"""
        self.model.eval()
        
        with torch.no_grad():
            # If using pre-computed features
            test_features = test_features.to(self.device)
            test_labels = test_labels.to(self.device)
            
            # Get text features (if needed)
            text_features = text_weights.to(self.device)
            
            # Normalize features
            image_features = test_features / test_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate logits
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            
            preds = logits.argmax(dim=1)
            acc = accuracy_score(test_labels.cpu().numpy(), preds.cpu().numpy()) * 100
            f1 = f1_score(test_labels.cpu().numpy(), preds.cpu().numpy()) * 100
            
            print(f"\nTest Accuracy: {acc:.2f}%")
            print(f"Test F1 Score: {f1:.2f}%")
            
        return acc