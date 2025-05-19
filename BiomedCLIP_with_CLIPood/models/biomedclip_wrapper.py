import torch
from open_clip import create_model_and_transforms, get_tokenizer

class BiomedCLIP(torch.nn.Module):
    def __init__(self, model_name="hf-hub:mbzuai/biomedclip-vit-base-patch16"):
        super().__init__()
        self.model, _, _ = create_model_and_transforms(model_name, pretrained=True)
        self.tokenizer = get_tokenizer(model_name)

    def encode_image(self, x):
        return self.model.encode_image(x)

    def encode_text(self, texts):
        tokens = self.tokenizer(texts)
        return self.model.encode_text(tokens)
