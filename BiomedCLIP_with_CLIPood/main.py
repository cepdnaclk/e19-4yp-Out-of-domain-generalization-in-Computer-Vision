import torch
from config import config
from data.camelyon17_loader import get_dataloaders
from models.biomedclip_wrapper import BiomedCLIP
from models.bma_utils import init_bma_model
from utils.eval import evaluate
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader = get_dataloaders(config["batch_size"])

model = BiomedCLIP().to(device)
text_labels = ["tumor", "normal"]
text_features = model.encode_text(text_labels).to(device)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

bma_model = init_bma_model(model)

train(model, bma_model, train_loader, text_features, config, device)

acc = evaluate(bma_model, val_loader, text_features, device)
print(f"Final Accuracy: {acc:.4f}")
