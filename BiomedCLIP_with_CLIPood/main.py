import torch
from models.biomedclip_wrapper import BiomedCLIP
from models.bma_utils import init_bma_model
from utils.eval import evaluate
from train import train
from data.camelyon17_loader import get_dataloaders
from config import config  



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data loaders
    train_loader, val_loader, _, _ = get_dataloaders(
        metadata_path=config["metadata_path"],
        data_root=config["data_root"],
        batch_size=config["batch_size"]
    )

    # Initialize BiomedCLIP model
    model = BiomedCLIP().to(device)

    # Define text labels and encode them
    text_labels = ["tumor", "normal"]
    text_features = model.encode_text(text_labels).to(device)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Initialize BMA wrapper
    bma_model = init_bma_model(model)

    # Train the model
    train(model, bma_model, train_loader, text_features, config, device)

    # Evaluate on validation set
    acc = evaluate(bma_model, val_loader, text_features, device)
    print(f"Final Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
