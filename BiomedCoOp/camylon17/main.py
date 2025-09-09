import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data import get_dataloaders
from biomed_coop import BiomedCoOp


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    train_loader, val_loader, test_loader, classnames = get_dataloaders(
        args.data_path, args.metadata_path, batch_size=args.batch_size, num_workers=4
    )

    # Model
    model = BiomedCoOp(classnames, device=device).to(device)

    # Only optimize prompts
    optimizer = optim.AdamW(model.prompt_learner.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} | Loss {total_loss/len(train_loader):.4f}")

        # Validation
        acc = evaluate(model, val_loader, device)
        print(f"Val Acc: {acc:.4f}")

    # Final test
    test_acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    train(args)
