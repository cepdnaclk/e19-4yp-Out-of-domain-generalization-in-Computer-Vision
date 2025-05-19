import torch
from models.margin_softmax import compute_class_margins, MarginMetricSoftmax
from models.bma_utils import update_bma_model
from tqdm import tqdm

def train(model, bma_model, train_loader, text_features, config, device):
    margins = compute_class_margins(text_features, config["alpha_margin"])
    criterion = MarginMetricSoftmax(margins)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    total_steps = len(train_loader) * config["epochs"]
    step = 0

    for epoch in range(config["epochs"]):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            image_features = model.encode_image(images)
            logits = image_features @ text_features.T
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_bma_model(model, bma_model, step, total_steps)
            step += 1
