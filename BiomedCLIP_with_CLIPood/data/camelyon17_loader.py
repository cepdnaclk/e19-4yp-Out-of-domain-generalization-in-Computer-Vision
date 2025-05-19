from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

def get_dataloaders(batch_size):
    dataset = get_dataset(dataset="camelyon17", download=True)
    train_loader = get_train_loader("standard", dataset, batch_size=batch_size)
    val_loader = get_eval_loader("standard", dataset, batch_size=batch_size)
    return train_loader, val_loader
