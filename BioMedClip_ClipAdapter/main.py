import os
import time

from utils import *
from open_clip import create_model_from_pretrained
from clip_adapter_biomedclip import ClipAdapter_BiomedCLIP

from data import get_dataloaders


def main():

   
    cfg = {
        'metadata_path': '/home/E19_FYP_Domain_Gen_Data/metadata.csv',
        'data_path': '/home/E19_FYP_Domain_Gen_Data/organized_by_center',
        'lr': 0.001,                   # Learning rate
        'train_epoch': 100,            # Number of training epochs
        'alpha_ca': 0.5,               # Initial alpha value for CLIP-Adapter
        # 'search_alpha_ca': False,      # Whether to search for best alpha
        'cache_dir': "./cache",        # Directory to save adapter checkpoints
        'WARMUP_EPOCH': 5,             # Warmup epochs for scheduler
        'WARMUP_CONS_LR': 1e-5,        # Constant LR during warmup
}

    cache_dir = "caches"
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    # Define cache path for dataloaders
    dataloader_cache_path = os.path.join(cfg['cache_dir'], 'dataloaders.pt')

    # Load or create dataloaders
    if os.path.exists(dataloader_cache_path):
        print("Loading dataloaders from cache...")
        train_loader, val_loader, test_loader = torch.load(dataloader_cache_path)
    else:
        print("Creating new dataloaders (first run)...")
        train_loader, val_loader, test_loader = get_dataloaders(
            metadata_path=cfg['metadata_path'],
            data_root=cfg['data_path']
        )
        torch.save((train_loader, val_loader, test_loader), dataloader_cache_path)
        print(f"Dataloaders cached to {dataloader_cache_path}")



    print("\nRunning configs.")
    print(cfg, "\n")

    method = ClipAdapter_BiomedCLIP(args=cfg)

    # Load the model and config files from the Hugging Face Hub
    clip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    clip_model = clip_model.cuda()
    clip_model.eval()

   
    # Textual features
    
    classnames = ["no tumor", "tumor present"]
    positive_prompt = "This is an image of a tumor"
    negative_prompt = "Tumor is not present in this image"

    text_weights = biomedclip_classifier(
    classnames=classnames,
    positive_prompt=positive_prompt,
    negative_prompt=negative_prompt,
    clip_model=clip_model
    )   
    

    # Pre-load test features
    f_test_time = time.time()
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(
        cfg, "test", clip_model, test_loader)

    total_acc = 0
    predictions = []
    
    

    loss, acc = method(train_loader=train_loader,
                    val_loader=val_loader,
                    test_features=test_features,
                    test_labels=test_labels,
                    text_weights=text_weights,
                    model=clip_model,
                    classnames=classnames)
    print(f'Final Accuracy {acc}')


if __name__ == '__main__':
    main()

