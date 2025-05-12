import os
import time

from utils import *
from open_clip import create_model_from_pretrained
from biomed_coop import CoOpTrainer

from data import get_dataloaders


def main():
    print("Starting CoOp-BiomedCLIP...")
   
    cfg = {
    'metadata_path': '/home/E19_FYP_Domain_Gen_Data/metadata.csv',
    'data_path': '/home/E19_FYP_Domain_Gen_Data/organized_by_center',
    'MODEL': {
        'BACKBONE': {
            'NAME': 'BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        },
        'INIT_WEIGHTS': None  # Path to pretrained weights if available
    },
    'INPUT': {
        'SIZE': (224, 224)  # Must match BiomedCLIP's expected input size
    },
    'TRAINER': {
        'COOP': {
            'PREC': "amp",  # Can be "fp16", "fp32", or "amp"
            'N_CTX': 4,     # Number of context tokens to learn
            'CTX_INIT': "", # Initial context words (e.g., "a photo of a")
            'CSC': False,   # Class-specific context (False for generic context)
            'CLASS_TOKEN_POSITION': "end"  # "end", "middle", or "front"
        }
    },
    'OPTIM': {
        'NAME': 'adamw',    # Optimizer name
        'LR': 0.002,        # Learning rate
        'MAX_EPOCH': 50,    # Total training epochs
        'WEIGHT_DECAY': 0.0005,  # Weight decay
        'LR_SCHEDULER': 'cosine',  # Learning rate scheduler
        'WARMUP_EPOCH': 5,  # Warmup epochs
        'WARMUP_TYPE': 'linear',
        'WARMUP_CONS_LR': True
    },
    'DATASET': {
        'CLASSNAMES':  ["no tumor", "tumor present"]  # Your binary class names
    }
    }

    cache_dir = "caches"
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    # # Define cache path for dataloaders
    # dataloader_cache_path = os.path.join(cfg['cache_dir'], 'dataloaders.pt')

    # # Load or create dataloaders
    # if os.path.exists(dataloader_cache_path):
    #     print("Loading dataloaders from cache...")
    #     train_loader, val_loader, test_loader = torch.load(dataloader_cache_path)
    # else:
    #     print("Creating new dataloaders (first run)...")
    train_loader, val_loader, test_loader, id_test_loaders= get_dataloaders(
        metadata_path=cfg['metadata_path'],
        data_root=cfg['data_path']
    )
        # torch.save((train_loader, val_loader, test_loader), dataloader_cache_path)
        # print(f"Dataloaders cached to {dataloader_cache_path}")



    print("\nRunning configs.")
    print(cfg, "\n")

    method = CoOpTrainer(args=cfg)

    # Load the model and config files from the Hugging Face Hub
    clip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    clip_model = clip_model.cuda()
    clip_model.eval()


    # Pre-load test features
    # f_test_time = time.time()
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(
        cfg, "test", clip_model, test_loader)
    
    # Pre-load features for all in-distribution test sets
    id_test_features = {}
    id_test_labels = {}

    for center_name, loader in id_test_loaders.items():
        features, labels = pre_load_features(cfg, center_name, clip_model, loader)
        id_test_features[center_name] = features
        id_test_labels[center_name] = labels

       
    
    

    loss, acc = method.train(train_loader=train_loader,
                    val_loader=val_loader,
                    test_features=test_features,
                    id_test_features=id_test_features,
                    id_test_labels=id_test_labels,
                    test_labels=test_labels,
                    # text_weights=text_weights,
                    model=clip_model,
                    classnames=classnames)
    # print(f'Final Accuracy {acc}')


if __name__ == '__main__':
    main()

