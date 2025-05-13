import os
import time

from utils import *
from open_clip import create_model_from_pretrained
from biomed_coop import CoOpTrainer

from data import get_dataloaders
import pickle

# save_dataloader implementation
def save_dataloader(dataloader, path):
    with open(path, 'wb') as f:
        pickle.dump(dataloader, f)

# load_dataloader implementation
def load_dataloader(path):
    with open(path, 'rb') as f:
        return pickle.load(f)




def main():
    print("Starting CoOp-BiomedCLIP...")
   
    # cfg = {
    # 'metadata_path': '/home/E19_FYP_Domain_Gen_Data/metadata.csv',
    # 'data_path': '/home/E19_FYP_Domain_Gen_Data/organized_by_center',
    # 'MODEL': {
    #     'BACKBONE': {
    #         'NAME': 'BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    #     },
    #     'INIT_WEIGHTS': None  # Path to pretrained weights if available
    # },
    # 'INPUT': {
    #     'SIZE': (224, 224)  # Must match BiomedCLIP's expected input size
    # },
    # 'TRAINER': {
    #     'COOP': {
    #         'PREC': "amp",  # Can be "fp16", "fp32", or "amp"
    #         'N_CTX': 4,     # Number of context tokens to learn
    #         'CTX_INIT': "", # Initial context words (e.g., "a photo of a")
    #         'CSC': False,   # Class-specific context (False for generic context)
    #         'CLASS_TOKEN_POSITION': "end"  # "end", "middle", or "front"
    #     }
    # },
    # 'OPTIM': {
    #     'NAME': 'adamw',    # Optimizer name
    #     'LR': 0.002,        # Learning rate
    #     'MAX_EPOCH': 50,    # Total training epochs
    #     'WEIGHT_DECAY': 0.0005,  # Weight decay
    #     'LR_SCHEDULER': 'cosine',  # Learning rate scheduler
    #     'WARMUP_EPOCH': 5,  # Warmup epochs
    #     'WARMUP_TYPE': 'linear',
    #     'WARMUP_CONS_LR': True
    # },
    # 'DATASET': {
    #     'CLASSNAMES':  ["no tumor", "tumor present"]  # Your binary class names
    # }
    # }

    args = {
    'metadata_path': '/home/E19_FYP_Domain_Gen_Data/metadata.csv',
    'data_path': '/home/E19_FYP_Domain_Gen_Data/organized_by_center',
    # Basic training parameters
    'lr': 0.002,                    # Learning rate
    'train_epoch': 50,              # Number of training epochs
    'batch_size': 32,               # Batch size
    
    # Precision and hardware
    'precision': 'amp',             # 'fp16', 'fp32', or 'amp' (recommended)
    
    # CoOp-specific parameters
    'n_ctx': 4,                     # Number of context tokens (X X X X)
    'class_token_position': 'end',  # Position of class token: 'end', 'middle', or 'front'
    'csc': False,                   # Class-specific context (False for generic context)
    
    # (Optional) Advanced parameters
    'weight_decay': 0.0005,         # L2 regularization
    'warmup_epoch': 5,              # Warmup epochs for learning rate
    'ctx_init': None,               # Initial context words (e.g., "a photo of a")
    
    # Dataset-specific
    'classnames': ["no tumor", "tumor present"],  # Must match your dataset classes
    'shots': None                   # Number of shots (if using few-shot)
    }


    cache_dir = "caches"
    os.makedirs(cache_dir, exist_ok=True)
    args['cache_dir'] = cache_dir

    # load saved dataloaders from cache folder if they exist if not create them
    if os.path.exists(os.path.join(cache_dir, 'train_loader.pkl')) and \
       os.path.exists(os.path.join(cache_dir, 'val_loader.pkl')) and \
       os.path.exists(os.path.join(cache_dir, 'test_loader.pkl')) and \
       os.path.exists(os.path.join(cache_dir, 'id_test_loaders.pkl')):
        print("Loading dataloaders from cache...")
        train_loader = load_dataloader(os.path.join(cache_dir, 'train_loader.pkl'))
        val_loader = load_dataloader(os.path.join(cache_dir, 'val_loader.pkl'))
        test_loader = load_dataloader(os.path.join(cache_dir, 'test_loader.pkl'))
        id_test_loaders = load_dataloader(os.path.join(cache_dir, 'id_test_loaders.pkl'))
    else:
        print("Creating dataloaders...")
        train_loader, val_loader, test_loader, id_test_loaders= get_dataloaders(
            metadata_path=args['metadata_path'],
            data_root=args['data_path']
        )
        print("Finished Creating dataloaders...")
        save_dataloader(train_loader, os.path.join(cache_dir, 'train_loader.pkl'))
        save_dataloader(val_loader, os.path.join(cache_dir, 'val_loader.pkl'))
        save_dataloader(test_loader, os.path.join(cache_dir, 'test_loader.pkl'))
        save_dataloader(id_test_loaders, os.path.join(cache_dir, 'id_test_loaders.pkl'))
        print("Dataloaders saved to cache.")

    
    # print("Creating dataloaders...")
    # train_loader, val_loader, test_loader, id_test_loaders= get_dataloaders(
    #     metadata_path=args['metadata_path'],
    #     data_root=args['data_path']
    # )
    # print("Finished Creating dataloaders...")

   
    print("Building model...")
    trainer = CoOpTrainer(args=args)
    
    trainer.build_model(args['classnames'])  
    

    print("Starting training...")
    best_val_acc, test_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        id_test_loaders=id_test_loaders
    )
    
    print(f"\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("Test Results:", test_results)



if __name__ == '__main__':
    main()

