import os
import time

from utils import *
# from open_clip import create_model_from_pretrained
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
   
    args = {
    'keyword': 'class_specific',   
    'metadata_path': '/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/metadata.csv',
    'organized_data_path': '/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/organized_by_center',
    'data_path': '/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/patches',
    # Basic training parameters
    'lr': 0.002,                    # Learning rate
    'train_epoch': 100,              # Number of training epochs
    'batch_size': 64,               # Batch size
    
    # Precision and hardware
    'precision': 'amp',             # 'fp16', 'fp32', or 'amp' (recommended)
    
    # CoOp-specific parameters
    'n_ctx': 4,                     # Number of context tokens (X X X X)
    'class_token_position': 'end',  # Position of class token: 'end', 'middle', or 'front'
    'csc': True,                   # Class-specific context (False for generic context)
    
    # (Optional) Advanced parameters
    'weight_decay': 0.0005,         # L2 regularization
    'warmup_epoch': 5,              # Warmup epochs for learning rate
    'ctx_init': None,               # Initial context words (e.g., "a photo of a")
    
    # Dataset-specific
    'classnames': ["no tumor", "tumor present"],  # Must match your dataset classes
    'shots': None                   # Number of shots (if using few-shot)
    }


    print(f"Starting CoOp-BiomedCLIP... KeyWord: {args['keyword']}")
    send_slack_message(f"Starting CoOp-BiomedCLIP... KeyWord: {args['keyword']}")

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
            data_root=args['data_path'],
            batch_size=args['batch_size'],
        )
        print("Finished Creating dataloaders...")
        save_dataloader(train_loader, os.path.join(cache_dir, 'train_loader.pkl'))
        save_dataloader(val_loader, os.path.join(cache_dir, 'val_loader.pkl'))
        save_dataloader(test_loader, os.path.join(cache_dir, 'test_loader.pkl'))
        save_dataloader(id_test_loaders, os.path.join(cache_dir, 'id_test_loaders.pkl'))
        print("Dataloaders saved to cache.")

    
    

   
    print("Building model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device in Main: {device}")
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

