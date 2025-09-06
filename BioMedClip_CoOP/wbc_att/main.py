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
    import pandas as pd
    train_csv_path = '/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/wbc_att/pbc_attr_v1_train.csv'
    train_df = pd.read_csv(train_csv_path)
    sorted_classnames = sorted(train_df['label'].unique())
    args = {
        'keyword': 'WBCAtt_CoOp',
        'train_csv': train_csv_path,
        'val_csv': '/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/wbc_att/pbc_attr_v1_val.csv',
        'test_csv': '/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/wbc_att/pbc_attr_v1_test.csv',
        'image_base': '/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/wbc_att/',
        'label_col': 'label',
        # Basic training parameters
        'lr': 0.002,
        'train_epoch': 100,
        'batch_size': 32,
        # Precision and hardware
        'precision': 'amp',
        # CoOp-specific parameters
        'n_ctx': 4,
        'class_token_position': 'end',
        'csc': False,
        # (Optional) Advanced parameters
        'weight_decay': 0.0005,
        'warmup_epoch': 5,
        'ctx_init': None,
        # Dataset-specific
        'classnames': list(sorted_classnames),
        'few_shot': False,  # Set to True to enable few-shot learning
        'few_shot_no': 2,   # Number of samples per class for few-shot
    }


    print(f"Starting CoOp-BiomedCLIP... KeyWord: {args['keyword']}")

    cache_dir = f"caches_{args['keyword']}"
    os.makedirs(cache_dir, exist_ok=True)
    args['cache_dir'] = cache_dir

    # load saved dataloaders from cache folder if they exist if not create them
    if os.path.exists(os.path.join(cache_dir, 'train_loader.pkl')) and \
       os.path.exists(os.path.join(cache_dir, 'val_loader.pkl')) and \
       os.path.exists(os.path.join(cache_dir, 'test_loader.pkl')):
        print("Loading dataloaders from cache...")
        train_loader = load_dataloader(os.path.join(cache_dir, 'train_loader.pkl'))
        val_loader = load_dataloader(os.path.join(cache_dir, 'val_loader.pkl'))
        test_loader = load_dataloader(os.path.join(cache_dir, 'test_loader.pkl'))
    else:
        print("Creating dataloaders...")
        train_loader, val_loader, test_loader = get_dataloaders(
            train_csv=args['train_csv'],
            val_csv=args['val_csv'],
            test_csv=args['test_csv'],
            image_base=args['image_base'],
            batch_size=args['batch_size'],
            label_col=args['label_col'],
            few_shot=args['few_shot'],
            few_shot_no=args['few_shot_no']
        )
        print("Finished Creating dataloaders...")
        save_dataloader(train_loader, os.path.join(cache_dir, 'train_loader.pkl'))
        save_dataloader(val_loader, os.path.join(cache_dir, 'val_loader.pkl'))
        save_dataloader(test_loader, os.path.join(cache_dir, 'test_loader.pkl'))
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
    )
    
    print(f"\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("Test Results:", test_results)



if __name__ == '__main__':
    main()

