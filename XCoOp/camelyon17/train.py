
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # TODO:  GPU choice

import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import datasets.camelyon17

# trainer
import prompting.xcoop


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.no_wandb:
        cfg.WANDB = False
    else:
        cfg.WANDB = True
    
    if args.no_wandb:
        cfg.WANDB = False
    else:
        cfg.WANDB = True


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # only use full-shots setting
    cfg.DATASET.INCLUDE_ALL_CLASSES = False

    ##### ------ UniveralOP ------- ######
    cfg.TRAINER.XCoOp = CN()
    cfg.TRAINER.XCoOp.ENABLE = True 
    cfg.TRAINER.XCoOp.XCoOp_PROMPTS = ['a photo of {}']
    cfg.TRAINER.XCoOp.XCoOp_LOSS_WEIGHT = 1.0
    cfg.TRAINER.XCoOp.N_CTX = 16 
    cfg.TRAINER.XCoOp.CTX_INIT = ""
    cfg.TRAINER.XCoOp.PREC = "amp"
    
    cfg.TRAINER.XCoOp.ENABLE_W = False
    cfg.TRAINER.XCoOp.ENABLE_IMPLICIT_OP = 'sum' # mul
    cfg.TRAINER.XCoOp.PRETRAINED_PROMPTS_DIR = None
    cfg.TRAINER.XCoOp.CLINICAL_PROMPTS = [
      "a photo of a {}, no evidence of tumor present.",  # non-tumor
      "a photo of a {}, tumor detected in tissue section.", # tumor
    ]
    cfg.TRAINER.XCoOp.XCoOp_TOKEN_LOSS_WEIGHT = 1.0
    cfg.TRAINER.XCoOp.XCoOp_VOCAB_LOSS_WEIGHT = 0.0  
    cfg.TRAINER.XCoOp.VISUAL_FEATURE_NUMBER = 1
    cfg.TRAINER.XCoOp.TEXT_PROMPT_NUMBER = 1 

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    trainer.load_model(None, 10)

    if not args.no_train:
        trainer.train()


def main():
    # Hardcoded variables for camelyon17 dataset
    
    dataset_root = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/patches" 
    metadata_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/metadata.csv" 
    output_dir = "./output"
    config_file = "configs/XCoOp/vit_b16_c4_batch32_camelyon17.yaml"
    dataset_config_file = "configs/datasets/camelyon17.yaml"
    seed = 42
    trainer = "XCoOp"
    backbone = "ViT-B/16"
    head = ""
    eval_only = False
    model_dir = ""
    load_epoch = None
    no_train = False
    no_wandb = False
    opts = []

    class Args:
        pass

    args = Args()
    args.root = dataset_root
    args.metadata_path = metadata_path
    args.output_dir = output_dir
    args.resume = ""
    args.seed = seed
    args.source_domains = None
    args.target_domains = None
    args.transforms = None
    args.trainer = trainer
    args.backbone = backbone
    args.head = head
    args.eval_only = eval_only
    args.model_dir = model_dir
    args.load_epoch = load_epoch
    args.no_train = no_train
    args.opts = opts
    args.no_wandb = no_wandb
    args.config_file = config_file
    args.dataset_config_file = dataset_config_file

    print("The program starts!")
    main_entry(args)

# Rename main to main_entry to avoid recursion
def main_entry(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    trainer.load_model(None, 10)

    if not args.no_train:
        trainer.train()

# Entry point for script execution
if __name__ == "__main__":
    main()