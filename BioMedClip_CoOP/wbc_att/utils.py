import os
import copy
import yaml
from ast import literal_eval
from tqdm import tqdm
from typing import Dict, List
# from open_clip import get_tokenizer
from open_clip.src.open_clip import get_tokenizer



import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# import clip
from transformers import AutoTokenizer

import os
from slack_sdk import WebClient
from dotenv import load_dotenv

tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        print(f"Path:::: {cfg['cache_dir'] + '/' + split + '_f.pt'}")
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")

    return features, labels



def biomedclip_classifier(classnames, prompts, clip_model):
    """
    For wbcatt, pass a list of prompts (one per class).
    """
    with torch.no_grad():
        clip_weights = []
        for prompt in prompts:
            texts = tokenizer([prompt]).cuda()
            class_embedding = clip_model.encode_text(texts)
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embedding.squeeze(0))
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc



