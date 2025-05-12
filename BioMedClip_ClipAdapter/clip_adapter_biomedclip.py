import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import argparse
import numpy as np

from tqdm import tqdm
from utils import cls_acc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class ClipAdapter_BiomedCLIP():
    '''
    CLIP Adapter method
        @article{gao2021clip,
            title={CLIP-Adapter: Better Vision-Language Models with Feature Adapters},
            author={Gao, Peng and Geng, Shijie and Zhang, Renrui and Ma, Teli and Fang, Rongyao and Zhang, Yongfeng and Li, Hongsheng and Qiao, Yu},
            journal={arXiv preprint arXiv:2110.04544},
            year={2021}
        }
    
    '''

    def __init__(self, args: argparse.Namespace):
        self.lr = args['lr']
        self.epoch = args['train_epoch']
        self.alpha = args['alpha_ca']
        self.cfg = args

    def forward(self,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_features: torch.tensor,
                test_labels: torch.tensor,
                text_weights: torch.tensor,
                model: nn.Module,
                id_test_features: torch.tensor,
                id_test_labels: torch.tensor,
                classnames):
        """
        inputs:
            train_loader : torch.utils.data.DataLoader
            test_features : torch.Tensor of shape [test_data_size, 1024]
            test_labels : torch.Tensor of shape [test_data_size]
            text_weights : torch.Tensor of shape [num_shot*num_classes, 1024]
        """
        # model.float()
        
        cfg = self.cfg
        """
        if cfg["shots"] == 1:
            self.cfg['train_epoch'] = 50
        elif cfg["shots"] == 2 or cfg["shots"] == 4:
            self.cfg['train_epoch'] = 100
        else:
            self.cfg['train_epoch'] = 200
        """
        print(self.cfg['train_epoch'])
        print('Building custom CLIP')
        model.eval()
        clip_ad_model = CustomCLIP(model)
        clip_ad_model_val = copy.deepcopy(clip_ad_model)
        
        # # New - For load and TEst -  Start
        # # Load the pre-trained adapter
        # adapter_path = self.cfg['cache_dir'] + "/best_clipAdapterModel.pt"
        # print(f"Loading adapter from: {adapter_path}")

        #  # Add the Adapter class to safe globals before loading
        # # torch.serialization.add_safe_globals([Adapter])
    
        # # Load the adapter with weights_only=False (only if you trust the source)
        # clip_ad_model.adapter = torch.load(adapter_path, weights_only=False)

        # clip_ad_model.adapter = torch.load(adapter_path)
        # clip_ad_model.cuda()
        # # New - For load and TEst -  End




        print('Turning off gradients in both the image and the text encoder')
        for name, param in clip_ad_model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)
                
        for name, param in clip_ad_model_val.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)
        
        clip_ad_model.cuda()
        clip_ad_model_val.cuda()
    

        # Feature Extraction for Validation
        print("\nExtracting visual features and labels from val set.")
        val_features, val_labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(val_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                val_features.append(image_features)
                val_labels.append(target)
        val_features, val_labels = torch.cat(val_features), torch.cat(val_labels)
        start_time = time.time() 
       
        alpha = cfg["alpha_ca"]
        print(alpha)

        
        optimizer = torch.optim.SGD(clip_ad_model.adapter.parameters(), self.lr)
        
        # Train
        print('\nStart Training procedure')
           
        best_acc, best_epoch = 0.0, 0
        for train_idx in range(self.cfg['train_epoch']):
            # Train
            clip_ad_model.adapter.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.cfg['train_epoch']))

            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                logits = clip_ad_model(image_features, text_weights, alpha)

                loss = F.cross_entropy(logits, target)

                acc = cls_acc(logits, target)
                correct_samples += acc / 100 * len(logits)
                all_samples += len(logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                
            clip_ad_model.adapter.eval()
            # current_lr = scheduler.get_last_lr()[0]
            print('Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format( correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))
            clip_ad_model.eval()
            logits = clip_ad_model(val_features, text_weights, self.alpha)
            acc = cls_acc(logits, val_labels)
            
            print("**** Clip-Adapter's val accuracy: {:.4f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = train_idx
                torch.save(clip_ad_model.adapter, self.cfg['cache_dir'] + "/best_clipAdapterModel.pt")
        # Evaluation
        print("Total time = {:.4f}".format(time.time()-start_time))
        # clip_ad_model.adapter = torch.load(self.cfg['cache_dir'] + "/best_clipA_" + str(self.cfg['shots']) + "shots.pt")
        
        print('\nStart evaluation on test sets')
        clip_ad_model.eval()
        # # logits_test = clip_ad_model(test_features, text_weights, self.alpha) 

        # # acc_test = np.mean(logits_test.argmax(dim=1).cpu().numpy() ==  test_labels.cpu().numpy())*100.0



        # Newly Added
        # Evaluate main test set (center 4)
        print("\nEvaluating main test set (center 4):")
        logits_test = clip_ad_model(test_features, text_weights, self.alpha)
        preds_test = logits_test.argmax(dim=1).cpu().numpy()
        labels_test = test_labels.cpu().numpy()

        print(f"Accuracy: {accuracy_score(labels_test, preds_test)*100:.2f}%")
        print(f"F1: {f1_score(labels_test, preds_test)*100:.2f}%")
        print(f"Precision: {precision_score(labels_test, preds_test)*100:.2f}%")
        print(f"Recall: {recall_score(labels_test, preds_test)*100:.2f}%")

        # Evaluate each in-distribution test set (centers 0-2)
        for i in range(3):
            center_name = f'id_test_center_{i}'
            print(f"\nEvaluating {center_name}:")
            
            features = id_test_features[center_name]
            labels = id_test_labels[center_name]
            
            logits = clip_ad_model(features, text_weights, self.alpha)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            print(f"Accuracy: {accuracy_score(labels, preds)*100:.2f}%")
            print(f"F1: {f1_score(labels, preds)*100:.2f}%")
            print(f"Precision: {precision_score(labels, preds)*100:.2f}%")
            print(f"Recall: {recall_score(labels, preds)*100:.2f}%")
        # Return dummy values for acc and loss
        return 100, 100


    
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class CustomCLIP(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.text.transformer.dtype
        self.adapter = Adapter(512, 4).to(clip_model.text.transformer.dtype)

            
    def forward(self, image_features, text_features, alpha):
        x = self.adapter(image_features)

        # alpha = 0.2
        image_features = alpha * x + (1 - alpha) * image_features
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features

        return logits
