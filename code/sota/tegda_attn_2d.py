from copy import deepcopy
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.jit
from monai.losses import DiceCELoss
import numpy as np
from collections import defaultdict
import re
from sota.aetta2d import AETTA

dicece_loss = DiceCELoss(4)

def group_keys_by_prefix(key_list):
    grouped = defaultdict(list)
    for key in key_list:
        match = re.search(r'^(.\w+?\.\d+)\.', key)
        if match:
            prefix = match.group(1)
        else:
            parts = key.split('.')
            prefix = '.'.join(parts[:1]) if len(parts) >= 2 else key
        
        grouped[prefix].append(key)
    
    return grouped

def normalize_fisher_by_layer(fisher, layer_groups):
    layer_scores = {}
    for layer_name, param_names in layer_groups.items():
        total_fisher = sum(fisher[name].sum().item() for name in param_names)
        num_params = sum(fisher[name].numel() for name in param_names)
        layer_scores[layer_name] = total_fisher / num_params
    
    max_score = max(layer_scores.values())
    return {layer: score / max_score for layer, score in layer_scores.items()}

def selective_restore_fisher(ema_model, source_model, fisher, revert_ratio=0.1):
    """Selectively restore EMA model parameters based on Fisher information"""
    source_state = source_model.state_dict()
    teacher_state = ema_model.state_dict()
    trainable_params = [name for name, param in ema_model.named_parameters()]
    layer_groups = group_keys_by_prefix(trainable_params)
    layer_scores = normalize_fisher_by_layer(fisher, layer_groups)
    
    with torch.no_grad():
        for layer_idx, (layer_name, param_names) in enumerate(layer_groups.items()):
            layer_revert_ratio = revert_ratio * (layer_idx + 1)
            for name in param_names:
                teacher_param = teacher_state[name]
                source_param = source_state[name]
                n_params = teacher_param.numel()
                n_revert = int(n_params * layer_revert_ratio)
                if n_revert == 0:
                    continue
                param_fisher = fisher[name].view(-1)
                _, indices = torch.topk(param_fisher, k=n_revert, largest=False)
                teacher_flat = teacher_param.view(-1)
                source_flat = source_param.view(-1)
                teacher_flat[indices] = source_flat[indices]
                teacher_state[name] = teacher_flat.view(teacher_param.shape)
    ema_model.load_state_dict(teacher_state)
    return ema_model

class Prototype_Pool(nn.Module):
    def __init__(self, class_num=4, max=50, scale_num=5):
        super(Prototype_Pool, self).__init__()
        self.class_num = class_num
        self.max_length = max
        self.feature_bank = [[] for _ in range(scale_num)]
        self.name_list = []
    
    def get_pool_feature(self, scale_idx=None):
        if scale_idx is not None:
            return self.feature_bank[scale_idx]
        return self.feature_bank
    
    def update_feature_pool(self, feature, feat_idx):
        for scale_idx in range(len(feature)):
            current_feat = feature[scale_idx][feat_idx].detach()
            if len(self.feature_bank[scale_idx]) == 0:
                self.feature_bank[scale_idx] = current_feat.unsqueeze(0)
            else:
                updated_pool = torch.cat([self.feature_bank[scale_idx], current_feat.unsqueeze(0)], dim=0)
                if updated_pool.size(0) > self.max_length:
                    updated_pool = updated_pool[-self.max_length:]
                
                self.feature_bank[scale_idx] = updated_pool

class TTA(nn.Module):
    """TEGDA+ 2D variant with attention mechanism for test-time adaptation"""
    def __init__(self, model, anchor_model, optimizer, steps=2, episodic=False, mt_alpha=0.99, rst_m=0.1):
        super().__init__()
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.optimizer = optimizer
        self.model_ema = anchor_model
        self.num_classes = 4 
        self.mt = mt_alpha
        self.rst = rst_m
        self.model = model
        self.source_model = deepcopy(model)
        self.pool = Prototype_Pool(class_num=4, max=10, scale_num=5)
        for param in self.source_model.parameters():
            param.requires_grad = False
        self.est = AETTA()
        self.est_list = []

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(1):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs
    
    @torch.no_grad()
    def forward_no_adapt(self, x):
        outputs = self.model(x)
        return outputs
    
    def get_index(self):
        return self.index
   
    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer, multi_eval=False):
        pred = self.forward_no_adapt(x)
        pred = torch.argmax(pred, dim=1).cpu().numpy()[0]
        eval_model = deepcopy(model)
        
        if multi_eval:
            est_1, est_2, est_3, est_avg, mismatch_mask, entropy, var, acc = self.est.aetta(input=x, pred=pred, model=eval_model, multi_eval=multi_eval)
        else:
            est_1, est_2, est_3, est_avg, mismatch_mask, entropy = self.est.aetta(input=x, pred=pred, model=eval_model, multi_eval=multi_eval)
        
        est_avg = np.array([est_avg])
        self.est_list.extend(est_avg)
        adapt_alpha = est_avg.mean() / 100
        estp90 = np.percentile(self.est_list, 90)
        
        # Update feature pool with high-confidence samples
        batch_feats = model.get_feature(x)
        for i in range(len(est_avg)):
            if estp90 <= est_avg[i]:
                self.pool.update_feature_pool(batch_feats, i)
        
        bank_features = self.pool.get_pool_feature()
        
        # Original output
        outputs = model.get_output(batch_feats)
        
        # Attention-enhanced output
        updated_outputs = model.get_output_attn(batch_feats, bank_features, est_avg / 100)
        
        # EMA reference
        standard_ema = self.model_ema(x)
        
        # Compute losses
        sem_loss = adapt_alpha * ((softmax_entropy(outputs, updated_outputs)).mean(0) + (softmax_entropy(updated_outputs, outputs)).mean(0)) / 2.0
        ce_loss = ((softmax_entropy(outputs, standard_ema)).mean(0) + (softmax_entropy(standard_ema, outputs)).mean(0)) / 2.0
        loss = ce_loss + sem_loss
        
        print(f'ce_loss:{ce_loss}, sem_loss:{sem_loss}, loss:{loss}')
        
        loss.backward()
        
        # Compute Fisher information
        fisher_enc = {n: torch.zeros_like(p) for n, p in model.enc.named_parameters()}
        fisher_dec = {n: torch.zeros_like(p) for n, p in model.dec1.named_parameters()}
        for name, param in model.enc.named_parameters():
            if param.grad is not None:
                fisher_enc[name] += param.grad ** 2
        for name, param in model.dec1.named_parameters():
            if param.grad is not None:
                fisher_dec[name] += param.grad ** 2
        
        optimizer.step()
        optimizer.zero_grad()

        # Use selective restore with Fisher information instead of global EMA
        self.model_ema.enc = selective_restore_fisher(self.model_ema.enc, self.source_model.enc, fisher_enc, revert_ratio=0.05*adapt_alpha)
        # self.model_ema.dec1 = selective_restore_fisher(self.model_ema.dec1, self.source_model.dec1, fisher_dec, revert_ratio=0.1*adapt_alpha)

        return model(x)


@torch.jit.script
def softmax_entropy(x, x_ema):
    """Entropy of softmax distribution from logits (2D variant)."""
    n, c, h, w = x.shape
    entropy1 = -(x_ema.softmax(1) * x.log_softmax(1)).sum() / \
        (n * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
    return entropy1
