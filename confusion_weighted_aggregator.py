import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfusionWeightedAggregator(nn.Module):
    def __init__(self, feat_dim=64, K=3):
        super().__init__()
        self.K = K
        self.temperature = 0.05

    def forward(self, t_feats, s_feat, w):
        sn = F.normalize(s_feat.detach(), dim=1)  
        scores = []
        for tf in t_feats:
            tn  = F.normalize(tf, dim=1)           
            cos = (tn * sn).sum(1, keepdim=True)   
            score = w * (-cos) + (1 - w) * cos
            scores.append(score)

        logits = torch.cat(scores, dim=1) / self.temperature  
        attn   = F.softmax(logits, dim=1)                      
        feat_stack = torch.stack(t_feats, dim=1)               
        agg = (attn.unsqueeze(-1) * feat_stack).sum(1)         
        return agg, attn
