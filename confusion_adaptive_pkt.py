import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfusionAdaptivePKT(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def _cond_prob(self, f):
        fn   = F.normalize(f, dim=1, eps=self.eps)
        sim  = 0.5 * (torch.mm(fn, fn.T) + 1.0)
        mask = 1.0 - torch.eye(f.size(0), device=f.device)
        sim  = sim * mask
        return sim / (sim.sum(0, keepdim=True) + self.eps) + self.eps

    def forward(self, f_teacher, f_student, w):
        p  = self._cond_prob(f_teacher)
        q  = self._cond_prob(f_student)
        wv = w.squeeze(1)                    
        W  = torch.outer(wv, wv)             
        kl = p * torch.log(p / q)
        return (W * kl).sum(0).mean() / (W.mean() + self.eps)
