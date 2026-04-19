import torch
import torch.nn as nn

class StudentConfusionMap(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.3))
        self.eps   = eps

    def forward(self, s_pred, t_preds):
        stack  = torch.stack(t_preds, dim=0)          
        t_mean = stack.mean(0)                         
        t_std  = stack.std(0).clamp(min=self.eps)      

        err  = (s_pred.detach() - t_mean.detach()).abs()
        raw  = err / (t_std.detach() + self.eps)

        raw_norm = (raw - raw.mean()) / (raw.std().clamp(min=0.1))

        w = torch.sigmoid(self.scale * raw_norm).unsqueeze(1) 
        return w, t_mean, t_std
