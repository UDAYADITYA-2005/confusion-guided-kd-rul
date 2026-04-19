import torch
from config import DEVICE, MAX_RUL

def rmse_loss(y_hat, y):
    return torch.sqrt(torch.mean((y_hat - y) ** 2))

def score_metric(y_hat, y):
    s = 0.0
    for i in range(len(y_hat)):
        d = y_hat[i] - y[i]
        s += (torch.exp(-d / 10) - 1) if d <= 0 else (torch.exp(d / 13) - 1)
    return float(s)

@torch.no_grad()
def evaluate(model, X, y, max_rul=MAX_RUL):
    model.eval()
    X, y = X.to(DEVICE), y.to(DEVICE)
    pred, _, _ = model(X)
    pred_scaled = pred * max_rul   
    rmse  = float(rmse_loss(pred_scaled, y))
    score = score_metric(pred_scaled, y)
    return rmse, score
