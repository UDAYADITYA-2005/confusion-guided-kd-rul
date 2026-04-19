import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config import DEVICE, N_FEATURES
from data_processing_train_valid_test import MyDataset
from models import LSTM_RUL

def train_ncl_teachers(data_cache, data_id='FD001', n_teachers=20, epochs=80,
                       lam=0.5, lr=1e-3, batch_size=64):
    os.makedirs('teacher_models/ncl', exist_ok=True)

    tr_x, va_x, te_x, tr_y, va_y, te_y = data_cache[data_id]
    train_ds = MyDataset(tr_x, tr_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=0)

    teachers, optims = [], []
    for i in range(n_teachers):
        torch.manual_seed(i * 7)
        m = LSTM_RUL(input_dim=N_FEATURES, hidden_dim=32, n_layers=5,
                     dropout=0.5, bid=True).to(DEVICE)
        teachers.append(m)
        optims.append(optim.AdamW(m.parameters(), lr=lr))

    crit = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        for m in teachers: m.train()

        for batch_x, batch_y, _ in train_dl:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            preds_nograd = []
            for m in teachers:
                m.eval()
                with torch.no_grad():
                    p, _ = m(batch_x)
                    preds_nograd.append(Variable(p, requires_grad=False))

            f_bar = torch.mean(torch.stack(preds_nograd), dim=0)

            for i, (m, opt) in enumerate(zip(teachers, optims)):
                m.train()
                opt.zero_grad()
                pred, _ = m(batch_x)

                corr = torch.zeros_like(f_bar)
                for j, pj in enumerate(preds_nograd):
                    if j != i:
                        corr = corr + (pj - f_bar)

                penalty = torch.mean((pred - f_bar) * corr)
                loss    = crit(pred, batch_y) + lam * penalty
                loss.backward()
                opt.step()

    for i, m in enumerate(teachers):
        torch.save(m.state_dict(),
                   f'teacher_models/ncl/{data_id}_ncl_{i+1}.pt')

    return teachers
