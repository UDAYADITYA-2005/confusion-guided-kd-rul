import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from config import DEVICE, N_FEATURES, MAX_RUL
from data_processing_train_valid_test import MyDataset
from models import CNN_Student
from student_confusion_map import StudentConfusionMap
from confusion_weighted_aggregator import ConfusionWeightedAggregator
from confusion_adaptive_pkt import ConfusionAdaptivePKT
from utils import rmse_loss, score_metric

def train_sikd_final(
    data_cache,
    data_id,
    teachers,
    K          = 3,
    n_epochs   = 200,
    lr_student = 1e-4, 
    paper_rmse = None
):
    M = len(teachers)
    tr_x, va_x, te_x, tr_y, va_y, te_y = data_cache[data_id]

    train_ds = MyDataset(tr_x, tr_y)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

    te_x_t = torch.FloatTensor(te_x).to(DEVICE)
    te_y_t = torch.FloatTensor(te_y).to(DEVICE)
    va_x_t = torch.FloatTensor(va_x).to(DEVICE)
    va_y_t = torch.FloatTensor(va_y).to(DEVICE)

    student    = CNN_Student(input_dim=N_FEATURES).to(DEVICE)
    scm        = StudentConfusionMap().to(DEVICE)
    aggregator = ConfusionWeightedAggregator(feat_dim=64, K=K).to(DEVICE)
    cpkt_fn    = ConfusionAdaptivePKT().to(DEVICE)

    all_params = (list(student.parameters()) + list(scm.parameters()) + list(aggregator.parameters()))
    opt   = optim.Adam(all_params, lr=lr_student, weight_decay=1e-5) 
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    history = {'val_rmse':[], 'conf_mean':[], 'conf_std':[], 'scm_scale':[], 'attn_entropy':[]}
    best_rmse, best_state = float('inf'), None
    rolling_val = []

    for epoch in range(n_epochs):
        student.train(); scm.train(); aggregator.train()

        progress = epoch / n_epochs
        if progress < 0.4:
            a1, a2, a3 = 0.10, 0.30, 0.60
        elif progress < 0.8:
            t_ = (progress - 0.4) / 0.4
            a3 = 0.60 * (1 - t_)
            a2 = 0.30 + 0.40 * t_
            a1 = 0.10 + 0.20 * t_
        else:
            t_ = (progress - 0.8) / 0.2
            a3 = 0.00
            a2 = 0.70 * (1 - t_)
            a1 = 0.30 + 0.70 * t_

        s = a1 + a2 + a3
        a1 /= s; a2 /= s; a3 /= s

        ep_conf, ep_cstd, ep_entr = [], [], []

        for batch_x, batch_y, _ in train_dl:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            sel_idxs = random.sample(range(M), K)
            sel_ts   = [teachers[i] for i in sel_idxs]

            with torch.no_grad():
                t_preds = [t(batch_x)[0] for t in sel_ts]
                t_feats = [t(batch_x)[1] for t in sel_ts]

            s_pred, _, s_feat = student(batch_x)

            conf_w, t_mean, _ = scm(s_pred, t_preds)
            agg_feat, attn_w = aggregator(t_feats, s_feat, conf_w)
            
            ent = -(attn_w * (attn_w + 1e-8).log()).sum(1).mean().item()

            L_hard = F.mse_loss(s_pred, batch_y)
            L_soft = F.mse_loss(s_pred, t_mean.detach())
            
            L_cpkt = cpkt_fn(agg_feat.detach(), s_feat, conf_w)

            conf_spread = -((conf_w - 0.5).pow(2)).mean()
            conf_reg    = 0.01 * conf_spread

            total_loss = a1 * L_hard + a2 * L_soft + a3 * L_cpkt + conf_reg

            opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

            ep_conf.append(conf_w.mean().item())
            ep_cstd.append(conf_w.std().item())
            ep_entr.append(ent)

        sched.step()

        student.eval()
        with torch.no_grad():
            vp, _, _ = student(va_x_t)
            vp = torch.clamp(vp, 0.0, 1.0)
            vr = float(rmse_loss(vp * MAX_RUL, va_y_t * MAX_RUL))

        scm_scale_val = scm.scale.item()
        history['val_rmse'].append(vr)
        history['conf_mean'].append(np.mean(ep_conf))
        history['conf_std'].append(np.mean(ep_cstd))
        history['scm_scale'].append(scm_scale_val)
        history['attn_entropy'].append(np.mean(ep_entr))

        rolling_val.append(vr)
        if len(rolling_val) > 5: rolling_val.pop(0)
        if np.mean(rolling_val) < best_rmse:
            best_rmse  = np.mean(rolling_val)
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

    student.load_state_dict(best_state)
    student.eval()
    with torch.no_grad():
        tp, _, _ = student(te_x_t)
        tp = torch.clamp(tp, 0.0, 1.0)
        tp_sc = tp * MAX_RUL
        test_rmse  = float(rmse_loss(tp_sc, te_y_t))
        test_score = score_metric(tp_sc, te_y_t)

    return student, test_rmse, test_score, history
