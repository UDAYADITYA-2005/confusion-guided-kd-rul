import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import json
import gc  
from torch.utils.data import DataLoader

from config import DEVICE, N_FEATURES, MAX_RUL, set_seed
from data_processing_train_valid_test import MyDataset, process_data
from models import CNN_Student, PKT, DuelingDDQN
from utils import rmse_loss, score_metric
from replay_memory import ReplayBuffer
from lstm_teacher_negative_correlation_learning import train_ncl_teachers

def train_rlkd_original(
    data_cache,
    data_id,
    teachers,
    K          = 3,         
    n_epochs   = 150,       
    lr_student = 5e-4,      
    lr_ddqn    = 1e-4,
    gamma      = 0.9,
    delta      = 0.999,
    batch_size = 128,       
    buf_size   = 10000,
):
    M = len(teachers)
    tr_x, va_x, te_x, tr_y, va_y, te_y = data_cache[data_id]

    train_ds = MyDataset(tr_x, tr_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    te_x_t = torch.FloatTensor(te_x).to(DEVICE)
    te_y_t = torch.FloatTensor(te_y).to(DEVICE)
    va_x_t = torch.FloatTensor(va_x).to(DEVICE)
    va_y_t = torch.FloatTensor(va_y).to(DEVICE)

    student    = CNN_Student(input_dim=N_FEATURES).to(DEVICE)
    pkt        = PKT().to(DEVICE)
    
    online_net = DuelingDDQN(state_dim=64, num_actions=M).to(DEVICE)
    target_net = DuelingDDQN(state_dim=64, num_actions=M).to(DEVICE)
    target_net.load_state_dict(online_net.state_dict())

    for t in teachers: t.eval()

    opt_s = optim.Adam(student.parameters(), lr=lr_student, weight_decay=1e-5)
    opt_q = optim.Adam(online_net.parameters(), lr=lr_ddqn)
    
    sched = optim.lr_scheduler.StepLR(opt_s, step_size=40, gamma=0.5)
    buf   = ReplayBuffer(buf_size)

    best_val_rmse, best_state = float('inf'), None
    history = {'val_rmse': [], 'test_rmse': []}

    print(f'\nRL-KD Training: {data_id} | K={K} | {M} teachers')

    for epoch in range(n_epochs):
        student.train()

        p  = epoch / n_epochs
        a3 = max(0.0, 0.5 * (1 - p))   
        a1 = 0.5 * p                   
        a2 = max(0.0, 1 - a1 - a3)      

        for batch_x, batch_y, _ in train_dl:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            with torch.no_grad():
                all_feats = torch.stack(
                    [teachers[i](batch_x)[1] for i in range(M)], dim=1
                )  

            state = all_feats.mean(dim=(0, 1)).unsqueeze(0)  

            epsilon = max(0.05, 0.3 - epoch * 0.003)
            sel_idxs = []
            avail    = list(range(M))
            
            for _ in range(K):
                if random.random() < epsilon or len(buf) < 256:
                    choice = random.choice(avail)
                    sel_idxs.append(choice)
                else:
                    online_net.eval()
                    with torch.no_grad():
                        q = online_net(state.to(DEVICE))[0]
                        mask = torch.zeros(M).to(DEVICE)
                        for ex in sel_idxs: mask[ex] = -1e9
                        choice = int((q + mask).argmax())
                        sel_idxs.append(choice)
                avail = [a for a in avail if a != choice]

            sel_teachers = [teachers[i] for i in sel_idxs]

            with torch.no_grad():
                t_preds = [t(batch_x)[0] for t in sel_teachers]
                t_feats = [t(batch_x)[1] for t in sel_teachers]
                t_mean_pred = torch.stack(t_preds).mean(0)
                t_mean_feat = torch.stack(t_feats).mean(0)

            s_pred, s_feat_raw, s_feat = student(batch_x)

            L_hard = F.mse_loss(s_pred, batch_y)
            L_soft = F.mse_loss(s_pred, t_mean_pred)
            L_pkt  = pkt(t_mean_feat, s_feat)
            total  = a1 * L_hard + a2 * L_soft + a3 * L_pkt

            student.eval()  
            with torch.no_grad():
                rmse_before = float(rmse_loss(student(va_x_t[:64])[0], va_y_t[:64]))
            
            student.train() 
            opt_s.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt_s.step()
            
            student.eval()
            with torch.no_grad():
                rmse_after = float(rmse_loss(student(va_x_t[:64])[0], va_y_t[:64]))
            student.train()

            if rmse_before > 1e-6:
                r = float(np.clip(np.tanh(4*(rmse_before-rmse_after)/rmse_before), -1, 1))
            else:
                r = 0.0

            with torch.no_grad():
                all_feats_next = torch.stack([teachers[i](batch_x)[1] for i in range(M)], dim=1)
            state1 = all_feats_next.mean(dim=(0, 1)).unsqueeze(0)
            
            buf.push(state.cpu(), sel_idxs[0], r, state1.cpu(), 1.0)

            if len(buf) >= 256:
                _update_ddqn(online_net, target_net, opt_q, buf, M, gamma, delta)

        sched.step()

        student.eval()
        with torch.no_grad():
            vp, _, _ = student(va_x_t)
            vr = float(rmse_loss(vp * MAX_RUL, va_y_t * MAX_RUL))
        history['val_rmse'].append(vr)

        if vr < best_val_rmse:
            best_val_rmse = vr
            best_state    = {k: v.clone() for k, v in student.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f'Ep {epoch+1:3d} | α=({a1:.2f},{a2:.2f},{a3:.2f}) | Val RMSE={vr:.3f}')

    student.load_state_dict(best_state)
    student.eval()
    with torch.no_grad():
        tp, _, _ = student(te_x_t)
        
        tp = torch.clamp(tp, 0.0, 1.0)
        
        tp_sc    = tp * MAX_RUL
        test_rmse  = float(rmse_loss(tp_sc, te_y_t))
        test_score = score_metric(tp_sc, te_y_t)

    print(f'\n[RL-KD {data_id}] Test RMSE={test_rmse:.2f}  Score={test_score:.1f}')
    return student, test_rmse, test_score, history

def _update_ddqn(online, target, opt, buf, M, gamma, delta, batch_size=32):
    batch  = buf.sample(batch_size)
    states = torch.cat([b[0] for b in batch]).to(DEVICE)
    acts   = torch.tensor([b[1] for b in batch], dtype=torch.long).to(DEVICE)
    rews   = torch.tensor([b[2] for b in batch], dtype=torch.float).to(DEVICE)
    next_s = torch.cat([b[3] for b in batch]).to(DEVICE)
    betas  = torch.tensor([b[4] for b in batch], dtype=torch.float).to(DEVICE)

    online.train()
    q_est  = online(states).gather(1, acts.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        a_next = online(next_s).argmax(1)
        q_tar  = target(next_s).gather(1, a_next.unsqueeze(1)).squeeze(1)
        target_val = rews + betas * gamma * q_tar

    loss = F.smooth_l1_loss(q_est, target_val)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 1.0)
    opt.step()

    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.copy_(delta * tp.data + (1 - delta) * op.data)

if __name__ == '__main__':
    set_seed(99)
    
    CACHE = {}
    print("Processing datasets...")
    for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
        CACHE[ds] = process_data('data', ds)

    PAPER_RESULTS = {
        'FD001': {'rmse': 13.07, 'score': 288.82},
        'FD002': {'rmse': 14.22, 'score': 901.74},
        'FD003': {'rmse': 12.82, 'score': 311.55},
        'FD004': {'rmse': 15.71, 'score': 1241.31},
    }

    K_MAP = {'FD001': 3, 'FD002': 5, 'FD003': 3, 'FD004': 5}

    all_rlkd_results = {}

    for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
        print(f'\n{"="*50}\n{ds}\n{"="*50}')
        
        teachers_ds = train_ncl_teachers(CACHE, data_id=ds, n_teachers=20, epochs=40)
        
        student, rmse, score, hist = train_rlkd_original(
            data_cache=CACHE, data_id=ds, teachers=teachers_ds, K=K_MAP[ds], n_epochs=150, batch_size=128)
            
        torch.save(student.state_dict(), f'rlkd_student_{ds}_pure.pt')
        all_rlkd_results[ds] = {'rmse': rmse, 'score': score}
        
        del teachers_ds
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('\n' + '='*65)
    print(f'{"Dataset":<10} {"Paper RMSE":>12} {"Ours RMSE":>12} {"Diff":>8}')
    print('-'*65)

    for ds in ['FD001','FD002','FD003','FD004']:
        paper_r = PAPER_RESULTS[ds]['rmse']
        ours_r  = all_rlkd_results[ds]['rmse']
        diff    = ours_r - paper_r
        print(f'{ds:<10} {paper_r:>12.2f} {ours_r:>12.2f} {diff:>+8.2f}')
    print('='*65)

    with open('baseline_results_all.json', 'w') as f:
        json.dump(all_rlkd_results, f, indent=2)

    print('\nResults saved → baseline_results_all.json')
    print('\nBaseline complete')
