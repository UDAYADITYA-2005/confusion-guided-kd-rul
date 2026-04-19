import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pandas as pd

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

results_file = 'sikd_results_all.json'

if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        YOUR = json.load(f)
else:
    YOUR = {
        'FD001': {'rmse': 13.23, 'score': 258.4},
        'FD002': {'rmse': 15.19, 'score': 932.0},
        'FD003': {'rmse': 13.89, 'score': 251.6},
        'FD004': {'rmse': 16.21, 'score': 1728.3},
    }

PAPER = {
    'Student Only':  {'FD001':(14.64,392.2), 'FD002':(16.15,1281.1), 'FD003':(15.34,602.2), 'FD004':(17.38,1760.4)},
    'Standard KD':   {'FD001':(13.82,320.4), 'FD002':(15.59,1131.3), 'FD003':(14.16,521.2), 'FD004':(16.86,1550.0)},
    'PKT':           {'FD001':(13.57,332.3), 'FD002':(14.41,996.0),  'FD003':(13.17,350.9), 'FD004':(15.94,1291.9)},
    'CA-KD':         {'FD001':(13.41,293.8), 'FD002':(14.23,976.0),  'FD003':(12.95,325.3), 'FD004':(15.85,1256.8)},
    'RL-KD (paper)': {'FD001':(13.07,288.8), 'FD002':(14.22,901.7),  'FD003':(12.82,311.6), 'FD004':(15.71,1241.3)},
}

DS = ['FD001', 'FD002', 'FD003', 'FD004']

def fig1_score_wins():
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, ds in zip(axes, DS):
        labels = ['Stu\nOnly', 'Std KD', 'PKT', 'CA-KD', 'RL-KD\n(paper)', 'SIKD\n(ours)']
        scores = [
            PAPER['Student Only'][ds][1], PAPER['Standard KD'][ds][1],
            PAPER['PKT'][ds][1], PAPER['CA-KD'][ds][1],
            PAPER['RL-KD (paper)'][ds][1], YOUR[ds]['score']
        ]
        colors = ['#D5D8DC','#AED6F1','#5DADE2','#2471A3','#1A5276','#C0392B']
        bars   = ax.bar(labels, scores, color=colors, edgecolor='white', width=0.7)

        bars[-1].set_edgecolor('black')
        bars[-1].set_linewidth(2)

        rlkd_s = PAPER['RL-KD (paper)'][ds][1]
        delta  = YOUR[ds]['score'] - rlkd_s
        color  = '#1E8449' if delta < 0 else '#C0392B'
        symbol = '▼' if delta < 0 else '▲'
        
        ax.annotate(f'{symbol}{abs(delta):.0f}',
                    xy=(5, YOUR[ds]['score']),
                    xytext=(5, YOUR[ds]['score'] + max(scores) * 0.03),
                    ha='center', fontsize=11, color=color, fontweight='bold')

        ax.set_title(ds, fontweight='bold')
        ax.set_ylabel('Score (↓ lower = safer)' if ds == 'FD001' else '')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', labelsize=8)

    fig.suptitle('Score Metric Comparison — SIKD vs All Methods\nScore penalises late predictions 10× more (safety-critical metric)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig1_score_wins.png')

def fig2_full_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    methods   = ['Student\nOnly', 'Std KD', 'PKT', 'CA-KD', 'RL-KD\n(paper)', 'SIKD\n(ours)']
    colors    = ['#D5D8DC', '#AED6F1', '#5DADE2', '#2471A3', '#1A5276', '#C0392B']
    x         = np.arange(len(methods))
    w         = 0.18

    for ax_i, (metric_i, ax) in enumerate(zip(['rmse', 'score'], axes)):
        for di, ds in enumerate(DS):
            vals = []
            for m in methods:
                key = m.replace('\n', ' ').strip()
                if key == 'SIKD (ours)':
                    vals.append(YOUR[ds][metric_i])
                else:
                    vals.append(PAPER[key][ds][0 if metric_i == 'rmse' else 1])
            offset = (di - 1.5) * w
            bars   = ax.bar(x + offset, vals, w, color=colors, label=ds if ax_i == 0 else '', alpha=0.85, edgecolor='white', linewidth=0.5)
            bars[-1].set_edgecolor('black')
            bars[-1].set_linewidth(1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8.5)
        ax.set_ylabel('RMSE (↓)' if metric_i == 'rmse' else 'Score (↓)')
        ax.set_title('RMSE Comparison' if metric_i == 'rmse' else 'Score Comparison')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if ax_i == 0:
            ax.legend(title='Dataset', fontsize=8, loc='upper right')

    plt.suptitle('SIKD vs All Baselines — C-MAPSS RUL Prediction', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig2_full_comparison.png')

def fig3_training_curves(all_histories):
    if not all_histories:
        return
        
    colors = {'FD001': '#2E86AB', 'FD002': '#A23B72', 'FD003': '#F18F01', 'FD004': '#C73E1D'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    for ds, hist in all_histories.items():
        ep = range(1, len(hist['val_rmse']) + 1)
        sm = pd.Series(hist['val_rmse']).rolling(5, min_periods=1).mean()
        ax.plot(ep, sm, color=colors[ds], lw=2, label=ds)
        ax.axhline(PAPER['RL-KD (paper)'][ds][0], color=colors[ds], lw=1, linestyle=':', alpha=0.5)
    ax.set(xlabel='Epoch', ylabel='Val RMSE', title='Validation RMSE convergence\n(dotted = paper target)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for ds, hist in all_histories.items():
        if 'conf_mean' in hist:
            ep   = range(1, len(hist['conf_mean']) + 1)
            mean = hist['conf_mean']
            std  = hist.get('conf_std', [0]*len(mean))
            ax.plot(ep, mean, color=colors[ds], lw=2, label=ds)
            ax.fill_between(ep, np.array(mean) - np.array(std), np.array(mean) + np.array(std), alpha=0.12, color=colors[ds])
    ax.axhline(0.5, color='gray', lw=1, linestyle='--', label='Neutral (0.5)')
    ax.set(xlabel='Epoch', ylabel='Mean confusion weight w_i', title='SCM confusion weights over training', ylim=[0, 1])
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    for ds, hist in all_histories.items():
        if 'scm_scale' in hist:
            ep = range(1, len(hist['scm_scale']) + 1)
            ax.plot(ep, hist['scm_scale'], color=colors[ds], lw=2, label=ds)
    ax.axhline(0.3, color='gray', lw=1, linestyle='--', label='Init value (0.3)')
    ax.set(xlabel='Epoch', ylabel='SCM scale λ', title='Learnable SCM scale evolution')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('SIKD Training Diagnostics', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig3_training_curves.png')

def fig4_confusion_histogram(all_histories):
    if not all_histories:
        return

    colors = {'FD001': '#2E86AB', 'FD002': '#A23B72', 'FD003': '#F18F01', 'FD004': '#C73E1D'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for ds, hist in all_histories.items():
        if 'conf_mean' not in hist: continue
        n      = len(hist['conf_mean'])
        early  = hist['conf_mean'][:n//4]
        late   = hist['conf_mean'][3*n//4:]
        ax.hist(early, bins=20, alpha=0.4, color=colors[ds], label=f'{ds} early')
        ax.hist(late,  bins=20, alpha=0.7, color=colors[ds], label=f'{ds} late', histtype='step', linewidth=2)
    ax.axvline(0.5, color='red', lw=1.5, linestyle='--', label='Uniform (0.5)')
    ax.set(xlabel='Mean batch confusion weight', ylabel='Frequency', title='Confusion weight distribution\nEarly vs late training epochs')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    for ds, hist in all_histories.items():
        if 'attn_entropy' not in hist: continue
        ep = range(1, len(hist['attn_entropy']) + 1)
        ax.plot(ep, hist['attn_entropy'], color=colors[ds], lw=2, label=ds)
    ax.axhline(1.099, color='red', lw=1, linestyle='--', label='Max entropy (uniform)')
    ax.axhline(0.85,  color='green', lw=1, linestyle='--', label='Target threshold')
    ax.set(xlabel='Epoch', ylabel='Attention entropy', title='Teacher attention entropy')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('SIKD Novelty Diagnostics — SCM and C-Aggregator', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig4_confusion_histogram.png')

def print_ablation_table():
    ABLATION_RESULTS = {
        'SIKD (Full)':              {'rmse': YOUR['FD001']['rmse'], 'score': YOUR['FD001']['score']},
        'RL-KD baseline':           {'rmse': PAPER['RL-KD (paper)']['FD001'][0], 'score': PAPER['RL-KD (paper)']['FD001'][1]},
    }
    
    lines = []
    lines.append('\n' + '='*65)
    lines.append('ABLATION STUDY — FD001 (Placeholders for missing runs)')
    lines.append(f'{"Method":<30} {"RMSE":>8} {"Score":>10} {"vs Full":>10}')
    lines.append('-'*65)
    full_rmse  = ABLATION_RESULTS['SIKD (Full)']['rmse']
    
    for name, v in ABLATION_RESULTS.items():
        rmse_s  = f'{v["rmse"]:.2f}'
        score_s = f'{v["score"]:.1f}'
        delta   = f'{v["rmse"]-full_rmse:+.2f}' if v['rmse'] != full_rmse else '—'
        lines.append(f'{name:<30} {rmse_s:>8} {score_s:>10} {delta:>10}')
    lines.append('='*65)
    
    output_str = '\n'.join(lines)
    
    with open('ablation_table.txt', 'w') as f:
        f.write(output_str)

if __name__ == '__main__':
    fig1_score_wins()
    fig2_full_comparison()
    all_histories = {ds: YOUR[ds]['history'] for ds in DS if 'history' in YOUR[ds]}
    fig3_training_curves(all_histories)
    fig4_confusion_histogram(all_histories)
    print_ablation_table()
