import json
import gc
import torch

from config import set_seed
from data_processing_train_valid_test import process_data
from lstm_teacher_negative_correlation_learning import train_ncl_teachers
from sikd_train import train_sikd_final

if __name__ == '__main__':
    # 1. Setup and Initialization
    set_seed(99)
    CACHE = {}
    
    print("Processing datasets...")
    for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
        CACHE[ds] = process_data('data', ds)

    all_sikd_results = {}
    PAPER_RMSE = {'FD001': 13.07, 'FD002': 14.22, 'FD003': 12.82, 'FD004': 15.71}
    PAPER_SCORE = {'FD001': 288.82, 'FD002': 901.74, 'FD003': 311.55, 'FD004': 1241.31}
    K_MAP = {'FD001': 3, 'FD002': 5, 'FD003': 3, 'FD004': 5}

    # 2. Master Execution Loop
    for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
        print(f'\n{"="*60}\nRunning SIKD on {ds}\n{"="*60}')
        print(f"Training 20 NCL Teachers for {ds}...")
        
        teachers_ds = train_ncl_teachers(CACHE, data_id=ds, n_teachers=20, epochs=80)
        
        stu, rmse, score, hist = train_sikd_final(
            data_cache = CACHE, 
            data_id    = ds, 
            teachers   = teachers_ds, 
            K          = K_MAP[ds], 
            n_epochs   = 200,
            lr_student = 1e-4,
            paper_rmse = PAPER_RMSE[ds]
        )
            
        torch.save(stu.state_dict(), f'sikd_student_{ds}.pt')
        all_sikd_results[ds] = {'rmse': rmse, 'score': score, 'history': hist}
        
        # Memory management
        del teachers_ds
        del stu
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. Save JSON Data
    with open('sikd_results_all.json', 'w') as f:
        json.dump(all_sikd_results, f, indent=2)

    # 4. Print Final Results Table
    print('\n' + '='*75)
    print(f'{"Dataset":<10} {"Paper RMSE":>12} {"SIKD RMSE":>12} '
          f'{"Δ RMSE":>10} {"Paper Score":>14} {"SIKD Score":>12}')
    print('-'*75)

    for ds in ['FD001','FD002','FD003','FD004']:
        pr = PAPER_RMSE[ds]
        ps = PAPER_SCORE[ds]
        sr = all_sikd_results[ds]['rmse']
        ss = all_sikd_results[ds]['score']
        delta = sr - pr
        print(f'{ds:<10} {pr:>12.2f} {sr:>12.2f} {delta:>+10.2f} {ps:>14.1f} {ss:>12.1f}')
    print('='*75)
    print('\nResults saved to sikd_results_all.json ✓')
