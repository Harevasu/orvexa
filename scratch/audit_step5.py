import os
import sys
import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add src to path
sys.path.insert(0, os.path.abspath("src"))
from orvexa.ranking_metrics import compute_ranking_metrics

def compute_all_metrics(csv_path, threshold_log10=-5.0):
    df = pd.read_csv(csv_path)
    y_true = df['final_risk'].values
    y_pred = df['predicted_risk'].values
    n_samples = len(y_true)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    spearman_rho, _ = stats.spearmanr(y_true, y_pred)
    
    # Operational ranking using canonical module
    ranking = compute_ranking_metrics(
        y_true_risk=list(y_true),
        y_pred_risk=list(y_pred),
        threshold_log10=threshold_log10
    )
    
    # Tail diagnostics on critical events
    is_crit = y_true >= threshold_log10
    n_crit = int(np.sum(is_crit))
    if n_crit > 0:
        residuals = y_pred[is_crit] - y_true[is_crit]
        tail_mean_res = float(np.mean(residuals))
        tail_med_res = float(np.median(residuals))
        tail_mae = float(np.mean(np.abs(residuals)))
        tail_rmse = float(np.sqrt(np.mean(residuals**2)))
    else:
        tail_mean_res = tail_med_res = tail_mae = tail_rmse = 0.0
        
    return {
        'n_samples': n_samples,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson': float(pearson_r),
        'spearman': float(spearman_rho),
        'n_crit': n_crit,
        'ranking': ranking,
        'tail_mean_res': tail_mean_res,
        'tail_med_res': tail_med_res,
        'tail_mae': tail_mae,
        'tail_rmse': tail_rmse,
        'target_mean': float(np.mean(y_true)),
        'target_std': float(np.std(y_true, ddof=0)),
        'pred_mean': float(np.mean(y_pred)),
        'pred_std': float(np.std(y_pred, ddof=0)),
    }

predictions_map = {
    'H2': 'data/processed/predictions/phase3b/blind_test/tcn_M4_h2.0_test_predictions.csv',
    'H3': 'data/processed/predictions/phase3b/blind_test/tcn_M4_h3.0_test_predictions.csv',
    'H5': 'data/processed/predictions/phase3b/blind_test/tcn_M4_h5.0_test_predictions.csv',
    'H6': 'data/processed/predictions/phase4a/blind_test/tcn_M4_h6.0_test_predictions.csv'
}

results = {}
for h, path in predictions_map.items():
    m = compute_all_metrics(path)
    results[h] = m
    print(f"=== Horizon {h} ({path}) ===")
    print(f"N={m['n_samples']}, MAE={m['mae']:.5f}, RMSE={m['rmse']:.5f}, R2={m['r2']:.5f}, Pearson={m['pearson']:.5f}, Spearman={m['spearman']:.5f}")
    print(f"Critical Count={m['n_crit']}")
    b1 = m['ranking']['budget_pct_1']
    b5 = m['ranking']['budget_pct_5']
    b10 = m['ranking']['budget_pct_10']
    print(f"Recall@1%={b1['recall']*100:.2f}% ({b1['true_positives']}/{m['n_crit']}), Precision@1%={b1['precision']*100:.2f}% (k={b1['cutoff_rank_k']})")
    print(f"Recall@5%={b5['recall']*100:.2f}% ({b5['true_positives']}/{m['n_crit']}), Precision@5%={b5['precision']*100:.2f}% (k={b5['cutoff_rank_k']})")
    print(f"Recall@10%={b10['recall']*100:.2f}% ({b10['true_positives']}/{m['n_crit']}), Precision@10%={b10['precision']*100:.2f}% (k={b10['cutoff_rank_k']}), Missed@10%={b10['missed_high_risk']}")
    print(f"Tail Mean Res={m['tail_mean_res']:.5f}, Tail Med Res={m['tail_med_res']:.5f}")
    print()

# Now compare with authoritative JSON/CSV metrics
with open('reports/phase3b/step4b_blind_test_summary.json', 'r') as f:
    p3b_json = json.load(f)

with open('reports/phase4a/step4_blind_test_summary.json', 'r') as f:
    p4a_json = json.load(f)

print("=== Checking Discrepancies with Authoritative JSON artifacts ===")
for h in ['H2', 'H3', 'H5']:
    auth = p3b_json['horizons'][h]
    reg = auth['regression_metrics']
    comp = results[h]
    mae_diff = abs(reg['mae'] - comp['mae'])
    rmse_diff = abs(reg['rmse'] - comp['rmse'])
    r2_diff = abs(reg['r2'] - comp['r2'])
    pear_diff = abs(reg['pearson_correlation'] - comp['pearson'])
    spear_diff = abs(reg['spearman_correlation'] - comp['spearman'])
    print(f"{h}: MAE diff={mae_diff:.2e}, RMSE diff={rmse_diff:.2e}, R2 diff={r2_diff:.2e}, Pear diff={pear_diff:.2e}, Spear diff={spear_diff:.2e}")

auth_h6 = p4a_json['metrics']
comp_h6 = results['H6']
print(f"H6: MAE diff={abs(auth_h6['test_mae'] - comp_h6['mae']):.2e}, RMSE diff={abs(auth_h6['test_rmse'] - comp_h6['rmse']):.2e}, R2 diff={abs(auth_h6['test_r2'] - comp_h6['r2']):.2e}, Pear diff={abs(auth_h6['test_pearson'] - comp_h6['pearson']):.2e}, Spear diff={abs(auth_h6['test_spearman'] - comp_h6['spearman']):.2e}")
