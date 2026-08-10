import json
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy import stats

def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    s_pooled = np.sqrt(((n1-1)*np.std(x1,ddof=1)**2 + (n2-1)*np.std(x2,ddof=1)**2) / (n1+n2-2))
    return (np.mean(x1) - np.mean(x2)) / s_pooled if s_pooled > 0 else 0.0

def bootstrap_ci(data, stat_func=np.mean, n_boot=10000, ci=0.95):
    boots = [stat_func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return [round(np.percentile(boots, alpha*100),4), round(np.percentile(boots, (1-alpha)*100),4)]

def benjamini_hochberg(p_values):
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adjusted = np.zeros(n)
    adjusted[-1] = sorted_p[-1]
    for i in range(n-2, -1, -1):
        adjusted[i] = min(adjusted[i+1], sorted_p[i] * n / (i+1))
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return np.clip(result, 0, 1)

np.random.seed(42)
results_dir = Path('results')
comp = json.loads((results_dir / 'comprehensive_comparison.json').read_text())

esm2_task2 = json.loads((results_dir / 'esm2_task2.json').read_text())
esm2_task3 = json.loads((results_dir / 'esm2_task3.json').read_text())

esm2_entries = []
for esm2_data in [esm2_task2, esm2_task3]:
    if isinstance(esm2_data, dict):
        esm2_entries.append({
            'model': esm2_data['model'].lower().replace('-','_'),
            'task': esm2_data['task'],
            'type': esm2_data.get('model_type', 'pLM'),
            'auc_mean': esm2_data['ROC-AUC_mean'],
            'auc_std': esm2_data['ROC-AUC_std'],
            'auc_folds': esm2_data['ROC-AUC_folds'],
            'n_samples': esm2_data['n_samples'],
            'params_M': esm2_data['params_M'],
        })

all_results = list(comp) + esm2_entries

all_comparisons = []
all_p_values = []

for task in ['task2_missense', 'task3_synonymous']:
    task_models = [r for r in all_results if r.get('task') == task and 'auc_folds' in r]
    for (m1, m2) in combinations(task_models, 2):
        folds1, folds2 = np.array(m1['auc_folds']), np.array(m2['auc_folds'])
        if len(folds1) != len(folds2):
            continue
        d = cohens_d(folds1, folds2)
        t_stat, p_val = stats.ttest_rel(folds1, folds2)
        ci1 = bootstrap_ci(folds1)
        ci2 = bootstrap_ci(folds2)
        diff = m1['auc_mean'] - m2['auc_mean']
        all_comparisons.append({
            'task': task,
            'model_1': m1['model'],
            'model_2': m2['model'],
            'type_1': m1.get('type',''),
            'type_2': m2.get('type',''),
            'auc_1': round(m1['auc_mean'], 4),
            'auc_2': round(m2['auc_mean'], 4),
            'diff': round(diff, 4),
            'cohens_d': round(d, 3),
            'effect_size': 'large' if abs(d)>0.8 else 'medium' if abs(d)>0.5 else 'small' if abs(d)>0.2 else 'negligible',
            'p_value': float(p_val),
            'significant_005': bool(p_val < 0.05),
            'bootstrap_ci_1': ci1,
            'bootstrap_ci_2': ci2,
        })
        all_p_values.append(p_val)

if all_p_values:
    fdr_adjusted = benjamini_hochberg(all_p_values)
    for i, comp_item in enumerate(all_comparisons):
        comp_item['p_fdr'] = round(float(fdr_adjusted[i]), 6)
        comp_item['significant_fdr_005'] = bool(fdr_adjusted[i] < 0.05)

model_ci = []
for r in all_results:
    if 'auc_folds' in r:
        folds = np.array(r['auc_folds'])
        ci = bootstrap_ci(folds)
        model_ci.append({
            'model': r['model'], 'task': r['task'], 'type': r.get('type',''),
            'auc_mean': r['auc_mean'], 'auc_std': r['auc_std'],
            'bootstrap_ci_95': ci, 'n_folds': len(folds)
        })

output = {
    'pairwise_comparisons': all_comparisons,
    'n_comparisons': len(all_comparisons),
    'n_significant_raw': sum(1 for c in all_comparisons if c.get('significant_005')),
    'n_significant_fdr': sum(1 for c in all_comparisons if c.get('significant_fdr_005')),
    'model_bootstrap_ci': model_ci
}

(results_dir / 'statistical_analysis.json').write_text(json.dumps(output, indent=2))

print(f"Total comparisons: {output['n_comparisons']}")
print(f"Significant (raw p<0.05): {output['n_significant_raw']}")
print(f"Significant (FDR q<0.05): {output['n_significant_fdr']}")

print("\n=== Key Comparisons (cLM vs pLM, cLM vs DNA LM) ===")
key_pairs = [
    ('task3_synonymous', 'codonbert', 'esm-2-650m', 'cLM vs pLM on synonymous'),
    ('task2_missense', 'codonbert', 'esm-2-650m', 'cLM vs pLM on missense'),
    ('task3_synonymous', 'codonbert', 'nt-500m', 'cLM vs DNA LM on synonymous'),
    ('task2_missense', 'codonbert', 'encodon-80m', 'cLM vs cLM on missense'),
    ('task3_synonymous', 'codonbert', 'codonbert_hf', 'cLM vs cLM on synonymous'),
    ('task2_missense', 'onehot_pos', 'codonbert', 'Traditional vs cLM on missense'),
    ('task3_synonymous', 'onehot_pos', 'codonbert', 'Traditional vs cLM on synonymous'),
]
for task, m1, m2, label in key_pairs:
    match = [c for c in all_comparisons if c['task']==task and 
             ((c['model_1']==m1 and c['model_2']==m2) or (c['model_1']==m2 and c['model_2']==m1))]
    if match:
        c = match[0]
        print(f"  {label}: d={c['cohens_d']:+.3f}({c['effect_size']}) | p={c['p_value']:.4f} | p_FDR={c.get('p_fdr',0):.4f}")

print("\n=== Bootstrap 95% CI for all models ===")
for m in model_ci:
    ci = m['bootstrap_ci_95']
    print(f"  {m['model']:15s} {m['task']:20s}: AUC={m['auc_mean']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

print("\nDone. Saved to results/statistical_analysis.json")