import json, os
from scipy import stats
import numpy as np

results = {}

# 1. comprehensive_comparison.json (original models)
with open('results/comprehensive_comparison.json') as f:
    data = json.load(f)
for d in data:
    if d['task'] == 'task3_synonymous':
        results[d['model']] = {'folds': d['auc_folds'], 'mean': d['auc_mean'], 'std': d['auc_std']}

# 2. ESM-2
fname = 'results/esm2_task3.json'
if os.path.exists(fname):
    with open(fname) as f:
        d = json.load(f)
    folds_key = None
    for k in ['auc_folds', 'ROC-AUC_folds']:
        if k in d:
            folds_key = k
            break
    if folds_key:
        mean_key = folds_key.replace('_folds', '_mean')
        std_key = folds_key.replace('_folds', '_std')
        results['esm2-650m'] = {'folds': d[folds_key], 'mean': d.get(mean_key, d.get('auc_mean')), 'std': d.get(std_key, d.get('auc_std'))}

# 3. EnCodon-620M
fname = 'results/encodon-620m_task3_synonymous.json'
if os.path.exists(fname):
    with open(fname) as f:
        d = json.load(f)
    folds_key = None
    for k in ['auc_folds', 'lr_auc_folds', 'ROC-AUC_folds']:
        if k in d:
            folds_key = k
            break
    if folds_key:
        mean_key = folds_key.replace('_folds', '_mean')
        std_key = folds_key.replace('_folds', '_std')
        results['encodon-620m'] = {'folds': d[folds_key], 'mean': d.get(mean_key, d.get('auc_mean')), 'std': d.get(std_key, d.get('auc_std'))}

# 4. CodonTransformer (array format, no folds)
fname = 'results/codontransformer_summary.json'
if os.path.exists(fname):
    with open(fname) as f:
        d = json.load(f)
    if isinstance(d, list):
        for item in d:
            if item.get('task') == 'task3_synonymous':
                folds = item.get('lr_auc_folds', item.get('auc_folds', None))
                if folds:
                    results['codontransformer'] = {'folds': folds, 'mean': item.get('lr_auc_mean', item.get('auc_mean')), 'std': item.get('lr_auc_std', item.get('auc_std'))}
                else:
                    results['codontransformer'] = {'folds': None, 'mean': item.get('lr_auc_mean', 0.713), 'std': item.get('lr_auc_std', 0.016)}
                break

# 5. CaLM (check format)
fname = 'results/calm_summary.json'
if os.path.exists(fname):
    with open(fname) as f:
        d = json.load(f)
    if isinstance(d, list):
        for item in d:
            if item.get('task') == 'task3_synonymous':
                folds = item.get('lr_auc_folds', item.get('auc_folds', None))
                results['calm'] = {'folds': folds, 'mean': item.get('lr_auc_mean', item.get('auc_mean')), 'std': item.get('lr_auc_std', item.get('auc_std'))}
                break
    elif isinstance(d, dict):
        for key in ['task3_synonymous', 'task3']:
            if key in d:
                sub = d[key]
                folds = sub.get('lr_auc_folds', sub.get('auc_folds', None))
                results['calm'] = {'folds': folds, 'mean': sub.get('lr_auc_mean', sub.get('auc_mean')), 'std': sub.get('lr_auc_std', sub.get('auc_std'))}
                break

# 6. Mistral (array format)
fname = 'results/mistral_codon_summary.json'
if os.path.exists(fname):
    with open(fname) as f:
        d = json.load(f)
    if isinstance(d, list):
        for item in d:
            if item.get('task') == 'task3_synonymous':
                model_name = item.get('model', '').lower().replace('codon', 'codon-')
                folds = item.get('lr_auc_folds', item.get('auc_folds', None))
                results[model_name] = {'folds': folds, 'mean': item.get('lr_auc_mean', item.get('auc_mean')), 'std': item.get('lr_auc_std', item.get('auc_std'))}
    elif isinstance(d, dict):
        for model_key in d:
            subd = d[model_key]
            if isinstance(subd, dict) and 'task3' in subd:
                sub = subd['task3']
                folds = sub.get('lr_auc_folds', sub.get('auc_folds', None))
                results[model_key] = {'folds': folds, 'mean': sub.get('lr_auc_mean', sub.get('auc_mean')), 'std': sub.get('lr_auc_std', sub.get('auc_std'))}

# 7. cdsBERT (array format)
fname = 'results/cdsbert_char_summary.json'
if os.path.exists(fname):
    with open(fname) as f:
        d = json.load(f)
    if isinstance(d, list):
        for item in d:
            if item.get('task') == 'task3_synonymous':
                model_name = item.get('model', '').lower().replace('plus', '-plus')
                folds = item.get('lr_auc_folds', item.get('auc_folds', None))
                results[model_name] = {'folds': folds, 'mean': item.get('lr_auc_mean', item.get('auc_mean')), 'std': item.get('lr_auc_std', item.get('auc_std'))}
    elif isinstance(d, dict):
        for model_key in d:
            subd = d[model_key]
            if isinstance(subd, dict) and 'task3' in subd:
                sub = subd['task3']
                folds = sub.get('lr_auc_folds', sub.get('auc_folds', None))
                results[model_key] = {'folds': folds, 'mean': sub.get('lr_auc_mean', sub.get('auc_mean')), 'std': sub.get('lr_auc_std', sub.get('auc_std'))}

print("=== All Task 3 per-fold AUCs ===")
for m in sorted(results.keys()):
    r = results[m]
    folds_str = [round(x,4) for x in r['folds']] if r['folds'] is not None else "NO FOLDS"
    print(f"  {m}: mean={r['mean']:.4f}, std={r['std']:.4f}, folds={folds_str}")

# Now compute p-values
codon_level_keys = ['encodon-620m', 'codonbert', 'codontransformer', 'codonbert_hf', 'encodon-80m']
esm2_key = 'esm2-650m'

if esm2_key not in results:
    print(f"\nERROR: {esm2_key} not found in results!")
else:
    esm2_folds = np.array(results[esm2_key]['folds'])
    
    print("\n=== Paired t-tests: codon-level cLM > ESM-2 (one-sided) ===")
    for m in codon_level_keys:
        if m in results:
            if results[m]['folds'] is None:
                print(f"  {m}: NO per-fold data available")
                continue
            cLM_folds = np.array(results[m]['folds'])
            if len(cLM_folds) == len(esm2_folds):
                diff = cLM_folds - esm2_folds
                t_stat, p_val = stats.ttest_1samp(diff, 0, alternative='greater')
                print(f"  {m}: mean_diff={diff.mean():.4f}, t={t_stat:.3f}, p={p_val:.4f}")
            else:
                print(f"  {m}: fold count mismatch ({len(cLM_folds)} vs {len(esm2_folds)})")
        else:
            print(f"  {m}: NOT FOUND in results")
    
    # Group-level test
    print("\n=== Group-level test: mean(codon-level cLMs) vs ESM-2 ===")
    codon_level_folds_list = []
    available = []
    for m in codon_level_keys:
        if m in results and results[m]['folds'] is not None and len(results[m]['folds']) == len(esm2_folds):
            codon_level_folds_list.append(results[m]['folds'])
            available.append(m)
    if codon_level_folds_list:
        codon_level_folds = np.array(codon_level_folds_list)
        codon_mean_per_fold = codon_level_folds.mean(axis=0)
        diff_group = codon_mean_per_fold - esm2_folds
        t_group, p_group = stats.ttest_1samp(diff_group, 0, alternative='greater')
        print(f"  Available models: {available}")
        print(f"  Codon-level mean per fold: {[round(x,4) for x in codon_mean_per_fold]}")
        print(f"  ESM-2 per fold: {[round(x,4) for x in esm2_folds]}")
        print(f"  Diff per fold: {[round(x,4) for x in diff_group]}")
        print(f"  Mean diff: {diff_group.mean():.4f}, t={t_group:.3f}, p={p_group:.4f}")
    
    # Character-level vs ESM-2
    print("\n=== Character-level cLMs vs ESM-2 ===")
    for m in ['cdsbert', 'cdsbert-plus']:
        if m in results:
            if results[m]['folds'] is None:
                print(f"  {m}: NO per-fold data")
                continue
            cLM_folds = np.array(results[m]['folds'])
            if len(cLM_folds) == len(esm2_folds):
                diff = cLM_folds - esm2_folds
                t_stat, p_val = stats.ttest_1samp(diff, 0, alternative='greater')
                print(f"  {m}: mean_diff={diff.mean():.4f}, t={t_stat:.3f}, p={p_val:.4f}")

    # MoE vs ESM-2
    print("\n=== MoE cLMs vs ESM-2 ===")
    for m in ['mistral-codon-117m', 'mistral-codon-16m', 'mistral-codon-1m']:
        if m in results:
            if results[m]['folds'] is None:
                print(f"  {m}: NO per-fold data")
                continue
            cLM_folds = np.array(results[m]['folds'])
            if len(cLM_folds) == len(esm2_folds):
                diff = cLM_folds - esm2_folds
                t_stat, p_val = stats.ttest_1samp(diff, 0, alternative='greater')
                print(f"  {m}: mean_diff={diff.mean():.4f}, t={t_stat:.3f}, p={p_val:.4f}")

    print("\n=== SUMMARY FOR PAPER ===")
    if codon_level_folds_list:
        print(f"  Codon-level cLMs (n={len(available)}) mean Task 3 AUC = {codon_level_folds.mean():.3f}")
        print(f"  ESM-2 Task 3 AUC = {esm2_folds.mean():.3f}")
        print(f"  Group-level one-sided p = {p_group:.4f}")
        if p_group < 0.05:
            print(f"  => SIGNIFICANT at p<0.05: codon-level cLMs > ESM-2 on synonymous variants")
        else:
            print(f"  => NOT significant at p<0.05")