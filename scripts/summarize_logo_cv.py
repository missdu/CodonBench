import json
with open('./results/logo_cv/logo_cv_missense_results.json') as f:
    d = json.load(f)
print('=== Task 1 (Missense) LOGO-CV Results ===')
for k,v in d.items():
    if 'auc_mean' in v:
        std = v.get('standard_cv_auc', 'N/A')
        print('%-20s LOGO=%.4f+/-%.4f  StdCV=%s  n_genes=%d' % (k, v["auc_mean"], v["auc_std"], std, v["n_genes"]))
    elif 'error' in v:
        print('%-20s ERROR: %s' % (k, v["error"][:80]))

with open('./results/logo_cv/logo_cv_results.json') as f:
    syn = json.load(f)
print()
print('=== Task 2 (Synonymous) LOGO-CV Results (existing) ===')
for k,v in syn.items():
    if 'auc_mean' in v:
        std = v.get('standard_cv_auc', 'N/A')
        print('%-20s LOGO=%.4f+/-%.4f  StdCV=%s  n_genes=%d' % (k, v["auc_mean"], v["auc_std"], std, v["n_genes"]))

print()
print('=== Cross-task comparison ===')
print('%-20s | %10s | %10s | %10s | %10s' % ('Method', 'T1 LOGO', 'T1 StdCV', 'T2 LOGO', 'T2 StdCV'))
print('-' * 70)
common = set(d.keys()) & set(syn.keys())
for k in sorted(common):
    t1 = d.get(k, {})
    t2 = syn.get(k, {})
    t1_logo = '%.4f' % t1["auc_mean"] if 'auc_mean' in t1 else 'N/A'
    t1_std = str(t1.get("standard_cv_auc", 'N/A')) if 'standard_cv_auc' in t1 else 'N/A'
    t2_logo = '%.4f' % t2["auc_mean"] if 'auc_mean' in t2 else 'N/A'
    t2_std = str(t2.get("standard_cv_auc", 'N/A')) if 'standard_cv_auc' in t2 else 'N/A'
    print('%-20s | %10s | %10s | %10s | %10s' % (k, t1_logo, t1_std, t2_logo, t2_std))
