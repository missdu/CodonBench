"""Comprehensive data audit: paper text vs experiment results."""
import json, os

BASE = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\results'

# Load all data sources
with open(os.path.join(BASE, 'comprehensive_comparison.json'), encoding='utf-8') as f:
    comp = json.load(f)
with open(os.path.join(BASE, 'cLM_mlp_independent_results.json'), encoding='utf-8') as f:
    mlp_data = json.load(f)
with open(os.path.join(BASE, 'supplementary', 'esm_mlp_probing_results.json'), encoding='utf-8') as f:
    esm_mlp = json.load(f)
with open(os.path.join(BASE, 'supplementary', 'from_scratch_ablation_all_conditions.json'), encoding='utf-8') as f:
    ablation = json.load(f)
with open(os.path.join(BASE, 'within_protein_llr_results.json'), encoding='utf-8') as f:
    llr_data = json.load(f)
with open(os.path.join(BASE, 'supplementary', 'layerwise_analysis_results.json'), encoding='utf-8') as f:
    layer_data = json.load(f)
with open(os.path.join(BASE, 'delong_from_scratch_results.json'), encoding='utf-8') as f:
    delong = json.load(f)

# Also load individual model files for missing data
extra_models = {}
for fn in os.listdir(BASE):
    if fn.endswith('.json') and 'task3' in fn:
        with open(os.path.join(BASE, fn), encoding='utf-8') as f:
            d = json.load(f)
        if isinstance(d, dict) and 'lr_auc_mean' in d:
            extra_models[d.get('model', fn)] = d

# Build lookup tables
lr_syn = {}
lr_mis = {}
for e in comp:
    if 'task3' in e.get('task', ''):
        lr_syn[e['model']] = e['auc_mean']
    elif 'task2' in e.get('task', ''):
        lr_mis[e['model']] = e['auc_mean']

# Add extra models
for name, d in extra_models.items():
    if 'task3' in d.get('task', ''):
        lr_syn[name] = d['lr_auc_mean']
    if 'task2' in d.get('task', ''):
        lr_mis[name] = d['lr_auc_mean']

mlp_syn = {}
mlp_mis = {}
for e in mlp_data:
    if 'task3' in e.get('task', ''):
        mlp_syn[e['model']] = e['test_mlp']
    elif 'task2' in e.get('task', ''):
        mlp_mis[e['model']] = e['test_mlp']
for e in esm_mlp:
    key = e['model'].replace('ESM-2-650M', 'esm2_650m').replace('ESM-1b-650M', 'esm1b').replace('-', '_').replace(' ', '').lower()
    if 'esm2' in e['model'].lower() or 'esm-2' in e['model'].lower():
        key = 'esm2_650m'
    elif 'esm1b' in e['model'].lower() or 'esm-1b' in e['model'].lower():
        key = 'esm1b'
    if 'task3' in e.get('task', ''):
        mlp_syn[key] = e['test_mlp']
    elif 'task2' in e.get('task', ''):
        mlp_mis[key] = e['test_mlp']

llr_syn = {}
llr_mis = {}
for e in llr_data:
    if 'task3' in e.get('task', ''):
        llr_syn[e['model']] = e['auc']
    elif 'task2' in e.get('task', ''):
        llr_mis[e['model']] = e['auc']

print("=" * 80)
print("R1: SynPath LR AUC (paper vs data)")
print("=" * 80)
paper_r1_syn = {
    'EnCodon-620M': 0.785, 'CodonBERT': 0.734, 'CodonTransformer': 0.713,
    'CodonBERT-HF': 0.705, 'EnCodon-80M': 0.686, 'CaLM': 0.673,
    'NT-v2-500M': 0.699, 'cdsBERT': 0.598, 'ESM-2': 0.680, 'ESM-1b': 0.682,
    'onehot_pos': 0.891, 'AlphaMissense': 0.440,
}
key_map = {
    'EnCodon-620M': 'encodon-620m', 'CodonBERT': 'codonbert', 'CodonTransformer': 'codontransformer',
    'CodonBERT-HF': 'codonbert_hf', 'EnCodon-80M': 'encodon-80m', 'CaLM': 'calm',
    'NT-v2-500M': 'nt-500m', 'cdsBERT': 'cdsbert', 'ESM-2': 'esm2_650m', 'ESM-1b': 'esm1b',
    'onehot_pos': 'onehot_pos',
}
for name, paper_val in paper_r1_syn.items():
    if name == 'AlphaMissense':
        print(f"  {name}: paper={paper_val:.3f}, data=external (no local file)")
        continue
    key = key_map.get(name, name.lower())
    data_val = lr_syn.get(key)
    if data_val is not None:
        match = "OK" if abs(paper_val - data_val) < 0.002 else "MISMATCH"
        print(f"  {name}: paper={paper_val:.3f}, data={data_val:.4f} {match}")
    else:
        print(f"  {name}: paper={paper_val:.3f}, data=NOT FOUND")

print()
print("=" * 80)
print("R1: MisPath LR AUC (paper vs data)")
print("=" * 80)
paper_r1_mis = {
    'ESM-2': 0.719, 'ESM-1b': 0.711, 'CodonTransformer': 0.694,
}
for name, paper_val in paper_r1_mis.items():
    key = key_map.get(name, name.lower())
    data_val = lr_mis.get(key)
    if data_val is not None:
        match = "OK" if abs(paper_val - data_val) < 0.002 else "MISMATCH"
        print(f"  {name}: paper={paper_val:.3f}, data={data_val:.4f} {match}")
    else:
        print(f"  {name}: paper={paper_val:.3f}, data=NOT FOUND")

print()
print("=" * 80)
print("R2: From-scratch ablation (paper vs S5 data)")
print("=" * 80)
# S5 data from ablation JSON
dsl = ablation['data_scale_layer']
msl = ablation.get('model_scale_layer', {})
conditions = [
    ('v1', dsl['v1']), ('v3a', dsl['v3a']), ('v3b', dsl['v3b']),
    ('v2', dsl['v2']), ('v4', msl.get('v4', {})),
]
paper_r2 = {
    'v1': {'codon_lr': 0.701, 'char_lr': 0.711, 'codon_mlp': 0.719, 'char_mlp': 0.627},
    'v3a': {'codon_lr': 0.670, 'char_lr': 0.647, 'codon_mlp': 0.720, 'char_mlp': 0.606},
    'v3b': {'codon_lr': 0.706, 'char_lr': 0.675, 'codon_mlp': 0.792, 'char_mlp': 0.647},
    'v4': {'codon_lr': 0.706, 'char_lr': 0.567, 'codon_mlp': 0.831, 'char_mlp': 0.590},
    'v2': {'codon_lr': 0.669, 'char_lr': 0.667, 'codon_mlp': 0.643, 'char_mlp': 0.662},
}
for vname, vdata in conditions:
    p = paper_r2[vname]
    codon = vdata.get('codon', {})
    char = vdata.get('char', {})
    print(f"  {vname}:")
    for field, data_key in [('codon_lr', 'synpath_lr_mean'), ('codon_mlp', 'synpath_mlp'),
                             ('char_lr', 'synpath_lr_mean'), ('char_mlp', 'synpath_mlp')]:
        pval = p[field]
        src = codon if 'codon' in field else char
        dval = src.get(data_key)
        if dval is not None:
            match = "OK" if abs(pval - dval) < 0.002 else "MISMATCH"
            print(f"    {field}: paper={pval:.3f}, data={dval:.3f} {match}")
        else:
            print(f"    {field}: paper={pval:.3f}, data=NOT FOUND (v4 char training in progress)")

print()
print("=" * 80)
print("R2: DeLong stats (paper vs delong JSON)")
print("=" * 80)
paper_delong = {
    'v1': {'mlp_delta_pp': 9.2, 'lr_p': 0.816},
    'v3a': {'mlp_delta_pp': 11.4, 'lr_p': 0.556},
    'v3b': {'mlp_delta_pp': 14.5, 'lr_p': 0.559},
    'v4': {'mlp_delta_pp': 24.1, 'lr_p': 0.000},
}
for vname in ['v1', 'v3a', 'v3b', 'v4']:
    p = paper_delong[vname]
    d = delong.get(vname, {})
    if 'mlp' in d:
        d_mlp_delta = d['mlp']['delta'] * 100
        match_mlp = "OK" if abs(p['mlp_delta_pp'] - d_mlp_delta) < 0.5 else f"MISMATCH (DeLong={d_mlp_delta:.1f}, paper uses S5反算={p['mlp_delta_pp']:.1f})"
        print(f"  {vname} MLP delta: paper(S5)={p['mlp_delta_pp']:.1f}pp, DeLong={d_mlp_delta:.1f}pp {match_mlp}")
    if 'lr' in d:
        d_lr_p = d['lr']['mean_p']
        match_lr = "OK" if abs(p['lr_p'] - d_lr_p) < 0.01 else f"MISMATCH"
        print(f"  {vname} LR p: paper={p['lr_p']:.3f}, DeLong={d_lr_p:.3f} {match_lr}")

print()
print("=" * 80)
print("R3: LLR AUC (paper vs data)")
print("=" * 80)
paper_llr = {
    'CodonBERT SynPath': 0.526, 'CodonBERT-HF SynPath': 0.484, 'EnCodon-80M SynPath': 0.495,
    'CodonBERT-HF MisPath': 0.763, 'EnCodon-80M MisPath': 0.656,
}
for key, pval in paper_llr.items():
    parts = key.split()
    model = parts[0]
    task = parts[1]
    if task == 'SynPath':
        dval = llr_syn.get(model.lower().replace('-', '_'))
    else:
        dval = llr_mis.get(model.lower().replace('-', '_'))
    if dval is not None:
        match = "OK" if abs(pval - dval) < 0.002 else "MISMATCH"
        print(f"  {key}: paper={pval:.3f}, data={dval:.4f} {match}")
    else:
        print(f"  {key}: paper={pval:.3f}, data=NOT FOUND")

print()
print("=" * 80)
print("R3: MLP AUC (paper vs data)")
print("=" * 80)
paper_mlp = {
    'CodonBERT SynPath': 0.849, 'CodonBERT-HF SynPath': 0.816, 'EnCodon-80M SynPath': 0.812,
}
for key, pval in paper_mlp.items():
    parts = key.split()
    model = parts[0].lower().replace('-', '_')
    dval = mlp_syn.get(model)
    if dval is not None:
        match = "OK" if abs(pval - dval) < 0.002 else "MISMATCH"
        print(f"  {key}: paper={pval:.3f}, data={dval:.4f} {match}")
    else:
        print(f"  {key}: paper={pval:.3f}, data=NOT FOUND")

print()
print("=" * 80)
print("R3: LR-to-MLP gains (paper vs data)")
print("=" * 80)
paper_gains = "+1.5 to +24.3 pp"
all_gains = []
for m in ['encodon-620m', 'codonbert', 'codontransformer', 'codonbert_hf', 'encodon-80m',
          'calm', 'mistralcodon117m', 'mistralcodon16m', 'mistralcodon1m']:
    lr = lr_syn.get(m)
    ml = mlp_syn.get(m)
    if lr and ml:
        gain = (ml - lr) * 100
        all_gains.append((m, gain))
if all_gains:
    gains_sorted = sorted(all_gains, key=lambda x: x[1])
    print(f"  Paper says: {paper_gains}")
    print(f"  Data range: {gains_sorted[0][1]:+.1f} to {gains_sorted[-1][1]:+.1f} pp")
    for m, g in gains_sorted:
        print(f"    {m}: {g:+.1f} pp")

print()
print("=" * 80)
print("Abstract: MLP gap range")
print("=" * 80)
print("  Abstract says: +7.7 to +25.3 pp (OLD, from V28 original)")
print("  Should be: +9.2 to +24.1 pp (S5反算锁死)")