import json, os

base = './results/'

models_files = {
    'calm': ('calm_eval/calm_task3_synonymous.json', 'calm_eval/calm_task2_missense.json'),
    'cdsbert': ('cdsbert_char/cdsBERT_task3_synonymous.json', 'cdsbert_char/cdsBERT_task2_missense.json'),
    'cdsbert_plus': ('cdsbert_char/cdsBERT-plus_task3_synonymous.json', 'cdsbert_char/cdsBERT-plus_task2_missense.json'),
    'mistral_117m': ('mistral_codon/mistral-codon-117m_task3_synonymous.json', 'mistral_codon/mistral-codon-117m_task2_missense.json'),
    'mistral_16m': ('mistral_codon/mistral-codon-16m_task3_synonymous.json', 'mistral_codon/mistral-codon-16m_task2_missense.json'),
    'mistral_1m': ('mistral_codon/mistral-codon-1m_task3_synonymous.json', 'mistral_codon/mistral-codon-1m_task2_missense.json'),
}

for name, (syn_f, mis_f) in models_files.items():
    with open(os.path.join(base, syn_f)) as f:
        syn = json.load(f)
    with open(os.path.join(base, mis_f)) as f:
        mis = json.load(f)
    print(f'{name}:')
    print(f'  SynPath: LR={syn["lr_auc_mean"]:.4f}, MLP={syn["mlp_auc_test"]:.4f}, gain={syn["mlp_auc_test"]-syn["lr_auc_mean"]:+.4f}')
    print(f'  MisPath: LR={mis["lr_auc_mean"]:.4f}, MLP={mis["mlp_auc_test"]:.4f}, gain={mis["mlp_auc_test"]-mis["lr_auc_mean"]:+.4f}')