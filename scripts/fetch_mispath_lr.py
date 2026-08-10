import json, sys

base = './results/'
versions = {
    'v1': ('from_scratch_codon_results.json', 'from_scratch_char_results.json'),
    'v3a': ('from_scratch_codon-v3a-real100ep_results.json', 'from_scratch_char-v3a-real100ep_results.json'),
    'v3b': ('from_scratch_codon-v3b-ensembl_results.json', 'from_scratch_char-v3b-ensembl_results.json'),
    'v2': ('from_scratch_codon-v2_results.json', 'from_scratch_char-v2_results.json'),
}

output = {}
for ver, (cf, chf) in versions.items():
    with open(base + cf) as f:
        codon = json.load(f)
    with open(base + chf) as f:
        char = json.load(f)
    
    def extract(data, key):
        if key in data:
            return data[key]
        if 'results' in data and key in data['results']:
            return data['results'][key]
        return None
    
    ver_data = {}
    for task in ['synpath', 'mispath']:
        for probe in ['lr', 'mlp']:
            key = f'{task}_{probe}'
            c_val = extract(codon, key)
            h_val = extract(char, key)
            ver_data[f'codon_{key}'] = c_val
            ver_data[f'char_{key}'] = h_val
    
    # Also try to get LR fold-level data
    for task in ['synpath', 'mispath']:
        key = f'{task}_lr_folds'
        c_val = extract(codon, key)
        h_val = extract(char, key)
        if c_val: ver_data[f'codon_{key}'] = c_val
        if h_val: ver_data[f'char_{key}'] = h_val
    
    # Try lr_mean and lr_std
    for task in ['synpath', 'mispath']:
        for stat in ['lr_mean', 'lr_std']:
            key = f'{task}_{stat}'
            c_val = extract(codon, key)
            h_val = extract(char, key)
            if c_val is not None: ver_data[f'codon_{key}'] = c_val
            if h_val is not None: ver_data[f'char_{key}'] = h_val
    
    output[ver] = ver_data

print(json.dumps(output, indent=2))