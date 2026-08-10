#!/usr/bin/env python3
"""
Evaluate cLMs on Fungal expression dataset (independent data source).
This supplements the regression tasks to address the limitation that
all three regression tasks come from the CodonBERT benchmark.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

sys.path.insert(0, './src')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

DATA_DIR = './data/regression'
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_fungal_data():
    df = pd.read_csv(f'{DATA_DIR}/Fungal_expression.csv')
    # Convert RNA (U) to DNA (T) for DNA-tokenized models
    df['CDS'] = df['Sequence'].str.replace('U', 'T')
    return df

def get_embeddings(model_name, sequences, batch_size=8):
    """Extract embeddings for a list of CDS sequences."""
    import torch
    
    if model_name == 'codonbert':
        from models.loader import load_model
        model, tokenizer = load_model('codonbert')
        model.eval()
        device = next(model.parameters()).device
        
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                # Mean pool over sequence length
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                embeds = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
                all_embeds.append(embeds.cpu().numpy())
        return np.vstack(all_embeds)
    
    elif model_name == 'codonbert_hf':
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained('lhallee/CodonBERT', trust_remote_code=True)
        model = AutoModel.from_pretrained('lhallee/CodonBERT', trust_remote_code=True)
        model.eval()
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                # CodonBERT-HF uses RNA (U not T)
                rna_batch = [s.replace('T', 'U') for s in batch]
                inputs = tokenizer(rna_batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                embeds = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
                all_embeds.append(embeds.cpu().numpy())
        return np.vstack(all_embeds)
    
    elif model_name == 'encodon-80m':
        from transformers import AutoTokenizer, AutoModel
        import src.models.xformers_compat
        tokenizer = AutoTokenizer.from_pretrained('goodarzilab/encodon-80M', trust_remote_code=True)
        model = AutoModel.from_pretrained('goodarzilab/encodon-80M', trust_remote_code=True)
        model.eval()
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if 'token_type_ids' in inputs:
                    del inputs['token_type_ids']
                outputs = model(**inputs)
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                embeds = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
                all_embeds.append(embeds.cpu().numpy())
        return np.vstack(all_embeds)

def evaluate_regression(X_train, y_train, X_test, y_test):
    """Evaluate regression with Ridge."""
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    spearman = spearmanr(y_test, y_pred).statistic
    
    return {'r2': r2, 'spearman': spearman, 'n_train': len(y_train), 'n_test': len(y_test)}

if __name__ == '__main__':
    print("Loading Fungal expression data...")
    df = load_fungal_data()
    print(f"Total: {len(df)}, Train: {(df['Split']=='train').sum()}, Val: {(df['Split']=='val').sum()}, Test: {(df['Split']=='test').sum()}")
    
    # Use train+val for training, test for evaluation
    train_df = df[df['Split'].isin(['train', 'val'])]
    test_df = df[df['Split'] == 'test']
    
    results = {}
    
    for model_name in ['codonbert', 'codonbert_hf', 'encodon-80m']:
        print(f"\n{'='*60}")
        print(f"Processing {model_name}...")
        
        try:
            # Extract embeddings
            train_seqs = train_df['CDS'].tolist()
            test_seqs = test_df['CDS'].tolist()
            
            print(f"  Extracting train embeddings (n={len(train_seqs)})...")
            X_train = get_embeddings(model_name, train_seqs, batch_size=4)
            print(f"  Train embeddings shape: {X_train.shape}")
            
            print(f"  Extracting test embeddings (n={len(test_seqs)})...")
            X_test = get_embeddings(model_name, test_seqs, batch_size=4)
            print(f"  Test embeddings shape: {X_test.shape}")
            
            y_train = train_df['Value'].values
            y_test = test_df['Value'].values
            
            # Evaluate
            metrics = evaluate_regression(X_train, y_train, X_test, y_test)
            results[model_name] = metrics
            print(f"  R2: {metrics['r2']:.4f}, Spearman: {metrics['spearman']:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
    
    # Save results
    out_path = f'{RESULTS_DIR}/fungal_expression_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(json.dumps(results, indent=2))