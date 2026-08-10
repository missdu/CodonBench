#!/usr/bin/env python3
import pandas as pd
import numpy as np

data_dir = "./data/regression"
df = pd.read_csv(f"{data_dir}/Fungal_expression.csv")

print(f"Total samples: {len(df)}")
print(f"Unique sequences: {df['Sequence'].nunique()}")
print(f"Dataset values: {df['Dataset'].unique()}")
print(f"Split values: {df['Split'].unique()}")

# Check sequence lengths
df['seq_len'] = df['Sequence'].str.len()
print(f"\nSequence length stats:")
print(df['seq_len'].describe())

# Check if sequences look like codon-level (length divisible by 3)
df['len_mod3'] = df['seq_len'] % 3
print(f"\nLength % 3 distribution:")
print(df['len_mod3'].value_counts())

# Check for within-protein structure (groups of sequences with same AA but different codons)
# Take first 10 chars and check if there are multiple sequences with same AA prefix
print(f"\nFirst sequence (first 100 chars): {df['Sequence'].iloc[0][:100]}")
print(f"Second sequence (first 100 chars): {df['Sequence'].iloc[1][:100]}")

# Check if T or U is used
has_T = df['Sequence'].str.contains('T').any()
has_U = df['Sequence'].str.contains('U').any()
print(f"\nContains T: {has_T}, Contains U: {has_U}")

# Value distribution
print(f"\nValue stats:")
print(df['Value'].describe())

# Check Dataset column for grouping info
for ds in df['Dataset'].unique():
    sub = df[df['Dataset'] == ds]
    print(f"\nDataset={ds}: n={len(sub)}, seq_len_mean={sub['seq_len'].mean():.0f}, value_mean={sub['Value'].mean():.3f}")