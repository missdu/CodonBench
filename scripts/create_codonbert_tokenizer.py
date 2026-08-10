import sys
sys.path.insert(0, ".")
import os

# Create tokenizer files for CodonBERT (vocab_size=69, same as EnCodon)
# 64 codons + 5 special tokens = 69

CODONS = [
    'TTT','TTC','TTA','TTG','CTT','CTC','CTA','CTG',
    'ATT','ATC','ATA','ATG','GTT','GTC','GTA','GTG',
    'TCT','TCC','TCA','TCG','CCT','CCC','CCA','CCG',
    'ACT','ACC','ACA','ACG','GCT','GCC','GCA','GCG',
    'TAT','TAC','TAA','TAG','CAT','CAC','CAA','CAG',
    'AAT','AAC','AAA','AAG','GAT','GAC','GAA','GAG',
    'TGT','TGC','TGA','TGG','CGT','CGC','CGA','CGG',
    'AGT','AGC','AGA','AGG','GGT','GGC','GGA','GGG',
]

# Special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
# Then codons 5-68
# This matches EnCodon's tokenizer layout

output_dir = os.path.expanduser("./cLMs/CodonBERT/codonbert")

# Create vocab.txt
vocab_lines = [
    "[PAD]",    # 0
    "[UNK]",    # 1
    "[CLS]",    # 2
    "[SEP]",    # 3
    "[MASK]",   # 4
] + CODONS  # 5-68

vocab_path = os.path.join(output_dir, "vocab.txt")
with open(vocab_path, "w") as f:
    for token in vocab_lines:
        f.write(token + "\n")
print(f"Created vocab.txt with {len(vocab_lines)} tokens at {vocab_path}")

# Create tokenizer_config.json
import json
tokenizer_config = {
    "do_lower_case": False,
    "model_type": "bert",
    "vocab_size": 69,
    "max_len": 1024,
}
with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
    json.dump(tokenizer_config, f, indent=2)
print("Created tokenizer_config.json")

# Verify
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(output_dir)
print(f"Tokenizer loaded: vocab_size={tok.vocab_size}")
test = tok("ATG GCT AAA", return_tensors="pt")
print(f"Test tokenize 'ATG GCT AAA': input_ids={test['input_ids'].tolist()}")