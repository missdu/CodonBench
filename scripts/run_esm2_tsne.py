import numpy as np
import json
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

base = Path("./results/supplementary")

print("Loading ESM-2 embeddings...")
esm2_syn_emb = np.load(base / "ESM-2-650M_task3_synonymous_emb.npy")
esm2_syn_labels = np.load(base / "ESM-2-650M_task3_synonymous_labels.npy")
esm2_mis_emb = np.load(base / "ESM-2-650M_task2_missense_emb.npy")
esm2_mis_labels = np.load(base / "ESM-2-650M_task2_missense_labels.npy")

print(f"ESM-2 SynPath: {esm2_syn_emb.shape}, MisPath: {esm2_mis_emb.shape}")

np.random.seed(42)
if len(esm2_syn_emb) > 1000:
    idx = np.random.choice(len(esm2_syn_emb), 1000, replace=False)
    esm2_syn_emb = esm2_syn_emb[idx]
    esm2_syn_labels = esm2_syn_labels[idx]
if len(esm2_mis_emb) > 1000:
    idx = np.random.choice(len(esm2_mis_emb), 1000, replace=False)
    esm2_mis_emb = esm2_mis_emb[idx]
    esm2_mis_labels = esm2_mis_labels[idx]

print("PCA to 50 dims...")
esm2_syn_pca = PCA(n_components=50, random_state=42).fit_transform(esm2_syn_emb)
esm2_mis_pca = PCA(n_components=50, random_state=42).fit_transform(esm2_mis_emb)
del esm2_syn_emb, esm2_mis_emb

print("t-SNE on ESM-2 SynPath...")
esm2_syn_tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500).fit_transform(esm2_syn_pca)
del esm2_syn_pca
print("t-SNE on ESM-2 MisPath...")
esm2_mis_tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500).fit_transform(esm2_mis_pca)
del esm2_mis_pca

np.save(base / "esm2_task3_synonymous_tsne.npy", esm2_syn_tsne.astype(np.float32))
np.save(base / "esm2_task3_synonymous_tsne_labels.npy", esm2_syn_labels)
np.save(base / "esm2_task2_missense_tsne.npy", esm2_mis_tsne.astype(np.float32))
np.save(base / "esm2_task2_missense_tsne_labels.npy", esm2_mis_labels)
print("Done! Saved ESM-2 t-SNE coordinates")
