from huggingface_hub import list_models
models = list(list_models(search="codon", limit=30))
for m in models:
    print(m.id, m.downloads or 0)