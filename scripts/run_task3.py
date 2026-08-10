import sys
sys.path.insert(0, ".")
from src.data.download import prepare_task3
from pathlib import Path

df = prepare_task3(Path("./data"))
print(f"Result: {len(df)} samples, cols={list(df.columns)}")
if "label" in df.columns:
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")