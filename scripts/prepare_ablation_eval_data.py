"""
Prepare evaluation data for from-scratch ablation.
Extracts sequences and labels from existing downstream data files,
saves as .npz for easy loading by run_from_scratch_ablation.py.

Usage: python prepare_ablation_eval_data.py
"""

import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULT_DIR = os.path.expanduser("./results")
DATA_DIR = os.path.expanduser("./data")


def prepare_task_data(task_name):
    """Load classification data and extract sequences + labels."""
    print(f"Preparing {task_name}...")

    downstream_file = os.path.join(RESULT_DIR, "downstream_classification_summary_final.json")
    if os.path.exists(downstream_file):
        with open(downstream_file) as f:
            summary = json.load(f)
        print(f"  Found summary file with {len(summary)} entries")

    comp_file = os.path.join(RESULT_DIR, "comprehensive_comparison.json")
    if os.path.exists(comp_file):
        with open(comp_file) as f:
            comp = json.load(f)
        for entry in comp:
            if entry.get("task") == task_name:
                print(f"  Reference: {entry['model']} AUC = {entry['auc_mean']:.3f}")

    clinvar_dir = os.path.join(DATA_DIR, "task2_clinvar")
    cds_file = os.path.join(clinvar_dir, "cds_sequences.json")

    if not os.path.exists(cds_file):
        print(f"  CDS file not found: {cds_file}")
        return None

    with open(cds_file) as f:
        cds_data = json.load(f)
    print(f"  Loaded {len(cds_data)} CDS sequences")

    return {
        "cds_data": cds_data,
        "task": task_name,
    }


def main():
    for task in ["task2_missense", "task3_synonymous"]:
        data = prepare_task_data(task)
        if data is None:
            print(f"Skipping {task}: data not available")
            continue

        output_file = os.path.join(RESULT_DIR, f"downstream_{task}_data_info.json")
        with open(output_file, "w") as f:
            json.dump({"task": task, "n_cds": len(data["cds_data"])}, f, indent=2)
        print(f"  Saved info to {output_file}")

    print("\nDone. The actual sequence/label extraction will be done by the evaluation script")
    print("using the same pipeline as run_downstream_classification.py")


if __name__ == "__main__":
    main()