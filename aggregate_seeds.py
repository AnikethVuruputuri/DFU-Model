"""
Aggregate Multi-Seed Evaluation Results for DFU-SIAM v3.
Computes Mean and Standard Deviation across runs for:
- Macro-F1, Accuracy, UAR, Macro-AUC, MCC, and Per-Class F1 (Pathology)
- Accuracy, MAE, Adjacent Accuracy, Macro-F1, and QWK (Wagner)
- Inference Runtime and Peak Memory (if recorded)
"""

import os
import json
import argparse
import numpy as np


def aggregate_results(result_json_paths, output_json=None):
    loaded = []
    for p in result_json_paths:
        if not os.path.exists(p):
            print(f"Warning: Result file {p} not found, skipping.")
            continue
        with open(p, 'r') as f:
            loaded.append(json.load(f))

    if len(loaded) == 0:
        raise ValueError("No valid result JSON files provided for aggregation.")

    print(f"\n=======================================================")
    print(f" Aggregated Results Across {len(loaded)} Seeds")
    print(f"=======================================================\n")

    summary = {'num_seeds': len(loaded), 'seeds_data': loaded}

    # 1. Pathology Aggregation
    if 'pathology' in loaded[0]:
        print("--- Pathology Performance (DFUC2021) ---")
        summary['pathology'] = {}
        for metric_key in ['macro_f1', 'accuracy', 'macro_precision', 'macro_recall', 'uar', 'mcc', 'macro_auc']:
            vals = [run['pathology'].get(metric_key, 0.0) for run in loaded if 'pathology' in run and metric_key in run['pathology']]
            if vals:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                summary['pathology'][metric_key] = {'mean': mean_val, 'std': std_val, 'raw': vals}
                print(f"  {metric_key:18s}: {mean_val:.4f} +/- {std_val:.4f}")

        # Per-class F1
        if 'per_class_f1' in loaded[0]['pathology']:
            print("\n  Per-Class F1 Breakdown:")
            summary['pathology']['per_class_f1'] = {}
            for cls_name in ['none', 'infection', 'ischaemia', 'both']:
                vals = [run['pathology']['per_class_f1'].get(cls_name, 0.0) for run in loaded if 'pathology' in run and 'per_class_f1' in run['pathology']]
                if vals:
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals))
                    summary['pathology']['per_class_f1'][cls_name] = {'mean': mean_val, 'std': std_val}
                    print(f"    {cls_name:14s}: {mean_val:.4f} +/- {std_val:.4f}")

    # 2. Wagner Depth Aggregation
    if 'wagner' in loaded[0]:
        print("\n--- Wagner Depth Classification ---")
        summary['wagner'] = {}
        for metric_key in ['accuracy', 'mae', 'adjacent_accuracy', 'macro_f1', 'quadratic_weighted_kappa']:
            vals = [run['wagner'].get(metric_key, 0.0) for run in loaded if 'wagner' in run and metric_key in run['wagner']]
            if vals:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                summary['wagner'][metric_key] = {'mean': mean_val, 'std': std_val, 'raw': vals}
                print(f"  {metric_key:24s}: {mean_val:.4f} +/- {std_val:.4f}")

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nAggregated results written to: {output_json}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate evaluation results across seeds")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to evaluation JSON files from each seed")
    parser.add_argument("--output", type=str, default="aggregated_results.json", help="Output summary JSON path")
    args = parser.parse_args()

    aggregate_results(args.inputs, args.output)
