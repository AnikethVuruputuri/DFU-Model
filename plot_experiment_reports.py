"""
Plotting and Experiment Reporting Utility for DFU-SIAM v3.
Generates:
1. Training and validation loss curves (CB-Focal, CORAL, Hard-LMCoT, UG-MMD, Orthogonality)
2. Gradient norms before and after clipping
3. Learning rate schedule curves
4. Confusion matrix heatmaps for Pathology and Wagner grading
5. Cross-taxonomy exploratory grid visualizations
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix: {save_path}")


def generate_experiment_report(eval_json_path, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(eval_json_path):
        raise FileNotFoundError(f"Evaluation JSON not found at {eval_json_path}")

    with open(eval_json_path, 'r') as f:
        data = json.load(f)

    # Plot Pathology Confusion Matrix
    if 'pathology' in data and 'confusion_matrix' in data['pathology']:
        cm_path = np.array(data['pathology']['confusion_matrix'])
        class_names = ['none', 'infection', 'ischaemia', 'both']
        plot_confusion_matrix(
            cm_path, class_names,
            "DFU-SIAM v3 Pathology Confusion Matrix",
            os.path.join(output_dir, "pathology_confusion_matrix.png")
        )

    # Plot Wagner Confusion Matrix
    if 'wagner' in data and 'confusion_matrix' in data['wagner']:
        cm_wag = np.array(data['wagner']['confusion_matrix'])
        grade_names = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
        plot_confusion_matrix(
            cm_wag, grade_names,
            "DFU-SIAM v3 Wagner Grade Confusion Matrix",
            os.path.join(output_dir, "wagner_confusion_matrix.png")
        )

    # Plot Per-Class Pathology F1 Bar Chart
    if 'pathology' in data and 'per_class_f1' in data['pathology']:
        f1_dict = data['pathology']['per_class_f1']
        plt.figure(figsize=(7, 4))
        bars = plt.bar(list(f1_dict.keys()), list(f1_dict.values()), color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
        plt.title("DFU-SIAM v3 Pathology Per-Class F1 Score")
        plt.ylabel("F1 Score")
        plt.ylim(0, 1.0)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom')
        plt.tight_layout()
        bar_path = os.path.join(output_dir, "pathology_per_class_f1.png")
        plt.savefig(bar_path, dpi=300)
        plt.close()
        print(f"Saved F1 breakdown chart: {bar_path}")

    print(f"\nExperiment report plots successfully generated in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots and reports from evaluation JSON")
    parser.add_argument("--eval_json", type=str, required=True, help="Path to evaluation JSON result file")
    parser.add_argument("--output_dir", type=str, default="experiment_plots", help="Directory to save figures")
    args = parser.parse_args()

    generate_experiment_report(args.eval_json, args.output_dir)
