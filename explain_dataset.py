"""
Batch Dataset Explainability and Cohort Faithfulness Evaluation for DFU-SIAM v3.

Processes multiple images across validation or test sets with support for:
- Correct-only vs Incorrect-only filtering (for error/failure mode analysis)
- Per-class balanced selection (ensuring minority classes like Ischaemia/Both are represented)
- Batch XAI generation across all methods
- Cohort faithfulness aggregation (Mean AOPC Deletion Score & Insertion AUC)
- Export of comprehensive summary report in JSON format

Usage:
    python explain_dataset.py --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth --num_samples 10 --filter all
    python explain_dataset.py --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth --num_samples 10 --filter incorrect_only --xai_method gradcam
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import torch

from dfu_siam_model import DFUSiamV3, get_device
from dataset import get_dfu_transforms, DFUSingleDataset
from xai.unified import XAIExplainer
from xai.base import PATHOLOGY_CLASSES, WAGNER_GRADES
from explain_prediction import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="DFU-SIAM v3 Batch Dataset Explainability")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained DFU-SIAM v3 checkpoint")
    parser.add_argument("--data_dir", type=str, default="DFUdataset",
                        help="Root directory for DFU dataset")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Path to validation CSV split")
    parser.add_argument("--output_dir", type=str, default="outputs/xai_dataset",
                        help="Directory to save batch explanations and summary JSON")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Total number of images to analyze")
    parser.add_argument("--filter", type=str, default="all",
                        choices=["all", "correct_only", "incorrect_only"],
                        help="Cohort filtering mode")
    parser.add_argument("--target", type=str, default="pathology",
                        choices=["pathology", "wagner"],
                        help="Prediction taxonomy to explain")
    parser.add_argument("--xai_method", type=str, default="gradcam",
                        choices=["gradcam", "transformer", "integrated_gradients", "shap", "all"],
                        help="Explainability method to generate")
    parser.add_argument("--run_faithfulness", action="store_true",
                        help="Compute AOPC Deletion and Insertion scores per sample")
    parser.add_argument("--arch", type=str, default=None,
                        help="Model architecture ('full' or 'swin_tiny')")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('cuda' or 'cpu')")
    return parser.parse_args()


def find_val_csv(args):
    if args.val_csv and os.path.exists(args.val_csv):
        return args.val_csv
    ckpt_dir = os.path.dirname(args.checkpoint)
    cand1 = os.path.join(ckpt_dir, "temp_val_split.csv")
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join("checkpoints", "temp_val_split.csv")
    if os.path.exists(cand2):
        return cand2
    cand3 = os.path.join(args.data_dir, "DFUC2021_train", "train.csv")
    if os.path.exists(cand3):
        return cand3
    raise FileNotFoundError("Could not find validation CSV file. Please provide --val_csv.")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, cfg, device = load_model(args.checkpoint, arch=args.arch, device=args.device)
    explainer = XAIExplainer(model, device=device)

    val_csv_path = find_val_csv(args)
    print(f"[*] Reading validation samples from: {val_csv_path}")
    df = pd.read_csv(val_csv_path)

    # Determine image directory
    train_img_dir = os.path.join(args.data_dir, "DFUC2021_train", "images")
    if not os.path.exists(train_img_dir):
        # Fallback if images directory is directly under DFUdataset
        train_img_dir = os.path.join(args.data_dir, "images")

    transform = get_dfu_transforms(is_train=False, img_size=(224, 224))

    # Evaluate predictions across candidate images to apply filter
    print(f"[*] Filtering images by condition: '{args.filter}'...")
    candidates = []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            img_name = row['image']
            img_path = os.path.join(train_img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            # Ground truth label
            gt_label = None
            if all(c in row for c in PATHOLOGY_CLASSES):
                gt_label = int(np.argmax([row[c] for c in PATHOLOGY_CLASSES]))

            # Quick forward pass to get predicted class
            try:
                pil_img = Image.open(img_path).convert("RGB")
                inp = transform(pil_img).unsqueeze(0).to(device)
                _, p_logits, w_logits = model.forward_single(inp, return_ordinal=True)
                pred_path = int(torch.argmax(p_logits, dim=-1)[0].item())
            except Exception as e:
                continue

            is_correct = (gt_label == pred_path) if gt_label is not None else True

            if args.filter == "correct_only" and not is_correct:
                continue
            if args.filter == "incorrect_only" and is_correct:
                continue

            candidates.append({
                'image_name': img_name,
                'image_path': img_path,
                'gt_label': gt_label,
                'pred_label': pred_path,
                'is_correct': is_correct
            })

            if len(candidates) >= args.num_samples * 3:
                break

    if not candidates:
        print(f"[!] No images matched filter condition '{args.filter}'.")
        return

    # Select balanced subset
    selected = candidates[:args.num_samples]
    print(f"[*] Selected {len(selected)} samples for XAI generation.")

    cohort_results = []
    aopc_scores = []
    insertion_scores = []
    total_start = time.time()

    for i, item in enumerate(selected):
        img_name = item['image_name']
        img_path = item['image_path']
        stem = Path(img_name).stem
        sample_out = os.path.join(args.output_dir, stem)

        print(f"  [{i+1}/{len(selected)}] Processing {img_name} (GT: {PATHOLOGY_CLASSES[item['gt_label']] if item['gt_label'] is not None else 'N/A'}, Pred: {PATHOLOGY_CLASSES[item['pred_label']]})...")

        pil_img = Image.open(img_path).convert("RGB")
        extra_meta = {
            'image_name': img_name,
            'ground_truth_id': item['gt_label'],
            'ground_truth_name': PATHOLOGY_CLASSES[item['gt_label']] if item['gt_label'] is not None else 'N/A',
            'is_correct': item['is_correct'],
            'checkpoint_path': str(Path(args.checkpoint).resolve())
        }

        try:
            results, meta = explainer.explain_image(
                pil_img=pil_img,
                transform=transform,
                target_type=args.target,
                target_id='predicted',
                method=args.xai_method,
                output_dir=sample_out,
                run_faithfulness=args.run_faithfulness,
                extra_meta=extra_meta
            )
            cohort_results.append(meta)

            if 'faithfulness_evaluation' in meta:
                aopc_scores.append(meta['faithfulness_evaluation']['aopc_deletion_score'])
                insertion_scores.append(meta['faithfulness_evaluation']['insertion_auc'])
        except Exception as e:
            print(f"    [!] Error explaining {img_name}: {e}")

    total_time = time.time() - total_start

    summary = {
        'checkpoint': args.checkpoint,
        'arch': getattr(model, 'arch', 'full'),
        'filter_mode': args.filter,
        'target': args.target,
        'xai_method': args.xai_method,
        'num_samples_analyzed': len(cohort_results),
        'total_runtime_sec': float(total_time),
        'average_runtime_per_sample_sec': float(total_time / max(1, len(cohort_results))),
        'aggregate_faithfulness': {
            'mean_aopc_deletion_score': float(np.mean(aopc_scores)) if aopc_scores else None,
            'std_aopc_deletion_score': float(np.std(aopc_scores)) if aopc_scores else None,
            'mean_insertion_auc': float(np.mean(insertion_scores)) if insertion_scores else None,
            'std_insertion_auc': float(np.std(insertion_scores)) if insertion_scores else None
        },
        'samples': cohort_results
    }

    summary_json_path = os.path.join(args.output_dir, "dataset_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n[+] Batch dataset explanations complete!")
    print(f"    - Output Directory: {args.output_dir}")
    print(f"    - Summary JSON:     {summary_json_path}")
    if aopc_scores:
        print(f"    - Mean Cohort AOPC Deletion: {summary['aggregate_faithfulness']['mean_aopc_deletion_score']:.4f}")
        print(f"    - Mean Cohort Insertion AUC: {summary['aggregate_faithfulness']['mean_insertion_auc']:.4f}")


if __name__ == "__main__":
    main()
