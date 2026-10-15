"""
Inference Pipeline for DFU-SIAM v3 Architecture with Test-Time Augmentation (TTA).
Extracts:
1. 4-Class Pathology Softmax Probabilities [None, Infection, Ischaemia, Both]
2. 5-Grade Wagner Depth via CORAL Cumulative Sigmoid Cutoffs (Grades 0-4)
3. Exploratory 16-Cell University of Texas (UT) Probabilistic Wound Matrix (Stage A-D x Depth 0-3)
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from dfu_siam_model import DFUSiamV3, get_device
from dataset import get_tta_transforms
from fusion_eval import predict_ut_grid


def extract_tta_test_predictions(model, test_img_dir, tta_transforms, device, img_size=(224, 224), max_samples=None):
    """
    Extracts averaged Softmax class probabilities and ordinal depth predictions for test images
    using Test-Time Augmentation (TTA) through direct task heads.
    """
    model.eval()
    test_img_names = sorted([f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if max_samples is not None:
        test_img_names = test_img_names[:max_samples]

    print(f"Extracting TTA predictions for {len(test_img_names)} test images...")

    all_test_probs = []
    all_ord_outputs = []

    with torch.no_grad():
        for img_name in test_img_names:
            img_path = os.path.join(test_img_dir, img_name)
            pil_img = Image.open(img_path).convert('RGB')

            # Apply all TTA transform variations
            tta_tensors = torch.stack([transform(pil_img) for transform in tta_transforms]).to(device)

            # Direct forward pass through DFUSiamV3 task heads
            _, logits, ord_outs = model.forward_single(tta_tensors, return_ordinal=True)

            # Softmax class probabilities
            probs = F.softmax(logits, dim=1)
            avg_prob = torch.mean(probs, dim=0).cpu().numpy()
            avg_ord = torch.mean(ord_outs, dim=0).cpu().numpy()

            all_test_probs.append(avg_prob)
            all_ord_outputs.append(avg_ord)

    all_test_probs = np.vstack(all_test_probs)
    all_ord_outputs = np.vstack(all_ord_outputs)
    return all_test_probs, all_ord_outputs, test_img_names


def run_inference(args):
    device = get_device()
    print(f"=== Starting DFU-SIAM v3 Direct Inference on {device} ===")

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

    # Load Model Checkpoint with strict version check
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    ckpt_ver = checkpoint.get('model_version', 'legacy')
    if ckpt_ver != 'v3':
        raise ValueError(
            f"Incompatible checkpoint: expected model_version='v3', but found {ckpt_ver!r}. "
            "DFU-SIAM v3 uses task-disentangled representations and requires a v3 checkpoint."
        )

    print(f"Loaded DFU-SIAM v3 checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

    cfg = checkpoint.get('config', {})
    arch = cfg.get('arch', 'full')
    model = DFUSiamV3(
        arch=arch,
        pretrained=False,
        shared_dim=cfg.get('shared_dim', None),
        path_dim=cfg.get('path_dim', None),
        depth_dim=cfg.get('depth_dim', None)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extract Test Set Predictions with TTA
    test_img_dir = os.path.join(args.data_dir, "DFUC2021_test")
    if not os.path.exists(test_img_dir):
        if os.path.exists(os.path.join(test_img_dir, "images")):
            test_img_dir = os.path.join(test_img_dir, "images")

    tta_transforms = get_tta_transforms(img_size=(args.img_size, args.img_size))
    probs, test_ord_outputs, test_names = extract_tta_test_predictions(
        model, test_img_dir, tta_transforms, device, img_size=(args.img_size, args.img_size), max_samples=args.max_test_samples
    )

    preds = np.argmax(probs, axis=1)
    class_names = ['none', 'infection', 'ischaemia', 'both']
    stage_letters = ['A', 'B', 'C', 'D']
    results = []

    for i, img_name in enumerate(test_names):
        row = {'image': img_name}
        for cls_idx, cls_name in enumerate(class_names):
            row[cls_name] = float(probs[i, cls_idx])
        row['predicted_class'] = class_names[preds[i]]

        # Exploratory UT 16-cell Wound Grid mapping
        ut_grid, depth_probs = predict_ut_grid(probs[i], test_ord_outputs[i])
        pred_depth_idx = int(np.argmax(depth_probs))
        pred_stage_idx = int(preds[i])

        row['predicted_wagner_depth'] = pred_depth_idx
        row['predicted_ut_grade'] = f"Stage {stage_letters[pred_stage_idx]} (Depth {pred_depth_idx})"
        row['ut_top_prob'] = float(ut_grid[pred_depth_idx, pred_stage_idx])
        results.append(row)

    out_df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, "dfu_siam_v3_predictions.csv")
    out_df.to_csv(out_path, index=False)

    print(f"\n=== Inference Complete! ===")
    print(f"Predictions saved to {out_path}")
    print("\nSample Predictions:")
    print(out_df[['image', 'predicted_class', 'predicted_wagner_depth', 'predicted_ut_grade', 'ut_top_prob']].head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Direct Inference for DFU-SIAM v3 Model")
    parser.add_argument("--data_dir", type=str, default="DFUdataset", help="Path to DFUdataset directory")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints_v3/dfu_siam_v3_best.pth", help="Path to trained v3 checkpoint")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save predictions CSV")
    parser.add_argument("--img_size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max test samples to process")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num workers")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_inference(args)
