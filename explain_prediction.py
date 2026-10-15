"""
Single-Image Explainability (XAI) CLI for DFU-SIAM v3.

Produces production-quality and publication-ready explanations for DFU-SIAM v3 predictions.
Supports:
- Grad-CAM (CNN, ViT, Fused)
- Transformer Window-Attention Attribution
- Integrated Gradients with configurable baselines
- Superpixel-partitioned KernelSHAP
- Dual targets: Pathology classes & Wagner ordinal depth grades

Usage:
    python explain_prediction.py --checkpoint checkpoints_a1_full/best_checkpoint.pth --image path/to/img.jpg --target pathology --xai_method gradcam
    python explain_prediction.py --checkpoint checkpoints_a1_full_balanced/best_checkpoint.pth --image path/to/img.jpg --target wagner --xai_method all
"""

import os
import sys
import argparse
import time
from pathlib import Path
from PIL import Image
import torch

from dfu_siam_model import DFUSiamV3, get_device
from dataset import get_dfu_transforms
from xai.unified import XAIExplainer
from xai.base import PATHOLOGY_CLASSES, WAGNER_GRADES


def parse_args():
    parser = argparse.ArgumentParser(description="DFU-SIAM v3 Explainability Tool")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained DFU-SIAM v3 model checkpoint (.pth)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input ulcer image")
    parser.add_argument("--target", type=str, default="pathology", choices=["pathology", "wagner"],
                        help="Prediction taxonomy to explain ('pathology' or 'wagner')")
    parser.add_argument("--target_id", type=str, default="predicted",
                        help="Target class/grade to explain: 'predicted', class name (e.g. 'ischaemia'), or int")
    parser.add_argument("--xai_method", type=str, default="all",
                        choices=["gradcam", "transformer", "integrated_gradients", "shap", "all"],
                        help="Explainability method to execute")
    parser.add_argument("--output_dir", type=str, default="outputs/xai",
                        help="Base directory for XAI visual outputs and metadata")
    parser.add_argument("--arch", type=str, default=None, choices=["full", "swin_tiny"],
                        help="Model architecture (auto-detected from checkpoint if omitted)")
    parser.add_argument("--baseline", type=str, default="black", choices=["black", "white", "blur", "mean"],
                        help="Baseline image type for Integrated Gradients")
    parser.add_argument("--ig_steps", type=int, default=25,
                        help="Path interpolation steps for Integrated Gradients")
    parser.add_argument("--shap_grid", type=int, default=14,
                        help="Grid superpixel partition size for SHAP (e.g. 14 -> 14x14=196 segments)")
    parser.add_argument("--shap_evals", type=int, default=100,
                        help="Number of background evaluations for KernelSHAP")
    parser.add_argument("--run_faithfulness", action="store_true",
                        help="Run quantitative AOPC deletion and insertion faithfulness tests")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('cuda' or 'cpu')")
    return parser.parse_args()


def load_model(checkpoint_path, arch=None, device=None):
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)

    print(f"[*] Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = checkpoint.get('config', {})

    if arch is None:
        arch = cfg.get('arch', 'full')

    print(f"[*] Instantiating DFUSiamV3 with arch='{arch}' (device: {device})...")
    model = DFUSiamV3(
        arch=arch,
        pretrained=False,
        shared_dim=cfg.get('shared_dim', None),
        path_dim=cfg.get('path_dim', None),
        depth_dim=cfg.get('depth_dim', None)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, cfg, device


def main():
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    model, cfg, device = load_model(args.checkpoint, arch=args.arch, device=args.device)

    # Initialize unified XAI explainer
    explainer = XAIExplainer(model, device=device)
    explainer.ig.steps = args.ig_steps
    explainer.ig.baseline_type = args.baseline
    explainer.shap.grid_size = args.shap_grid
    explainer.shap.n_evals = args.shap_evals

    # Build image-specific output directory
    image_stem = img_path.stem
    img_out_dir = os.path.join(args.output_dir, image_stem)
    os.makedirs(img_out_dir, exist_ok=True)

    # Load image and preprocessing transforms
    pil_img = Image.open(str(img_path)).convert("RGB")
    transform = get_dfu_transforms(is_train=False, img_size=(224, 224))

    print(f"[*] Generating '{args.xai_method}' explanations for target='{args.target}' (target_id='{args.target_id}')...")
    start_time = time.time()

    extra_meta = {
        'image_path': str(img_path.resolve()),
        'checkpoint_path': str(Path(args.checkpoint).resolve()),
        'epoch_checkpoint': cfg.get('epoch', 'N/A')
    }

    results, meta = explainer.explain_image(
        pil_img=pil_img,
        transform=transform,
        target_type=args.target,
        target_id=args.target_id,
        method=args.xai_method,
        output_dir=img_out_dir,
        run_faithfulness=args.run_faithfulness,
        extra_meta=extra_meta
    )

    elapsed = time.time() - start_time
    print(f"[+] Explanations completed in {elapsed:.2f}s!")
    print(f"    - Output directory: {img_out_dir}")
    print(f"    - Predicted Pathology: {meta['pred_pathology_name'].upper()} (Prob: {max(meta['pathology_probs']):.3f})")
    print(f"    - Predicted Wagner:    {meta['pred_wagner_name']} (Prob: {max(meta['wagner_probs']):.3f})")
    print(f"    - Target Explained:    {meta['target_type'].upper()} -> '{meta['target_name']}' (Score: {meta['target_score']:.3f})")

    if 'faithfulness_evaluation' in meta:
        faith = meta['faithfulness_evaluation']
        print(f"    - AOPC Deletion Score: {faith['aopc_deletion_score']:.4f} (Higher indicates higher fidelity)")
        print(f"    - Insertion AUC:       {faith['insertion_auc']:.4f} (Higher indicates faster feature recovery)")

    print(f"    - Generated Visualizations:")
    for f in os.listdir(img_out_dir):
        print(f"      * {f}")


if __name__ == "__main__":
    main()
