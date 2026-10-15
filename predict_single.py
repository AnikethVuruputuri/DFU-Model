"""
Single-Image Clinical Inference & Diagnostics for DFU-SIAM v3 Architecture.
Runs:
1. 4-Class DFUC Pathology Softmax Classification
2. 5-Grade Wagner Depth Grade via CORAL Sigmoid Cumulative Logits (Grades 0-4)
3. Exploratory 16-Cell University of Texas (UT) Wound Matrix Synthesis
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dfu_siam_model import DFUSiamV3, get_device
from dataset import get_tta_transforms
from fusion_eval import predict_ut_grid


def predict_wagner_5_probs_coral(wagner_out):
    """
    Decodes 4 cumulative cutoffs from CORAL into a valid 5-grade probability distribution.
    q_k = sigmoid(f_k) for k in {0, 1, 2, 3}.
    P(0) = 1 - q_0
    P(k) = clamp(q_{k-1} - q_k, 0)
    P(4) = q_3
    """
    wagner_arr = np.asarray(wagner_out).squeeze()
    cum_probs = 1.0 / (1.0 + np.exp(-wagner_arr[:4]))
    p0 = max(1.0 - cum_probs[0], 0.0)
    p1 = max(cum_probs[0] - cum_probs[1], 0.0)
    p2 = max(cum_probs[1] - cum_probs[2], 0.0)
    p3 = max(cum_probs[2] - cum_probs[3], 0.0)
    p4 = max(cum_probs[3], 0.0)

    probs = np.array([p0, p1, p2, p3, p4])
    total = np.sum(probs)
    return probs / (total + 1e-8)


def predict_single_image(image_path, checkpoint_path, img_size=224, save_visual=True, output_dir="output"):
    """
    Runs multi-task cross-taxonomy inference on a single Diabetic Foot Ulcer image using DFU-SIAM v3.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    device = get_device()
    print(f"\n=======================================================")
    print(f" DFU-SIAM v3 SINGLE IMAGE INFERENCE & DIAGNOSIS")
    print(f"=======================================================")
    print(f" Input Image  : {image_path}")
    print(f" Checkpoint   : {checkpoint_path}")
    print(f" Computing On : {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_ver = checkpoint.get('model_version', 'legacy')
    if ckpt_ver != 'v3':
        raise ValueError(
            f"Incompatible checkpoint: expected model_version='v3', but found {ckpt_ver!r}. "
            "DFU-SIAM v3 requires a v3 checkpoint."
        )

    config = checkpoint.get('config', {})
    arch = config.get('arch', 'full')
    model = DFUSiamV3(
        arch=arch,
        pretrained=False,
        shared_dim=config.get('shared_dim', None),
        path_dim=config.get('path_dim', None),
        depth_dim=config.get('depth_dim', None)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    pil_img = Image.open(image_path).convert('RGB')
    tta_transforms = get_tta_transforms(img_size=(img_size, img_size))
    tta_tensors = torch.stack([transform(pil_img) for transform in tta_transforms]).to(device)

    with torch.no_grad():
        _, logits, ord_outs = model.forward_single(tta_tensors, return_ordinal=True)
        probs = F.softmax(logits, dim=1)
        avg_prob = torch.mean(probs, dim=0).cpu().numpy()
        avg_ord = torch.mean(ord_outs, dim=0).cpu().numpy()

    class_names = ['none', 'infection', 'ischaemia', 'both']
    stage_letters = ['A', 'B', 'C', 'D']
    pred_cls_idx = int(np.argmax(avg_prob))
    pred_class_name = class_names[pred_cls_idx]
    pred_confidence = float(avg_prob[pred_cls_idx]) * 100

    wagner_5_probs = predict_wagner_5_probs_coral(avg_ord)
    pred_wagner_grade = int(np.argmax(wagner_5_probs))
    pred_wagner_conf = float(wagner_5_probs[pred_wagner_grade]) * 100

    ut_grid, ut_depth_probs = predict_ut_grid(avg_prob, avg_ord)
    pred_ut_depth_idx = int(np.argmax(ut_depth_probs))
    pred_stage_idx = pred_cls_idx
    ut_grade_str = f"Stage {stage_letters[pred_stage_idx]} (Depth {pred_ut_depth_idx})"
    top_ut_prob = float(ut_grid[pred_ut_depth_idx, pred_stage_idx]) * 100

    print("\n-------------------------------------------------------")
    print(" CLINICAL DIAGNOSTIC RESULTS SUMMARY")
    print("-------------------------------------------------------")
    print(f" ==> FINAL PREDICTED PATHOLOGY CLASS : {pred_class_name.upper()} ({pred_confidence:.2f}% confidence)")
    print(f" ==> FINAL PREDICTED WAGNER GRADE    : GRADE {pred_wagner_grade} ({pred_wagner_conf:.2f}% confidence)")
    print(f" ==> FINAL PREDICTED UT WOUND STAGE  : {ut_grade_str} ({top_ut_prob:.2f}% joint probability)")
    print("-------------------------------------------------------")

    print(f"\n Detailed DFUC Pathology Probabilities (Winner: {pred_class_name.upper()}):")
    for cls_name, prob_val in zip(class_names, avg_prob):
        bar = "#" * int(prob_val * 25)
        tag = " <== [FINAL PREDICTION]" if cls_name == pred_class_name else ""
        print(f"    * {cls_name.capitalize():<10} : {prob_val * 100:6.2f}% | {bar:<25}{tag}")

    print(f"\n Detailed Wagner Grade Probabilities (Winner: GRADE {pred_wagner_grade}):")
    wagner_descriptions = [
        "Grade 0: Intact skin / Pre-ulcerative lesion",
        "Grade 1: Superficial ulcer (dermis/subcutaneous)",
        "Grade 2: Deep ulcer to tendon/capsule/bone",
        "Grade 3: Deep ulcer with abscess/osteomyelitis",
        "Grade 4: Gangrene of forefoot/partial foot"
    ]
    for g_idx, g_prob in enumerate(wagner_5_probs):
        bar = "#" * int(g_prob * 25)
        desc = wagner_descriptions[g_idx]
        tag = " <== [FINAL PREDICTION]" if g_idx == pred_wagner_grade else ""
        print(f"    * Grade {g_idx} : {g_prob * 100:6.2f}% | {bar:<25} ({desc}){tag}")

    print(f"\n Detailed Exploratory UT Grid Depth Axis Probabilities (Winner: UT DEPTH {pred_ut_depth_idx}):")
    for d_idx, d_prob in enumerate(ut_depth_probs):
        bar = "#" * int(d_prob * 25)
        tag = " <== [FINAL PREDICTION]" if d_idx == pred_ut_depth_idx else ""
        print(f"    * UT Depth {d_idx} : {d_prob * 100:6.2f}% | {bar:<25}{tag}")
    print("=======================================================\n")

    return {
        'pathology_class': pred_class_name,
        'pathology_prob': pred_confidence,
        'wagner_grade': pred_wagner_grade,
        'wagner_prob': pred_wagner_conf,
        'ut_stage': ut_grade_str,
        'ut_prob': top_ut_prob
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Image Diagnosis with DFU-SIAM v3")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input ulcer image")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints_v3/dfu_siam_v3_best.pth", help="Path to v3 checkpoint")
    parser.add_argument("--img_size", type=int, default=224, help="Image resolution")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save diagnosis outputs")

    args = parser.parse_args()
    predict_single_image(args.image_path, args.checkpoint_path, img_size=args.img_size, output_dir=args.output_dir)
