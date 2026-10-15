"""
Standalone Evaluation Script for DFU-SIAM v3 and Checkpoints.
Computes comprehensive metrics:
- Accuracy, Macro-Precision, Macro-Recall, Macro-F1
- Per-class F1, Precision, Recall (especially Ischaemia & Both minority classes)
- Confusion Matrix
- Wagner Depth Metrics: Accuracy, MAE, Adjacent Accuracy, QWK (Quadratic Weighted Kappa)
- Zero-Shot Cross-Taxonomy Consistency
- Optional export to JSON
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report, cohen_kappa_score
)

from dfu_siam_model import DFUSiamV3, CORALLoss, get_device
from dataset import DFUSingleDataset, WagnerSingleDataset, get_dfu_transforms
from fusion_eval import evaluate_zero_shot_cross_taxonomy


def evaluate_pathology_full(model, dataloader, device, class_names=None):
    """Full DFUC pathology evaluation with per-class F1 and confusion matrix."""
    if class_names is None:
        class_names = ['none', 'infection', 'ischaemia', 'both']

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, lbls, _ in dataloader:
            imgs = imgs.to(device)
            out = model.forward_single(imgs)
            logits = out[1] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Core metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    # Per-class F1, precision, recall
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_prec = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_rec = recall_score(all_labels, all_preds, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)

    metrics = {
        'accuracy': float(acc),
        'macro_precision': float(macro_prec),
        'macro_recall': float(macro_rec),
        'macro_f1': float(macro_f1),
        'per_class_f1': {name: float(f) for name, f in zip(class_names, per_class_f1)},
        'per_class_precision': {name: float(p) for name, p in zip(class_names, per_class_prec)},
        'per_class_recall': {name: float(r) for name, r in zip(class_names, per_class_rec)},
        'confusion_matrix': cm.tolist(),
        'num_samples': len(all_labels),
        'class_distribution': {name: int((all_labels == i).sum()) for i, name in enumerate(class_names)},
        'prediction_distribution': {name: int((all_preds == i).sum()) for i, name in enumerate(class_names)},
    }

    return metrics, report


def evaluate_wagner_full(model, dataloader, device, ordinal_criterion, grade_names=None):
    """Full Wagner evaluation with MAE, QWK, and confusion matrix."""
    if grade_names is None:
        grade_names = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, lbls, _ in dataloader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            out = model.forward_wagner(imgs)
            ord_out = out[-1]

            _, logits, preds = ordinal_criterion(ord_out, lbls)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    mae = np.mean(np.abs(all_preds - all_labels))
    adj_acc = np.mean(np.abs(all_preds - all_labels) <= 1)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(grade_names))))

    report = classification_report(all_labels, all_preds, target_names=grade_names, zero_division=0)

    metrics = {
        'accuracy': float(acc),
        'mae': float(mae),
        'adjacent_accuracy': float(adj_acc),
        'macro_f1': float(macro_f1),
        'quadratic_weighted_kappa': float(qwk),
        'confusion_matrix': cm.tolist(),
        'num_samples': len(all_labels),
    }

    return metrics, report


def main():
    parser = argparse.ArgumentParser(description="Standalone DFU-SIAM v3 Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data_dir", type=str, default="DFUdataset", help="Path to DFUdataset directory")
    parser.add_argument("--val_csv", type=str, default=None, help="Path to validation split CSV")
    parser.add_argument("--wagner_dir", type=str, default="WagnerDataset", help="Path to WagnerDataset directory")
    parser.add_argument("--img_size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num workers")
    parser.add_argument("--output_json", type=str, default=None, help="Save metrics to JSON file")
    args = parser.parse_args()

    device = get_device()
    print(f"=== DFU-SIAM Standalone Evaluation on {device} ===")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_ver = checkpoint.get('model_version', 'legacy')
    epoch = checkpoint.get('epoch', 'N/A')
    saved_f1 = checkpoint.get('macro_f1', checkpoint.get('best_metric', 'N/A'))
    print(f"Checkpoint: version={ckpt_ver}, epoch={epoch}, saved_f1={saved_f1}")

    # Load v3 model (auto-detects full, swin_tiny, or lightweight from checkpoint config)
    cfg = checkpoint.get('config', {})
    arch = cfg.get('arch', 'swin_tiny' if 'swin' in str(cfg).lower() else 'full')
    model = DFUSiamV3(
        arch=arch,
        pretrained=False,
        shared_dim=cfg.get('shared_dim', None),
        path_dim=cfg.get('path_dim', None),
        depth_dim=cfg.get('depth_dim', None)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ordinal_criterion = CORALLoss(num_classes=5).to(device)

    # Find validation CSV
    val_csv = args.val_csv
    if val_csv is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        val_csv = os.path.join(ckpt_dir, "temp_val_split.csv")
        if not os.path.exists(val_csv):
            val_csv = os.path.join("checkpoints", "temp_val_split.csv")

    all_results = {}
    transform = get_dfu_transforms(is_train=False, img_size=(args.img_size, args.img_size))

    # Evaluate DFUC2021 pathology classification
    train_img_dir = os.path.join(args.data_dir, "DFUC2021_train", "images")
    if os.path.exists(val_csv) and os.path.exists(train_img_dir):
        print(f"\n--- DFUC2021 Pathology Classification ---")
        print(f"Val CSV: {val_csv}")
        val_dataset = DFUSingleDataset(
            img_dir=train_img_dir, csv_file=val_csv, transform=transform
        )
        print(f"Validation samples: {len(val_dataset)}")
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        dfu_metrics, dfu_report = evaluate_pathology_full(model, val_loader, device)
        all_results['pathology'] = dfu_metrics

        print(f"Accuracy:         {dfu_metrics['accuracy']:.4f}")
        print(f"Macro Precision:  {dfu_metrics['macro_precision']:.4f}")
        print(f"Macro Recall:     {dfu_metrics['macro_recall']:.4f}")
        print(f"Macro F1:         {dfu_metrics['macro_f1']:.4f}")
        print(f"\nPer-Class F1:")
        for name, f1 in dfu_metrics['per_class_f1'].items():
            print(f"  {name:12s}: {f1:.4f}")
        print(f"\nClassification Report:\n{dfu_report}")

    # Evaluate Wagner ordinal depth
    wagner_val_dir = os.path.join(args.wagner_dir, "valid")
    if os.path.exists(wagner_val_dir):
        print(f"\n--- Wagner Ordinal Classification ---")
        print(f"Wagner Val Dir: {wagner_val_dir}")
        wagner_val_ds = WagnerSingleDataset(wagner_val_dir, is_byol=False, transform=transform)
        print(f"Validation samples: {len(wagner_val_ds)}")
        wagner_val_loader = DataLoader(
            wagner_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        wagner_metrics, wagner_report = evaluate_wagner_full(
            model, wagner_val_loader, device, ordinal_criterion
        )
        all_results['wagner'] = wagner_metrics

        print(f"Accuracy:              {wagner_metrics['accuracy']:.4f}")
        print(f"MAE:                   {wagner_metrics['mae']:.4f}")
        print(f"Adjacent Acc (+-1):    {wagner_metrics['adjacent_accuracy']:.4f}")
        print(f"Macro F1:              {wagner_metrics['macro_f1']:.4f}")
        print(f"Quadratic Kappa (QWK): {wagner_metrics['quadratic_weighted_kappa']:.4f}")
        print(f"\nClassification Report:\n{wagner_report}")

        # Cross-taxonomy consistency
        if 'val_loader' in locals():
            print(f"\n--- Zero-Shot Cross-Taxonomy Consistency ---")
            cross_tax = evaluate_zero_shot_cross_taxonomy(
                model, val_loader, wagner_val_loader, device, ordinal_criterion
            )
            all_results['cross_taxonomy'] = cross_tax
            print(f"Spearman Rho:  {cross_tax['wagner_spearman_rho']:.4f} (p={cross_tax['wagner_spearman_p']:.4e})")
            print(f"Pearson R:     {cross_tax['wagner_pearson_r']:.4f}")
            print(f"Depth Delta:   {cross_tax['pathological_vs_clean_depth_delta']:.4f}")
            print(f"Mean Depth per DFUC Class:")
            for c_name, d_val in cross_tax['mean_depth_per_dfu_class'].items():
                print(f"  {c_name:12s}: {d_val:.4f}")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nMetrics exported to: {args.output_json}")


if __name__ == "__main__":
    main()
