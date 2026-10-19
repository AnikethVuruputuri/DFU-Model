"""
Training Pipeline for DFU-SIAM v3 Architecture.
Integrates:
- DFUSiamV3 Model with TaskDisentangler (z_shared, z_path_private, z_depth_private)
- Class-Balanced Focal Loss for Long-Tailed Pathology Classification
- CORAL Cumulative Ordinal Loss for Wagner Depth Grading (Grades 0-4)
- Hard-Pair Tangent LMCoT Metric Learning
- Uncertainty-Gated MMD (UG-MMD) on shared representations with detached confidence weights
- Orthogonality / Disentanglement Regularization (z_shared ⟂ z_private)
- Strict Checkpointing (v3 namespace, legacy checkpoint rejection)
- Reproducibility & AMP GradScaler Stability (init_scale=1024.0)
"""

import os
import sys

# Prevent CUDA memory fragmentation on GPUs with <= 16GB VRAM (e.g. NVIDIA T4, RTX 3050)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import signal
import random
import argparse
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

from dfu_siam_model import (
    DFUSiamV3, ClassBalancedFocalLoss, CORALLoss, HardPairLMCoTLoss,
    UncertaintyGatedMMDLoss, OrthogonalityLoss, get_device
)
from dataset import (
    DFUSiamesePairDataset, DFUSingleDataset, WagnerSingleDataset,
    InterleavedDataLoader, get_dfu_transforms, get_dfu_transforms_v2
)
from fusion_eval import evaluate_wagner_ordinal, evaluate_zero_shot_cross_taxonomy
from leakage_guard import assert_pre_training_leakage_free, compute_file_md5


def save_atomic_checkpoint(state_dict: dict, target_path: str):
    """
    Saves a checkpoint atomically using a unique temporary file and os.replace.
    Ensures existing valid checkpoints are never corrupted if training is interrupted mid-write.
    """
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(prefix="ckpt_", suffix=".tmp", dir=target_dir)
    os.close(temp_fd)
    try:
        torch.save(state_dict, temp_path)
        os.replace(temp_path, target_path)
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise e


def get_rng_states():
    """Captures all random number generator states across Python, NumPy, PyTorch CPU, and CUDA."""
    states = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    return states


def set_rng_states(states):
    """Restores random number generator states across Python, NumPy, PyTorch CPU, and CUDA."""
    if not states or not isinstance(states, dict):
        return
    try:
        if 'python' in states and states['python'] is not None:
            random.setstate(states['python'])
        if 'numpy' in states and states['numpy'] is not None:
            np.random.set_state(states['numpy'])
        if 'torch_cpu' in states and states['torch_cpu'] is not None:
            cpu_state = states['torch_cpu']
            if isinstance(cpu_state, torch.Tensor):
                cpu_state = cpu_state.cpu().to(torch.uint8)
            torch.set_rng_state(cpu_state)
        if 'torch_cuda' in states and states['torch_cuda'] is not None and torch.cuda.is_available():
            cuda_states = states['torch_cuda']
            clean_cuda_states = []
            for s in cuda_states:
                if isinstance(s, torch.Tensor):
                    clean_cuda_states.append(s.cpu().to(torch.uint8))
                else:
                    clean_cuda_states.append(s)
            torch.cuda.set_rng_state_all(clean_cuda_states)
    except Exception as e:
        print(f"[Warning] Could not restore all RNG states: {e}")



def evaluate_dfu_classifier(model, dataloader, device):
    """
    Evaluates 4-class DFU pathology using the direct ANN + Softmax classification head.
    Fast, GPU-accelerated evaluation under torch.no_grad().
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, lbls, _ in dataloader:
            imgs = imgs.to(device)
            _, logits = model.forward_single(imgs)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)  # UAR
    cm = confusion_matrix(all_labels, all_preds)

    # Matthews Correlation Coefficient (MCC)
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    mcc = matthews_corrcoef(all_labels, all_preds)

    # Macro-AUC (multi-class OvR)
    try:
        macro_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except Exception:
        macro_auc = 0.0

    # Per-class metrics
    class_names = ['none', 'infection', 'ischaemia', 'both']
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_prec = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_rec = recall_score(all_labels, all_preds, average=None, zero_division=0)

    metrics = {
        'macro_f1': float(macro_f1),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'uar': float(rec),
        'mcc': float(mcc),
        'macro_auc': float(macro_auc),
        'per_class_f1': {c: float(f) for c, f in zip(class_names, per_class_f1)},
        'per_class_precision': {c: float(p) for c, p in zip(class_names, per_class_prec)},
        'per_class_recall': {c: float(r) for c, r in zip(class_names, per_class_rec)},
        'confusion_matrix': cm.tolist()
    }
    return macro_f1, metrics


def train_dfu_siam(args):
    # Set global random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if getattr(args, 'device', None):
        device = torch.device(args.device)
    else:
        device = get_device()
    if device.type == 'cuda':
        if getattr(args, 'deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("[Reproducibility] cuDNN deterministic mode enabled (slower but reproducible).")
        else:
            torch.backends.cudnn.benchmark = True

    mode_str = "DFU-SIAM v3 Multi-Task Disentangled Architecture (DFUC + Wagner)" if args.enable_wagner else "DFU-SIAM v3 Pathology-Only Baseline"
    print(f"=== Starting DFU-SIAM v3 Training [{mode_str}] on {device} ===")

    if getattr(args, 'dry_run', False):
        print("\n[DRY RUN MODE] Printing resolved training configuration:")
        for k, v in sorted(vars(args).items()):
            print(f"  {k:24s}: {v}")
        print("\n[DRY RUN MODE] Configuration validated successfully. Exiting without training.\n")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    train_img_dir = os.path.join(args.data_dir, "DFUC2021_train", "images")
    train_csv = os.path.join(args.data_dir, "DFUC2021_train", "train.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv not found at {train_csv}")

    # Read CSV and perform Train/Val Stratified Split for DFUC
    df = pd.read_csv(train_csv)
    class_cols = ['none', 'infection', 'ischaemia', 'both']
    total_csv_rows = len(df)
    # Filter out unlabelled rows where all class columns are NaN to prevent false Class 0 ("none") labels
    df = df.dropna(subset=class_cols, how='all').reset_index(drop=True)
    labeled_rows = len(df)
    unlabeled_rows = total_csv_rows - labeled_rows
    df['class'] = np.argmax(df[class_cols].values, axis=1)

    # Data integrity assertions
    assert labeled_rows > 0, f"No labeled samples found after filtering! CSV has {total_csv_rows} total rows."
    assert not df[class_cols].isna().all(axis=1).any(), "Found rows with all-NaN labels after filtering!"
    class_counts = df['class'].value_counts().sort_index()
    samples_per_cls = [int(class_counts.get(i, 0)) for i in range(4)]
    print(f"DFUC CSV: {total_csv_rows} total rows, {unlabeled_rows} unlabeled (dropped), {labeled_rows} labeled.")
    print(f"  Class distribution: {dict(zip(class_cols, samples_per_cls))}")

    train_df, val_df = train_test_split(
        df, test_size=args.val_split, random_state=args.seed, stratify=df['class']
    )

    # Clean Manifest Quarantine Protocol
    clean_manifest_path = getattr(args, 'clean_manifest', None)
    if clean_manifest_path and os.path.exists(clean_manifest_path):
        import json
        with open(clean_manifest_path, 'r') as f:
            cmanifest = json.load(f)
        dfu_exclusions = set(cmanifest.get('dfu_train_exclusions', []))
        orig_train_count = len(train_df)
        train_df = train_df[~train_df['image'].isin(dfu_exclusions)].reset_index(drop=True)
        print(f"[Clean Split Protocol] Quarantined {orig_train_count - len(train_df)} cross-colliding images from DFU training set.")
        print(f"  --> Clean DFU Training Samples: {len(train_df)}")
        print(f"  --> Clean DFU Validation Samples: {len(val_df)} (preserved, untouched)")

    temp_train_csv = os.path.join(args.output_dir, "temp_train_split.csv")
    temp_val_csv = os.path.join(args.output_dir, "temp_val_split.csv")
    train_df.to_csv(temp_train_csv, index=False)
    val_df.to_csv(temp_val_csv, index=False)

    print(f"DFUC Dataset split: {len(train_df)} training samples, {len(val_df)} validation samples.")

    # Initialize DFUC Datasets and Loaders
    train_transform = get_dfu_transforms_v2(level=getattr(args, 'augmentation_level', 'baseline'), img_size=(args.img_size, args.img_size))
    train_pair_dataset = DFUSiamesePairDataset(
        img_dir=train_img_dir,
        csv_file=temp_train_csv,
        transform=train_transform,
        num_pairs=args.num_pairs_per_epoch
    )

    val_single_dataset = DFUSingleDataset(
        img_dir=train_img_dir,
        csv_file=temp_val_csv,
        transform=get_dfu_transforms(is_train=False, img_size=(args.img_size, args.img_size))
    )

    train_pair_loader = DataLoader(
        train_pair_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )
    val_single_loader = DataLoader(
        val_single_dataset,
        batch_size=max(args.batch_size * 4, 32),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )

    # Initialize Wagner dataset loaders if multi-task enabled
    wagner_train_ds = None
    wagner_val_ds = None
    wagner_te_ds = None
    if args.enable_wagner:
        wagner_train_dir = os.path.join(args.wagner_dir, "train")
        wagner_val_dir = os.path.join(args.wagner_dir, "valid")
        wagner_test_dir = os.path.join(args.wagner_dir, "test")
        if not os.path.exists(wagner_train_dir):
            wagner_train_dir = args.wagner_dir

        wagner_train_ds = WagnerSingleDataset(
            wagner_train_dir,
            is_byol=False,
            transform=train_transform,
            clean_manifest=clean_manifest_path
        )
        wagner_val_ds = WagnerSingleDataset(
            wagner_val_dir if os.path.exists(wagner_val_dir) else wagner_train_dir,
            is_byol=False,
            transform=get_dfu_transforms(is_train=False, img_size=(args.img_size, args.img_size)),
            clean_manifest=clean_manifest_path
        )
        wagner_te_ds = WagnerSingleDataset(
            wagner_test_dir,
            is_byol=False,
            transform=get_dfu_transforms(is_train=False, img_size=(args.img_size, args.img_size)),
            clean_manifest=clean_manifest_path
        ) if os.path.exists(wagner_test_dir) else None

        wagner_train_loader = DataLoader(
            wagner_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0)
        )
        wagner_val_loader = DataLoader(
            wagner_val_ds,
            batch_size=max(args.batch_size * 2, 16),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0)
        )

        interleaved_train_loader = InterleavedDataLoader(train_pair_loader, wagner_train_loader, mode='paired_step')
        print(f"Wagner Dataset loaded: {len(wagner_train_ds)} train samples, {len(wagner_val_ds)} val samples.")
    elif args.wagner_dir and os.path.exists(args.wagner_dir):
        # Wagner training is disabled (e.g. Pathology-only A0_Path baseline)
        # Load evaluation sets to verify zero cross-dataset leakage into DFU training
        wagner_val_dir = os.path.join(args.wagner_dir, "valid")
        wagner_test_dir = os.path.join(args.wagner_dir, "test")
        if os.path.exists(wagner_val_dir):
            wagner_val_ds = WagnerSingleDataset(wagner_val_dir, clean_manifest=clean_manifest_path)
        if os.path.exists(wagner_test_dir):
            wagner_te_ds = WagnerSingleDataset(wagner_test_dir, clean_manifest=clean_manifest_path)

    # Mandatory Pre-Training Cryptographic Leakage Assertion
    leakage_report = None
    if clean_manifest_path and not getattr(args, 'skip_leakage_assertion', False):
        leakage_report = assert_pre_training_leakage_free(
            dfu_train_dataset=train_pair_dataset,
            dfu_val_dataset=val_single_dataset,
            wagner_train_dataset=wagner_train_ds,
            wagner_val_dataset=wagner_val_ds,
            wagner_test_dataset=wagner_te_ds,
            clean_manifest_path=clean_manifest_path
        )

    # Initialize DFU-SIAM v3 Model
    model = DFUSiamV3(
        arch=args.arch,
        pretrained=True,
        shared_dim=args.shared_dim,
        path_dim=args.path_dim,
        depth_dim=args.depth_dim,
        dropout=args.dropout
    ).to(device)

    # Backbone Freezing Strategies
    if args.freeze_backbone or args.finetune_strategy in ['gradual_unfreeze', 'head_warmup']:
        model.freeze_backbones()
    elif args.partial_finetuning:
        model.partial_finetune_backbones()

    p_info = model.get_parameter_breakdown()
    print(f"\n[Model Architecture: {args.arch.upper()}]")
    print(f"  Total Parameters: {p_info['total']:,} ({p_info['fp32_size_mb']:.2f} MB FP32, {p_info['fp16_size_mb']:.2f} MB FP16)")
    print(f"  Trainable: {p_info['trainable']:,} | Non-trainable: {p_info['non_trainable']:,}")
    print(f"  CNN ({p_info['cnn']:,}) | ViT ({p_info['vit']:,})")
    print(f"  Projections ({p_info['cnn_proj'] + p_info['vit_proj']:,}) | Disentangler ({p_info['disentangler']:,})")
    print(f"  Pathology Head ({p_info['classifier']:,}) | Wagner Head ({p_info['wagner_head']:,})\n")


    # Save experiment config
    import json
    import datetime
    import subprocess
    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode('utf-8').strip()
    except Exception:
        git_rev = "git_unavailable"

    manifest_hash = None
    if clean_manifest_path and os.path.exists(clean_manifest_path):
        manifest_hash = compute_file_md5(clean_manifest_path)

    config_data = {
        "args": vars(args),
        "model_params": p_info,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_revision": git_rev,
        "clean_manifest_hash": manifest_hash,
        "clean_manifest_path": clean_manifest_path,
        "leakage_verification": leakage_report["status"] if leakage_report else "skipped",
        "versions": {
            "python": sys.version,
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
    }
    with open(os.path.join(args.output_dir, "experiment_config.json"), "w") as f:
        json.dump(config_data, f, indent=4)

    # Loss Functions Initialization
    if args.loss_pathology == 'cbfocal':
        pathology_criterion = ClassBalancedFocalLoss(
            samples_per_cls=samples_per_cls,
            num_classes=4,
            beta=args.focal_beta,
            gamma=args.focal_gamma
        ).to(device)
        print(f"[Loss] Pathology Supervised Loss: ClassBalancedFocalLoss (beta={args.focal_beta}, gamma={args.focal_gamma})")
    elif args.loss_pathology == 'ce':
        pathology_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
        print(f"[Loss] Pathology Supervised Loss: CrossEntropyLoss (label_smoothing={args.label_smoothing})")
    elif args.loss_pathology == 'wce':
        total_samples = sum(samples_per_cls)
        # Handle empty classes safely
        w = [total_samples / (4.0 * c) if c > 0 else 0.0 for c in samples_per_cls]
        w_tensor = torch.tensor(w, dtype=torch.float32).to(device)
        pathology_criterion = nn.CrossEntropyLoss(weight=w_tensor, label_smoothing=args.label_smoothing).to(device)
        print(f"[Loss] Pathology Supervised Loss: Weighted CrossEntropyLoss (label_smoothing={args.label_smoothing})")

    coral_criterion = CORALLoss(num_classes=5).to(device)
    lmcot_criterion = HardPairLMCoTLoss(margin=args.margin, warmup_epochs=args.lmcot_warmup_epochs).to(device)
    ug_mmd_criterion = UncertaintyGatedMMDLoss(min_weight=args.mmd_min_weight).to(device)
    orth_criterion = OrthogonalityLoss().to(device)

    # Optimizer Parameter Groups
    # Include all parameters so they can be unfrozen later
    backbone_params = list(model.backbone.cnn.parameters()) + list(model.backbone.vit.parameters())
    head_params = (
        list(model.backbone.vit_proj.parameters()) +
        list(model.backbone.cnn_proj.parameters()) +
        list(model.disentangler.parameters()) +
        list(model.classifier.parameters()) +
        list(model.wagner_head.parameters())
    )

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.lr_backbone, 'weight_decay': args.weight_decay_backbone})
    if head_params:
        param_groups.append({'params': head_params, 'lr': args.lr_head, 'weight_decay': args.weight_decay})
    optimizer = torch.optim.AdamW(param_groups)

    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-7)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=1024.0)

    best_monitored_metric = float('inf') if args.monitor_metric == 'mae' else -1.0
    
    # Paths for separated checkpoint files
    latest_model_path = os.path.join(args.output_dir, "latest_checkpoint.pth")
    best_model_path = os.path.join(args.output_dir, "best_checkpoint.pth")
    epoch_model_path = os.path.join(args.output_dir, "epoch_checkpoint.pth")
    legacy_best_path = os.path.join(args.output_dir, "dfu_siam_v3_best.pth")
    legacy_last_path = os.path.join(args.output_dir, "dfu_siam_v3_last.pth")

    start_epoch = 1
    start_batch_idx = 0
    global_step = 0
    resumed_pair_state = None

    # Resolve resume checkpoint path if --resume, --resume_from, or --resume_checkpoint is specified
    resume_target = args.resume_from or args.resume_checkpoint
    if args.resume and not resume_target:
        if os.path.isfile(latest_model_path):
            resume_target = latest_model_path
        elif os.path.isfile(legacy_last_path):
            resume_target = legacy_last_path

    # Checkpoint Resume Logic with Strict Version Validation and Mid-Epoch Restoration
    if resume_target:
        if not os.path.isfile(resume_target):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_target}")

        print(f"\n[Resume] Loading checkpoint from: {resume_target}")
        checkpoint = torch.load(resume_target, map_location=device, weights_only=False)
        ckpt_ver = checkpoint.get('model_version', 'legacy')
        if ckpt_ver != 'v3':
            raise ValueError(
                f"Incompatible checkpoint: expected model_version='v3', but found {ckpt_ver!r}. "
                "DFU-SIAM v3 uses a task-disentangled architecture and cannot load legacy v1/v2 weights."
            )

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None and use_amp:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore RNG states if available
        if 'rng_states' in checkpoint:
            set_rng_states(checkpoint['rng_states'])
            print("  --> RNG states successfully restored across Python, NumPy, and PyTorch.")

        # Determine exact resumption point (mid-epoch vs new epoch)
        ckpt_epoch = int(checkpoint.get('epoch', 1))
        ckpt_batch = int(checkpoint.get('batch', -1))
        global_step = int(checkpoint.get('global_step', 0))
        best_monitored_metric = float(checkpoint.get('best_metric', best_monitored_metric))

        if ckpt_batch >= 0:
            # Checkpoint saved mid-epoch or after batch completed
            start_epoch = ckpt_epoch
            start_batch_idx = ckpt_batch + 1
            resumed_pair_state = checkpoint.get('pair_state', None)
            print(f"  --> Resuming MID-EPOCH at Epoch {start_epoch}, Batch {start_batch_idx} (Global Step {global_step}).")
        else:
            # Checkpoint saved at end of epoch
            start_epoch = ckpt_epoch + 1
            start_batch_idx = 0
            print(f"  --> Resuming at START of Epoch {start_epoch} (Global Step {global_step}).")

        print(f"  --> Current best {args.monitor_metric}: {best_monitored_metric:.4f}")

    # Graceful Interruption Handling (Ctrl+C / SIGINT / SIGTERM)
    interrupted = {'flag': False}

    def handle_interrupt(signum, frame):
        if not interrupted['flag']:
            print(f"\n[Signal Received ({signum})] Graceful shutdown requested. Completing active batch and saving checkpoint...")
            interrupted['flag'] = True
        else:
            print(f"\n[Force Exit] Second signal received. Exiting immediately.")
            sys.exit(1)

    try:
        signal.signal(signal.SIGINT, handle_interrupt)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, handle_interrupt)
    except Exception:
        pass

    def build_checkpoint_payload(cur_epoch: int, cur_batch: int, cur_gstep: int, cur_best: float, pair_state_dict=None):
        payload = {
            'model_version': 'v3',
            'epoch': cur_epoch,
            'batch': cur_batch,
            'global_step': cur_gstep,
            'best_metric': cur_best,
            'monitor_metric': args.monitor_metric,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if use_amp else None,
            'rng_states': get_rng_states(),
            'pair_state': pair_state_dict,
            'config': vars(args)
        }
        return payload

    print(f"Training DFU-SIAM v3 for {args.epochs} epochs (LR Backbone={args.lr_backbone}, LR Head={args.lr_head}, AMP={use_amp})...")
    start_time = time.time()


    epochs_without_improvement = 0
    learning_curves = {
        'train_loss_total': [],
        'train_loss_components': [],
        'val_metrics_dfu': [],
        'val_metrics_wagner': [],
        'gap': [],
        'epoch_diagnostics': []
    }
    lc_file = os.path.join(args.output_dir, "learning_curves.json")
    if resume_target and os.path.exists(lc_file):
        try:
            with open(lc_file, "r") as f:
                loaded_lc = json.load(f)
            for k in learning_curves.keys():
                if k in loaded_lc:
                    learning_curves[k] = loaded_lc[k]
            print(f"[Resume] Preserved {len(learning_curves['train_loss_total'])} epochs of learning curve history.")
        except Exception as e:
            print(f"[Resume Warning] Could not parse existing learning_curves.json: {e}")

    for epoch in range(start_epoch, args.epochs + 1):
        if interrupted['flag']:
            print(f"[Training Halted] Exiting loop before Epoch {epoch} due to interrupt signal.")
            break


        if epoch == args.unfreeze_epoch and args.finetune_strategy in ['gradual_unfreeze', 'head_warmup']:
            print(f"\n[Finetune] Epoch {epoch}: Unfreezing backbones ({args.finetune_strategy} strategy).")
            model.unfreeze_backbones()
            if args.finetune_strategy == 'head_warmup':
                print("[Finetune] Reducing head learning rate.")
                for param_group in optimizer.param_groups:
                    # simplistic check: if it's the head group
                    if param_group.get('weight_decay') == args.weight_decay and len(optimizer.param_groups) > 1:
                        param_group['lr'] = args.lr_head * 0.1

        epoch_start_time = time.time()
        print(f"\n>>> Starting Epoch {epoch:02d}/{args.epochs:02d} ...")
        model.train()
        total_loss = 0.0
        num_batches = 0
        loss_components = {'path': 0.0, 'coral': 0.0, 'lmcot': 0.0, 'mmd': 0.0, 'orth': 0.0}

        # Determine start batch for this epoch
        current_start_batch = start_batch_idx if epoch == start_epoch else 0

        # Regenerate or restore class-balanced pairs for each epoch (only when DFUC is trained)
        if args.lambda_path > 0.0 or args.lambda_lmcot > 0.0 or args.lambda_mmd > 0.0:
            if epoch == start_epoch and resumed_pair_state is not None:
                train_pair_dataset.pairs = resumed_pair_state['pairs']
                train_pair_dataset.pair_labels = resumed_pair_state['pair_labels']
                train_pair_dataset.class_targets = resumed_pair_state['class_targets']
                print(f"[Resume] Restored exact {len(train_pair_dataset.pairs)} pairs from mid-epoch checkpoint.")
                resumed_pair_state = None  # Consume once
            else:
                train_pair_dataset.pairs, train_pair_dataset.pair_labels, train_pair_dataset.class_targets = train_pair_dataset._generate_pairs()

        pair_state_for_saving = {
            'pairs': getattr(train_pair_dataset, 'pairs', None),
            'pair_labels': getattr(train_pair_dataset, 'pair_labels', None),
            'class_targets': getattr(train_pair_dataset, 'class_targets', None)
        } if (args.lambda_path > 0.0 or args.lambda_lmcot > 0.0 or args.lambda_mmd > 0.0) else None

        if args.enable_wagner and args.lambda_path == 0.0 and args.lambda_lmcot == 0.0 and args.lambda_mmd == 0.0 and args.lambda_orth == 0.0:
            # Genuine Single-Task Wagner Ordinal Baseline (A0-Wag)
            for batch_idx, (w_img, w_lbl, _) in enumerate(wagner_train_loader):
                if batch_idx < current_start_batch:
                    continue  # Fast forward past already completed batches

                w_img = w_img.to(device, non_blocking=True)
                w_lbl = w_lbl.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    z_s_wag, z_d_wag, w_logits = model.forward_wagner(w_img)
                    l_coral, _, _ = coral_criterion(w_logits, w_lbl)
                    loss = args.lambda_coral * l_coral

                scaler.scale(loss).backward()
                if getattr(args, 'grad_clip', 0.0) > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                loss_components['coral'] += l_coral.item()
                total_loss += loss.item()
                num_batches += 1
                global_step += 1

                # Periodic step-based checkpoint saving
                if args.checkpoint_every > 0 and (global_step % args.checkpoint_every == 0):
                    step_payload = build_checkpoint_payload(epoch, batch_idx, global_step, best_monitored_metric, pair_state_for_saving)
                    save_atomic_checkpoint(step_payload, latest_model_path)
                    print(f" [Epoch {epoch:02d} | Batch {batch_idx:04d} | Global Step {global_step}] Saved latest checkpoint to {latest_model_path}")

                if interrupted['flag']:
                    intr_payload = build_checkpoint_payload(epoch, batch_idx, global_step, best_monitored_metric, pair_state_for_saving)
                    save_atomic_checkpoint(intr_payload, latest_model_path)
                    print(f"\n[Interrupted] Saved resumable state at Epoch {epoch}, Batch {batch_idx} to {latest_model_path}.")
                    break

                if args.max_batches_per_epoch is not None and num_batches >= args.max_batches_per_epoch:
                    break
        elif args.enable_wagner:
            for batch_idx, (dfu_batch, wagner_batch) in enumerate(interleaved_train_loader):
                if batch_idx < current_start_batch:
                    continue  # Fast forward past already completed batches

                img1, img2, pair_target, c1, c2 = dfu_batch
                w_img, w_lbl, _ = wagner_batch

                img1 = img1.to(device, non_blocking=True)
                img2 = img2.to(device, non_blocking=True)
                pair_target = pair_target.to(device, non_blocking=True)
                c1 = c1.to(device, non_blocking=True)
                c2 = c2.to(device, non_blocking=True)
                w_img = w_img.to(device, non_blocking=True)
                w_lbl = w_lbl.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    # 1. Forward DFUC images (Pathology Branch)
                    z_s1, z_p1, logits1 = model.forward_pathology(img1)
                    z_s2, z_p2, logits2 = model.forward_pathology(img2)

                    # Pathology supervised loss
                    if args.lambda_path > 0.0:
                        l_path = 0.5 * (pathology_criterion(logits1, c1) + pathology_criterion(logits2, c2))
                    else:
                        l_path = torch.tensor(0.0, device=device)

                    # LMCoT Metric Learning on Shared Representations
                    if args.use_hard_lmcot:
                        all_zs = torch.cat([z_s1, z_s2], dim=0)
                        all_c = torch.cat([c1, c2], dim=0)
                        l_lmcot = lmcot_criterion.forward_mined(all_zs, all_c, epoch=epoch)
                    else:
                        l_lmcot = lmcot_criterion.forward_paired(z_s1, z_s2, pair_target)

                    # Orthogonality for DFUC images: z_shared ⟂ z_path_private
                    l_orth_dfu = 0.5 * (orth_criterion(z_s1, z_path_private=z_p1) + orth_criterion(z_s2, z_path_private=z_p2))

                    # 2. Forward Wagner images (Ordinal Branch)
                    z_s_wag, z_d_wag, w_logits = model.forward_wagner(w_img)

                    # Wagner ordinal CORAL loss
                    l_coral, _, _ = coral_criterion(w_logits, w_lbl)

                    # Orthogonality for Wagner images: z_shared ⟂ z_depth_private
                    l_orth_wag = orth_criterion(z_s_wag, z_depth_private=z_d_wag)

                    # Combined Orthogonality Regularization
                    l_orth = 0.5 * (l_orth_dfu + l_orth_wag)

                    # 3. Uncertainty-Gated MMD on Shared Representations Only
                    w_dfu = UncertaintyGatedMMDLoss.compute_pathology_uncertainty(logits1, min_weight=args.mmd_min_weight)
                    w_wag = UncertaintyGatedMMDLoss.compute_wagner_uncertainty(w_logits, min_weight=args.mmd_min_weight)
                    l_mmd = ug_mmd_criterion(z_s1, z_s_wag, weights_x=w_dfu, weights_y=w_wag)

                    # 4. Total DFU-SIAM v3 Objective
                    loss = (
                        args.lambda_path * l_path +
                        args.lambda_coral * l_coral +
                        args.lambda_lmcot * l_lmcot +
                        args.lambda_mmd * l_mmd +
                        args.lambda_orth * l_orth
                    )

                scaler.scale(loss).backward()
                if getattr(args, 'grad_clip', 0.0) > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                if args.lambda_path > 0.0:
                    loss_components['path'] += l_path.item()
                loss_components['coral'] += l_coral.item()
                loss_components['lmcot'] += l_lmcot.item()
                loss_components['mmd'] += l_mmd.item()
                loss_components['orth'] += l_orth.item()

                total_loss += loss.item()
                num_batches += 1
                global_step += 1

                # Periodic step-based checkpoint saving
                if args.checkpoint_every > 0 and (global_step % args.checkpoint_every == 0):
                    step_payload = build_checkpoint_payload(epoch, batch_idx, global_step, best_monitored_metric, pair_state_for_saving)
                    save_atomic_checkpoint(step_payload, latest_model_path)
                    print(f" [Epoch {epoch:02d} | Batch {batch_idx:04d} | Global Step {global_step}] Saved latest checkpoint to {latest_model_path}")

                if interrupted['flag']:
                    intr_payload = build_checkpoint_payload(epoch, batch_idx, global_step, best_monitored_metric, pair_state_for_saving)
                    save_atomic_checkpoint(intr_payload, latest_model_path)
                    print(f"\n[Interrupted] Saved resumable state at Epoch {epoch}, Batch {batch_idx} to {latest_model_path}.")
                    break

                if args.max_batches_per_epoch is not None and num_batches >= args.max_batches_per_epoch:
                    break
        else:
            # Baseline Pathology-Only Mode
            for batch_idx, (img1, img2, pair_target, c1, c2) in enumerate(train_pair_loader):
                if batch_idx < current_start_batch:
                    continue  # Fast forward past already completed batches

                img1, img2 = img1.to(device), img2.to(device)
                pair_target = pair_target.to(device)
                c1, c2 = c1.to(device), c2.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    z_s1, z_p1, logits1 = model.forward_pathology(img1)
                    z_s2, z_p2, logits2 = model.forward_pathology(img2)

                    l_path = 0.5 * (pathology_criterion(logits1, c1) + pathology_criterion(logits2, c2))
                    if args.use_hard_lmcot:
                        all_zs = torch.cat([z_s1, z_s2], dim=0)
                        all_c = torch.cat([c1, c2], dim=0)
                        l_lmcot = lmcot_criterion.forward_mined(all_zs, all_c, epoch=epoch)
                    else:
                        l_lmcot = lmcot_criterion.forward_paired(z_s1, z_s2, pair_target)

                    l_orth = 0.5 * (orth_criterion(z_s1, z_path_private=z_p1) + orth_criterion(z_s2, z_path_private=z_p2))
                    loss = l_path + args.lambda_lmcot * l_lmcot + args.lambda_orth * l_orth

                scaler.scale(loss).backward()
                if getattr(args, 'grad_clip', 0.0) > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                loss_components['path'] += l_path.item()
                loss_components['lmcot'] += l_lmcot.item()
                loss_components['orth'] += l_orth.item()

                total_loss += loss.item()
                num_batches += 1
                global_step += 1

                # Periodic step-based checkpoint saving
                if args.checkpoint_every > 0 and (global_step % args.checkpoint_every == 0):
                    step_payload = build_checkpoint_payload(epoch, batch_idx, global_step, best_monitored_metric, pair_state_for_saving)
                    save_atomic_checkpoint(step_payload, latest_model_path)
                    print(f" [Epoch {epoch:02d} | Batch {batch_idx:04d} | Global Step {global_step}] Saved latest checkpoint to {latest_model_path}")

                if interrupted['flag']:
                    intr_payload = build_checkpoint_payload(epoch, batch_idx, global_step, best_monitored_metric, pair_state_for_saving)
                    save_atomic_checkpoint(intr_payload, latest_model_path)
                    print(f"\n[Interrupted] Saved resumable state at Epoch {epoch}, Batch {batch_idx} to {latest_model_path}.")
                    break

                if args.max_batches_per_epoch is not None and num_batches >= args.max_batches_per_epoch:
                    break

        if interrupted['flag']:
            print(f"[Training Stopped] Stopped during Epoch {epoch}.")
            break

        scheduler.step()
        avg_loss = total_loss / max(1, num_batches)
        comp_summary = " | ".join([f"{k.capitalize()}={v/max(1, num_batches):.4f}" for k, v in loss_components.items() if v > 0])
        
        gpu_mem_str = ""
        if device.type == 'cuda':
            alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            max_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            gpu_mem_str = f" | GPU Mem: {alloc_mb:.1f}MB (Peak: {max_alloc_mb:.1f}MB)"

        print(f"\n[Epoch {epoch:02d}/{args.epochs:02d}] Train Total Loss: {avg_loss:.4f} ({comp_summary}){gpu_mem_str}")

        # Validation & Evaluation
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            if args.lambda_path > 0.0:
                macro_f1, dfu_metrics = evaluate_dfu_classifier(model, val_single_loader, device)
                dfu_str = f"Macro-F1: {macro_f1:.4f} | Acc: {dfu_metrics['accuracy']:.4f} | UAR: {dfu_metrics.get('uar', 0.0):.4f} | MCC: {dfu_metrics.get('mcc', 0.0):.4f}"
            else:
                macro_f1 = 0.0
                dfu_metrics = {'accuracy': 0.0, 'macro_f1': 0.0}
                dfu_str = "Pathology: Disabled (Single-Task Wagner Baseline)"

            wagner_str = ""
            if args.enable_wagner:
                w_eval = evaluate_wagner_ordinal(model, wagner_val_loader, device, coral_criterion)
                zs_eval = evaluate_zero_shot_cross_taxonomy(model, val_single_loader, wagner_val_loader, device, coral_criterion) if args.lambda_path > 0.0 else {}
                wagner_str = f" | Wagner Val Loss: {w_eval['loss']:.4f} | Wagner Acc: {w_eval['accuracy']:.4f} | MAE: {w_eval['mae']:.4f} | QWK: {w_eval.get('quadratic_weighted_kappa', 0.0):.4f}"

            print(f"  --> Val Metrics: {dfu_str}{wagner_str}")

            if args.monitor_metric == 'joint':
                # Harmonic mean of DFUC Accuracy and Wagner Adjacent/Exact Accuracy
                dfu_acc = dfu_metrics.get('accuracy', 0.0)
                wag_acc = w_eval.get('accuracy', 0.0) if args.enable_wagner else 0.0
                wag_qwk = max(0.0, w_eval.get('quadratic_weighted_kappa', 0.0)) if args.enable_wagner else 0.0
                # Balanced score combining DFUC accuracy, Wagner accuracy and QWK
                monitored_val = 2.0 * (dfu_acc * (0.5 * wag_acc + 0.5 * wag_qwk)) / max(1e-6, dfu_acc + (0.5 * wag_acc + 0.5 * wag_qwk))
                is_better = monitored_val > best_monitored_metric + args.early_stopping_min_delta
            elif args.monitor_metric == 'qwk':
                monitored_val = w_eval.get('quadratic_weighted_kappa', 0.0) if args.enable_wagner else 0.0
                is_better = monitored_val > best_monitored_metric + args.early_stopping_min_delta
            elif args.monitor_metric == 'mae':
                monitored_val = w_eval.get('mae', float('inf')) if args.enable_wagner else float('inf')
                is_better = monitored_val < best_monitored_metric - args.early_stopping_min_delta
            elif args.monitor_metric == 'wagner_acc':
                monitored_val = w_eval.get('accuracy', 0.0) if args.enable_wagner else 0.0
                is_better = monitored_val > best_monitored_metric + args.early_stopping_min_delta
            elif args.monitor_metric == 'accuracy':
                monitored_val = dfu_metrics.get('accuracy', 0.0)
                is_better = monitored_val > best_monitored_metric + args.early_stopping_min_delta
            else:  # 'macro_f1'
                monitored_val = macro_f1
                is_better = monitored_val > best_monitored_metric + args.early_stopping_min_delta

            if is_better:
                best_monitored_metric = monitored_val
                best_ckpt_dict = build_checkpoint_payload(epoch, -1, global_step, best_monitored_metric, pair_state_for_saving)
                best_ckpt_dict.update({
                    'macro_f1': macro_f1,
                    'dfu_metrics': dfu_metrics,
                })
                if args.enable_wagner:
                    best_ckpt_dict['wagner_metrics'] = w_eval
                    best_ckpt_dict['zero_shot_metrics'] = zs_eval

                save_atomic_checkpoint(best_ckpt_dict, best_model_path)
                print(f" --> Best DFU-SIAM v3 model saved to {best_model_path} ({args.monitor_metric} = {monitored_val:.4f})")
        else:
            print(f"[Epoch {epoch:02d}/{args.epochs:02d}] Loss: {avg_loss:.4f}")


        # Update learning curves
        epoch_train_loss = avg_loss
        epoch_components = {k: v/max(1, num_batches) for k, v in loss_components.items() if v > 0}
        
        learning_curves['train_loss_total'].append(epoch_train_loss)
        learning_curves['train_loss_components'].append(epoch_components)
        
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            learning_curves['val_metrics_dfu'].append(dfu_metrics)
            if args.enable_wagner:
                learning_curves['val_metrics_wagner'].append(w_eval)
            
            # gap calculation (simple difference between train accuracy and val accuracy if available, or just keeping track of both)
            val_acc = dfu_metrics.get('accuracy', 0.0)
            # pseudo train acc could be derived but we just store the gap as train_loss vs val_loss for simplicity
            learning_curves['gap'].append({'train_loss': epoch_train_loss, 'val_acc': val_acc})

        # Epoch Diagnostics (runtime, GPU memory, learning rates)
        epoch_dur = time.time() - epoch_start_time
        gpu_alloc = (torch.cuda.memory_allocated(device) / (1024**2)) if device.type == 'cuda' else 0.0
        gpu_res = (torch.cuda.memory_reserved(device) / (1024**2)) if device.type == 'cuda' else 0.0
        lr_bb = float(optimizer.param_groups[0]['lr']) if optimizer.param_groups else 0.0
        lr_hd = float(optimizer.param_groups[1]['lr']) if len(optimizer.param_groups) > 1 else lr_bb
        diag_entry = {
            'epoch': epoch,
            'duration_sec': float(round(epoch_dur, 2)),
            'gpu_mem_alloc_mb': float(round(gpu_alloc, 2)),
            'gpu_mem_reserved_mb': float(round(gpu_res, 2)),
            'lr_backbone': lr_bb,
            'lr_head': lr_hd
        }
        if 'epoch_diagnostics' not in learning_curves:
            learning_curves['epoch_diagnostics'] = []
        learning_curves['epoch_diagnostics'].append(diag_entry)
        
        with open(os.path.join(args.output_dir, "learning_curves.json"), "w") as f:
            json.dump(learning_curves, f, indent=4)

        if (epoch % args.val_interval == 0 or epoch == args.epochs) and args.early_stopping_patience > 0:
            if is_better:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"  --> Early stopping counter: {epochs_without_improvement} / {args.early_stopping_patience}")
                if epochs_without_improvement >= args.early_stopping_patience:
                    print(f"\n[Early Stopping] No improvement for {args.early_stopping_patience} epochs. Stopping training.")
                    break

        # Save epoch checkpoint (batch = -1 signifies completed epoch)
        epoch_ckpt_dict = build_checkpoint_payload(epoch, -1, global_step, best_monitored_metric, pair_state_for_saving)
        save_atomic_checkpoint(epoch_ckpt_dict, latest_model_path)
        if args.keep_all_epochs:
            per_epoch_path = os.path.join(args.output_dir, f"epoch_{epoch:03d}.pth")
            save_atomic_checkpoint(epoch_ckpt_dict, per_epoch_path)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    total_duration = time.time() - start_time
    if not interrupted['flag']:
        print(f"\n=== Training Complete in {total_duration/60:.2f} minutes ===")
        print(f"Best Validation {args.monitor_metric}: {best_monitored_metric:.4f}")
        if getattr(args, 'save_only_best', False):
            # Clean up all non-best checkpoints to enforce storage limit (e.g. Kaggle 20GB limit)
            for non_best in [latest_model_path, epoch_model_path, legacy_last_path]:
                if os.path.isfile(non_best):
                    try:
                        os.remove(non_best)
                        print(f"🧹 Removed non-best checkpoint to conserve disk: {os.path.basename(non_best)}")
                    except OSError:
                        pass
            print(f"Saved Checkpoint: {best_model_path} (Only best_checkpoint.pth retained for disk quota)")
        else:
            print(f"Saved Checkpoints: {best_model_path}, {latest_model_path}")
    else:
        print(f"\n=== Training Gracefully Halted in {total_duration/60:.2f} minutes ===")
        print(f"Latest Resumable Checkpoint Saved to: {latest_model_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DFU-SIAM v3 Task-Disentangled Multi-Task Architecture", allow_abbrev=False)
    parser.add_argument("--data_dir", type=str, default="DFUdataset", help="Path to DFUdataset directory")
    parser.add_argument("--wagner_dir", type=str, default="WagnerDataset", help="Path to WagnerDataset directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints_v3", help="Output directory for v3 checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training (e.g. 'cuda', 'cuda:0', 'cuda:1', or 'cpu')")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint in output_dir")
    parser.add_argument("--resume_from", type=str, default=None, help="Explicit checkpoint path to resume training from")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Alias for --resume_from for backward compatibility")
    parser.add_argument("--checkpoint_every", type=int, default=100, help="Save latest_checkpoint.pth every N training steps (0 to disable mid-epoch saving)")
    parser.add_argument("--save_every_epoch", action="store_true", default=True, help="Save epoch_checkpoint.pth at the end of each completed epoch")
    parser.add_argument("--keep_all_epochs", action="store_true", help="Save distinct epoch_{epoch}.pth checkpoints in addition to epoch_checkpoint.pth")
    parser.add_argument("--save_only_best", action="store_true", help="Purge intermediate/latest checkpoints upon completion to strictly preserve disk space (e.g. Kaggle 20GB limit)")
    parser.add_argument("--enable_wagner", action="store_true", help="Enable Wagner dataset joint ordinal training & alignment")

    # Architecture Configuration
    parser.add_argument("--arch", type=str, default="swin_tiny", choices=["swin_tiny", "full", "lightweight"], help="Model architecture: 'swin_tiny' (EfficientNetV2-B0 + Swin-Tiny ~35.2M), 'full' (EfficientNetV2-S + Swin-Tiny ~59.8M), or 'lightweight' (EfficientNetV2-B0 + MobileViT-S ~12.5M)")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze both CNN and ViT backbones, training only projection, disentangler, and heads")
    parser.add_argument("--partial_finetuning", action="store_true", help="Freeze lower backbone stages while fine-tuning top stages and heads")

    # Representation Subspace Dimensions
    parser.add_argument("--shared_dim", type=int, default=None, help="Dimension of shared transferable subspace z_shared (default: 768 for full, 256 for swin_tiny/lightweight)")
    parser.add_argument("--path_dim", type=int, default=None, help="Dimension of pathology-private subspace z_path_private (default: 512 for full, 256 for swin_tiny/lightweight)")
    parser.add_argument("--depth_dim", type=int, default=None, help="Dimension of depth-private subspace z_depth_private (default: 512 for full, 256 for swin_tiny/lightweight)")

    # Loss Configuration
    parser.add_argument("--loss_pathology", type=str, default="cbfocal", choices=["cbfocal", "ce", "wce"], help="Pathology supervised loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focusing parameter gamma for Focal Loss")
    parser.add_argument("--focal_beta", type=float, default=0.999, help="Class balancing parameter beta (effective number of samples)")

    # Loss Hyperparameter Weights & Ablation Toggles
    parser.add_argument("--lambda_path", type=float, default=1.0, help="Weight factor for DFUC pathology supervised loss (set 0.0 to disable for Wagner-only baseline)")
    parser.add_argument("--lambda_coral", type=float, default=1.0, help="Weight factor for Wagner CORAL ordinal loss (set 0.0 to disable)")
    parser.add_argument("--lambda_lmcot", type=float, default=1.0, help="Weight factor for Siamese LMCoT loss (set 0.0 to disable)")
    parser.add_argument("--lambda_mmd", type=float, default=0.1, help="Weight factor for Uncertainty-Gated MMD alignment loss (set 0.0 to disable)")
    parser.add_argument("--lambda_orth", type=float, default=0.1, help="Weight factor for Orthogonality / Disentanglement loss (set 0.0 to disable)")

    # Metric Learning & Gating
    parser.add_argument("--margin", type=float, default=2.0, help="Margin parameter for LMCoT loss")
    parser.add_argument("--use_hard_lmcot", action="store_true", help="Enable hard-pair mining for LMCoT")
    parser.add_argument("--lmcot_warmup_epochs", type=int, default=5, help="Epochs before switching from semi-hard to hardest pair mining")
    parser.add_argument("--mmd_min_weight", type=float, default=0.1, help="Minimum sample weight clamp for uncertainty gating")

    # Optimization & Stability
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr_backbone", type=float, default=3e-6, help="Learning rate for pretrained backbones")
    parser.add_argument("--lr_head", type=float, default=1e-4, help="Learning rate for projection, disentanglement, and task heads")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--num_pairs_per_epoch", type=int, default=12000, help="Number of class-balanced pairs generated per epoch")
    parser.add_argument("--img_size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation set split fraction")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation evaluation interval (epochs)")
    parser.add_argument("--monitor_metric", type=str, default="macro_f1", choices=["macro_f1", "accuracy", "qwk", "mae", "wagner_acc", "joint"], help="Validation metric for checkpoint selection ('joint' balances DFUC accuracy and Wagner metrics)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Enable cuDNN deterministic mode for full reproducibility")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Maximum gradient norm threshold (0.0 to disable)")
    parser.add_argument("--pilot", action="store_true", help="Run short pilot mode for training pipeline verification")
    parser.add_argument("--max_batches_per_epoch", type=int, default=None, help="Cap training batches per epoch (for rapid sanity checks)")
    parser.add_argument("--dry_run", action="store_true", help="Print resolved configuration and exit without training")

    # Early Stopping
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="Epochs without improvement before stopping (0=disabled)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001, help="Minimum improvement to count as progress")

    # Label Smoothing  
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor (0.0=disabled, 0.1=recommended)")

    # Learning Rate Scheduling
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs with linear LR ramp")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for AdamW optimizer")
    parser.add_argument("--weight_decay_backbone", type=float, default=1e-3, help="Weight decay for backbone parameters")

    # Training Mode
    parser.add_argument("--finetune_strategy", type=str, default="none", choices=["none","gradual_unfreeze","head_warmup"], help="Fine-tuning strategy")
    parser.add_argument("--unfreeze_epoch", type=int, default=5, help="Epoch to start unfreezing backbone (for gradual_unfreeze and head_warmup)")

    # Weighted Sampling
    parser.add_argument("--use_weighted_sampler", action="store_true", help="Use class-balanced weighted random sampler")
    parser.add_argument("--sampler_mode", type=str, default="inverse_freq", choices=["inverse_freq","sqrt_inverse","effective_num"], help="Sampling weight strategy")

    # Augmentation
    parser.add_argument("--augmentation_level", type=str, default="baseline", choices=["baseline","mild","strong","clinical"], help="Data augmentation intensity level")

    # Experiment Naming & Clean Manifest Protocol
    parser.add_argument("--experiment_name", type=str, default=None, help="Name for this experiment run")
    parser.add_argument("--clean_manifest", type=str, default="research_paper/clean_leak_free_split_manifest.json", help="Path to clean leak-free split quarantine manifest")
    parser.add_argument("--skip_leakage_assertion", action="store_true", help="Skip pre-training cryptographic leakage assertion (not recommended)")

    args = parser.parse_args()
    train_dfu_siam(args)
