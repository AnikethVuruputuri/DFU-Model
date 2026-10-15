"""
Automated Orchestration Runner for DFU-SIAM v3 Experiment Suite.
Supports:
- Running A0-Path, A0-Wag, A1-Full, and ablations A2-A7 across seeds [42, 123, 2026]
- Automatic evaluation via evaluate_checkpoint.py on validation & test splits
- Multi-seed metric aggregation via aggregate_seeds.py
- Progress logging and state resumption
"""

import os
import sys
import json
import time
import argparse
import subprocess

PYTHON_EXE = sys.executable

EXPERIMENT_CONFIGS = {
    "A0_Path": {
        "name": "Pathology Supervised Baseline (A0-Path)",
        "flags": [
            "--loss_pathology", "ce",
            "--lambda_path", "1.0",
            "--lambda_coral", "0.0",
            "--lambda_lmcot", "0.0",
            "--lambda_mmd", "0.0",
            "--lambda_orth", "0.0",
            "--monitor_metric", "macro_f1"
        ]
    },
    "A0_Wag": {
        "name": "Wagner Ordinal Baseline (A0-Wag)",
        "flags": [
            "--enable_wagner",
            "--lambda_path", "0.0",
            "--lambda_coral", "1.0",
            "--lambda_lmcot", "0.0",
            "--lambda_mmd", "0.0",
            "--lambda_orth", "0.0",
            "--monitor_metric", "qwk"
        ]
    },
    "A1_Full": {
        "name": "Full Proposed DFU-SIAM v3 (A1)",
        "flags": [
            "--arch", "full",
            "--enable_wagner",
            "--use_hard_lmcot",
            "--loss_pathology", "cbfocal",
            "--focal_beta", "0.999",
            "--focal_gamma", "2.0",
            "--lambda_path", "1.0",
            "--lambda_coral", "1.0",
            "--lambda_lmcot", "0.5",
            "--lambda_mmd", "0.1",
            "--lambda_orth", "0.1",
            "--monitor_metric", "macro_f1"
        ]
    },
    "A2_NoLMCoT": {
        "name": "No LMCoT Metric Learning Ablation (A2)",
        "flags": [
            "--enable_wagner",
            "--use_hard_lmcot",
            "--loss_pathology", "cbfocal",
            "--focal_beta", "0.999",
            "--focal_gamma", "2.0",
            "--lambda_path", "1.0",
            "--lambda_coral", "1.0",
            "--lambda_lmcot", "0.0",
            "--lambda_mmd", "0.1",
            "--lambda_orth", "0.1",
            "--monitor_metric", "macro_f1"
        ]
    },
    "A3_NoUGMMD": {
        "name": "No Uncertainty-Gated MMD Ablation (A3)",
        "flags": [
            "--enable_wagner",
            "--use_hard_lmcot",
            "--loss_pathology", "cbfocal",
            "--focal_beta", "0.999",
            "--focal_gamma", "2.0",
            "--lambda_path", "1.0",
            "--lambda_coral", "1.0",
            "--lambda_lmcot", "0.5",
            "--lambda_mmd", "0.0",
            "--lambda_orth", "0.1",
            "--monitor_metric", "macro_f1"
        ]
    },
    "A4_NoOrth": {
        "name": "No Orthogonality Disentanglement Ablation (A4)",
        "flags": [
            "--enable_wagner",
            "--use_hard_lmcot",
            "--loss_pathology", "cbfocal",
            "--focal_beta", "0.999",
            "--focal_gamma", "2.0",
            "--lambda_path", "1.0",
            "--lambda_coral", "1.0",
            "--lambda_lmcot", "0.5",
            "--lambda_mmd", "0.1",
            "--lambda_orth", "0.0",
            "--monitor_metric", "macro_f1"
        ]
    },
    "A6_StandardCE": {
        "name": "Standard CE Instead of CB-Focal (A6)",
        "flags": [
            "--enable_wagner",
            "--use_hard_lmcot",
            "--loss_pathology", "ce",
            "--lambda_path", "1.0",
            "--lambda_coral", "1.0",
            "--lambda_lmcot", "0.5",
            "--lambda_mmd", "0.1",
            "--lambda_orth", "0.1",
            "--monitor_metric", "macro_f1"
        ]
    },
    "A7_NoHardMining": {
        "name": "Uniform LMCoT Sampling Without Hard Mining (A7)",
        "flags": [
            "--enable_wagner",
            "--loss_pathology", "cbfocal",
            "--focal_beta", "0.999",
            "--focal_gamma", "2.0",
            "--lambda_path", "1.0",
            "--lambda_coral", "1.0",
            "--lambda_lmcot", "0.5",
            "--lambda_mmd", "0.1",
            "--lambda_orth", "0.1",
            "--monitor_metric", "macro_f1"
        ]
    },
    "A1_Lightweight": {
        "name": "Lightweight DFU-SIAM v3 (EfficientNetV2-B0 + MobileViT-S ~12.5M)",
        "flags": [
            "--arch", "lightweight",
            "--enable_wagner",
            "--use_hard_lmcot",
            "--loss_pathology", "cbfocal",
            "--focal_beta", "0.999",
            "--focal_gamma", "2.0",
            "--lambda_path", "1.0",
            "--lambda_coral", "1.0",
            "--lambda_lmcot", "0.5",
            "--lambda_mmd", "0.1",
            "--lambda_orth", "0.1",
            "--monitor_metric", "macro_f1"
        ]
    },
    "A1_Lightweight_Frozen": {
        "name": "Lightweight DFU-SIAM v3 with Frozen Backbones (1.75M Trainable)",
        "flags": [
            "--arch", "lightweight",
            "--freeze_backbone",
            "--enable_wagner",
            "--use_hard_lmcot",
            "--loss_pathology", "cbfocal",
            "--focal_beta", "0.999",
            "--focal_gamma", "2.0",
            "--lambda_path", "1.0",
            "--lambda_coral", "1.0",
            "--lambda_lmcot", "0.5",
            "--lambda_mmd", "0.1",
            "--lambda_orth", "0.1",
            "--monitor_metric", "macro_f1"
        ]
    },
    "A1_Lightweight_Partial": {
        "name": "Lightweight DFU-SIAM v3 with Partial Fine-Tuning (8.55M Trainable)",
        "flags": [
            "--arch", "lightweight",
            "--partial_finetuning",
            "--enable_wagner",
            "--use_hard_lmcot",
            "--loss_pathology", "cbfocal",
            "--focal_beta", "0.999",
            "--focal_gamma", "2.0",
            "--lambda_path", "1.0",
            "--lambda_coral", "1.0",
            "--lambda_lmcot", "0.5",
            "--lambda_mmd", "0.1",
            "--lambda_orth", "0.1",
            "--monitor_metric", "macro_f1"
        ]
    },
    "A1_SwinTiny_Optimized": {
        "name": "Swin-Tiny Optimized DFU-SIAM v3 (EfficientNetV2-B0 + Swin-Tiny ~35.2M)",
        "flags": [
            "--arch", "swin_tiny",
            "--enable_wagner",
            "--use_hard_lmcot",
            "--loss_pathology", "cbfocal",
            "--focal_beta", "0.999",
            "--focal_gamma", "2.0",
            "--lambda_path", "1.0",
            "--lambda_coral", "1.0",
            "--lambda_lmcot", "0.5",
            "--lambda_mmd", "0.1",
            "--lambda_orth", "0.1",
            "--monitor_metric", "macro_f1"
        ]
    },
    "A1_SwinTiny_Partial": {
        "name": "Swin-Tiny DFU-SIAM v3 Partial Fine-Tuning (21.6M Trainable)",
        "flags": [
            "--arch", "swin_tiny",
            "--partial_finetuning",
            "--enable_wagner",
            "--use_hard_lmcot",
            "--loss_pathology", "cbfocal",
            "--focal_beta", "0.999",
            "--focal_gamma", "2.0",
            "--lambda_path", "1.0",
            "--lambda_coral", "1.0",
            "--lambda_lmcot", "0.5",
            "--lambda_mmd", "0.1",
            "--lambda_orth", "0.1",
            "--monitor_metric", "macro_f1"
        ]
    },
    "A1_SwinTiny_Frozen": {
        "name": "Swin-Tiny DFU-SIAM v3 Frozen Feature Extraction (1.78M Trainable)",
        "flags": [
            "--arch", "swin_tiny",
            "--freeze_backbone",
            "--enable_wagner",
            "--use_hard_lmcot",
            "--loss_pathology", "cbfocal",
            "--focal_beta", "0.999",
            "--focal_gamma", "2.0",
            "--lambda_path", "1.0",
            "--lambda_coral", "1.0",
            "--lambda_lmcot", "0.5",
            "--lambda_mmd", "0.1",
            "--lambda_orth", "0.1",
            "--monitor_metric", "macro_f1"
        ]
    }
}


def run_experiment(exp_id, seed, epochs=40, batch_size=8, grad_clip=1.0, results_dir="experiments_runs_20261002"):
    exp_cfg = EXPERIMENT_CONFIGS[exp_id]
    output_dir = os.path.join(results_dir, f"{exp_id}_seed{seed}")
    os.makedirs(output_dir, exist_ok=True)
    best_ckpt = os.path.join(output_dir, "dfu_siam_v3_best.pth")
    eval_json = os.path.join(output_dir, "eval_val_metrics.json")
    eval_test_json = os.path.join(output_dir, "eval_test_metrics.json")
    log_file = os.path.join(output_dir, "train.log")

    print(f"\n=======================================================")
    print(f" Starting Experiment: {exp_cfg['name']} (Seed {seed})")
    print(f" Output Dir: {output_dir}")
    print(f"=======================================================\n")

    # Check if experiment has already completed successfully
    if os.path.exists(best_ckpt) and os.path.exists(eval_json):
        print(f"[Skip] Experiment {exp_id} seed {seed} already completed. Best checkpoint & evaluation exist.")
        return True

    train_cmd = [
        PYTHON_EXE, "train_dfu_siam.py",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--grad_clip", str(grad_clip),
        "--output_dir", output_dir,
        "--seed", str(seed)
    ] + exp_cfg["flags"]

    # Auto-resume if an interrupted checkpoint exists in this experiment directory
    latest_ckpt = os.path.join(output_dir, "latest_checkpoint.pth")
    if os.path.isfile(latest_ckpt):
        print(f"[Auto-Resume] Found existing checkpoint {latest_ckpt}. Resuming from interruption point.")
        train_cmd.append("--resume")

    print(f"Executing: {' '.join(train_cmd)}")
    with open(log_file, "w") as log_f:
        p = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        p.wait()

    if p.returncode != 0:
        print(f"[Error] Experiment {exp_id} seed {seed} failed with returncode {p.returncode}")
        return False

    if not os.path.exists(best_ckpt):
        print(f"[Error] Best checkpoint not found at {best_ckpt}")
        return False

    # Standalone validation evaluation
    val_csv = os.path.join(output_dir, "temp_val_split.csv")
    eval_cmd = [
        PYTHON_EXE, "evaluate_checkpoint.py",
        "--checkpoint", best_ckpt,
        "--val_csv", val_csv,
        "--output_json", eval_json
    ]
    subprocess.run(eval_cmd, check=True)
    print(f"Validation evaluation saved to {eval_json}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DFU-SIAM v3 Full Experiment Suite")
    parser.add_argument("--experiments", nargs="+", default=["A0_Wag", "A1_Full", "A0_Path", "A2_NoLMCoT", "A3_NoUGMMD", "A4_NoOrth", "A6_StandardCE", "A7_NoHardMining"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--results_dir", type=str, default="experiments_runs_20261002")

    args = parser.parse_args()

    for exp_id in args.experiments:
        if exp_id not in EXPERIMENT_CONFIGS:
            print(f"Unknown experiment ID: {exp_id}. Valid choices: {list(EXPERIMENT_CONFIGS.keys())}")
            continue
        for seed in args.seeds:
            success = run_experiment(exp_id, seed, epochs=args.epochs, batch_size=args.batch_size, grad_clip=args.grad_clip, results_dir=args.results_dir)
            if not success:
                print(f"Aborting sequence due to error in {exp_id} seed {seed}")
                break
