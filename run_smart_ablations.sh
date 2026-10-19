#!/bin/bash
# DFU-SIAM v3 Smart Multi-Ablation Runner for Cloud / Cluster (Lightning AI H100)
# Automatically skips finished ablations and resumes paused runs.

EPOCHS=25
BATCH_SIZE=32
WORKERS=8
LR_BACKBONE=2e-5
LR_HEAD=4e-4

# Clean Split Manifest (0 data leakage)
CLEAN_MANIFEST="research_paper/clean_leak_free_split_manifest.json"

run_ablation() {
    NAME=$1
    DIR=$2
    shift 2
    EXTRA_ARGS="$@"

    echo "=========================================================="
    echo "Checking: $NAME in $DIR"
    echo "=========================================================="

    if [ -f "$DIR/best_checkpoint.pth" ] && [ -f "$DIR/learning_curves.json" ]; then
        echo "[SKIP] $NAME is already finished! Saving compute."
        return 0
    fi

    RESUME_FLAG=""
    if [ -f "$DIR/latest_checkpoint.pth" ] || [ -f "$DIR/epoch_checkpoint.pth" ]; then
        echo "[RESUME] Found existing checkpoint for $NAME. Resuming..."
        RESUME_FLAG="--resume"
    else
        echo "[START] Starting fresh run for $NAME..."
    fi

    python train_dfu_siam.py \
      --arch full \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --lr_backbone $LR_BACKBONE \
      --lr_head $LR_HEAD \
      --warmup_epochs 3 \
      --augmentation_level clinical \
      --label_smoothing 0.05 \
      --num_pairs_per_epoch 6000 \
      --num_workers $WORKERS \
      --grad_clip 1.0 \
      --output_dir "$DIR" \
      --clean_manifest "$CLEAN_MANIFEST" \
      --seed 42 \
      $RESUME_FLAG \
      $EXTRA_ARGS
}

# 1. Full Model (Reference Baseline)
run_ablation "A1_Full" "checkpoints_clean_ablation_A1_full" \
  --enable_wagner --use_hard_lmcot --loss_pathology cbfocal --focal_beta 0.999 --focal_gamma 2.0 \
  --lambda_path 1.0 --lambda_coral 2.5 --lambda_lmcot 0.5 --lambda_mmd 0.1 --lambda_orth 0.1 --monitor_metric joint

# 2. Pathology Only (A0_Path)
run_ablation "A0_Path" "checkpoints_clean_ablation_A0_path" \
  --loss_pathology cbfocal --focal_beta 0.999 --focal_gamma 2.0 \
  --lambda_path 1.0 --lambda_coral 0.0 --lambda_lmcot 0.0 --lambda_mmd 0.0 --lambda_orth 0.0 --monitor_metric macro_f1

# 3. Wagner Only (A0_Wag)
run_ablation "A0_Wag" "checkpoints_clean_ablation_A0_wag" \
  --enable_wagner --lambda_path 0.0 --lambda_coral 2.5 --lambda_lmcot 0.0 --lambda_mmd 0.0 --lambda_orth 0.0 --monitor_metric qwk

# 4. Without LMCoT Metric Loss (A2_NoLMCoT)
run_ablation "A2_NoLMCoT" "checkpoints_clean_ablation_A2_no_lmcot" \
  --enable_wagner --loss_pathology cbfocal --focal_beta 0.999 --focal_gamma 2.0 \
  --lambda_path 1.0 --lambda_coral 2.5 --lambda_lmcot 0.0 --lambda_mmd 0.1 --lambda_orth 0.1 --monitor_metric joint

# 5. Without MMD Alignment (A3_NoUGMMD)
run_ablation "A3_NoUGMMD" "checkpoints_clean_ablation_A3_no_mmd" \
  --enable_wagner --use_hard_lmcot --loss_pathology cbfocal --focal_beta 0.999 --focal_gamma 2.0 \
  --lambda_path 1.0 --lambda_coral 2.5 --lambda_lmcot 0.5 --lambda_mmd 0.0 --lambda_orth 0.1 --monitor_metric joint

# 6. Without Orthogonality Disentanglement (A4_NoOrth)
run_ablation "A4_NoOrth" "checkpoints_clean_ablation_A4_no_orth" \
  --enable_wagner --use_hard_lmcot --loss_pathology cbfocal --focal_beta 0.999 --focal_gamma 2.0 \
  --lambda_path 1.0 --lambda_coral 2.5 --lambda_lmcot 0.5 --lambda_mmd 0.1 --lambda_orth 0.0 --monitor_metric joint

# 7. Standard Cross-Entropy (A6_StandardCE)
run_ablation "A6_StandardCE" "checkpoints_clean_ablation_A6_standard_ce" \
  --enable_wagner --use_hard_lmcot --loss_pathology ce \
  --lambda_path 1.0 --lambda_coral 2.5 --lambda_lmcot 0.5 --lambda_mmd 0.1 --lambda_orth 0.1 --label_smoothing 0.0 --monitor_metric joint

# 8. Random Pairs without Hard-Mining (A7_NoHardMining)
run_ablation "A7_NoHardMining" "checkpoints_clean_ablation_A7_no_hard_mining" \
  --enable_wagner --loss_pathology cbfocal --focal_beta 0.999 --focal_gamma 2.0 \
  --lambda_path 1.0 --lambda_coral 2.5 --lambda_lmcot 0.5 --lambda_mmd 0.1 --lambda_orth 0.1 --monitor_metric joint

echo "=========================================================="
echo ">>> ALL CLEAN ABLATIONS FINISHED SUCCESSFULLY! <<<"
echo "=========================================================="
