#!/bin/bash
# DFU-SIAM v3 Kaggle Dual-T4 Ablation Study Launcher
# Runs parallel ablations on GPU 0 and GPU 1 simultaneously.

echo "=========================================================="
echo "    DFU-SIAM v3 KAGGLE DUAL-T4 ABLATION RUNNER            "
echo "=========================================================="

# Check GPU availability
nvidia-smi

# Run Python parallel orchestrator
python run_kaggle_ablations_2x_t4.py --epochs 25 --batch_size 16 --workers 4

echo "=========================================================="
echo "    ALL KAGGLE ABLATIONS COMPLETED!                       "
echo "=========================================================="
