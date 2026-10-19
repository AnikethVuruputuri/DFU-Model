# Kaggle Dual-T4 Clean-Split Ablation Campaign Guide

This guide details how to execute the 8 matched clean-split ablation experiments on Kaggle using a Dual-T4 GPU environment (or single GPU).

---

## 1. Prerequisites on Kaggle

- **Accelerator**: GPU T4 x2 (Dual NVIDIA Tesla T4 16GB)
- **Internet**: Enabled (for cloning GitHub repository / installing packages)
- **Datasets**:
  - DFUdataset (DFUC2021)
  - WagnerDataset

---

## 2. Setup Commands (In Kaggle Notebook)

```python
# Cell 1: Clone or pull the latest repository
import os
if not os.path.exists("WagnerDS"):
    !git clone https://github.com/YourUsername/WagnerDS.git
%cd WagnerDS
!git pull origin main
```

```python
# Cell 2: Verify environment and clean manifest
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count:   {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Check clean split manifest presence
import os
manifest = "research_paper/clean_leak_free_split_manifest.json"
assert os.path.exists(manifest), f"Manifest missing at {manifest}!"
print(f"✅ Clean Split Manifest Verified ({os.path.getsize(manifest):,} bytes)")
```

---

## 3. Verify Execution with Dry-Run

Run the pre-flight verification to inspect the paths, configurations, and auto-scaled learning rates:

```bash
!python run_kaggle_ablations_2x_t4.py --dry_run
```

---

## 4. Launch Clean Ablation Campaign

### Full 8-Ablation Parallel Run (Dual T4)
Distributes all 8 experiments across GPU 0 and GPU 1 automatically:
```bash
!python run_kaggle_ablations_2x_t4.py --epochs 25 --batch_size 16
```
- **Total Runtime**: ~4.5 to 5.0 hours on Dual T4 (2x speedup).
- **Auto-resumption**: If the notebook times out or is restarted, re-running the exact same command resumes from the latest completed epoch automatically.
- **Disk Safety & 20GB Limit**: `--save_only_best` is active by default. Intermediate checkpoints (`latest_checkpoint.pth`, `epoch_checkpoint.pth`) are purged upon run completion, retaining strictly `best_checkpoint.pth` and `learning_curves.json` (~720MB per ablation $\times$ 8 = ~5.7GB total). This leaves >14GB of free headroom on Kaggle's 20GB disk quota!
- **Artifact Isolation**: Clean checkpoints are stored in dedicated directories (`checkpoints_clean_ablation_*`), preserving all previous preliminary results without collision.

### Running a Specific Ablation
If you wish to run only one or two ablations at a time:
```bash
!python run_kaggle_ablations_2x_t4.py --epochs 25 --batch_size 16 --only A1_Full
```

---

## 5. Summary Table & Results Artifacts

Upon completion of the ablations, the runner outputs a publication-ready Markdown table and writes:
- `research_paper/clean_kaggle_ablation_results.csv`
- `research_paper/clean_kaggle_ablation_results.md`

Commit or download these files to update the manuscript table with the final verified numbers!
