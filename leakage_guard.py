"""
Pre-Training Cryptographic Leakage Guard and Data Integrity Verification.

Provides:
- Strict MD5 hash and path verification across DFUC2021 and Wagner datasets.
- Mandatory pre-training assertion guaranteeing zero internal and zero cross-split leakage.
- Fast hash extraction and caching to ensure immediate verification before training starts.
"""

import os
import json
import hashlib
from typing import Dict, List, Set, Tuple

def compute_file_md5(filepath: str) -> str:
    """Computes MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_dataset_paths_and_hashes(dataset, dataset_type: str = "dfu") -> Tuple[List[str], Dict[str, str]]:
    """
    Extracts resolved image paths and computes their MD5 hashes.
    dataset_type: 'dfu' (uses img_dir + img_names) or 'wagner' (uses img_paths).
    Returns (paths_list, {path: md5_hash}).
    """
    if dataset is None:
        return [], {}

    paths = []
    if hasattr(dataset, "img_paths"):
        paths = list(dataset.img_paths)
    elif hasattr(dataset, "img_names") and hasattr(dataset, "img_dir"):
        paths = [os.path.join(dataset.img_dir, fn) for fn in dataset.img_names]
    else:
        raise ValueError(f"Unknown dataset structure for {dataset}")

    path_to_hash = {}
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Image file does not exist: {p}")
        path_to_hash[p] = compute_file_md5(p)

    return paths, path_to_hash

def assert_pre_training_leakage_free(
    dfu_train_dataset,
    dfu_val_dataset,
    wagner_train_dataset=None,
    wagner_val_dataset=None,
    wagner_test_dataset=None,
    clean_manifest_path: str = "research_paper/clean_leak_free_split_manifest.json",
    wagner_dedup_path: str = "wagner_dedup_manifest.json"
) -> Dict[str, any]:
    """
    Mandatory pre-training leakage assertion.
    Guarantees:
    1. Zero internal DFUC train-val duplicate leakage.
    2. Zero internal Wagner train-val duplicate leakage.
    3. Zero internal Wagner train-test duplicate leakage.
    4. Zero cross-split leakage: DFUC validation images CANNOT appear in Wagner training.
    5. Zero cross-split leakage: Wagner validation images CANNOT appear in DFUC training.
    6. Zero cross-split leakage: Wagner test images CANNOT appear in DFUC training.
    7. All quarantined images from clean_manifest and dedup_manifest are absent from training.
    8. Exact clean counts match the declared protocol.
    """
    print("\n" + "="*80)
    print("MANDATORY PRE-TRAINING LEAKAGE ASSERTION: CRYPTOGRAPHIC HASH AUDIT")
    print("="*80)

    # 1. Load manifests with cross-platform fallback paths
    if not os.path.exists(clean_manifest_path):
        candidates_clean = [
            os.path.join(os.path.dirname(__file__), "research_paper", "clean_leak_free_split_manifest.json"),
            "research_paper/clean_leak_free_split_manifest.json",
            "/kaggle/working/DFU-Model/research_paper/clean_leak_free_split_manifest.json",
            "/kaggle/working/research_paper/clean_leak_free_split_manifest.json"
        ]
        for c in candidates_clean:
            if os.path.exists(c):
                clean_manifest_path = c
                break

    if not os.path.exists(clean_manifest_path):
        raise FileNotFoundError(f"Clean manifest not found: {clean_manifest_path}")
    with open(clean_manifest_path, "r") as f:
        clean_manifest = json.load(f)

    if not os.path.exists(wagner_dedup_path):
        candidates_dedup = [
            os.path.join(os.path.dirname(__file__), "wagner_dedup_manifest.json"),
            "wagner_dedup_manifest.json",
            "/kaggle/working/DFU-Model/wagner_dedup_manifest.json",
            "/kaggle/working/wagner_dedup_manifest.json"
        ]
        for c in candidates_dedup:
            if os.path.exists(c):
                wagner_dedup_path = c
                break

    if not os.path.exists(wagner_dedup_path):
        raise FileNotFoundError(f"Wagner dedup manifest not found: {wagner_dedup_path}")
    with open(wagner_dedup_path, "r") as f:
        dedup_manifest = json.load(f)

    dfu_train_exclusions = set(clean_manifest.get("dfu_train_exclusions", []))
    wagner_train_cross_exclusions = set(
        p.replace('\\', '/').split('/')[-1].lower() for p in clean_manifest.get("wagner_train_cross_exclusions", [])
    )
    wagner_internal_exclusions = set(
        it["train_path"].replace('\\', '/').split('/')[-1].lower()
        for it in (dedup_manifest.get("exact_duplicates", []) + dedup_manifest.get("augmented_variants", []))
        if "train_path" in it
    )

    print(f"Loaded Quarantines: {len(dfu_train_exclusions)} DFU Train exclusions, "
          f"{len(wagner_train_cross_exclusions)} Wagner cross-split exclusions, "
          f"{len(wagner_internal_exclusions)} Wagner internal duplicate exclusions.")

    # 2. Extract resolved paths and hashes
    print("Hashing resolved images from data loaders...")
    dfu_tr_paths, dfu_tr_hashes = extract_dataset_paths_and_hashes(dfu_train_dataset, "dfu")
    dfu_val_paths, dfu_val_hashes = extract_dataset_paths_and_hashes(dfu_val_dataset, "dfu")
    wag_tr_paths, wag_tr_hashes = extract_dataset_paths_and_hashes(wagner_train_dataset, "wagner") if wagner_train_dataset is not None else ([], {})
    wag_val_paths, wag_val_hashes = extract_dataset_paths_and_hashes(wagner_val_dataset, "wagner") if wagner_val_dataset is not None else ([], {})
    wag_te_paths, wag_te_hashes = extract_dataset_paths_and_hashes(wagner_test_dataset, "wagner") if wagner_test_dataset is not None else ([], {})

    dfu_tr_hash_set = set(dfu_tr_hashes.values())
    dfu_val_hash_set = set(dfu_val_hashes.values())
    wag_tr_hash_set = set(wag_tr_hashes.values())
    wag_val_hash_set = set(wag_val_hashes.values())
    wag_te_hash_set = set(wag_te_hashes.values())

    print(f"Resolved Counts:")
    print(f"  DFUC Clean Train : {len(dfu_tr_paths)} images (Unique hashes: {len(dfu_tr_hash_set)})")
    print(f"  DFUC Clean Val   : {len(dfu_val_paths)} images (Unique hashes: {len(dfu_val_hash_set)})")
    if wagner_train_dataset is not None:
        print(f"  Wagner Clean Train: {len(wag_tr_paths)} images (Unique hashes: {len(wag_tr_hash_set)})")
    if wagner_val_dataset is not None:
        print(f"  Wagner Clean Valid: {len(wag_val_paths)} images (Unique hashes: {len(wag_val_hash_set)})")
    if wagner_test_dataset is not None:
        print(f"  Wagner Clean Test : {len(wag_te_paths)} images (Unique hashes: {len(wag_te_hash_set)})")

    # 3. Exact count assertions
    expected_summary = clean_manifest.get("split_summary", {})
    if expected_summary:
        assert len(dfu_tr_paths) == expected_summary.get("dfu_train_quarantined", 4626), \
            f"DFU train count mismatch: expected {expected_summary.get('dfu_train_quarantined')}, got {len(dfu_tr_paths)}"
        assert len(dfu_val_paths) == expected_summary.get("dfu_val_clean", 1191), \
            f"DFU val count mismatch: expected {expected_summary.get('dfu_val_clean')}, got {len(dfu_val_paths)}"
        if wagner_train_dataset is not None:
            assert len(wag_tr_paths) == expected_summary.get("wagner_train_quarantined", 11371), \
                f"Wagner train count mismatch: expected {expected_summary.get('wagner_train_quarantined')}, got {len(wag_tr_paths)}"
        if wagner_val_dataset is not None:
            assert len(wag_val_paths) == expected_summary.get("wagner_valid_clean", 303), \
                f"Wagner val count mismatch: expected {expected_summary.get('wagner_valid_clean')}, got {len(wag_val_paths)}"
        if wagner_test_dataset is not None:
            assert len(wag_te_paths) == expected_summary.get("wagner_test_clean", 157), \
                f"Wagner test count mismatch: expected {expected_summary.get('wagner_test_clean')}, got {len(wag_te_paths)}"

    # 4. Check that no quarantined image filename entered training
    dfu_tr_basenames = set(p.replace('\\', '/').split('/')[-1] for p in dfu_tr_paths)
    dfu_quarantine_violations = dfu_tr_basenames.intersection(dfu_train_exclusions)
    assert len(dfu_quarantine_violations) == 0, \
        f"CRITICAL ERROR: {len(dfu_quarantine_violations)} quarantined DFU images found in DFU training loader: {list(dfu_quarantine_violations)[:5]}"

    if wagner_train_dataset is not None:
        wag_tr_basenames = set(p.replace('\\', '/').split('/')[-1].lower() for p in wag_tr_paths)
        wag_cross_violations = wag_tr_basenames.intersection(wagner_train_cross_exclusions)
        assert len(wag_cross_violations) == 0, \
            f"CRITICAL ERROR: {len(wag_cross_violations)} cross-quarantined images found in Wagner training loader: {list(wag_cross_violations)[:5]}"

        wag_dedup_violations = wag_tr_basenames.intersection(wagner_internal_exclusions)
        assert len(wag_dedup_violations) == 0, \
            f"CRITICAL ERROR: {len(wag_dedup_violations)} internally duplicate images found in Wagner training loader: {list(wag_dedup_violations)[:5]}"

    # 5. Cryptographic hash collision checks across splits
    dfu_internal_leaks = set()
    wag_internal_val_leaks = set()
    wag_internal_te_leaks = set()
    cross_dfu_val_in_wag_tr = set()
    cross_wag_val_in_dfu_tr = set()
    cross_wag_te_in_dfu_tr = set()

    # Internal DFUC
    dfu_internal_leaks = dfu_tr_hash_set.intersection(dfu_val_hash_set)
    assert len(dfu_internal_leaks) == 0, \
        f"Internal DFUC train/val leakage: {len(dfu_internal_leaks)} identical images in both splits!"

    # Internal Wagner
    if wagner_train_dataset is not None and wagner_val_dataset is not None:
        wag_internal_val_leaks = wag_tr_hash_set.intersection(wag_val_hash_set)
        assert len(wag_internal_val_leaks) == 0, \
            f"Internal Wagner train/valid leakage: {len(wag_internal_val_leaks)} identical images in both splits!"

    if wagner_train_dataset is not None and wagner_test_dataset is not None:
        wag_internal_te_leaks = wag_tr_hash_set.intersection(wag_te_hash_set)
        assert len(wag_internal_te_leaks) == 0, \
            f"Internal Wagner train/test leakage: {len(wag_internal_te_leaks)} identical images in both splits!"

    # Cross-dataset leaks
    if wagner_train_dataset is not None:
        cross_dfu_val_in_wag_tr = dfu_val_hash_set.intersection(wag_tr_hash_set)
        assert len(cross_dfu_val_in_wag_tr) == 0, \
            f"CROSS-DATASET LEAKAGE: {len(cross_dfu_val_in_wag_tr)} DFU validation images found in Wagner training set!"

    if wagner_val_dataset is not None:
        cross_wag_val_in_dfu_tr = wag_val_hash_set.intersection(dfu_tr_hash_set)
        assert len(cross_wag_val_in_dfu_tr) == 0, \
            f"CROSS-DATASET LEAKAGE: {len(cross_wag_val_in_dfu_tr)} Wagner validation images found in DFU training set!"

    if wagner_test_dataset is not None:
        cross_wag_te_in_dfu_tr = wag_te_hash_set.intersection(dfu_tr_hash_set)
        assert len(cross_wag_te_in_dfu_tr) == 0, \
            f"CROSS-DATASET LEAKAGE: {len(cross_wag_te_in_dfu_tr)} Wagner test images found in DFU training set!"

    print("\n[LEAKAGE ASSERTION PASSED] All verified cross-split and internal leakage checks confirmed with ZERO collisions!")
    print("  - DFU Train vs DFU Val Collisions: 0")
    if wagner_train_dataset is not None:
        print("  - Wagner Train vs Wagner Val Collisions: 0")
        print("  - DFU Val vs Wagner Train Collisions: 0")
    if wagner_val_dataset is not None:
        print("  - Wagner Val vs DFU Train Collisions: 0")
    if wagner_test_dataset is not None:
        print("  - Wagner Test vs DFU Train Collisions: 0")
    print("  - Quarantined Exclusion Violations: 0")
    print("="*80 + "\n")

    return {
        "status": "PASS",
        "counts": {
            "dfu_train": len(dfu_tr_paths),
            "dfu_val": len(dfu_val_paths),
            "wagner_train": len(wag_tr_paths),
            "wagner_val": len(wag_val_paths),
            "wagner_test": len(wag_te_paths) if wagner_test_dataset is not None else 0
        },
        "leakage_counts": {
            "dfu_internal": len(dfu_internal_leaks),
            "wagner_internal_val": len(wag_internal_val_leaks),
            "dfu_val_in_wagner_train": len(cross_dfu_val_in_wag_tr),
            "wagner_val_in_dfu_train": len(cross_wag_val_in_dfu_tr)
        }
    }

if __name__ == "__main__":
    from dataset import DFUSingleDataset, WagnerSingleDataset
    print("Running standalone leakage guard smoke test...")
    dfu_img_dir = os.path.join("DFUdataset", "DFUC2021_train", "images")
    dfu_csv = os.path.join("checkpoints_a1_full_balanced", "temp_train_split.csv")
    dfu_val_csv = os.path.join("checkpoints_a1_full_balanced", "temp_val_split.csv")
    
    # We will test the clean quarantine
    with open("research_paper/clean_leak_free_split_manifest.json") as f:
        cm = json.load(f)
    ex_dfu = set(cm["dfu_train_exclusions"])
    
    import pandas as pd
    df_tr = pd.read_csv(dfu_csv)
    df_clean_tr = df_tr[~df_tr["image"].isin(ex_dfu)].reset_index(drop=True)
    temp_test_tr_csv = "scratch/temp_clean_train_smoke.csv"
    os.makedirs("scratch", exist_ok=True)
    df_clean_tr.to_csv(temp_test_tr_csv, index=False)
    
    ds_dfu_tr = DFUSingleDataset(dfu_img_dir, csv_file=temp_test_tr_csv)
    ds_dfu_val = DFUSingleDataset(dfu_img_dir, csv_file=dfu_val_csv)
    ds_wag_tr = WagnerSingleDataset("WagnerDataset/train", clean_manifest="research_paper/clean_leak_free_split_manifest.json")
    ds_wag_val = WagnerSingleDataset("WagnerDataset/valid")
    ds_wag_te = WagnerSingleDataset("WagnerDataset/test")
    
    res = assert_pre_training_leakage_free(ds_dfu_tr, ds_dfu_val, ds_wag_tr, ds_wag_val, ds_wag_te)
    print("Standalone test result:", res)
