import os
import pandas as pd
import hashlib
from PIL import Image
import numpy as np
import json
from collections import defaultdict
from itertools import combinations

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None

def get_ahash(file_path, hash_size=8):
    try:
        image = Image.open(file_path).convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(image)
        avg = pixels.mean()
        diff = pixels > avg
        return "".join(["1" if b else "0" for b in diff.flatten()])
    except:
        return None

def audit_dfu():
    print("--- Auditing DFUC2021 Dataset ---")
    csv_path = os.path.join("DFUdataset", "DFUC2021_train", "train.csv")
    img_dir = os.path.join("DFUdataset", "DFUC2021_train", "images")
    
    report = {}
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return report
        
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    class_cols = ['none', 'infection', 'ischaemia', 'both']
    
    labeled_rows = df.dropna(subset=class_cols, how='all')
    unlabeled_count = total_rows - len(labeled_rows)
    
    report["total_rows"] = total_rows
    report["unlabeled_rows"] = unlabeled_count
    report["labeled_rows"] = len(labeled_rows)
    
    class_counts = {}
    if len(labeled_rows) > 0:
        for c in class_cols:
            class_counts[c] = int(labeled_rows[c].sum())
    report["class_distribution"] = class_counts
    
    print(f"Total Rows: {total_rows}, Labeled: {len(labeled_rows)}, Unlabeled (NaN): {unlabeled_count}")
    print(f"Class distribution: {class_counts}")
    
    # Check images on disk
    missing_images = []
    duplicate_filenames = df['image'].duplicated().sum()
    report["duplicate_filenames_in_csv"] = int(duplicate_filenames)
    print(f"Duplicate filenames in CSV: {duplicate_filenames}")
    
    md5_dict = defaultdict(list)
    ahash_dict = defaultdict(list)
    
    for img_name in df['image'].dropna().unique():
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            missing_images.append(img_name)
        else:
            md5_val = get_md5(img_path)
            if md5_val:
                md5_dict[md5_val].append(img_name)
            ahash_val = get_ahash(img_path)
            if ahash_val:
                ahash_dict[ahash_val].append(img_name)
                
    report["missing_images_on_disk"] = len(missing_images)
    print(f"Missing images referenced in CSV: {len(missing_images)}")
    
    exact_duplicates = {k: v for k, v in md5_dict.items() if len(v) > 1}
    near_duplicates = {k: v for k, v in ahash_dict.items() if len(v) > 1}
    
    report["exact_duplicates_count"] = sum([len(v) - 1 for v in exact_duplicates.values()])
    report["near_duplicates_count"] = sum([len(v) - 1 for v in near_duplicates.values()])
    
    print(f"Found {len(exact_duplicates)} groups of exact duplicates (MD5).")
    print(f"Found {len(near_duplicates)} groups of near duplicates (Perceptual Hash).")
    
    return report, md5_dict

def audit_wagner():
    print("\n--- Auditing Wagner Dataset ---")
    base_dir = "WagnerDataset"
    splits = ["train", "valid", "test"]
    
    report = {}
    split_md5 = {}
    
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        grade_dist = {}
        split_md5[split] = defaultdict(list)
        
        for grade in ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]:
            grade_dir = os.path.join(split_dir, grade)
            if os.path.exists(grade_dir):
                files = [f for f in os.listdir(grade_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                grade_dist[grade] = len(files)
                for f in files:
                    img_path = os.path.join(grade_dir, f)
                    md5_val = get_md5(img_path)
                    if md5_val:
                        split_md5[split][md5_val].append(img_path)
                        
        report[split] = {"grade_distribution": grade_dist}
        print(f"{split} split distribution: {grade_dist}")
        
    # Cross-split exact duplicates
    leakage_report = []
    all_splits = list(split_md5.keys())
    for s1, s2 in combinations(all_splits, 2):
        common_md5 = set(split_md5[s1].keys()).intersection(set(split_md5[s2].keys()))
        if common_md5:
            print(f"WARNING: Cross-split leakage detected between {s1} and {s2}. {len(common_md5)} images leaked.")
            for h in common_md5:
                leakage_report.append({
                    "md5": h,
                    s1: split_md5[s1][h],
                    s2: split_md5[s2][h]
                })
    report["cross_split_leakage"] = leakage_report
    
    return report, split_md5

def cross_dataset_check(dfu_md5, wagner_md5):
    print("\n--- Cross-Dataset Audit ---")
    wagner_all_md5 = {}
    for split, md5_dict in wagner_md5.items():
        for h, paths in md5_dict.items():
            if h not in wagner_all_md5:
                wagner_all_md5[h] = []
            wagner_all_md5[h].extend(paths)
            
    dfu_set = set(dfu_md5.keys())
    wagner_set = set(wagner_all_md5.keys())
    
    common = dfu_set.intersection(wagner_set)
    print(f"Found {len(common)} exact matches between DFUC2021 and Wagner dataset.")
    print("NOTE: Patient-level separation cannot be guaranteed without patient IDs.")
    
    report = {
        "dfu_wagner_collisions_count": len(common),
        "collisions": [{"dfu": dfu_md5[h], "wagner": wagner_all_md5[h]} for h in common],
        "note": "Patient-level separation cannot be guaranteed without patient IDs."
    }
    return report

def main():
    final_report = {}
    dfu_report, dfu_md5 = audit_dfu()
    final_report["dfu_audit"] = dfu_report
    
    wagner_report, wagner_md5 = audit_wagner()
    final_report["wagner_audit"] = wagner_report
    
    cross_report = cross_dataset_check(dfu_md5, wagner_md5)
    final_report["cross_dataset"] = cross_report
    
    with open("dataset_audit_report.json", "w") as f:
        json.dump(final_report, f, indent=4)
    print("\nReport saved to dataset_audit_report.json")

if __name__ == "__main__":
    main()
