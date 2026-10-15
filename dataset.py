import os
import random
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class RandomAutocontrastPIL:
    """Randomly apply autocontrast (min-max contrast stretching) with probability p."""
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return TF.autocontrast(img)
        return img

# Retain alias for backward compatibility
RandomEqualizePIL = RandomAutocontrastPIL


def get_dfu_transforms(is_train=True, img_size=(224, 224)):
    """
    Enhanced Data Augmentations:
    - Resize to 224x224
    - RandomAffine (degrees=15, translate=10%, scale=0.9-1.1, shear=10)
    - ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    - RandomHorizontalFlip (p=0.5)
    - RandomVerticalFlip (p=0.5)
    - RandomErasing (Cutout p=0.25)
    - Normalization
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        transform = T.Compose([
            T.Resize(img_size),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            RandomEqualizePIL(p=0.3),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2), value=0)
        ])
    else:
        transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    return transform


def get_tta_transforms(img_size=(224, 224)):
    """
    Test-Time Augmentation (TTA) transforms list.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transforms_list = [
        # Original
        T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize(mean=mean, std=std)]),
        # Horizontal Flip
        T.Compose([T.Resize(img_size), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize(mean=mean, std=std)]),
        # Vertical Flip
        T.Compose([T.Resize(img_size), T.RandomVerticalFlip(p=1.0), T.ToTensor(), T.Normalize(mean=mean, std=std)]),
        # Color Jitter
        T.Compose([T.Resize(img_size), T.ColorJitter(brightness=0.15, contrast=0.15), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    ]
    return transforms_list


def get_dfu_transforms_v2(level='baseline', img_size=(224, 224)):
    """
    Configurable augmentation pipeline.
    Levels: 'baseline' (preserves original), 'mild', 'strong', 'clinical'
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if level == 'baseline':
        return get_dfu_transforms(is_train=True, img_size=img_size)
    elif level == 'mild':
        return T.Compose([
            T.Resize(img_size),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    elif level == 'strong':
        return T.Compose([
            T.Resize(img_size),
            T.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=15),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.GaussianBlur(kernel_size=3),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    elif level == 'clinical':
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            T.RandomAffine(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2), # NO hue change
            T.RandomHorizontalFlip(p=0.5), # NO vertical flip
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        raise ValueError(f"Unknown augmentation level: {level}")


class MixUpCutMixCollator:
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, mixup_prob=0.5, cutmix_prob=0.0, num_classes=4):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.num_classes = num_classes

    def __call__(self, batch):
        from torch.utils.data.dataloader import default_collate
        batch = default_collate(batch)
        if len(batch) == 3:
            imgs, labels, names = batch
        else:
            imgs, labels = batch[0], batch[1]
            names = None

        batch_size = imgs.size(0)
        
        # Convert labels to one-hot if they are integers
        if labels.dim() == 1:
            labels = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()

        r = random.random()
        if r < self.mixup_prob:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1
            index = torch.randperm(batch_size)
            
            mixed_imgs = lam * imgs + (1 - lam) * imgs[index, :]
            soft_labels = lam * labels + (1 - lam) * labels[index, :]
            
        elif r < self.mixup_prob + self.cutmix_prob:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if self.cutmix_alpha > 0 else 1
            index = torch.randperm(batch_size)
            
            W, H = imgs.size(2), imgs.size(3)
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            
            mixed_imgs = imgs.clone()
            mixed_imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[index, :, bbx1:bbx2, bby1:bby2]
            
            lam_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            soft_labels = lam_adjusted * labels + (1 - lam_adjusted) * labels[index, :]
            
        else:
            mixed_imgs = imgs
            soft_labels = labels
            
        if names is not None:
            return mixed_imgs, soft_labels, names
        return mixed_imgs, soft_labels


def get_class_balanced_sampler(labels, mode='inverse_freq'):
    from torch.utils.data import WeightedRandomSampler
    
    if isinstance(labels, list):
        labels = np.array(labels)
        
    class_counts = np.bincount(labels)
    
    if mode == 'inverse_freq':
        weights = 1.0 / np.where(class_counts == 0, 1e-6, class_counts)
    elif mode == 'sqrt_inverse':
        weights = 1.0 / np.sqrt(np.where(class_counts == 0, 1e-6, class_counts))
    elif mode == 'effective_num':
        beta = 0.999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.where(effective_num == 0, 1e-6, effective_num)
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")
        
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


class DFUSingleDataset(Dataset):
    """
    Dataset for single image loading (used for validation feature extraction and testing).
    """
    def __init__(self, img_dir, csv_file=None, transform=None, is_test=False):
        self.img_dir = img_dir
        self.transform = transform if transform is not None else get_dfu_transforms(is_train=False)
        self.is_test = is_test

        if not self.is_test and csv_file is not None and os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            class_cols = ['none', 'infection', 'ischaemia', 'both']
            if all(c in df.columns for c in class_cols):
                # Filter out unlabelled/NaN rows
                df = df.dropna(subset=class_cols, how='all').reset_index(drop=True)
                self.img_names = df['image'].values
                self.labels = np.argmax(df[class_cols].values, axis=1).astype(np.int64)
            elif 'class' in df.columns:
                df = df.dropna(subset=['class']).reset_index(drop=True)
                self.img_names = df['image'].values
                self.labels = df['class'].values.astype(np.int64)
            else:
                self.img_names = df['image'].values
                self.labels = None
        else:
            self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            self.labels = None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx], img_name
        return image, img_name


class DFUSiamesePairDataset(Dataset):
    """
    Class-Balanced Equal Probability Pair Generator for Siamese Training.
    Gives equal 25% sampling probability to each class anchor to heavily oversample minority classes (ischaemia & both).
    """
    def __init__(self, img_dir, csv_file, transform=None, num_pairs=12000):
        self.img_dir = img_dir
        self.transform = transform if transform is not None else get_dfu_transforms(is_train=True)
        self.num_pairs = num_pairs

        df = pd.read_csv(csv_file)
        class_cols = ['none', 'infection', 'ischaemia', 'both']
        if all(c in df.columns for c in class_cols):
            df = df.dropna(subset=class_cols, how='all').reset_index(drop=True)
            self.img_names = df['image'].values
            self.labels = np.argmax(df[class_cols].values, axis=1).astype(np.int64)
        elif 'class' in df.columns:
            df = df.dropna(subset=['class']).reset_index(drop=True)
            self.img_names = df['image'].values
            self.labels = df['class'].values.astype(np.int64)
        else:
            raise ValueError(f"CSV {csv_file} must contain either {class_cols} or 'class' column")

        # Index images by class label
        self.label_to_indices = {c: np.where(self.labels == c)[0] for c in range(4)}
        self.pairs, self.pair_labels, self.class_targets = self._generate_pairs()

    def _generate_pairs(self):
        pairs = []
        pair_labels = []
        class_targets = []
        classes = list(self.label_to_indices.keys())

        num_pos = self.num_pairs // 2
        num_neg = self.num_pairs - num_pos

        # Positive pairs (same class) - Equal class representation (25% per class)
        for _ in range(num_pos):
            c = random.choice(classes)
            idx_list = self.label_to_indices[c]
            if len(idx_list) > 1:
                idx1, idx2 = np.random.choice(idx_list, size=2, replace=False)
            else:
                idx1 = idx2 = idx_list[0]
            pairs.append((idx1, idx2))
            pair_labels.append(1)
            class_targets.append((c, c))

        # Negative pairs (different class) - Equal class representation for anchor
        for _ in range(num_neg):
            c1 = random.choice(classes)
            c2 = random.choice([c for c in classes if c != c1])
            idx1 = np.random.choice(self.label_to_indices[c1])
            idx2 = np.random.choice(self.label_to_indices[c2])
            pairs.append((idx1, idx2))
            pair_labels.append(0)
            class_targets.append((c1, c2))

        return pairs, np.array(pair_labels, dtype=np.int64), class_targets

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        target = self.pair_labels[idx]
        c1, c2 = self.class_targets[idx]

        img1_path = os.path.join(self.img_dir, self.img_names[idx1])
        img2_path = os.path.join(self.img_dir, self.img_names[idx2])

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(target, dtype=torch.float32), torch.tensor(c1, dtype=torch.long), torch.tensor(c2, dtype=torch.long)


class BYOLDualTransform:
    """
    Generates two independently augmented stochastic views (v1, v2) for BYOL self-supervised learning.
    Each view undergoes independent affine, color jitter, autocontrast, and cutout transforms.
    """
    def __init__(self, img_size=(224, 224)):
        self.transform1 = get_dfu_transforms(is_train=True, img_size=img_size)
        self.transform2 = get_dfu_transforms(is_train=True, img_size=img_size)

    def __call__(self, img):
        v1 = self.transform1(img)
        v2 = self.transform2(img)
        return v1, v2


class WagnerSingleDataset(Dataset):
    """
    Dataset for Wagner-grade depth dataset (Grades 0-4).
    Supports single view (standard evaluation/training) or dual stochastic views (BYOL).
    Supports deduplication exclusion manifests to prevent evaluation split leakage.
    """
    def __init__(self, data_dir, transform=None, is_byol=False, exclude_manifest="wagner_dedup_manifest.json"):
        self.data_dir = data_dir
        self.is_byol = is_byol
        if transform is not None:
            self.transform = transform
        elif self.is_byol:
            self.transform = BYOLDualTransform(img_size=(224, 224))
        else:
            self.transform = get_dfu_transforms(is_train=False)

        # Load exclusion list if manifest exists and is provided
        self.excluded_paths = set()
        if exclude_manifest and os.path.exists(exclude_manifest):
            try:
                import json
                with open(exclude_manifest, 'r') as f:
                    manifest = json.load(f)
                for item in manifest.get('exact_duplicates', []) + manifest.get('augmented_variants', []):
                    if 'train_path' in item:
                        self.excluded_paths.add(os.path.normpath(item['train_path']).lower())
            except Exception as e:
                print(f"[Warning] Failed to load exclude_manifest {exclude_manifest}: {e}")

        self.img_paths = []
        self.labels = []
        self.img_names = []
        self.class_names = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']

        for label, grade_str in enumerate(self.class_names):
            grade_dir = os.path.join(data_dir, grade_str)
            if os.path.exists(grade_dir):
                for fname in os.listdir(grade_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(grade_dir, fname)
                        if self.excluded_paths and os.path.normpath(full_path).lower() in self.excluded_paths:
                            continue
                        self.img_paths.append(full_path)
                        self.labels.append(label)
                        self.img_names.append(fname)

        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            transformed = self.transform(image)
        else:
            transformed = image

        if self.is_byol and isinstance(transformed, tuple):
            v1, v2 = transformed
            return (v1, v2), self.labels[idx], self.img_names[idx]

        return transformed, self.labels[idx], self.img_names[idx]


class InterleavedDataLoader:
    """
    Yields interleaved or paired batches from DFUC2021 DataLoader and Wagner DataLoader.
    Ensures images from the two datasets are never mixed image-to-image.
    - Mode 'paired_step': yields (dfu_batch, wagner_batch) per iteration step.
    - Mode 'alternating': yields (batch, dataset_tag) where dataset_tag is 'dfu' or 'wagner'.
    Epoch length is controlled by loader_dfu.
    """
    def __init__(self, loader_dfu, loader_wagner, mode='paired_step'):
        self.loader_dfu = loader_dfu
        self.loader_wagner = loader_wagner
        self.mode = mode

    def __len__(self):
        if self.mode == 'paired_step':
            return len(self.loader_dfu)
        else:
            return len(self.loader_dfu) + len(self.loader_wagner)

    def __iter__(self):
        iter_dfu = iter(self.loader_dfu)
        iter_wagner = iter(self.loader_wagner)

        if self.mode == 'paired_step':
            max_batches = len(self)
            for _ in range(max_batches):
                try:
                    dfu_batch = next(iter_dfu)
                except StopIteration:
                    iter_dfu = iter(self.loader_dfu)
                    dfu_batch = next(iter_dfu)

                try:
                    wagner_batch = next(iter_wagner)
                except StopIteration:
                    iter_wagner = iter(self.loader_wagner)
                    wagner_batch = next(iter_wagner)

                yield dfu_batch, wagner_batch

        else:
            active_dfu, active_wagner = True, True
            while active_dfu or active_wagner:
                if active_dfu:
                    try:
                        dfu_batch = next(iter_dfu)
                        yield dfu_batch, 'dfu'
                    except StopIteration:
                        active_dfu = False
                if active_wagner:
                    try:
                        wagner_batch = next(iter_wagner)
                        yield wagner_batch, 'wagner'
                    except StopIteration:
                        active_wagner = False


if __name__ == "__main__":
    train_dir = r"DFUdataset\DFUC2021_train\images"
    csv_file = r"DFUdataset\DFUC2021_train\train.csv"
    wagner_train_dir = r"WagnerDataset\train"
    
    print("=== Phase 1 Data Layer Verification ===")
    dfu_class_names = ['none', 'infection', 'ischaemia', 'both']
    wagner_class_names = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']

    if os.path.exists(csv_file):
        ds_dfu = DFUSiamesePairDataset(train_dir, csv_file, num_pairs=100)
        loader_dfu = DataLoader(ds_dfu, batch_size=4, shuffle=True)
        img1, img2, target, c1, c2 = ds_dfu[0]
        print(f"[DFU Loader] Pair sample - Img1 shape: {img1.shape}, Img2 shape: {img2.shape}, Target: {target}")
        print(f"            Decoded DFU labels: ({dfu_class_names[c1]}, {dfu_class_names[c2]})")
    
    if os.path.exists(wagner_train_dir):
        ds_wagner = WagnerSingleDataset(wagner_train_dir, transform=get_dfu_transforms(is_train=True))
        loader_wagner = DataLoader(ds_wagner, batch_size=4, shuffle=True)
        w_img, w_lbl, w_name = ds_wagner[0]
        print(f"[Wagner Loader] Single sample - Image shape: {w_img.shape}, Label: {w_lbl} ({wagner_class_names[w_lbl]}), File: {w_name}")
        print(f"               Total Wagner train samples loaded: {len(ds_wagner)}")

    if os.path.exists(csv_file) and os.path.exists(wagner_train_dir):
        interleaved_loader = InterleavedDataLoader(loader_dfu, loader_wagner, mode='paired_step')
        print(f"\n[Interleaved Batching Test] Total paired batches per epoch: {len(interleaved_loader)}")
        for step, (dfu_batch, wagner_batch) in enumerate(interleaved_loader):
            b_img1, b_img2, b_target, b_c1, b_c2 = dfu_batch
            w_imgs, w_lbls, w_names = wagner_batch
            print(f"\n--- Batch Step {step+1} Sample Decoded Labels ---")
            print(f"DFUC Batch shape: {b_img1.shape}, Anchors DFU labels: {[dfu_class_names[c.item()] for c in b_c1]}")
            print(f"Wagner Batch shape: {w_imgs.shape}, Wagner labels: {[wagner_class_names[l.item()] for l in w_lbls]}")
            if step >= 2:
                break
        print("\nPhase 1 verification complete: Both loaders run and produce correctly shaped batches!")

