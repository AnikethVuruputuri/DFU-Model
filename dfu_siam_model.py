"""
DFU-SIAM v3: Diabetic Foot Ulcer - Shared Intelligence and Alignment Model (v3 Architecture).

Architectural Highlights:
1. Asymmetric Hybrid Dual-Backbone (EfficientNetV2-S local inductive bias + Swin-Tiny global attention).
2. Symmetrical 1280-D linear projections for both CNN and ViT branches.
3. Task-Disentangled Representation Learning:
   - Fused 2560-D features decomposed into:
     * z_shared (768-D): Cross-domain transferable anatomical/tissue features.
     * z_path_private (512-D): Pathology-specific microbiological/ischaemic features.
     * z_depth_private (512-D): Ulcer-depth-specific tissue breakdown features.
4. Selective Task Branching:
   - Pathology Head receives concat(z_shared, z_path_private) -> 4 logits.
   - Wagner Head receives concat(z_shared, z_depth_private) -> 4 cumulative cutoffs (CORAL).
   - z_depth_private is NEVER passed to pathology; z_path_private is NEVER passed to Wagner.
5. Novel Objectives:
   - Class-Balanced Focal Loss (Cui et al., CVPR 2019) for severe class imbalance.
   - Consistent Rank Logits (CORAL; Cao et al., PRL 2020) for ordinal Wagner depth.
   - Hard-Pair Tangent LMCoT for clinical boundary separation.
   - Uncertainty-Gated MMD (UG-MMD) aligning ONLY z_shared, weighted by detached predictive confidence.
   - Orthogonality Disentanglement Loss enforcing z_shared ⟂ z_private.

Note: Legacy modules (BYOL, AnchorMetric, Sinkhorn) have been completely removed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def get_parameter_count(model):
    """
    Computes comprehensive parameter counts and estimated model memory footprints.
    Returns:
        dict containing total, trainable, non-trainable counts, module breakdowns,
        and FP32/FP16 memory sizes in MB.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    cnn_params = sum(p.numel() for p in model.backbone.cnn.parameters()) if hasattr(model, 'backbone') and hasattr(model.backbone, 'cnn') else 0
    vit_params = sum(p.numel() for p in model.backbone.vit.parameters()) if hasattr(model, 'backbone') and hasattr(model.backbone, 'vit') else 0
    cnn_proj_params = sum(p.numel() for p in model.backbone.cnn_proj.parameters()) if hasattr(model, 'backbone') and hasattr(model.backbone, 'cnn_proj') else 0
    vit_proj_params = sum(p.numel() for p in model.backbone.vit_proj.parameters()) if hasattr(model, 'backbone') and hasattr(model.backbone, 'vit_proj') else 0
    
    disentangler_params = sum(p.numel() for p in model.disentangler.parameters()) if hasattr(model, 'disentangler') else 0
    classifier_params = sum(p.numel() for p in model.classifier.parameters()) if hasattr(model, 'classifier') else 0
    wagner_params = sum(p.numel() for p in model.wagner_head.parameters()) if hasattr(model, 'wagner_head') else 0

    fp32_size_mb = (total * 4) / (1024 ** 2)
    fp16_size_mb = (total * 2) / (1024 ** 2)

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable,
        'cnn': cnn_params,
        'vit': vit_params,
        'cnn_proj': cnn_proj_params,
        'vit_proj': vit_proj_params,
        'disentangler': disentangler_params,
        'classifier': classifier_params,
        'wagner_head': wagner_params,
        'fp32_size_mb': fp32_size_mb,
        'fp16_size_mb': fp16_size_mb,
    }


class DFUSiamV3Backbone(nn.Module):
    """
    Dual-Backbone Feature Extractor with Symmetrical Projections.
    Supports three architectural configurations:
    - 'swin_tiny' (Optimized ~35.2M): EfficientNetV2-B0 (1280-D raw) + Swin-Tiny (768-D raw) -> Proj 256 -> Fused 512-D
    - 'full' (Original Standard ~59.8M): EfficientNetV2-S (1280-D raw) + Swin-Tiny (768-D raw) -> Proj 1280 -> Fused 2560-D
    - 'lightweight' (~12.5M): EfficientNetV2-B0 (1280-D raw) + MobileViT-S (640-D raw) -> Proj 256 -> Fused 512-D
    """
    def __init__(
        self,
        arch="full",
        cnn_name=None,
        vit_name=None,
        proj_dim=None,
        dropout=0.3,
        pretrained=True
    ):
        super(DFUSiamV3Backbone, self).__init__()
        self.arch = arch

        # Determine backbone defaults based on architecture mode
        if arch in ("swin_tiny", "swin_tiny_optimized", "swin_optimized"):
            if cnn_name is None:
                cnn_name = "tf_efficientnetv2_b0"
            if vit_name is None:
                vit_name = "swin_tiny_patch4_window7_224"
            if proj_dim is None:
                proj_dim = 256
        elif arch in ("lightweight", "mobilevit"):
            if cnn_name is None:
                cnn_name = "tf_efficientnetv2_b0"
            if vit_name is None:
                vit_name = "mobilevit_s"
            if proj_dim is None:
                proj_dim = 256
        else:  # "full"
            if cnn_name is None:
                cnn_name = "tf_efficientnetv2_s.in21k_ft_in1k"
            if vit_name is None:
                vit_name = "swin_tiny_patch4_window7_224"
            if proj_dim is None:
                proj_dim = 1280

        self.proj_dim = proj_dim

        # Branch A: CNN Backbone
        try:
            self.cnn = timm.create_model(cnn_name, pretrained=pretrained, num_classes=0)
            self.cnn_dim = self.cnn.num_features
        except Exception as e:
            fallback_cnn = "tf_efficientnetv2_b0" if "b0" in cnn_name else "efficientnet_b0"
            print(f"[Warning] Failed to load {cnn_name}, falling back to {fallback_cnn}: {e}")
            self.cnn = timm.create_model(fallback_cnn, pretrained=pretrained, num_classes=0)
            self.cnn_dim = self.cnn.num_features

        # Branch B: ViT Backbone
        try:
            self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
            self.vit_dim = self.vit.num_features
        except Exception as e:
            fallback_vit = "swin_tiny_patch4_window7_224" if "swin" in vit_name else "vit_tiny_patch16_224"
            print(f"[Warning] Failed to load {vit_name}, falling back to {fallback_vit}: {e}")
            self.vit = timm.create_model(fallback_vit, pretrained=pretrained, num_classes=0)
            self.vit_dim = self.vit.num_features

        # Symmetrical Projections (both branches project to proj_dim)
        self.cnn_proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.cnn_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.GELU()
        )

        self.vit_proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.vit_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.GELU()
        )

        self.fused_dim = self.proj_dim * 2

    def forward(self, x):
        """
        Input: (N, 3, H, W)
        Output: Raw concatenated fused representation (N, fused_dim)
        """
        feat_cnn = self.cnn(x)
        feat_cnn = self.cnn_proj(feat_cnn)

        feat_vit = self.vit(x)
        feat_vit = self.vit_proj(feat_vit)

        fused = torch.cat([feat_cnn, feat_vit], dim=1)
        return fused


class TaskDisentangler(nn.Module):
    """
    Decomposes fused representations into three disentangled subspaces:
    1. z_shared: Cross-domain transferable anatomical features (aligned via UG-MMD).
    2. z_path_private: Pathology-specific microbiological/ischaemic features.
    3. z_depth_private: Ulcer-depth-specific anatomical invasion features.

    All subspaces are L2-normalized onto their respective unit hyperspheres.
    """
    def __init__(self, fused_dim=2560, shared_dim=768, path_dim=512, depth_dim=512, dropout=0.3):
        super(TaskDisentangler, self).__init__()
        self.shared_dim = shared_dim
        self.path_dim = path_dim
        self.depth_dim = depth_dim

        # Dynamically scale intermediate hidden dimensions based on fused_dim
        shared_hidden = min(1024, fused_dim)
        private_hidden = min(768, int(fused_dim * 0.75)) if fused_dim > 512 else 384

        # Shared subspace projector (2-layer MLP with BN and GELU)
        self.shared_proj = nn.Sequential(
            nn.Linear(fused_dim, shared_hidden),
            nn.BatchNorm1d(shared_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(shared_hidden, shared_dim),
            nn.BatchNorm1d(shared_dim)
        )

        # Pathology-private subspace projector
        self.path_proj = nn.Sequential(
            nn.Linear(fused_dim, private_hidden),
            nn.BatchNorm1d(private_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(private_hidden, path_dim),
            nn.BatchNorm1d(path_dim)
        )

        # Depth-private subspace projector
        self.depth_proj = nn.Sequential(
            nn.Linear(fused_dim, private_hidden),
            nn.BatchNorm1d(private_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(private_hidden, depth_dim),
            nn.BatchNorm1d(depth_dim)
        )

    def forward(self, fused):
        """
        Input: (N, fused_dim)
        Output:
            z_shared: (N, shared_dim), L2-normalized
            z_path_private: (N, path_dim), L2-normalized
            z_depth_private: (N, depth_dim), L2-normalized
        """
        raw_shared = self.shared_proj(fused)
        raw_path = self.path_proj(fused)
        raw_depth = self.depth_proj(fused)

        z_shared = F.normalize(raw_shared, p=2, dim=1)
        z_path_private = F.normalize(raw_path, p=2, dim=1)
        z_depth_private = F.normalize(raw_depth, p=2, dim=1)

        return z_shared, z_path_private, z_depth_private


class DFUSiamV3(nn.Module):
    """
    DFU-SIAM v3 Model Architecture.
    - Symmetrical Dual-Backbone:
        * 'swin_tiny' (Optimized ~35.2M): EfficientNetV2-B0 + Swin-Tiny (256-D projections, 512-D fusion).
        * 'full' (Original Standard ~59.8M): EfficientNetV2-S + Swin-Tiny (1280-D projections, 2560-D fusion).
        * 'lightweight' (~12.5M): EfficientNetV2-B0 + MobileViT-S (retained for comparison).
    - Task-Disentangled Latent Space (z_shared, z_path_private, z_depth_private).
    - Selective Task-Specific Feature Construction:
        * Pathology input = concat(z_shared, z_path_private) -> 4 logits
        * Wagner input    = concat(z_shared, z_depth_private) -> 4 cumulative logits (CORAL)
    - Complete isolation: depth-private features never enter pathology head;
      pathology-private features never enter Wagner head.
    """
    def __init__(
        self,
        arch="full",
        backbone=None,
        shared_dim=None,
        path_dim=None,
        depth_dim=None,
        num_classes=4,
        num_wagner_classes=5,
        dropout=0.3,
        **backbone_kwargs
    ):
        super(DFUSiamV3, self).__init__()
        self.model_version = "v3"
        self.arch = arch

        # Resolve subspace dimensions and head capacity based on architecture
        if arch in ("swin_tiny", "swin_tiny_optimized", "swin_optimized", "lightweight", "mobilevit"):
            self.shared_dim = shared_dim if shared_dim is not None else 256
            self.path_dim = path_dim if path_dim is not None else 256
            self.depth_dim = depth_dim if depth_dim is not None else 256
            head_hidden = 256
        else:  # "full"
            self.shared_dim = shared_dim if shared_dim is not None else 768
            self.path_dim = path_dim if path_dim is not None else 512
            self.depth_dim = depth_dim if depth_dim is not None else 512
            head_hidden = 512

        self.num_classes = num_classes
        self.num_wagner_classes = num_wagner_classes

        # Backbone
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = DFUSiamV3Backbone(arch=arch, dropout=dropout, **backbone_kwargs)

        # Disentanglement Module
        self.disentangler = TaskDisentangler(
            fused_dim=self.backbone.fused_dim,
            shared_dim=self.shared_dim,
            path_dim=self.path_dim,
            depth_dim=self.depth_dim,
            dropout=dropout
        )

        # Pathology Classification Head (receives z_shared + z_path_private)
        path_in_dim = self.shared_dim + self.path_dim
        self.classifier = nn.Sequential(
            nn.Linear(path_in_dim, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, num_classes)
        )

        # Wagner Ordinal Head (CORAL: receives z_shared + z_depth_private)
        # Predicts K-1 = 4 cumulative binary cutoffs for Grades 0-4
        depth_in_dim = self.shared_dim + self.depth_dim
        self.wagner_head = nn.Sequential(
            nn.Linear(depth_in_dim, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, num_wagner_classes - 1, bias=False)
        )

    def freeze_backbones(self):
        """Freezes all parameters in both CNN and ViT backbones."""
        for param in self.backbone.cnn.parameters():
            param.requires_grad = False
        for param in self.backbone.vit.parameters():
            param.requires_grad = False
        print("[Model] Both CNN and ViT backbones frozen.")

    def unfreeze_backbones(self):
        """Unfreezes all parameters in both CNN and ViT backbones."""
        for param in self.backbone.cnn.parameters():
            param.requires_grad = True
        for param in self.backbone.vit.parameters():
            param.requires_grad = True
        print("[Model] Both CNN and ViT backbones unfrozen.")

    def partial_finetune_backbones(self):
        """
        Freezes lower backbone stages while unfreezing top feature layers.
        Adapts high-level representation without full backbone computational/gradient load.
        - For CNN (EfficientNetV2): unfreezes conv_head, bn2, and blocks[-1].
        - For ViT (Swin-Tiny): unfreezes layers[-1] (Stage 3), norm, and head.
        - For MobileViT: unfreezes stages[-1], norm, head.
        """
        self.freeze_backbones()

        # Unfreeze top layers of CNN (e.g. conv_head, bn2, last block)
        cnn = self.backbone.cnn
        if hasattr(cnn, 'conv_head'):
            for p in cnn.conv_head.parameters():
                p.requires_grad = True
        if hasattr(cnn, 'bn2'):
            for p in cnn.bn2.parameters():
                p.requires_grad = True
        if hasattr(cnn, 'blocks') and len(cnn.blocks) > 0:
            for p in cnn.blocks[-1].parameters():
                p.requires_grad = True

        # Unfreeze top layers of ViT (Swin-Tiny layers[-1] or MobileViT stages[-1])
        vit = self.backbone.vit
        if hasattr(vit, 'head'):
            for p in vit.head.parameters():
                p.requires_grad = True
        if hasattr(vit, 'norm'):
            for p in vit.norm.parameters():
                p.requires_grad = True
        if hasattr(vit, 'layers') and len(vit.layers) > 0:
            # Swin-Tiny: Stage 3 (last stage)
            for p in vit.layers[-1].parameters():
                p.requires_grad = True
        elif hasattr(vit, 'stages') and len(vit.stages) > 0:
            # MobileViT: last stage
            for p in vit.stages[-1].parameters():
                p.requires_grad = True

        print("[Model] Configured partial fine-tuning: top stages unfrozen, early stages frozen.")

    def get_cam_target_layers(self):
        """
        Returns target layers for Grad-CAM model-attribution visualizations:
        - CNN: last convolutional block of EfficientNetV2
        - Transformer: last Swin Transformer block norm in Stage 3
        Note: Resulting heatmaps represent feature-attribution saliency maps,
        NOT segmentation masks or definitive clinical proof.
        """
        layers = {}
        cnn = self.backbone.cnn
        if hasattr(cnn, 'conv_head'):
            layers['cnn'] = cnn.conv_head
        elif hasattr(cnn, 'blocks') and len(cnn.blocks) > 0:
            layers['cnn'] = cnn.blocks[-1]

        vit = self.backbone.vit
        if hasattr(vit, 'layers') and len(vit.layers) > 0:
            layers['vit'] = vit.layers[-1].blocks[-1].norm1
        elif hasattr(vit, 'norm'):
            layers['vit'] = vit.norm
        elif hasattr(vit, 'stages') and len(vit.stages) > 0:
            layers['vit'] = vit.stages[-1]

        return layers

    def get_parameter_breakdown(self):
        """Returns parameter count breakdown dictionary."""
        return get_parameter_count(self)

    def forward_features(self, x):
        """Extracts disentangled feature representations for input x."""
        fused = self.backbone(x)
        z_shared, z_path_private, z_depth_private = self.disentangler(fused)
        return z_shared, z_path_private, z_depth_private

    def forward_pathology(self, x):
        """
        Pathology forward pass (for DFUC images).
        Combines z_shared and z_path_private. Isolates z_depth_private.
        Returns: z_shared, z_path_private, path_logits
        """
        z_shared, z_path_private, _ = self.forward_features(x)
        feat_path = torch.cat([z_shared, z_path_private], dim=1)
        path_logits = self.classifier(feat_path)
        return z_shared, z_path_private, path_logits

    def forward_wagner(self, x):
        """
        Wagner forward pass (for Wagner images).
        Combines z_shared and z_depth_private. Isolates z_path_private.
        Returns: z_shared, z_depth_private, wagner_logits
        """
        z_shared, _, z_depth_private = self.forward_features(x)
        feat_depth = torch.cat([z_shared, z_depth_private], dim=1)
        wagner_logits = self.wagner_head(feat_depth)
        return z_shared, z_depth_private, wagner_logits

    def forward_single(self, x, return_ordinal=False):
        """
        General inference/evaluation forward pass for a single image batch.
        Predicts both pathology and Wagner depth from their respective disentangled representations.
        Returns: (z_shared, path_logits) or (z_shared, path_logits, wagner_logits)
        """
        z_shared, z_path_private, z_depth_private = self.forward_features(x)
        feat_path = torch.cat([z_shared, z_path_private], dim=1)
        feat_depth = torch.cat([z_shared, z_depth_private], dim=1)

        path_logits = self.classifier(feat_path)
        if return_ordinal:
            wagner_logits = self.wagner_head(feat_depth)
            return z_shared, path_logits, wagner_logits
        return z_shared, path_logits

    def forward(self, img1, img2):
        """
        Siamese paired forward pass for DFUC pathology pairs.
        Returns:
            (z_shared1, z_shared2): shared representations for UG-MMD/LMCoT
            (logits1, logits2): pathology predictions
            (z_path1, z_path2): pathology-private representations
        """
        z_s1, z_p1, _ = self.forward_features(img1)
        z_s2, z_p2, _ = self.forward_features(img2)

        feat1 = torch.cat([z_s1, z_p1], dim=1)
        feat2 = torch.cat([z_s2, z_p2], dim=1)

        logits1 = self.classifier(feat1)
        logits2 = self.classifier(feat2)

        return (z_s1, z_s2), (logits1, logits2), (z_p1, z_p2)


# Alias for backward compatibility
DFUSiamNet = DFUSiamV3


# =============================================================================
# LOSS FUNCTIONS (DFU-SIAM v3)
# =============================================================================

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss for Long-Tailed DFU Pathology Classification.
    Reference: Cui et al. (CVPR 2019) 'Class-Balanced Loss Based on Effective Number of Samples'.

    Effective number of samples: E_c = (1 - beta^{n_c}) / (1 - beta)
    Weight: w_c = 1 / E_c, normalized across classes so sum(w) = num_classes.
    Focal Loss: FL = - alpha_y * (1 - p_y)^gamma * log(p_y)
    """
    def __init__(self, samples_per_cls=None, num_classes=4, beta=0.999, gamma=2.0, reduction='mean', eps=1e-7):
        super(ClassBalancedFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

        if samples_per_cls is not None:
            weights = self._compute_cb_weights(samples_per_cls, beta, num_classes)
            self.register_buffer('class_weights', torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer('class_weights', torch.ones(num_classes, dtype=torch.float32))

    @staticmethod
    def _compute_cb_weights(samples_per_cls, beta, num_classes):
        effective_num = [1.0 - math.pow(beta, n) if n > 0 else 1.0 for n in samples_per_cls]
        weights = [(1.0 - beta) / max(en, 1e-8) for en in effective_num]
        total = sum(weights)
        norm_weights = [w * (num_classes / total) for w in weights]
        return norm_weights

    def forward(self, logits, targets):
        """
        logits: (N, C) unnormalized class logits
        targets: (N,) ground-truth integer class labels
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Target class probabilities
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = torch.clamp(target_probs, self.eps, 1.0 - self.eps)

        # Class balancing weight
        alpha = self.class_weights.to(logits.device)[targets]

        # Focal modulation factor
        focal_weight = torch.pow(1.0 - target_probs, self.gamma)

        loss = -alpha * focal_weight * target_log_probs

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss


class CORALLoss(nn.Module):
    """
    Consistent Rank Logits (CORAL) Loss for Wagner Ordinal Grading (Grades 0-4).
    Reference: Cao et al. (Pattern Recognition Letters 2020).

    Uses K-1 = 4 binary cutoffs to model ordinal cumulative probabilities:
    P(y > k) for k in {0, 1, 2, 3}.
    Grade 0 -> [0, 0, 0, 0]
    Grade 1 -> [1, 0, 0, 0]
    Grade 2 -> [1, 1, 0, 0]
    Grade 3 -> [1, 1, 1, 0]
    Grade 4 -> [1, 1, 1, 1]
    """
    def __init__(self, num_classes=5):
        super(CORALLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        """
        logits: (N, num_classes - 1) cumulative binary cutoff logits
        targets: (N,) ground-truth ordinal integer grades (0 to num_classes - 1)
        """
        num_cutoffs = self.num_classes - 1
        cutoffs = torch.arange(num_cutoffs, device=targets.device).unsqueeze(0)
        binary_targets = (targets.unsqueeze(1) > cutoffs).float()

        loss = F.binary_cross_entropy_with_logits(logits, binary_targets)
        probs = torch.sigmoid(logits.detach())
        preds = torch.sum(probs > 0.5, dim=1).long()
        return loss, logits, preds


class HardPairLMCoTLoss(nn.Module):
    """
    Hard-Pair Large Margin Cotangent Loss (Hard-LMCoT).
    Applies tangent-of-half-angle metric learning with active hard-pair mining:
    - Positive pairs: same pathology class with highest distance (hardest positives).
    - Negative pairs: different pathology classes with lowest distance (hardest negatives).
    - Clinical boundary prioritization (None <-> Infection, Ischaemia <-> Both).
    - Warmup support: uses semi-hard / uniform pairs for early epochs, hard mining afterwards.
    - Numerical safeguards: float32 computation, safe normalization (eps=1e-8),
      bounded half_angle (max_dist ceiling avoiding tan(pi/2) singularity).
    """
    def __init__(self, margin=2.0, max_dist=10.0, eps=1e-4, warmup_epochs=5):
        super(HardPairLMCoTLoss, self).__init__()
        self.margin = margin
        self.max_dist = max_dist
        self.eps = eps
        self.warmup_epochs = warmup_epochs

    def _angular_distance(self, emb1, emb2):
        """Computes stable tan(theta/2) angular distance between embeddings in float32."""
        dev = emb1.device.type if emb1.is_cuda else 'cpu'
        with torch.amp.autocast(dev, enabled=False):
            emb1_f = emb1.float()
            emb2_f = emb2.float()
            emb1_norm = F.normalize(emb1_f, p=2, dim=-1, eps=1e-8)
            emb2_norm = F.normalize(emb2_f, p=2, dim=-1, eps=1e-8)
            cos_sim = torch.sum(emb1_norm * emb2_norm, dim=-1)
            cos_sim = torch.clamp(cos_sim, -1.0 + self.eps, 1.0 - self.eps)
            theta = torch.acos(cos_sim)
            max_ha = math.atan(self.max_dist)
            half_angle = torch.clamp(theta / 2.0, min=self.eps, max=max_ha)
            return torch.tan(half_angle)

    def forward_paired(self, emb1, emb2, target):
        """
        Standard paired forward pass on explicit pairs (N, D).
        target: (N,), 1 = same class, 0 = different class.
        """
        dev = emb1.device.type if emb1.is_cuda else 'cpu'
        with torch.amp.autocast(dev, enabled=False):
            ang_dist = self._angular_distance(emb1, emb2)
            target = target.float()
            loss_pos = target * torch.square(ang_dist)
            loss_neg = (1.0 - target) * torch.square(torch.clamp(self.margin - ang_dist, min=0.0))
            return torch.mean(loss_pos + loss_neg)

    def forward_mined(self, embeddings, labels, epoch=0):
        """
        Batch hard-pair mining mode.
        embeddings: (N, D) normalized embeddings
        labels: (N,) integer class labels
        epoch: current training epoch (controls warmup transition)
        """
        N = embeddings.size(0)
        if N < 2:
            return 0.0 * embeddings.sum()

        dev = embeddings.device.type if embeddings.is_cuda else 'cpu'
        with torch.amp.autocast(dev, enabled=False):
            # Execute metric learning in float32 to prevent half-precision overflow under AMP
            emb_f = embeddings.float()
            emb_norm = F.normalize(emb_f, p=2, dim=-1, eps=1e-8)

            # Pairwise angular distance matrix
            cos_sim = torch.matmul(emb_norm, emb_norm.t())
            cos_sim = torch.clamp(cos_sim, -1.0 + self.eps, 1.0 - self.eps)
            theta = torch.acos(cos_sim)
            max_ha = math.atan(self.max_dist)
            half_angle = torch.clamp(theta / 2.0, min=self.eps, max=max_ha)
            dist_matrix = torch.tan(half_angle)  # (N, N)

        # Pairwise identity masks
        labels_row = labels.unsqueeze(0)
        labels_col = labels.unsqueeze(1)
        pos_mask = (labels_row == labels_col)
        neg_mask = (labels_row != labels_col)

        # Exclude self-similarity diagonal
        pos_mask.fill_diagonal_(False)

        is_warmup = (epoch < self.warmup_epochs)

        losses = []

        # 1. Positive Pairs Mining (Same Class)
        has_pos = False
        for i in range(N):
            valid_pos = torch.where(pos_mask[i])[0]
            if len(valid_pos) > 0:
                has_pos = True
                if is_warmup:
                    # Warmup: average over all valid positive pairs
                    d_pos = dist_matrix[i, valid_pos]
                    losses.append(torch.mean(torch.square(d_pos)))
                else:
                    # Hard mining: pick the farthest positive (hardest positive)
                    hard_idx = valid_pos[torch.argmax(dist_matrix[i, valid_pos])]
                    d_pos = dist_matrix[i, hard_idx]
                    losses.append(torch.square(d_pos))

        # 2. Negative Pairs Mining (Different Classes)
        has_neg = False
        for i in range(N):
            valid_neg = torch.where(neg_mask[i])[0]
            if len(valid_neg) > 0:
                has_neg = True
                if is_warmup:
                    # Warmup: average over all valid negative pairs within margin
                    d_neg = dist_matrix[i, valid_neg]
                    losses.append(torch.mean(torch.square(torch.clamp(self.margin - d_neg, min=0.0))))
                else:
                    # Hard mining: pick the closest negative (hardest negative)
                    hard_idx = valid_neg[torch.argmin(dist_matrix[i, valid_neg])]
                    d_neg = dist_matrix[i, hard_idx]
                    losses.append(torch.square(torch.clamp(self.margin - d_neg, min=0.0)))

        if len(losses) == 0:
            return 0.0 * embeddings.sum()

        return torch.stack(losses).mean()

    def forward(self, *args, **kwargs):
        """Unified dispatch: accepts (emb1, emb2, target) or (embeddings, labels, [epoch])."""
        if len(args) == 3:
            return self.forward_paired(args[0], args[1], args[2])
        elif len(args) == 2:
            epoch = kwargs.get('epoch', 0)
            return self.forward_mined(args[0], args[1], epoch=epoch)
        raise ValueError(f"HardPairLMCoTLoss received unsupported argument count: {len(args)}")


# Alias for backward compatibility
LMCoTLoss = HardPairLMCoTLoss


class UncertaintyGatedMMDLoss(nn.Module):
    """
    Uncertainty-Gated Maximum Mean Discrepancy (UG-MMD) Loss.
    Aligns ONLY the shared representation z_shared across DFUC and Wagner cohorts.
    Samples are weighted by predictive confidence (1 - normalized entropy).

    CRITICAL SAFEGUARD:
    Confidence weights are strictly DETACHED (w = confidence.detach()) to prevent
    the network from gaming the alignment objective by artificially lowering confidence.
    """
    def __init__(self, kernel_scales=[0.1, 1.0, 10.0], min_weight=0.1, eps=1e-8):
        super(UncertaintyGatedMMDLoss, self).__init__()
        self.kernel_scales = kernel_scales
        self.min_weight = min_weight
        self.eps = eps

    @staticmethod
    def compute_pathology_uncertainty(logits, min_weight=0.1, eps=1e-8):
        """
        Computes normalized predictive entropy confidence from 4-class pathology logits.
        confidence = 1 - (H / log(4))
        Returns detached weights clamped in [min_weight, 1.0].
        """
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs + eps)
        # Normalized entropy in [0, 1]
        entropy = -torch.sum(probs * log_probs, dim=1) / math.log(4.0)
        confidence = torch.clamp(1.0 - entropy, min_weight, 1.0)
        return confidence.detach()

    @staticmethod
    def compute_wagner_uncertainty(coral_logits, min_weight=0.1, eps=1e-8):
        """
        Computes normalized predictive entropy confidence from CORAL cumulative logits.
        Constructs valid 5-class depth distribution from monotonic cumulative cutoffs.
        Returns detached weights clamped in [min_weight, 1.0].
        """
        q = torch.sigmoid(coral_logits)  # (N, 4) cumulative probabilities P(y > k)

        # Construct discrete grade probabilities [P(0), P(1), P(2), P(3), P(4)]
        p0 = torch.clamp(1.0 - q[:, 0], min=0.0)
        p1 = torch.clamp(q[:, 0] - q[:, 1], min=0.0)
        p2 = torch.clamp(q[:, 1] - q[:, 2], min=0.0)
        p3 = torch.clamp(q[:, 2] - q[:, 3], min=0.0)
        p4 = torch.clamp(q[:, 3], min=0.0)

        probs = torch.stack([p0, p1, p2, p3, p4], dim=1)  # (N, 5)
        probs = probs / (torch.sum(probs, dim=1, keepdim=True) + eps)

        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(probs * log_probs, dim=1) / math.log(5.0)
        confidence = torch.clamp(1.0 - entropy, min_weight, 1.0)
        return confidence.detach()

    def _weighted_rbf_kernel(self, x, y):
        """Computes multi-scale RBF kernel matrix between x (N, D) and y (M, D)."""
        d = x.size(1)
        x_norm = torch.sum(x**2, dim=1, keepdim=True)
        y_norm = torch.sum(y**2, dim=1, keepdim=True)
        dist = x_norm + y_norm.t() - 2.0 * torch.matmul(x, y.t())
        dist = torch.clamp(dist, min=0.0)

        kernel_val = 0.0
        for scale in self.kernel_scales:
            gamma = 1.0 / (2.0 * scale * d)
            kernel_val = kernel_val + torch.exp(-gamma * dist)
        return kernel_val

    def forward(self, x_shared, y_shared, weights_x=None, weights_y=None):
        """
        x_shared: DFUC shared representations (N, D)
        y_shared: Wagner shared representations (M, D)
        weights_x: (N,) detached sample confidence weights (optional)
        weights_y: (M,) detached sample confidence weights (optional)
        """
        N = x_shared.size(0)
        M = y_shared.size(0)

        if N == 0 or M == 0:
            return torch.tensor(0.0, device=x_shared.device, requires_grad=True)

        # Normalize sample weights so sum(w) = 1
        if weights_x is not None:
            wx = weights_x / (torch.sum(weights_x) + self.eps)
        else:
            wx = torch.ones(N, device=x_shared.device) / N

        if weights_y is not None:
            wy = weights_y / (torch.sum(weights_y) + self.eps)
        else:
            wy = torch.ones(M, device=y_shared.device) / M

        # Kernel matrices
        K_xx = self._weighted_rbf_kernel(x_shared, x_shared)
        K_yy = self._weighted_rbf_kernel(y_shared, y_shared)
        K_xy = self._weighted_rbf_kernel(x_shared, y_shared)

        # Weighted MMD terms: w^T K w
        mmd_xx = torch.sum(wx.unsqueeze(1) * wx.unsqueeze(0) * K_xx)
        mmd_yy = torch.sum(wy.unsqueeze(1) * wy.unsqueeze(0) * K_yy)
        mmd_xy = torch.sum(wx.unsqueeze(1) * wy.unsqueeze(0) * K_xy)

        mmd = mmd_xx + mmd_yy - 2.0 * mmd_xy
        return torch.clamp(mmd, min=0.0)


# Alias for backward compatibility
MMDLoss = UncertaintyGatedMMDLoss


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality Regularization Loss for Task Disentanglement.
    Discourages task-private representations from redundantly occupying shared space:
    L_orth = mean( (z_shared · z_path_private)^2 ) + mean( (z_shared · z_depth_private)^2 )

    Encourages: shared information ⟂ task-private information.
    Computation is linear O(B * D) with normalized cosine similarity.
    """
    def __init__(self, eps=1e-8):
        super(OrthogonalityLoss, self).__init__()
        self.eps = eps

    def _cosine_orthogonality(self, u, v):
        """Computes mean squared cosine similarity between normalized representations."""
        cos_sim = torch.sum(u * v, dim=-1)  # (N,)
        return torch.mean(torch.square(cos_sim))

    def forward(self, z_shared, z_path_private=None, z_depth_private=None):
        """
        Computes total orthogonality penalty across available task-private representations.
        z_shared: (N, D_shared)
        z_path_private: (N, D_path) [optional]
        z_depth_private: (M, D_depth) [optional]
        """
        loss = 0.0
        terms = 0

        if z_path_private is not None:
            # If dimensions match, compute direct dot product; else project or slice
            if z_shared.size(1) == z_path_private.size(1):
                loss = loss + self._cosine_orthogonality(z_shared, z_path_private)
            else:
                min_d = min(z_shared.size(1), z_path_private.size(1))
                loss = loss + self._cosine_orthogonality(z_shared[:, :min_d], z_path_private[:, :min_d])
            terms += 1

        if z_depth_private is not None:
            if z_shared.size(1) == z_depth_private.size(1):
                loss = loss + self._cosine_orthogonality(z_shared, z_depth_private)
            else:
                min_d = min(z_shared.size(1), z_depth_private.size(1))
                loss = loss + self._cosine_orthogonality(z_shared[:, :min_d], z_depth_private[:, :min_d])
            terms += 1

        if terms == 0:
            return torch.tensor(0.0, device=z_shared.device, requires_grad=True)

        return loss / terms


def get_device():
    """Returns available computation device (cuda or cpu)."""
    if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_available'):
        try:
            if torch.cuda.is_available():
                return torch.device('cuda')
        except Exception:
            pass
    return torch.device('cpu')


if __name__ == "__main__":
    print("[Testing DFU-SIAM v3 Architecture & Losses]")
    device = get_device()
    print(f"Device: {device}")

    # 1. Swin-Tiny Optimized Architecture Verification (~35.2M)
    print("\n--- 1. Testing Swin-Tiny Optimized Architecture (arch='swin_tiny') ---")
    model_swin = DFUSiamV3(arch="swin_tiny", pretrained=False).to(device)
    x = torch.randn(4, 3, 224, 224, device=device)

    counts_swin = model_swin.get_parameter_breakdown()
    print(f"Swin-Tiny Optimized Total Parameters: {counts_swin['total']:,}")
    print(f"  CNN ({counts_swin['cnn']:,}) + ViT Swin-Tiny ({counts_swin['vit']:,})")
    print(f"  CNN Proj ({counts_swin['cnn_proj']:,}) + ViT Proj ({counts_swin['vit_proj']:,})")
    print(f"  Disentangler ({counts_swin['disentangler']:,})")
    print(f"  Pathology Head ({counts_swin['classifier']:,}) + Wagner Head ({counts_swin['wagner_head']:,})")
    print(f"  Model Size: FP32={counts_swin['fp32_size_mb']:.2f} MB, FP16={counts_swin['fp16_size_mb']:.2f} MB")
    assert counts_swin['total'] == 35_158_798, f"Expected 35,158,798, got {counts_swin['total']}"

    z_s, z_p, z_d = model_swin.forward_features(x)
    assert z_s.shape == (4, 256)
    assert z_p.shape == (4, 256)
    assert z_d.shape == (4, 256)

    _, _, path_logits = model_swin.forward_pathology(x)
    _, _, wagner_logits = model_swin.forward_wagner(x)
    assert path_logits.shape == (4, 4)
    assert wagner_logits.shape == (4, 4)

    # Test explainability CAM targets
    cam_targets = model_swin.get_cam_target_layers()
    print(f"CAM Targets: CNN={type(cam_targets['cnn']).__name__}, ViT={type(cam_targets['vit']).__name__}")
    assert 'cnn' in cam_targets and 'vit' in cam_targets

    # 2. Backbone Freezing Modes for Swin-Tiny
    print("\n--- 2. Testing Swin-Tiny Freezing Modes ---")
    model_swin.freeze_backbones()
    counts_frozen = model_swin.get_parameter_breakdown()
    print(f"Frozen Backbone: Trainable={counts_frozen['trainable']:,}, Non-trainable={counts_frozen['non_trainable']:,}")
    assert counts_frozen['trainable'] == 1_780_740

    model_swin.partial_finetune_backbones()
    counts_partial = model_swin.get_parameter_breakdown()
    print(f"Partial Fine-Tuning: Trainable={counts_partial['trainable']:,}, Non-trainable={counts_partial['non_trainable']:,}")
    assert counts_partial['trainable'] == 21_635_552

    model_swin.unfreeze_backbones()
    counts_unfrozen = model_swin.get_parameter_breakdown()
    assert counts_unfrozen['trainable'] == counts_swin['total']

    # 3. Test Losses & Backward on Swin-Tiny Model
    print("\n--- 3. Testing Multi-Task Losses and Backward Pass on Swin-Tiny ---")
    cb_focal = ClassBalancedFocalLoss(samples_per_cls=[2552, 2555, 227, 621]).to(device)
    coral = CORALLoss(num_classes=5).to(device)
    hard_lmcot = HardPairLMCoTLoss().to(device)
    ug_mmd = UncertaintyGatedMMDLoss().to(device)
    orth_loss = OrthogonalityLoss().to(device)

    targets_path = torch.tensor([0, 1, 2, 3], device=device)
    targets_wagner = torch.tensor([0, 1, 3, 4], device=device)

    l_focal = cb_focal(path_logits, targets_path)
    l_coral, _, _ = coral(wagner_logits, targets_wagner)
    l_lmcot = hard_lmcot.forward_mined(z_s, targets_path, epoch=1)
    
    w_dfu = UncertaintyGatedMMDLoss.compute_pathology_uncertainty(path_logits)
    w_wag = UncertaintyGatedMMDLoss.compute_wagner_uncertainty(wagner_logits)
    l_mmd = ug_mmd(z_s, z_s, w_dfu, w_wag)
    l_orth = orth_loss(z_s, z_p, z_d)

    total_loss = l_focal + 1.0 * l_coral + 1.0 * l_lmcot + 0.1 * l_mmd + 0.1 * l_orth
    print(f"Total loss: {total_loss.item():.4f}")
    assert total_loss.requires_grad
    total_loss.backward()

    # 4. Full Architecture Verification (~59.8M)
    print("\n--- 4. Testing Full Architecture (arch='full') ---")
    model_full = DFUSiamV3(arch="full", pretrained=False).to(device)
    counts_full = model_full.get_parameter_breakdown()
    print(f"Full Architecture Total Parameters: {counts_full['total']:,}")
    assert counts_full['total'] == 59_783_374

    print("Backward pass succeeded. All DFU-SIAM v3 self-tests passed successfully!")
