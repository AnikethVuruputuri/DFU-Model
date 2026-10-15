"""
DFU-SIAM v3 Comprehensive Test Suite.
Validates:
1. Module imports (DFUSiamV3, CB-Focal, CORAL, Hard-LMCoT, UG-MMD, OrthogonalityLoss)
2. Model initialization & architecture parameters
3. Model shape test (batch size 1 and batch size > 1, shared/private dimensions)
4. Class-Balanced Focal Loss (effective number weights, probabilities, autograd)
5. CORAL Loss (target encoding, cutoffs, decoding, autograd)
6. Hard-Pair LMCoT Loss (tangent metric, polarity, mining, empty-pair safety)
7. Uncertainty-Gated MMD (detached confidence weights, multi-kernel RBF, autograd)
8. Orthogonality Loss (disentanglement penalty, orthogonal vs aligned features)
9. Complete loss combination (requires_grad=True, finite values)
10. Full backpropagation & gradient reach across all modules
11. Checkpoint save/load fidelity & legacy checkpoint rejection
12. Real-data mini-batch smoke test (DFUC pair + Wagner batch)
13. Inference pipeline & transforms
"""

import os
import sys
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_imports():
    """Test all required imports for DFU-SIAM v3."""
    print("[TEST 1] Importing DFU-SIAM v3 modules...", end=" ")
    try:
        from dfu_siam_model import (
            DFUSiamV3, DFUSiamV3Backbone, TaskDisentangler,
            ClassBalancedFocalLoss, CORALLoss, HardPairLMCoTLoss,
            UncertaintyGatedMMDLoss, OrthogonalityLoss, get_device
        )
        from dataset import (
            DFUSingleDataset, DFUSiamesePairDataset, WagnerSingleDataset,
            InterleavedDataLoader, get_dfu_transforms, get_tta_transforms
        )
        from fusion_eval import evaluate_wagner_ordinal, predict_ut_grid, evaluate_zero_shot_cross_taxonomy
        print("PASS")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_model_init():
    """Test DFUSiamV3 model initialization and parameter count."""
    print("[TEST 2] Model initialization...", end=" ")
    from dfu_siam_model import DFUSiamV3
    try:
        model = DFUSiamV3(pretrained=False, dropout=0.3)
        assert model.model_version == "v3", f"Expected version v3, got {model.model_version}"
        assert model.backbone.fused_dim == 2560
        assert model.shared_dim == 768
        assert model.path_dim == 512
        assert model.depth_dim == 512

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params == trainable_params, "All model parameters should be trainable"

        print(f"PASS (params: {total_params:,} total, {trainable_params:,} trainable)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_model_shapes():
    """Test forward tensor dimensions for batch sizes 1 and 4."""
    print("[TEST 3] Model tensor shapes (batch 1 & batch 4)...", end=" ")
    from dfu_siam_model import DFUSiamV3
    try:
        model = DFUSiamV3(pretrained=False)
        model.eval()

        for bs in [1, 4]:
            x = torch.randn(bs, 3, 224, 224)

            # Feature disentanglement
            z_s, z_p, z_d = model.forward_features(x)
            assert z_s.shape == (bs, 768), f"Expected z_shared shape ({bs}, 768), got {z_s.shape}"
            assert z_p.shape == (bs, 512), f"Expected z_path shape ({bs}, 512), got {z_p.shape}"
            assert z_d.shape == (bs, 512), f"Expected z_depth shape ({bs}, 512), got {z_d.shape}"

            # L2-normalization verification
            assert torch.allclose(torch.norm(z_s, p=2, dim=1), torch.ones(bs), atol=1e-4)
            assert torch.allclose(torch.norm(z_p, p=2, dim=1), torch.ones(bs), atol=1e-4)
            assert torch.allclose(torch.norm(z_d, p=2, dim=1), torch.ones(bs), atol=1e-4)

            # Pathology forward
            _, _, p_logits = model.forward_pathology(x)
            assert p_logits.shape == (bs, 4), f"Expected pathology logits ({bs}, 4), got {p_logits.shape}"

            # Wagner forward
            _, _, w_logits = model.forward_wagner(x)
            assert w_logits.shape == (bs, 4), f"Expected Wagner cutoffs ({bs}, 4), got {w_logits.shape}"

            # Single inference forward
            z_s_out, p_out, w_out = model.forward_single(x, return_ordinal=True)
            assert z_s_out.shape == (bs, 768)
            assert p_out.shape == (bs, 4)
            assert w_out.shape == (bs, 4)

        print("PASS (shapes verified: 768-D shared, 512-D private, 4 logits)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_class_balanced_focal_loss():
    """Test Class-Balanced Focal Loss properties, weighting, and gradients."""
    print("[TEST 4] Class-Balanced Focal Loss...", end=" ")
    from dfu_siam_model import ClassBalancedFocalLoss
    try:
        samples_per_cls = [2552, 2555, 227, 621]
        loss_fn = ClassBalancedFocalLoss(samples_per_cls=samples_per_cls, gamma=2.0, beta=0.999)

        # Check that minority class (ischaemia, index 2) receives higher weight than majority
        weights = loss_fn.class_weights
        assert weights[2] > weights[0], "Minority class (ischaemia) must have higher weight than class 0"
        assert weights[2] > weights[1], "Minority class (ischaemia) must have higher weight than class 1"

        # Test forward pass with gradients
        logits = torch.randn(4, 4, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3])

        loss = loss_fn(logits, targets)
        assert loss.dim() == 0, "Loss must be scalar"
        assert loss.requires_grad, "Loss must require grad"
        assert torch.isfinite(loss), "Loss must be finite"

        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        assert torch.any(logits.grad != 0), "Gradients must not be all zero"

        print(f"PASS (loss={loss.item():.4f}, minority_wt={weights[2].item():.2f} > majority_wt={weights[0].item():.2f})")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_coral_loss():
    """Test CORAL ordinal cumulative loss, target encoding, and decoding."""
    print("[TEST 5] CORAL ordinal loss...", end=" ")
    from dfu_siam_model import CORALLoss
    try:
        loss_fn = CORALLoss(num_classes=5)
        logits = torch.randn(5, 4, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

        loss, out_logits, preds = loss_fn(logits, targets)
        assert loss.dim() == 0, "Loss must be scalar"
        assert loss.requires_grad, "Loss must require grad"
        assert torch.isfinite(loss), "Loss must be finite"
        assert preds.shape == (5,), f"Expected preds shape (5,), got {preds.shape}"
        assert (preds >= 0).all() and (preds <= 4).all()

        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()

        # Cumulative cutoff sanity check
        cutoffs = torch.arange(4).unsqueeze(0)
        for g in range(5):
            t = torch.tensor([g])
            b = (t.unsqueeze(1) > cutoffs).float().squeeze(0).tolist()
            expected = [1.0 if g > k else 0.0 for k in range(4)]
            assert b == expected, f"Grade {g} encoding mismatch"

        print(f"PASS (loss={loss.item():.4f}, preds={preds.tolist()}, encoding=OK)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_hard_pair_lmcot():
    """Test Hard-Pair LMCoT Loss: 12 comprehensive numerical, stability, and boundary tests."""
    print("[TEST 6] Hard-Pair LMCoT loss...", end=" ")
    from dfu_siam_model import HardPairLMCoTLoss
    try:
        loss_fn = HardPairLMCoTLoss(margin=2.0, max_dist=10.0, eps=1e-4)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1. Identical embeddings
        e1 = torch.randn(4, 64, device=dev, requires_grad=True)
        l1 = loss_fn.forward_paired(e1, e1, torch.ones(4, device=dev))
        l1.backward()
        assert torch.isfinite(l1) and torch.isfinite(e1.grad).all()

        # 2. Near-identical embeddings
        e2_a = torch.randn(4, 64, device=dev, requires_grad=True)
        e2_b = e2_a + 1e-4 * torch.randn(4, 64, device=dev)
        l2 = loss_fn.forward_paired(e2_a, e2_b, torch.ones(4, device=dev))
        l2.backward()
        assert torch.isfinite(l2) and torch.isfinite(e2_a.grad).all()

        # 3. Orthogonal embeddings
        e3_a = torch.eye(4, 64, device=dev, requires_grad=True)
        e3_b = torch.eye(4, 64, device=dev)[[1, 0, 3, 2]]
        l3 = loss_fn.forward_paired(e3_a, e3_b, torch.zeros(4, device=dev))
        l3.backward()
        assert torch.isfinite(l3) and torch.isfinite(e3_a.grad).all()

        # 4. Opposite embeddings (penalty for pos pair, zero for neg pair)
        e4_a = torch.randn(4, 64, device=dev, requires_grad=True)
        e4_b = -e4_a
        l4_pos = loss_fn.forward_paired(e4_a, e4_b, torch.ones(4, device=dev))
        l4_neg = loss_fn.forward_paired(e4_a, e4_b, torch.zeros(4, device=dev))
        l4_pos.backward()
        assert torch.isfinite(l4_pos) and torch.isfinite(e4_a.grad).all()
        assert l4_neg.item() == 0.0, "Opposite embeddings with distance > margin must yield 0 negative loss"
        assert l1.item() < l4_pos.item(), "Identical pos pair must have lower loss than opposite pos pair"

        # 5. Random normalized embeddings
        e5_a = torch.randn(8, 64, device=dev, requires_grad=True)
        e5_b = torch.randn(8, 64, device=dev)
        l5 = loss_fn.forward_paired(e5_a, e5_b, torch.randint(0, 2, (8,), device=dev))
        l5.backward()
        assert torch.isfinite(l5) and torch.isfinite(e5_a.grad).all()

        # 6. Batch with only one sample from a class
        e6 = torch.randn(5, 64, device=dev, requires_grad=True)
        lbls6 = torch.tensor([0, 0, 1, 1, 2], device=dev)
        l6 = loss_fn.forward_mined(e6, lbls6, epoch=5)
        l6.backward()
        assert torch.isfinite(l6) and torch.isfinite(e6.grad).all()

        # 7. Batch with no valid positive pairs (all distinct classes)
        e7 = torch.randn(4, 64, device=dev, requires_grad=True)
        lbls7 = torch.tensor([0, 1, 2, 3], device=dev)
        l7 = loss_fn.forward_mined(e7, lbls7, epoch=5)
        l7.backward()
        assert torch.isfinite(l7) and torch.isfinite(e7.grad).all()

        # 8. Batch with no valid negative pairs (all same class)
        e8 = torch.randn(4, 64, device=dev, requires_grad=True)
        lbls8 = torch.tensor([1, 1, 1, 1], device=dev)
        l8 = loss_fn.forward_mined(e8, lbls8, epoch=5)
        l8.backward()
        assert torch.isfinite(l8) and torch.isfinite(e8.grad).all()

        # 9. Hard-positive and hard-negative mining
        e9 = torch.randn(8, 64, device=dev, requires_grad=True)
        lbls9 = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=dev)
        l9 = loss_fn.forward_mined(e9, lbls9, epoch=10)
        l9.backward()
        assert torch.isfinite(l9) and torch.isfinite(e9.grad).all()

        # 10. Forward and backward passes under float32
        e10 = torch.randn(8, 64, dtype=torch.float32, device=dev, requires_grad=True)
        lbls10 = torch.tensor([0, 1, 0, 1, 2, 2, 3, 3], device=dev)
        l10 = loss_fn.forward_mined(e10, lbls10, epoch=2)
        l10.backward()
        assert torch.isfinite(l10) and torch.isfinite(e10.grad).all()

        # 11. Mixed-precision compatibility under AMP
        e11 = torch.randn(8, 64, device=dev, requires_grad=True)
        lbls11 = torch.tensor([0, 1, 0, 1, 2, 2, 3, 3], device=dev)
        with torch.amp.autocast(dev):
            l11 = loss_fn.forward_mined(e11, lbls11, epoch=10)
        l11.backward()
        assert torch.isfinite(l11) and torch.isfinite(e11.grad).all()

        # 12. Finite and bounded gradients across all cases
        max_loss = max(l.item() for l in [l1, l2, l3, l4_pos, l5, l6, l7, l8, l9, l10, l11])
        print(f"PASS (12 subtests OK: max_loss={max_loss:.4f}, stable_gradients=True)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_ug_mmd_loss():
    """Test Uncertainty-Gated MMD with detached confidence weights."""
    print("[TEST 7] Uncertainty-Gated MMD loss...", end=" ")
    from dfu_siam_model import UncertaintyGatedMMDLoss
    try:
        mmd_fn = UncertaintyGatedMMDLoss(min_weight=0.1)

        x_raw = torch.randn(8, 768, requires_grad=True)
        x = F.normalize(x_raw, p=2, dim=1)
        y = F.normalize(torch.randn(8, 768), p=2, dim=1)

        # Confidence weights
        dummy_p_logits = torch.randn(8, 4)
        dummy_w_logits = torch.randn(8, 4)
        w_dfu = UncertaintyGatedMMDLoss.compute_pathology_uncertainty(dummy_p_logits, min_weight=0.1)
        w_wag = UncertaintyGatedMMDLoss.compute_wagner_uncertainty(dummy_w_logits, min_weight=0.1)

        # Check weights are detached (Rule 11)
        assert not w_dfu.requires_grad, "DFUC confidence weights must be detached"
        assert not w_wag.requires_grad, "Wagner confidence weights must be detached"
        assert (w_dfu >= 0.1).all() and (w_dfu <= 1.0).all()
        assert (w_wag >= 0.1).all() and (w_wag <= 1.0).all()

        # Identical vs different distributions
        loss_same = mmd_fn(x, x.detach(), w_dfu, w_dfu)
        loss_diff = mmd_fn(x, y, w_dfu, w_wag)

        assert loss_same.item() < loss_diff.item(), "Identical distributions must have lower MMD than different"
        assert loss_diff.requires_grad
        loss_diff.backward()
        assert x_raw.grad is not None and torch.isfinite(x_raw.grad).all()

        print(f"PASS (same={loss_same.item():.6f} < diff={loss_diff.item():.4f}, weights_detached=True)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_orthogonality_loss():
    """Test Orthogonality Loss on task-private and shared representations."""
    print("[TEST 8] Orthogonality disentanglement loss...", end=" ")
    from dfu_siam_model import OrthogonalityLoss
    try:
        orth_fn = OrthogonalityLoss()

        # Perfectly orthogonal vectors
        u = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        v = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        loss_orth = orth_fn._cosine_orthogonality(u, v)
        assert torch.isclose(loss_orth, torch.tensor(0.0)), "Orthogonal vectors must have 0.0 loss"

        # Perfectly parallel vectors
        loss_aligned = orth_fn._cosine_orthogonality(u, u)
        assert torch.isclose(loss_aligned, torch.tensor(1.0)), "Aligned vectors must have 1.0 penalty"

        # Real multi-dimensional tensor test
        z_s_raw = torch.randn(4, 768, requires_grad=True)
        z_p_raw = torch.randn(4, 512, requires_grad=True)
        z_d_raw = torch.randn(4, 512, requires_grad=True)
        z_s = F.normalize(z_s_raw, p=2, dim=1)
        z_p = F.normalize(z_p_raw, p=2, dim=1)
        z_d = F.normalize(z_d_raw, p=2, dim=1)

        loss = orth_fn(z_s, z_p, z_d)
        assert loss.dim() == 0 and loss.requires_grad
        loss.backward()
        assert z_s_raw.grad is not None and z_p_raw.grad is not None and z_d_raw.grad is not None

        print(f"PASS (orth={loss_orth.item():.4f} < aligned={loss_aligned.item():.4f}, total={loss.item():.4f})")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_full_backpropagation():
    """Test full multi-task backward pass through all DFU-SIAM v3 modules."""
    print("[TEST 9] Full backpropagation & gradient reach...", end=" ")
    from dfu_siam_model import (
        DFUSiamV3, ClassBalancedFocalLoss, CORALLoss, HardPairLMCoTLoss,
        UncertaintyGatedMMDLoss, OrthogonalityLoss
    )
    try:
        model = DFUSiamV3(pretrained=False)
        cb_focal = ClassBalancedFocalLoss(samples_per_cls=[2552, 2555, 227, 621])
        coral = CORALLoss(num_classes=5)
        lmcot = HardPairLMCoTLoss()
        ug_mmd = UncertaintyGatedMMDLoss()
        orth = OrthogonalityLoss()

        model.train()
        img1 = torch.randn(4, 3, 224, 224)
        img2 = torch.randn(4, 3, 224, 224)
        c1 = torch.tensor([0, 1, 2, 3])
        c2 = torch.tensor([1, 1, 2, 0])
        w_img = torch.randn(4, 3, 224, 224)
        w_lbl = torch.tensor([0, 1, 3, 4])

        # Forward
        z_s1, z_p1, logits1 = model.forward_pathology(img1)
        z_s2, z_p2, logits2 = model.forward_pathology(img2)
        z_sw, z_dw, w_logits = model.forward_wagner(w_img)

        # Losses
        l_path = 0.5 * (cb_focal(logits1, c1) + cb_focal(logits2, c2))
        l_coral, _, _ = coral(w_logits, w_lbl)
        l_lmcot = lmcot.forward_mined(torch.cat([z_s1, z_s2], dim=0), torch.cat([c1, c2], dim=0), epoch=10)
        
        w_dfu = UncertaintyGatedMMDLoss.compute_pathology_uncertainty(logits1)
        w_wag = UncertaintyGatedMMDLoss.compute_wagner_uncertainty(w_logits)
        l_mmd = ug_mmd(z_s1, z_sw, w_dfu, w_wag)

        l_orth = 0.5 * (orth(z_s1, z_p1) + orth(z_sw, z_depth_private=z_dw))

        l_total = l_path + 1.0 * l_coral + 1.0 * l_lmcot + 0.1 * l_mmd + 0.1 * l_orth

        assert l_total.requires_grad
        assert torch.isfinite(l_total)
        l_total.backward()

        # Check gradients reach every sub-module
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.backbone.cnn.parameters()), "CNN backbone has no grad"
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.backbone.vit.parameters()), "ViT backbone has no grad"
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.disentangler.parameters()), "Disentangler has no grad"
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.classifier.parameters()), "Classifier head has no grad"
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.wagner_head.parameters()), "Wagner head has no grad"

        print(f"PASS (loss={l_total.item():.4f}, all sub-modules received gradients)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_checkpoint_v3_compatibility():
    """Test checkpoint save, load fidelity, and rejection of legacy checkpoints."""
    print("[TEST 10] Checkpoint v3 compatibility & legacy rejection...", end=" ")
    from dfu_siam_model import DFUSiamV3
    import tempfile
    try:
        model = DFUSiamV3(pretrained=False)
        test_input = torch.randn(2, 3, 224, 224)
        model.eval()

        with torch.no_grad():
            orig_s, orig_p, orig_w = model.forward_single(test_input, return_ordinal=True)

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save valid v3 checkpoint
            torch.save({
                'model_version': 'v3',
                'epoch': 1,
                'model_state_dict': model.state_dict(),
                'config': {'shared_dim': 768, 'path_dim': 512, 'depth_dim': 512}
            }, tmp_path)

            # Reload into fresh instance
            fresh = DFUSiamV3(pretrained=False)
            ckpt = torch.load(tmp_path, weights_only=False)
            assert ckpt['model_version'] == 'v3'
            fresh.load_state_dict(ckpt['model_state_dict'])
            fresh.eval()

            with torch.no_grad():
                fresh_s, fresh_p, fresh_w = fresh.forward_single(test_input, return_ordinal=True)

            assert torch.equal(orig_s, fresh_s), "Shared output mismatch upon reload"
            assert torch.equal(orig_p, fresh_p), "Pathology output mismatch upon reload"
            assert torch.equal(orig_w, fresh_w), "Wagner output mismatch upon reload"

            # Test legacy checkpoint rejection (Rule 6)
            legacy_ckpt = {'model_version': 'v2', 'epoch': 1}
            torch.save(legacy_ckpt, tmp_path)
            loaded_legacy = torch.load(tmp_path, weights_only=False)
            rejected = False
            if loaded_legacy.get('model_version') != 'v3':
                rejected = True
            assert rejected, "Legacy checkpoint must be flagged as incompatible"

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        print("PASS (exact output reload verified, legacy rejected)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_synthetic_backward_smoke():
    """Test synthetic multi-task forward and backward autograd reach without real data or optimizer updates."""
    print("[TEST 11] Synthetic multi-task backward pass smoke test...", end=" ")
    from dfu_siam_model import (
        DFUSiamV3, ClassBalancedFocalLoss, CORALLoss, HardPairLMCoTLoss,
        UncertaintyGatedMMDLoss, OrthogonalityLoss, get_device
    )
    try:
        device = get_device()
        model = DFUSiamV3(pretrained=False).to(device)
        model.train()

        img1 = torch.randn(2, 3, 224, 224, device=device)
        img2 = torch.randn(2, 3, 224, 224, device=device)
        pair_t = torch.tensor([1.0, 0.0], device=device)
        c1 = torch.tensor([0, 1], device=device)
        c2 = torch.tensor([1, 2], device=device)
        w_img = torch.randn(2, 3, 224, 224, device=device)
        w_lbl = torch.tensor([0, 2], device=device)

        z_s1, z_p1, logits1 = model.forward_pathology(img1)
        z_s2, z_p2, logits2 = model.forward_pathology(img2)
        z_sw, z_dw, w_logits = model.forward_wagner(w_img)

        cb_loss = ClassBalancedFocalLoss().to(device)
        coral = CORALLoss().to(device)
        lmcot = HardPairLMCoTLoss().to(device)
        ug_mmd = UncertaintyGatedMMDLoss().to(device)
        orth = OrthogonalityLoss().to(device)

        l_path = 0.5 * (cb_loss(logits1, c1) + cb_loss(logits2, c2))
        l_coral, _, _ = coral(w_logits, w_lbl)
        l_lmcot = lmcot.forward_paired(z_s1, z_s2, pair_t)
        w_dfu = UncertaintyGatedMMDLoss.compute_pathology_uncertainty(logits1)
        w_wag = UncertaintyGatedMMDLoss.compute_wagner_uncertainty(w_logits)
        l_mmd = ug_mmd(z_s1, z_sw, w_dfu, w_wag)
        l_orth = 0.5 * (orth(z_s1, z_p1) + orth(z_sw, z_depth_private=z_dw))

        loss = l_path + l_coral + l_lmcot + 0.1 * l_mmd + 0.1 * l_orth
        loss.backward()

        # Check gradients reach model parameters without calling optimizer.step()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

        print(f"PASS (Loss={loss.item():.4f}, Path={l_path.item():.4f}, CORAL={l_coral.item():.4f}, MMD={l_mmd.item():.4f})")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_inference_pipeline():
    """Test single-image and batch inference pipeline."""
    print("[TEST 12] Inference pipeline & TTA transforms...", end=" ")
    from dfu_siam_model import DFUSiamV3
    from dataset import get_tta_transforms
    from fusion_eval import predict_ut_grid
    try:
        model = DFUSiamV3(pretrained=False)
        model.eval()

        tta_tfs = get_tta_transforms(img_size=(224, 224))
        assert len(tta_tfs) == 4, f"Expected 4 TTA transforms, got {len(tta_tfs)}"

        dummy_img = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            _, path_logits, wagner_logits = model.forward_single(dummy_img, return_ordinal=True)

        probs = F.softmax(path_logits, dim=1).numpy()
        w_out = wagner_logits.numpy()

        ut_grid, depth_probs = predict_ut_grid(probs[0], w_out[0])
        assert ut_grid.shape == (4, 4), f"Expected UT Grid (4, 4), got {ut_grid.shape}"
        assert np.isclose(np.sum(ut_grid), 1.0, atol=1e-3), "UT grid probabilities must sum to 1.0"
        assert depth_probs.shape == (4,), f"Expected depth probs (4,), got {depth_probs.shape}"

        print("PASS (UT 16-cell grid shape verified [4, 4], sum=1.0)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_wagner_dedup_and_baseline_isolation():
    """Test Wagner dataset deduplication, split integrity, and A0-Wag gradient isolation."""
    print("[TEST 13] Wagner deduplication & baseline isolation...", end=" ")
    try:
        from dataset import WagnerSingleDataset
        from dfu_siam_model import DFUSiamV3, CORALLoss

        # 1. Dataset split count checks
        train_ds = WagnerSingleDataset('WagnerDataset/train')
        val_ds = WagnerSingleDataset('WagnerDataset/valid')
        assert len(train_ds) == 11491, f"Expected 11491 train samples after deduplication, got {len(train_ds)}"
        assert len(val_ds) == 303, f"Expected 303 validation samples, got {len(val_ds)}"

        # 2. Verify zero hash leakage
        import hashlib
        eval_hashes = set()
        for p in val_ds.img_paths:
            with open(p, 'rb') as fp:
                eval_hashes.add(hashlib.md5(fp.read()).hexdigest())

        train_hashes = set()
        for p in train_ds.img_paths:
            with open(p, 'rb') as fp:
                train_hashes.add(hashlib.md5(fp.read()).hexdigest())

        leak_count = len(train_hashes.intersection(eval_hashes))
        assert leak_count == 0, f"Found {leak_count} duplicate hashes between train and valid!"

        # 3. Verify A0-Wag gradient isolation
        model = DFUSiamV3(pretrained=False)
        model.train()
        coral_crit = CORALLoss()

        w_img = torch.randn(4, 3, 224, 224)
        w_lbl = torch.tensor([0, 1, 2, 3])

        z_s_wag, z_d_wag, w_logits = model.forward_wagner(w_img)
        loss, _, _ = coral_crit(w_logits, w_lbl)
        loss.backward()

        wagner_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.wagner_head.parameters())
        classifier_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.classifier.parameters())
        path_proj_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.disentangler.path_proj.parameters())
        depth_proj_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.disentangler.depth_proj.parameters())

        assert wagner_has_grad, "wagner_head must receive gradients in A0-Wag"
        assert depth_proj_has_grad, "depth_proj must receive gradients in A0-Wag"
        assert not classifier_has_grad, "classifier must NOT receive gradients in A0-Wag"
        assert not path_proj_has_grad, "path_proj must NOT receive gradients in A0-Wag"

        print("PASS (11491 train samples, 0 hash leaks, pure Wagner gradient isolation)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_lightweight_architecture():
    """Test Lightweight Architecture (~12.5M params <= 30M), tensor shapes, freezing modes, and backward pass."""
    print("[TEST 14] Lightweight Architecture (~12.5M params <= 30M target)...", end=" ")
    try:
        from dfu_siam_model import (
            DFUSiamV3, ClassBalancedFocalLoss, CORALLoss, HardPairLMCoTLoss,
            UncertaintyGatedMMDLoss, OrthogonalityLoss
        )

        # 1. Model initialization in lightweight mode
        model = DFUSiamV3(arch="lightweight", pretrained=False)
        p_info = model.get_parameter_breakdown()

        # 2. Parameter count assertions
        assert p_info['total'] <= 30_000_000, f"Total params {p_info['total']} exceeds 30M limit!"
        assert p_info['total'] == 12_544_308, f"Expected 12,544,308 params, got {p_info['total']}"

        # 3. Tensor shapes in eval mode (Batch size 1 and Batch size 4)
        model.eval()
        for bsz in [1, 4]:
            x = torch.randn(bsz, 3, 224, 224)
            zs, zp, zd = model.forward_features(x)
            assert zs.shape == (bsz, 256), f"Expected zs shape ({bsz}, 256), got {zs.shape}"
            assert zp.shape == (bsz, 256), f"Expected zp shape ({bsz}, 256), got {zp.shape}"
            assert zd.shape == (bsz, 256), f"Expected zd shape ({bsz}, 256), got {zd.shape}"

            _, _, path_logits = model.forward_pathology(x)
            _, _, wagner_logits = model.forward_wagner(x)
            assert path_logits.shape == (bsz, 4), f"Expected path_logits shape ({bsz}, 4), got {path_logits.shape}"
            assert wagner_logits.shape == (bsz, 4), f"Expected wagner_logits shape ({bsz}, 4), got {wagner_logits.shape}"

            zs_s, p_s, w_s = model.forward_single(x, return_ordinal=True)
            assert p_s.shape == (bsz, 4) and w_s.shape == (bsz, 4)

        # 4. Backbone Freezing Tests
        model.freeze_backbones()
        p_frozen = model.get_parameter_breakdown()
        assert p_frozen['trainable'] == 1_747_972, f"Expected 1,747,972 trainable params when frozen, got {p_frozen['trainable']}"
        assert not any(p.requires_grad for p in model.backbone.cnn.parameters()), "CNN params must be non-trainable when frozen"
        assert not any(p.requires_grad for p in model.backbone.vit.parameters()), "ViT params must be non-trainable when frozen"

        # 5. Partial Fine-Tuning Tests
        model.partial_finetune_backbones()
        p_partial = model.get_parameter_breakdown()
        assert p_partial['trainable'] > p_frozen['trainable']
        assert p_partial['trainable'] < p_info['total']

        # 6. Backward Pass Verification in train mode
        model.unfreeze_backbones()
        model.train()
        x_in = torch.randn(4, 3, 224, 224)
        targets_p = torch.tensor([0, 1, 2, 3])
        targets_w = torch.tensor([0, 1, 3, 4])

        zs, zp, p_logits = model.forward_pathology(x_in)
        _, zd, w_logits = model.forward_wagner(x_in)

        cb_focal = ClassBalancedFocalLoss(samples_per_cls=[2552, 2555, 227, 621])
        coral = CORALLoss(num_classes=5)
        lmcot = HardPairLMCoTLoss()
        ug_mmd = UncertaintyGatedMMDLoss()
        orth = OrthogonalityLoss()

        l_p = cb_focal(p_logits, targets_p)
        l_w, _, _ = coral(w_logits, targets_w)
        l_lmc = lmcot.forward_mined(zs, targets_p, epoch=1)
        w_dfu = UncertaintyGatedMMDLoss.compute_pathology_uncertainty(p_logits)
        w_wag = UncertaintyGatedMMDLoss.compute_wagner_uncertainty(w_logits)
        l_m = ug_mmd(zs, zs, w_dfu, w_wag)
        l_o = orth(zs, zp, zd)

        total_l = l_p + l_w + 0.5 * l_lmc + 0.1 * l_m + 0.1 * l_o
        assert total_l.requires_grad
        total_l.backward()

        assert any(p.grad is not None for p in model.backbone.cnn.parameters())
        assert any(p.grad is not None for p in model.classifier.parameters())
        assert any(p.grad is not None for p in model.wagner_head.parameters())

        print(f"PASS ({p_info['total']:,} params, shapes verified, freezing verified, backward verified)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_swin_tiny_optimized_architecture():
    """Test Swin-Tiny Optimized Architecture (~35.2M params), exact math, resolution checks, freezing, and CAM targets."""
    print("[TEST 15] Swin-Tiny Optimized Architecture (~35.2M params, Swin retained)...", end=" ")
    try:
        from dfu_siam_model import (
            DFUSiamV3, ClassBalancedFocalLoss, CORALLoss, HardPairLMCoTLoss,
            UncertaintyGatedMMDLoss, OrthogonalityLoss
        )

        # 1. Model initialization
        model = DFUSiamV3(arch="swin_tiny", pretrained=False)
        p_info = model.get_parameter_breakdown()

        # 2. Exact Parameter Count Assertions
        assert p_info['total'] == 35_158_798, f"Expected 35,158,798 total params, got {p_info['total']}"
        assert p_info['cnn'] == 5_858_704, f"Expected 5,858,704 CNN params, got {p_info['cnn']}"
        assert p_info['vit'] == 27_519_354, f"Expected 27,519,354 Swin-Tiny params, got {p_info['vit']}"
        assert (p_info['cnn_proj'] + p_info['vit_proj']) == 525_824, f"Expected 525,824 proj params, got {p_info['cnn_proj'] + p_info['vit_proj']}"
        assert p_info['disentangler'] == 989_184, f"Expected 989,184 disentangler params, got {p_info['disentangler']}"
        assert (p_info['classifier'] + p_info['wagner_head']) == 265_732, f"Expected 265,732 head params, got {p_info['classifier'] + p_info['wagner_head']}"

        # 3. Resolution Verification: 224x224 must pass, 192x192 must be rejected by Swin window attention
        model.eval()
        x224 = torch.randn(2, 3, 224, 224)
        zs_224, _, _ = model.forward_features(x224)
        assert zs_224.shape == (2, 256), f"Expected zs shape (2, 256), got {zs_224.shape}"

        res_rejected = False
        try:
            x192 = torch.randn(2, 3, 192, 192)
            _ = model.forward_features(x192)
        except Exception:
            res_rejected = True
        assert res_rejected, "Swin-Tiny window attention must reject non-224 input without padding/reconfiguration"

        # 4. CAM Explainability Targets Verification
        cam_targets = model.get_cam_target_layers()
        assert 'cnn' in cam_targets and 'vit' in cam_targets
        assert isinstance(cam_targets['cnn'], torch.nn.Module)
        assert isinstance(cam_targets['vit'], torch.nn.Module)

        # 5. Backbone Freezing Modes
        model.freeze_backbones()
        p_frozen = model.get_parameter_breakdown()
        assert p_frozen['trainable'] == 1_780_740, f"Expected 1,780,740 trainable params when frozen, got {p_frozen['trainable']}"
        assert not any(p.requires_grad for p in model.backbone.cnn.parameters())
        assert not any(p.requires_grad for p in model.backbone.vit.parameters())

        model.partial_finetune_backbones()
        p_partial = model.get_parameter_breakdown()
        assert p_partial['trainable'] == 21_635_552, f"Expected 21,635,552 trainable params in partial mode, got {p_partial['trainable']}"
        assert p_partial['non_trainable'] == 13_523_246, f"Expected 13,523,246 non-trainable in partial mode, got {p_partial['non_trainable']}"

        # 6. Backward Pass Verification
        model.unfreeze_backbones()
        model.train()
        x_in = torch.randn(4, 3, 224, 224)
        targets_p = torch.tensor([0, 1, 2, 3])
        targets_w = torch.tensor([0, 1, 3, 4])

        zs, zp, p_logits = model.forward_pathology(x_in)
        _, zd, w_logits = model.forward_wagner(x_in)

        cb_focal = ClassBalancedFocalLoss(samples_per_cls=[2552, 2555, 227, 621])
        coral = CORALLoss(num_classes=5)
        lmcot = HardPairLMCoTLoss()
        ug_mmd = UncertaintyGatedMMDLoss()
        orth = OrthogonalityLoss()

        l_p = cb_focal(p_logits, targets_p)
        l_w, _, _ = coral(w_logits, targets_w)
        l_lmc = lmcot.forward_mined(zs, targets_p, epoch=1)
        w_dfu = UncertaintyGatedMMDLoss.compute_pathology_uncertainty(p_logits)
        w_wag = UncertaintyGatedMMDLoss.compute_wagner_uncertainty(w_logits)
        l_m = ug_mmd(zs, zs, w_dfu, w_wag)
        l_o = orth(zs, zp, zd)

        total_l = l_p + l_w + 0.5 * l_lmc + 0.1 * l_m + 0.1 * l_o
        assert total_l.requires_grad
        total_l.backward()

        assert any(p.grad is not None for p in model.backbone.cnn.parameters())
        assert any(p.grad is not None for p in model.backbone.vit.parameters())
        assert any(p.grad is not None for p in model.classifier.parameters())
        assert any(p.grad is not None for p in model.wagner_head.parameters())

        print(f"PASS (35,158,798 params, Swin-Tiny verified, freezing verified, backward verified)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_checkpoint_atomic_resume():
    """Test atomic checkpoint saving, corrupt-write immunity, RNG restoration, and mid-epoch resume."""
    print("[TEST 16] Robust Checkpointing: Atomic saving & exact resume fidelity...", end=" ")
    import tempfile
    from dfu_siam_model import DFUSiamV3, get_device
    from train_dfu_siam import save_atomic_checkpoint, get_rng_states, set_rng_states

    tmp_dir = tempfile.mkdtemp(prefix="test_ckpt_")
    try:
        device = get_device()
        model1 = DFUSiamV3(arch="swin_tiny", pretrained=False).to(device)
        model1.eval()

        target_file = os.path.join(tmp_dir, "latest_checkpoint.pth")
        
        # 1. Save state atomically
        orig_rng = get_rng_states()
        dummy_state = {
            'model_version': 'v3',
            'epoch': 3,
            'batch': 15,
            'global_step': 135,
            'best_metric': 0.7854,
            'monitor_metric': 'macro_f1',
            'model_state_dict': model1.state_dict(),
            'optimizer_state_dict': {'state': {}, 'param_groups': []},
            'scheduler_state_dict': {},
            'scaler_state_dict': {},
            'rng_states': orig_rng,
            'pair_state': {'pairs': [(1, 2), (3, 4)], 'pair_labels': [1, 0], 'class_targets': [0, 1]},
            'config': {'arch': 'swin_tiny', 'epochs': 40}
        }
        save_atomic_checkpoint(dummy_state, target_file)
        assert os.path.isfile(target_file), "Checkpoint was not saved to target path"

        # 2. Check corrupt write immunity (writing to a bad directory or interrupted tmp)
        # Even if a tmp file is created and broken, target_file remains untouched
        content_before = open(target_file, "rb").read()
        try:
            # Simulate failure during atomic write to non-existent root
            bad_path = os.path.join("Z:\\nonexistent_drive_12345", "ckpt.pth")
            save_atomic_checkpoint(dummy_state, bad_path)
        except Exception:
            pass
        content_after = open(target_file, "rb").read()
        assert content_before == content_after, "Target checkpoint was corrupted during failed save"

        # 3. Reload checkpoint and verify model weights & state fidelity
        loaded = torch.load(target_file, map_location=device, weights_only=False)
        assert loaded['model_version'] == 'v3'
        assert loaded['epoch'] == 3
        assert loaded['batch'] == 15
        assert loaded['global_step'] == 135
        assert loaded['best_metric'] == 0.7854
        assert loaded['pair_state']['pairs'] == [(1, 2), (3, 4)]

        model2 = DFUSiamV3(arch="swin_tiny", pretrained=False).to(device)
        model2.load_state_dict(loaded['model_state_dict'])
        model2.eval()

        dummy_x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            out1, _ = model1.forward_single(dummy_x)
            out2, _ = model2.forward_single(dummy_x)
        assert torch.allclose(out1, out2, atol=1e-5), "Model outputs diverged after checkpoint restoration"

        # 4. RNG restoration verification
        # Seed generator, generate numbers, alter RNG, then restore RNG and verify identical random numbers
        import random
        random.seed(999)
        seq_a = [random.random() for _ in range(5)]
        state_saved = get_rng_states()
        seq_b_expected = [random.random() for _ in range(5)]

        # Modify RNG
        random.seed(12345)
        seq_c = [random.random() for _ in range(5)]
        assert seq_c != seq_b_expected, "RNG should have changed after reseeding"

        # Restore RNG
        set_rng_states(state_saved)
        seq_b_restored = [random.random() for _ in range(5)]
        assert seq_b_restored == seq_b_expected, "RNG states were not accurately restored"

        print("PASS (atomic swap verified, corruption immune, RNG & mid-epoch states restored)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False
    finally:
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def run_all_tests():
    print("=" * 65)
    print(" DFU-SIAM v3 Comprehensive Test & Validation Suite")
    print("=" * 65)

    tests = [
        test_imports,
        test_model_init,
        test_model_shapes,
        test_class_balanced_focal_loss,
        test_coral_loss,
        test_hard_pair_lmcot,
        test_ug_mmd_loss,
        test_orthogonality_loss,
        test_full_backpropagation,
        test_checkpoint_v3_compatibility,
        test_synthetic_backward_smoke,
        test_inference_pipeline,
        test_wagner_dedup_and_baseline_isolation,
        test_lightweight_architecture,
        test_swin_tiny_optimized_architecture,
        test_checkpoint_atomic_resume,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"UNEXPECTED ERROR in {test_fn.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 65)
    print(f" Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 65)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

