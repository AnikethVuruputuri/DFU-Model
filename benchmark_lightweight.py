"""
Empirical Benchmark: DFU-SIAM v3 Full Architecture vs Lightweight Architecture.
Measures:
- Parameter count and memory footprint (FP32 & FP16)
- GPU Forward + Backward latency (ms/batch) with PyTorch AMP GradScaler
- Throughput (samples / sec)
- Peak GPU VRAM allocated (MB)
- Extrapolated epoch duration and full 40-epoch training duration on RTX 3050 Laptop GPU (6GB).
"""

import time
import torch
import torch.nn as nn
from dfu_siam_model import (
    DFUSiamV3, ClassBalancedFocalLoss, CORALLoss, HardPairLMCoTLoss,
    UncertaintyGatedMMDLoss, OrthogonalityLoss, get_device
)

def benchmark_variant(name, arch, freeze_mode=None, batch_size=8, num_warmup=5, num_steps=20):
    device = get_device()
    print(f"\n=======================================================", flush=True)
    print(f" Benchmarking: {name} (Device: {device})", flush=True)
    print(f"=======================================================", flush=True)

    # Initialize model
    model = DFUSiamV3(arch=arch, pretrained=False).to(device)
    if freeze_mode == 'frozen':
        model.freeze_backbones()
    elif freeze_mode == 'partial':
        model.partial_finetune_backbones()

    p_info = model.get_parameter_breakdown()
    print(f" Total Parameters      : {p_info['total']:,}", flush=True)
    print(f" Trainable Parameters  : {p_info['trainable']:,}", flush=True)
    print(f" Non-trainable Params  : {p_info['non_trainable']:,}", flush=True)
    print(f" FP32 Size             : {p_info['fp32_size_mb']:.2f} MB", flush=True)
    print(f" FP16 Size             : {p_info['fp16_size_mb']:.2f} MB", flush=True)

    # Criteria
    cb_focal = ClassBalancedFocalLoss(samples_per_cls=[2552, 2555, 227, 621]).to(device)
    coral = CORALLoss(num_classes=5).to(device)
    hard_lmcot = HardPairLMCoTLoss(margin=2.0).to(device)
    ug_mmd = UncertaintyGatedMMDLoss().to(device)
    orth = OrthogonalityLoss().to(device)

    # Optimizer & AMP
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'), init_scale=1024.0)

    # Reset peak memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model.train()

    # Pre-generate synthetic batch
    img1 = torch.randn(batch_size, 3, 224, 224, device=device)
    img2 = torch.randn(batch_size, 3, 224, 224, device=device)
    w_img = torch.randn(batch_size, 3, 224, 224, device=device)
    c1 = torch.randint(0, 4, (batch_size,), device=device)
    c2 = torch.randint(0, 4, (batch_size,), device=device)
    w_lbl = torch.randint(0, 5, (batch_size,), device=device)

    # Warmup
    for _ in range(num_warmup):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            z_s1, z_p1, log1 = model.forward_pathology(img1)
            z_s2, z_p2, log2 = model.forward_pathology(img2)
            z_sw, z_dw, w_log = model.forward_wagner(w_img)

            l_path = 0.5 * (cb_focal(log1, c1) + cb_focal(log2, c2))
            l_coral, _, _ = coral(w_log, w_lbl)
            all_zs = torch.cat([z_s1, z_s2], dim=0)
            all_c = torch.cat([c1, c2], dim=0)
            l_lmcot = hard_lmcot.forward_mined(all_zs, all_c, epoch=1)
            w_dfu = UncertaintyGatedMMDLoss.compute_pathology_uncertainty(log1)
            w_wag = UncertaintyGatedMMDLoss.compute_wagner_uncertainty(w_log)
            l_mmd = ug_mmd(z_s1, z_sw, w_dfu, w_wag)
            l_orth = 0.5 * (orth(z_s1, z_p1) + orth(z_sw, z_depth_private=z_dw))

            loss = l_path + l_coral + 0.5 * l_lmcot + 0.1 * l_mmd + 0.1 * l_orth

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark loop
    start_time = time.perf_counter()
    for _ in range(num_steps):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            z_s1, z_p1, log1 = model.forward_pathology(img1)
            z_s2, z_p2, log2 = model.forward_pathology(img2)
            z_sw, z_dw, w_log = model.forward_wagner(w_img)

            l_path = 0.5 * (cb_focal(log1, c1) + cb_focal(log2, c2))
            l_coral, _, _ = coral(w_log, w_lbl)
            all_zs = torch.cat([z_s1, z_s2], dim=0)
            all_c = torch.cat([c1, c2], dim=0)
            l_lmcot = hard_lmcot.forward_mined(all_zs, all_c, epoch=1)
            w_dfu = UncertaintyGatedMMDLoss.compute_pathology_uncertainty(log1)
            w_wag = UncertaintyGatedMMDLoss.compute_wagner_uncertainty(w_log)
            l_mmd = ug_mmd(z_s1, z_sw, w_dfu, w_wag)
            l_orth = 0.5 * (orth(z_s1, z_p1) + orth(z_sw, z_depth_private=z_dw))

            loss = l_path + l_coral + 0.5 * l_lmcot + 0.1 * l_mmd + 0.1 * l_orth

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    avg_step_ms = (elapsed / num_steps) * 1000.0
    # In each step, we process 3 * batch_size images (img1, img2, w_img)
    total_imgs = num_steps * batch_size * 3
    throughput_fps = total_imgs / elapsed

    peak_vram_mb = 0.0
    alloc_vram_mb = 0.0
    if device.type == 'cuda':
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        alloc_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    # Extrapolations: 1 epoch = 1500 interleaved steps (12,000 DFUC pairs / 8)
    batches_per_epoch = 1500
    epoch_sec = (avg_step_ms / 1000.0) * batches_per_epoch
    epoch_min = epoch_sec / 60.0
    total_40ep_hrs = (epoch_sec * 40) / 3600.0

    print(f" Latency per Step      : {avg_step_ms:.2f} ms", flush=True)
    print(f" Throughput            : {throughput_fps:.1f} images/sec", flush=True)
    print(f" Peak GPU VRAM         : {peak_vram_mb:.1f} MB (Allocated: {alloc_vram_mb:.1f} MB)", flush=True)
    print(f" Estimated 1 Epoch     : {epoch_min:.2f} minutes ({epoch_sec:.1f} s)", flush=True)
    print(f" Extrapolated 40 Epochs: {total_40ep_hrs:.2f} hours", flush=True)

    return {
        'name': name,
        'arch': arch,
        'freeze_mode': freeze_mode,
        'total_params': p_info['total'],
        'trainable_params': p_info['trainable'],
        'fp32_mb': p_info['fp32_size_mb'],
        'fp16_mb': p_info['fp16_size_mb'],
        'avg_step_ms': avg_step_ms,
        'throughput_fps': throughput_fps,
        'peak_vram_mb': peak_vram_mb,
        'epoch_min': epoch_min,
        'total_40ep_hrs': total_40ep_hrs
    }

if __name__ == '__main__':
    results = []
    # 1. Full Architecture (~59.8M)
    results.append(benchmark_variant("Full DFU-SIAM v3 (59.8M)", arch="full"))
    # 2. Swin-Tiny Optimized (35.2M, Full Fine-Tuning)
    results.append(benchmark_variant("Swin-Tiny Optimized (35.2M, Full)", arch="swin_tiny"))
    # 3. Swin-Tiny Optimized (Partial Fine-Tuning, 21.6M Trainable)
    results.append(benchmark_variant("Swin-Tiny Optimized (Partial 21.6M)", arch="swin_tiny", freeze_mode="partial"))
    # 4. Swin-Tiny Optimized (Frozen Feature Extraction, 1.78M Trainable)
    results.append(benchmark_variant("Swin-Tiny Optimized (Frozen 1.78M)", arch="swin_tiny", freeze_mode="frozen"))
    # 5. MobileViT-S Architecture (~12.5M, reference comparison)
    results.append(benchmark_variant("MobileViT-S Reference (~12.5M)", arch="lightweight"))

    print("\n\n" + "=" * 85, flush=True)
    print(" SUMMARY BENCHMARK COMPARISON TABLE (RTX 3050 Laptop GPU 6GB)", flush=True)
    print("=" * 85, flush=True)
    header = f"{'Configuration':<40} | {'Params':<9} | {'VRAM (MB)':<10} | {'Step (ms)':<9} | {'40-Epoch'}"
    print(header, flush=True)
    print("-" * 85, flush=True)
    for r in results:
        param_str = f"{r['trainable_params']/1e6:.2f}M"
        print(f"{r['name']:<40} | {param_str:<9} | {r['peak_vram_mb']:<10.1f} | {r['avg_step_ms']:<9.1f} | {r['total_40ep_hrs']:.2f} hrs", flush=True)
    print("=" * 85, flush=True)
