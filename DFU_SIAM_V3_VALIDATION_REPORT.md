# DFU-SIAM v3: Codebase Validation & Integrity Report

## 1. Verification Status Summary

| Check Category | Verification Item | Status | Result / Measurement |
| :--- | :--- | :---: | :--- |
| **Static / Imports** | Module loading (`dfu_siam_model`, `dataset`, etc.) | **PASSED** | Clean imports across all project modules. |
| **Architecture** | Total and trainable parameter count | **PASSED** | **59,783,374** parameters (100% trainable). |
| **Tensor Shapes** | Representation dimensions | **PASSED** | $z_{\text{shared}} \in \mathbb{R}^{768}$, $z_{\text{private}} \in \mathbb{R}^{512}$, Logits $\in \mathbb{R}^{4}$. |
| **Supervised Loss** | Class-Balanced Focal Loss | **PASSED** | Minority weight: 2.13 > Majority: 0.47. |
| **Ordinal Loss** | CORAL Cumulative Ordinal Loss | **PASSED** | Cutoffs validated across Grades 0 to 4. |
| **LMCoT Metric** | 12-Case Numerical & Boundary Battery | **PASSED** | Max observed loss = 99.9999 (bounded at $d_{\max}^2=100.0$). Stable gradients under FP32 and AMP. |
| **Domain Alignment**| Uncertainty-Gated MMD (detached weights) | **PASSED** | Identical distributions yield 0.0 loss; confidence weights verified non-differentiable. |
| **Disentanglement** | Orthogonality Loss | **PASSED** | Orthogonal vectors yield 0.0 loss; parallel vectors yield 1.0 penalty. |
| **Autograd Reach** | Full multi-task backward reach | **PASSED** | Every CNN, ViT, projection, disentanglement, and head parameter received non-zero finite gradients. |
| **Checkpoints** | Version namespace and legacy rejection | **PASSED** | Strict `v3` verified; legacy v1/v2 checkpoints safely rejected. |
| **Synthetic Smoke Test** | Synthetic forward/backward autograd reach | **PASSED** | Finite loss (2.6874); gradients confirmed reaching all parameters without optimizer updates. |
| **Inference Pipeline**| Test-time augmentation & UT grid decoding | **PASSED** | 16-cell grid probabilities verified ($\sum = 1.0$). |
| **Training Pipeline**| Full multi-epoch training | **NOT STARTED** | **TRAINING NOT STARTED** (Rule enforced). |

---

## 1.1 Historical Investigation: Real-Data Smoke Step in Previous Audit
In response to Issue 1, a comprehensive inspection of code, logs, and disk checkpoints was conducted:
1. **Historical Execution**: An earlier version of `test_all.py` contained `test_real_data_smoke()`, which loaded 2 samples from `DFUC2021_train` and 2 samples from `WagnerDataset/train` to verify that dataset loaders and loss forward passes functioned on real image formats. That test instantiated an ephemeral, untrained `model = DFUSiamV3(pretrained=False)` and executed an in-memory `optimizer.step()`.
2. **No Checkpoint Saved / No Weights Modified**: File system inspection verifies that **zero** `.pth` checkpoints were written or modified during that execution. All existing model checkpoints on disk bear older timestamps (e.g. `checkpoints/dfu_siam_best.pth` from earlier baseline experiments). The in-memory model was immediately garbage collected when the test exited.
3. **Rectification**: To ensure absolute methodological consistency with the rule that no real-data optimization steps occur before full training, `test_all.py` Test 11 has been converted to `test_synthetic_backward_smoke()`. It generates synthetic tensors in memory and verifies forward and backward autograd reach without touching real datasets or calling `optimizer.step()`. Real data remains 100% pristine.

---

## 2. LMCoT Stabilization Audit Findings

### 2.1 Mathematical Analysis of Distance Ceiling & Gradient Behavior
1. **Geometric Formulation**:
   $$D(z_i, z_j) = \tan(\theta/2), \quad \theta = \arccos\left(\text{clamp}(z_i^\top z_j, -1+\epsilon, 1-\epsilon)\right)$$
2. **Ceiling Parameterization ($d_{\max} = 10.0$)**:
   $$\text{half\_angle} = \text{clamp}\left(\frac{\theta}{2}, \epsilon, \arctan(d_{\max})\right)$$
   - $\arctan(10.0) \approx 1.4711\text{ rad} = 84.29^\circ$.
   - The corresponding threshold angle is $\theta_{\text{thresh}} = 2 \times 84.29^\circ = 168.58^\circ$.
   - In cosine space: $\cos(\theta_{\text{thresh}}) = \cos(2 \arctan(10)) = \frac{1 - 10^2}{1 + 10^2} = -\frac{99}{101} \approx -0.9802$.
   - **Cap Activation Frequency**: In high-dimensional hyperspheres ($D \ge 64$), random vectors concentrate orthogonally around $\cos \approx 0$. Over $10,000$ synthetic random unit vector pairs, the ceiling activation frequency is exactly **$0.00\%$**. The ceiling activates only when representations are nearly diametrically opposed ($\theta \ge 168.6^\circ$).
3. **Gradient Continuity & Saturation**:
   - For $\theta < 168.58^\circ$, gradients propagate with standard cotangent metric dynamics:
     $$\frac{\partial D}{\partial \theta} = \frac{1}{2} \sec^2(\theta/2) = \frac{1 + D^2}{2}$$
   - For $\theta \ge 168.58^\circ$, the clamp holds distance constant at $10.0$, saturating the gradient to $0.0$ for that specific pair. This acts as a mathematical gradient ceiling, preventing gradient explosion without introducing NaN or infinite values.
4. **Distinction Between Numerical Verification & Scientific Validation**:
   - Passing synthetic numerical boundary tests confirms that the implementation does not crash, overflow, or emit NaN/Inf gradients.
   - **It does NOT constitute proof of clinical superiority or scientific novelty**. The actual generalization benefit of the cotangent metric compared to standard Euclidean contrastive loss or cosine angular margins must be empirically evaluated through the planned ablation experiments (**A1 vs A2 vs A7**).

### Implemented Fixes
1. **Mathematical Distance Ceiling ($d_{\max} = 10.0$)**: The half-angle is clamped to $\text{atan}(d_{\max}) \approx 1.4711$ radians ($< \pi/2 \approx 1.5708$). This bounds the maximum possible pairwise loss strictly to $d_{\max}^2 = 100.0$.
2. **Autocast Disabling with Float32 Execution**: Metric learning operations are wrapped in `torch.amp.autocast(enabled=False)`, with tensors cast to `float32`.
3. **Safe Cosine Clamping ($\epsilon = 10^{-4}$)**: Guarantees $\sqrt{1 - x^2} \ge \sqrt{2 \times 10^{-4}} > 0.014$, completely eliminating autograd singularities.

### 12-Case Test Battery Results
1. **Identical embeddings**: loss $= 0.000050$, finite gradient = True
2. **Near-identical embeddings**: loss $= 0.000050$, finite gradient = True
3. **Orthogonal embeddings**: loss $= 1.000000$, finite gradient = True
4. **Opposite embeddings**: loss $= 0.000000$ (neg pair with distance $> m$), finite gradient = True
5. **Random normalized embeddings**: loss $= 1.036754$, finite gradient = True
6. **Batch with singleton class**: loss $= 1.378515$, finite gradient = True
7. **Batch with no valid positive pairs**: loss $= 1.156598$, finite gradient = True
8. **Batch with no valid negative pairs**: loss $= 1.201471$, finite gradient = True
9. **Hard-pair mining**: loss $= 1.223282$, finite gradient = True
10. **Float32 precision pass**: loss $= 1.076987$, finite gradient = True
11. **AMP mixed-precision pass**: loss $= 1.205267$, finite gradient = True
12. **All gradients finite and bounded**: max observed loss $= 99.9999 \le 100.0$, all finite = True

---

## 3. Dataset Integrity & Split Audit

### DFUC2021 Pathology Dataset
- Total rows in `train.csv`: **9,949**
- Rows with all class labels NaN (unlabelled): **3,994** (dropped by loader)
- Clean labeled samples: **5,955**
- Class distribution:
  - `none`: 2,552 samples (42.85%)
  - `infection`: 2,555 samples (42.91%)
  - `ischaemia`: 227 samples (3.81%) — severe minority
  - `both`: 621 samples (10.43%) — moderate minority
- Duplicates in image filenames: **0**
- Train/Val split: 80% train (4,764 images), 20% validation (1,191 images), stratified by pathology class.

### Wagner Depth Dataset
- `train/`: **11,639** images (Grade 0: 2,000, Grade 1: 2,240, Grade 2: 2,345, Grade 3: 2,744, Grade 4: 2,310)
- `valid/`: **303** images (Grade 0: 21, Grade 1: 86, Grade 2: 78, Grade 3: 40, Grade 4: 78)
- `test/`: **157** images (Grade 0: 16, Grade 1: 44, Grade 2: 37, Grade 3: 18, Grade 4: 42)

### Methodological Boundaries & Limitations
1. **Missing Wagner Patient Metadata**: Patient IDs are unavailable in `WagnerDataset`. Cross-validation on Wagner relies on image-level separation.
2. **Disjoint Cohort Modalities**: DFUC2021 contains pathology annotations only; WagnerDataset contains depth grades only. No single patient image possesses paired ground-truth for both labels simultaneously.
3. **Exploratory UT Grid**: The 4×4 University of Texas grid synthesized in `fusion_eval.py` is an exploratory probabilistic joint distribution, not a ground-truth paired classifier.
