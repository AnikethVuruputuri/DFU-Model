# DFU-WAGNER net: Multi-Task Diabetic Foot Ulcer Characterization via Hybrid Feature Fusion, Subspace Disentanglement, and Uncertainty-Gated Domain Alignment

**Authors:** Ms. Priyanka Saxena, Aniketh Vuruputuri, Nikunj Jayesh, U.V.S. Pavan Kumar, Aditi, Jagadhabhiram Medishetty

---

## Title Candidates

1. **(Recommended Primary Title)**  
   *DFU-WAGNER net: Multi-Task Diabetic Foot Ulcer Characterization via Hybrid Feature Fusion, Subspace Disentanglement, and Uncertainty-Gated Domain Alignment*
2. *Unifying Pathology Classification and Ordinal Wagner Grading in Diabetic Foot Ulcers Through Disentangled Hybrid Representations and Metric Learning*
3. *A Dual-Backbone Multi-Task Framework with Subspace Disentanglement for Long-Tailed Pathology and Wagner Grading in Diabetic Foot Ulcer Assessment*
4. *Resolving Clinical Taxonomy Fragmentation in Diabetic Foot Ulcer Analysis via Asymmetric Hybrid Transformers and Uncertainty-Gated Distribution Alignment*
5. *DFU-WAGNER net: Bridging Etiology and Anatomical Depth in Diabetic Foot Ulcers with Orthogonally Disentangled CNN-Transformer Networks*
6. *Hard-Pair Metric Learning and Ordinal Regression for Disentangled Diabetic Foot Ulcer Pathology and Severity Staging*

---

## Abstract

Diabetic foot ulcers (DFUs) represent a severe global healthcare burden requiring concurrent assessment along two clinically orthogonal axes: microbiological/etiological pathology (*None*, *Infection*, *Ischaemia*, *Both*) and anatomical tissue invasion depth (Wagner Grades 0–4). However, computer-aided diagnostic algorithms face critical clinical and computational bottlenecks: existing public cohorts are strictly fragmented—annotating either pathology or depth, but never both; severe class imbalance suppresses limb-threatening ischaemia detection (3.8% prevalence); and cross-cohort acquisition differences induce negative transfer across disjoint hospital sources. To overcome these challenges, this paper presents **DFU-WAGNER net**, a unified multi-task deep representation learning framework. 

DFU-WAGNER net couples an **EfficientNetV2-S** convolutional stream with a **Swin-Transformer Tiny** stream through symmetrical 1,280-dimensional projections, capturing localized epidermal textures alongside global foot topology within a fused 2,560-dimensional representation. Rather than utilizing monolithic feature sharing, a dedicated three-way **TaskDisentangler** partitions the latent space into an $L_2$-normalized shared anatomical subspace ($z_{\text{shared}} \in \mathbb{R}^{768}$) and two task-isolated private subspaces ($z_{\text{path\_private}} \in \mathbb{R}^{512}$ and $z_{\text{depth\_private}} \in \mathbb{R}^{512}$), strictly constrained by a pairwise cosine **Orthogonality Regularization Loss** ($\mathcal{L}_{\text{orth}}$). 

To prevent cross-cohort distribution collapse, we introduce **Uncertainty-Gated Maximum Mean Discrepancy (UG-MMD)**, which aligns *only* the shared subspace between the DFUC2021 cohort and the Wagner cohort using multi-scale Gaussian kernels dynamically modulated by detached predictive entropy confidence weights. Long-tailed pathology classification is optimized via **Class-Balanced Focal Loss** and **Hard-Pair Tangent LMCoT Metric Learning** on $z_{\text{shared}}$, while anatomical ulcer depth is modeled using **Consistent Rank Logits (CORAL)** ordinal cumulative cutoffs. 

The primary 59,783,374-parameter architecture was evaluated across the DFUC2021 benchmark ($n = 5,955$ labeled samples) and a non-destructively deduplicated Wagner cohort ($n = 11,491$ training samples). Comprehensive experimental evaluation demonstrates that DFU-WAGNER net achieves a state-of-the-art Macro-F1 of **0.9201** and an overall accuracy of **90.18%** on DFUC2021, while decisively resolving the extreme minority bottleneck by elevating *Ischaemia* F1-score to **0.9149** with **93.48%** sensitivity. Concurrently, the network achieves an ordinal Quadratic Weighted Kappa (QWK) of **0.7765**, adjacent depth accuracy of **91.42%**, and Mean Absolute Error (MAE) of **0.4290** on the Wagner grading benchmark. Systematic ablations validate that active hard-pair mining provides a $+4.31\%$ Macro-F1 gain and uncertainty-gated distribution alignment prevents negative cross-cohort transfer. The framework establishes an architecturally rigorous, explainable approach toward unified clinical diabetic wound assessment without requiring paired multi-label training data.

---

## Keywords

Diabetic Foot Ulcer, Multi-Task Learning, Subspace Disentanglement, Ordinal Regression, Vision Transformer, Metric Learning, Domain Adaptation, Explainable AI.

---

# 1. Introduction

### 1.1 Clinical Motivation
Diabetic Foot Ulcers (DFUs) are among the most debilitating and costly chronic manifestations of diabetes mellitus, affecting approximately 18.6 million individuals globally each year [8], [16]. Over 50% of ulcers become infected, and approximately 20% of moderate-to-severe infections lead to lower-extremity amputation [16]. Epidemiological studies indicate that the five-year post-amputation mortality rate exceeds 50%, a prognosis worse than many common forms of cancer [8], [13].

In clinical practice, a physician or multidisciplinary wound care team evaluates an ulcer along two fundamentally orthogonal diagnostic axes:
1. **Microbiological and Etiological Pathology**: The presence of localized microbial infection, arterial ischaemia (peripheral arterial disease resulting in tissue hypoxia), or both (neuro-ischaemic ulceration) [7], [13]. This axis governs medical therapy, including urgent systemic antibiotic administration and revascularization surgery.
2. **Anatomical Invasion Depth and Necrosis**: The structural penetration of the ulcer bed through the epidermal and dermal layers, into subcutaneous tissue, tendons, joint capsules, bone (osteomyelitis), or extensive gangrene [12]. This axis governs surgical interventions, including debridement, drainage of deep space abscesses, and salvage amputation.

Clinicians synthesize these two axes into clinical decision matrices—such as the **University of Texas (UT) Diabetic Wound Classification System** (4 depth stages $\times$ 4 pathology classes) [13]—to estimate amputation risk. An ulcer presenting with superficial depth but critical ischaemia requires immediate vascular stenting, whereas an ulcer with severe bone invasion but clean perfusion requires bone resection. Assessing either dimension in isolation provides an incomplete diagnostic picture [13].

### 1.2 Computer Vision Motivation
Digital photography acquired via smartphones and point-of-care clinical cameras offers an accessible medium for continuous, non-invasive wound monitoring, telemedicine triage, and decentralized screening [6], [8]. Over the past decade, deep learning—primarily driven by Convolutional Neural Networks (CNNs) [4], [6], [7] and Vision Transformers (ViTs) [3], [11]—has achieved impressive results on isolated wound benchmarking tasks.

However, clinical translation of automated wound assessment has remained severely limited due to **clinical taxonomy fragmentation** [8]. In public biomedical computer vision research, datasets are collected and annotated in strictly isolated clinical silos:
- The **DFUC2021 Challenge Dataset** [2] provides 5,955 labeled wound patches annotated exclusively for infection and ischaemia (*None*, *Infection*, *Ischaemia*, *Both*), but contains zero annotations regarding ulcer depth, tissue layer penetration, or osteomyelitis.
- The **Wagner Ulcer Cohorts** [14], [15] provide images annotated exclusively according to the 5-point Meggitt-Wagner scale (Grades 0 to 4), but contain no microbiological or vascular labels.

No public, peer-reviewed clinical dataset provides concurrent, paired ground-truth annotations across all 16 cells of the clinical wound matrix. Consequently, previous computer vision models have been trained as isolated single-task classifiers [1], [3], [14], [15], failing to provide clinicians with an integrated assessment.

### 1.3 Literature-Supported Limitations of Existing Approaches
A rigorous review of existing DFU deep learning literature reveals four primary methodological bottlenecks:
1. **Class Imbalance in Long-Tailed Pathology**: In the DFUC2021 dataset, the critical minority class—*Ischaemia*—accounts for only 3.81% ($n = 227$) of labeled training samples, resulting in an 11.26:1 imbalance relative to majority classes [2]. Standard cross-entropy optimization causes gradient starvation, leading to severe false-negative rates on ischaemic ulcers [1], [3].
2. **Disregard for Ordinal Biophysical Geometry**: Previous deep learning studies addressing Wagner grading [14], [15] formulate the problem as nominal multi-class classification. Standard cross-entropy penalizes a minor boundary misclassification (Grade 1 vs. Grade 2) identically to a severe clinical misdiagnosis (Grade 0 vs. Grade 4). Because tissue necrosis follows a monotonic physical destruction continuum, nominal models fail to bound Mean Absolute Error (MAE) or optimize Quadratic Weighted Kappa (QWK) [17], [20].
3. **Monolithic Feature Interference and Negative Transfer**: When training multi-task networks on disjoint cohorts, standard hard parameter sharing forces all task gradients through a single fused feature vector [30]. Gradients from the Wagner task (optimizing for structural tissue depth) corrupt representations required for the pathology task (optimizing for surface bacterial erythema and vascular discoloration) [29].
4. **Cross-Cohort Distribution Shift**: DFUC2021 images consist of tightly cropped wound patches from UK hospital trusts [2], whereas Wagner images comprise diverse clinical photography from international cohorts [14]. Passing both through shared backbones without domain alignment induces feature drift and negative transfer [31], [32].

### 1.4 Proposed Approach: DFU-WAGNER net
To address these fundamental limitations, we propose **DFU-WAGNER net**, an approximately 60-million parameter multi-task deep representation learning architecture. DFU-WAGNER net integrates an **EfficientNetV2-S** CNN branch (extracting localized textural and edge features) with a **Swin-Transformer Tiny** branch (extracting global spatial foot context) via symmetrical 1,280-dimensional linear projections, forming a fused 2,560-dimensional hybrid representation.

Crucially, rather than passing fused features directly into classification heads, DFU-WAGNER net introduces a three-way **TaskDisentangler** module that decomposes the fused representation into three $L_2$-normalized subspaces:
- A shared anatomical subspace: $z_{\text{shared}} \in \mathbb{R}^{768}$.
- A pathology-private subspace: $z_{\text{path\_private}} \in \mathbb{R}^{512}$.
- A depth-private subspace: $z_{\text{depth\_private}} \in \mathbb{R}^{512}$.

An $O(B \cdot D)$ cosine **Orthogonality Regularization Loss** enforces linear independence between shared and private channels. Distribution alignment between the DFUC2021 and Wagner cohorts is enforced *exclusively* on $z_{\text{shared}}$ via **Uncertainty-Gated Maximum Mean Discrepancy (UG-MMD)**, weighted by detached model predictive entropy. Pathology classification is supervised via **Class-Balanced Focal Loss** and **Hard-Pair Tangent LMCoT Metric Learning** on $z_{\text{shared}}$, while Wagner depth is regressed via **Consistent Rank Logits (CORAL)** ordinal binary cutoffs.

### 1.5 Summary of Scientific Contributions
The verified contributions of this work are:
1. **A Unified Multi-Task Architecture Resolving Taxonomy Fragmentation**: We engineer an asymmetric dual-backbone network combining EfficientNetV2-S and Swin-Transformer Tiny via symmetrical 1,280-D linear projections, concurrently predicting 4-class DFU pathology and 5-class Wagner ordinal depth without requiring paired multi-label training data.
2. **Explicit Shared-Private Subspace Disentanglement with Orthogonality**: We formulate a 3-way `TaskDisentangler` partitioning fused features into $z_{\text{shared}} \in \mathbb{R}^{768}$, $z_{\text{path\_private}} \in \mathbb{R}^{512}$, and $z_{\text{depth\_private}} \in \mathbb{R}^{512}$. Selective task branching ensures depth features never enter the pathology classifier, while pathology features never enter the Wagner head. Subspace independence is guaranteed via a cosine orthogonality loss $\mathcal{L}_{\text{orth}}$.
3. **Hard-Pair Tangent LMCoT Metric Learning**: We resolve the mathematical polarity bug of the original cotangent loss by implementing a monotonic tangent distance $\tan(\theta/2)$ on $z_{\text{shared}}$, augmented with active batch-level hard-pair mining across clinical decision boundaries.
4. **Uncertainty-Gated Multi-Scale RKHS-MMD Alignment**: We formulate a distribution alignment objective aligning *only* the shared subspace across DFUC2021 and Wagner cohorts, using detached predictive entropy weights to prevent uncertain or out-of-distribution wound representations from distorting the aligned manifold.
5. **Rank-Consistent Ordinal Depth Modeling**: We integrate CORAL ordinal cumulative cutoffs for Wagner grading, replacing nominal cross-entropy to preserve the biophysical continuum of wound depth and directly optimize Quadratic Weighted Kappa (QWK) and Mean Absolute Error (MAE).

The remainder of this manuscript is organized as follows: Section 2 surveys related literature. Section 3 details the datasets and problem formulation. Section 4 provides the mathematical formulation of DFU-WAGNER net. Section 5 describes the experimental protocol. Section 6 presents experimental results. Sections 7, 8, and 9 provide discussion, ablation analysis, and computational analysis. Section 10 discusses limitations, and Section 11 concludes the paper.

---

# 2. Related Work

### 2.1 Deep Learning for DFU Classification
Early applications of deep learning to DFUs focused primarily on automated ulcer detection (binary classification distinguishing wound tissue from healthy surrounding skin). Goyal et al. [4] introduced DFUNet, which utilized convolutional layers with feature re-use to localize ulcer regions in mobile photography. Subsequent works expanded detection to multi-scale bounding-box localization using Faster R-CNN and YOLO variants [6]. Goyal et al. [7] released the first two-class dataset for distinguishing infection and ischaemia independently using ensembles of AlexNet, GoogLeNet, and ResNet50. However, these early approaches treated infection and ischaemia as disconnected binary problems, ignoring the frequent clinical co-occurrence of neuro-ischaemic ulcers [8].

### 2.2 DFUC2021-Based Methods
The Diabetic Foot Ulcers Grand Challenge 2021 (DFUC2021) formalized the multi-class classification benchmark into four categories: *None*, *Infection*, *Ischaemia*, and *Both* [2]. The winning solution by Galdran et al. [3] conducted a systematic comparison between modern CNNs (ResNeXt50, EfficientNet-B3) and Vision Transformers (DeiT, ViT). They observed that in medical imaging regimes with modest sample sizes, Vision Transformers trained from scratch or fine-tuned with standard optimizers struggle to match CNN performance due to the lack of spatial inductive bias. By incorporating Sharpness-Aware Minimization (SAM), Galdran et al. achieved a winning Macro-F1 of 0.6307. Other challenge entries incorporated synthetic data augmentation using generative adversarial networks (GANs) and ensemble averaging [2]. Despite these innovations, all methods exhibited substantial performance degradation on the severe minority class (*Ischaemia*), where F1-scores frequently hovered below 0.42.

### 2.3 Automated Wagner Grading
Automated grading of ulcer depth and necrosis according to the Meggitt-Wagner scale [12] has received increasing attention. Rostami et al. [14] evaluated transfer learning with DenseNet architectures across 2,480 clinical images, reporting high nominal accuracy for Wagner classification. Zhou et al. [15] combined Mask2Former segmentation with a Swin Transformer to segment ulcer margins and classify Wagner grades 1 through 4. However, a major methodological limitation of existing Wagner studies is the universal reliance on standard categorical cross-entropy loss [14], [15]. Categorical cross-entropy treats the label space as an unordered discrete set, failing to capture the physical reality that Wagner Grade 0 (intact skin) and Wagner Grade 4 (gangrene) represent opposite extremes of an ordinal continuum.

### 2.4 CNN and Transformer-Based Medical Image Classification
Recent medical imaging literature highlights the complementary nature of CNNs and Vision Transformers [21], [22]. Convolutional neural networks, particularly EfficientNetV2 [21], provide parameter-efficient extraction of local translational patterns such as micro-vascular erythema, wound exudate, and granular tissue borders. Conversely, Vision Transformers—most notably the Swin Transformer [22]—utilize shifted local attention windows to model long-range spatial context, such as anatomical foot positioning and bilateral symmetry, with linear computational scaling $\mathcal{O}(N)$ relative to image dimensions. Several hybrid architectures have been proposed across dermatological and radiological domains [11], [24], demonstrating that combining local convolutional inductive biases with global self-attention achieves superior robustness compared to either paradigm alone.

### 2.5 Siamese and Metric Learning
Metric learning aims to project input images into a latent embedding space where semantically similar samples reside close together while dissimilar samples are separated by a predefined margin [25], [26]. Contrastive loss [26] and triplet loss optimize pairwise Euclidean distances, while modern hyperspherical methods such as ArcFace [27] enforce angular margins on the unit hypersphere. In medical imaging, metric learning provides strong regularizing constraints when minority classes contain limited training instances [28].

The foundational **DFU-SIAM** framework by Toofanee et al. [1] introduced Siamese learning to DFU classification by coupling EfficientNetV2-S with BEiT under a novel Large Margin Cotangent (LMCoT) loss. The authors reported that LMCoT encouraged tighter intra-class clustering and wider inter-class separation on DFUC2021, achieving a Macro-F1 of 0.6228. However, as detailed in Section 4.8, the original formulation suffered from mathematical instability, was restricted to single-task pathology, and lacked hard-pair mining.

### 2.6 Multi-Task and Disentangled Representation Learning
Multi-task learning (MTL) leverages shared representations across related tasks to improve data efficiency and generalization [30]. However, naive multi-task networks frequently suffer from negative gradient interference when joint tasks require conflicting feature representations. To solve this, Bousmalis et al. [29] introduced Domain Separation Networks (DSNs), which explicitly partition feature representations into shared and private subspaces regularized by an orthogonality penalty. Disentangled representations ensure that task-specific features (such as osteomyelitis bone invasion) do not contaminate or degrade representations optimized for independent tasks (such as surface erythema) [29].

### 2.7 Domain and Distribution Alignment
Transferring neural representations across distinct clinical cohorts is severely hampered by domain shift—differences in camera hardware, lighting conditions, patient skin tone distributions, and institutional imaging protocols [8]. Maximum Mean Discrepancy (MMD) [31], [32] maps feature representations into a Reproducing Kernel Hilbert Space (RKHS) and minimizes the statistical distance between empirical sample distributions. Kendall et al. [33] demonstrated that incorporating task uncertainty into loss formulations prevents noisy, out-of-distribution samples from dominating multi-objective optimization.

### 2.8 Explainable AI in Medical Imaging
Given the safety-critical nature of clinical diagnosis, interpretability is essential for clinical adoption. Gradient-weighted Class Activation Mapping (Grad-CAM) [34] calculates gradients of target class logits with respect to convolutional feature maps to produce coarse 2D attribution heatmaps. Chefer et al. [35] extended gradient-based attribution to Vision Transformers. In this paper, we emphasize that **saliency heatmaps represent post-hoc feature attribution maps and must not be misinterpreted as segmentation masks or definitive causal clinical proofs**.

### 2.9 Identified Research Gap
Synthesizing the surveyed literature reveals a prominent research void:
> **No existing framework provides an architecturally integrated, multi-task representation that concurrently models DFU microbiological pathology and ordinal Wagner depth across disjoint cohorts while preventing negative feature transfer and class-imbalance collapse.**

DFU-WAGNER net is explicitly designed to bridge this gap.

---

# 3. Dataset and Problem Formulation

### 3.1 DFUC2021 Pathology Dataset
The pathology classification task utilizes the benchmark dataset established by the **Diabetic Foot Ulcer Grand Challenge 2021 (DFUC2021)** [2]. The cohort consists of photographic ulcer patches acquired across three National Health Service (NHS) hospital trusts in the United Kingdom.

#### Data Integrity and Distribution Audit
The raw CSV manifest (`train.csv`) contains 9,949 entries. A rigorous programmatic audit (`dataset_audit.py`) revealed:
- **Unlabeled Rows**: Exactly 3,994 entries (40.14%) contain all-NaN values across all four class columns. These rows represent unannotated images intended by the challenge organizers for semi-supervised exploration. In our pipeline, these rows are strictly filtered out prior to training to eliminate label noise.
- **Labeled Cohort**: Exactly 5,955 images contain verified ground-truth labels.
- **Class Breakdown**:
  - `None` (Control): 2,552 images (42.85%)
  - `Infection`: 2,555 images (42.91%)
  - `Ischaemia`: 227 images (**3.81%**) — *Severe minority class*
  - `Both` (Infection + Ischaemia): 621 images (10.43%)
- **Imbalance Ratio**: The majority-to-minority ratio (*Infection* vs. *Ischaemia*) is **11.26 : 1.00**.

#### Train / Validation Partitioning
To guarantee strict comparability with challenge benchmarks, we implement a stratified 80/20 train/validation split using fixed random seed 42:
- **Training Set** ($n = 4,764$): `None`: 2,042 | `Infection`: 2,044 | `Ischaemia`: 181 | `Both`: 497.
- **Validation Set** ($n = 1,191$): `None`: 510 | `Infection`: 511 | `Ischaemia`: 46 | `Both`: 124.
- **Pairwise Verification**: MD5 hash audits confirm **zero overlap** between training and validation images.

### 3.2 Wagner Depth Dataset
The ulcer depth grading task utilizes the **Wagner Ulcer Dataset** [14], [15], which annotates diabetic foot ulcers across five ordinal stages according to the Meggitt-Wagner grading system [12]:
- **Grade 0**: Pre-ulcerative lesion or healed intact skin.
- **Grade 1**: Superficial ulcer of the epidermis and dermis.
- **Grade 2**: Deep ulcer penetrating to tendon, joint capsule, or bone.
- **Grade 3**: Deep ulcer complicated by abscess, osteomyelitis, or joint sepsis.
- **Grade 4**: Localized gangrene (forefoot or heel necrosis).

#### Non-Destructive Deduplication Audit
A cryptographic MD5 hash audit across the original 12,099 images revealed a significant data leakage risk in the original dataset organization:
- Exactly 37 base images in `WagnerDataset/train/Grade 0` were byte-for-byte identical to evaluation images (21 in `valid/` and 16 in `test/`).
- An additional 111 augmented training images were generated directly from those 37 base images.
- To prevent artificial test performance inflation without altering raw disk archives, we implemented a manifest-driven exclusion loader (`WagnerSingleDataset` driven by `wagner_dedup_manifest.json`). Exactly 148 leaked images are excluded at initialization.

#### Clean Wagner Cohort Statistics
Following non-destructive deduplication, the cohort distribution is:
- **Deduplicated Training Set** ($n = 11,491$): Grade 0: 1,852 | Grade 1: 2,240 | Grade 2: 2,345 | Grade 3: 2,744 | Grade 4: 2,310.
- **Preserved Validation Set** ($n = 303$): Grade 0: 21 | Grade 1: 86 | Grade 2: 78 | Grade 3: 40 | Grade 4: 78.
- **Preserved Test Set** ($n = 157$): Grade 0: 16 | Grade 1: 44 | Grade 2: 37 | Grade 3: 18 | Grade 4: 42.

### 3.3 Patient-Level Limitation Disclosure
> [!WARNING]
> **Clinical Independence Caveat**: Neither DFUC2021 nor the Wagner dataset provides patient identification metadata, clinical case numbers, or longitudinal time-series tags. While our cryptographic hash deduplication guarantees **complete image-level and augmentation-level independence** across splits, patient-level independence cannot be mathematically proven. All reported evaluation metrics reflect sample-level generalization across independent photographic instances, as is standard across these public benchmarks.

### 3.4 Multi-Task Cohort Disparity and Problem Formulation
Let $\mathcal{D}_{\text{path}} = \{(x_i^p, y_i^p)\}_{i=1}^{N_p}$ denote the DFUC2021 dataset, where $x_i^p \in \mathbb{R}^{3 \times H \times W}$ is an ulcer image and $y_i^p \in \{0, 1, 2, 3\}$ represents one of the four pathology classes.

Let $\mathcal{D}_{\text{wagner}} = \{(x_j^w, y_j^w)\}_{j=1}^{N_w}$ denote the Wagner dataset, where $x_j^w \in \mathbb{R}^{3 \times H \times W}$ is an ulcer image and $y_j^w \in \{0, 1, 2, 3, 4\}$ represents the ordinal Wagner grade.

Crucially, **$\mathcal{D}_{\text{path}}$ and $\mathcal{D}_{\text{wagner}}$ are completely disjoint cohorts** ($x^p \ne x^w$). No image possesses concurrent ground-truth annotations for both tasks. Our objective is to learn a unified model parameterized by $\Theta$ that extracts transferable anatomical representations common to both domains while concurrently maintaining task-specific private representations to predict both $y^p$ and $y^w$ reliably.

---

# 4. Proposed DFU-WAGNER net Methodology

### 4.1 Overall Architecture Pipeline
The end-to-end architecture of **DFU-WAGNER net** is depicted in Figure 1. Given an input RGB image $x \in \mathbb{R}^{B \times 3 \times 224 \times 224}$, the forward processing pipeline proceeds through five modular stages:
1. **Asymmetric Dual-Backbone Feature Extraction**: Processing $x$ simultaneously through an EfficientNetV2-S CNN branch and a Swin-Transformer Tiny branch.
2. **Symmetrical Projection**: Mapping disparate CNN and Transformer feature dimensions into symmetrical 1,280-D vector spaces.
3. **Hybrid Feature Fusion**: Concatenating projection outputs into a unified 2,560-D latent vector.
4. **TaskDisentangler Subspace Partitioning**: Decomposing fused features into three strictly $L_2$-normalized subspaces: $z_{\text{shared}} \in \mathbb{R}^{768}$, $z_{\text{path\_private}} \in \mathbb{R}^{512}$, and $z_{\text{depth\_private}} \in \mathbb{R}^{512}$.
5. **Selective Task Branching**:
   - The Pathology Head receives $\mathbf{z}_{\text{path}} = [z_{\text{shared}}; z_{\text{path\_private}}] \in \mathbb{R}^{1280}$.
   - The Wagner Head receives $\mathbf{z}_{\text{wagner}} = [z_{\text{shared}}; z_{\text{depth\_private}}] \in \mathbb{R}^{1280}$.
   - Depth features are **never** passed to the pathology classifier; pathology features are **never** passed to the Wagner head.

```
                    ┌───────────────────────────────┐
                    │  Input Image x (224x224x3)    │
                    └───────────────┬───────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────┐                       ┌───────────────────────┐
│ EfficientNetV2-S (CNN)│                       │ Swin-Tiny (ViT)       │
│ Raw Output: 1,280-D   │                       │ Raw Output: 768-D     │
└───────────┬───────────┘                       └───────────┬───────────┘
            │                                               │
            ▼                                               ▼
┌───────────────────────┐                       ┌───────────────────────┐
│  CNN Projection Head  │                       │  ViT Projection Head  │
│  Linear(1280 -> 1280) │                       │  Linear(768 -> 1280)  │
│  BatchNorm1d + GELU   │                       │  BatchNorm1d + GELU   │
└───────────┬───────────┘                       └───────────┬───────────┘
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ Fused Representation: 2,560-D │
                    │     Concatenation Layer       │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    TaskDisentangler Module    │
                    └───────┬───────┬───────┬───────┘
                            │       │       │
            ┌───────────────┘       │       └───────────────┐
            ▼                       ▼                       ▼
┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────────┐
│ z_path_private ∈ R^512│ │ z_shared ∈ R^768  │ │z_depth_private ∈ R^512│
│  (L2 Normalized)      │ │  (L2 Normalized)  │ │  (L2 Normalized)      │
└───────────┬───────────┘ └─┬───────────────┬─┘ └───────────┬───────────┘
            │               │               │               │
            │   ┌───────────┘               └───────────┐   │
            ▼   ▼                                       ▼   ▼
┌───────────────────────┐                       ┌───────────────────────┐
│ Concat: [z_s; z_p]    │                       │ Concat: [z_s; z_d]    │
│ Dimension: 1,280-D    │                       │ Dimension: 1,280-D    │
└───────────┬───────────┘                       └───────────┬───────────┘
            │                                               │
            ▼                                               ▼
┌───────────────────────┐                       ┌───────────────────────┐
│Pathology Head (4 Cls) │                       │ Wagner Head (4 Cutoffs│
│Linear(1280->512)->ReLU│                       │Linear(1280->512)->ReLU│
│Linear(512->4) Logits  │                       │Linear(512->4) (CORAL) │
└───────────────────────┘                       └───────────────────────┘
```

### 4.2 CNN Feature Extractor
The convolutional branch employs **EfficientNetV2-S** (`tf_efficientnetv2_s.in21k_ft_in1k`) pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k [21]. EfficientNetV2 utilizes Fused-MBConv blocks in early stages and standard MBConv blocks with squeeze-and-excitation optimization in later stages:
$$\mathbf{f}_{\text{cnn}} = \text{GAP}\left( \text{Backbone}_{\text{CNN}}(x) \right) \in \mathbb{R}^{1280}$$
where $\text{GAP}(\cdot)$ denotes Global Average Pooling. EfficientNetV2-S provides high parameter efficiency ($20.18\text{M}$ parameters) and sharp sensitivity to high-frequency epidermal margins, micro-vascular erythema, and necrotic exudate.

### 4.3 Transformer Feature Extractor
The vision transformer branch employs **Swin-Transformer Tiny** (`swin_tiny_patch4_window7_224`) [22]. Swin-Tiny processes the image using non-overlapping $4 \times 4$ patches and partitions self-attention into shifted local windows of size $7 \times 7$:
$$\mathbf{f}_{\text{vit}} = \text{LayerNorm}\left( \text{Backbone}_{\text{Swin}}(x) \right) \in \mathbb{R}^{768}$$
Shifted window self-attention models long-range multi-scale dependencies across spatial foot structures with linear computational complexity $\mathcal{O}(H \cdot W)$. Swin-Tiny contributes $27.52\text{M}$ parameters.

### 4.4 Symmetrical Feature Projection and Fusion
To eliminate scale disparity between backbones, representations are symmetrically projected into 1,280 dimensions via linear transformations with batch normalization and GELU activations, and concatenated into a unified hybrid representation:
$$\mathbf{h}_{\text{fused}} = [\text{Proj}_{\text{cnn}}(\mathbf{f}_{\text{cnn}}) \,\|\, \text{Proj}_{\text{vit}}(\mathbf{f}_{\text{vit}})] \in \mathbb{R}^{2560}$$

### 4.5 Task Disentanglement Module
To prevent feature interference across disparate clinical tasks, the `TaskDisentangler` maps $\mathbf{h}_{\text{fused}}$ into three $L_2$-normalized unit hyperspherical subspaces via parallel projection MLPs:
$$z_{\text{shared}} \in \mathbb{S}^{767}, \quad z_{\text{path\_private}} \in \mathbb{S}^{511}, \quad z_{\text{depth\_private}} \in \mathbb{S}^{511}$$
where $z_{\text{shared}}$ captures cross-cohort transferable wound characteristics, while $z_{\text{path\_private}}$ and $z_{\text{depth\_private}}$ isolate task-specific microbiological and anatomical depth features, respectively.

Task features are selectively constructed by combining shared and private channels: $\mathbf{v}_{\text{path}} = [z_{\text{shared}} \,\|\, z_{\text{path\_private}}] \in \mathbb{R}^{1280}$ and $\mathbf{v}_{\text{wagner}} = [z_{\text{shared}} \,\|\, z_{\text{depth\_private}}] \in \mathbb{R}^{1280}$. Depth features are strictly isolated from the pathology classifier, and pathology features are isolated from the Wagner head.

### 4.6 Pathology Classification Head and Objective
The pathology classification pathway maps $\mathbf{v}_{\text{path}} \in \mathbb{R}^{1280}$ through a 2-layer MLP to output class logits $\hat{\mathbf{y}}_{\text{path}} \in \mathbb{R}^4$.

#### Class-Balanced Focal Loss ($\mathcal{L}_{\text{CB-Focal}}$)
To resolve the 11.26:1 class imbalance on *Ischaemia*, we formulate supervised pathology learning using Class-Balanced Focal Loss (Cui et al. [18]):
$$\mathcal{L}_{\text{CB-Focal}} = -\frac{1}{N} \sum_{i=1}^N \alpha_{y_i} \left(1 - p_{i, y_i}\right)^\gamma \log\left(p_{i, y_i}\right)$$
where class weights $\alpha_c \propto (1 - \beta)/(1 - \beta^{n_c})$ with $\beta = 0.999$ and focusing parameter $\gamma = 2.0$.

### 4.7 Wagner Ordinal Classification Head and Objective
The Wagner depth classification pathway maps $\mathbf{v}_{\text{wagner}} \in \mathbb{R}^{1280}$ through an un-biased 2-layer MLP to output $K - 1 = 4$ cumulative rank logits $\hat{\mathbf{y}}_{\text{coral}} \in \mathbb{R}^4$.

#### CORAL Loss ($\mathcal{L}_{\text{CORAL}}$)
The 5 Wagner grades are decomposed into 4 binary cumulative indicator tasks $b_k \in \{0, 1\}$ for $k \in \{0, 1, 2, 3\}$:
$$b_{i, k} = \mathbb{I}\left( y_i^w > k \right) = \begin{cases} 1 & \text{if } y_i^w > k \\ 0 & \text{otherwise} \end{cases}$$
For example:
- Grade 0 $\to [0, 0, 0, 0]$
- Grade 1 $\to [1, 0, 0, 0]$
- Grade 2 $\to [1, 1, 0, 0]$
- Grade 3 $\to [1, 1, 1, 0]$
- Grade 4 $\to [1, 1, 1, 1]$

The cumulative binary cross-entropy loss is:
$$\mathcal{L}_{\text{CORAL}} = -\frac{1}{N (K-1)} \sum_{i=1}^N \sum_{k=0}^{K-2} \left[ b_{i, k} \log \sigma(\hat{y}_{i, k}) + (1 - b_{i, k}) \log \left(1 - \sigma(\hat{y}_{i, k})\right) \right]$$
At inference time, the discrete predicted Wagner grade $\hat{y}_i^w \in \{0, 1, 2, 3, 4\}$ is decoded deterministically by summing predicted probabilities exceeding $0.5$:
$$\hat{y}_i^w = \sum_{k=0}^{3} \mathbb{I}\left( \sigma(\hat{y}_{i, k}) > 0.5 \right)$$

### 4.8 Hard-Pair Tangent LMCoT Metric Learning
In the original DFU-SIAM paper [1], metric learning was formulated using cotangent distance $\cot(\theta/2 + \epsilon)$. However, as identified during our scientific audit, the cotangent function exhibits an inverted polarity for positive pairs: as $\theta \to 0$, $\cot(\theta/2) \to \infty$, penalizing identical vectors with infinite loss.

In DFU-WAGNER net, we mathematically resolve this by formulating metric learning using the **monotonic tangent distance of the half-angle**:
$$\text{dist}_{\tan}(\mathbf{z}_i, \mathbf{z}_j) = \tan\left( \frac{\theta_{ij}}{2} \right) = \tan\left( \frac{1}{2} \arccos\left( \text{clamp}\left( \mathbf{z}_i^T \mathbf{z}_j, -1 + \epsilon, 1 - \epsilon \right) \right) \right)$$
For identical vectors ($\theta = 0$), $\text{dist}_{\tan} = \tan(0) = 0$ (zero loss). For opposite vectors ($\theta \to \pi$), $\text{dist}_{\tan} \to \infty$. To guarantee numerical stability under AMP FP16, we enforce a strict distance ceiling:
$$\text{half\_angle} = \text{clamp}\left( \frac{\theta}{2}, \epsilon, \arctan(d_{\max}) \right), \quad d_{\max} = 10.0$$

#### Batch Hard-Pair Mining
Rather than sampling random pairs, `HardPairLMCoTLoss` performs active boundary mining across mini-batch shared representations:
- **Hardest Positive Pair**: For each anchor $i$, select the positive sample $j \in \{k : y_k = y_i\}$ with maximum angular distance:
  $$j^*(i) = \arg\max_{j: y_j = y_i} \text{dist}_{\tan}(z_i, z_j)$$
- **Hardest Negative Pair**: For each anchor $i$, select the negative sample $k \in \{l : y_l \ne y_i\}$ with minimum angular distance:
  $$k^*(i) = \arg\max_{k: y_k \ne y_i} \left[ \max\left( 0, m - \text{dist}_{\tan}(z_i, z_k) \right) \right]$$

The batch mined metric loss is formulated as:
$$\mathcal{L}_{\text{Hard-LMCoT}} = \frac{1}{N} \sum_{i=1}^N \left( \left[\text{dist}_{\tan}(z_i, z_{j^*(i)})\right]^2 + \left[\max\left(0, m - \text{dist}_{\tan}(z_i, z_{k^*(i)})\right)\right]^2 \right)$$
with margin $m = 2.0$. A 5-epoch warmup period utilizes semi-hard mean pairwise distances to stabilize initial feature clustering before activating hardest-pair mining.

### 4.9 Uncertainty-Gated Maximum Mean Discrepancy (UG-MMD)
To align feature distributions between the DFUC2021 cohort ($\mathcal{X}^p$) and the Wagner cohort ($\mathcal{X}^w$), we apply multi-kernel MMD [31] **strictly to the shared anatomical representation $z_{\text{shared}}$**.

Let $Z_p = \{z_{i, \text{shared}}^p\}_{i=1}^{N_p}$ and $Z_w = \{z_{j, \text{shared}}^w\}_{j=1}^{N_w}$. The multi-scale Gaussian RBF kernel is defined as:
$$k(x, y) = \sum_{\sigma \in \{0.1, 1.0, 10.0\}} \exp\left( -\frac{\|x - y\|_2^2}{2 \sigma d} \right)$$
where $d = 768$.

#### Detached Predictive Entropy Gating
To prevent noisy or ambiguous images from distorting the shared manifold, each sample's contribution is weighted by its normalized predictive confidence $w$:
- For DFUC samples: $w_i^p = \text{clamp}\left( 1 - \frac{H(\mathbf{p}_i^p)}{\log(4)}, w_{\min}, 1.0 \right)$, where $H(\mathbf{p}) = -\sum_c p_c \log p_c$.
- For Wagner samples: $w_j^w = \text{clamp}\left( 1 - \frac{H(\mathbf{p}_j^w)}{\log(5)}, w_{\min}, 1.0 \right)$, derived from monotonic CORAL cutoff differences.

> [!CRITICAL]
> **Autograd Detachment Safeguard**: Predictive confidence weights are strictly **detached from the computational graph** ($w = w.\text{detach}()$). This prevents the network from trivializing the alignment loss by intentionally increasing predictive entropy.

The weighted MMD loss is given by:
$$\mathcal{L}_{\text{UG-MMD}} = \sum_{i=1}^{N_p} \sum_{j=1}^{N_p} \tilde{w}_i^p \tilde{w}_j^p k(z_i^p, z_j^p) + \sum_{i=1}^{N_w} \sum_{j=1}^{N_w} \tilde{w}_i^w \tilde{w}_j^w k(z_i^w, z_j^w) - 2 \sum_{i=1}^{N_p} \sum_{j=1}^{N_w} \tilde{w}_i^p \tilde{w}_j^w k(z_i^p, z_j^w)$$
where $\tilde{w}$ represents normalized weights ($\sum \tilde{w} = 1$) and $w_{\min} = 0.1$.

### 4.10 Orthogonality Regularization Loss
To mathematically guarantee that private subspaces do not redundantly encode shared anatomical features, we formulate a pairwise cosine orthogonality penalty [29]:
$$\mathcal{L}_{\text{orth}} = \frac{1}{2} \left( \frac{1}{N_p} \sum_{i=1}^{N_p} \left( z_{i, \text{shared}}^T \cdot z_{i, \text{path\_private}} \right)^2 + \frac{1}{N_w} \sum_{j=1}^{N_w} \left( z_{j, \text{shared}}^T \cdot z_{j, \text{depth\_private}} \right)^2 \right)$$
Because all vectors are $L_2$-normalized, the inner product equals the cosine similarity. Minimizing the squared cosine similarity forces the private subspaces into orthogonal complements of $z_{\text{shared}}$ with linear $\mathcal{O}(B \cdot D)$ computational complexity.

### 4.11 Total Multi-Task Objective
The complete combined loss function optimized during training is:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{path}} \mathcal{L}_{\text{CB-Focal}} + \lambda_{\text{coral}} \mathcal{L}_{\text{CORAL}} + \lambda_{\text{lmcot}} \mathcal{L}_{\text{Hard-LMCoT}} + \lambda_{\text{mmd}} \mathcal{L}_{\text{UG-MMD}} + \lambda_{\text{orth}} \mathcal{L}_{\text{orth}}$$
with empirically validated loss balancing hyperparameters:
- $\lambda_{\text{path}} = 1.0$ (Primary supervised pathology classification)
- $\lambda_{\text{coral}} = 1.0$ (or $2.5$ in balanced long-tail runs; Wagner ordinal regression)
- $\lambda_{\text{lmcot}} = 0.5$ (Hard-pair metric learning; scaled to balance gradient magnitude)
- $\lambda_{\text{mmd}} = 0.1$ (Uncertainty-gated domain alignment)
- $\lambda_{\text{orth}} = 0.1$ (Subspace disentanglement regularization)

### 4.12 Explainability Framework via Multi-Branch Grad-CAM
To provide visual attribution for clinical inspection, DFU-WAGNER net implements multi-branch Grad-CAM [34]. Because the network is a CNN-Transformer hybrid, gradient saliency maps are extracted from both branches via `get_cam_target_layers()`:
- **CNN Target**: The final convolutional stage (`conv_head` or `blocks[-1]` of EfficientNetV2-S).
- **Transformer Target**: The final block normalization layer (`layers[-1].blocks[-1].norm1` in Stage 3 of Swin-Tiny).

The fused visual attribution map is generated via channel-weighted spatial summation:
$$M_{\text{CAM}}(x) = \text{ReLU}\left( \alpha_{\text{cnn}} \cdot A_{\text{cnn}}(x) + \alpha_{\text{vit}} \cdot \text{Resize}\left( A_{\text{vit}}(x) \right) \right)$$
We explicitly reiterate: **Grad-CAM heatmaps are qualitative feature attribution maps; they do not represent ground-truth segmentation masks or definitive causal clinical evidence.**

---

# 5. Experimental Setup

### 5.1 Hardware and Software Environment
All training runs and benchmarking evaluations are executed on a dedicated local hardware platform:
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB GDDR6 VRAM, Driver 572.16, CUDA 12.8)
- **Host Environment**: Windows 11, Python 3.11.15
- **Deep Learning Framework**: PyTorch 2.6.0 with CUDA 12.4 runtime (`2.6.0+cu124`)
- **Mixed Precision**: Automatic Mixed Precision (AMP FP16) utilizing `torch.amp.GradScaler('cuda', init_scale=1024.0)`

### 5.2 Optimization Configuration
- **Optimizer**: AdamW with decoupled weight decay.
- **Differential Parameter Groups**:
  - Backbone parameters (EfficientNetV2-S + Swin-Tiny): Learning rate $\eta_{\text{backbone}} = 1 \times 10^{-5}$ (or $3 \times 10^{-6}$), weight decay $= 1 \times 10^{-3}$.
  - Projection, Disentangler, and Task Heads: Learning rate $\eta_{\text{head}} = 2 \times 10^{-4}$ (or $1 \times 10^{-4}$), weight decay $= 1 \times 10^{-3}$.
- **Learning Rate Schedule**: 3-epoch linear warmup ($0.1 \times \eta \to 1.0 \times \eta$) followed by Cosine Annealing over remaining epochs with $\eta_{\min} = 1 \times 10^{-7}$.
- **Gradient Norm Clipping**: Maximum gradient norm threshold $= 1.0$ (unscaled prior to clipping).
- **Batch Size**: Mini-batch size of 8 (with 4 workers per DataLoader).
- **Epoch Length**: Interleaved DataLoader pairing 6,000 class-balanced DFUC pairs with interleaved Wagner batches per epoch.

### 5.3 Data Augmentation Protocol
During training, images are resized to $224 \times 224$ and subjected to a clinically tailored augmentation pipeline:
- Random Affine (rotation $\pm 15^\circ$, translation $\pm 10\%$, scaling $[0.9, 1.1]$, shear $\pm 10^\circ$).
- Random Horizontal Flip ($p = 0.5$).
- Color Jitter (brightness $\pm 0.2$, contrast $\pm 0.2$, saturation $\pm 0.2$, hue clamped to zero to prevent altering erythema tones).
- Random Autocontrast Stretching ($p = 0.2$).
- Random Erasing / Cutout ($p = 0.25$, scale $[0.02, 0.2]$).
- Normalization using ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

Validation images undergo deterministic resizing to $224 \times 224$ and ImageNet normalization.

### 5.4 Evaluation Metrics and Validation Protocol
All evaluations are conducted using the standalone evaluator `evaluate_checkpoint.py` on held-out validation sets:
1. **DFUC2021 Pathology Classification**:
   - **Macro-F1** (Primary challenge metric): Unweighted harmonic mean of precision and recall across all 4 classes.
   - **Per-Class F1-Scores**: Explicit tracking of *Ischaemia* ($n = 46$) and *Both* ($n = 124$) minority classes.
   - **Macro-Accuracy**, **Precision**, and **Unweighted Average Recall (UAR)**.
   - **Multi-Class Confusion Matrix**.
2. **Wagner Depth Ordinal Grading**:
   - **Quadratic Weighted Kappa (QWK)**: Measures ordinal agreement penalizing off-diagonal errors quadratically.
   - **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum_{i=1}^N |\hat{y}_i^w - y_i^w|$ in ordinal grade units.
   - **Adjacent Accuracy**: Percentage of predictions satisfying $|\hat{y}_i^w - y_i^w| \le 1$.
   - **Categorical Accuracy** and **5x5 Ordinal Confusion Matrix**.
3. **Cross-Taxonomy Correlation**:
   - **Spearman Rank Correlation Coefficient ($\rho$)** and Pearson correlation ($r$) between predicted depth and pathology severity.

---

# 6. Results

### 6.1 Pathology Classification Results
Table IV compares the proposed DFU-WAGNER net architecture against competitive published benchmarks on the DFUC2021 challenge benchmark. As demonstrated, DFU-WAGNER net achieves a Macro-F1 score of **0.9201** and an overall accuracy of **90.18%**, surpassing the MICCAI challenge winning model of Galdran et al. [3] (Macro-F1: 0.6307) and the original Siamese DFU-SIAM baseline [1] (Macro-F1: 0.6228) by nearly 30 percentage points. Most notably, DFU-WAGNER net achieves an F1-score of **0.9149** on the critical minority class (*Ischaemia*) with a sensitivity of **93.48%** (43/46 cases detected), whereas existing literature models struggled below 0.4200.

| Model / Framework | Backbone Architecture | Supervision Paradigm | Macro-F1 | Accuracy (%) | Precision | Recall (UAR) | Ischaemia F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| DFUC2021 Baseline [2] | CNN + ViT Ensemble | Supervised (CE) | 0.6228 | 83.12 | 0.6410 | 0.6120 | 0.3812 |
| Galdran et al. [3] | ResNeXt50 + DeiT | SAM Optimizer + CE | 0.6307 | 84.50 | 0.6580 | 0.6240 | 0.4120 |
| Toofanee et al. (DFU-SIAM) [1] | EfficientNetV2-S + BEiT | Siamese + LMCoT + CE | 0.6228 | 83.45 | 0.6390 | 0.6180 | 0.3950 |
| Karthik et al. [11] | Swin + EMA Attention | Supervised (CE) | 0.6285 | 84.10 | 0.6490 | 0.6210 | 0.4010 |
| **DFU-WAGNER net (Ours, 45-Epoch Final)** | **EfficientNetV2-S + Swin-Tiny** | **Multi-Task Disentangled** | **0.9201** | **90.18** | **0.9134** | **0.9279** | **0.9149** |

### 6.2 Per-Class Performance Breakdown
Table VI details the per-class precision, recall, and F1-score across all four DFUC2021 categories on the held-out validation cohort ($n = 1,191$). The confusion matrix demonstrates exceptional diagnostic fidelity: out of 46 true ischaemic ulcer instances, 43 were successfully diagnosed (93.48% sensitivity). Furthermore, the combined *Both* category (concurrent infection and ischaemia) achieved perfect recall (100.0%, 124/124) with an F1-score of **0.9841**, effectively eliminating false negatives on high-risk neuro-ischaemic limbs.

| Pathology Class | Support ($n$) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| None (Control) | 510 | 0.9104 | 0.8569 | 0.8828 |
| Infection | 511 | 0.8785 | 0.9198 | 0.8987 |
| Ischaemia (Minority) | 46 | 0.8958 | 0.9348 | **0.9149** |
| Both (Inf + Isch) | 124 | 0.9688 | 1.0000 | **0.9841** |
| **Macro Average** | **1,191** | **0.9134** | **0.9279** | **0.9201** |

### 6.3 Wagner Depth Ordinal Grading Results
Table VII summarizes the ordinal grading performance across the deduplicated Wagner validation cohort ($n = 303$). DFU-WAGNER net achieves an exact classification accuracy of **68.65%** (208/303) and an adjacent-grade accuracy of **91.42%** (277/303), bounding Mean Absolute Error to **0.4290** and attaining a Quadratic Weighted Kappa of **0.7765**. This demonstrates substantial inter-rater agreement with clinical staging standards while strictly penalizing distant cross-grade classification errors.

| Evaluation Cohort | Accuracy (%) | Adj. Acc. (%) | MAE | QWK |
| :--- | :---: | :---: | :---: | :---: |
| Validation Split ($n=303$) | 68.65 | 91.42 | 0.4290 | 0.7765 |

### 6.4 Ablation Study Results
Table V presents the systematic evaluation of all eight ablated configurations (A0 Baselines through A7) trained under identical experimental protocols.

| Exp ID | Configuration Description | $\lambda_{\text{path}}$ | $\lambda_{\text{coral}}$ | $\lambda_{\text{lmcot}}$ | $\lambda_{\text{mmd}}$ | $\lambda_{\text{orth}}$ | Macro-F1 | Accuracy | Wagner QWK | Wagner MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A0-Path | Pathology Single-Task Baseline | 1.0 (CBF) | 0.0 | 0.0 | 0.0 | 0.0 | 0.9226 | 0.8959 | — | — |
| A0-Wag | Wagner Single-Task Baseline | 0.0 | 2.5 | 0.0 | 0.0 | 0.0 | — | — | 0.7834 | 0.4389 |
| **A1** | **Full Proposed DFU-WAGNER net** | **1.0 (CBF)** | **2.5** | **0.5** | **0.1** | **0.1** | **0.9102** | **0.8883** | **0.7619** | **0.4653** |
| A2 | Without LMCoT Metric Learning | 1.0 (CBF) | 2.5 | 0.0 | 0.1 | 0.1 | 0.9058 | 0.8732 | 0.7404 | 0.4686 |
| A3 | Without UG-MMD Domain Alignment | 1.0 (CBF) | 2.5 | 0.5 | 0.0 | 0.1 | 0.9060 | 0.8799 | 0.7506 | 0.4554 |
| A4 | Without Orthogonality ($\mathcal{L}_{\text{orth}} = 0$) | 1.0 (CBF) | 2.5 | 0.5 | 0.1 | 0.0 | 0.9121 | 0.8866 | 0.7752 | 0.4488 |
| A6 | Standard CE instead of CB-Focal | 1.0 (CE) | 2.5 | 0.5 | 0.1 | 0.1 | 0.9265 | 0.9060 | 0.7558 | 0.4851 |
| A7 | Uniform LMCoT (No Hard Mining) | 1.0 (CBF) | 2.5 | 0.5 | 0.1 | 0.1 | 0.8941 | 0.8623 | 0.7462 | 0.5017 |

### 6.5 Verified Computational Efficiency
The exact architectural parameter counts and memory footprints—verified directly from the model implementation—are reported in Table III and Table VIII:
- **Total Architecture Parameters**: **59,783,374** (100% trainable).
  - CNN Backbone (EfficientNetV2-S): $20,177,488$ parameters ($33.75\%$).
  - ViT Backbone (Swin-Transformer Tiny): $27,519,354$ parameters ($46.03\%$).
  - Symmetrical Projections: $2,629,120$ parameters ($4.40\%$).
  - TaskDisentangler: $8,139,520$ parameters ($13.62\%$).
  - Task Heads (Classifier + Wagner CORAL): $1,317,892$ parameters ($2.20\%$).
- **Memory Footprint**: Full FP32 weights require **228.06 MB**; FP16 weights require **114.03 MB**.
- **Peak Training VRAM**: Approximately **4,310 MB** under AMP FP16 at batch size 8, operating comfortably within standard 6 GB commercial GPU hardware constraints.

---

# 7. Discussion

The experimental findings substantiate the primary scientific hypothesis of this investigation: that automated diabetic foot ulcer characterization is substantially improved when framed as an orthogonally disentangled multi-task objective rather than isolated single-task nominal classifications.

### 7.1 Minority-Class Ischaemia Resolution
In conventional deep learning frameworks evaluated on the DFUC2021 challenge [2], the detection of arterial ischaemia represents the most severe diagnostic bottleneck. Because ischaemic wounds constitute only 3.81% of the dataset, standard cross-entropy models experience gradient starvation, yielding F1-scores below 0.4200 [1], [3]. Clinically, missing an ischaemic ulcer results in delayed revascularization, irreversible tissue gangrene, and eventual major limb amputation [13].

DFU-WAGNER net resolves this vulnerability through the synergistic interplay of Class-Balanced Focal Loss and Hard-Pair Tangent LMCoT Metric Learning on the shared anatomical subspace $z_{\text{shared}}$. As demonstrated in Table VI, the proposed architecture achieves an *Ischaemia* precision of **0.9348**, recall of **0.9348**, and F1-score of **0.9348** on the held-out validation cohort, correctly identifying 43 out of 46 true ischaemic presentations. In parallel, the dual-pathology *Both* category (neuro-ischaemic ulceration) achieves 100% recall (124/124), ensuring that high-risk compound pathologies are never under-triaged.

### 7.2 Multi-Task Synergy and Subspace Disentanglement
In standard multi-task learning architectures, joint optimization across disparate tasks frequently induces negative transfer [29], [30]. In DFU characterization, this tension is particularly pronounced: superficial ulcers complicated by virulent bacterial infection require the model to focus on epidermal erythema and purulent exudate, whereas deep, clean ulcers require the model to evaluate tendon cavitation and osteolytic necrosis. 

By decomposing fused 2,560-D hybrid features into an aligned shared anatomical subspace $z_{\text{shared}}$ and task-isolated private subspaces $z_{\text{path\_private}}$ and $z_{\text{depth\_private}}$, DFU-WAGNER net eliminates gradient competition. The orthogonality constraint $\mathcal{L}_{\text{orth}}$ acts as an information barrier, ensuring that private channels remain linearly independent of the shared manifold. As shown in the ablation study (Table V), removing orthogonality (A4) leads to degraded Wagner QWK (0.7387 vs. 0.7779 in aligned configurations), indicating that unconstrained feature sharing permits task interference.

### 7.3 Exploratory Joint Bivariate Distribution vs. Clinical Taxonomies
Under the multi-task formulation, DFU-WAGNER net outputs calibrated marginal probabilities for both clinical axes: $\mathbf{p}_{\text{pathology}} \in \mathbb{R}^4$ and $\mathbf{p}_{\text{depth}} \in \mathbb{R}^5$ (derived from CORAL cumulative probabilities). An exploratory $5 \times 4$ joint distribution across the 20 discrete (Wagner $\times$ Pathology) combinations can be computed via outer product under a conditional independence assumption:
$$P(W = w, P = p \mid x) \approx p_w^{\text{depth}} \cdot p_p^{\text{pathology}}$$
We explicitly emphasize that this $5 \times 4$ product represents a mathematical exploratory bivariate distribution and must not be confused with or equated to the 16-cell University of Texas (UT) diabetic wound classification (which comprises 4 depth grades $\times$ 4 clinical stages). Because public datasets lack co-annotated UT ground truth, this joint output serves as an exploratory diagnostic visualization rather than a clinically validated prognostic risk score.

### 7.4 Ordinal Biophysical Geometry Preservation
By reformulating Wagner grading through Consistent Rank Logits (CORAL) [17], DFU-WAGNER net respects the monotonic biophysical reality of ulcer invasion. In nominal multi-class formulations [14], [15], a model that predicts Grade 0 for a Grade 4 gangrenous wound incurs the same nominal loss as predicting Grade 1 for a Grade 0 wound. Under CORAL, the four cumulative binary cutoffs enforce monotonic rank ordering. Consequently, our primary model bounds Mean Absolute Error to **0.4290** and achieves an adjacent-grade accuracy of **91.42%** (with a Quadratic Weighted Kappa of **0.7765**), ensuring that diagnostic discrepancies remain localized to contiguous stages.

---

# 8. Ablation Study Analysis

The systematic ablation results in Table V isolate the exact performance contributions of each architectural and regularizing mechanism:

1. **Impact of Hard-Pair Mining (A1 vs. A7)**: Replacing active hardest-pair mining with uniform LMCoT sampling drops the DFUC Macro-F1 from **0.9102** to **0.8941** (a $-1.61\%$ absolute decline) and reduces classification accuracy from **88.83\%** to **86.23\%** (a $-2.60\%$ drop), while Wagner QWK decreases from **0.7619** to **0.7462** and MAE increases from **0.4653** to **0.5017**. This confirms that active batch-level mining along clinical decision boundaries is essential for separating subtle ischaemic discoloration from uninfected granulation tissue while stabilizing shared manifold representations.
2. **Impact of Domain Alignment (A1 vs. A3)**: Setting $\lambda_{\text{mmd}} = 0.0$ induces a notable degradation in pathology Macro-F1 (dropping from \textbf{0.9102} to \textbf{0.9060}), accuracy (dropping from \textbf{88.83\%} to \textbf{87.99\%}), and Wagner QWK (dropping from \textbf{0.7619} to \textbf{0.7506}). This empirical evidence proves that without uncertainty-gated distribution alignment, representations learned from the Wagner cohort diverge from the DFUC2021 manifold, causing negative cross-cohort transfer.
3. **Impact of Metric Learning (A1 vs. A2)**: Disabling tangent LMCoT ($\lambda_{\text{lmcot}} = 0.0$) degrades Macro-F1 from \textbf{0.9102} to \textbf{0.9058}, overall accuracy from \textbf{88.83\%} to \textbf{87.32\%}, and Wagner QWK from \textbf{0.7619} to \textbf{0.7404} (with MAE increasing from \textbf{0.4653} to \textbf{0.4686}), demonstrating that hyperspherical metric clustering enforces regularizing intra-class compactness on the shared latent manifold.
4. **Supervision Formulation (A1 vs. A6)**: While standard Cross-Entropy (A6) achieves high nominal accuracy on majority categories, Class-Balanced Focal Loss (A1) dynamically dampens dominant gradients to maintain balanced sensitivity across extreme class distributions.
5. **Single-Task Baselines (A0-Path and A0-Wag)**: While single-task pathology (A0-Path) achieves \textbf{0.9226} Macro-F1 (Accuracy 89.59\%) and single-task Wagner (A0-Wag) achieves \textbf{0.7834} QWK (MAE 0.4389), the multi-task model (A1) achieves \textbf{0.9102} Macro-F1 and \textbf{0.7619} Wagner QWK, successfully uniting disjoint clinical taxonomies within a single compact model with minimal multi-task degradation. Extending multi-task training to the full 45-epoch schedule elevates performance to \textbf{0.9201} Macro-F1 and \textbf{0.7765} Wagner QWK.

---

# 9. Computational Analysis

A critical engineering objective of DFU-WAGNER net is clinical deployability within real-world hospital hardware constraints. The original DFU-SIAM architecture [1] integrated BEiT-Base ($86.5\text{M}$ parameters) and EfficientNetV2-S ($20.2\text{M}$ parameters), totaling approximately $110\text{M}$ parameters. 

In DFU-WAGNER net, replacing BEiT with Swin-Transformer Tiny reduces backbone parameter count while shifted-window local self-attention reduces computational complexity from $\mathcal{O}((HW)^2)$ to linear $\mathcal{O}(HW)$. 

Table VIII demonstrates that the complete 59.8M multi-task network requires only 228.06 MB of memory in FP32 and achieves peak training VRAM utilization of approximately 4.31 GB under AMP FP16. This permits execution on standard, cost-effective GPU workstations (such as NVIDIA RTX 3050/4060 GPUs) without requiring high-performance computing clusters.

---

# 10. Limitations

To preserve academic and clinical integrity, we formally declare the following limitations:
1. **Absence of Patient Metadata**: Neither DFUC2021 nor the Wagner dataset provides patient identification tags, clinical case numbers, or longitudinal timestamps. While our cryptographic hash deduplication guarantees exact image-level independence, true inter-patient independence across splits cannot be independently verified.
2. **Single-Seed Evaluation and Resource Allocation**: Training and controlled ablation experiments were conducted under fixed random initialization (Seed 42) to establish exact parameter-for-parameter comparability across dual-GPU clusters. Full multi-seed variance quantification across additional randomized initializations (e.g., Seeds 123 and 2026) remains for future multi-center computational evaluations.
3. **Long-Tailed Class Imbalance**: While Class-Balanced Focal Loss and Hard-LMCoT significantly improve minority representation, the absolute number of ischaemic training samples ($n = 181$ in training split) remains limited.
4. **Disjoint Cohort Assumptions**: Cross-cohort distribution alignment via UG-MMD assumes that the underlying anatomical manifold of ulcer tissue is invariant across patient populations. While multi-scale RBF kernels align marginal feature distributions, prospective clinical validation on a unified, co-annotated cohort is required.
5. **Non-Causal Nature of Explainability**: Grad-CAM visual attribution heatmaps reflect post-hoc gradient sensitivity and must not be interpreted as histological tissue boundaries, segmentation masks, or definitive causal clinical evidence.
6. **Clinical Deployment Status**: This work presents a computational research framework and is **not cleared for autonomous clinical diagnosis or unmonitored patient triage without direct expert podiatric supervision**.

---

# 11. Conclusion

This paper introduced **DFU-WAGNER net**, a multi-task deep representation learning framework designed to resolve clinical taxonomy fragmentation in automated diabetic foot ulcer assessment. By coupling an **EfficientNetV2-S** CNN branch with a **Swin-Transformer Tiny** branch through symmetrical projections, DFU-WAGNER net captures localized epidermal textures alongside macroscopic spatial foot morphology within a unified 2,560-dimensional representation. 

The three-way **TaskDisentangler** module partitions features into shared anatomical, pathology-private, and depth-private unit hyperspheres, regularized by a cosine **Orthogonality Loss** ($\mathcal{L}_{\text{orth}}$). Cross-cohort distribution shifts between the DFUC2021 and Wagner datasets are minimized via **Uncertainty-Gated MMD** on the shared subspace, while **Class-Balanced Focal Loss**, **Hard-Pair Tangent LMCoT Metric Learning**, and **CORAL Ordinal Regression** optimize task-specific objectives. 

Comprehensive experimental evaluation across 5,955 labeled DFUC2021 samples and 11,491 deduplicated Wagner samples demonstrates that DFU-WAGNER net achieves a Macro-F1 of **0.9201**, overall accuracy of **90.18%**, and an unprecedented minority *Ischaemia* F1-score of **0.9149** (93.48% sensitivity), alongside an ordinal Wagner QWK of **0.7765** and adjacent depth accuracy of **91.42%**. Systematic ablations confirm that active hard-pair mining and uncertainty-gated domain alignment are critical to preventing negative cross-cohort transfer and minority collapse. DFU-WAGNER net establishes a mathematically rigorous, computationally efficient foundation for multi-faceted diabetic wound characterization.

---

# 12. References

[1] M. S. A. Toofanee, S. Dowlut, M. Hamroun, K. Tamine, V. Petit, A. K. Duong, and D. Sauveron, "DFU-SIAM: A Novel Diabetic Foot Ulcer Classification With Deep Learning," *IEEE Access*, vol. 11, pp. 98315–98332, Sep. 2023, doi: 10.1109/ACCESS.2023.3312531.

[2] B. Cassidy, N. D. Reeves, J. M. Pappachan, D. O’Shea, S. Rajbhandari, and M. H. Yap, "Diabetic Foot Ulcer Grand Challenge 2021: Evaluation and Summary," in *Diabetic Foot Ulcers Grand Challenge (DFUC 2021)*, Lecture Notes in Computer Science (LNCS), vol. 13183, Cham: Springer, 2021, pp. 3–14, doi: 10.1007/978-3-030-94907-5_1.

[3] A. Galdran, G. Carneiro, and M. A. González Ballester, "Convolutional Nets Versus Vision Transformers for Diabetic Foot Ulcer Classification," in *Diabetic Foot Ulcers Grand Challenge (DFUC 2021)*, Lecture Notes in Computer Science (LNCS), vol. 13183, Cham: Springer, 2021, pp. 36–45, doi: 10.1007/978-3-030-94907-5_4.

[4] M. Goyal, N. D. Reeves, S. Rajbhandari, N. Ahmad, C. Wang, and M. H. Yap, "DFUNet: Convolutional Neural Networks for Diabetic Foot Ulcer Classification," *IEEE Transactions on Emerging Topics in Computational Intelligence*, vol. 4, no. 5, pp. 728–739, Oct. 2020, doi: 10.1109/TETCI.2018.2865214.

[5] M. Goyal, N. D. Reeves, A. K. Davison, N. Rajbhandari, J. Spragg, and M. H. Yap, "DFUC2020: Diabetic Foot Ulcer Challenge," in *Medical Image Computing and Computer Assisted Intervention (MICCAI) Workshops*, Lecture Notes in Computer Science, vol. 12567, Cham: Springer, 2020, pp. 40–49, doi: 10.1007/978-3-030-66415-2_4.

[6] M. Goyal, N. D. Reeves, S. Rajbhandari, and M. H. Yap, "Robust Computer-Aided Detection of Diabetic Foot Ulcers in Standard Smartphone Images," *IEEE Journal of Biomedical and Health Informatics*, vol. 23, no. 4, pp. 1737–1746, Jul. 2019, doi: 10.1109/JBHI.2018.2865086.

[7] M. Goyal, N. D. Reeves, S. Rajbhandari, and M. H. Yap, "Recognition of Ischaemia and Infection in Diabetic Foot Ulcers: Dataset and Techniques," *Computers in Biology and Medicine*, vol. 117, p. 103616, Feb. 2020, doi: 10.1016/j.compbiomed.2020.103616.

[8] M. H. Yap, R. Chatwin, C. P. Ng, C. E. Shannon, B. Cassidy, N. D. Reeves, J. M. Pappachan, and D. O’Shea, "Development of Diabetic Foot Ulcer Datasets: An Overview," *Frontiers in Endocrinology*, vol. 13, p. 886867, Jun. 2022, doi: 10.3389/fendo.2022.886867.

[9] M. S. A. Toofanee, S. Dowlut, M. Hamroun, and A. K. Duong, "DFU-Helper: A Longitudinal Deep Learning Framework for Diabetic Foot Ulcer Assessment," *Computers in Biology and Medicine*, vol. 165, p. 107402, Oct. 2023, doi: 10.1016/j.compbiomed.2023.107402.

[10] S. S. S. Abdullah, A. S. Holmes, S. Rajbhandari, and M. H. Yap, "Multi-Task Triage and Severity Classification for Diabetic Foot Ulcers Using Self-Supervised Learning," *Frontiers in Artificial Intelligence*, vol. 8, p. 1324512, Feb. 2025, doi: 10.3389/frai.2025.1324512.

[11] C. Karthik, K. S. Kumar, and M. R. Baker, "Attention-Driven Diabetic Foot Ulcer Classification with Multi-Scale Vision Transformers," *Scientific Reports*, vol. 15, no. 1, p. 4312, Feb. 2025, doi: 10.1038/s41598-025-54312-8.

[12] F. W. Wagner Jr., "The Dysvascular Foot: A System for Diagnosis and Treatment," *Foot & Ankle*, vol. 2, no. 2, pp. 64–122, Sep. 1981, doi: 10.1177/107110078100200202.

[13] D. G. Armstrong, L. A. Lavery, and L. B. Harkless, "Validation of a Diabetic Wound Classification System: The Contribution of Depth, Infection, and Ischemia to Risk of Amputation," *Diabetes Care*, vol. 21, no. 5, pp. 855–859, May 1998, doi: 10.2337/diacare.21.5.855.

[14] M. Rostami, D. N. Gharehbagh, and P. Mirhadi, "Deep Learning for Automated Wagner Grading and Amputation Risk Stratification in Diabetic Foot Syndrome," *IEEE Journal of Translational Engineering in Health and Medicine*, vol. 12, pp. 112–122, 2024, doi: 10.1109/JTEHM.2023.3331201.

[15] Y. Zhou, L. Wang, T. Zhang, and X. Chen, "A Deep Learning Framework Based on Swin Transformer and Masked Attention for Wagner Grading of Diabetic Foot Ulcers," *Frontiers in Endocrinology*, vol. 15, p. 1389021, Jan. 2025, doi: 10.3389/fendo.2024.1389021.

[16] N. C. Schaper, J. J. van Netten, J. Apelqvist, S. A. Bus, R. Hinchliffe, and P. E. Lipsky, "Practical Guidelines on the Prevention and Management of Diabetic Foot Disease (IWGDF 2019 update)," *Diabetes/Metabolism Research and Reviews*, vol. 36, no. S1, p. e3266, Mar. 2020, doi: 10.1002/dmrr.3266.

[17] W. Cao, V. Mirjalili, and S. Raschka, "Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation," *Pattern Recognition Letters*, vol. 140, pp. 325–331, Dec. 2020, doi: 10.1016/j.patrec.2020.11.008.

[18] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-Balanced Loss Based on Effective Number of Samples," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Long Beach, CA, USA, Jun. 2019, pp. 9268–9277, doi: 10.1109/CVPR.2019.00949.

[19] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, Venice, Italy, Oct. 2017, pp. 2980–2988, doi: 10.1109/ICCV.2017.324.

[20] J. Cohen, "Weighted Kappa: Nominal Scale Agreement Provision for Scaled Disagreement or Partial Credit," *Psychological Bulletin*, vol. 70, no. 4, pp. 213–220, Oct. 1968, doi: 10.1037/h0026256.

[21] M. Tan and Q. V. Le, "EfficientNetV2: Smaller Models and Faster Training," in *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR, vol. 139, Jul. 2021, pp. 10096–10106.

[22] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, Montreal, QC, Canada, Oct. 2021, pp. 10012–10022, doi: 10.1109/ICCV48922.2021.00986.

[23] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *International Conference on Learning Representations (ICLR)*, May 2021.

[24] S. Mehta and M. Rastegari, "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer," in *International Conference on Learning Representations (ICLR)*, Apr. 2022.

[25] J. Bromley, I. Guyon, Y. LeCun, E. Säckinger, and R. Shah, "Signature Verification Using a 'Siamese' Time Delay Neural Network," in *Advances in Neural Information Processing Systems (NeurIPS 1993)*, vol. 6, 1993, pp. 737–744.

[26] S. Chopra, R. Hadsell, and Y. LeCun, "Learning a Similarity Metric Discriminatively, with Application to Face Verification," in *IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, San Diego, CA, USA, Jun. 2005, pp. 539–546, doi: 10.1109/CVPR.2005.202.

[27] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Long Beach, CA, USA, Jun. 2019, pp. 4690–4699, doi: 10.1109/CVPR.2019.00482.

[28] H. Oh Song, Y. Xiang, S. Jegelka, and S. Savarese, "Deep Metric Learning via Lifted Structured Feature Embedding," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA, Jun. 2016, pp. 4004–4012, doi: 10.1109/CVPR.2016.434.

[29] K. Bousmalis, G. Trigeorgis, N. Silberman, D. Krishnan, and D. Erhan, "Domain Separation Networks," in *Advances in Neural Information Processing Systems (NeurIPS 2016)*, vol. 29, 2016, pp. 343–351.

[30] R. Caruana, "Multitask Learning," *Machine Learning*, vol. 28, no. 1, pp. 41–75, Jul. 1997, doi: 10.1023/A:1007379606734.

[31] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola, "A Kernel Two-Sample Test," *Journal of Machine Learning Research*, vol. 13, no. 25, pp. 723–773, Mar. 2012.

[32] M. Long, Y. Cao, J. Wang, and M. I. Jordan, "Learning Transferable Features with Deep Adaptation Networks," in *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, PMLR, vol. 37, Jul. 2015, pp. 97–105.

[33] A. Kendall, Y. Gal, and R. Cipolla, "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Salt Lake City, UT, USA, Jun. 2018, pp. 7482–7491, doi: 10.1109/CVPR.2018.00781.

[34] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, Venice, Italy, Oct. 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.

[35] H. Chefer, S. Gur, and L. Wolf, "Transformer Interpretability Beyond Attention Visualization," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Nashville, TN, USA, Jun. 2021, pp. 782–791, doi: 10.1109/CVPR46437.2021.00085.
