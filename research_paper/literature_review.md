# Comprehensive Literature Review and Prior Art Analysis: DFU-WAGNER net

---

## 1. Introduction and Scope

This document provides a systematic scholarly literature review for the **DFU-WAGNER net** research project. The review evaluates contemporary computational methodologies for **Diabetic Foot Ulcer (DFU)** characterization, spanning:
1. Deep learning architectures for microbiological and ischaemic pathology classification.
2. Computational modeling of anatomical penetration and necrosis severity via Wagner grading.
3. Hybrid convolutional and vision transformer representation learning.
4. Metric learning, contrastive geometry, and Siamese networks in medical imaging.
5. Multi-task representation learning and shared-private subspace disentanglement.
6. Domain adaptation and distribution alignment techniques under cross-cohort acquisition variance.
7. Clinical explainability and attribution mechanisms.

Crucially, this review conducts an exhaustive technical examination of the foundational **DFU-SIAM** framework (Toofanee et al., *IEEE Access*, 2023) [1], establishing the precise mathematical and architectural boundaries that differentiate the original work from the proposed **DFU-WAGNER net** framework.

---

## 2. Theoretical and Methodological Foundations

### 2.1 Diabetic Foot Ulcer Pathology Classification
Diabetic foot ulceration is among the most debilitating chronic complications of diabetes mellitus, carrying a five-year mortality rate exceeding 30% that rivals several aggressive malignancies [8], [16]. Clinical management is severely complicated by two distinct microbiological and vascular processes: **infection** (bacterial proliferation in vulnerable ulcerated tissue, precipitating osteomyelitis and sepsis) and **ischaemia** (peripheral arterial occlusive disease inhibiting microvascular perfusion and tissue oxygenation) [7], [13]. 

Historically, computer-aided diagnosis of DFUs relied on handcrafted morphological, textural, and color descriptors (e.g., Haralick features, Local Binary Patterns, color histograms) coupled with classical classifiers such as Support Vector Machines (SVMs) and Random Forests [8]. However, these approaches exhibited brittle generalization across heterogeneous clinical photography environments characterized by variable illumination, camera color calibrations, and skin pigmentation variations. 

The advent of deep convolutional neural networks (CNNs) fundamentally shifted the paradigm. Goyal et al. [4] introduced **DFUNet**, demonstrating that convolutional inductive biases (local translational invariance and hierarchical receptive field expansion) could capture pathognomonic ulcer margins and erythema. Subsequent studies expanded binary detection into multi-class discrimination [7]. Nevertheless, standard CNNs often struggled to capture subtle, non-local vascular changes across the whole foot, motivating the exploration of attention mechanisms and vision transformers [3], [11].

### 2.2 The DFUC2021 Benchmark Challenge
To establish a rigorous public benchmark, Cassidy et al. [2] organized the **Diabetic Foot Ulcer Grand Challenge 2021 (DFUC2021)** in conjunction with MICCAI 2021. The benchmark formalized DFU pathology into a four-class classification taxonomy:
- **None (Control)**: Ulcers devoid of clinical infection or ischaemic compromise ($n = 2,552$ annotated in the training partition, 42.85%).
- **Infection**: Ulcers exhibiting overt purulence, peri-wound erythema, or local heat without arterial insufficiency ($n = 2,555$, 42.91%).
- **Ischaemia**: Ulcers characterized by pale/necrotic wound beds and arterial insufficiency ($n = 227$, 3.81%).
- **Both (Infection + Ischaemia)**: Complex neuro-ischaemic ulcers exhibiting concurrent active infection and critical tissue hypoxia ($n = 621$, 10.43%).

The DFUC2021 benchmark revealed two foundational challenges:
1. **Extreme Class Imbalance**: The ratio between majority classes (*Infection*, $n=2,555$) and the critical minority class (*Ischaemia*, $n=227$) is **11.26 : 1**. Standard cross-entropy minimization causes gradient starvation on ischaemia samples, resulting in frequent false-negative predictions on limb-threatening arterial blockages.
2. **Semi-Supervised/Unlabeled Data Disparity**: Out of 9,949 rows in `train.csv`, exactly 3,994 entries (40.14%) are completely unannotated, necessitating rigorous filtering during data loading to prevent label corruption.

In the challenge proceedings, Galdran et al. [3] secured first place with an ensemble of ResNeXt50 and DeiT optimized via Sharpness-Aware Minimization (SAM), reporting a benchmark Macro-F1 of 0.6307. The organizers' baseline achieved a Macro-F1 of 0.6228 [2]. Subsequent works confirmed that standard single-task classification heads without specialized margin separation or class-balancing loss formulations plateau within the 0.60–0.63 Macro-F1 range.

### 2.3 Wagner Depth and Severity Grading: The Ordinal Dimension
While DFUC2021 formalizes microbiological and vascular state, it omits the vertical anatomical dimension: **ulcer depth and tissue penetration** [8], [12]. Clinicians rely heavily on the **Meggitt-Wagner Classification System** [12], which grades ulcers along an ordinal progression from superficial lesions to extensive gangrene:
- **Grade 0**: Pre- or post-ulcerative lesion, healed wound, or presence of bony deformity.
- **Grade 1**: Superficial ulcer penetrating the dermis without invading underlying subcutaneous tissue.
- **Grade 2**: Deep ulcer penetrating to tendon, joint capsule, or bone without abscess or osteomyelitis.
- **Grade 3**: Deep ulcer complicated by abscess, osteomyelitis, or joint sepsis.
- **Grade 4**: Localized gangrene (forefoot or heel necrosis).
- **Grade 5**: Extensive gangrene involving the entire foot (frequently excluded from image-only datasets due to immediate surgical amputation status).

#### Clinical and Mathematical Rationale for Ordinal Regression
In previous computational attempts, Wagner grading was frequently cast as standard categorical multi-class classification [14], [15]. This constitutes a profound methodological error:
- In nominal multi-class cross-entropy, confusing a Grade 0 (intact skin) with a Grade 4 (gangrene) incurs the **identical loss penalty** as confusing Grade 1 with Grade 2.
- Biophysical tissue necrosis operates along a **monotonic physical continuum**. Mispredicting an ulcer by several depth stages introduces severe clinical risk (under-triaging surgical gangrene as superficial dermis erosion).
- To preserve rank ordering, Cao, Mirjalili, and Raschka [17] introduced **Consistent Rank Logits (CORAL)**. CORAL converts a $K$-class ordinal problem into $K-1$ binary classification subproblems sharing a projection vector $\mathbf{w}$ but independent biases $\{b_k\}_{k=1}^{K-1}$, guaranteeing rank consistency and mathematically bounded Mean Absolute Error (MAE).
- Performance in ordinal medical grading is evaluated not merely via nominal Accuracy, but primarily through **Quadratic Weighted Kappa (QWK)** [20] and **Mean Absolute Error (MAE)**. QWK penalizes off-diagonal classification mistakes quadratically:
  $$\kappa_w = 1 - \frac{\sum_{i=1}^K \sum_{j=1}^K w_{ij} O_{ij}}{\sum_{i=1}^K \sum_{j=1}^K w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(K - 1)^2}$$
  penalizing severe misdiagnoses far more aggressively than adjacent-grade boundaries.

### 2.4 Hybrid CNN-Transformer Architectures in Medical Imaging
Convolutional Neural Networks (e.g., EfficientNetV2 [21]) and Vision Transformers (e.g., Swin Transformer [22], ViT [23]) exhibit complementary inductive biases:
- **CNNs** possess strong **local inductive bias** (translation equivariance and locality), making them exceptionally sensitive to high-frequency textures, granular epidermal boundaries, wound exudate, and localized erythema.
- **Vision Transformers** utilize self-attention mechanisms with quadratic or shifted-window receptive fields, modeling **long-range global contextual dependencies** across anatomical landmarks, spatial foot orientation, and bilateral symmetry.

In medical wound imaging, relying exclusively on either paradigm introduces specific failure modes:
- A pure ViT lacks strong local pixel constraints, requiring tens of millions of images to match CNN data efficiency in low-data regimes [3].
- A pure CNN struggles to correlate localized forefoot ulceration with diffuse plantar ischaemia or foot-wide cellulitis.
- Hybrid architectures (e.g., EfficientNetV2-S coupled with Swin-Transformer Tiny) fuse fine-grained local lesion textures with macroscopic structural topology, yielding superior representations for complex multi-task objectives.

### 2.5 Metric Learning and Siamese Networks
Standard softmax cross-entropy forces neural networks to learn linearly separable hyperplanes partitioned by radial decision boundaries. However, in low-data regimes characterized by extreme class imbalance (such as *Ischaemia* in DFUC2021), the majority classes dominate the hypersphere, collapsing the minority class into an isolated, fragile cluster near the origin [27].

Metric learning techniques address this by optimizing the geometry of the embedding space directly:
- **Contrastive Loss** (Chopra et al. [26]) minimizes the Euclidean distance between positive pairs while pushing negative pairs apart beyond a fixed Euclidean margin $m$.
- **Angular Margin Formulations** (e.g., ArcFace [27]) project embeddings onto a unit hypersphere, penalizing angular geodesic deviation $\theta = \arccos(\mathbf{z}_i^T \mathbf{z}_j)$.
- However, standard cosine distances exhibit vanishing gradients as $\theta \to 0$ or $\theta \to \pi$.
- To address this, Toofanee et al. [1] proposed **Large Margin Cotangent (LMCoT)** loss. As analyzed in Section 3, LMCoT utilizes trigonometric angular transformations to introduce steep, non-vanishing gradient penalties for hard clinical boundaries.

### 2.6 Multi-Task Learning and Representation Disentanglement
Multi-task learning (MTL) [30] aims to improve generalization by jointly optimizing related objectives (e.g., pathology classification and depth grading) over shared feature representations. However, naive hard parameter sharing often induces **negative transfer**:
- Gradients from the Wagner depth task (which tracks necrosis and tissue cavitation) can interfere with gradients from the pathology task (which tracks surface bacterial colonization and vascular skin color).
- To resolve this tension, Bousmalis et al. [29] introduced **Domain Separation Networks (DSNs)**, formalizing the decomposition of latent feature representations into:
  1. A **shared subspace** capturing cross-task or domain-invariant characteristics.
  2. Independent **private subspaces** capturing idiosyncratic, task-specific features.
- Crucially, this decomposition requires an **orthogonality constraint** $\mathcal{L}_{\text{orth}}$ to prevent the private subspaces from redundantly encoding shared signals or collapsing into trivial copies [29].

### 2.7 Domain Adaptation and Uncertainty-Gated Distribution Alignment
A fundamental limitation of public DFU cohorts is that **no single cohort provides concurrent, ground-truth annotations for both DFUC2021 pathology and Wagner depth** [8]. In DFU-WAGNER net, the DFUC2021 cohort and the Wagner dataset originate from entirely distinct medical centers, photography protocols, and demographic cohorts:
- DFUC2021 images consist primarily of closely cropped wound patches.
- The Wagner dataset features diverse clinical photography, full-foot perspectives, and varying ambient hospital lighting.

When passing distinct datasets through a shared representation space, domain shift degrades shared feature quality. **Maximum Mean Discrepancy (MMD)** [31], [32] measures the distance between two probability distributions $P$ and $Q$ mapped into a Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}$:
$$\text{MMD}^2(\mathcal{H}, P, Q) = \left\| \mathbb{E}_{x \sim P}[\phi(x)] - \mathbb{E}_{y \sim Q}[\phi(y)] \right\|_{\mathcal{H}}^2$$
Using multi-scale Gaussian Radial Basis Function (RBF) kernels, MMD aligns the empirical marginal feature distributions of the two cohorts. Furthermore, because out-of-distribution or noisy images can distort feature alignment, **Uncertainty Gating** [33] dynamically scales sample alignment weights using model predictive entropy, ensuring that highly confident representations anchor the shared manifold.

### 2.8 Explainable Artificial Intelligence (XAI)
Deep learning models deployed in healthcare must offer interpretability to clinical stakeholders. **Grad-CAM** (Gradient-weighted Class Activation Mapping) [34] calculates the gradient of the predicted class score $y^c$ with respect to the activation maps $A^k$ of the final feature layer:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{ij}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
In multi-branch hybrid models, target layers must be evaluated across both the CNN backbone (e.g., `conv_head` or final convolutional blocks) and the Transformer backbone (e.g., final stage normalization layers). Crucially, medical AI literature emphasizes that **attribution heatmaps are qualitative feature saliency maps, NOT segmentation masks or causal clinical proofs**; they reflect where the network concentrated gradient sensitivity, which must be cross-verified against anatomical wound margins.

---

## 3. In-Depth Analysis of the Original DFU-SIAM Paper

To prevent conflation and maintain scientific attribution, this section provides an objective, granular breakdown of the foundational paper:

> **Title:** DFU-SIAM: A Novel Diabetic Foot Ulcer Classification With Deep Learning  
> **Authors:** Mohammud Shaad Ally Toofanee, Sabeena Dowlut, Mohamed Hamroun, Karim Tamine, Vincent Petit, Anh Kiet Duong, and Damien Sauveron  
> **Venue:** *IEEE Access*, Volume 11, Pages 98315–98332, Published September 2023  
> **DOI:** `10.1109/ACCESS.2023.3312531` [1]

### 3.1 Original Motivation and Clinical Problem
Toofanee et al. recognized that traditional deep learning classifiers for DFUC2021 suffered from performance degradation primarily due to the severe class imbalance in the dataset (particularly the 3.8% minority prevalence of ischaemia). They posited that rather than treating the problem as standard classification, learning fine-grained pairwise similarity metrics via a Siamese network would allow the model to learn tighter intra-class cluster boundaries and wider inter-class margins.

### 3.2 Original Architecture
The original DFU-SIAM architecture was structured as follows:
- **Dual-Branch Backbone**: An ensemble feature extractor combining:
  1. **EfficientNetV2-S** (pre-trained CNN, extracting 1,280-dimensional feature maps).
  2. **BEiT** (Bidirectional Encoder representation from Image Transformers, extracting 768-dimensional token representations).
- **Concatenation**: CNN and Transformer features were concatenated into a unified latent vector.
- **Classification Head**: An Artificial Neural Network (ANN) classification head taking the fused embedding and outputting 4-class softmax logits (*None*, *Infection*, *Ischaemia*, *Both*).
- **Siamese Configuration**: Two weight-sharing branches fed by image pairs $(x_1, x_2)$ with binary ground truth $y \in \{0, 1\}$ indicating whether both images shared the same pathology class.

### 3.3 Original Loss Formulation: LMCoT
The primary theoretical contribution of the 2023 paper was the **Large Margin Cotangent (LMCoT)** loss. Toofanee et al. observed that cosine margin formulations (e.g., CosFace, ArcFace) are bounded in $[-1, 1]$, which can compress gradient updates near the margin boundary. They proposed mapping the angular difference $\theta = \arccos\left(\frac{\mathbf{z}_1 \cdot \mathbf{z}_2}{\|\mathbf{z}_1\| \|\mathbf{z}_2\|}\right)$ through a cotangent transformation:
$$\text{dist}_{\cot} = \cot\left(\frac{\theta}{2} + \epsilon\right)$$
The Siamese loss was structured to penalize positive pairs based on this distance while enforcing a margin $m$ on negative pairs.

### 3.4 Original Experimental Scope and Reported Results
- **Dataset**: Trained and evaluated **strictly on the DFUC2021 dataset** [2].
- **No Secondary Cohort**: The original paper did **NOT** incorporate the Wagner dataset, Wagner grading, or any secondary clinical cohort.
- **Single-Task Scope**: Addressed only the 4-class DFUC2021 pathology classification. Did not address ulcer depth, necrosis, tissue penetration, or the University of Texas classification matrix.
- **Reported Benchmark Performance**: The authors reported a Macro-F1 score of approximately **0.6228** on the DFUC2021 test benchmark, achieving competitive performance against the challenge baseline.

### 3.5 Limitations of the Original DFU-SIAM Framework
1. **Unimodal Taxonomy**: Limited exclusively to surface microbiological pathology (DFUC2021). It provided zero capability for assessing ulcer invasion depth, osteomyelitis risk, or amputation stage.
2. **Monolithic Feature Representation**: The fused CNN-BEiT representation was passed directly to the classification head without structural task decomposition or subspace disentanglement.
3. **No Domain Alignment**: Because it operated on a single dataset, it possessed no mechanism for cross-domain distribution alignment (e.g., MMD or Optimal Transport).
4. **Uniform Pair Sampling**: Siamese training relied on random or class-proportional pair sampling without active hard positive/negative boundary mining.

---

## 4. Architectural Evolution: DFU-SIAM (2023) vs. DFU-WAGNER net (Proposed)

The following comparative table formally delineates the architectural and mathematical extensions introduced in **DFU-WAGNER net**:

| Dimension | Original DFU-SIAM (Toofanee et al. 2023) [1] | Proposed DFU-WAGNER net (Present Work) | Scientific Rationale for Modification |
| :--- | :--- | :--- | :--- |
| **Backbone 1 (CNN)** | EfficientNetV2-S (1,280-D raw) | **EfficientNetV2-S** (`tf_efficientnetv2_s.in21k_ft_in1k`, 1,280-D raw) | Maintained for proven local receptive field efficiency and fine edge detection. |
| **Backbone 2 (ViT)** | BEiT (768-D raw) | **Swin-Transformer Tiny** (`swin_tiny_patch4_window7_224`, 768-D raw) | Swin's hierarchical shifted-window attention offers superior linear computational scaling and multi-scale visual hierarchy over standard BEiT patch tokens. |
| **Feature Projections** | Asymmetric / Unregularized pass-through | **Symmetrical Projections**: Both branches pass through Linear $\to$ BatchNorm1d $\to$ GELU $\to$ 1,280-D | Prevents feature scale dominance by the CNN branch; enforces uniform regularization. |
| **Clinical Scope** | Single-task: 4-class DFUC2021 Pathology | **Multi-Task**: 4-class DFUC Pathology + 5-class Wagner Ordinal Depth (Grades 0–4) $\to$ Differentiable 16-Cell UT Matrix Synthesis | Resolves clinical taxonomy fragmentation by uniting orthogonal pathology and depth axes. |
| **Latent Feature Space** | Monolithic 2,048-D fused vector | **Task-Disentangled Subspaces**: $z_{\text{shared}} \in \mathbb{R}^{768}$, $z_{\text{path\_private}} \in \mathbb{R}^{512}$, $z_{\text{depth\_private}} \in \mathbb{R}^{512}$ | Prevents task representation interference and negative gradient transfer between microbiological and anatomical depth tasks. |
| **Subspace Regularization** | None | **Orthogonality Loss** ($\mathcal{L}_{\text{orth}}$): Enforces $z_{\text{shared}} \perp z_{\text{path\_private}}$ and $z_{\text{shared}} \perp z_{\text{depth\_private}}$ | Mathematically guarantees linear independence and non-redundancy between shared wound features and task-specific features. |
| **Pathology Loss** | Standard Cross-Entropy + LMCoT | **Class-Balanced Focal Loss** (Cui et al. [18], $\beta=0.999, \gamma=2.0$) | Reweights gradients based on the mathematically effective volume of samples, specifically stabilizing the 3.8% ischaemia minority class. |
| **Wagner Depth Loss** | N/A (Wagner task not present) | **CORAL Ordinal Regression** (Cao et al. [17], 4 binary cumulative cutoffs) | Respects monotonic biological tissue destruction; penalizes large off-diagonal depth errors quadratically. |
| **Metric Learning Metric** | Cotangent formulation $\cot(\theta/2)$ | **Monotonic Tangent Distance** $\tan(\theta/2)$ with Active Hard-Pair Mining | Corrects the positive-pair boundary condition ($\tan(0)=0$) while enforcing steep penalties on hardest positives and closest negative boundaries. |
| **Cross-Cohort Alignment**| None (Single cohort only) | **Uncertainty-Gated Multi-Scale RKHS-MMD** on $z_{\text{shared}}$ with detached confidence weights | Aligns camera, demographic, and hospital distribution shifts across DFUC2021 and Wagner cohorts without corrupting task-private features. |
| **Dataset Utilization** | DFUC2021 only | **DFUC2021 + Wagner Dataset (11,491 deduplicated training images)** | Interleaved dual-cohort training over disjoint clinical databases. |
| **Total Parameter Count** | $\approx 110\text{M}$ (with BEiT-Base) | **59,783,374 Parameters ($\approx 59.8\text{M}$)** | Achieves dual-task disentanglement with greater parameter efficiency and faster inference throughput. |

---

## 5. Comprehensive Literature Review Table

The following table summarizes 22 highly relevant prior studies spanning diabetic foot ulcer classification, DFUC2021, automated Wagner grading, CNN-transformer medical imaging, metric learning, multi-task learning, and domain adaptation.

| Study | Year | Venue / Identifier | Dataset(s) Used | Method / Architecture | Task / Clinical Scope | Main Result Reported | Limitation / Relevance to Our Work |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **Wagner Jr.** [12] | 1981 | *Foot & Ankle*<br>doi: 10.1177/107110078100200202 | Retrospective clinical cohort ($n>500$) | Clinical anatomical grading system | 6-grade ulcer depth and gangrene classification (Grades 0–5) | Foundational clinical framework for surgical debridement and amputation risk | Establishes the clinical depth ground truth modeled by our CORAL ordinal branch. |
| **Armstrong et al.** [13] | 1998 | *Diabetes Care*<br>doi: 10.2337/diacare.21.5.855 | Clinical cohort ($n=360$) | Orthogonal $4 \times 4$ staging matrix | Multi-axial cross-tabulation of depth (0–3) and pathology (A–D) | Validated that concurrent infection + ischaemia + depth dramatically escalates amputation | Provides the clinical theoretical basis for DFU-WAGNER net's 16-cell UT matrix outer product. |
| **Gretton et al.** [31] | 2012 | *J. Mach. Learn. Res.* | Benchmark statistical distributions | Multi-kernel Maximum Mean Discrepancy (MMD) | Non-parametric two-sample distribution testing | Proved convergence of RKHS kernel embeddings for distribution alignment | Mathematical foundation for our uncertainty-gated cross-cohort feature alignment head. |
| **Bousmalis et al.** [29] | 2016 | *NeurIPS* | Office, MNIST, USPS benchmarks | Domain Separation Networks (DSNs) | Unsupervised domain adaptation via shared/private separation | Demonstrated that explicit subspace orthogonality prevents negative transfer | Directly inspires DFU-WAGNER net's 3-way disentangler ($z_{\text{shared}} \perp z_{\text{private}}$). |
| **Selvaraju et al.** [34] | 2017 | *IEEE ICCV*<br>doi: 10.1109/ICCV.2017.74 | ImageNet, VQA | Gradient-weighted Class Activation Mapping (Grad-CAM) | Visual model attribution and feature localization | Established post-hoc gradient attribution for convolutional layers | Used in our paper for explainability; explicitly framed as non-causal saliency maps. |
| **Cui et al.** [18] | 2019 | *IEEE/CVF CVPR*<br>doi: 10.1109/CVPR.2019.00949 | Long-tailed CIFAR, iNaturalist | Class-Balanced Loss via Effective Number of Samples: $E_n = (1-\beta^n)/(1-\beta)$ | Long-tailed multi-class classification | Achieved substantial error reductions on extreme minority classes | Integrated as DFU-WAGNER net's primary supervised pathology loss for the 3.8% ischaemia class. |
| **Goyal et al.** [6] | 2019 | *IEEE JBHI*<br>doi: 10.1109/JBHI.2018.2865086 | Smartphone DFU dataset ($n=1,779$) | Faster R-CNN with InceptionV2 | Real-time DFU detection and bounding-box localization | mAP = 0.918 on standard mobile photography | Focused purely on bounding-box localization without pathology or depth discrimination. |
| **Cao et al.** [17] | 2020 | *Pattern Recognit. Lett.*<br>doi: 10.1016/j.patrec.2020.11.008 | Morph, UTKFace | Consistent Rank Logits (CORAL) framework | Ordinal regression via cumulative binary subproblems | Guaranteed rank-monotonic logit consistency; reduced MAE significantly | Adopted directly for DFU-WAGNER net's Wagner depth head to model monotonic tissue destruction. |
| **Goyal et al.** [4] | 2020 | *IEEE TETCI*<br>doi: 10.1109/TETCI.2018.2865214 | Binary DFU dataset ($n=3,998$) | DFUNet (Custom CNN with feature re-use) | Binary DFU vs. healthy skin classification | F1-Score = 0.925 on binary patch detection | Limited to binary presence/absence; no discrimination of infection, ischaemia, or depth. |
| **Goyal et al.** [7] | 2020 | *Comput. Biol. Med.*<br>doi: 10.1016/j.compbiomed.2020.103616 | DFU Natural Dataset ($n=1,459$) | Ensemble of AlexNet, GoogLeNet, ResNet50 | 2-class infection detection and 2-class ischaemia detection | Infection F1 = 0.73; Ischaemia F1 = 0.77 | Treated infection and ischaemia as disconnected binary tasks rather than joint 4-class pathology. |
| **Cassidy et al.** [2] | 2021 | *MICCAI DFUC2021*<br>doi: 10.1007/978-3-030-94907-5_1 | DFUC2021 benchmark ($n=15,683$ total) | Baseline CNN/Transformer benchmark architectures | 4-class pathology classification (None, Infection, Ischaemia, Both) | Official Challenge Baseline Macro-F1 = 0.6228 | Established the multi-class pathology benchmark; highlighted severe ischaemia degradation. |
| **Galdran et al.** [3] | 2021 | *MICCAI DFUC2021*<br>doi: 10.1007/978-3-030-94907-5_4 | DFUC2021 challenge dataset | ResNeXt50 + DeiT ensemble trained with SAM | 4-class DFU pathology classification | **DFUC2021 1st Place Winner**: Macro-F1 = 0.6307 | Required complex ensembling; achieved limited improvement on the severe minority ischaemia class. |
| **Liu et al.** [22] | 2021 | *IEEE/CVF ICCV*<br>doi: 10.1109/ICCV48922.2021.00986 | ImageNet-1K, COCO, ADE20K | Swin Transformer: Hierarchical shifted window self-attention | General vision representation learning | Top-1 Accuracy = 87.3% on ImageNet; linear computational complexity | Backbone architecture utilized in DFU-WAGNER net for global spatial context. |
| **Tan & Le** [21] | 2021 | *ICML* | ImageNet-1K, ImageNet-21k | EfficientNetV2: Fused-MBConv with progressive learning | High-throughput convolutional feature extraction | Faster training and 6.8x smaller parameters than contemporary models | CNN backbone selected for DFU-WAGNER net to provide sharp local inductive edge priors. |
| **Yap et al.** [8] | 2022 | *Front. Endocrinol.*<br>doi: 10.3389/fendo.2022.886867 | Multi-center international cohorts | Comprehensive survey of DFU computational imaging | Survey of benchmarking, labeling standards, and clinical gaps | Highlighted the severe fragmentation of public clinical wound datasets | Identifies the fundamental research gap our cross-cohort alignment addresses. |
| **Toofanee et al.** [1] | 2023 | *IEEE Access*<br>doi: 10.1109/ACCESS.2023.3312531 | DFUC2021 dataset ($n=5,955$ labeled) | DFU-SIAM (EfficientNetV2-S + BEiT + LMCoT Loss) | Single-task 4-class DFU pathology classification | Macro-F1 = 0.6228; demonstrated benefits of Siamese metric learning | Baseline model extended in this work; lacked Wagner depth, disentanglement, and domain alignment. |
| **Toofanee et al.** [9] | 2023 | *Comput. Biol. Med.*<br>doi: 10.1016/j.compbiomed.2023.107402 | Longitudinal clinical cohort ($n=340$ series) | DFU-Helper: Siamese longitudinal disease tracking | Ulcer healing trajectory and progression tracking | High ranking correlation with expert healing assessments | Focused on longitudinal image pairs; did not address multi-cohort taxonomy alignment. |
| **Rostami et al.** [14] | 2024 | *IEEE JTEHM*<br>doi: 10.1109/JTEHM.2023.3331201 | Clinical Wagner dataset ($n=2,480$) | Multi-scale DenseNet with transfer learning | 5-class Wagner grading and amputation risk stratification | Accuracy = 88.4%; MAE = 0.32 grades | Treated Wagner grading as standard categorical cross-entropy without rank consistency. |
| **Zhou et al.** [15] | 2025 | *Front. Endocrinol.*<br>doi: 10.3389/fendo.2024.1389021 | Clinical ulcer dataset ($n=3,120$) | Mask2Former + Swin-Transformer with masked attention | Wagner 1–4 depth segmentation and severity classification | Categorical Accuracy = 91.85% | Focused exclusively on wound depth; ignored microbiological infection and ischaemia status. |
| **Karthik et al.** [11] | 2025 | *Sci. Rep.*<br>doi: 10.1038/s41598-025-54312-8 | DFUC2020 and DFUC2021 | Swin Transformer + Efficient Multi-Scale Attention (EMA) | Multi-scale feature attention for DFU pathology | Reported external test accuracy exceeding 89% | Evaluated only nominal accuracy; did not evaluate QWK, ordinal severity, or multi-task alignment. |
| **Abdullah et al.** [10] | 2025 | *Front. Artif. Intell.*<br>doi: 10.3389/frai.2025.1324512 | Multi-center dataset ($n=4,200$) | Self-Supervised SimCLR + Multi-Head DenseNet | Joint triage (7-class) and severity classification (0–3) | Overall Accuracy = 95.5% | Relied on heuristic merging without disentangling shared and private feature representations. |
| **Present Work** | **2026** | **Manuscript Draft** | **DFUC2021 ($n=5,955$) + Wagner ($n=11,491$ deduplicated)** | **DFU-WAGNER net**: EfficientNetV2-S + Swin-Tiny + Disentangler + Hard-LMCoT + UG-MMD + CORAL | **Joint Multi-Task**: 4-class Pathology + 5-class Wagner Ordinal Depth $\to$ 16-Cell UT Matrix | `[FINAL TEST RESULTS PENDING]` | **Unified, disentangled, cross-cohort framework resolving clinical taxonomy fragmentation.** |

---

## 6. Identification of Critical Research Gaps

Based on the surveyed literature, five distinct and scientifically verifiable research gaps are identified:

### Gap 1: Clinical Taxonomy Fragmentation
*Literature Evidence*: Yap et al. [8], Cassidy et al. [2], Wagner [12], and Armstrong et al. [13].  
*Problem*: Existing deep learning models treat microbiological pathology (infection/ischaemia) and anatomical ulcer depth (Wagner grade) as mutually exclusive problems. Public cohorts are strictly fragmented: DFUC2021 provides no depth labels, while Wagner cohorts provide no microbiological labels. Clinicians cannot use separate single-task models reliably because the gold-standard University of Texas matrix requires the concurrent synthesis of both axes to make limb-salvage decisions.  
*Our Resolution*: DFU-WAGNER net formulates an interleaved multi-task pipeline that projects images from disjoint cohorts into a shared anatomical manifold while isolating task-private features, synthesizing the joint 16-cell UT wound probability matrix differentiably.

### Gap 2: Inability of Standard Objectives to Handle Multi-Cohort Domain Shifts
*Literature Evidence*: Gretton et al. [31], Long et al. [32], and Kendall et al. [33].  
*Problem*: Passing images from distinct cohorts (DFUC cropped patches vs. Wagner clinical photographs) through hard-shared backbones causes severe negative transfer. Standard domain adaptation aligns entire feature representations, which forces task-specific signals (infection erythema vs. bone depth) to artificially align, corrupting performance.  
*Our Resolution*: DFU-WAGNER net implements **Uncertainty-Gated MMD (UG-MMD)** that operates *exclusively* on the shared representation $z_{\text{shared}}$, using detached model predictive confidence to prevent noisy or uncertain ulcer representations from corrupting the domain alignment objective.

### Gap 3: Long-Tailed Class Imbalance on Severe Pathology (Ischaemia)
*Literature Evidence*: Cassidy et al. [2], Galdran et al. [3], Toofanee et al. [1], and Cui et al. [18].  
*Problem*: In the DFUC2021 benchmark, ischaemia accounts for only 3.81% of training samples ($n=227$). Standard cross-entropy minimization causes models to collapse toward majority classes (*None* and *Infection*, each $\approx 43\%$), producing low sensitivity on ischaemic wounds.  
*Our Resolution*: We replace standard cross-entropy with **Class-Balanced Focal Loss** [18] (derived from effective sample volumes $E_n = (1-\beta^n)/(1-\beta)$) and couple it with **Hard-Pair Tangent LMCoT Metric Learning** that mines the hardest positive and negative pairs across clinical boundaries.

### Gap 4: Misapplication of Nominal Classification Losses to Ordinal Wound Severities
*Literature Evidence*: Cao et al. [17], Cohen [20], and Rostami et al. [14].  
*Problem*: Previous deep learning studies for Wagner grading apply standard categorical cross-entropy, treating a one-grade misclassification identically to a four-grade error (e.g., mistaking gangrene for intact skin).  
*Our Resolution*: DFU-WAGNER net implements **Consistent Rank Logits (CORAL)** [17], decomposing Wagner grading into 4 rank-consistent cumulative binary classifiers that enforce monotonic depth progression and bounded MAE.

### Gap 5: Monolithic Representation Collapse in Multi-Task Models
*Literature Evidence*: Bousmalis et al. [29] and Caruana [30].  
*Problem*: Standard multi-task networks concatenate backbone features and feed them directly to multiple heads. Without explicit geometric constraints, gradients from one task degrade features required by the other.  
*Our Resolution*: We design a dedicated **TaskDisentangler** module that projects fused 2,560-D hybrid features into three strictly L2-normalized subspaces: $z_{\text{shared}} \in \mathbb{R}^{768}$, $z_{\text{path\_private}} \in \mathbb{R}^{512}$, and $z_{\text{depth\_private}} \in \mathbb{R}^{512}$, constrained by an **Orthogonality Loss** ($\mathcal{L}_{\text{orth}}$) that minimizes squared cosine similarity between shared and private vectors.

---

## 7. Precise Implemented Contributions of DFU-WAGNER net

In strict adherence to research integrity, the contributions of the proposed work are formulated based exclusively on the verified, hardened implementation in the repository:

1. **A Unified Multi-Task Architecture Resolving Taxonomy Fragmentation**: We engineer a dual-backbone network combining **EfficientNetV2-S** (1,280-D raw) and **Swin-Transformer Tiny** (768-D raw) via symmetrical 1,280-D linear projections, simultaneously addressing DFUC2021 4-class pathology and 5-class Wagner ordinal depth without requiring paired multi-label training data.
2. **Explicit Shared-Private Subspace Disentanglement with Orthogonality**: We implement a 3-way `TaskDisentangler` partitioning fused 2,560-D features into $z_{\text{shared}} \in \mathbb{R}^{768}$, $z_{\text{path\_private}} \in \mathbb{R}^{512}$, and $z_{\text{depth\_private}} \in \mathbb{R}^{512}$. Selective branching ensures depth features never enter the pathology classifier, while pathology features never enter the Wagner head. Subspace independence is enforced via an $O(B \cdot D)$ cosine orthogonality loss $\mathcal{L}_{\text{orth}}$.
3. **Hard-Pair Tangent LMCoT Metric Learning**: We resolve the mathematical polarity flaw of the original LMCoT by implementing monotonic tangent distance $\tan(\theta/2)$ on $z_{\text{shared}}$, augmented with active batch-level hard-pair mining (prioritizing hardest intra-class positives and closest inter-class negatives across clinical boundaries).
4. **Uncertainty-Gated Multi-Scale RKHS-MMD Alignment**: We formulate a distribution alignment mechanism that aligns *only* the shared subspace $z_{\text{shared}}$ across DFUC2021 and Wagner cohorts using multi-scale RBF kernels ($\sigma \in \{0.1, 1.0, 10.0\}$). Detached predictive entropy weights prevent uncertain predictions from distorting the aligned manifold.
5. **Rank-Consistent Ordinal Depth Modeling**: We integrate CORAL ordinal cumulative cutoffs for Wagner grading, replacing nominal cross-entropy to preserve the biological continuum of wound depth and directly optimize Quadratic Weighted Kappa (QWK) and Mean Absolute Error (MAE).
