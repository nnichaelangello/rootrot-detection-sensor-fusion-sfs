# IoT-based Electronic Nose and Environmental Sensor Fusion for Melon Root Rot Classification Using Adaptive Feature Optimization

[![DOI](https://img.shields.io/badge/DOI-10.22266%2Fijies2026.0930.55-blue)](https://doi.org/10.22266/ijies2026.0930.55)
[![Journal](https://img.shields.io/badge/Journal-IJIES%20Vol.19%20No.9%202026-green)](http://www.inass.org/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Python](https://img.shields.io/badge/Python-3.12.13-blue)](https://www.python.org/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20RTX%20PRO%206000%20Blackwell-76B900)](https://www.nvidia.com/)

---

## Authors

**Helmy Widyantara**<sup>1\*</sup>, **Ahmad Wali Satria Bahari Johan**<sup>2</sup>, **Khodijah Amiroh**<sup>1</sup>, **Michael Angello Qadosy Riyadi**<sup>1</sup>, **Muhammad Adib Kamali**<sup>1</sup>

<sup>1</sup> Information Technology Study Program, Telkom University, Surabaya Campus, Ketintang 156, Surabaya, 60231, Indonesia  
<sup>2</sup> Informatics Study Program, Telkom University, Surabaya Campus, Ketintang 156, Surabaya, 60231, Indonesia  
\* Corresponding author: helmywidyantara@telkomuniversity.ac.id

---

## Abstract

This study proposes an IoT-based sensor fusion framework for early detection and severity classification of melon root rot by combining a six-channel metal oxide semiconductor (MOS) electronic nose array with three environmental sensors — measuring soil pH, electrical conductivity (EC), and relative humidity — into a unified nine-dimensional feature space. The central methodological contribution is a classifier-adaptive sequential forward selection (SFS) scheme in which each of eight heterogeneous classifiers independently identifies its own optimal feature subset, rather than sharing a single globally optimised selection pipeline. Evaluated across six experimental scenarios under stratified ten-fold cross-validation, hold-out testing, and Wilcoxon signed-rank analysis — with bootstrap resampling and standardisation applied strictly within each fold's training partition — all eight classifiers achieved perfect hold-out accuracy of 1.0000 using only three to eight model-specific features, with cross-validation accuracy ranging from 0.9865 to 0.9973 under the proposed scenario. The framework also delivered training time reductions of up to 26.16% and model size reductions of up to 28.55% for classical and neural network architectures. Soil pH and relative humidity emerged as universal cross-architecture biomarkers, establishing an evidence-based foundation for minimal-sensor IoT node design in precision agriculture.

---

## Background

Melon (*Cucumis melo* L.) is among the most commercially valuable horticultural commodities in tropical and subtropical agriculture, yet its production is chronically threatened by soilborne root rot disease — predominantly caused by *Fusarium oxysporum*, *Phytophthora* spp., and *Pythium* spp. The pathological progression of root rot is subsurface in nature: by the time macroscopic above-ground symptoms appear, pathogen colonisation of the root vascular system is already at an advanced stage, rendering post-symptomatic intervention largely ineffective.

Conventional diagnostic methods — including visual assessment, laboratory culturing, and molecular assays — are inherently reactive and incompatible with the continuous, real-time monitoring requirements of modern precision agriculture. This fundamental temporal mismatch between disease progression and diagnostic response motivates the search for automated, non-invasive, and early-stage detection systems capable of identifying root rot at subclinical phases of infection.

---

## Motivation

Three convergent observations underpin this work. First, volatile organic compound (VOC) profiles emitted by infected root tissue undergo measurable compositional shifts prior to macroscopic symptom onset, making gas-phase sensing a biochemically sound and temporally advantageous detection channel. Second, rhizosphere physicochemical parameters — particularly soil pH and humidity — are systematically perturbed by root rot pathogens during tissue degradation, providing an orthogonal and complementary disease signal. Third, existing sensor-based disease detection pipelines apply a single, globally shared feature selection strategy uniformly across all classifiers, an assumption that is methodologically unjustified: kernel-based methods, ensemble learners, and attention-based deep learning architectures exploit fundamentally different representational mechanisms and therefore benefit from different optimal feature subsets within the same fused sensor space.

Addressing these gaps simultaneously — through fusion of both sensing modalities and per-classifier adaptive feature selection — constitutes the primary motivation for this research.

---

## Comparison with Prior Work

| Reference | Sensing Modality | Feature Selection | Task | Reported Performance |
|---|---|---|---|---|
| Herrmann et al. (2024) | E-nose only (7 TGS sensors) | None | Water stress detection in soybean | 94.4% accuracy |
| Gorijavolu et al. (2026) | UAV imagery + environmental sensors | None (single pipeline) | Sugarcane disease detection | 97.6% accuracy, F1 = 0.967 |
| Talaat et al. (2026) | Imaging + IoT sensors | None | Foliar disease surveillance | 99.63% accuracy |
| Widyantara et al. (2026) [prior] | E-nose only (6 TGS sensors) | None | Melon root rot severity | Macro F1 = 84.58% |
| **This work** | **E-nose + environmental sensors (fused)** | **Per-model SFS (classifier-adaptive)** | **Melon root rot severity (4 classes)** | **CV: 0.9865–0.9973; Hold-out: 1.0000** |

The principal distinction from all prior work lies in the per-model instantiation of sequential forward selection: rather than applying one shared selection pipeline across all classifiers, each of the eight evaluated models independently determines its own optimal sensor subset from the fused nine-dimensional space.

---

## Objectives

1. Design and implement a dual-modality IoT sensing system integrating a six-channel MOS e-nose array with soil pH, EC, and humidity sensors, collecting 460 samples across four root rot severity classes.
2. Introduce a per-model SFS methodology over the fused feature space, demonstrating that three to eight model-specific features recover the full discriminative capacity of the nine-sensor configuration without statistically significant accuracy loss.
3. Conduct a rigorous six-scenario comparative benchmark with stratified ten-fold cross-validation, hold-out evaluation, Wilcoxon signed-rank significance testing, and computational efficiency profiling across eight classifiers.
4. Identify universal cross-architecture biomarkers of root rot severity to support the rational design of minimal-sensor IoT nodes for precision agriculture deployment.

---

## Methodology

### Dataset

The dataset comprises 460 labelled samples across four severity classes — Healthy Root, Mild Root Rot, Moderate Root Rot, and Severe Root Rot — collected using a standardised three-phase acquisition protocol (delay, sampling, purge) in a controlled measurement chamber. Each observation is a nine-dimensional feature vector drawn simultaneously from both sensing modules.

| Class | Train (n) | Test (n) | Total (n) |
|---|---|---|---|
| Healthy Root | 80 | 20 | 100 |
| Mild Root Rot | 96 | 24 | 120 |
| Moderate Root Rot | 96 | 24 | 120 |
| Severe Root Rot | 96 | 24 | 120 |
| **Total** | **368** | **92** | **460** |

### Sensing System

- **E-nose module**: Six MOS gas sensors — TGS832, TGS2620, TGS2600, TGS2603, TGS822, TGS826 — each targeting distinct VOC classes.
- **Environmental module**: Soil pH (electrochemical), electrical conductivity (mS/cm), and relative humidity (capacitive).

### Preprocessing

All preprocessing — bootstrap resampling to 96 samples per class for class balancing, followed by Z-score standardisation — was applied exclusively within the training combination of each cross-validation fold after fold assignment. No preprocessing parameters were fitted on, or applied to, any validation or test partition prior to inference, ensuring a strictly leakage-free protocol throughout.

### Feature Selection

Sequential Forward Selection (SFS) was instantiated independently for each of the eight classifiers over the nine-dimensional fused feature space, using stratified five-fold cross-validated accuracy as the selection criterion. This per-model design contrasts with conventional globally shared pipelines: because different classifier families exploit different inductive biases, their individually optimal sensor subsets within the shared space may diverge substantially. Optimal subset sizes ranged from three features (Decision Tree) to eight (TabNet).

### Classifiers

Eight architecturally diverse classifiers were evaluated using published default hyperparameters without additional tuning:

- **Classical ML**: SVM-RBF, Logistic Regression, Random Forest, Decision Tree, XGBoost
- **Neural Network**: MLP
- **Deep Learning**: FT-Transformer, TabNet

### Experimental Scenarios

| ID | Description | Feature Source | Selection |
|---|---|---|---|
| S1 | Environmental Only | pH, EC, Humidity | None |
| S2 | E-nose Only | TGS832–TGS826 | None |
| S3 | All Features | All 9 | None |
| S4 | Environmental Only (SFS-projected) | pH, EC, Humidity | Per-model SFS |
| S5 | E-nose Only (SFS-projected) | TGS series | Per-model SFS |
| S6 | All Features (SFS) — **proposed** | All 9 | Per-model SFS |

---

## Results and Discussion

### Feature Selection (S6 Optimal Subsets)

Humidity was selected by all eight models without exception; pH was selected by seven of eight (FT-Transformer selected EC in its place). This near-universal preference for environmental parameters reflects a physiologically grounded mechanism: root rot pathogens systematically alter rhizosphere pH and moisture retention during tissue degradation, producing a model-agnostic disease signature. In contrast, e-nose channel selection was comparatively model-dependent, consistent with the partially overlapping VOC sensitivity profiles of MOS sensors across different classifier architectures.

| Model | Optimal Features | Best 5-Fold CV Accuracy |
|---|---|---|
| SVM-RBF | 7 | 0.9945 |
| Logistic Regression | 4 | 0.9973 |
| Random Forest | 5 | 1.0000 |
| Decision Tree | 3 | 0.9973 |
| XGBoost | 4 | 0.9946 |
| MLP | 4 | 0.9946 |
| FT-Transformer | 4 | 1.0000 |
| TabNet | 8 | 0.9945 |

### Cross-Validation Performance (S6)

Under the proposed S6 scenario, ten-fold cross-validation accuracy ranged from 0.9865 (XGBoost) to 0.9973 (FT-Transformer) across all eight classifiers.

### Hold-Out Test Performance

All eight classifiers achieved perfect accuracy, precision, recall, and F1-score of 1.0000 on the locked 92-sample hold-out partition under scenario S6, using only three to eight model-specific features rather than the full nine.

### Statistical Significance

Wilcoxon signed-rank testing (two-sided, α = 0.05) confirmed that S6 significantly outperformed e-nose-only (S2) and SFS-projected e-nose (S5) configurations for all eight models (p = 0.0020). The S6 vs. S3 comparison — the primary test of whether per-model SFS adds accuracy value beyond unselected full fusion — yielded no statistically significant difference for any model (all p = 1.0000), confirming that classifier-adaptive SFS recovers full discriminative capacity without measurable accuracy trade-off.

### Computational Efficiency

Per-model SFS delivered statistically significant training time reductions for classical and ensemble architectures — up to 26.16% for XGBoost (p = 0.0488) — and model size reductions of up to 28.55% for MLP. Deep learning architectures (FT-Transformer, TabNet) showed no statistically significant training time change, reflecting the dimensionality-insensitive nature of fixed-epoch attention-based training at this input scale.

---

## Conclusion

Integrating a MOS electronic nose with environmental sensors within a classifier-adaptive SFS framework yielded a robust and computationally efficient system for four-class severity classification of melon root rot. Key findings are as follows:

- Sensor fusion of VOC profiles and environmental parameters is the primary discriminative mechanism; the two modalities carry complementary rather than redundant information.
- Per-model SFS recovers full nine-sensor discriminative capacity from three to eight model-specific features, with no statistically detectable accuracy loss relative to the unselected full-feature baseline.
- Soil pH and relative humidity are universal cross-architecture biomarkers, establishing that a minimal two-sensor environmental node augmented by one to five model-specific gas channels can approach the performance of the full array.
- Computational benefits of feature reduction are model-class-dependent: classical ML and ensemble models realise measurable efficiency gains, while deep learning architectures at this input scale are largely dimensionality-insensitive.

All findings are based on controlled laboratory validation. Generalisation to field conditions — including sensor drift, soil heterogeneity, and pathogen strain variation — remains an open direction for future work.

---

## Citation

If this work or repository is useful for your research, please cite both the published article and this software repository.

### Paper

```bibtex
@article{widyantara2026iot,
  author    = {Widyantara, Helmy and Johan, Ahmad Wali Satria Bahari and Amiroh, Khodijah and Riyadi, Michael Angello Qadosy and Kamali, Muhammad Adib},
  title     = {IoT-based Electronic Nose and Environmental Sensor Fusion for Melon Root Rot Classification Using Adaptive Feature Optimization},
  journal   = {International Journal of Intelligent Engineering and Systems},
  volume    = {19},
  number    = {9},
  pages     = {1125--1145},
  year      = {2026},
  doi       = {10.22266/ijies2026.0930.55},
  note      = {Received: July 1, 2026. Revised: July 22, 2026}
}
```

### Software Repository

```bibtex
@software{riyadi2026rootrot,
  author    = {Riyadi, Michael Angello Qadosy and Widyantara, Helmy},
  title     = {Root Rot Detection via Sensor Fusion and Sequential Forward Selection},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/nnichaelangello/rootrot-detection-sensor-fusion-sfs},
  note      = {Implementation code for: IoT-based Electronic Nose and Environmental Sensor Fusion for Melon Root Rot Classification Using Adaptive Feature Optimization. DOI: 10.22266/ijies2026.0930.55}
}
```

---

## Acknowledgements

This work was supported by the Research and Community Service of Telkom University under the Non-External Collaboration Research Scheme (Period 1, Wave 1), Contract No. 024/LIT06/PPM-LIT/2025.

---

*International Journal of Intelligent Engineering and Systems, Vol. 19, No. 9, 2026 — DOI: [10.22266/ijies2026.0930.55](https://doi.org/10.22266/ijies2026.0930.55)*
