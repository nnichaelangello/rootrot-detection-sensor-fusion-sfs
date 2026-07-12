# Root Rot Severity Detection in Melon Plants Using Electronic Nose and Environmental Sensor Fusion with Sequential Forward Selection

---

## Overview

This repository provides the complete implementation associated with the study on automated detection of root rot disease severity in melon plants (*Cucumis melo* L.) using a multi-modal sensing approach. The system combines readings from an electronic nose (e-nose) array of six metal oxide semiconductor (MOS) gas sensors with three complementary environmental measurements, yielding a nine-dimensional feature space.

A key element of the methodology is a **per-model Sequential Forward Selection (SFS)** procedure that determines the most discriminative feature subset independently for each classifier. Eight classifiers — spanning classical machine learning, neural network, and deep learning paradigms — are evaluated under six experimental scenarios designed as an ablation study, allowing the individual and joint contributions of sensor modalities and feature selection to be quantified.

---

## Dataset

The dataset integrates two independently collected sensing streams. The table below describes each feature, its sensor category, and the underlying physical measurement principle.

<img width="731" height="252" alt="image" src="https://github.com/user-attachments/assets/b6b8b2e2-807a-4e27-8cdd-f24e9d7c431d" />

Four severity classes are defined: **Healthy Root**, **Mild Root Rot**, **Moderate Root Rot**, and **Severe Root Rot**.

The table below summarizes how samples are distributed across the training and test partitions. The original training counts reflect the raw class sizes following stratified splitting; the bootstrap training counts reflect the balanced set after applying within-class resampling to equalize class sizes at 96 samples per class.

<img width="726" height="158" alt="image" src="https://github.com/user-attachments/assets/1234c3df-d515-4bdf-b6ed-660ab803f1e1" />

> **Note:** Bootstrap resampling is applied exclusively to the training partition, after the train-test split is finalized. The test set (n = 92) is held out and not modified in any way.

> **Dataset availability:** The raw data files are subject to institutional data-sharing constraints. Parties interested in accessing the dataset for replication purposes are encouraged to contact the corresponding author directly.

---

## Models

Eight classifiers are benchmarked. All models are instantiated at their **published default hyperparameters**; no grid search or tuning is applied. Only `random_state = 42` is introduced where the implementation permits, solely to ensure reproducibility across runs.

<img width="876" height="327" alt="image" src="https://github.com/user-attachments/assets/ec0d1f98-c2e4-43b1-b5dd-a6ce849d0a52" />

---

## Experimental Scenarios

Six scenarios are defined to systematically decompose the contribution of each design choice. Scenarios S1 through S3 serve as fixed-feature references; S4 and S5 isolate SFS effects within each individual modality; S6 represents the complete proposed system.

<img width="617" height="182" alt="image" src="https://github.com/user-attachments/assets/2b0ca995-03cd-4fed-8a26-94a38e2f478d" />

The ablation logic addresses five research questions:

- **RQ1** — Does fusing both modalities (S6) outperform each individual modality in isolation (S1, S2)?
- **RQ2** — Does applying SFS to the full feature set (S6) surpass using all features without selection (S3)?
- **RQ3** — Does SFS improve classification when restricted to e-nose inputs alone (S4 vs. S2)?
- **RQ4** — Does SFS improve classification when restricted to environmental inputs alone (S5 vs. S1)?
- **RQ5** — Is the advantage of S6 consistent across all eight model architectures?

---

## Sequential Forward Selection (SFS) Configuration

SFS is executed independently for each model. At each step, the candidate feature that yields the highest mean cross-validation accuracy when added to the current selected set is retained. The process continues until all nine features have been evaluated. The optimal subset size is determined as the step index at which the accuracy curve reaches its maximum (argmax), without any threshold or stopping criterion.

```
SFS folds         : 5-fold stratified cross-validation
SFS seed          : random_state = 42
Optimality rule   : argmax of mean CV accuracy across steps 1–9
Feature pool      : All 9 features (TGS832, TGS2620, TGS2600, TGS2603,
                    TGS822, TGS826, pH, EC, Humidity)
Evaluator         : Each model evaluates its own SFS using itself as the scorer
```

The subsets derived from S6 SFS are also used to construct S4 and S5 by projecting the selected features back onto the e-nose and environmental subspaces, respectively.

---

## Reproducibility

A single global seed (`42`) is applied at every stochastic point in the pipeline. The table below enumerates all locations where this seed is set.

| Component | Seed Application |
|-----------|-----------------|
| NumPy global state | `np.random.seed(42)` |
| PyTorch global state | `torch.manual_seed(42)` |
| Stratified train-test split | `train_test_split(..., random_state=42)` |
| Bootstrap resampling (per class) | `np.random.RandomState(42)` |
| Shuffle of balanced training set | `np.random.RandomState(42).permutation(...)` |
| SFS cross-validation splitter | `StratifiedKFold(..., random_state=42)` |
| Final evaluation CV splitter | `StratifiedKFold(..., random_state=42)` |
| SVM-RBF | `random_state=42` |
| Logistic Regression | `random_state=42` |
| Random Forest | `random_state=42` |
| Decision Tree | `random_state=42` |
| XGBoost | `random_state=42` |
| MLP | `random_state=42` |
| FT-Transformer (internal seed) | `seed=42` passed to classifier wrapper |
| TabNet (internal seed) | `seed=42` passed to TabNetClassifier |

> Minor numerical differences may arise from GPU floating-point non-determinism or library version changes, but the relative rankings and conclusions are expected to remain stable.

---

## Results Summary

The table below reports test-set accuracy across all eight models and six scenarios. S6 denotes the proposed SFS-based sensor fusion method.

<img width="851" height="312" alt="image" src="https://github.com/user-attachments/assets/432d65cd-d588-4804-8497-ce97d5cdf2cf" />
