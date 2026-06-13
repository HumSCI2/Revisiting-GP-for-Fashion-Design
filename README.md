# Revisiting Kernel Complexity in Gaussian Process Regression: An Empirical Study on Text-to-Visual Embedding Mapping in Fashion Design

> **Does more expressive always mean better?** A controlled empirical study of GP kernel complexity in text-to-visual embedding regression.


---

## Abstract

Gaussian Process (GP) regression is widely used for modeling complex data due to its flexibility and principled uncertainty estimation, where the choice of kernel plays a critical role in capturing underlying data characteristics. While complex kernels such as **Spectral Mixture (SM) kernels** are often assumed to provide superior performance due to their expressive power, it remains unclear whether increased kernel complexity is always beneficial for real-world data with weak or ambiguous patterns.

In this paper, we revisit the role of kernel complexity in GP regression through a controlled empirical study based on a **text-to-visual embedding mapping task in the fashion domain**. We formulate multimodal relationships as a low-dimensional regression problem, where textual features are used to predict corresponding visual embeddings. This enables a systematic investigation of how different kernels behave under varying data conditions — including synthetic stationary and non-stationary data, as well as real-world non-Gaussian data.

Our findings show that **increased kernel complexity does not necessarily improve predictive performance** when data exhibit weak structures or limited multi-scale patterns. In several settings, simpler stationary kernels achieve performance comparable to or better than more expressive SM-based kernels while requiring substantially fewer parameters and lower computational overhead. Statistical validation using Friedman and pairwise Wilcoxon signed-rank tests further confirms that simpler kernels can remain **statistically competitive** with the best-performing complex kernels.

We also show that conventional metrics such as RMSE and MAPE do not fully characterize model behavior — **uncertainty estimation and predictive visualization** provide complementary insights for kernel selection.

---

## Repository Structure

```
GP_fashion_design/
├── data/                       # Raw and preprocessed fashion data
├── src/
│   ├── features/               # Stage 1 — BERT & CLIP feature extraction
│   ├── autoencoder/            # Stage 2 — Autoencoder for low-dim representations
│   ├── gp/                     # Stage 3 — GP kernels (RBF, Matérn, SM variants, …)
│   ├── baselines/              # Linear regression and MLP baselines
│   └── evaluation/             # Metrics, statistical tests, calibration
├── experiments/
│   ├── synthetic/              # Controlled experiments on synthetic signals
│   ├── fashion/                # Full-dataset and attribute-subset evaluations
│   └── ablation/               # SM components, latent dim, learning rate
├── notebooks/                  # Step-by-step demos and result reproduction
├── results/                    # Saved tables and figures
├── requirements.txt
└── README.md
```

---

## Method Overview

The framework operates in three stages: feature extraction from raw multimodal fashion data, dimensionality reduction via autoencoders, and GP regression with kernels of varying complexity.

![Method Overview](figs/model_framework.png)


| Stage | Input | Output | Key components |
|-------|-------|--------|----------------|
| 1 — Feature extraction | Image + text description | 768-dim text emb. / 512-dim visual emb. | DistilBERT, CLIP ViT-B/32 |
| 2 — Representation | High-dim embeddings | 1-dim latent code (ablations: 2/5/10) | 3-layer fully connected AE |
| 3 — GP regression | `t, u_c, u_p, u_m, u_s, u_d` | Visual latent `v` | RBF · Matérn · Periodic · Linear · SM (D/E/R) |

---
![Method Overview](figs/fashion_encoder.png)


## Key Results

### No single kernel consistently dominates

| Kernel   | Dior TAE rank | Dior DAE rank | Versace TAE rank | Versace DAE rank | Mean rank |
|----------|:---:|:---:|:---:|:---:|:---:|
| RBF      | 7.1 | 7.4 | 5.9 | 5.0 | 6.35 |
| Matérn   | 4.3 | 6.2 | 2.1 | **2.0** | 3.65 |
| Periodic | 5.6 | 4.9 | 7.2 | 6.1 | 5.95 |
| Linear   | 2.6 | 2.5 | 5.2 | 8.0 | 4.58 |
| SM       | 4.3 | 5.1 | 3.4 | 3.7 | 4.13 |
| SM_D     | 7.8 | 5.9 | 7.5 | 6.9 | 7.03 |
| SM_E     | 2.5 | **1.5** | **1.7** | 2.4 | 2.03 |
| **SM_R** | **1.8** | 2.5 | 3.0 | **1.9** | **2.30** |

*(Rank 1 = best. Lower is better.)*

**Key takeaways:**
- SM_E and SM_R achieve the best average ranks, but **Matérn remains statistically competitive** on Versace (p = 0.19 / 0.62 vs. best) with far fewer parameters.
- SM_D — despite equal complexity to SM_E / SM_R — consistently **ranks among the worst**, confirming that initialization strategy matters more than kernel expressiveness alone.
- Statistical validation (Friedman test: χ² ≥ 51.30, p < 0.001 in all settings) confirms that differences are not due to random seed variation.

### Computational cost vs. performance trade-off

| Kernel | Train / restart  |
|--------|:-----------:|
| RBF    | 0.16 ± 0.03 s |
| Matérn | 1.33 ± 1.94 s |
| Linear | 2.39 ± 0.12 s |
| SM (Q=6) | 3.74 ± 1.38 s |
| SM_R   | 2.12 ± 0.23 s  |

SM kernels are ~23× slower to train than RBF and require ~11× more hyperparameters, yet do not consistently outperform simpler kernels on real-world fashion data.

---

## Part Results

### Synthetic Data: When SM kernels shine
![Stationary data](figs/stationary_data.png)

**Stationary data** — SM (Q=3) successfully recovers all three frequency components (0.5, 1.0, 2.0 Hz) from the spectral representation, while RBF and Matérn fail to capture the multi-scale structure.

![Non-Stationary data](figs/strong_non_statinoary.png)


**Non-stationary regime shift** — All stationary kernels, including SM, struggle when the signal transitions abruptly between 0.5 Hz and 1.5 Hz, as the learned spectrum collapses to a single dominant component.

### Real-world Fashion Data: When complexity hurts
![Non-Stationary data](figs/fashion_data.png)

 
**Full dataset (Dior / Versace)** — Visual features vary across time without consistent patterns; spectral representations show no dominant frequencies. In this regime, SM kernels offer marginal gains at substantially higher computational cost.
![dior visualization](figs/visualization_dior.png)


**Popular attribute subsets** — Even in more structured subsets (e.g., filtering by the most frequent color "black"), complex SM kernels fail to achieve statistically significant improvements over simpler alternatives.
![subset data distribution](figs/subset_distribution.png)



### Calibration curves

Calibration curves (predicted vs. empirical coverage) show that the Periodic kernel deviates most from the ideal diagonal, particularly at high coverage levels. Some SM variants are slightly conservative (wider intervals), while most kernels maintain reasonable calibration on fashion data.
![subset data distribution](figs/callibration_curve.png)


---

## Installation

```bash
# Clone the repository
git clone https://github.com/HumSCI2/GP-for-Fashion-Design.git
cd GP-for-Fashion-Design

# Create environment
conda env create -f environment.yml
conda activate revisit-brand

# Or with pip
pip install -r requirements.txt
```

**Core dependencies:** `torch`, `gpytorch`, `transformers`, `open_clip_torch`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `numpy`, `pandas`

---

## Data

The fashion dataset is sourced from the [1stDibs](https://en.wikipedia.org/wiki/1stdibs) platform and available via the [Brand5 dataset](https://github.com/dm-mo/fashion-DNA).

| Brand | Samples | Decades covered |
|-------|:-------:|:---------------:|
| Christian Dior | 481 | ~6 decades |
| Versace | 521 | ~5 decades |

**Data splits:**
- 90% → Autoencoder training
- 10% → Generalization evaluation (DAE set)
- Within the 90%: 9:1:1 → GP train / val / test (TAE set)

---

## Running Experiments

```bash
# Synthetic experiments
python experiments/synthetic/stationary.py

# Full fashion dataset (all kernels, both brands)
python experiments/fashion/run_all_kernels.py --brand dior --seed 0

# Subset evaluation
python experiments/fashion/run_subsets.py --brand versace --attribute color

# Ablation: effect of SM mixture components
python experiments/ablation/mixture_components.py --Q 2 4 6 8

# Ablation: latent dimensionality
python experiments/ablation/latent_dim.py --dim 1 2 5 10
```
---
## More Details
Full details coming upon paper acceptance.
