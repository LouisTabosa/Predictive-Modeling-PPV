# Advancements in Predictive Modeling for Peak Particle Velocity in Rock Blasting

This repository contains the data, code, and results associated with the review article:

> **"Advancements in Predictive Modeling for Peak Particle Velocity in Rock Blasting: Tracing the Evolution from Empirical Models to Artificial Intelligence Techniques"**
> Louis D. G. Tabosa, José J. de Figueiredo, Vidal F. N. Torres, Saulo S. Martins
> Federal University of Pará (UFPA) · Universidade Federal do SUl e Sudeste do Pará (Unifesspa) · Vale Institute of Technology (ITV)

---

## Overview

The prediction of Peak Particle Velocity (PPV) is a fundamental task for operational safety in rock blasting activities. This work systematically examines the methodological evolution of predictive PPV models — from pioneering empirical approaches (USBM) to contemporary Artificial Intelligence techniques — and presents comparative numerical experiments using physics-consistent synthetic data to evaluate Empirical, ANN, SVR, and Empirical-Driven Machine Learning (EDML) strategies.

---

## Repository Structure

```
├── Chapter_6_Practical_Applications_and_Field_Validation/
│   ├── Dados_200.xlsx          # Field dataset (200 samples) used for USBM and ANN fitting
│   ├── USBM-FIT200.m           # MATLAB environment for USBM empirical fitting
│   ├── Fit_USBM.m              # MATLAB routine to fit USBM model to custom datasets
│   ├── train_ANN.m             # MATLAB script to reproduce ANN training
│   └── Fit_ANN.m               # Trained ANN model ready to apply to new data
│
└── Chapter_7_Comparative_Numerical_Analysis_Empirical_ANN_and_EDML_Approaches/
    ├── EDML-ANN.py             # Main Python script: synthetic data generation and model comparison
    └── Output/                 # Pre-computed results (figures and metrics) — no need to run the code
```

---

## Chapter 6 — Practical Applications and Field Validation

This chapter applies the USBM empirical model and an Artificial Neural Network to a real field dataset of 200 blast vibration measurements.

### Files

| File | Description |
|------|-------------|
| `Dados_200.xlsx` | Field dataset containing distance, charge per delay, and measured PPV values |
| `USBM-FIT200.m` | Full MATLAB environment where the USBM fitting was performed and validated |
| `Fit_USBM.m` | Standalone routine to fit the USBM power-law model to any dataset |
| `train_ANN.m` | Script to reproduce the ANN training procedure from scratch |
| `Fit_ANN.m` | Pre-trained ANN model to apply directly to new blast data |

### How to use `Fit_USBM.m`

1. Open MATLAB and load your dataset in the same format as `Dados_200.xlsx` (columns: Distance, Charge, PPV)
2. Run `Fit_USBM.m` — it will automatically fit the USBM constants *k* and *β* via log-log regression
3. The script outputs the fitted parameters and a scatter plot of measured vs. predicted PPV

### How to use `Fit_ANN.m`

1. Load `Fit_ANN.m` in MATLAB
2. Provide your input data (Distance and Charge per delay)
3. The script returns PPV predictions using the pre-trained network weights

---

## Chapter 7 — Comparative Numerical Analysis: Empirical, ANN, SVR, and EDML Approaches

This chapter presents controlled numerical experiments using physics-consistent synthetic data to compare four modeling strategies across two physical scenarios:

- **Scenario A** — Pure geometric spreading (power-law attenuation)
- **Scenario B** — Geometric spreading with material attenuation (exponential damping)

### Models compared

| Model | Description |
|-------|-------------|
| **Empirical** | Standard USBM power-law regression in log-log space |
| **ANN** | Feedforward neural network trained directly on blast parameters |
| **SVR** | Support Vector Regression with RBF kernel, hyperparameters tuned via cross-validation |
| **EDML** | Empirical-Driven Machine Learning — ANN trained on residuals from the empirical baseline |

### Files

| File | Description |
|------|-------------|
| `EDML-ANN.py` | Main script: generates synthetic data, trains all four models, and produces comparison figures |
| `Output/` | Pre-computed figures and performance metrics for both scenarios |

### Dependencies

```
numpy
pandas
matplotlib
scikit-learn
os
itertools
time
warnings
```

Install with:
```bash
pip install numpy pandas matplotlib scikit-learn
```

### How to run

```bash
python ann_svr_edml.py
```

The script will:
1. Generate synthetic PPV datasets for Scenario A and Scenario B
2. Train the Empirical, ANN, SVR, and EDML models
3. Output performance metrics (R², RMSE, training time)
4. Save comparison figures to the `Output/` folder

> If you only want to inspect the results without running the code, all figures and metrics are already available in the `Output/` folder.

---

## Key Results

| Scenario | Model | R² | RMSE | Training Time |
|----------|-------|----|------|---------------|
| A — Geometric only | Empirical | 0.782 | 0.425 | 1.3 ms |
| A — Geometric only | ANN | 0.770 | 0.437 | ~319 s |
| A — Geometric only | SVR | 0.763 | 0.443 | 1.44 s |
| A — Geometric only | EDML | 0.779 | 0.428 | ~126 s |
| B — With attenuation | Empirical | 0.883 | 0.685 | 1.3 ms |
| B — With attenuation | ANN | 0.952 | 0.439 | ~383 s |
| B — With attenuation | SVR | 0.951 | 0.445 | 3.22 s |
| B — With attenuation | EDML | **0.954** | **0.431** | ~273 s |

The EDML approach achieves the best accuracy in the physically realistic scenario (B). The SVR offers a compelling accuracy-to-cost ratio, achieving near-identical accuracy to ANN while being approximately 100× faster to train — making it particularly attractive for real-time or resource-constrained applications.

---

## Data Availability

The synthetic datasets used in the numerical experiments (Section 7) are generated programmatically and can be fully reproduced using the provided source code. The field dataset used in the validation analysis (Fig. 5, Section 6.3) is publicly available in:

> Hammed et al. (2018). *Peak particle velocity data acquisition for monitoring blast induced earthquakes in quarry sites*. Data in Brief, 19, 398–408.

---

## Citation

If you use this code or data in your research, please cite:

```
Tabosa, L. D. G., de Figueiredo, J. J., Torres, V. F. N., & Martins, S. S.
Advancements in Predictive Modeling for Peak Particle Velocity in Rock Blasting:
Tracing the Evolution from Empirical Models to Artificial Intelligence Techniques.
```

---

## Authors

- **Louis D. G. Tabosa** — Unifesspa / UFPA
- **José J. de Figueiredo** — UFPA
- **Vidal F. N. Torres** — ITV
- **Saulo S. Martins** — UFPA
