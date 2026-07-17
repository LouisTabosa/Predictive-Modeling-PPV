# Advancements in Predictive Modeling for Peak Particle Velocity in Rock Blasting

This repository contains the data, code, and results associated with the review article:

> **"Advancements in Predictive Modeling for Peak Particle Velocity in Rock Blasting: Tracing the Evolution from Empirical Models to Artificial Intelligence Techniques"**
> 
> Louis D. G. Tabosa, José J. S. de Figueiredo, Vidal F. N. Torres, Saulo S. Martins
> 
> Universidade Federal do Pará (UFPA) · Universidade Federal do Sul e Sudeste do Pará (Unifesspa) · Instituto Tecnológico Vale (ITV)

---

## Overview

The prediction of Peak Particle Velocity (PPV) is a fundamental task for operational safety in rock blasting activities. This work systematically examines the methodological evolution of predictive PPV models — from pioneering empirical approaches (USBM) to contemporary Artificial Intelligence techniques..

---

## Repository Structure

```
├── Chapter_6_Practical_Applications_and_Field_Validation/
│   ├── Dados_200.xlsx          # Field dataset (200 samples) used for USBM and ANN fitting
│   ├── USBM-FIT200.m           # MATLAB environment for USBM empirical fitting
│   ├── Fit_USBM.m              # MATLAB routine to fit USBM model to custom datasets
│   ├── train_ANN.m             # MATLAB script to reproduce ANN training
│   └── Fit_ANN.m               # Trained ANN model ready to apply to new data

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

## Chapter 7 — Removed to comply with peer review.

## Citation

If you use this code or data in your research, please cite:

```
Tabosa, L. D. G., de Figueiredo, J. J. S., Torres, V. F. N., & Martins, S. S.
Advancements in Predictive Modeling for Peak Particle Velocity in Rock Blasting:
Tracing the Evolution from Empirical Models to Artificial Intelligence Techniques.
Arch Computat Methods Eng (2026). https://doi.org/10.1007/s11831-026-10714-4
```

---

## Authors

- **Louis D. G. Tabosa** — Unifesspa / UFPA
- **José J. S. de Figueiredo** — UFPA
- **Vidal F. N. Torres** — ITV
- **Saulo S. Martins** — UFPA
