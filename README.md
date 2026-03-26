# Logistic Regression — Spam Classification

> **Skills:** NumPy · Mini-batch SGD · L1/L2 regularization · K-fold cross-validation · Hyperparameter search · Feature sparsity analysis

---

## Overview

Binary email spam classification on the [UCI Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase) — **4,601 emails**, 57 features (word/character frequencies, capital letter statistics).

The entire logistic regression pipeline was implemented **from scratch in NumPy** — no sklearn for the core model.

Features are standardized using training-set statistics only (no data leakage).

---

## What I Built

### Logistic Regression with SGD from Scratch
Implemented mini-batch SGD with:
- **L2 regularization** (weight decay)
- **Momentum** (β = 0.9)
- Per-epoch loss tracking and plotting

Key hyperparameters: `lr=0.001`, `batch_size=16`, `epochs=50`, `λ=0.001`

### Hyperparameter Search with K-Fold Cross-Validation
Grid search over learning rate, batch size, and number of epochs using **5-fold CV**.

| lr | batch_size | epochs |
|----|-----------|--------|
| 1e-3 / 1e-2 | 16 / 64 | 50 / 100 |

Best configuration selected by mean cross-validation log-loss.

### Regularization Strength (λ) Selection
K-fold CV over a grid of λ values to find the optimal regularization strength:

```
λ ∈ {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1}
```

### L1 Regularization & Feature Sparsity
Used L1 penalty (SAGA solver) over a log-spaced grid of `C` values to produce sparse solutions and reveal which features are most predictive of spam. Tracked coefficient paths and the number of non-zero weights vs. regularization strength.

---

## Results

| | Result |
|--|--------|
| SGD training | Loss converges cleanly across batch sizes |
| Best CV config | `lr=1e-3, batch=16, epochs=100` |
| Optimal λ | Selected via cross-validation |
| L1 sparsity | Zeroes out low-signal features; highlights key spam indicators |

---

## Files

| File | Description |
|------|-------------|
| `code.ipynb` | Full notebook with all outputs |
| `writeup.pdf` | Written report |
