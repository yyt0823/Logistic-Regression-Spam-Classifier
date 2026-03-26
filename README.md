# Logistic Regression — Spam Classification

**McGill COMP551** — Logistic Regression from Scratch on the UCI Spambase Dataset

---

## Dataset

The [Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase) from the UCI ML Repository contains **4,601 emails**, each described by 57 features (word/character frequencies, capital letter statistics). The binary label indicates whether the email is spam (1) or not (0).

Loaded via `ucimlrepo`:
```python
from ucimlrepo import fetch_ucirepo
spambase = fetch_ucirepo(id=94)
```

Features are standardized using training-set mean and standard deviation only (no data leakage).

---

## Tasks

### Task 1 — Logistic Regression with SGD from Scratch
Implemented mini-batch SGD logistic regression with:
- **L2 regularization** (weight decay)
- **Momentum** (β = 0.9) as an optional extension
- Loss tracked per epoch and plotted for both batch sizes (16 and 1)

Key hyperparameters: `lr=0.001`, `batch_size=16`, `epochs=50`, `λ=0.001`

### Task 2 — Hyperparameter Search with K-Fold Cross-Validation
Grid search over learning rate, batch size, and number of epochs using **5-fold CV**.

Grid searched:
| lr | batch_size | epochs |
|----|-----------|--------|
| 1e-3 / 1e-2 | 16 / 64 | 50 / 100 |

Best configuration selected by mean cross-validation log-loss.

### Task 3 — Regularization Strength Selection
Fixed architecture (`lr=0.1`, `batch_size=16`, `epochs=200`), K-fold CV over a grid of λ values:

```
λ ∈ {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1}
```

Optimal λ selected by minimizing validation cross-entropy.

### Task 4 — L1 Regularization & Feature Sparsity
Used `sklearn.LogisticRegression` with L1 penalty (SAGA solver) over a log-spaced grid of `C` values (1/λ). Tracked:
- Coefficient paths as regularization increases
- Number of non-zero coefficients vs. `C`

This reveals which features are most predictive of spam.

---

## Results

| Task | Metric | Result |
|------|--------|--------|
| Task 1 | Training loss converges | ✓ |
| Task 2 | Best CV config | `lr=1e-3, batch=16, epochs=100` (approx) |
| Task 3 | Optimal λ | selected via CV |
| Task 4 | Sparse solution | L1 zeros out low-signal features |

---

## Files

| File | Description |
|------|-------------|
| `code.ipynb` | Full notebook with all tasks and outputs |
| `writeup.pdf` | Written report |
