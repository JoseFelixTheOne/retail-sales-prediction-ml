# Retail Sales Prediction Pipeline

An end-to-end supervised machine learning regression pipeline designed to predict item-level retail sales across multiple supermarket formats using the BigMart Sales dataset.

The project follows reproducible Data Science best practices: structured exploratory data analysis (EDA), data quality audit, zero-data-leakage preprocessing, baseline benchmarking, and multi-model evaluation.

---

## Key Results & Business Impact

| Model | MAE ($) | RMSE ($) | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (`depth=6`)** | **$773.08** | **$1,092.42** | **0.5923** | **Selected** |
| Decision Tree (`depth=5`) | $773.20 | $1,095.87 | 0.5897 | Evaluated |
| Support Vector Regressor (RBF) | $794.82 | $1,124.84 | 0.5677 | Evaluated |
| Linear Regression | $853.99 | $1,132.94 | 0.5615 | Evaluated |
| Baseline (Mean Predictor) | $1,365.59 | $1,710.84 | -0.0000 | Baseline |

* **Accuracy Improvement:** The selected Random Forest model reduces mean prediction error by **~43.4%** compared to the naive baseline ($1,365.59 \rightarrow$773.08).
* **Primary Sales Drivers:** Feature importance confirms that item list price (`Item_MRP`) accounts for >50% of predictive importance, followed directly by store scale (`Supermarket Type 3` and `Supermarket Type 1`). Physical item attributes (weight, fat content) have marginal impact on gross revenue.

---

## Project Structure

```text
├── data/
│   ├── Test.csv                       # Unlabeled test dataset
│   └── Train.csv                      # Raw historical training dataset
├── TB1_grupo05_torres_felix.ipynb     # Main Jupyter Notebook (full ML pipeline)
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation

```

---

## Quickstart & Reproducibility

### 1. Clone the repository

```bash
git clone https://github.com/JoseFelixTheOne/retail-sales-prediction-ml.git
cd retail-sales-prediction-ml
```

### 2. Set up the virtual environment

Using `uv`:

```bash
uv venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

```

*(Or using standard `venv`):*

```bash
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the notebook

Open the project in VS Code or Jupyter and execute:

```bash
code TB1_grupo05_torres_felix.ipynb

```

---

## Pipeline Overview

1. **Global Data Quality Audit:** Automated inspection of missing values, category cardinality, and identifier detection.
2. **Preprocessing & Feature Engineering:**
* Missing value imputation: Mean imputation for `Item_Weight`, mode imputation for `Outlet_Size`.
* Label standardization on `Item_Fat_Content` (mapping abbreviations `LF`, `low fat`, and `reg`).
* One-Hot Encoding (`drop_first=True`) across categorical features.


3. **Data Leakage Prevention:** 80/20 train-test split executed **prior** to feature scaling (`StandardScaler` fitted strictly on the training set).
4. **Model Benchmarking:** Baseline evaluation vs. Linear Regression, SVR, Decision Tree, and Random Forest.
5. **Real-time Inference Simulation:** Sample-based test inference to validate single-product predictions against actual ground-truth values.

