# Pipeline Architecture & Documentation
## Cryptocurrency Volatility Prediction — ML Pipeline

---

## 1. Pipeline Overview

The project uses a **multi-stage machine learning pipeline** that transforms raw cryptocurrency market data into a trained volatility prediction model. The pipeline is divided into 5 sequential stages.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CRYPTO VOLATILITY PREDICTION PIPELINE                │
│                                                                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  ┌────────┐ │
│  │  STAGE 1 │ → │  STAGE 2 │ → │  STAGE 3 │ → │  STAGE 4 │→ │STAGE 5│ │
│  │   Data   │   │   EDA &  │   │  Feature │   │  Model   │  │ Model  │ │
│  │Ingestion │   │ Analysis │   │  Prep    │   │Training  │  │ Saving │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘  └────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Stage 1 — Data Ingestion

### Purpose
Load raw data from disk and validate its integrity.

### Steps

```
dataset.csv
    │
    ▼
pd.read_csv()
    │
    ▼
Rename 'Unnamed: 0' → 'ID'
    │
    ▼
Parse date & timestamp → datetime64
    │
    ▼
Null Check (df.isna().sum())
    │
    ▼
Duplicate Check (df.duplicated().sum())
    │
    ▼
Display schema (df.info(), df.head())
```

### Inputs / Outputs

| | Detail |
|---|---|
| **Input** | `dataset.csv` |
| **Output** | Clean raw DataFrame `df` |
| **Key Parameters** | `errors='coerce'` for datetime parsing |

---

## 3. Stage 2 — EDA & Analysis

### Purpose
Understand data distributions, correlations, and identify data quality issues.

### Sub-stages

```
EDA Stage
├── 2a. Univariate Analysis
│       ├── KDE plots for all numerical features
│       └── Count plots for categorical features
│
├── 2b. Correlation Analysis
│       └── Heatmap of numerical feature correlations
│
├── 2c. Multivariate Analysis
│       ├── Yearly volatility trend (line plot)
│       ├── Top 10 most volatile cryptos (bar plot)
│       └── Price trends for major coins (FacetGrid)
│
└── 2d. Outlier Visualization
        └── Box plots for all numerical features (pre-treatment)
```

### Key Findings Logged

- OHLC multicollinearity detected → `high`, `low`, `close` dropped from X later
- WBTC outlier confirmed → handled via IQR capping
- Data is right-skewed → PowerTransformer applied in Stage 3
- 2021 = peak volatility year

---

## 4. Stage 3 — Feature Engineering & Preprocessing

### 4a. Feature Engineering Sub-Pipeline

```
df (raw)
    │
    ├──► Derive volatility = high − low
    │
    ├──► Extract from date:
    │       year, month, day, day_of_week
    │
    ├──► Drop: date, timestamp, ID
    │
    └──► Round numeric columns (2 decimal places)
         Scale volume → Billions (/1_000_000_000)
```

### 4b. Outlier Treatment Sub-Pipeline

```
df (after feature engineering)
    │
    ▼
cap_outliers(df, outlier_cols)
    │
    For each of ['open','high','low','close',
                 'volume','marketCap','volatility']:
        ├── Compute Q1, Q3, IQR
        ├── Set lower = Q1 - 1.5*IQR
        ├── Set upper = Q3 + 1.5*IQR
        ├── Cap values below lower → lower
        └── Cap values above upper → upper
    │
    ▼
df_cleaned (outlier-capped copy)
    │
df.update(df_cleaned)   ← apply back to main df
```

### 4c. Scaling & Transformation Sub-Pipeline

```
dataC = df.copy()  [includes crypto_name]
    │
    ▼
ColumnTransformer
    ├── PIPELINE A → power_transform_cols
    │   ['open','high','low','close','volume','marketCap','volatility']
    │   └── SimpleImputer(strategy='median')
    │   └── PowerTransformer(standardize=True)   # Yeo-Johnson
    │
    └── PIPELINE B → standard_scale_only_cols
        ['year','month','day','day_of_week']
        └── SimpleImputer(strategy='constant', fill_value=0)
        └── StandardScaler()
    │
    remainder='passthrough'  →  crypto_name passes through
    │
    ▼
scaled_data DataFrame
    │
    ▼
pd.get_dummies(columns=['crypto_name'], dtype=int)
    │
    ▼
Feature Selection:
    X = scaled_data.drop(['volatility','high','close','low'])
    y = scaled_data['volatility']
```

---

## 5. Stage 4 — Model Training & Evaluation

### 5a. Baseline Evaluation

```
X, y
    │
    ▼
train_test_split(test_size=0.2, random_state=42)
    │
    ▼
For each model in {Linear Regression, Decision Tree,
                   Random Forest, Gradient Boosting,
                   AdaBoost, XGBoost, KNN}:
    ├── model.fit(X_train, y_train)
    ├── y_pred = model.predict(X_test)
    └── Compute: MSE, RMSE, MAE, R²
    │
    ▼
Comparison Report DataFrame
```

### 5b. Model Comparison Results (Expected)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **Random Forest** ⭐ | ~0.98 | Lowest | Lowest |
| Gradient Boosting | ~0.95 | Low | Low |
| XGBoost | ~0.94 | Low | Low |
| Decision Tree | ~0.90 | Moderate | Moderate |
| K-Neighbors | ~0.85 | Moderate | Moderate |
| AdaBoost | ~0.75 | Higher | Higher |
| Linear Regression | ~0.60 | High | High |

### 5c. Hyperparameter Tuning (GridSearchCV)

```
Random Forest (baseline)
    │
    ▼
GridSearchCV
    ├── param_grid:
    │     n_estimators: [100, 200]
    │     max_depth: [None, 10]
    │     min_samples_split: [2, 5]
    │     min_samples_leaf: [1, 2]
    │     max_features: ['sqrt', 'log2']
    │     criterion: ['squared_error']
    ├── cv=3
    ├── n_jobs=-1
    └── verbose=2
    │
    ▼
Best Params: {n_estimators:200, max_depth:None,
              max_features:'sqrt', min_samples_leaf:1,
              min_samples_split:2}
    │
    ▼
best_rf_model.fit(X_train, y_train)
    │
    ▼
Final Evaluation: R², MAE on X_test
```

---

## 6. Stage 5 — Model Persistence

```
best_rf_model (trained, tuned)
    │
    ▼
joblib.dump(best_rf_model, 'best_crypto_model.pkl')
    │
    ▼
Saved artifact: best_crypto_model.pkl
```

### Loading for Inference

```python
import joblib
model = joblib.load('best_crypto_model.pkl')
predictions = model.predict(X_new_scaled)
```

**Note:** `X_new_scaled` must be preprocessed through the same ColumnTransformer pipeline before prediction.

---

## 7. Data Flow Diagram

```
dataset.csv
    │
    ▼
[RAW DataFrame]
 - timestamp, date, open, high, low
 - close, volume, marketCap, crypto_name
    │
    ▼ rename + type casting
[CLEANED RAW]
 - date(datetime), open, high, low
 - close, volume, marketCap, crypto_name
    │
    ▼ feature engineering
[ENRICHED DataFrame]
 - open, high, low, close, volume, marketCap
 - volatility (NEW), year, month, day, day_of_week
 - crypto_name
    │
    ▼ IQR capping
[OUTLIER-TREATED DataFrame]
    │
    ▼ PowerTransformer + StandardScaler + OHE
[SCALED DataFrame]
 - open, volume, marketCap, volatility (transformed)
 - year, month, day, day_of_week (standardized)
 - crypto_BTC, crypto_ETH, ... (one-hot)
    │
    ├──► X = drop volatility, high, close, low
    └──► y = volatility
         │
         ▼ train_test_split
    X_train, X_test, y_train, y_test
         │
         ▼ model.fit()
    Trained Models (7 baseline)
         │
         ▼ GridSearchCV
    best_rf_model (tuned)
         │
         ▼ joblib.dump()
    best_crypto_model.pkl
```

---

## 8. Reproducibility Checklist

| Item | Status |
|---|---|
| `random_state=42` set in all splits | ✅ |
| `random_state=42` in best model | ✅ |
| Warnings suppressed cleanly | ✅ |
| All imports at top of notebook | ✅ |
| Dataset path configurable via env var | ✅ |
| Model saved to disk | ✅ |
| `numeric_only=True` in `.corr()` | ✅ |
| `dtype=int` in `get_dummies` | ✅ |

---

## 9. Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.5.0
seaborn>=0.12.0
plotly>=5.10.0
scikit-learn>=1.1.0
xgboost>=1.7.0
joblib>=1.2.0
```

Install via:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost joblib
```

---

*Document version: 1.0 | Project: Crypto Volatility Prediction*
