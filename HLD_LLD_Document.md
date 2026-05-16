# High-Level Design (HLD) & Low-Level Design (LLD)
## Cryptocurrency Volatility Prediction System

---

# PART 1 — HIGH-LEVEL DESIGN (HLD)

## 1.1 Project Objective

To build a machine learning system that **predicts cryptocurrency market volatility** (the intraday price swing: `high − low`) using historical OHLCV (Open, High, Low, Close, Volume) data across multiple cryptocurrencies.

---

## 1.2 System Overview

```
Raw CSV Data
     │
     ▼
Data Ingestion & Validation
     │
     ▼
Exploratory Data Analysis (EDA)
     │
     ▼
Feature Engineering & Preprocessing Pipeline
     │
     ▼
Model Training & Evaluation
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Best Model Serialization (best_crypto_model.pkl)
     │
     ▼
Predictions / Deployment
```

---

## 1.3 High-Level Components

| Component | Responsibility |
|---|---|
| **Data Layer** | Load CSV, validate schema, handle nulls/duplicates |
| **EDA Module** | Statistical summaries, univariate & multivariate plots |
| **Feature Engineering** | Volatility derivation, date decomposition, encoding |
| **Preprocessing Pipeline** | Outlier capping, power transformation, scaling |
| **Modeling Engine** | Train/evaluate 7 regressors, compare metrics |
| **Tuning Module** | GridSearchCV on the best model (Random Forest) |
| **Persistence Layer** | Save best model with joblib |

---

## 1.4 Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | scikit-learn, XGBoost |
| Model Persistence | joblib |
| Development Environment | Jupyter Notebook / Google Colab |

---

## 1.5 Input / Output Specification

| | Detail |
|---|---|
| **Input** | `dataset.csv` — multi-cryptocurrency OHLCV time series |
| **Output** | Predicted volatility value (continuous, normalized) |
| **Saved Model** | `best_crypto_model.pkl` (Random Forest Regressor) |

---

## 1.6 Evaluation Metrics

| Metric | Description |
|---|---|
| **R² Score** | Proportion of variance explained by the model |
| **MSE** | Mean Squared Error — average squared prediction error |
| **RMSE** | Root MSE — same scale as target variable |
| **MAE** | Mean Absolute Error — average absolute deviation |

**Best Achieved (Random Forest Tuned):** R² ≈ 0.98, MAE ≈ very low

---

## 1.7 Models Evaluated

| Model | Type |
|---|---|
| Linear Regression | Parametric baseline |
| Decision Tree Regressor | Non-linear, single tree |
| Random Forest Regressor ⭐ | Ensemble (bagging) — **winner** |
| Gradient Boosting Regressor | Ensemble (boosting) |
| AdaBoost Regressor | Ensemble (adaptive boosting) |
| XGBoost Regressor | Extreme gradient boosting |
| K-Neighbors Regressor | Instance-based |

---

---

# PART 2 — LOW-LEVEL DESIGN (LLD)

## 2.1 Data Ingestion Module

### Function: Load Dataset

```python
df = pd.read_csv(dataset_path)
df = df.rename(columns={'Unnamed: 0': 'ID'})
```

**Steps:**
1. Read CSV from configurable `dataset_path`.
2. Rename unnamed index column to `ID`.
3. Convert `date` and `timestamp` to `datetime64` using `errors='coerce'` (invalid values become `NaT`).
4. Check nulls: `df.isna().sum()` — assert all zeros.
5. Check duplicates: `df.duplicated().sum()` — assert zero.
6. Drop `timestamp` and `ID` (redundant/identifier columns).

---

## 2.2 Feature Engineering Module

### 2.2.1 Volatility Derivation (Target Variable)

```python
df['volatility'] = df['high'] - df['low']
```

Captures the **intraday price swing** — the core target variable.

### 2.2.2 Date Decomposition

```python
df['year']        = df['date'].dt.year
df['month']       = df['date'].dt.month
df['day']         = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df.drop(['date'], axis=1, inplace=True)
```

Decomposes the date into cyclic temporal features.

---

## 2.3 Outlier Treatment Module

### Function: `cap_outliers(df_input, columns)`

**Algorithm:** Winsorization via IQR

```
For each column:
    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    values > upper_limit  →  capped to upper_limit
    values < lower_limit  →  capped to lower_limit
```

**Input:** DataFrame, list of column names
**Output:** Cleaned DataFrame copy
**Columns treated:** `open`, `high`, `low`, `close`, `volume`, `marketCap`, `volatility`

---

## 2.4 Preprocessing Pipeline (sklearn ColumnTransformer)

```
ColumnTransformer
├── Pipeline 1: outlier_features_pipeline  →  applied to OHLCV + volatility
│   ├── SimpleImputer(strategy='median')
│   └── PowerTransformer(standardize=True)   # Yeo-Johnson transformation
│
└── Pipeline 2: numeric_pipeline  →  applied to year, month, day, day_of_week
    ├── SimpleImputer(strategy='constant', fill_value=0)
    └── StandardScaler()
```

**Remainder:** `crypto_name` passed through as-is → one-hot encoded separately.

### One-Hot Encoding

```python
scaled_data = pd.get_dummies(scaled_data, columns=['crypto_name'], dtype=int)
```

---

## 2.5 Feature Selection

After preprocessing:

```python
X = scaled_data.drop(columns=['volatility', 'high', 'close', 'low'])
y = scaled_data['volatility']
```

**Reason for dropping `high`, `close`, `low`:** Near-perfect multicollinearity with `open` (correlation ≈ 0.99). Retaining all would inflate variance with no additional information.

---

## 2.6 Model Training Module

### Function: `evaluate_models(X, y, models)`

```
Input:  X (features), y (target), models dict
Output: performance report DataFrame, trained_models dict,
        X_train, X_test, y_train, y_test

Algorithm:
  1. train_test_split(X, y, test_size=0.2, random_state=42)
  2. For each model:
       a. model.fit(X_train, y_train)
       b. y_pred = model.predict(X_test)
       c. Compute MSE, RMSE, MAE, R²
       d. Store results
  3. Return consolidated report + trained models
```

---

## 2.7 Hyperparameter Tuning Module

**Method:** GridSearchCV with 3-fold cross-validation

**Model:** RandomForestRegressor

**Search Space:**

| Parameter | Values Searched |
|---|---|
| `n_estimators` | [100, 200] |
| `max_depth` | [None, 10] |
| `min_samples_split` | [2, 5] |
| `min_samples_leaf` | [1, 2] |
| `max_features` | ['sqrt', 'log2'] |
| `criterion` | ['squared_error'] |

**Best Parameters Found:**

| Parameter | Best Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | None |
| `max_features` | 'sqrt' |
| `min_samples_leaf` | 1 |
| `min_samples_split` | 2 |
| `criterion` | 'squared_error' |

---

## 2.8 Model Serialization Module

```python
import joblib
joblib.dump(best_rf_model, "best_crypto_model.pkl")
```

Saves the trained, tuned Random Forest model to disk for future inference without retraining.

### Loading for Inference

```python
import joblib
model = joblib.load("best_crypto_model.pkl")
y_pred = model.predict(X_new)
```

---

## 2.9 Error Handling Strategy

| Scenario | Handling |
|---|---|
| File not found | `os.getenv('DATASET_PATH')` with fallback |
| Invalid date values | `errors='coerce'` → becomes `NaT` |
| Type mismatch in features | `pd.to_numeric(col, errors='coerce')` |
| Skewed distributions | PowerTransformer applied pre-modeling |
| Extreme outliers | IQR capping applied before pipeline |

---

## 2.10 Output Summary

| Artifact | Description |
|---|---|
| `Crypto_Project_Fixed.ipynb` | Cleaned, annotated source notebook |
| `best_crypto_model.pkl` | Serialized best model |
| Evaluation plots | Actual vs Predicted scatter, Residual distribution |
| Model report | DataFrame with MSE, RMSE, MAE, R² for all 7 models |

---

*Document version: 1.0 | Project: Crypto Volatility Prediction*
