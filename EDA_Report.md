# Exploratory Data Analysis (EDA) Report
## Cryptocurrency Market Data — Volatility Prediction Project

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| **File** | dataset.csv |
| **Target Variable** | `volatility` (derived: `high − low`) |
| **Domain** | Cryptocurrency Financial Markets |

### 1.1 Features Description

| Feature | Type | Description |
|---|---|---|
| `timestamp` | Temporal | Exact UTC datetime of observation |
| `date` | Temporal | Calendar date (daily granularity) |
| `open` | Numerical | Price at the start of trading period |
| `high` | Numerical | Maximum price during trading session |
| `low` | Numerical | Minimum price during trading session |
| `close` | Numerical | Price at end of trading period |
| `volume` | Numerical | Total units traded (scaled to Billions) |
| `marketCap` | Numerical | Total market capitalization at that time |
| `crypto_name` | Categorical | Name/symbol of the cryptocurrency |
| `ID` | Identifier | Internal unique row identifier |

---

## 2. Data Quality Assessment

### 2.1 Missing Values
- No significant missing values were found in the dataset.
- `pd.isna().sum()` was applied across all columns — all columns returned zero null counts.

### 2.2 Duplicate Records
- Checked using `df.duplicated().sum()`.
- No duplicate rows were found in the dataset.

### 2.3 Data Type Corrections
- `date` and `timestamp` columns were parsed from string to `datetime64` using `pd.to_datetime(..., errors='coerce')`.
- Columns with `dtype == 'O'` (object) were classified as categorical.

---

## 3. Feature Classification

| Category | Count | Columns |
|---|---|---|
| **Categorical** | 1 | `crypto_name` |
| **Numerical** | 7+ | `open`, `high`, `low`, `close`, `volume`, `marketCap`, `year`, `month`, `day`, `day_of_week` |
| **Dropped** | 2 | `timestamp` (redundant), `ID` (identifier only) |

---

## 4. Univariate Analysis

### 4.1 Numerical Features — KDE Plots

All numerical features were visualized using KDE (Kernel Density Estimation) plots. Key observations:

- **Price Skewness:** All OHLC features (`open`, `high`, `low`, `close`) are **highly right-skewed**. Prices cluster near lower values with rare extreme spikes.
- **Volume & MarketCap Outliers:** Long tail distributions indicate the presence of extreme outliers during market volatility events.
- **Year Distribution:** Dataset has dense records for **2020–2022**; very sparse data for years before 2017.
- **Constant Feature (Hour):** Hour is a constant zero — zero variance, provides no predictive value. **Dropped.**
- **Month & Day:** Relatively uniform distributions — data collection was consistent throughout the year.

### 4.2 Categorical Features — Count Plots

- `crypto_name` shows a large number of unique cryptocurrency entries.
- Some cryptocurrencies (Bitcoin, Ethereum) have more historical records than newer coins.
- Distribution is **not uniform** — major coins dominate data density.

---

## 5. Correlation Analysis

A **correlation heatmap** was generated over all numerical columns after feature engineering.

### Key Findings:
- `open`, `high`, `low`, `close` show **near-perfect positive correlation (~0.99)** with each other → **Multicollinearity detected.**
- `volatility` (high−low) is moderately correlated with OHLC features.
- `volume` shows **lower correlation** with OHLC and volatility.
- `date`-derived columns (`year`, `month`, `day`) have **near-zero correlation** with price features.

---

## 6. Feature Engineering

The following new features were derived from existing columns:

| New Feature | Formula / Source | Rationale |
|---|---|---|
| `volatility` | `high − low` | Core target variable; captures intraday price swing |
| `year` | `date.dt.year` | Captures long-term market cycles |
| `month` | `date.dt.month` | Captures seasonal patterns |
| `day` | `date.dt.day` | Captures intramonth patterns |
| `day_of_week` | `date.dt.dayofweek` | Captures weekday trading behavior |

The original `date` column was **dropped** after extraction to avoid redundancy.

---

## 7. Multivariate Analysis

### 7.1 Yearly Volatility Trend (Line Plot)

| Period | Observation |
|---|---|
| 2013–2016 | Near-zero volatility — market was tiny and illiquid |
| 2017 | **First major spike** — the crypto bull run |
| 2018–2020 | Elevated but stable baseline volatility |
| 2021 | **All-time peak volatility** — institutional adoption, DEFI boom |
| 2022 | Significant drop — market consolidation / bear market |

### 7.2 Top 10 Most Volatile Cryptocurrencies (Bar Plot)
- **Wrapped Bitcoin (WBTC)** showed an anomalous extreme spike — confirmed as a data outlier.
- Bitcoin itself does not show such extreme daily swings, confirming the WBTC data point is erroneous.

### 7.3 Price Trend of Major Coins (FacetGrid)
Coins analyzed: Bitcoin, Ethereum, XRP, Cardano, Litecoin.
- All coins show synchronised bull-runs, confirming **market-wide correlation**.
- Bitcoin and Ethereum's peaks dominate in absolute price terms.

---

## 8. Outlier Detection & Treatment

### 8.1 Detection: Box Plots
Box plots were generated for all numerical features before treatment. All features — especially `open`, `high`, `low`, `close`, `volume`, `marketCap`, `volatility` — showed heavy outliers due to extreme market events.

### 8.2 Treatment: IQR Capping
The **Interquartile Range (IQR)** method was used:

```
lower_limit = Q1 − 1.5 × IQR
upper_limit = Q3 + 1.5 × IQR
Values beyond these limits are capped (Winsorization)
```

Columns treated: `open`, `high`, `low`, `close`, `volume`, `marketCap`, `volatility`

Post-treatment box plots confirmed significant reduction in outlier severity.

---

## 9. Data Preprocessing for Modeling

### 9.1 Skewness Treatment
- **Power Transformer (Yeo-Johnson)** was applied to skewed features (OHLC, volume, marketCap, volatility) to achieve near-normal distributions.
- **StandardScaler** was applied to temporal features (year, month, day, day_of_week).

### 9.2 Categorical Encoding
- `crypto_name` was one-hot encoded using `pd.get_dummies()` with `dtype=int`.

### 9.3 Feature Selection for Modeling
Due to Multicollinearity among OHLC features, only `open` was retained as a representative price feature. Dropped from X: `high`, `close`, `low`.

Final feature set passed to models:
- `open`, `volume`, `marketCap`, `volatility` (target), `year`, `month`, `day`, `day_of_week`, all crypto one-hot columns.

---

## 10. EDA Summary & Key Insights

| Insight | Action Taken |
|---|---|
| OHLC columns are multicollinear | Dropped redundant price columns from X |
| Data is right-skewed | Applied PowerTransformer |
| Extreme outliers in financial data | Applied IQR capping (Winsorization) |
| Temporal features extracted | `year`, `month`, `day`, `day_of_week` added |
| `timestamp`, `ID` are redundant | Dropped early in pipeline |
| WBTC data point is an anomaly | Handled via outlier capping |
| `hour` has zero variance | Dropped (constant feature) |

---

*Report generated from: `Crypto_Project_Fixed.ipynb`*
