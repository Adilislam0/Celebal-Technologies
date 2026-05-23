# =============================================================================
# END-TO-END ML PIPELINE — Tesla Deliveries & Production Data (2015–2025)
# Dataset : https://www.kaggle.com/datasets/nalisha/tesla-ea-deliveries-and-production-data20152025
# Target  : Estimated_Deliveries
# Course  : Data Science001
# =============================================================================

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    cross_val_score, TimeSeriesSplit, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

TARGET = 'Estimated_Deliveries'


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


def eval_model(name, pipeline, X_tr, y_tr, X_te, y_te):
    """Fit pipeline, evaluate on train+test, return test predictions."""
    pipeline.fit(X_tr, y_tr)
    tr_pred = pipeline.predict(X_tr)
    te_pred = pipeline.predict(X_te)

    tr_rmse = np.sqrt(mean_squared_error(y_tr, tr_pred))
    te_rmse = np.sqrt(mean_squared_error(y_te, te_pred))
    tr_r2   = r2_score(y_tr, tr_pred)
    te_r2   = r2_score(y_te, te_pred)
    te_mae  = mean_absolute_error(y_te, te_pred)

    print(f"\n  ── {name}")
    print(f"     Train RMSE : {tr_rmse:>12,.0f}  |  Test RMSE : {te_rmse:>12,.0f}")
    print(f"     Train R²   : {tr_r2:>12.4f}  |  Test R²   : {te_r2:>12.4f}")
    print(f"     Test  MAE  : {te_mae:>12,.0f}")

    # Bias-Variance commentary
    gap = tr_r2 - te_r2
    if tr_r2 > 0.95 and gap > 0.20:
        verdict = "⚠  HIGH VARIANCE  → Overfitting (large train–test R² gap)"
    elif tr_r2 < 0.50:
        verdict = "⚠  HIGH BIAS      → Underfitting (poor train fit)"
    else:
        verdict = "✓  Balanced bias-variance tradeoff"
    print(f"     Verdict    : {verdict}")

    return te_pred, te_rmse, te_r2


# =============================================================================
# STEP 1 — TYPES OF ML
# =============================================================================
section("STEP 1 · TYPES OF ML")
print("""
  Problem Type : SUPERVISED LEARNING — REGRESSION
  ─────────────────────────────────────────────────
  • Input  (X) : time index, production figures, lag/rolling features, quarter dummies
  • Output (y) : Estimated_Deliveries  (continuous, numeric)
  • Goal        : Learn a mapping X → y so we can forecast future deliveries
""")


# =============================================================================
# STEP 2 — LOAD DATA
# =============================================================================
section("STEP 2 · LOAD DATA")

# ── locate CSV ────────────────────────────────────────────────────────────────
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print(f"  CSV files found in working directory: {csv_files}")

if not csv_files:
    raise FileNotFoundError(
        "No CSV found. Download the dataset from Kaggle and place it in "
        "the same directory as this script."
    )

df = pd.read_csv(csv_files[0])
print(f"  Loaded  : {csv_files[0]}")
print(f"  Shape   : {df.shape}")
print(f"\n  Columns : {df.columns.tolist()}")
print(f"\n  Head:\n{df.head().to_string()}")
print(f"\n  dtypes:\n{df.dtypes.to_string()}")


# =============================================================================
# STEP 3 — ML PIPELINE OVERVIEW
# =============================================================================
section("STEP 3 · ML PIPELINE OVERVIEW")
print("""
  Raw Data
     │
     ▼
  [EDA] ──── understand distributions, correlations, outliers
     │
     ▼
  [Data Cleaning] ──── duplicates, missing values, outlier flagging
     │
     ▼
  [Feature Engineering] ──── time index, quarter extraction
     │
     ▼
  [Time Series Analysis] ──── decompose, stationarity (ADF)
     │
     ▼
  [Lag Features + Rolling Stats] ──── temporal predictors (no leakage)
     │
     ▼
  [Encoding] ──── one-hot for quarter dummies
     │
     ▼
  [Chronological Split] ──── 80 % train | 20 % test  (no shuffling)
     │
     ┌────────────────────────────────────┐
     ▼                                    ▼
  [sklearn Pipeline]           [sklearn Pipeline]  ...
  StandardScaler + LR      StandardScaler + Ridge/Lasso
     │
     ▼
  [Cross-Validation → TimeSeriesSplit]
     │
     ▼
  [Hyperparameter Tuning → GridSearchCV]
     │
     ▼
  [Evaluation Metrics: RMSE, MAE, R², MAPE]
     │
     ▼
  [Forecasting + Bias-Variance Tradeoff Plot]
""")


# =============================================================================
# STEP 4 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
section("STEP 4 · EXPLORATORY DATA ANALYSIS (EDA)")

print("\n  --- Basic Info ---")
df.info()

print("\n  --- Descriptive Statistics ---")
print(df.describe().to_string())

print("\n  --- Missing Values ---")
print(df.isnull().sum().to_string())

print("\n  --- Cardinality per Column ---")
print(df.nunique().to_string())

# ── EDA plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('EDA – Tesla Deliveries & Production Dataset', fontsize=14, fontweight='bold')

# 1. Target distribution
axes[0, 0].hist(df[TARGET].dropna(), bins=20, color='steelblue', edgecolor='black')
axes[0, 0].set_title(f'Distribution of {TARGET}')
axes[0, 0].set_xlabel(TARGET)
axes[0, 0].set_ylabel('Frequency')

# 2. Correlation heatmap (numeric only)
num_df = df.select_dtypes(include=[np.number])
if num_df.shape[1] > 1:
    sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                ax=axes[0, 1], linewidths=0.5)
    axes[0, 1].set_title('Correlation Heatmap')

# 3. Target over time (index order)
axes[1, 0].plot(df[TARGET].values, color='darkblue', linewidth=2, marker='o', markersize=4)
axes[1, 0].set_title(f'{TARGET} Over Records')
axes[1, 0].set_xlabel('Record Index')
axes[1, 0].set_ylabel(TARGET)

# 4. Boxplot of target
axes[1, 1].boxplot(df[TARGET].dropna(), patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
axes[1, 1].set_title(f'Boxplot – {TARGET}')
axes[1, 1].set_ylabel(TARGET)

plt.tight_layout()
plt.savefig('01_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved → 01_eda.png")


# =============================================================================
# STEP 5 — DATA CLEANING
# =============================================================================
section("STEP 5 · DATA CLEANING")

print(f"  Shape before : {df.shape}")

# Duplicates
dup = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"  Duplicate rows removed : {dup}")

# Missing values
for col in df.columns:
    n_miss = df[col].isnull().sum()
    if n_miss == 0:
        continue
    if df[col].dtype in [np.float64, np.int64]:
        fill = df[col].median()
        strategy = "median"
    else:
        fill = df[col].mode()[0]
        strategy = "mode"
    df[col].fillna(fill, inplace=True)
    print(f"  Filled {n_miss:>3} NaN in '{col}' with {strategy} ({fill})")

# Outlier flagging (IQR) — NOT removed (real delivery spikes are meaningful)
Q1, Q3 = df[TARGET].quantile(0.25), df[TARGET].quantile(0.75)
IQR = Q3 - Q1
df['is_outlier'] = (
    (df[TARGET] < Q1 - 1.5 * IQR) | (df[TARGET] > Q3 + 1.5 * IQR)
).astype(int)
print(f"\n  Outlier records flagged (not dropped): {df['is_outlier'].sum()}")
print(f"  Shape after  : {df.shape}")


# =============================================================================
# STEP 6 — FEATURE ENGINEERING
# =============================================================================
section("STEP 6 · FEATURE ENGINEERING")

cols = df.columns.tolist()
print(f"  Columns available: {cols}")

# ── Parse time column ─────────────────────────────────────────────────────────
# Expected: 'Quarter' column like 'Q1 2020' or 'Q1-2020'
# Adjust the regex below if your dataset uses a different format.

if 'Quarter' in df.columns:
    raw = df['Quarter'].astype(str)
    df['Year']    = raw.str.extract(r'(\d{4})').astype(float)
    df['Q_num']   = raw.str.extract(r'Q(\d)').astype(float)
    df['time_idx'] = (df['Year'] - df['Year'].min()) * 4 + df['Q_num'] - 1
    print("  Extracted: Year, Q_num, time_idx  from 'Quarter'")
elif 'Year' in df.columns and 'Quarter' not in df.columns:
    df['time_idx'] = range(len(df))
    print("  Created sequential time_idx from Year column")
else:
    df['time_idx'] = range(len(df))
    print("  Created sequential time_idx (no Quarter column found)")

# Sort chronologically
df.sort_values('time_idx', inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  Sorted by time_idx. Shape: {df.shape}")


# =============================================================================
# STEP 7 — TIME SERIES COMPONENTS
# =============================================================================
section("STEP 7 · TIME SERIES COMPONENTS")

ts = df[TARGET].dropna().reset_index(drop=True)
print(f"  Series length: {len(ts)}")

if len(ts) >= 8:
    try:
        period = 4  # quarterly data → annual seasonality
        decomp = seasonal_decompose(ts, model='additive', period=period)

        fig2, axes2 = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
        fig2.suptitle('Time Series Decomposition – Estimated Deliveries',
                      fontsize=13, fontweight='bold')
        for ax, data, label, color in zip(
            axes2,
            [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
            ['Observed', 'Trend', 'Seasonal', 'Residual'],
            ['steelblue', 'green', 'orange', 'red']
        ):
            ax.plot(data.values, color=color, linewidth=1.8)
            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, alpha=0.4)

        plt.tight_layout()
        plt.savefig('02_ts_decomposition.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("  Saved → 02_ts_decomposition.png")
    except Exception as exc:
        print(f"  Decomposition skipped: {exc}")
else:
    print("  Series too short for decomposition (need ≥ 8 records)")


# =============================================================================
# STEP 8 — STATIONARITY (ADF TEST)
# =============================================================================
section("STEP 8 · STATIONARITY — Augmented Dickey-Fuller Test")

adf = adfuller(ts)
print(f"\n  ADF Statistic  : {adf[0]:.4f}")
print(f"  p-value        : {adf[1]:.4f}")
print("  Critical Values:")
for k, v in adf[4].items():
    print(f"    {k}: {v:.4f}")

if adf[1] < 0.05:
    print("\n  → Series is STATIONARY (reject H₀ at 5 %)")
else:
    print("\n  → Series is NON-STATIONARY → applying first-order differencing")
    df['Deliveries_diff'] = df[TARGET].diff()


# =============================================================================
# STEP 9 — LAG FEATURES
# =============================================================================
section("STEP 9 · LAG FEATURES")

for lag in [1, 2, 4]:
    col = f'lag_{lag}Q'
    df[col] = df[TARGET].shift(lag)
    print(f"  Created {col}  (shift={lag})")

print("\n  Why shift? → prevents data leakage; lag_1Q uses last quarter's value,")
print("  which IS available at prediction time.")


# =============================================================================
# STEP 10 — ROLLING STATISTICS
# =============================================================================
section("STEP 10 · ROLLING STATISTICS")

# Always shift(1) BEFORE rolling so future information never leaks
df['roll_mean_2Q'] = df[TARGET].shift(1).rolling(window=2).mean()
df['roll_mean_4Q'] = df[TARGET].shift(1).rolling(window=4).mean()
df['roll_std_4Q']  = df[TARGET].shift(1).rolling(window=4).std()
print("  Created: roll_mean_2Q, roll_mean_4Q, roll_std_4Q  (shift(1) applied first)")

fig3, ax3 = plt.subplots(figsize=(13, 5))
ax3.plot(df[TARGET].values,      label='Actual Deliveries', color='steelblue', linewidth=2)
ax3.plot(df['roll_mean_4Q'].values, label='4Q Rolling Mean',  color='orange',   linewidth=2, linestyle='--')
ax3.plot(df['roll_mean_2Q'].values, label='2Q Rolling Mean',  color='green',    linewidth=1.5, linestyle=':')
ax3.set_title('Estimated Deliveries vs Rolling Means', fontsize=13)
ax3.set_xlabel('Record Index')
ax3.set_ylabel('Deliveries')
ax3.legend()
plt.tight_layout()
plt.savefig('03_rolling_stats.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved → 03_rolling_stats.png")


# =============================================================================
# STEP 11 — ENCODING TECHNIQUES
# =============================================================================
section("STEP 11 · ENCODING TECHNIQUES")

# One-hot encode Q_num (nominal, 4 categories)
if 'Q_num' in df.columns:
    df = pd.get_dummies(df, columns=['Q_num'], prefix='Q', dtype=int)
    print("  One-hot encoded Q_num → quarter dummy variables (Q_1 … Q_4)")

# Handle any remaining object columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"  Remaining categorical columns: {cat_cols}")
for col in cat_cols:
    if df[col].nunique() <= 15:
        df = pd.get_dummies(df, columns=[col], drop_first=True, dtype=int)
        print(f"  One-hot encoded: {col}")
    else:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Label encoded  : {col}")


# =============================================================================
# STEP 12 — DATA LEAKAGE CHECK + FEATURE MATRIX
# =============================================================================
section("STEP 12 · DATA LEAKAGE — Feature Matrix Preparation")

# Drop rows with NaN (from lag/rolling creation)
before = len(df)
df.dropna(inplace=True)
print(f"  Dropped {before - len(df)} rows with NaN (from lag/rolling windows)")
print(f"  Final clean shape: {df.shape}")

# Columns to exclude from X
DROP_COLS = [TARGET, 'Quarter', 'Quarter_str', 'Deliveries_diff']
DROP_COLS = [c for c in DROP_COLS if c in df.columns]

feature_cols = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in DROP_COLS
]
print(f"\n  Features ({len(feature_cols)}): {feature_cols}")
print(f"  Target : {TARGET}")

print("""
  DATA LEAKAGE CHECKLIST
  ──────────────────────
  ✓ Lag features: shift(n) ensures only past values used
  ✓ Rolling stats: shift(1) before rolling → no future leakage
  ✓ Chronological split: train < test in time → no shuffling
  ✓ StandardScaler: fit ONLY on X_train → no test info leaks
  ✓ GridSearchCV: uses TimeSeriesSplit → respects temporal order
""")

X = df[feature_cols].copy()
y = df[TARGET].copy()


# =============================================================================
# STEP 13 — CHRONOLOGICAL SPLIT
# =============================================================================
section("STEP 13 · CHRONOLOGICAL SPLIT (Time-Aware Train / Test Split)")

split_idx = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"  Total samples : {len(df)}")
print(f"  Train         : {len(X_train)} rows  (80 %)")
print(f"  Test          : {len(X_test)}  rows  (20 %)")
print("  → NO random shuffling — temporal order preserved")

# Visualize split
fig4, ax4 = plt.subplots(figsize=(13, 4))
ax4.plot(range(len(y_train)), y_train.values,
         color='steelblue', linewidth=2, label='Train')
ax4.plot(range(len(y_train), len(y_train) + len(y_test)), y_test.values,
         color='green', linewidth=2, label='Test')
ax4.axvline(x=len(y_train) - 1, color='red', linestyle='--', linewidth=1.5, label='Split')
ax4.set_title('Chronological Train / Test Split — Estimated Deliveries')
ax4.set_xlabel('Time Index')
ax4.set_ylabel('Deliveries')
ax4.legend()
plt.tight_layout()
plt.savefig('04_chronological_split.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved → 04_chronological_split.png")


# =============================================================================
# STEP 14 — SKLEARN PIPELINES  (Scaler + Model)
# =============================================================================
section("STEP 14 · SKLEARN PIPELINES — Feature Scaling inside Pipeline")

print("""
  Why use sklearn Pipeline?
  • Chains preprocessing + model in one object
  • Prevents accidental data leakage (scaler sees only training folds)
  • Clean, reproducible, deployable

  StandardScaler: z = (x - μ) / σ  — applied per feature
""")

# ── 15. LINEAR REGRESSION ─────────────────────────────────────────────────────
section("STEP 15 · LINEAR REGRESSION")

lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LinearRegression())
])
lr_preds, lr_rmse, lr_r2 = eval_model(
    "Linear Regression", lr_pipe, X_train, y_train, X_test, y_test
)

# Coefficients
coef_series = pd.Series(
    lr_pipe.named_steps['model'].coef_,
    index=feature_cols
).sort_values(key=abs, ascending=False)
print("\n  Top-5 coefficients by magnitude:")
print(coef_series.head(5).to_string())


# ── 16. RIDGE REGRESSION (L2) ─────────────────────────────────────────────────
section("STEP 16 · RIDGE REGRESSION (L2 Regularization)")

print("  L2 penalty shrinks all coefficients toward zero — no feature elimination")

ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=1.0))
])
ridge_preds, ridge_rmse, ridge_r2 = eval_model(
    "Ridge (L2)  α=1", ridge_pipe, X_train, y_train, X_test, y_test
)


# ── 17. LASSO REGRESSION (L1) ─────────────────────────────────────────────────
section("STEP 17 · LASSO REGRESSION (L1 Regularization)")

print("  L1 penalty can drive coefficients to exactly zero → implicit feature selection")

lasso_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Lasso(alpha=1.0, max_iter=20_000))
])
lasso_preds, lasso_rmse, lasso_r2 = eval_model(
    "Lasso (L1)  α=1", lasso_pipe, X_train, y_train, X_test, y_test
)

lasso_coef = pd.Series(
    lasso_pipe.named_steps['model'].coef_,
    index=feature_cols
)
zeroed = (lasso_coef == 0).sum()
print(f"\n  Features zeroed out by Lasso: {zeroed} / {len(feature_cols)}")


# =============================================================================
# STEP 18 — CROSS VALIDATION  (TimeSeriesSplit)
# =============================================================================
section("STEP 18 · CROSS VALIDATION — TimeSeriesSplit (5 folds)")

tscv = TimeSeriesSplit(n_splits=5)
print(f"  Folds: 5  |  Strategy: TimeSeriesSplit (no data leakage across folds)\n")

for name, pipe in [
    ("Linear Regression", lr_pipe),
    ("Ridge (L2)        ", ridge_pipe),
    ("Lasso (L1)        ", lasso_pipe),
]:
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring='r2')
    print(f"  {name}  CV R²: {cv_scores.round(3)}  "
          f"→ Mean: {cv_scores.mean():.3f}  Std: {cv_scores.std():.3f}")


# =============================================================================
# STEP 19 — HYPERPARAMETER TUNING  (GridSearchCV)
# =============================================================================
section("STEP 19 · HYPERPARAMETER TUNING — GridSearchCV")

alpha_grid = [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]

# Ridge tuning
gs_ridge = GridSearchCV(
    Pipeline([('sc', StandardScaler()), ('m', Ridge())]),
    {'m__alpha': alpha_grid},
    cv=tscv, scoring='r2', n_jobs=-1
)
gs_ridge.fit(X_train, y_train)
best_ridge_alpha = gs_ridge.best_params_['m__alpha']
print(f"\n  Ridge  — Best α: {best_ridge_alpha:<8}  CV R²: {gs_ridge.best_score_:.4f}")

# Lasso tuning
gs_lasso = GridSearchCV(
    Pipeline([('sc', StandardScaler()), ('m', Lasso(max_iter=20_000))]),
    {'m__alpha': alpha_grid},
    cv=tscv, scoring='r2', n_jobs=-1
)
gs_lasso.fit(X_train, y_train)
best_lasso_alpha = gs_lasso.best_params_['m__alpha']
print(f"  Lasso  — Best α: {best_lasso_alpha:<8}  CV R²: {gs_lasso.best_score_:.4f}")

# Test performance after tuning
best_ridge_preds = gs_ridge.predict(X_test)
best_lasso_preds = gs_lasso.predict(X_test)

for name, preds in [("Tuned Ridge", best_ridge_preds), ("Tuned Lasso", best_lasso_preds)]:
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"  {name:<12} Test R²: {r2:.4f}   RMSE: {rmse:,.0f}")


# =============================================================================
# STEP 20 — EVALUATION METRICS SUMMARY
# =============================================================================
section("STEP 20 · EVALUATION METRICS SUMMARY")

all_results = {
    'Linear Regression': (lr_preds,         lr_rmse),
    'Ridge (tuned)':     (best_ridge_preds,  np.sqrt(mean_squared_error(y_test, best_ridge_preds))),
    'Lasso (tuned)':     (best_lasso_preds,  np.sqrt(mean_squared_error(y_test, best_lasso_preds))),
}

rows = []
for mname, (preds, _) in all_results.items():
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test.values - preds) / np.maximum(np.abs(y_test.values), 1))) * 100
    rows.append({'Model': mname, 'RMSE': f'{rmse:,.0f}', 'MAE': f'{mae:,.0f}',
                 'R²': f'{r2:.4f}', 'MAPE%': f'{mape:.2f}'})

summary = pd.DataFrame(rows)
print(f"\n{summary.to_string(index=False)}")
print("""
  Metric Guide
  ────────────
  RMSE  : penalises large errors heavily (same unit as target)
  MAE   : average absolute error (more robust to outliers)
  R²    : proportion of variance explained (1.0 = perfect)
  MAPE% : percentage error (scale-independent)
""")

# Best model by RMSE
best_name = min(all_results, key=lambda k: np.sqrt(mean_squared_error(y_test, all_results[k][0])))
best_preds = all_results[best_name][0]
print(f"  🏆 Best model: {best_name}")


# =============================================================================
# STEP 21 — FORECASTING PLOT
# =============================================================================
section("STEP 21 · FORECASTING")

fig5, ax5 = plt.subplots(figsize=(13, 5))
ax5.plot(range(len(y_train)), y_train.values,
         color='steelblue', linewidth=2, label='Train Actuals')
ax5.plot(range(len(y_train), len(y_train) + len(y_test)), y_test.values,
         color='green', linewidth=2.5, label='Test Actuals')
ax5.plot(range(len(y_train), len(y_train) + len(y_test)), best_preds,
         color='red', linewidth=2, linestyle='--',
         label=f'Forecast — {best_name}')
ax5.axvline(x=len(y_train) - 1, color='black', linestyle=':', linewidth=1.2)
ax5.set_title('Tesla Estimated Deliveries — Forecast vs Actual', fontsize=13)
ax5.set_xlabel('Time Index (Quarters)')
ax5.set_ylabel('Estimated Deliveries')
ax5.legend()
plt.tight_layout()
plt.savefig('05_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved → 05_forecast.png")


# =============================================================================
# STEP 22 — BIAS-VARIANCE TRADEOFF
# =============================================================================
section("STEP 22 · BIAS-VARIANCE TRADEOFF")

alphas_bv = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10_000]
train_r2s, test_r2s = [], []

for a in alphas_bv:
    p = Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=a))])
    p.fit(X_train, y_train)
    train_r2s.append(r2_score(y_train, p.predict(X_train)))
    test_r2s.append(r2_score(y_test,  p.predict(X_test)))

fig6, ax6 = plt.subplots(figsize=(11, 5))
ax6.semilogx(alphas_bv, train_r2s, 'b-o', linewidth=2, label='Train R²')
ax6.semilogx(alphas_bv, test_r2s,  'r-o', linewidth=2, label='Test R²')
ax6.axvline(x=best_ridge_alpha, color='green', linestyle='--', linewidth=1.5,
            label=f'Best α = {best_ridge_alpha}')
ax6.fill_between(alphas_bv, train_r2s, test_r2s, alpha=0.15, color='purple',
                 label='Variance gap')
ax6.set_title('Bias-Variance Tradeoff — Ridge Regularization Strength vs R²',
              fontsize=12)
ax6.set_xlabel('Alpha (log scale)  →  higher α = stronger regularisation = higher bias')
ax6.set_ylabel('R² Score')
ax6.legend()
ax6.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.savefig('06_bias_variance.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved → 06_bias_variance.png")

print("""
  INTERPRETATION
  ──────────────
  Low  α  → Model too flexible → memorises training noise
            → High Variance, Low Bias  → OVERFITTING
  High α  → Model too constrained → misses real patterns
            → Low Variance, High Bias → UNDERFITTING
  Sweet spot (green line) → best generalisation on test data
""")


# =============================================================================
# ✅  PIPELINE COMPLETE
# =============================================================================
section("✅  END-TO-END ML PIPELINE COMPLETE")

print(f"""
  Plots generated
  ───────────────
  01_eda.png                  — EDA distributions & correlations
  02_ts_decomposition.png     — Trend / Seasonal / Residual
  03_rolling_stats.png        — Rolling means overlay
  04_chronological_split.png  — Train-test boundary
  05_forecast.png             — Model forecast vs actuals
  06_bias_variance.png        — Regularisation tradeoff

  Best Model  : {best_name}
  Dataset     : Tesla EA Deliveries & Production 2015–2025
  Target      : {TARGET}

    This pipeline demonstrates a complete workflow for time series regression.
""")
