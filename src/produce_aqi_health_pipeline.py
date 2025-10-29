# produce_aqi_health_pipeline.py
# Updated: cleaned warnings, improved anchoring, robust date handling
# Requirements: pandas, numpy, matplotlib, scipy, sklearn
# Install if needed: pip install pandas numpy matplotlib scipy scikit-learn

import os
import math
from datetime import date, timedelta, datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# ---------- CONFIG ----------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LIVE_AQI_PATH = "cleaned/live_pune_aqi.csv"   # this should exist from your gov API fetch
HIST_AQI_CSV = os.path.join(OUTPUT_DIR, "pune_aqi_12mo_synthetic.csv")
HEALTH_CSV = os.path.join(OUTPUT_DIR, "pune_health_synthetic.csv")
JOINED_CSV = os.path.join(OUTPUT_DIR, "pune_aqi_health_joined.csv")

DAYS = 365    # length of historical series
TODAY = date.today()
START_DATE = TODAY - timedelta(days=DAYS-1)  # inclusive

# Parameters for synthetic generation (tweak if needed)
BASELINE_AQI = 80        # will be adjusted by live anchor if available
SEASONAL_AMPLITUDE = 25  # seasonal swing (annual)
WEEKDAY_EFFECT = {0: -5, 1: -3, 2: 0, 3: 0, 4: 2, 5: 5, 6: 3}  # Mon=0 ... Sun=6
NOISE_STD = 8.0
HEALTH_BASELINE = 40    # avg daily respiratory cases baseline
HEALTH_ALPHA = 0.18     # immediate effect per AQI unit
HEALTH_LAG_ALPHA = 0.08 # lagged effect weight (applied to AQI lag 2-4)
HEALTH_NOISE_STD = 6.0

pd.set_option("display.width", 140)

# ---------- Step 1: Read live aqi anchor if available ----------
anchor_value = None
if os.path.exists(LIVE_AQI_PATH):
    try:
        live_df = pd.read_csv(LIVE_AQI_PATH)
        # Accept either 'date' or 'Date' and either 'AQI' or lowercase; normalize
        if 'date' in live_df.columns:
            date_col = 'date'
        elif 'Date' in live_df.columns:
            date_col = 'Date'
        else:
            date_col = None

        if date_col is not None:
            # parse date column robustly
            try:
                live_df[date_col] = pd.to_datetime(live_df[date_col], errors='coerce')
            except Exception:
                live_df[date_col] = pd.to_datetime(live_df[date_col].astype(str), errors='coerce')

            # pick AQI column (case-insensitive)
            aqi_col = None
            for c in live_df.columns:
                if str(c).lower() == "aqi":
                    aqi_col = c
                    break

            if aqi_col is not None:
                live_df = live_df[[date_col, aqi_col]].dropna(subset=[date_col, aqi_col]).copy()
                if not live_df.empty:
                    # use mean of up to last 7 days (if present)
                    live_df['date_only'] = live_df[date_col].dt.date
                    recent = live_df.sort_values(date_col).tail(7)
                    anchor_value = float(recent[aqi_col].mean())
                    print(f"Live anchor found: mean AQI from live file (last up to 7 rows) = {anchor_value:.2f}")
            else:
                print("Warning: no AQI column found in live file.")
        else:
            print("Warning: no date column found in live file.")
    except Exception as e:
        print("Warning: couldn't read live AQI file:", e)

if anchor_value is None:
    print("No live gov anchor found or invalid file. Using default baseline.")
    anchor_value = BASELINE_AQI

# Adjust baseline to anchor (so historical trend centers near anchor)
BASELINE_AQI = float(anchor_value)

# ---------- Step 2: Generate synthetic daily AQI series (realistic) ----------
dates = pd.date_range(START_DATE, periods=DAYS, freq="D")
df_aqi = pd.DataFrame({"date": dates})
day_of_year = df_aqi["date"].dt.dayofyear.values

# seasonal yearly cycle: sine wave mapped to amplitude
seasonal = SEASONAL_AMPLITUDE * np.sin(2 * math.pi * (day_of_year / 365.25) - 0.2)

# weekday effect
weekday = df_aqi["date"].dt.weekday
weekday_effect = np.array([WEEKDAY_EFFECT.get(int(w), 0) for w in weekday])

# trending small random walk to simulate multi-week variability
np.random.seed(42)
walk = np.cumsum(np.random.normal(loc=0.0, scale=0.8, size=len(dates)))
walk = (walk - np.mean(walk))  # zero mean

# combine
aqi_raw = BASELINE_AQI + seasonal + weekday_effect + walk + np.random.normal(0, NOISE_STD, size=len(dates))

# force min and sensible max
aqi = np.clip(aqi_raw, a_min=5, a_max=500)  # AQI-like bounds

df_aqi["AQI"] = np.round(aqi, 2)

# Anchor check: replace most recent few days with actual live values if available (vectorized)
if os.path.exists(LIVE_AQI_PATH):
    try:
        live_df = pd.read_csv(LIVE_AQI_PATH)
        # normalize columns
        # find date column
        for c in live_df.columns:
            if str(c).lower() == "date":
                live_date_col = c
                break
        else:
            live_date_col = None

        # find aqi column
        for c in live_df.columns:
            if str(c).lower() == "aqi":
                live_aqi_col = c
                break
        else:
            live_aqi_col = None

        if live_date_col and live_aqi_col:
            live_df[live_date_col] = pd.to_datetime(live_df[live_date_col], errors='coerce')
            live_df = live_df.dropna(subset=[live_date_col, live_aqi_col]).copy()
            if not live_df.empty:
                live_df['date_only'] = live_df[live_date_col].dt.date
                # Map date strings to values
                live_map = dict(zip(live_df['date_only'].astype(str), live_df[live_aqi_col].astype(float)))
                # vectorized replace
                df_aqi['date_str'] = df_aqi['date'].dt.date.astype(str)
                mask = df_aqi['date_str'].isin(live_map.keys())
                if mask.any():
                    df_aqi.loc[mask, 'AQI'] = df_aqi.loc[mask, 'date_str'].map(live_map).astype(float)
                    replaced = mask.sum()
                    print(f"Anchored {replaced} days of synthetic history to live AQI values.")
                df_aqi.drop(columns=['date_str'], inplace=True)
        else:
            # no usable columns
            pass
    except Exception as e:
        print("Warning: couldn't anchor to live file:", e)

# Save synthetic historical AQI
df_aqi.to_csv(HIST_AQI_CSV, index=False)
print(f"\nSaved synthetic historical AQI -> {HIST_AQI_CSV}  (rows: {len(df_aqi)})")

# ---------- Step 3: Create synthetic healthcare (respiratory) daily counts ----------
df_health = df_aqi.copy()
# Introduce lag columns for AQI (lag 1..7) and use bfill to avoid deprecated fillna(method=...)
for L in [1, 2, 3, 4, 7]:
    df_health[f"AQI_lag{L}"] = df_health["AQI"].shift(L)
# fill start values using backward fill (bfill) then forward fill as safety
df_health[[f"AQI_lag{L}" for L in [1,2,3,4,7]]] = df_health[[f"AQI_lag{L}" for L in [1,2,3,4,7]]].bfill().ffill()

# Admissions model: baseline + immediate effect + small multi-day lag effect + noise + weekly seasonality
lag_effect = df_health["AQI_lag2"] * 0.05 + df_health["AQI_lag3"] * 0.03 + df_health["AQI_lag4"] * 0.02
immediate = HEALTH_ALPHA * df_health["AQI"]
weekday_factor = df_health["date"].dt.weekday.map(WEEKDAY_EFFECT) * 0.2  # small day effect
noise = np.random.normal(0, HEALTH_NOISE_STD, size=len(df_health))

df_health["respiratory_cases"] = (HEALTH_BASELINE + immediate + lag_effect + weekday_factor + noise).round().astype(int)
df_health["respiratory_cases"] = df_health["respiratory_cases"].clip(lower=0)

# Save health CSV
df_health[["date", "AQI", "respiratory_cases"]].to_csv(HEALTH_CSV, index=False)
print(f"Saved synthetic healthcare data -> {HEALTH_CSV}")

# ---------- Step 4: Join and prepare analysis dataset ----------
df_join = pd.merge(df_aqi, df_health[["date", "respiratory_cases"]], on="date", how="left")
# Ensure date column is datetime
df_join['date'] = pd.to_datetime(df_join['date'])
df_join.to_csv(JOINED_CSV, index=False)
print(f"Saved joined AQI+Health -> {JOINED_CSV}")

# ---------- Step 5: Analysis: correlation, lagged correlation, simple regression ----------
# Pearson correlation immediate
r, p = pearsonr(df_join["AQI"], df_join["respiratory_cases"])
print(f"\nPearson correlation (AQI vs Respiratory cases): r = {r:.3f}, p = {p:.3g}")

# lagged correlations (AQI lag 0..7 vs cases)
print("\nLagged correlations (AQI lag d vs respiratory_cases):")
for lag in range(0, 8):
    col = df_join["AQI"].shift(lag).bfill()
    r_l, p_l = pearsonr(col, df_join["respiratory_cases"])
    print(f"  lag {lag:>2}: r = {r_l:.3f}, p = {p_l:.3g}")

# Simple linear regression (cases ~ AQI + day_of_week)
X = pd.DataFrame()
X["AQI"] = df_join["AQI"]
X["dow"] = df_join["date"].dt.weekday
# One-hot encode dow for regression
X = pd.get_dummies(X, columns=["dow"], drop_first=True)
y = df_join["respiratory_cases"].values

model = LinearRegression()
model.fit(X, y)
coef_aqi = model.coef_[0]
r2 = model.score(X, y)
print(f"\nLinear regression: respiratory_cases ~ AQI + day_of_week")
print(f"  coefficient for AQI = {coef_aqi:.3f} (per 1 AQI unit)")
print(f"  R^2 = {r2:.3f}")

# ---------- Step 6: Plots ----------
plt.style.use("seaborn-v0_8")
# 1) Time series: AQI and cases
fig, ax1 = plt.subplots(figsize=(12,5))
ax1.plot(df_join["date"], df_join["AQI"], label="AQI", linewidth=1.2)
ax1.set_ylabel("AQI")
ax2 = ax1.twinx()
ax2.plot(df_join["date"], df_join["respiratory_cases"], color="tab:orange", label="Resp Cases", linewidth=1)
ax2.set_ylabel("Respiratory cases")
ax1.set_title("Pune: Synthetic 12-month AQI (anchored) and Respiratory Cases")
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "timeseries_aqi_cases.png"))
plt.close(fig)

# 2) Scatter and regression line
fig2, ax = plt.subplots(figsize=(7,5))
ax.scatter(df_join["AQI"], df_join["respiratory_cases"], alpha=0.6, s=18)
# regression line on AQI only (use coef from model which corresponds to AQI)
xa = np.linspace(df_join["AQI"].min(), df_join["AQI"].max(), 100)
ax.plot(xa, coef_aqi*xa + model.intercept_, color="red", linewidth=1.5)
ax.set_xlabel("AQI")
ax.set_ylabel("Respiratory cases")
ax.set_title(f"AQI vs Respiratory cases (r={r:.2f})")
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_aqi_cases.png"))
plt.close(fig2)

# 3) Lag correlation bar plot
lags = list(range(0,8))
corrs = []
for lag in lags:
    col = df_join["AQI"].shift(lag).bfill()
    corr = pearsonr(col, df_join["respiratory_cases"])[0]
    corrs.append(corr)

fig3, ax3 = plt.subplots(figsize=(8,4))
ax3.bar(lags, corrs)
ax3.set_xlabel("AQI lag (days)")
ax3.set_ylabel("Pearson r")
ax3.set_title("Lagged correlation AQI -> respiratory cases")
plt.savefig(os.path.join(OUTPUT_DIR, "lag_corr.png"))
plt.close(fig3)

print(f"\nPlots saved in '{OUTPUT_DIR}': timeseries_aqi_cases.png, scatter_aqi_cases.png, lag_corr.png")

# ---------- Step 7: Quick sample prints for slides ----------
print("\nSample rows (joined):")
print(df_join.tail(8).to_string(index=False))

print("\nDone. Deliverables created:")
print(f" - Historical AQI (synthetic) : {HIST_AQI_CSV}")
print(f" - Healthcare (synthetic)     : {HEALTH_CSV}")
print(f" - Joined dataset             : {JOINED_CSV}")
print(f" - Plots folder               : {OUTPUT_DIR}")

# End of script
