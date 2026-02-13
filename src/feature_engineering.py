import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
INPUT_PATH = Path("data/processed/fd001_master.csv")
OUTPUT_PATH = Path("data/processed/fd001_features.csv")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
df = pd.read_csv(INPUT_PATH)

print("Initial engine count:", df["engine_id"].nunique())

print("After load:", df["engine_id"].nunique())

df = df.drop_duplicates()
print("After drop_duplicates:", df["engine_id"].nunique())

df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
print("After sort:", df["engine_id"].nunique())

# --------------------------------------------------
# Sensor Columns
# --------------------------------------------------
SENSOR_COLS = [c for c in df.columns if c.startswith("sensor_")]
df[SENSOR_COLS] = df[SENSOR_COLS].apply(pd.to_numeric, errors="coerce")

# --------------------------------------------------
# Rolling + Trend Features (Built Safely in Dict)
# --------------------------------------------------
WINDOWS = [5, 10, 20]
feature_dict = {}

# Rolling features
for window in WINDOWS:
    for col in SENSOR_COLS:
        feature_dict[f"{col}_mean_{window}"] = (
            df.groupby("engine_id")[col]
              .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

        feature_dict[f"{col}_std_{window}"] = (
            df.groupby("engine_id")[col]
              .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

print("After rolling:", df["engine_id"].nunique())

# Trend features
def rolling_slope(series):
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    return np.polyfit(x, series.values, 1)[0]

for col in SENSOR_COLS:
    feature_dict[f"{col}_trend_10"] = (
        df.groupby("engine_id")[col]
          .transform(lambda x: x.rolling(10, min_periods=2)
                                 .apply(rolling_slope, raw=False))
    )

print("After trend:", df["engine_id"].nunique())

# --------------------------------------------------
# Concatenate ALL engineered features at once
# --------------------------------------------------
feature_df = pd.DataFrame(feature_dict)

df = pd.concat([df, feature_df], axis=1)

# --------------------------------------------------
# Normalized lifecycle
# --------------------------------------------------
df["normalized_cycle"] = (
    df["cycle"] /
    df.groupby("engine_id")["cycle"].transform("max")
)

# --------------------------------------------------
# Handle NaNs safely
# --------------------------------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df = df.fillna(0)

print("After fillna:", df["engine_id"].nunique())

# --------------------------------------------------
# Clip extreme outliers
# --------------------------------------------------
# Clip ONLY engineered sensor features
clip_cols = [
    col for col in df.columns
    if (
        col.startswith("sensor_") or
        "_mean_" in col or
        "_std_" in col or
        "_trend_" in col
    )
]

for col in clip_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = np.clip(df[col], lower, upper)


print("After clipping:", df["engine_id"].nunique())

# --------------------------------------------------
# Validation
# --------------------------------------------------
print("Final engine count after feature engineering:",
      df["engine_id"].nunique())

print("Final dataset shape:", df.shape)

# --------------------------------------------------
# Save
# --------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Feature engineering completed successfully")
print("📁 Output saved to:", OUTPUT_PATH)