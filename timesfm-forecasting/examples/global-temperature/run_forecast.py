#!/usr/bin/env python3
"""
Run TimesFM forecast on global temperature anomaly data.
Generates forecast output CSV and JSON for the example.

Uses TimesFM 2.5 (PyTorch) for one-shot time series forecasting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Preflight check
print("=" * 60)
print("  TimesFM FORECAST - Global Temperature Anomaly Example")
print("=" * 60)

# Load data
data_path = Path(__file__).parent / "temperature_anomaly.csv"
df = pd.read_csv(data_path, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"\n📊 Input Data: {len(df)} months of temperature anomalies")
print(
    f"   Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}"
)
print(f"   Mean anomaly: {df['anomaly_c'].mean():.2f}°C")
print(
    f"   Trend: {df['anomaly_c'].iloc[-12:].mean() - df['anomaly_c'].iloc[:12].mean():.2f}°C change (first to last year)"
)

# Prepare input for TimesFM
# TimesFM expects a list of 1D numpy arrays
input_series = df["anomaly_c"].values.astype(np.float32)

# Load TimesFM 2.5 (PyTorch)
print("\n🤖 Loading TimesFM 2.5 (200M) PyTorch...")
import torch
import timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

print("   Compiling model for forecasting...")
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,  # Temperature anomalies can be negative
        fix_quantile_crossing=True,
    )
)

# Forecast
horizon = 12
print(f"\n📈 Running forecast ({horizon} months ahead)...")
point_forecast, quantile_forecast = model.forecast(
    horizon=horizon,
    inputs=[input_series],
)

print(f"   Point forecast shape: {point_forecast.shape}")
print(f"   Quantile forecast shape: {quantile_forecast.shape}")

# Extract results
point = point_forecast[0]  # Shape: (horizon,)
quantiles = quantile_forecast[0]  # Shape: (horizon, 10)
# Quantile indices (TimesFM 2.5):
#   0 = mean, 1 = 10%, 2 = 20%, ..., 5 = 50% (median), ..., 9 = 90%

# Create forecast dates (2025 monthly)
last_date = df["date"].max()
forecast_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS"
)

# Build output DataFrame
# TimesFM 2.5 quantiles: mean at index 0, then 10% to 90% at indices 1-9
output_df = pd.DataFrame(
    {
        "date": forecast_dates.strftime("%Y-%m-%d"),
        "point_forecast": point,
        "mean": quantiles[:, 0],
        "q10": quantiles[:, 1],
        "q20": quantiles[:, 2],
        "q30": quantiles[:, 3],
        "q40": quantiles[:, 4],
        "q50": quantiles[:, 5],  # Median (= point_forecast)
        "q60": quantiles[:, 6],
        "q70": quantiles[:, 7],
        "q80": quantiles[:, 8],
        "q90": quantiles[:, 9],
    }
)

# Save outputs
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
output_df.to_csv(output_dir / "forecast_output.csv", index=False)

# JSON output for the report
# Note: TimesFM 2.5 has 9 quantiles (10%-90%), no 99% quantile
quantile_labels = ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%"]
output_json = {
    "model": "TimesFM 2.5 (200M) PyTorch",
    "input": {
        "source": "NOAA GISTEMP Global Temperature Anomaly",
        "n_observations": len(df),
        "date_range": f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}",
        "mean_anomaly_c": round(df["anomaly_c"].mean(), 3),
    },
    "forecast": {
        "horizon": horizon,
        "dates": forecast_dates.strftime("%Y-%m").tolist(),
        "point": point.tolist(),
        "quantiles": {
            label: quantiles[:, i + 1].tolist()
            for i, label in enumerate(quantile_labels)
        },
    },
    "summary": {
        "forecast_mean_c": round(float(point.mean()), 3),
        "forecast_max_c": round(float(point.max()), 3),
        "forecast_min_c": round(float(point.min()), 3),
        "vs_last_year_mean": round(
            float(point.mean() - df["anomaly_c"].iloc[-12:].mean()), 3
        ),
    },
}

with open(output_dir / "forecast_output.json", "w") as f:
    json.dump(output_json, f, indent=2)

# Print summary
print("\n" + "=" * 60)
print("  FORECAST RESULTS")
print("=" * 60)
print(
    f"\n📅 Forecast period: {forecast_dates[0].strftime('%Y-%m')} to {forecast_dates[-1].strftime('%Y-%m')}"
)
print(f"\n🌡️  Temperature Anomaly Forecast (°C above 1951-1980 baseline):")
print(f"\n   {'Month':<10} {'Point':>8} {'80% CI':>15} {'90% CI':>15}")
print(f"   {'-' * 10} {'-' * 8} {'-' * 15} {'-' * 15}")
for i, (date, pt, q10, q90, q05, q95) in enumerate(
    zip(
        forecast_dates.strftime("%Y-%m"),
        point,
        quantiles[:, 2],  # 20% (lower 80% CI)
        quantiles[:, 8],  # 80% (upper 80% CI)
        quantiles[:, 1],  # 10% (lower 90% CI)
        quantiles[:, 9],  # 90% (upper 90% CI)
    )
):
    print(
        f"   {date:<10} {pt:>8.3f} [{q10:>6.3f}, {q90:>6.3f}] [{q05:>6.3f}, {q95:>6.3f}]"
    )

print(f"\n📊 Summary Statistics:")
print(f"   Mean forecast:  {point.mean():.3f}°C")
print(
    f"   Max forecast:   {point.max():.3f}°C (Month: {forecast_dates[point.argmax()].strftime('%Y-%m')})"
)
print(
    f"   Min forecast:   {point.min():.3f}°C (Month: {forecast_dates[point.argmin()].strftime('%Y-%m')})"
)
print(f"   vs 2024 mean:   {point.mean() - df['anomaly_c'].iloc[-12:].mean():+.3f}°C")

print(f"\n✅ Output saved to:")
print(f"   {output_dir / 'forecast_output.csv'}")
print(f"   {output_dir / 'forecast_output.json'}")
