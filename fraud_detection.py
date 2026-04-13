"""
Banking Fraud Detection System
================================
Statistical outlier detection pipeline to flag suspicious transactions.
Implements Min-Max normalization, IQR-based replacement, and Z-Score
detection as user-defined functions. Benchmarks both methods.

Tech Stack: Python, Pandas, NumPy
"""

import pandas as pd
import numpy as np
import os

# ── GENERATE SAMPLE CSV (if not present) ─────────────────────────────────────

CSV_FILE = "transactions.csv"

def generate_sample_csv(filepath: str, n: int = 300, seed: int = 42) -> None:
    """Generate a realistic transaction CSV with injected outliers."""
    np.random.seed(seed)

    # Normal transactions
    amounts  = np.round(np.random.normal(loc=200, scale=60, size=n), 2)
    # Inject ~5% extreme outliers (potential fraud)
    outlier_idx = np.random.choice(n, size=int(n * 0.05), replace=False)
    amounts[outlier_idx] = np.round(np.random.uniform(2000, 9999, size=len(outlier_idx)), 2)
    # Clamp negatives
    amounts = np.abs(amounts)

    df = pd.DataFrame({
        "Transaction_ID": [f"TXN{str(i).zfill(5)}" for i in range(1, n + 1)],
        "Account_ID":     [f"ACC{np.random.randint(1000, 9999)}" for _ in range(n)],
        "Amount":         amounts,
        "Merchant":       np.random.choice(
                              ["Amazon", "Flipkart", "Swiggy", "Unknown", "ATM"],
                              size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        "Location":       np.random.choice(
                              ["Delhi", "Mumbai", "Bangalore", "Foreign", "Online"],
                              size=n, p=[0.3, 0.25, 0.2, 0.1, 0.15]),
        "Timestamp":      pd.date_range("2024-01-01", periods=n, freq="2h"),
    })
    df.to_csv(filepath, index=False)
    print(f"Sample CSV generated: {filepath}  ({n} rows)")


if not os.path.exists(CSV_FILE):
    generate_sample_csv(CSV_FILE)

df = pd.read_csv(CSV_FILE)
print("\nDataset shape:", df.shape)
print(df[["Transaction_ID", "Amount", "Merchant", "Location"]].head(8))


# ── 1. MIN-MAX NORMALIZATION (user-defined) ───────────────────────────────────

def min_max_normalize(series: pd.Series) -> pd.Series:
    """
    Normalize a numeric Series to the [0, 1] range using Min-Max scaling.

    Formula:
        x_norm = (x - x_min) / (x_max - x_min)

    Returns:
        pd.Series: Normalized values.
    """
    x_min = series.min()
    x_max = series.max()
    if x_max == x_min:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - x_min) / (x_max - x_min)


df["Amount_MinMax"] = min_max_normalize(df["Amount"])
print("\n[1] Min-Max Normalized 'Amount' (first 5 rows):")
print(df[["Transaction_ID", "Amount", "Amount_MinMax"]].head())


# ── 2. IQR METHOD — OUTLIER REPLACEMENT ──────────────────────────────────────

def iqr_replace_outliers(series: pd.Series) -> tuple[pd.Series, int]:
    """
    Identify outliers using the IQR method and replace them with the column mean.

    Outlier bounds:
        Lower = Q1 - 1.5 * IQR
        Upper = Q3 + 1.5 * IQR

    Returns:
        cleaned_series (pd.Series): Series with outliers replaced by mean.
        outlier_count  (int):       Number of outliers detected.
    """
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    is_outlier   = (series < lower) | (series > upper)
    outlier_count = is_outlier.sum()

    col_mean = series[~is_outlier].mean()      # mean of inliers only
    cleaned  = series.copy()
    cleaned[is_outlier] = col_mean

    print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"  Lower bound={lower:.2f}, Upper bound={upper:.2f}")
    print(f"  Column mean (inliers)={col_mean:.2f}")

    return cleaned, int(outlier_count)


print("\n[2] IQR Method:")
df["Amount_IQR_Cleaned"], iqr_outliers = iqr_replace_outliers(df["Amount"])
print(f"  Outliers detected and replaced: {iqr_outliers}")


# ── 3. Z-SCORE METHOD (user-defined) ─────────────────────────────────────────

def zscore_detect_outliers(series: pd.Series,
                            threshold: float = 3.0) -> tuple[pd.Series, int]:
    """
    Detect outliers using Z-Score with a configurable threshold.

    Z-Score formula:
        z = (x - mean) / std

    Outlier condition:  |z| > threshold  (default ±3)

    Returns:
        outlier_flags (pd.Series[bool]): True where the value is an outlier.
        outlier_count (int):             Number of outliers detected.
    """
    mean = series.mean()
    std  = series.std()

    if std == 0:
        return pd.Series(np.zeros(len(series), dtype=bool), index=series.index), 0

    z_scores     = (series - mean) / std
    outlier_flags = z_scores.abs() > threshold
    outlier_count = int(outlier_flags.sum())

    print(f"  Mean={mean:.2f}, Std={std:.2f}, Threshold=±{threshold}")

    return outlier_flags, outlier_count


print("\n[3] Z-Score Method (threshold ±3):")
df["Is_Outlier_ZScore"], zscore_outliers = zscore_detect_outliers(df["Amount"])
print(f"  Outliers detected: {zscore_outliers}")
print(df[df["Is_Outlier_ZScore"]][["Transaction_ID", "Amount", "Merchant", "Location"]])


# ── 4. BENCHMARK COMPARISON ───────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  OUTLIER DETECTION COMPARISON")
print("=" * 55)
print(f"  {'Method':<25} {'Outliers Detected':>17}")
print("-" * 55)
print(f"  {'IQR Method':<25} {iqr_outliers:>17}")
print(f"  {'Z-Score Method (±3)':<25} {zscore_outliers:>17}")
print("=" * 55)

if iqr_outliers > zscore_outliers:
    print("\n  Verdict: IQR is more sensitive — flags more edge cases.")
    print("  Best for: skewed distributions with long tails.")
elif zscore_outliers > iqr_outliers:
    print("\n  Verdict: Z-Score flags more outliers in this dataset.")
    print("  Best for: normally distributed transaction amounts.")
else:
    print("\n  Both methods agree on the number of outliers.")

print("\n  Recommendation:")
print("  - Use IQR for real-world fraud screening (robust to skew).")
print("  - Use Z-Score when transaction amounts follow a Gaussian distribution.")
print("  - Combine both for maximum coverage in a production pipeline.")

# Save annotated results
output_file = "flagged_transactions.csv"
df.to_csv(output_file, index=False)
print(f"\nAnnotated results saved: {output_file}")
