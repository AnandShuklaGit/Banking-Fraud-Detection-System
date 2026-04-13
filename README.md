# Banking Fraud Detection System

A statistical outlier detection pipeline to flag suspicious transactions from CSV records.

## Overview
Banks need to screen transactions for fraud in real time. Extreme transaction amounts are a primary indicator. This pipeline implements and benchmarks three detection methods to identify the most effective approach.

## Features
- **Min-Max Normalization** (user-defined function) — scales amounts to [0, 1]
- **IQR Method** — computes Q1, Q3, IQR bounds; replaces outliers with column mean
- **Z-Score Method** (user-defined function, threshold ±3) — flags transactions deviating more than 3 standard deviations
- Benchmark comparison table with recommendation for real-world deployment

## Tech Stack
Python, Pandas, NumPy

## Output Files
| File | Description |
|------|-------------|
| `flagged_transactions.csv` | Original data annotated with normalized amounts and outlier flags |

## Methods Explained

**Min-Max Normalization**
