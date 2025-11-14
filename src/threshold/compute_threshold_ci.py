# -*- coding: utf-8 -*-
"""
Compute group odor threshold and 95% CI from yes/no data (n = 20 panelists).

Expected CSV format (comma-separated):
    Concentration (ppm),panelist A,panelist B,...,panelist U
    0,0,0,...,0
    0.1,1,0,...,0
    1,1,1,...,1
    ...

- The first column is the concentration in ppm (≈ mg/L in water).
- The remaining columns are binary responses for each panelist:
    1 = detected, 0 = not detected.

The group threshold is defined as the lowest concentration at which
at least 50% of the panel (≥10 of 20 assessors) report detection.

95% CI is estimated by non-parametric bootstrap resampling of panelists:
- Resample the 20 panelists with replacement (n_boot replicates).
- For each bootstrap sample, recompute the group threshold.
- Take the 2.5th and 97.5th percentiles as the 95% CI.

Author: (your name)
"""

import math
import numpy as np
import pandas as pd


def load_sensory_csv(path: str):
    """
    Load CSV with columns:
        'Concentration (ppm)', 'panelist A', ..., 'panelist U'
    Returns:
        conc: np.ndarray of shape (n_concs,)
        resp: np.ndarray of shape (n_concs, n_panelists) with 0/1
    """
    df = pd.read_csv(path)

    # First column: concentration
    conc = df["Concentration (ppm)"].to_numpy(dtype=float)

    # All other columns starting with "panelist" are responses
    panelist_cols = [c for c in df.columns if c.lower().startswith("panelist")]
    resp = df[panelist_cols].to_numpy(dtype=float)

    return conc, resp, panelist_cols


def compute_group_threshold(conc: np.ndarray,
                            resp: np.ndarray,
                            cutoff_prop: float = 0.5):
    """
    Compute group threshold for a given concentration × panelist matrix.

    Parameters
    ----------
    conc : array-like, shape (n_concs,)
        Concentration levels (ascending).
    resp : array-like, shape (n_concs, n_panelists)
        Binary responses, 1 = detected, 0 = not detected.
    cutoff_prop : float
        Proportion of panelists required for detection.
        For 20 panelists and 50% rule, this is 0.5.

    Returns
    -------
    threshold : float
        The lowest concentration where detected >= cutoff_n.
        Returns NaN if the cutoff is never reached.
    cutoff_n : int
        The number of panelists required (e.g., 10 for n=20).
    detected_counts : np.ndarray, shape (n_concs,)
        Number of "detected" responses at each concentration.
    """
    n_concs, n_panel = resp.shape
    # For 20 panelists and 50% rule, this becomes 10
    cutoff_n = int(round(cutoff_prop * n_panel))

    detected_counts = resp.sum(axis=1)  # sum over panelists for each conc
    reached = detected_counts >= cutoff_n

    if not reached.any():
        return math.nan, cutoff_n, detected_counts

    first_idx = np.where(reached)[0][0]
    return float(conc[first_idx]), cutoff_n, detected_counts


def bootstrap_threshold_ci(conc: np.ndarray,
                           resp: np.ndarray,
                           cutoff_prop: float = 0.5,
                           n_boot: int = 10000,
                           seed: int = 0):
    """
    Estimate 95% CI for the group threshold by bootstrap resampling
    of panelists (non-parametric bootstrap).

    Parameters
    ----------
    conc : array-like, shape (n_concs,)
    resp : array-like, shape (n_concs, n_panelists)
    cutoff_prop : float
        Proportion of panelists required for detection (e.g., 0.5 for 50%).
    n_boot : int
        Number of bootstrap replicates.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    median : float
        Bootstrap median of the threshold distribution.
    lower : float
        2.5th percentile (lower bound of 95% CI).
    upper : float
        97.5th percentile (upper bound of 95% CI).
    thresholds_all : np.ndarray
        All bootstrap threshold values.
    """
    rng = np.random.default_rng(seed)
    n_concs, n_panel = resp.shape
    cutoff_n = int(round(cutoff_prop * n_panel))

    thresholds = []

    for _ in range(n_boot):
        # Resample panelists (columns) with replacement
        idx = rng.integers(0, n_panel, size=n_panel)
        boot_resp = resp[:, idx]

        detected_counts = boot_resp.sum(axis=1)
        reached = detected_counts >= cutoff_n

        if reached.any():
            first_idx = np.where(reached)[0][0]
            thresholds.append(conc[first_idx])
        # If cutoff is never reached, we could append NaN,
        # but in this dataset it always reaches at the highest concentration.

    thresholds = np.array(thresholds, dtype=float)

    median = float(np.quantile(thresholds, 0.5))
    lower = float(np.quantile(thresholds, 0.025))
    upper = float(np.quantile(thresholds, 0.975))

    return median, lower, upper, thresholds


if __name__ == "__main__":
    # Example usage: replace with your actual CSV paths
    csv_paths = [
        "2-phenylethyl-acetate.csv",  # 2-phenylethyl acetate
        "menthyl-acetate.csv",        # menthyl acetate
    ]

    for path in csv_paths:
        print(f"\n=== {path} ===")
        conc, resp, panelists = load_sensory_csv(path)
        print(f"Loaded {len(conc)} concentrations, {resp.shape[1]} panelists.")

        threshold, cutoff_n, detected = compute_group_threshold(conc, resp)
        print(f"Detected counts at each concentration:")
        for c, d in zip(conc, detected):
            print(f"  {c:g} ppm: {int(d)} / {resp.shape[1]} panelists")

        median, lower, upper, all_ts = bootstrap_threshold_ci(
            conc, resp, cutoff_prop=0.5, n_boot=10000, seed=0
        )

        print(f"\nGroup threshold (50% rule): {threshold:g} ppm")
        print(f"Bootstrap median threshold: {median:g} ppm")
        print(f"95% CI: {lower:g} – {upper:g} ppm")
