from __future__ import annotations

import math
from typing import Dict

import numpy as np


def summarize_gamma_recovery(
    gamma_hats: np.ndarray,
    gamma_true: float,
) -> Dict[str, float]:
    """Return publication-safe Monte Carlo recovery statistics.

    ``empirical_95_interval`` is a descriptive interval for the distribution
    of estimators.  It is deliberately NOT called CI and is never used as a
    coverage calculation.
    """
    x = np.asarray(gamma_hats, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite gamma estimates")

    mean_hat = float(np.mean(x))
    bias = mean_hat - float(gamma_true)
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    rmse = float(np.sqrt(np.mean((x - gamma_true) ** 2)))
    mc_se_mean = sd / math.sqrt(x.size) if x.size > 1 else np.nan

    # Since bias = mean_hat - gamma_true and gamma_true is fixed,
    # Monte-Carlo SE(bias) is identical to MC SE(mean_hat).
    mc_se_bias = mc_se_mean

    q_low, q_high = np.percentile(x, [2.5, 97.5])

    return {
        "n_successful": int(x.size),
        "gamma_true": float(gamma_true),
        "mean_gamma_hat": mean_hat,
        "bias": float(bias),
        "relative_bias": float(bias / gamma_true) if gamma_true != 0 else np.nan,
        "sd_gamma_hat": sd,
        "rmse": rmse,
        "mc_se_mean": float(mc_se_mean),
        "mc_se_bias": float(mc_se_bias),
        "empirical_95_low": float(q_low),
        "empirical_95_high": float(q_high),
    }

