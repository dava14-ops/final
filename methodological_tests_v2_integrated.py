#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
methodological_tests_v2.py

Публикационный экспериментальный слой для верификации CF-Cox/2SRI.

Основные цели v2:
1. Сравнить Naive Cox, CF-Cox и Oracle Cox на ОДНОЙ и той же
   сгенерированной выборке в каждой Monte Carlo репликации.
2. Оценить bias, relative bias, RMSE, empirical SD и mean model SE.
3. Оценить coverage 95% CI для Naive / CF / Oracle.
4. Для bootstrap использовать именно исходный dataset репликации,
   а не генерировать его повторно.
5. Использовать детерминированную схему seeds без hash().
6. Сохранять результаты каждой репликации и bootstrap diagnostics в CSV.
7. Не зашивать DEFAULT_GAMMA в расчеты, а использовать gamma_true.
8. Сделать количество кластеров и силу инструмента явными параметрами.
9. Сохранить совместимость с существующим Итог.py настолько,
   насколько позволяют его текущие публичные/внутренние функции.

ВАЖНО:
- Скрипт НЕ изменяет Итог.py.
- Oracle использует истинный U, если колонка U присутствует.
- Для кластерного bootstrap каждому псевдовыбору кластера назначается
  новый bootstrap cluster_id. Это предотвращает ситуацию, когда один
  и тот же исходный cluster_id повторяется несколько раз.
- Для основной статьи рекомендуется использовать percentile bootstrap CI:
      [q_0.025(gamma*), q_0.975(gamma*)]
  Нормальный bootstrap CI также сохраняется как diagnostic.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("methodological_tests_v2")


# ---------------------------------------------------------------------------
# Imports from existing model implementation
# ---------------------------------------------------------------------------

try:
    from Итог import (
        DGPParameters,
        CFFitOptions,
        generate_data,
        fit_first_stage,
        fit_cf_cox,
        fit_naive_cox,
        _build_cf_columns,
        _fit_cox_model,
        _add_design_x_columns,
        _validate_survival_frame,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Не удалось импортировать функции из Итог.py. "
        "Положите methodological_tests_v2.py рядом с Итог.py."
    ) from exc

try:
    from constants import MTBF_BASELINE_HOURS
except Exception:
    MTBF_BASELINE_HOURS = 558.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_GAMMA = 0.5
DEFAULT_RHO = 0.7
DEFAULT_DELTA = 0.7
DEFAULT_N = 10_000
DEFAULT_G = 32
DEFAULT_INSTRUMENT_STRENGTH = 0.5
DEFAULT_BASELINE_HAZARD = 1.0 / MTBF_BASELINE_HOURS
DEFAULT_CENSORING_SCALE = 5000.0
DEFAULT_BASELINE_SHAPE = 1.88
DEFAULT_CONFIDENCE_LEVEL = 0.95

OUTPUT_DIR = Path("methodological_results_v2")


@dataclass(frozen=True)
class ExperimentConfig:
    gamma_true: float = DEFAULT_GAMMA
    rho: float = DEFAULT_RHO
    delta: float = DEFAULT_DELTA
    n: int = DEFAULT_N
    n_clusters: int = DEFAULT_G  # target/diagnostic only; current generate_data() does not accept G
    instrument_strength: float = DEFAULT_INSTRUMENT_STRENGTH
    baseline_shape: float = DEFAULT_BASELINE_SHAPE
    baseline_hazard: float = DEFAULT_BASELINE_HAZARD
    censoring_scale: float = DEFAULT_CENSORING_SCALE
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL

    # DGP implementation controls retained from the current script.
    contamination: bool = False
    contamination_probability: float = 1.0
    instrument_source: str = "normal"


# ---------------------------------------------------------------------------
# Deterministic seeds
# ---------------------------------------------------------------------------

def make_seed(seed_base: int, *parts: int) -> int:
    """
    Stable deterministic seed.

    No use of Python hash(), because hash randomization can make a factorial
    experiment non-reproducible across processes.
    """
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    x = np.uint64(seed_base) & mask
    for p in parts:
        x ^= np.uint64(int(p) + 0x9E3779B97F4A7C15) & mask
        x = (x * np.uint64(0xBF58476D1CE4E5B9)) & mask
        x ^= x >> np.uint64(27)
        x = (x * np.uint64(0x94D049BB133111EB)) & mask
        x ^= x >> np.uint64(31)
    return int(x & np.uint64(0x7FFFFFFF))


# ---------------------------------------------------------------------------
# DGP construction
# ---------------------------------------------------------------------------

def _make_dgp(
    gamma: float,
    rho: float,
    delta: float,
    baseline_shape: float,
) -> DGPParameters:
    """Создает DGPParameters, совместимый с текущим methodological_tests.py."""
    return DGPParameters(
        gamma=gamma,
        rho=rho,
        delta=delta,
        intercept=0.5,
        structural_intercept=0.5,
        first_stage_z_coef=0.5,
        fs_age_coef=0.15,
        fs_hours_coef=0.10,
        fs_climate_coef=0.20,
        fs_soil_coef=0.15,
        fs_brand_coef=0.10,
        fs_power_coef=0.08,
        beta_age=0.20,
        beta_hours=0.10,
        beta_age_hours=0.15,
        beta_climate=0.20,
        beta_soil=0.12,
        beta_brand=0.06,
        beta_power=-0.05,
        baseline_family="weibull",
        baseline_shape=baseline_shape,
        brand_encoding="dummies",
        brand_reference_code=0,
        competing_risks=True,
        minor_failure_rate=0.002,
        event_definition="major_claim",
        segment="light",
        peakload_target_mean=0.55,
        peakload_target_std=0.15,
    )


def _safe_generate_data(
    *,
    cfg: ExperimentConfig,
    seed: int,
) -> pd.DataFrame:
    """
    Generate exactly one dataset per replication.

    IMPORTANT:
    The current Итог.py::generate_data() signature accepts n, contamination,
    baseline_hazard, censoring_scale, rng, instrument_strength, dgp,
    instrument_source, price_instrument_path and contamination_probability,
    but DOES NOT accept n_clusters/G/clusters. Therefore cfg.n_clusters is
    recorded as a target/diagnostic only. The actual cluster count generated
    by Итог.py is measured from data["cluster_id"] and must be reported.

    The current default synthetic cluster construction in Итог.py is based on
    production_year × campaign_group, whereas real-covariate hybrid mode can
    provide cluster_indices from the real data layer.
    """
    rng = np.random.default_rng(seed)
    dgp = _make_dgp(
        gamma=cfg.gamma_true,
        rho=cfg.rho,
        delta=cfg.delta,
        baseline_shape=cfg.baseline_shape,
    )

    kwargs: Dict[str, Any] = {
        "n": cfg.n,
        "contamination": cfg.contamination,
        "baseline_hazard": cfg.baseline_hazard,
        "censoring_scale": cfg.censoring_scale,
        "rng": rng,
        "instrument_strength": cfg.instrument_strength,
        "dgp": dgp,
        "instrument_source": cfg.instrument_source,
        "price_instrument_path": None,
        "contamination_probability": cfg.contamination_probability,
    }

    # Current Итог.py explicitly supports price_instrument_path.
    # Keep it configurable in the DGP object if/when using price_bartik.
    price_path = getattr(dgp, "price_instrument_path", None)
    if cfg.instrument_source == "price_bartik" and price_path:
        kwargs["price_instrument_path"] = price_path

    data = generate_data(**kwargs)

    if not isinstance(data, pd.DataFrame):
        raise TypeError("generate_data() не вернула pandas.DataFrame.")

    required = {"time", "event", "PeakLoad", "Z", "U", "cluster_id"}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(
            f"generate_data() вернула данные без обязательных колонок: {sorted(missing)}"
        )

    actual_g = int(data["cluster_id"].nunique())
    if cfg.n_clusters is not None and actual_g != cfg.n_clusters:
        logger.warning(
            "TARGET G=%d не установлен: текущий generate_data() "
            "фактически создал G=%d кластеров.",
            cfg.n_clusters,
            actual_g,
        )

    return data


# ---------------------------------------------------------------------------
# Model options
# ---------------------------------------------------------------------------

def _make_opts(data: pd.DataFrame, n_bootstrap: int = 0) -> CFFitOptions:
    return CFFitOptions(
        cox_se_threshold=10.0,
        v_hat_basis="linear",
        v_hat_basis_params=None,
        extra_x_cols=None,
        center_peakload=None,
        brand_encoding="dummies",
        brand_reference_code=0,
        var_z_threshold=1e-8,
        min_first_stage_f=10.0,
        fail_on_weak_instrument=False,
        min_cox_events=10,
        min_events_per_covariate=5,
        save_tracebacks=True,
        cluster_col="cluster_id" if "cluster_id" in data.columns else None,
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Generic safe parsing helpers
# ---------------------------------------------------------------------------

def _finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _extract_f_stat(first_stage: Any) -> float:
    """
    Best-effort extraction of first-stage F statistic from the existing object.
    Returns NaN if the exact attribute name differs.
    """
    candidate_names = (
        "f_stat",
        "f_statistic",
        "first_stage_f",
        "f_cluster",
        "f_cluster_stat",
        "partial_f",
        "partial_f_stat",
        "weak_iv_f",
    )
    for name in candidate_names:
        if hasattr(first_stage, name):
            value = _finite_or_nan(getattr(first_stage, name))
            if math.isfinite(value):
                return value

    if isinstance(first_stage, dict):
        for name in candidate_names:
            if name in first_stage:
                value = _finite_or_nan(first_stage[name])
                if math.isfinite(value):
                    return value

    return float("nan")


def _extract_gamma_se(result: Any) -> Tuple[float, float]:
    gamma = _finite_or_nan(getattr(result, "gamma_hat", np.nan))

    se_candidates = (
        "naive_model_se",
        "naive_se",
        "se",
        "gamma_se",
        "model_se",
    )
    se = float("nan")
    for name in se_candidates:
        if hasattr(result, name):
            value = _finite_or_nan(getattr(result, name))
            if math.isfinite(value):
                se = value
                break

    return gamma, se


# ---------------------------------------------------------------------------
# Oracle model
# ---------------------------------------------------------------------------

def fit_oracle_cox(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit oracle Cox using the true latent U.

    Reuses the same internal design-building functions as the original script.
    """
    if "U" not in data.columns:
        raise KeyError("Oracle Cox невозможен: колонка U отсутствует.")

    time_col = "time" if "time" in data.columns else "T"
    event_col = "event"

    oracle_data = data[[time_col, event_col, "PeakLoad", "U"]].copy()
    x_cols = _add_design_x_columns(
        oracle_data,
        data,
        None,
        "dummies",
        0,
    )
    covariate_cols = ["PeakLoad", "U"] + list(x_cols)

    _validate_survival_frame(oracle_data, covariate_cols)

    cph = _fit_cox_model(
        oracle_data,
        covariate_cols,
        penalizer=0.0,
        robust=False,
    )

    gamma_hat = _finite_or_nan(cph.params_["PeakLoad"])

    se = float("nan")
    try:
        se = _finite_or_nan(cph.standard_errors_["PeakLoad"])
    except Exception:
        pass

    return {
        "gamma_hat": gamma_hat,
        "se": se,
        "cph": cph,
    }


# ---------------------------------------------------------------------------
# Single replication: generate once, fit all models once
# ---------------------------------------------------------------------------

def fit_all_models(
    data: pd.DataFrame,
    *,
    with_first_stage: bool = True,
) -> Dict[str, Any]:
    """
    Fit Naive, CF and Oracle on the SAME dataset.

    No data regeneration occurs inside this function.
    """
    opts = _make_opts(data, n_bootstrap=0)

    out: Dict[str, Any] = {
        "gamma_naive": np.nan,
        "se_naive": np.nan,
        "gamma_cf": np.nan,
        "se_cf": np.nan,
        "gamma_oracle": np.nan,
        "se_oracle": np.nan,
        "first_stage_f": np.nan,
        "cf_fit": None,
        "first_stage": None,
        "opts": opts,
        "oracle_fit": None,
        "naive_fit": None,
    }

    try:
        naive = fit_naive_cox(data, opts)
        gamma, se = _extract_gamma_se(naive)
        out["gamma_naive"] = gamma
        out["se_naive"] = se
        out["naive_fit"] = naive
    except Exception as exc:
        out["naive_error"] = repr(exc)
        logger.warning("Naive Cox failed: %s", exc)

    if with_first_stage:
        try:
            first_stage = fit_first_stage(data, opts)
            out["first_stage"] = first_stage
            out["first_stage_f"] = _extract_f_stat(first_stage)

            cf = fit_cf_cox(data, first_stage, opts)
            gamma, se = _extract_gamma_se(cf)
            out["gamma_cf"] = gamma
            out["se_cf"] = se
            out["cf_fit"] = cf
        except Exception as exc:
            out["cf_error"] = repr(exc)
            logger.warning("CF Cox failed: %s", exc)

    try:
        oracle = fit_oracle_cox(data)
        out["gamma_oracle"] = oracle["gamma_hat"]
        out["se_oracle"] = oracle["se"]
        out["oracle_fit"] = oracle
    except Exception as exc:
        out["oracle_error"] = repr(exc)
        logger.warning("Oracle Cox failed: %s", exc)

    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _rmse(values: pd.Series, truth: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((values.to_numpy() - truth) ** 2)))


def _mean_abs_bias(values: pd.Series, truth: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return float("nan")
    return float(np.mean(np.abs(values.to_numpy() - truth)))


def _summarize_model(
    df: pd.DataFrame,
    *,
    model: str,
    gamma_col: str,
    se_col: Optional[str],
    gamma_true: float,
) -> Dict[str, Any]:
    vals = pd.to_numeric(df[gamma_col], errors="coerce")
    valid = vals.dropna()

    result: Dict[str, Any] = {
        "model": model,
        "n_replications": int(len(df)),
        "n_success": int(len(valid)),
        "failure_rate": float(1.0 - len(valid) / len(df)) if len(df) else np.nan,
        "mean_gamma_hat": float(valid.mean()) if len(valid) else np.nan,
        "bias": float(valid.mean() - gamma_true) if len(valid) else np.nan,
        "relative_bias_pct": (
            float(100.0 * (valid.mean() - gamma_true) / gamma_true)
            if len(valid) and gamma_true != 0
            else np.nan
        ),
        "abs_bias": _mean_abs_bias(vals, gamma_true),
        "rmse": _rmse(vals, gamma_true),
        "empirical_sd": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
        "mean_se": np.nan,
        "se_ratio_empirical_to_model": np.nan,
    }

    if se_col is not None and se_col in df.columns:
        se = pd.to_numeric(df[se_col], errors="coerce").dropna()
        if len(se):
            mean_se = float(se.mean())
            result["mean_se"] = mean_se
            if math.isfinite(result["empirical_sd"]) and mean_se > 0:
                result["se_ratio_empirical_to_model"] = (
                    result["empirical_sd"] / mean_se
                )

    return result


def summarize_comparison(
    df: pd.DataFrame,
    gamma_true: float,
) -> pd.DataFrame:
    rows = [
        _summarize_model(
            df,
            model="Naive Cox",
            gamma_col="gamma_naive",
            se_col="se_naive",
            gamma_true=gamma_true,
        ),
        _summarize_model(
            df,
            model="CF-Cox (2SRI)",
            gamma_col="gamma_cf",
            se_col="se_cf",
            gamma_true=gamma_true,
        ),
        _summarize_model(
            df,
            model="Oracle Cox",
            gamma_col="gamma_oracle",
            se_col="se_oracle",
            gamma_true=gamma_true,
        ),
    ]

    summary = pd.DataFrame(rows)

    # Add relative RMSE reduction vs Naive, useful for the manuscript.
    naive_rmse = summary.loc[
        summary["model"] == "Naive Cox", "rmse"
    ].iloc[0]
    if math.isfinite(naive_rmse) and naive_rmse > 0:
        summary["rmse_reduction_vs_naive_pct"] = (
            100.0 * (1.0 - summary["rmse"] / naive_rmse)
        )
    else:
        summary["rmse_reduction_vs_naive_pct"] = np.nan

    return summary


# ---------------------------------------------------------------------------
# Bootstrap utilities
# ---------------------------------------------------------------------------

def _cluster_bootstrap_sample(
    data: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Resample clusters with replacement.

    Each sampled source cluster receives a NEW pseudo cluster id equal to the
    bootstrap draw number. Thus repeated selection of the same original
    cluster creates independent pseudo-cluster labels in the bootstrap sample.
    """
    if "cluster_id" not in data.columns:
        row_idx = rng.integers(0, len(data), size=len(data))
        return data.iloc[row_idx].copy().reset_index(drop=True)

    clusters = pd.unique(data["cluster_id"])
    n_clusters = len(clusters)

    chunks: List[pd.DataFrame] = []
    for draw_idx in range(n_clusters):
        source_cluster = rng.choice(clusters)
        chunk = data.loc[data["cluster_id"] == source_cluster].copy()
        chunk["_bootstrap_source_cluster"] = source_cluster
        chunk["_bootstrap_draw_cluster"] = draw_idx
        chunks.append(chunk)

    boot = pd.concat(chunks, axis=0, ignore_index=True)
    # Downstream code expects exactly cluster_id.
    boot["cluster_id"] = boot["_bootstrap_draw_cluster"].astype(int)
    boot.drop(
        columns=["_bootstrap_source_cluster", "_bootstrap_draw_cluster"],
        inplace=True,
    )
    return boot.reset_index(drop=True)


def _row_bootstrap_sample(
    data: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    idx = rng.integers(0, len(data), size=len(data))
    return data.iloc[idx].copy().reset_index(drop=True)


def _bootstrap_naive_gamma(
    boot_data: pd.DataFrame,
    opts: CFFitOptions,
) -> float:
    fit = fit_naive_cox(boot_data, opts)
    return _extract_gamma_se(fit)[0]


def _bootstrap_cf_gamma(
    boot_data: pd.DataFrame,
    opts: CFFitOptions,
) -> float:
    fs = fit_first_stage(boot_data, opts)
    fit = fit_cf_cox(boot_data, fs, opts)
    return _extract_gamma_se(fit)[0]


def _bootstrap_oracle_gamma(
    boot_data: pd.DataFrame,
) -> float:
    fit = fit_oracle_cox(boot_data)
    return fit["gamma_hat"]


def bootstrap_gamma_ci(
    data: pd.DataFrame,
    *,
    model: str,
    n_bootstrap: int,
    seed: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Dict[str, Any]:
    """
    Cluster bootstrap of gamma.

    Returns:
      - bootstrap_se
      - percentile_ci
      - normal_ci components can be constructed outside
      - n_success / n_failures
    """
    rng = np.random.default_rng(seed)
    opts = _make_opts(data, n_bootstrap=0)

    samples: List[float] = []
    failures = 0
    errors: List[str] = []

    model_key = model.strip().lower()

    if model_key not in {"naive", "cf", "oracle"}:
        raise ValueError(f"Неизвестная модель bootstrap: {model}")

    for b in range(n_bootstrap):
        try:
            boot_data = _cluster_bootstrap_sample(data, rng)

            if model_key == "naive":
                gamma_b = _bootstrap_naive_gamma(boot_data, opts)
            elif model_key == "cf":
                gamma_b = _bootstrap_cf_gamma(boot_data, opts)
            else:
                gamma_b = _bootstrap_oracle_gamma(boot_data)

            if math.isfinite(gamma_b):
                samples.append(float(gamma_b))
            else:
                failures += 1
        except Exception as exc:
            failures += 1
            if len(errors) < 10:
                errors.append(repr(exc))

    alpha = 1.0 - confidence_level

    if len(samples) < max(20, int(0.25 * n_bootstrap)):
        return {
            "bootstrap_se": np.nan,
            "percentile_ci_lower": np.nan,
            "percentile_ci_upper": np.nan,
            "normal_center": np.nan,
            "n_success": len(samples),
            "n_total": n_bootstrap,
            "n_failures": failures,
            "success_rate": len(samples) / n_bootstrap if n_bootstrap else np.nan,
            "errors": " | ".join(errors),
        }

    arr = np.asarray(samples, dtype=float)
    q_low, q_high = np.quantile(
        arr,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    return {
        "bootstrap_se": float(np.std(arr, ddof=1)),
        "percentile_ci_lower": float(q_low),
        "percentile_ci_upper": float(q_high),
        "normal_center": float(np.mean(arr)),
        "n_success": len(arr),
        "n_total": n_bootstrap,
        "n_failures": failures,
        "success_rate": len(arr) / n_bootstrap,
        "errors": " | ".join(errors),
    }


# ---------------------------------------------------------------------------
# Full coverage replication
# ---------------------------------------------------------------------------

def run_one_replication_with_coverage(
    *,
    rep_idx: int,
    seed: int,
    cfg: ExperimentConfig,
    n_bootstrap: int,
    coverage_models: Sequence[str],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Generate one dataset ONCE, fit all models, then bootstrap the same dataset.
    """
    t0 = time.perf_counter()

    data = _safe_generate_data(cfg=cfg, seed=seed)
    fits = fit_all_models(data, with_first_stage=True)

    actual_g = (
        int(data["cluster_id"].nunique())
        if "cluster_id" in data.columns
        else len(data)
    )
    n_events = int(data["event"].sum()) if "event" in data.columns else np.nan

    base = {
        "replication": rep_idx,
        "seed": seed,
        "gamma_true": cfg.gamma_true,
        "rho": cfg.rho,
        "delta": cfg.delta,
        "n": len(data),
        "G_requested": cfg.n_clusters,
        "G_actual": actual_g,
        "n_events": n_events,
        "instrument_strength": cfg.instrument_strength,
        "baseline_shape": cfg.baseline_shape,
        "first_stage_f": fits.get("first_stage_f", np.nan),
        "gamma_naive": fits["gamma_naive"],
        "se_naive": fits["se_naive"],
        "gamma_cf": fits["gamma_cf"],
        "se_cf": fits["se_cf"],
        "gamma_oracle": fits["gamma_oracle"],
        "se_oracle": fits["se_oracle"],
        "replication_elapsed_sec": np.nan,
    }

    # Point-estimate diagnostics for the same replication.
    for model, gamma_col in [
        ("naive", "gamma_naive"),
        ("cf", "gamma_cf"),
        ("oracle", "gamma_oracle"),
    ]:
        gamma_hat = _finite_or_nan(base[gamma_col])
        base[f"bias_{model}"] = (
            gamma_hat - cfg.gamma_true
            if math.isfinite(gamma_hat)
            else np.nan
        )

    coverage_rows: List[Dict[str, Any]] = []

    for model in coverage_models:
        model_key = model.strip().lower()
        gamma_col = {
            "naive": "gamma_naive",
            "cf": "gamma_cf",
            "oracle": "gamma_oracle",
        }[model_key]

        gamma_hat = _finite_or_nan(base[gamma_col])

        row = {
            "replication": rep_idx,
            "seed": seed,
            "gamma_true": cfg.gamma_true,
            "model": model_key,
            "gamma_hat": gamma_hat,
            "bias": (
                gamma_hat - cfg.gamma_true
                if math.isfinite(gamma_hat)
                else np.nan
            ),
            "first_stage_f": base["first_stage_f"],
            "n": len(data),
            "G_actual": actual_g,
            "n_events": n_events,
        }

        if not math.isfinite(gamma_hat):
            row.update(
                {
                    "bootstrap_se": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "ci_width": np.nan,
                    "normal_ci_lower": np.nan,
                    "normal_ci_upper": np.nan,
                    "covered": np.nan,
                    "normal_covered": np.nan,
                    "bootstrap_success_rate": np.nan,
                    "bootstrap_failures": np.nan,
                    "bootstrap_n_success": np.nan,
                    "bootstrap_n_total": n_bootstrap,
                }
            )
            coverage_rows.append(row)
            continue

        boot_seed = make_seed(seed, rep_idx, 991, {"naive": 1, "cf": 2, "oracle": 3}[model_key])
        boot = bootstrap_gamma_ci(
            data,
            model=model_key,
            n_bootstrap=n_bootstrap,
            seed=boot_seed,
            confidence_level=cfg.confidence_level,
        )

        ci_lo = boot["percentile_ci_lower"]
        ci_hi = boot["percentile_ci_upper"]

        # Diagnostic normal/bootstrap-SE CI around the original point estimate.
        z = float(
            {
                0.90: 1.6448536269514722,
                0.95: 1.959963984540054,
                0.99: 2.5758293035489004,
            }.get(
                round(cfg.confidence_level, 2),
                1.959963984540054,
            )
        )
        boot_se = _finite_or_nan(boot["bootstrap_se"])
        normal_lo = (
            gamma_hat - z * boot_se if math.isfinite(boot_se) else np.nan
        )
        normal_hi = (
            gamma_hat + z * boot_se if math.isfinite(boot_se) else np.nan
        )

        row.update(
            {
                "bootstrap_se": boot_se,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "ci_width": (
                    ci_hi - ci_lo
                    if math.isfinite(ci_lo) and math.isfinite(ci_hi)
                    else np.nan
                ),
                "normal_ci_lower": normal_lo,
                "normal_ci_upper": normal_hi,
                "covered": (
                    bool(ci_lo <= cfg.gamma_true <= ci_hi)
                    if math.isfinite(ci_lo) and math.isfinite(ci_hi)
                    else np.nan
                ),
                "normal_covered": (
                    bool(normal_lo <= cfg.gamma_true <= normal_hi)
                    if math.isfinite(normal_lo) and math.isfinite(normal_hi)
                    else np.nan
                ),
                "bootstrap_success_rate": boot["success_rate"],
                "bootstrap_failures": boot["n_failures"],
                "bootstrap_n_success": boot["n_success"],
                "bootstrap_n_total": boot["n_total"],
                "bootstrap_errors": boot.get("errors", ""),
            }
        )

        coverage_rows.append(row)

    base["replication_elapsed_sec"] = time.perf_counter() - t0
    return base, pd.DataFrame(coverage_rows)


# ---------------------------------------------------------------------------
# Experiment 1: point-estimate comparison
# ---------------------------------------------------------------------------

def run_comparison(
    *,
    cfg: ExperimentConfig,
    n_runs: int,
    seed_base: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 84)
    print("WEEK 1A — NAIVE vs CF-Cox vs ORACLE")
    print("=" * 84)
    print(
        f"gamma_true={cfg.gamma_true}, rho={cfg.rho}, delta={cfg.delta}, "
        f"n={cfg.n}, G_target={cfg.n_clusters}, pi_Z={cfg.instrument_strength}"
    )
    print(f"R={n_runs}")
    print("-" * 84)

    rows: List[Dict[str, Any]] = []

    for rep in range(1, n_runs + 1):
        seed = make_seed(seed_base, 101, rep)
        print(f"[{rep:>4}/{n_runs}] seed={seed} ... ", end="", flush=True)
        t0 = time.perf_counter()

        try:
            data = _safe_generate_data(cfg=cfg, seed=seed)
            fits = fit_all_models(data, with_first_stage=True)
            actual_g = (
                int(data["cluster_id"].nunique())
                if "cluster_id" in data.columns
                else len(data)
            )
            n_events = int(data["event"].sum()) if "event" in data.columns else np.nan

            row = {
                "replication": rep,
                "seed": seed,
                "gamma_true": cfg.gamma_true,
                "rho": cfg.rho,
                "delta": cfg.delta,
                "n": len(data),
                "G_requested": cfg.n_clusters,
                "G_actual": actual_g,
                "n_events": n_events,
                "instrument_strength": cfg.instrument_strength,
                "baseline_shape": cfg.baseline_shape,
                "first_stage_f": fits.get("first_stage_f", np.nan),
                "gamma_naive": fits["gamma_naive"],
                "se_naive": fits["se_naive"],
                "gamma_cf": fits["gamma_cf"],
                "se_cf": fits["se_cf"],
                "gamma_oracle": fits["gamma_oracle"],
                "se_oracle": fits["se_oracle"],
                "elapsed_sec": time.perf_counter() - t0,
            }

            for model, col in [
                ("naive", "gamma_naive"),
                ("cf", "gamma_cf"),
                ("oracle", "gamma_oracle"),
            ]:
                row[f"bias_{model}"] = (
                    row[col] - cfg.gamma_true
                    if math.isfinite(_finite_or_nan(row[col]))
                    else np.nan
                )

            rows.append(row)

            print(
                f"Naive={row['gamma_naive']:+.4f}, "
                f"CF={row['gamma_cf']:+.4f}, "
                f"Oracle={row['gamma_oracle']:+.4f}, "
                f"({row['elapsed_sec']:.1f}s)"
            )

        except Exception as exc:
            logger.exception("Репликация %d завершилась ошибкой.", rep)
            rows.append(
                {
                    "replication": rep,
                    "seed": seed,
                    "gamma_true": cfg.gamma_true,
                    "rho": cfg.rho,
                    "delta": cfg.delta,
                    "n": cfg.n,
                    "G_requested": cfg.n_clusters,
                    "G_actual": np.nan,
                    "n_events": np.nan,
                    "instrument_strength": cfg.instrument_strength,
                    "baseline_shape": cfg.baseline_shape,
                    "first_stage_f": np.nan,
                    "gamma_naive": np.nan,
                    "se_naive": np.nan,
                    "gamma_cf": np.nan,
                    "se_cf": np.nan,
                    "gamma_oracle": np.nan,
                    "se_oracle": np.nan,
                    "elapsed_sec": time.perf_counter() - t0,
                    "replication_error": repr(exc),
                }
            )
            print(f"ERROR: {exc}")

    df = pd.DataFrame(rows)
    summary = summarize_comparison(df, cfg.gamma_true)

    if "G_actual" in df.columns:
        g_actual = pd.to_numeric(df["G_actual"], errors="coerce").dropna()
        if len(g_actual):
            print(
                f"Actual cluster count: mean={g_actual.mean():.1f}, "
                f"min={g_actual.min():.0f}, max={g_actual.max():.0f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "week1A_comparison_replications.csv", index=False)
    summary.to_csv(output_dir / "week1A_comparison_summary.csv", index=False)

    print("\n" + "-" * 84)
    print(summary.to_string(index=False))
    print("-" * 84)

    return df, summary


# ---------------------------------------------------------------------------
# Experiment 2: coverage
# ---------------------------------------------------------------------------

def summarize_coverage(
    coverage_df: pd.DataFrame,
    *,
    confidence_level: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for model, group in coverage_df.groupby("model", dropna=False):
        covered = pd.to_numeric(group["covered"], errors="coerce").dropna()
        normal_covered = pd.to_numeric(
            group["normal_covered"], errors="coerce"
        ).dropna()
        widths = pd.to_numeric(group["ci_width"], errors="coerce").dropna()
        boot_rates = pd.to_numeric(
            group["bootstrap_success_rate"], errors="coerce"
        ).dropna()
        boot_se = pd.to_numeric(group["bootstrap_se"], errors="coerce").dropna()

        n_eligible = len(covered)

        rows.append(
            {
                "model": model,
                "confidence_level": confidence_level,
                "n_replications": int(len(group)),
                "n_eligible_for_coverage": int(n_eligible),
                "n_covered": int(covered.sum()) if n_eligible else 0,
                "coverage": float(covered.mean()) if n_eligible else np.nan,
                "coverage_mcse": (
                    float(
                        math.sqrt(
                            covered.mean() * (1.0 - covered.mean()) / n_eligible
                        )
                    )
                    if n_eligible
                    else np.nan
                ),
                "normal_ci_coverage": (
                    float(normal_covered.mean()) if len(normal_covered) else np.nan
                ),
                "mean_ci_width": float(widths.mean()) if len(widths) else np.nan,
                "median_ci_width": (
                    float(widths.median()) if len(widths) else np.nan
                ),
                "mean_bootstrap_se": (
                    float(boot_se.mean()) if len(boot_se) else np.nan
                ),
                "mean_bootstrap_success_rate": (
                    float(boot_rates.mean()) if len(boot_rates) else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def run_coverage(
    *,
    cfg: ExperimentConfig,
    n_runs: int,
    n_bootstrap: int,
    seed_base: int,
    output_dir: Path,
    coverage_models: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 84)
    print("WEEK 1B — 95% CI COVERAGE")
    print("=" * 84)
    print(
        f"gamma_true={cfg.gamma_true}, rho={cfg.rho}, delta={cfg.delta}, "
        f"n={cfg.n}, G_target={cfg.n_clusters}, pi_Z={cfg.instrument_strength}"
    )
    print(f"R={n_runs}, B={n_bootstrap}, models={list(coverage_models)}")
    print("CI for manuscript: percentile bootstrap")
    print("-" * 84)

    rep_rows: List[Dict[str, Any]] = []
    ci_rows: List[pd.DataFrame] = []

    for rep in range(1, n_runs + 1):
        seed = make_seed(seed_base, 202, rep)
        print(
            f"[{rep:>4}/{n_runs}] seed={seed} "
            f"(B={n_bootstrap}) ... ",
            end="",
            flush=True,
        )
        t0 = time.perf_counter()

        try:
            rep_row, rep_ci = run_one_replication_with_coverage(
                rep_idx=rep,
                seed=seed,
                cfg=cfg,
                n_bootstrap=n_bootstrap,
                coverage_models=coverage_models,
            )
            rep_rows.append(rep_row)
            ci_rows.append(rep_ci)

            coverage_now = []
            for _, r in rep_ci.iterrows():
                if pd.notna(r.get("covered")):
                    coverage_now.append(
                        f"{r['model']}={'Y' if bool(r['covered']) else 'N'}"
                    )

            print(
                f"{', '.join(coverage_now) if coverage_now else 'no valid CI'} "
                f"({time.perf_counter() - t0:.1f}s)"
            )

        except Exception as exc:
            logger.exception("Coverage replication %d failed.", rep)
            print(f"ERROR: {exc}")

            rep_rows.append(
                {
                    "replication": rep,
                    "seed": seed,
                    "gamma_true": cfg.gamma_true,
                    "rho": cfg.rho,
                    "delta": cfg.delta,
                    "n": cfg.n,
                    "G_requested": cfg.n_clusters,
                    "G_actual": np.nan,
                    "n_events": np.nan,
                    "instrument_strength": cfg.instrument_strength,
                    "baseline_shape": cfg.baseline_shape,
                    "first_stage_f": np.nan,
                    "replication_elapsed_sec": time.perf_counter() - t0,
                    "replication_error": repr(exc),
                }
            )

    rep_df = pd.DataFrame(rep_rows)
    ci_df = (
        pd.concat(ci_rows, ignore_index=True)
        if ci_rows
        else pd.DataFrame()
    )

    summary = summarize_coverage(
        ci_df,
        confidence_level=cfg.confidence_level,
    ) if not ci_df.empty else pd.DataFrame()

    if not ci_df.empty and "G_actual" in ci_df.columns:
        g_actual = pd.to_numeric(ci_df["G_actual"], errors="coerce").dropna()
        if len(g_actual):
            logger.info(
                "Coverage experiment actual G: mean=%.1f, min=%d, max=%d",
                g_actual.mean(),
                int(g_actual.min()),
                int(g_actual.max()),
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(output_dir / "week1B_coverage_replications.csv", index=False)
    ci_df.to_csv(output_dir / "week1B_coverage_details.csv", index=False)
    summary.to_csv(output_dir / "week1B_coverage_summary.csv", index=False)

    print("\n" + "-" * 84)
    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("Нет результатов coverage.")
    print("-" * 84)

    return ci_df, summary


# ---------------------------------------------------------------------------
# Combined Week 1
# ---------------------------------------------------------------------------

def run_week1(
    *,
    cfg: ExperimentConfig,
    n_runs: int,
    n_bootstrap: int,
    seed_base: int,
    output_dir: Path,
    coverage_models: Sequence[str],
    skip_comparison: bool = False,
    skip_coverage: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save exact config for reproducibility.
    # NOTE: n_clusters is a target/diagnostic in the current Итог.py, not an enforced DGP parameter.
    config_payload = asdict(cfg)
    config_payload.update(
        {
            "n_runs": n_runs,
            "n_bootstrap": n_bootstrap,
            "seed_base": seed_base,
            "coverage_models": list(coverage_models),
            "python": sys.version,
            "pid": os.getpid(),
        }
    )
    pd.DataFrame([config_payload]).to_json(
        output_dir / "week1_config.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )

    t0 = time.perf_counter()

    if not skip_comparison:
        run_comparison(
            cfg=cfg,
            n_runs=n_runs,
            seed_base=seed_base,
            output_dir=output_dir,
        )

    if not skip_coverage:
        run_coverage(
            cfg=cfg,
            n_runs=n_runs,
            n_bootstrap=n_bootstrap,
            seed_base=seed_base,
            output_dir=output_dir,
            coverage_models=coverage_models,
        )

    print("\n" + "=" * 84)
    print("WEEK 1 COMPLETED")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Elapsed: {time.perf_counter() - t0:.1f} sec")
    print("=" * 84)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_models(value: str) -> List[str]:
    models = [x.strip().lower() for x in value.split(",") if x.strip()]
    allowed = {"naive", "cf", "oracle"}
    bad = [x for x in models if x not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Неизвестные модели: {bad}. Допустимо: naive,cf,oracle."
        )
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Week 1 Monte Carlo validation for CF-Cox/2SRI"
    )

    parser.add_argument(
        "--mode",
        choices=["comparison", "coverage", "all"],
        default="all",
        help="Что запускать.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test: маленькие R/B/n.",
    )

    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=42)

    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--clusters", type=int, default=DEFAULT_G)
    parser.add_argument(
        "--instrument-strength",
        type=float,
        default=DEFAULT_INSTRUMENT_STRENGTH,
    )
    parser.add_argument(
        "--baseline-shape",
        type=float,
        default=DEFAULT_BASELINE_SHAPE,
    )
    parser.add_argument(
        "--censoring-scale",
        type=float,
        default=DEFAULT_CENSORING_SCALE,
    )
    parser.add_argument(
        "--baseline-hazard",
        type=float,
        default=DEFAULT_BASELINE_HAZARD,
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        choices=[0.90, 0.95, 0.99],
    )

    parser.add_argument(
        "--coverage-models",
        type=_parse_models,
        default=["naive", "cf", "oracle"],
        help="Например: naive,cf,oracle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.quick:
        n = args.n or 2_000
        n_runs = args.n_runs or 3
        n_bootstrap = args.n_bootstrap or 20
    else:
        n = args.n or DEFAULT_N
        n_runs = args.n_runs or 100
        n_bootstrap = args.n_bootstrap or 200

    cfg = ExperimentConfig(
        gamma_true=args.gamma,
        rho=args.rho,
        delta=args.delta,
        n=n,
        n_clusters=args.clusters,
        instrument_strength=args.instrument_strength,
        baseline_shape=args.baseline_shape,
        baseline_hazard=args.baseline_hazard,
        censoring_scale=args.censoring_scale,
        confidence_level=args.confidence_level,
    )

    print("=" * 84)
    print("METHODOLOGICAL TESTS v2")
    print("=" * 84)
    print(
        f"mode={args.mode}, quick={args.quick}, "
        f"n={cfg.n}, R={n_runs}, B={n_bootstrap}"
    )
    print(
        f"gamma={cfg.gamma_true}, rho={cfg.rho}, delta={cfg.delta}, "
        f"G={cfg.n_clusters}, pi_Z={cfg.instrument_strength}"
    )
    print(f"output={args.output_dir.resolve()}")
    print("=" * 84)

    if args.mode in {"comparison", "all"}:
        run_comparison(
            cfg=cfg,
            n_runs=n_runs,
            seed_base=args.seed_base,
            output_dir=args.output_dir,
        )

    if args.mode in {"coverage", "all"}:
        run_coverage(
            cfg=cfg,
            n_runs=n_runs,
            n_bootstrap=n_bootstrap,
            seed_base=args.seed_base,
            output_dir=args.output_dir,
            coverage_models=args.coverage_models,
        )

    print("\nГотово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
