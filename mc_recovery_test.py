#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo recovery test for CF-Cox.

Purpose
-------
Verify that the estimator can recover a known structural coefficient
``gamma_true`` under the DGP used to generate the data.

Important statistical distinction
---------------------------------
The Monte Carlo distribution of gamma_hat is NOT a confidence interval for
one fitted sample. Therefore the old implementation's ``coverage`` value,
which put gamma_true inside one percentile interval of all gamma_hat values,
was not a valid coverage calculation.

This module now reports:
* bias, relative bias, SD and RMSE of gamma_hat;
* Monte Carlo SE of the reported mean and bias;
* empirical 95% interval of the estimator distribution (descriptive only);
* failure rate and penalized-fit rate;
* first-stage classical and cluster F statistics.

Optional cluster-bootstrap coverage can be requested with ``n_bootstrap > 0``.
That coverage is computed per Monte Carlo replication from a cluster bootstrap
CI and is the only field labelled as 95% coverage.

Hardware recommendations
------------------------
AMD Ryzen 5 7500F (6 cores / 12 threads):
    --n-jobs 6   (physical cores, recommended)
    --n-jobs -1  (all 12 threads, marginal gain)
32 GB RAM: sufficient for 6 parallel workers with n_tractors=40000.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from Итог import (
    CFFitOptions,
    DGPParameters,
    fit_cf_cox,
    fit_first_stage,
    generate_data,
)
from train_model import make_rng
from mc_recovery_stats import summarize_gamma_recovery

DEFAULT_N_SIMS = 100
DEFAULT_N_TRACTORS = 40000
DEFAULT_GAMMA_TRUE = 0.5
DEFAULT_SEED = 42


def _make_fit_options() -> CFFitOptions:
    """CF-Cox options used consistently by every MC replication."""
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
        save_tracebacks=False,
        cluster_col="cluster_id",
    )


def _cluster_bootstrap_gamma(
    data: pd.DataFrame,
    opts: CFFitOptions,
    n_bootstrap: int,
    seed: int,
) -> Tuple[Optional[float], Optional[Tuple[float, float]], int]:
    """
    Cluster bootstrap SE and percentile CI for one MC replication.

    Clusters are sampled with replacement and all rows in a sampled cluster
    are retained. This preserves within-cluster dependence represented by
    ``cluster_id`` in the DGP.
    """
    if n_bootstrap <= 1 or "cluster_id" not in data.columns:
        return None, None, 0

    clusters = pd.unique(data["cluster_id"])
    if len(clusters) < 2:
        return None, None, 0

    rng = np.random.default_rng(seed)
    estimates: List[float] = []

    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        parts = []
        for cluster in sampled:
            block = data.loc[data["cluster_id"] == cluster]
            if not block.empty:
                parts.append(block)
        if not parts:
            continue

        bdata = pd.concat(parts, ignore_index=True)

        repeated_ids: List[str] = []
        for boot_id, block in enumerate(parts):
            repeated_ids.extend([str(boot_id)] * len(block))
        bdata["cluster_id"] = repeated_ids

        try:
            if np.var(bdata["Z"].to_numpy(dtype=float), ddof=0) < opts.var_z_threshold:
                continue
            fs = fit_first_stage(bdata, opts)
            cf = fit_cf_cox(bdata, fs, opts)
            if np.isfinite(cf.gamma_hat):
                estimates.append(float(cf.gamma_hat))
        except Exception:
            continue

    if len(estimates) < 50:
        return None, None, len(estimates)

    arr = np.asarray(estimates, dtype=float)
    se = float(np.std(arr, ddof=1))
    low, high = np.percentile(arr, [2.5, 97.5])
    return se, (float(low), float(high)), len(estimates)


def _run_single_replication(
    rep_id: int,
    seed_base: int,
    n_tractors: int,
    gamma_true: float,
    n_bootstrap: int,
) -> Dict[str, Any]:
    """
    Одна MC-репликация. Возвращает dict с результатами или ошибкой.

    Эта функция вызывается либо sequentially, либо через joblib.Parallel.
    """
    result: Dict[str, Any] = {
        "rep_id": rep_id,
        "_success": False,
        "_error": None,
    }

    try:
        rng = make_rng(seed_base, rep_id)
        dgp = DGPParameters(
            gamma=gamma_true,
            rho=0.7,
            delta=0.5,
            intercept=10.0,
            structural_intercept=10.0,
            first_stage_z_coef=0.5,
            use_real_covariates=True,
            weather_campaign="sowing",
            soil_source="soil_real",
        )

        data = generate_data(
            n=n_tractors,
            contamination=False,
            baseline_hazard=1.19e-8,
            censoring_scale=1382.0,
            rng=rng,
            dgp=dgp,
            instrument_source="weather_real",
        )

        opts = _make_fit_options()
        fs = fit_first_stage(data, opts)
        cf = fit_cf_cox(data, fs, opts)

        result.update({
            "gamma_hat": float(cf.gamma_hat),
            "lambda_hat": float(cf.cf_coef_signed),
            "gamma_naive_se": float(cf.naive_model_se),
            "f_stat_classical": float(fs.report.classical_f),
            "f_stat_cluster": float(fs.report.cluster_f),
            "n_clusters": int(fs.report.n_clusters),
            "penalized": bool(cf.is_penalized),
        })

        if n_bootstrap > 0:
            boot_seed = int(seed_base + 10_000_000 + rep_id)
            se_b, ci_b, n_ok = _cluster_bootstrap_gamma(
                data=data,
                opts=opts,
                n_bootstrap=n_bootstrap,
                seed=boot_seed,
            )
            result["bootstrap_successes"] = int(n_ok)
            if se_b is not None and ci_b is not None:
                result["bootstrap_se"] = float(se_b)
                result["bootstrap_ci_low"] = float(ci_b[0])
                result["bootstrap_ci_high"] = float(ci_b[1])
                result["bootstrap_covered"] = bool(
                    ci_b[0] <= gamma_true <= ci_b[1]
                )
            else:
                result["bootstrap_se"] = np.nan
                result["bootstrap_ci_low"] = np.nan
                result["bootstrap_ci_high"] = np.nan
                result["bootstrap_covered"] = False
        else:
            result["bootstrap_se"] = np.nan
            result["bootstrap_ci_low"] = np.nan
            result["bootstrap_ci_high"] = np.nan
            result["bootstrap_covered"] = np.nan
            result["bootstrap_successes"] = 0

        result["_success"] = True

    except Exception as exc:
        result["_error"] = f"{type(exc).__name__}: {exc}"

    return result


def run_mc_recovery(
    n_sims: int = DEFAULT_N_SIMS,
    n_tractors: int = DEFAULT_N_TRACTORS,
    gamma_true: float = DEFAULT_GAMMA_TRUE,
    seed_base: int = DEFAULT_SEED,
    n_bootstrap: int = 0,
    n_jobs: int = 1,
    output_path: Path | str = "mc_recovery_results.csv",
):
    """
    Run Monte Carlo recovery of the known structural gamma.

    Parameters
    ----------
    n_sims : int
        Number of Monte Carlo replications.
    n_tractors : int
        Number of tractors per replication.
    gamma_true : float
        True structural coefficient to recover.
    seed_base : int
        Base seed for reproducibility.
    n_bootstrap : int
        Cluster-bootstrap replications per MC dataset.
        0 = fast recovery run (no coverage).
        >0 = per-replication 95% bootstrap coverage.
    n_jobs : int
        Number of parallel workers.
        1 = sequential (default, for debugging).
        -1 = all available CPU cores.
        Recommended for Ryzen 5 7500F: 6 (physical cores).
    output_path : Path
        Output CSV path.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be > 0")
    if n_tractors <= 0:
        raise ValueError("n_tractors must be > 0")
    if not np.isfinite(gamma_true):
        raise ValueError("gamma_true must be finite")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be >= 0")

    print("=" * 70)
    print("MONTE CARLO RECOVERY TEST")
    print("=" * 70)
    print(f"  n_sims:       {n_sims}")
    print(f"  n_tractors:   {n_tractors:,}")
    print(f"  gamma_true:   {gamma_true}")
    print(f"  seed:         {seed_base}")
    print(f"  n_bootstrap:  {n_bootstrap}")
    print(f"  n_jobs:       {n_jobs}")
    print("=" * 70)

    t_start = time.time()

    # ─── Sequential или Parallel выполнение ──────────────────────────
    if n_jobs == 1 or not HAS_JOBLIB:
        if not HAS_JOBLIB and n_jobs != 1:
            print(
                "WARNING: joblib not installed. "
                "Falling back to sequential execution. "
                "Install with: pip install joblib"
            )

        all_results: List[Dict[str, Any]] = []
        for i in range(n_sims):
            t_rep_start = time.time()
            res = _run_single_replication(
                rep_id=i,
                seed_base=seed_base,
                n_tractors=n_tractors,
                gamma_true=gamma_true,
                n_bootstrap=n_bootstrap,
            )
            all_results.append(res)
            t_rep = time.time() - t_rep_start

            if res["_success"]:
                print(
                    f"[{i+1}/{n_sims}] ({t_rep:.1f}s) "
                    f"gamma_hat = {res['gamma_hat']:+.4f} "
                    f"(true = {gamma_true})"
                )
                print(f"  lambda_hat = {res['lambda_hat']:+.4f}")
                print(f"  F classical = {res['f_stat_classical']:.1f}")
                print(f"  F cluster = {res['f_stat_cluster']:.1f}")
                print(f"  n_clusters = {res['n_clusters']}")
                print(f"  penalized = {res['penalized']}")
            else:
                print(f"[{i+1}/{n_sims}] FAILED: {res['_error']}")

    else:
        print(f"\nParallel execution: n_jobs={n_jobs}")
        print("Progress will be shown below.\n")

        all_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_run_single_replication)(
                rep_id=i,
                seed_base=seed_base,
                n_tractors=n_tractors,
                gamma_true=gamma_true,
                n_bootstrap=n_bootstrap,
            )
            for i in range(n_sims)
        )

        for res in all_results:
            if res["_success"]:
                print(
                    f"[{res['rep_id']+1}/{n_sims}] "
                    f"gamma_hat = {res['gamma_hat']:+.4f} "
                    f"F_cluster = {res['f_stat_cluster']:.0f} "
                    f"penalized = {res['penalized']}"
                )
            else:
                print(
                    f"[{res['rep_id']+1}/{n_sims}] FAILED: {res['_error']}"
                )

    t_total = time.time() - t_start

    # ─── Разделение успешных и неудачных ─────────────────────────────
    successful = [r for r in all_results if r["_success"]]
    failed = [r for r in all_results if not r["_success"]]

    gamma_hats = [r["gamma_hat"] for r in successful]
    lambda_hats = [r["lambda_hat"] for r in successful]
    gamma_naive_se = [r["gamma_naive_se"] for r in successful]
    f_stats = [r["f_stat_classical"] for r in successful]
    f_stats_cluster = [r["f_stat_cluster"] for r in successful]
    n_clusters_list = [r["n_clusters"] for r in successful]
    penalized_list = [r["penalized"] for r in successful]
    bootstrap_se = [r["bootstrap_se"] for r in successful]
    bootstrap_ci_low = [r["bootstrap_ci_low"] for r in successful]
    bootstrap_ci_high = [r["bootstrap_ci_high"] for r in successful]
    bootstrap_covered = [r["bootstrap_covered"] for r in successful]
    bootstrap_successes = [r["bootstrap_successes"] for r in successful]

    failures = [
        f"simulation_{r['rep_id']}: {r['_error']}" for r in failed
    ]

    # ─── Summary statistics ──────────────────────────────────────────
    gamma_arr = np.asarray(gamma_hats, dtype=float)
    summary = (
        summarize_gamma_recovery(gamma_arr, gamma_true)
        if gamma_arr.size
        else {}
    )

    bootstrap_coverage_values = [
        r["bootstrap_covered"]
        for r in successful
        if isinstance(r["bootstrap_covered"], bool)
    ]

    summary.update({
        "requested_sims": int(n_sims),
        "failure_count": int(len(failures)),
        "failure_rate": float(len(failures) / n_sims) if n_sims > 0 else 0.0,
        "mean_f_classical": (
            float(np.nanmean(f_stats)) if f_stats else np.nan
        ),
        "min_f_classical": (
            float(np.nanmin(f_stats)) if f_stats else np.nan
        ),
        "max_f_classical": (
            float(np.nanmax(f_stats)) if f_stats else np.nan
        ),
        "mean_f_cluster": (
            float(np.nanmean(f_stats_cluster)) if f_stats_cluster else np.nan
        ),
        "min_f_cluster": (
            float(np.nanmin(f_stats_cluster)) if f_stats_cluster else np.nan
        ),
        "max_f_cluster": (
            float(np.nanmax(f_stats_cluster)) if f_stats_cluster else np.nan
        ),
        "mean_n_clusters": (
            float(np.mean(n_clusters_list)) if n_clusters_list else np.nan
        ),
        "penalized_fit_rate": (
            float(np.mean(penalized_list)) if penalized_list else np.nan
        ),
        "coverage_method": (
            "cluster_bootstrap_percentile"
            if n_bootstrap > 0
            else "not_computed"
        ),
        "coverage_95": (
            float(np.mean(bootstrap_coverage_values))
            if bootstrap_coverage_values and n_bootstrap > 0
            else np.nan
        ),
        "coverage_n": (
            int(len(bootstrap_coverage_values)) if n_bootstrap > 0 else 0
        ),
        "bootstrap_success_median": (
            float(np.median(bootstrap_successes))
            if bootstrap_successes
            else np.nan
        ),
        "n_jobs": int(n_jobs),
        "total_time_seconds": float(t_total),
        "mean_time_per_sim_seconds": float(t_total / n_sims) if n_sims > 0 else 0.0,
    })

    # ─── Вывод результатов ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MONTE CARLO RECOVERY RESULTS")
    print("=" * 70)
    for key in (
        "gamma_true",
        "n_successful",
        "mean_gamma_hat",
        "bias",
        "relative_bias",
        "sd_gamma_hat",
        "rmse",
        "mc_se_mean",
        "empirical_95_low",
        "empirical_95_high",
        "failure_rate",
        "mean_f_cluster",
        "coverage_95",
        "total_time_seconds",
        "mean_time_per_sim_seconds",
    ):
        if key in summary:
            val = summary[key]
            if isinstance(val, float):
                print(f"{key}: {val:.6f}")
            else:
                print(f"{key}: {val}")

    # ─── Сохранение результатов ──────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_success = len(successful)
    results = pd.DataFrame({
        "rep_id": [r["rep_id"] for r in successful],
        "gamma_hat": gamma_hats,
        "lambda_hat": lambda_hats,
        "gamma_naive_se": gamma_naive_se,
        "f_stat_classical": f_stats,
        "f_stat_cluster": f_stats_cluster,
        "n_clusters": n_clusters_list,
        "penalized": penalized_list,
        "bootstrap_se": bootstrap_se,
        "bootstrap_ci_low": bootstrap_ci_low,
        "bootstrap_ci_high": bootstrap_ci_high,
        "bootstrap_covered": bootstrap_covered,
        "bootstrap_successes": bootstrap_successes,
    })

    results.to_csv(output_path, index=False)

    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    failures_path = output_path.with_name(output_path.stem + "_failures.txt")
    failures_path.write_text("\n".join(failures), encoding="utf-8")

    print(f"\nResults saved to {output_path}")
    print(f"Summary saved to {summary_path}")
    if failures:
        print(f"Failures saved to {failures_path}")

    print(f"\nTotal time: {t_total:.1f}s "
          f"({t_total/60:.1f} min)")

    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monte Carlo recovery test for CF-Cox"
    )
    parser.add_argument(
        "--n-sims", type=int, default=DEFAULT_N_SIMS,
        help="Number of Monte Carlo replications.",
    )
    parser.add_argument(
        "--n-tractors", type=int, default=DEFAULT_N_TRACTORS,
        help="Number of tractors per replication.",
    )
    parser.add_argument(
        "--gamma-true", type=float, default=DEFAULT_GAMMA_TRUE,
        help="True structural coefficient gamma.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Base seed for reproducibility.",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=0,
        help=(
            "Cluster-bootstrap replications per MC dataset. "
            ">0 enables valid per-replication 95%% coverage."
        ),
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help=(
            "Number of parallel CPU workers. "
            "1 = sequential (default). "
            "-1 = all available cores. "
            "Recommended for Ryzen 5 7500F: 6."
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("mc_recovery_results.csv"),
        help="Output CSV path.",
    )

    args = parser.parse_args()

    run_mc_recovery(
        n_sims=args.n_sims,
        n_tractors=args.n_tractors,
        gamma_true=args.gamma_true,
        seed_base=args.seed,
        n_bootstrap=args.n_bootstrap,
        n_jobs=args.n_jobs,
        output_path=args.output,
    )