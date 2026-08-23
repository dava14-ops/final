#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MC-2 Recovery Experiment: Enterprise Quality Index (EQI)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from Итог import DGPParameters, generate_data, fit_first_stage, fit_cf_cox, CFFitOptions
from enterprise_quality import validate_enterprise_structure

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def run_single_replication(
    replication_id: int,
    dgp: DGPParameters,
    n: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed + replication_id)

    # Генерация данных
    data = generate_data(
        n=n,
        contamination=False,
        baseline_hazard=1.19e-8,  # Используем реалистичный baseline из ваших прошлых запусков
        censoring_scale=1382.0,
        rng=rng,
        dgp=dgp,
        contamination_probability=0.0,
        instrument_strength=0.5,
        instrument_source="normal",
    )

    # Валидация enterprise structure
    if dgp.use_enterprise_quality:
        if not validate_enterprise_structure(
            data["x_enterprise_quality"].values,
            data["enterprise_id"].values,
        ):
            raise ValueError(
                f"Replication {replication_id}: enterprise structure invalid"
            )

    # Настройки для CF-Cox
    opts = CFFitOptions(
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
    )

    # Первая стадия
    first_stage = fit_first_stage(data, opts)

    # CF-Cox
    cf_result = fit_cf_cox(data, first_stage, opts)

    # Извлечение коэффициентов
    coefs = cf_result.cph.params_
    se = cf_result.cph.standard_errors_

    # γ recovery
    gamma_hat = float(coefs.get("PeakLoad", np.nan))
    gamma_se = float(se.get("PeakLoad", np.nan))
    gamma_ci_low = gamma_hat - 1.96 * gamma_se
    gamma_ci_high = gamma_hat + 1.96 * gamma_se
    gamma_covered = (dgp.gamma >= gamma_ci_low) and (dgp.gamma <= gamma_ci_high)

    # β_EQI recovery
    beta_eqi_hat = float(coefs.get("x_enterprise_quality", np.nan))
    beta_eqi_se = float(se.get("x_enterprise_quality", np.nan))
    beta_eqi_ci_low = beta_eqi_hat - 1.96 * beta_eqi_se
    beta_eqi_ci_high = beta_eqi_hat + 1.96 * beta_eqi_se
    beta_eqi_covered = (dgp.beta_eqi >= beta_eqi_ci_low) and (
        dgp.beta_eqi <= beta_eqi_ci_high
    )

    # CF λ recovery
    lambda_hat = float(coefs.get("v_hat", np.nan))

    # First stage diagnostics
    f_statistic = float(first_stage.report.classical_f)

    return {
        "replication_id": replication_id,
        "gamma_hat": gamma_hat,
        "gamma_se": gamma_se,
        "gamma_ci_low": gamma_ci_low,
        "gamma_ci_high": gamma_ci_high,
        "gamma_covered": gamma_covered,
        "beta_eqi_hat": beta_eqi_hat,
        "beta_eqi_se": beta_eqi_se,
        "beta_eqi_ci_low": beta_eqi_ci_low,
        "beta_eqi_ci_high": beta_eqi_ci_high,
        "beta_eqi_covered": beta_eqi_covered,
        "beta_eqi_negative": beta_eqi_hat < 0,
        "lambda_hat": lambda_hat,
        "f_statistic": f_statistic,
    }


def main():
    N_REPLICATIONS = 100
    N_OBSERVATIONS = 40000
    N_ENTERPRISES = 500
    BASE_SEED = 12345

    TRUE_GAMMA = 0.5
    TRUE_BETA_EQI = -0.5

    logger.info("=" * 70)
    logger.info("MC-2 RECOVERY EXPERIMENT: Enterprise Quality Index")
    logger.info("=" * 70)
    logger.info(f"Replications: {N_REPLICATIONS}")
    logger.info(f"Observations per replication: {N_OBSERVATIONS}")
    logger.info(f"Enterprises: {N_ENTERPRISES}")
    logger.info(f"Tractors per enterprise: {N_OBSERVATIONS // N_ENTERPRISES}")
    logger.info(f"True γ: {TRUE_GAMMA}")
    logger.info(f"True β_EQI: {TRUE_BETA_EQI}")

    dgp = DGPParameters(
        gamma=TRUE_GAMMA,
        rho=0.7,
        delta=0.5,
        intercept=10.0,
        structural_intercept=10.0,
        first_stage_z_coef=0.5,
        baseline_family="weibull",
        baseline_shape=1.88,
        brand_encoding="dummies",
        competing_risks=True,
        minor_failure_rate=0.002,
        event_definition="major_claim",
        segment="light",
        # EQI parameters
        beta_eqi=TRUE_BETA_EQI,
        n_enterprises=N_ENTERPRISES,
        use_enterprise_quality=True,
    )

    results: List[Dict[str, float]] = []

    for rep_id in range(N_REPLICATIONS):
        logger.info(f"Replication {rep_id + 1}/{N_REPLICATIONS}")
        try:
            result = run_single_replication(rep_id, dgp, N_OBSERVATIONS, BASE_SEED)
            results.append(result)
        except Exception as exc:
            logger.error(f"Replication {rep_id} failed: {exc}")
            continue

    df = pd.DataFrame(results)

    logger.info("")
    logger.info("=" * 70)
    logger.info("MC-2 RECOVERY RESULTS")
    logger.info("=" * 70)
    logger.info(f"Successful replications: {len(df)}/{N_REPLICATIONS}")

    # γ recovery
    logger.info("--- γ recovery (true = %.2f) ---", TRUE_GAMMA)
    logger.info(f"  Mean:    {df['gamma_hat'].mean():+.6f}")
    logger.info(f"  SD:      {df['gamma_hat'].std():.6f}")
    logger.info(f"  Coverage: {df['gamma_covered'].mean() * 100:.1f}%")

    gamma_bias = df["gamma_hat"].mean() - TRUE_GAMMA
    logger.info(f"  Bias:    {gamma_bias:+.6f}")

    # β_EQI recovery
    logger.info("--- β_EQI recovery (true = %.2f) ---", TRUE_BETA_EQI)
    logger.info(f"  Mean:    {df['beta_eqi_hat'].mean():+.6f}")
    logger.info(f"  SD:      {df['beta_eqi_hat'].std():.6f}")
    logger.info(f"  Coverage: {df['beta_eqi_covered'].mean() * 100:.1f}%")
    logger.info(f"  P(β<0):  {100 * df['beta_eqi_negative'].mean():.1f}%")

    beta_eqi_bias = df["beta_eqi_hat"].mean() - TRUE_BETA_EQI
    logger.info(f"  Bias:    {beta_eqi_bias:+.6f}")

    # Сохранение
    output_dir = Path("mc_results")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "mc2_recovery_eqi_detailed.csv", index=False)

    summary = {
        "experiment": "MC-2 EQI recovery",
        "n_replications": len(df),
        "true_gamma": TRUE_GAMMA,
        "true_beta_eqi": TRUE_BETA_EQI,
        "gamma_recovery": {
            "mean": df["gamma_hat"].mean(),
            "bias": gamma_bias,
            "coverage": df["gamma_covered"].mean(),
        },
        "beta_eqi_recovery": {
            "mean": df["beta_eqi_hat"].mean(),
            "bias": beta_eqi_bias,
            "coverage": df["beta_eqi_covered"].mean(),
            "sign_check": df["beta_eqi_negative"].mean(),
        },
    }

    with open(output_dir / "mc2_recovery_eqi_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Результаты сохранены в {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
