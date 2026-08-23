#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conftest.py
Общие fixtures для всех тестов проекта.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Add project root to sys.path for imports
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ---------------------------------------------------------------------------
# Импорт модулей проекта
# ---------------------------------------------------------------------------
from constants import (
    BRAND_MAP,
    BRAND_TO_CODE,
    CALIBRATION_HORIZON_DAYS,
    CALIBRATION_HORIZON_ENGINE_HOURS,
    DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY,
    FREQ_SHARES,
    MAJOR_FAILURE_SHARE,
    MODEL_TIME_UNIT,
    MTBF_BASELINE_HOURS,
    POWER_SEGMENT_THRESHOLDS,
    SEVERITY_WEIGHTS,
    VALID_EVENT_DEFINITIONS,
)
from prediction_engine import (
    ModelParameters,
    load_model_params,
    save_model_params,
    validate_model,
    predict_probability,
    predict_many,
    baseline_cumulative_hazard,
    transform_peak,
)
from premium_engine import (
    calculate_single_premium,
    calculate_premium,
)


# ---------------------------------------------------------------------------
# Фикстура: минимальная валидная модель
# ---------------------------------------------------------------------------
def _build_minimal_model() -> ModelParameters:
    """Создать минимальную валидную модель для тестов."""
    times = [0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 1712.0, 2000.0]
    values = [0.0, 0.005, 0.012, 0.035, 0.075, 0.12, 0.14, 0.17]

    return ModelParameters(
        model_version="0.2",
        prediction_engine_version="3.2.1-fixed",
        metadata={},
        transform_info={
            "type": "standardize",
            "center": 0.7099,
            "scale": 0.2053,
        },
        first_stage={
            "exog_names": ["const", "Z", "x_age", "x_hours"],
            "params": {
                "const": 0.71,
                "Z": 0.5,
                "x_age": 0.15,
                "x_hours": 0.10,
            },
        },
        cox={
            "exog_names": ["PeakLoad", "v_hat", "x_age", "x_hours"],
            "coefs": {
                "PeakLoad": 0.5,
                "v_hat": 0.3,
                "x_age": 0.2,
                "x_hours": 0.1,
            },
        },
        baseline_cumulative_hazard={
            "times": times,
            "values": values,
        },
        template_covariates={
            "Z": 0.0,
            "x_age": 0.0,
            "x_hours": 0.0,
        },
        partial_out_X_beta=0.0,
        training_pl_hat_mean=0.71,
        training_x_mean=0.0,
        training_x_means={"x_age": 0.0, "x_hours": 0.0},
        partial_out_all_betas={},
        training_residuals_mean=0.0,
        training_residuals_std=1.0,
        training_first_stage_fitted=[],
        training_residuals_arr=[],
        training_meta={
            "model_version": "3.0",
            "time_unit": "engine_hours",
            "calibration_time_horizon": 1712.0,
            "calibration_time_horizon_days": 214.0,
            "peakload_min": 0.0,
            "peakload_max": 2.0,
            "event_definition": "major_claim",
            "allow_baseline_extrapolation": True,
            "default_engine_hours_per_calendar_day": 8.0,
            "x_standardization": {
                "x_age": {"raw_col": "Age", "shift": 10.0, "scale": 10.0},
                "x_hours": {"raw_col": "Hours", "shift": 1000.0, "scale": 1000.0},
            },
            "brand_mapping": BRAND_TO_CODE,
        },
        cf_basis_metadata={
            "v_hat_basis": "linear",
            "v_hat_cols": ["v_hat"],
            "residuals_mean": 0.0,
            "residuals_std": 1.0,
            "linear_standardized": True,
        },
        calibration_time_horizon=1712.0,
        brand_mapping=BRAND_TO_CODE,
        brand_effects={},
        event_definition="major_claim",
        competing_risks=False,
        minor_failure_rate=0.002,
        segment="light",
        allow_baseline_extrapolation=True,
    )


@pytest.fixture
def minimal_model() -> ModelParameters:
    """Минимальная валидная модель."""
    return _build_minimal_model()


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    """Сохранённая модель во временном файле."""
    model = _build_minimal_model()
    path = tmp_path / "model_params.json"
    save_model_params(path, model)
    return path


@pytest.fixture
def loaded_model(model_file: Path) -> ModelParameters:
    """Модель, загруженная из файла."""
    return load_model_params(model_file, validate=True)


# ---------------------------------------------------------------------------
# Фикстура: параметры для premium
# ---------------------------------------------------------------------------
@pytest.fixture
def premium_params() -> Dict[str, Any]:
    """Стандартные параметры для расчёта премии."""
    return {
        "probability": 0.028,
        "sum_insured": 5_000_000.0,
        "theta": 0.15,
        "discount_rate": 0.08,
        "policy_horizon_days": 214.0,
    }


# ---------------------------------------------------------------------------
# Фикстура: синтетические claims
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_claims() -> "pd.DataFrame":
    """Синтетические claims-данные для тестов severity/backtesting."""
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 300
    n_events = 80

    events_mask = np.zeros(n, dtype=bool)
    events_mask[:n_events] = True
    rng.shuffle(events_mask)

    failure_times = rng.exponential(2000.0, size=n)
    failure_times = np.clip(failure_times, 1.0, 5000.0)

    repair_costs = np.where(
        events_mask,
        rng.lognormal(11.0, 1.0, size=n),
        0.0,
    )
    repair_costs = np.clip(repair_costs, 0.0, None)

    downtime_hours = np.where(
        events_mask,
        rng.exponential(20.0, size=n),
        0.0,
    )

    major_flags = np.where(
        events_mask,
        (rng.random(n) < 0.30).astype(int),
        0,
    )

    systems = rng.choice(
        ["гидравлика", "электроника", "двигатель", "трансмиссия", "прочее"],
        size=n,
        p=[0.30, 0.30, 0.12, 0.20, 0.08],
    )

    brands = rng.choice(
        list(BRAND_MAP.values()),
        size=n,
        p=[0.35, 0.08, 0.07, 0.10, 0.40],
    )

    return pd.DataFrame({
        "machine_id": [f"M{i:04d}" for i in range(n)],
        "brand": brands,
        "power_hp": rng.uniform(80.0, 350.0, size=n),
        "production_year": rng.integers(2000, 2020, size=n),
        "age_at_event": rng.uniform(1.0, 25.0, size=n),
        "hours_at_event": rng.uniform(100.0, 15000.0, size=n),
        "failure_time": failure_times,
        "event_flag": events_mask.astype(int),
        "failure_system": systems,
        "major_failure_flag": major_flags,
        "repair_cost": repair_costs,
        "downtime_hours": downtime_hours,
        "claim_amount": repair_costs * 1.2,
    })