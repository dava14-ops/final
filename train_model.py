#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py (v3.0 hardened + v0.2 patches + review fixes)

Review fixes:
- TrainingConfig defaults aligned with constants.py;
- fit_first_stage_and_cf now accepts and returns transform_info /
  achieved_event_rate / peak_stats / training_meta_flat, so artifacts
  built after weak-instrument auto-correction use fresh metadata;
- MODEL_SEMANTIC_VERSION is now persisted in training_meta.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import platform
import sys
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger("train_model")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# noinspection PyUnresolvedReferences
from Итог import generate_data  # type: ignore[attr-defined]

from Итог import partial_f_statistic_for_z

# noinspection PyUnresolvedReferences
from Итог import DGPParameters  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from Итог import fit_first_stage as _fit_first_stage_original  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from Итог import (
    fit_first_stage_cluster_robust,  # type: ignore[attr-defined]
)  # cluster-robust F-statistic

# noinspection PyUnresolvedReferences
from Итог import fit_cf_cox  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from Итог import calibrate_censoring_scale_deterministic  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from Итог import classical_first_stage_f_statistic  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from Итог import cragg_donald_stat  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from Итог import endogeneity_lr_test  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from Итог import interaction_lr_test  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from Итог import X_STANDARDIZATION  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from Итог import CFFitOptions  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from Итог import FirstStageFit  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from Итог import (  # type: ignore[attr-defined]
    load_claims_for_training,
    prepare_claims_for_cf,
    run_first_stage_on_claims,
    fit_cf_cox_on_claims,
    partial_f_statistic_for_z,
)

# Import centralized constants
from constants import (
    MODEL_TIME_UNIT,
    DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY,
    CALIBRATION_HORIZON_DAYS,
    CALIBRATION_HORIZON_ENGINE_HOURS,
    MAJOR_FAILURE_SHARE,
    MTBF_BASELINE_HOURS,
    DEFAULT_WEIBULL_SHAPE,
    BRAND_MAP,
    BRAND_TO_CODE,
    CLIMATE_INDEX_REFERENCE,
    SOIL_INDEX_REFERENCE,
    DEFAULT_BRAND_PROB_BY_CODE,
    PL_HAT_EXOG_CONVENTION,
)

# ---------------------------------------------------------------------------
# v0.2 imports with safe fallbacks
# ---------------------------------------------------------------------------
try:
    # noinspection PyUnresolvedReferences
    from Итог import FREQ_SHARES  # type: ignore[attr-defined]
except ImportError:
    logger.warning("FREQ_SHARES не найден в Итог.py, используется fallback.")
    FREQ_SHARES = {
        "minor": 0.70,
        "major": 0.30,
    }

try:
    # noinspection PyUnresolvedReferences
    from Итог import SEVERITY_WEIGHTS  # type: ignore[attr-defined]
except ImportError:
    logger.warning("SEVERITY_WEIGHTS не найден в Итог.py, используется fallback.")
    SEVERITY_WEIGHTS = {
        "minor": 0.25,
        "major": 1.00,
    }

try:
    # noinspection PyUnresolvedReferences
    from Итог import EVENT_DEFINITIONS  # type: ignore[attr-defined]
except ImportError:
    logger.warning("EVENT_DEFINITIONS не найден в Итог.py, используется fallback.")
    EVENT_DEFINITIONS = ("total_loss", "major_claim", "any_failure")

try:
    # noinspection PyUnresolvedReferences
    from Итог import major_failure_beta_prior  # type: ignore[attr-defined]
except ImportError:
    logger.warning("major_failure_beta_prior не найден в Итог.py, используется fallback.")

    def major_failure_beta_prior(mean: float, effective_n: float) -> Dict[str, float]:
        """
        P-04: Beta-prior для major failure share.
        alpha = mean * effective_n
        beta  = (1 - mean) * effective_n
        """
        mean = float(mean)
        effective_n = float(effective_n)
        if not (0.0 < mean < 1.0):
            raise ValueError("major_failure_beta_prior: mean должно быть в (0, 1)")
        if effective_n <= 0.0:
            raise ValueError("major_failure_beta_prior: effective_n должно быть положительным")
        alpha = mean * effective_n
        beta = (1.0 - mean) * effective_n
        return {
            "alpha": alpha,
            "beta": beta,
            "mean": mean,
            "effective_n": effective_n,
        }


try:
    # noinspection PyUnresolvedReferences
    from Итог import RF_HEAVY_BRAND_CATALOG  # type: ignore[attr-defined]
except ImportError:
    logger.warning("RF_HEAVY_BRAND_CATALOG не найден в Итог.py, используется пустой fallback.")
    RF_HEAVY_BRAND_CATALOG = {}

# noinspection PyUnresolvedReferences
from prediction_engine import ModelParameters  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from prediction_engine import default_metadata  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from prediction_engine import save_model_params  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from prediction_engine import load_model_params  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from prediction_engine import validate_model  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from prediction_engine import predict_probability  # type: ignore[attr-defined]

# noinspection PyUnresolvedReferences
from prediction_engine import kaplan_meier_check  # type: ignore[attr-defined]

# ─── Параметрический базовый риск (v3.1) ───────────────────────────────────
try:
    from parametric_baseline import (
        fit_parametric_baseline,
        BaselineSpec,
        VALID_PARAMETRIC_FAMILIES,
    )

    HAS_PARAMETRIC_BASELINE = True
except ImportError:
    HAS_PARAMETRIC_BASELINE = False
    VALID_PARAMETRIC_FAMILIES = frozenset()
    logger.info("parametric_baseline.py не найден: параметрический базовый риск отключён")


# ★ FIX: локальные helpers для bootstrap SE (не экспортируются из prediction_engine)
def _dict_get_normalized(d: Any, key: Any, default: Any = None) -> Any:
    """Безопасное получение значения из dict с case-insensitive ключом."""
    if not isinstance(d, dict):
        return default
    target = str(key).lower()
    for k, v in d.items():
        if str(k).lower() == target:
            return v
    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    """Безопасное приведение к bool."""
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "т", "да"}
    return default


def _try_int(value: Any, default: int) -> int:
    """Безопасное приведение к int."""
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _try_int_optional(value: Any) -> Optional[int]:
    """Безопасное приведение к int или None."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# ★ FIX: _safe_series_scalar определён в Итог.py, импортируем
# noinspection PyUnresolvedReferences
from Итог import _safe_series_scalar  # type: ignore[attr-defined]

# Model registry for v1.0 claims-based models
from model_registry import generate_model_filename  # type: ignore[attr-defined]

# Model provenance helpers (weather campaign tracking)
try:
    from model_provenance import normalize_model_campaign_metadata  # type: ignore[attr-defined]
except ImportError:
    logger.info("model_provenance не найден: пропуск campaign provenance")
    normalize_model_campaign_metadata = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Safe type converters
# ---------------------------------------------------------------------------
def _as_float_or_none(value: Any) -> Optional[float]:
    """Безопасное приведение к float или None."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _require_float(value: Any, name: str) -> float:
    """Требовательное приведение к float с понятной ошибкой."""
    if value is None:
        raise ValueError(f"{name}: значение отсутствует")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: не удалось привести к float") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name}: значение должно быть конечным числом")
    return result


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_FORMAT_VERSION = "3.0"
ENGINE_CONVENTION = "cf_cox_v3"

# Фаза 8.3: семантическая версия модели
# major: 0 = simulation, 1 = real data, 2+ = architecture change
# minor: 0 = base, 1 = covariates/baseline change, 2 = CF/segment change
MODEL_SEMANTIC_VERSION = "0.2"

AUTO_INSTRUMENT_THRESHOLD = 14.18

# ─── Фаза 5.4: IV-режим ─────────────────────────────────────────────────────
# causal — инструмент валиден, γ интерпретируется каузально
# predictive — инструмент слаб/не подтверждён, γ только предсказательный
IV_MODE_CAUSAL = "causal"
IV_MODE_PREDICTIVE = "predictive"
VALID_IV_MODES = frozenset({IV_MODE_CAUSAL, IV_MODE_PREDICTIVE})

# УДАЛЕНО: MAX_AUTO_INCREASES и INCREASE_FACTOR больше не используются.
# Автокоррекция слабого инструмента методологически некорректна (см. Stock-Yogo 2005).
# Вместо этого при F < 10.4 происходит принудительное переключение в predictive режим.

# Time conventions
MTBF_INPUT_UNIT = "engine_hours"
MTBF_TO_MODEL_TIME_FACTOR = 1.0

# P-04: Beta-prior вместо точечной константы
MAJOR_FAILURE_SHARE_PRIOR = major_failure_beta_prior(
    mean=MAJOR_FAILURE_SHARE,
    effective_n=30.0,
)

# v0.2 reference metadata
# P-09: downtime convention
DOWNTIME_MODEL = "mttr"
# P-10: power segment reference threshold
POWER_SEGMENT_THRESHOLD = 150.0


# ---------------------------------------------------------------------------
# Training configuration container
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Все параметры обучения, собранные от пользователя."""

    # Dataset
    n: int = 40000
    seed: int = 12345

    # Contamination
    contamination: bool = False
    contamination_probability: float = 0.0
    stress_test_mode: bool = False

    # DGP core
    gamma: float = 0.5
    rho: float = 0.7
    delta: float = 0.7
    beta_age_hours: float = 0.15
    fs_intercept: float = 10.0
    structural_intercept: float = 10.0
    fs_z: float = 0.5
    fs_z_initial: float = 0.5

    # Baseline
    baseline_family: str = "weibull"
    baseline_shape: Optional[float] = DEFAULT_WEIBULL_SHAPE

    # Event model (P-03 / D1 / D2)
    target_event_rate: float = 0.02
    event_definition: str = "major_claim"
    competing_risks: bool = True
    minor_failure_rate: float = 0.002
    segment: str = "light"

    # Calibration
    calib_method: str = "probability"
    target_time: float = CALIBRATION_HORIZON_ENGINE_HOURS
    target_quantile: float = 0.028
    target_probability: Optional[float] = 0.028
    baseline_hazard_initial: Optional[float] = None

    # Transform
    do_standardize: bool = True
    do_center_only: bool = False

    # CF basis
    v_hat_basis: str = "linear"
    v_hat_basis_params: Optional[Dict[str, Any]] = None

    # Output
    out_path: str = "model_params.json"

    # TUM calibration (Фаза 3)
    tum_peakload_target_mean: Optional[float] = None
    tum_peakload_target_std: Optional[float] = None
    tum_stats_path: str = ""

    # IV mode (Фаза 5.4 / 6.6)
    weather_instrument_validated: bool = False
    instrument_source: str = "normal"  # normal | weather | weather_real
    weather_campaign: str = "sowing"  # sowing | harvest

    # ─── НОВОЕ ПОЛЕ: путь к ценовому инструменту Bartik ───────────
    price_instrument_path: str = "instrument_z_bartik.csv"

    # Soil source (Фаза 6.6)
    soil_source: str = "synthetic"  # synthetic | claims | soil_real

    # Фаза 8: Interaction Age × Hours
    beta_age_hours: float = 0.15  # синергетический эффект

    # ─── Фаза 9: Параметрический базовый риск ────────────────────────────
    # Семейство для подгонки под кривую Бреслоу.
    # "weibull" | "gompertz" | "exponential" | "none"
    # "none" = не подгонять (обратная совместимость).
    parametric_baseline_fit: str = "weibull"

    # ─── Фаза 9: форма модели ────────────────────────────────────────────
    # "control_function" — текущий путь с 2SRI / v_hat (казуальная попытка)
    # "reduced_form"     — упрощённый предиктивный путь: Кокс от [PeakLoad, X, Z]
    model_form: str = "control_function"

    # УДАЛЕНО: allow_auto_instrument_correction больше не используется.
    # Автокоррекция слабого инструмента удалена как методологически некорректная.
    # При F < 10.4 происходит автоматическое переключение в predictive режим.

    @property
    def instrument_strength(self) -> float:
        return float(self.fs_z)


# P-12: Kaplan-Meier validator flag
KAPLAN_MEIER_VALIDATOR_ENABLED = True

# DGP defaults
DEFAULT_GOMPERTZ_RATE = 0.01
DEFAULT_BETA_AGE = 0.20
DEFAULT_BETA_HOURS = 0.10
DEFAULT_BETA_CLIMATE = 0.20
DEFAULT_BETA_SOIL = 0.12
DEFAULT_BETA_POWER = -0.05
DEFAULT_GAMMA = 0.5
DEFAULT_DELTA = 0.7

# Calibration defaults
DEFAULT_TARGET_PROBABILITY = 0.028
CALIBRATION_TOLERANCE_ABS = 0.01
POST_FIT_TIGHT_TOLERANCE = 0.03
POST_FIT_FATAL_TOLERANCE = 0.10
MIN_BASELINE_SEARCH = 1e-12
MAX_BASELINE_SEARCH = 1e6
MAX_GOMPERTZ_EXP_ARGUMENT = 700.0

COVARIATE_MAPPING: Dict[str, str] = {
    "age_years": "x_age",
    "hours": "x_hours",
    "age_hours": "x_age_hours",  # Фаза 8: interaction Age × Hours
    "climate_index": "x_climate",
    "soil_index": "x_soil",
    "power": "x_power",
    "brand": "x_brand",
}


# ---------------------------------------------------------------------------
# Safe object/mapping helpers
# ---------------------------------------------------------------------------
def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Безопасно превращает dataclass-like объект в dict.
    Не использует asdict() напрямую, чтобы избежать тайпчекер-ошибок.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return {f.name: getattr(obj, f.name, None) for f in fields(obj)}
        except (TypeError, AttributeError):
            pass
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    return {"repr": repr(obj)}


def _safe_items(obj: Any) -> List[Tuple[Any, Any]]:
    """
    Безопасно возвращает список пар (key, value) из dict-like объекта.
    """
    if obj is None:
        return []
    if isinstance(obj, dict):
        return list(obj.items())

    items_fn = getattr(obj, "items", None)
    if items_fn is not None and callable(items_fn):
        try:
            return list(items_fn())
        except (TypeError, ValueError, AttributeError):
            return []
    return []


def _safe_keys(obj: Any) -> List[Any]:
    """
    Безопасно возвращает ключи из dict/Series/params-like объекта.
    """
    if obj is None:
        return []
    if isinstance(obj, dict):
        return list(obj.keys())

    index_obj = getattr(obj, "index", None)
    if index_obj is not None:
        try:
            return list(index_obj)
        except (TypeError, ValueError, AttributeError):
            pass

    keys_fn = getattr(obj, "keys", None)
    if keys_fn is not None and callable(keys_fn):
        try:
            return list(keys_fn())
        except (TypeError, ValueError, AttributeError):
            return []
    return []


def _safe_exog_names(model_obj: Any) -> Optional[List[str]]:
    """
    Безопасно извлекает exog_names из statsmodels-like model объекта.
    """
    if model_obj is None:
        return None

    names_obj = getattr(model_obj, "exog_names", None)
    if names_obj is None:
        return None

    try:
        return [str(name) for name in names_obj]
    except (TypeError, ValueError):
        return None


def _safe_params_to_float_dict(params_obj: Any) -> Dict[str, float]:
    """
    Безопасно превращает params-like объект в Dict[str, float].
    """
    result: Dict[str, float] = {}
    for key, value in _safe_items(params_obj):
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


# ---------------------------------------------------------------------------
# First-stage adapter: bridges old tuple API with new FirstStageFit dataclass
# ---------------------------------------------------------------------------
@dataclass
class _FirstStageResult:
    """Thin wrapper that exposes both FirstStageFit dataclass and legacy attributes."""

    fs: "FirstStageFit"

    @property
    def fitted(self) -> Any:
        return self.fs.fitted

    @property
    def residuals(self) -> np.ndarray:
        return self.fs.residuals

    @property
    def design(self) -> np.ndarray:
        return self.fs.design


class _ReducedFormFirstStageStub:
    """
    Фиктивный объект первой стадии для совместимости с
    build_model_artifact() в режиме reduced_form.
    .fitted.fittedvalues возвращает массив нулей длиной n.
    """

    def __init__(self, n: int):
        self._n = int(n)
        self.fittedvalues = np.zeros(self._n, dtype=float)
        self.resid = np.zeros(self._n, dtype=float)
        self.params = {}
        # build_model_artifact обращается к fit["fitted_fs"].fitted.fittedvalues
        self.fitted = self

    @property
    def nobs(self) -> int:
        return self._n


def fit_reduced_form_pipeline(
    cfg: TrainingConfig,
    dgp: DGPParameters,
    data: pd.DataFrame,
    data_mod: pd.DataFrame,
    baseline_h: float,
    calibrated_censoring_scale: float,
    baseline_diag: Dict[str, Any],
    transform_info: Dict[str, Any],
    achieved_event_rate: float,
    peak_stats: Dict[str, float],
    training_meta_flat: Dict[str, float],
) -> Dict[str, Any]:
    """
    Reduced Form pipeline: без первой стадии, без v_hat, без бутстрапа.
    Возвращает словарь с теми же ключами, что и fit_first_stage_and_cf().
    """
    print()
    print("=" * 70)
    print("РЕЖИМ: REDUCED FORM (предиктивная модель)")
    print("=" * 70)
    print("Инструмент Z включается в Кокс как обычная ковариата.")
    print("Первая стадия, контрольная функция и бутстрап не используются.")
    print("Каузальная интерпрет γ невозможна (нарушение exclusion restriction).")
    print("=" * 70)

    # Импорт новой функции из Итог.py
    try:
        from Итог import fit_reduced_form_cox  # type: ignore[attr-defined]
    except ImportError as exc:
        raise RuntimeError(
            "fit_reduced_form_cox не найдена в Итог.py. Примените Патч C.1."
        ) from exc

    opts = CFFitOptions(
        cox_se_threshold=10.0,
        v_hat_basis="none",
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
        cluster_col="cluster_id",
        n_bootstrap=0,
    )

    cf = fit_reduced_form_cox(data=data_mod, opts=opts)
    for warning in getattr(cf, "warnings", []) or []:
        logger.warning("[RF Cox WARNING] %s", warning)

    cph_obj = getattr(cf, "cph", None)
    if cph_obj is None:
        raise RuntimeError("fit_reduced_form_cox did not return Cox model")

    params_obj = getattr(cph_obj, "params_", None)
    cox_names = [str(name) for name in _safe_keys(params_obj)]
    cox_coefs = _safe_params_to_float_dict(params_obj)
    cox_ses = _safe_params_to_float_dict(getattr(cph_obj, "standard_errors_", None))
    validate_finite_mapping("cox_coefs", cox_coefs)
    validate_finite_mapping("cox_standard_errors", cox_ses)

    # PH diagnostics
    ph_diagnostics = run_ph_diagnostics(cf, data_mod)
    if ph_diagnostics.get("error"):
        logger.warning("RF Cox PH diagnostics failed: %s", ph_diagnostics["error"])

    # Baseline serialization + параметрическая подгонка (Задача 1)
    baseline_hazard_obj = getattr(cph_obj, "baseline_cumulative_hazard_", None)
    baseline_cumulative_hazard = serialize_baseline(baseline_hazard_obj)
    validate_cox_baseline(baseline_cumulative_hazard)

    # ─── Параметрическая подгонка базового риска (Задача 1) ────────────
    baseline_spec_dict: Dict[str, Any] = {"family": "breslow"}
    parametric_family = str(getattr(cfg, "parametric_baseline_fit", "weibull")).lower()
    if HAS_PARAMETRIC_BASELINE and parametric_family in VALID_PARAMETRIC_FAMILIES:
        try:
            spec_obj = fit_parametric_baseline(
                breslow_times=np.asarray(baseline_cumulative_hazard["times"], dtype=float),
                breslow_values=np.asarray(baseline_cumulative_hazard["values"], dtype=float),
                family=parametric_family,
            )
            baseline_spec_dict = spec_obj.to_dict()
            logger.info(
                "✅ [RF] Параметрический базовый риск подогнан: family=%s, R²(log)=%.4f",
                baseline_spec_dict["family"],
                baseline_spec_dict.get("fit_r2", float("nan")),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RF] Параметрическая подгонка не удалась (%s).", exc)
            baseline_spec_dict = {"family": "breslow"}

    # Template covariates
    template_dict = build_raw_template_covariates(data)

    # CF metadata (пустая для reduced form)
    cf_basis_metadata = cf.cf_basis_metadata or {}

    # Фиктивная первая стадия для совместимости с build_model_artifact
    n = len(data_mod)
    fitted_fs_stub = _ReducedFormFirstStageStub(n)

    return {
        "fitted_fs": fitted_fs_stub,
        "resid": np.zeros(n, dtype=float),
        "fs_names": [],
        "fs_params": {},
        "iv_diagnostics": {
            "f_statistic": None,
            "f_statistic_weak": None,
            "cragg_donald_stat": None,
            "cragg_donald_weak": None,
            "endogenous": None,
            "instrument_adequate": False,
            "model_form": "reduced_form",
            "note": "No first stage in reduced-form model.",
        },
        "iv_mode": IV_MODE_PREDICTIVE,
        "weak_instrument_fixed": False,
        "cf": cf,
        "cox_names": cox_names,
        "cox_coefs": cox_coefs,
        "cox_ses": cox_ses,
        "ph_diagnostics": ph_diagnostics,
        "baseline_cumulative_hazard": baseline_cumulative_hazard,
        "baseline_spec": baseline_spec_dict,
        "template_dict": template_dict,
        "partial_out_all_betas": {},
        "training_x_means": {},
        "training_pl_hat_mean": 0.0,
        "partial_out_X_beta": 0.0,
        "training_x_mean": 0.0,
        "cf_basis_metadata": cf_basis_metadata,
        "data": data,
        "data_mod": data_mod,
        "transform_info": transform_info,
        "achieved_event_rate": achieved_event_rate,
        "peak_stats": peak_stats,
        "training_meta_flat": training_meta_flat,
        "baseline_h": baseline_h,
        "calibrated_censoring_scale": calibrated_censoring_scale,
        "baseline_diag": baseline_diag,
        "interaction_lr": None,
        "interaction_lr_test": None,
        "bootstrap_se": None,
    }


_DEFAULT_CF_OPTIONS = CFFitOptions(
    cox_se_threshold=10.0,
    v_hat_basis="linear",
    v_hat_basis_params={"n_knots": 2},
    extra_x_cols=None,
    center_peakload=None,
    brand_encoding="dummies",
    brand_reference_code=0,
    var_z_threshold=1e-8,
    min_first_stage_f=10.0,
    fail_on_weak_instrument=True,
    min_cox_events=10,
    min_events_per_covariate=5,
    save_tracebacks=True,
    cluster_col="cluster_id",
    # ★ FIX 2.3: Bootstrap SE включён по умолчанию
    # PATCH-11: 200 итераций — минимально допустимое для
    # относительной ошибки SE ≈ 1/√(2·200) ≈ 5%.
    n_bootstrap=200,  # БЫЛО: 0
)


def _fit_first_stage_adapter(data: pd.DataFrame) -> Any:
    """Adapter that translates old API to new FirstStageFit API."""
    fs = _fit_first_stage_original(data, _DEFAULT_CF_OPTIONS)
    return _FirstStageResult(fs)


# ---------------------------------------------------------------------------
# Logging / environment helpers
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_dependency_versions() -> Dict[str, str]:
    return {
        "numpy": getattr(np, "__version__", "unknown"),
        "pandas": getattr(pd, "__version__", "unknown"),
        "statsmodels": getattr(sm, "__version__", "unknown"),
        "python": sys.version.split()[0],
    }


def validate_constants() -> None:
    brand_sum = float(sum(DEFAULT_BRAND_PROB_BY_CODE.values()))
    if abs(brand_sum - 1.0) > 1e-6:
        logger.warning(
            "DEFAULT_BRAND_PROB_BY_CODE сумма вероятностей = %.6f, ожидается 1.0",
            brand_sum,
        )

    if not (0.0 <= MAJOR_FAILURE_SHARE <= 1.0):
        logger.warning("MAJOR_FAILURE_SHARE вне диапазона [0, 1]")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_event_definition(value: str) -> str:
    """P-03: проверка определения события."""
    v = str(value).strip().lower()
    if v not in EVENT_DEFINITIONS:
        raise ValueError(
            f"event_definition должно быть одним из {EVENT_DEFINITIONS}, получено: {value!r}"
        )
    return v


def sanity_check_target_probability(
    target_probability: float,
    event_definition: str,
) -> None:
    """P-03: правдоподобность целевой вероятности по определению события."""
    p = float(target_probability)
    ed = str(event_definition).lower()

    if ed == "any_failure" and p < 0.20:
        print(
            "⚠️  target_probability < 20% при event_definition='any_failure' "
            "выглядит заниженным (литература: λ≈0.002/мч)."
        )

    if ed == "total_loss" and p > 0.10:
        print("⚠️  target_probability > 10% при event_definition='total_loss' выглядит завышенным.")

    if ed == "major_claim" and not (0.02 <= p <= 0.35):
        print(
            f"⚠️  target_probability={p:.3f} вне типового диапазона "
            "[0.02, 0.35] для 'major_claim'. Проверьте определение события."
        )


def validate_probability(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name}: должно быть конечным числом")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name}: должно быть в диапазоне [0, 1]")


def validate_open_probability(value: float, name: str) -> None:
    validate_probability(value, name)
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name}: должно быть в открытом диапазоне (0, 1)")


def validate_positive(value: float, name: str) -> None:
    if value is None:
        raise ValueError(f"{name}: значение отсутствует")
    if not np.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name}: должно быть конечным положительным числом")


def validate_finite(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name}: должно быть конечным числом (не nan/inf)")


def validate_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("Сгенерированный набор данных пуст")

    missing = df.isna().sum()
    if missing.any():
        raise RuntimeError(
            f"Обнаружены отсутствующие значения (NaN): {missing[missing > 0].to_dict()}"
        )

    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty and not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("Обнаружены бесконечные значения (inf)")


def validate_finite_mapping(name: str, mapping: Dict[str, Any]) -> None:
    for key, value in mapping.items():
        try:
            value_float = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise RuntimeError(f"{name}: ключ {key} имеет нечисловое значение") from exc
        if not math.isfinite(value_float):
            raise RuntimeError(f"{name}: ключ {key} содержит non-finite значение")


# ---------------------------------------------------------------------------
# Interactive helpers (EOF-safe)
# ---------------------------------------------------------------------------
def _safe_input(prompt: str) -> str:
    """
    Обёртка над input(), корректно обрабатывающая EOF.
    Бросает EOFError только если default невозможен (обрабатывается выше).
    """
    return input(prompt)


def ask(prompt: str, default: Optional[str] = None) -> str:
    try:
        if default is None:
            return _safe_input(f"{prompt}: ").strip()
        value = _safe_input(f"{prompt} [{default}]: ").strip()
        return value if value else default
    except EOFError:
        if default is not None:
            logger.info(
                "EOF при вводе '%s': использую значение по умолчанию '%s'",
                prompt,
                default,
            )
            return default
        raise


def ask_float(
    prompt: str,
    default: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    while True:
        try:
            raw = ask(prompt, str(default))
        except EOFError:
            logger.info("EOF при вводе '%s': использую %s", prompt, default)
            return float(default)

        try:
            value = float(raw)
        except (TypeError, ValueError):
            print("Введите число.")
            continue

        if not math.isfinite(value):
            print("Значение должно быть конечным числом (не nan/inf).")
            continue

        if min_value is not None and value < min_value:
            print(f"Значение должно быть >= {min_value}.")
            continue

        if max_value is not None and value > max_value:
            print(f"Значение должно быть <= {max_value}.")
            continue

        return value


def ask_int(prompt: str, default: int) -> int:
    while True:
        try:
            raw = ask(prompt, str(default))
        except EOFError:
            logger.info("EOF при вводе '%s': использую %s", prompt, default)
            return int(default)

        try:
            return int(raw)
        except ValueError:
            print("Введите целое число.")
            continue


def ask_nonnegative_int(prompt: str, default: int) -> int:
    while True:
        value = ask_int(prompt, default)
        if value >= 0:
            return value
        print("Значение должно быть >= 0.")


def ask_yesno(prompt: str, default: bool) -> bool:
    default_text = "да" if default else "нет"
    while True:
        try:
            value = ask(prompt + " (да/нет)", default_text).lower()
        except EOFError:
            logger.info("EOF при вводе '%s': использую %s", prompt, default)
            return default

        if value in ("да", "yes", "y", "д"):
            return True
        if value in ("нет", "no", "n"):
            return False
        print("Введите да или нет.")


def ask_open_probability(prompt: str, default: float) -> float:
    while True:
        value = ask_float(prompt, default, min_value=0.0, max_value=1.0)
        if 0.0 < value < 1.0:
            return value
        print("Значение должно быть строго в интервале (0, 1).")


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------
def make_rng(global_seed: int, stream_id: int) -> np.random.Generator:
    if int(global_seed) < 0:
        raise ValueError("global_seed должен быть >= 0")
    seq = np.random.SeedSequence([int(global_seed), int(stream_id)])
    return np.random.default_rng(seq)


# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------
def _call_with_supported_kwargs(
    func: Any,
    required_kwargs: Dict[str, Any],
    optional_kwargs: Dict[str, Any],
) -> Any:
    if not callable(func):
        raise TypeError("Переданный объект не является вызываемым.")

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**required_kwargs)

    kwargs = dict(required_kwargs)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    for key, value in optional_kwargs.items():
        if value is None:
            continue
        if has_var_keyword or key in sig.parameters:
            kwargs[key] = value

    return func(**kwargs)


def generate_data_compat(
    *,
    n: int,
    contamination: bool,
    baseline_hazard: float,
    censoring_scale: float,
    rng: np.random.Generator,
    dgp: Optional[DGPParameters] = None,
    instrument_strength: Optional[float] = None,
    instrument_source: str = "normal",
    contamination_probability: float = 1.0,
) -> pd.DataFrame:
    required_kwargs = {
        "n": n,
        "contamination": contamination,
        "baseline_hazard": baseline_hazard,
        "censoring_scale": censoring_scale,
        "rng": rng,
    }
    optional_kwargs = {
        "dgp": dgp,
        "instrument_strength": instrument_strength,
        "instrument_source": instrument_source,
        "contamination_probability": contamination_probability,
    }
    return _call_with_supported_kwargs(
        generate_data,
        required_kwargs,
        optional_kwargs,
    )


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def _describe_object_fields(obj: Any) -> str:
    try:
        if is_dataclass(obj) and not isinstance(obj, type):
            return str(_dataclass_to_dict(obj))
        if isinstance(obj, dict):
            return str(obj)

        attrs: Dict[str, Any] = {}
        for key in dir(obj):
            if key.startswith("_"):
                continue
            try:
                value = getattr(obj, key)
            except AttributeError:
                continue
            if callable(value):
                continue
            attrs[key] = value
        return str(attrs)
    except (AttributeError, TypeError, ValueError):
        return repr(obj)


def _extract_scalar_calibration_result(obj: Any, name: str) -> float:
    """
    Извлекает float из результата калибровки.
    """
    if obj is None:
        raise ValueError(f"{name}: результат отсутствует")

    try:
        x = float(obj)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass

    if isinstance(obj, (tuple, list)):
        for item in obj:
            if _is_finite_number(item):
                return float(item)
        for item in obj:
            try:
                return _extract_scalar_calibration_result(item, name)
            except (TypeError, ValueError, AttributeError, RuntimeError):
                continue

    preferred_names = (
        "censoring_scale",
        "scale",
        "censoring",
        "value",
        "result",
        "estimate",
        "parameter",
        "x",
        "theta",
        "baseline_hazard",
        "hazard",
    )

    for attr in preferred_names:
        if hasattr(obj, attr):
            value = getattr(obj, attr, None)
            if _is_finite_number(value):
                return float(value)

    if isinstance(obj, dict):
        for attr in preferred_names:
            if attr in obj and _is_finite_number(obj[attr]):
                return float(obj[attr])
        else:
            getter = getattr(obj, "get", None)
            if getter is not None and callable(getter):
                for attr in preferred_names:
                    try:
                        value = getter(attr, None)
                    except (TypeError, ValueError, AttributeError):
                        value = None
                    if _is_finite_number(value):
                        return float(value)

    if is_dataclass(obj) and not isinstance(obj, type):
        data = _dataclass_to_dict(obj)
        for attr in preferred_names:
            if attr in data and _is_finite_number(data[attr]):
                return float(data[attr])

        numeric_fields = {k: v for k, v in data.items() if _is_finite_number(v)}
        if len(numeric_fields) == 1:
            return float(next(iter(numeric_fields.values())))
        if numeric_fields:
            raise RuntimeError(
                f"Не удалось однозначно извлечь {name}. "
                f"Числовые поля: {numeric_fields}. "
                "Добавьте нужное имя поля в preferred_names."
            )

    try:
        arr = np.asarray(obj)
        if arr.size == 1:
            value = arr.reshape(-1)[0]
            if _is_finite_number(value):
                return float(value)
    except (TypeError, ValueError):
        pass

    raise RuntimeError(
        f"Не удалось извлечь float из {type(obj).__name__} для {name}. "
        f"Содержимое: {_describe_object_fields(obj)}"
    )


def calibrate_censoring_scale_compat(
    *,
    target_event_rate: float,
    n_calib: int,
    rng: np.random.Generator,
    dgp: Optional[DGPParameters] = None,
    baseline_hazard: float,
    instrument_strength: Optional[float] = None,
    instrument_source: str = "normal",
    contamination: bool = False,
    contamination_probability: float = 1.0,
) -> float:
    effective_tol = min(0.005, max(1e-6, 0.1 * float(target_event_rate)))

    required_kwargs = {
        "target_event_rate": target_event_rate,
        "n_calib": n_calib,
        "rng": rng,
    }
    optional_kwargs = {
        "dgp": dgp,
        "baseline_hazard": baseline_hazard,
        "instrument_strength": instrument_strength,
        "instrument_source": instrument_source,
        "contamination": contamination,
        "contamination_probability": contamination_probability,
        "tol": effective_tol,
    }

    raw_result = _call_with_supported_kwargs(
        calibrate_censoring_scale_deterministic,
        required_kwargs,
        optional_kwargs,
    )
    result = _extract_scalar_calibration_result(
        raw_result,
        "censoring_scale",
    )
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError("censoring_scale должен быть конечным положительным числом")
    return result


def fit_cf_cox_compat(
    *,
    data: pd.DataFrame,
    first_stage: "FirstStageFit",
    v_hat_basis: str,
    v_hat_basis_params: Optional[Dict[str, Any]],
) -> Any:
    """Compatibility wrapper: calls new fit_cf_cox(data, first_stage, opts)."""
    extra_x_cols = getattr(first_stage, "x_cols", None)
    opts = CFFitOptions(
        cox_se_threshold=10.0,
        v_hat_basis=v_hat_basis,
        v_hat_basis_params=v_hat_basis_params,
        extra_x_cols=extra_x_cols,
        center_peakload=None,
        brand_encoding="dummies",
        brand_reference_code=0,
        var_z_threshold=1e-8,
        min_first_stage_f=10.0,
        fail_on_weak_instrument=True,
        min_cox_events=10,
        min_events_per_covariate=5,
        save_tracebacks=True,
        cluster_col="cluster_id",
        n_bootstrap=0,
    )
    return fit_cf_cox(data=data, first_stage=first_stage, opts=opts)


def _log_dropped_kwargs(label: str, dropped: set) -> None:
    if dropped:
        logger.info(
            "%s: пропущены неподдерживаемые поля: %s",
            label,
            sorted(str(x) for x in dropped),
        )


def _construct_object(cls: Any, kwargs: Dict[str, Any], label: str) -> Any:
    if is_dataclass(cls):
        try:
            field_names = {f.name for f in fields(cls)}
            filtered = {k: v for k, v in kwargs.items() if k in field_names}
            _log_dropped_kwargs(label, set(kwargs) - set(filtered))
            return cls(**filtered)
        except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
            logger.warning(
                "%s: filtered dataclass construction failed: %s",
                label,
                exc,
            )

    try:
        return cls(**kwargs)
    except TypeError:
        try:
            sig = inspect.signature(cls)
        except (TypeError, ValueError):
            raise
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        _log_dropped_kwargs(label, set(kwargs) - set(filtered))
        return cls(**filtered)


def construct_dgp(kwargs: Dict[str, Any]) -> DGPParameters:
    return _construct_object(DGPParameters, kwargs, "DGPParameters")


def construct_model_parameters(kwargs: Dict[str, Any]) -> Any:
    return _construct_object(ModelParameters, kwargs, "ModelParameters")


def set_dgp_field(dgp: Any, name: str, value: Any) -> Any:
    """
    Обновить поле DGP безопасно.
    Работает и с mutable dataclass, и с frozen dataclass через пересоздание.
    """
    if is_dataclass(dgp) and not isinstance(dgp, type):
        field_names = {f.name for f in fields(dgp)}
        if name not in field_names:
            logger.warning(
                "DGP field '%s' не найдено в полях DGPParameters. "
                "Изменение может быть проигнорировано.",
                name,
            )

        try:
            setattr(dgp, name, value)
            return dgp
        except (AttributeError, TypeError, ValueError) as exc:
            if is_dataclass(dgp) and not isinstance(dgp, type):
                try:
                    data = {f.name: getattr(dgp, f.name, None) for f in fields(dgp)}
                    data[name] = value
                    return type(dgp)(**data)
                except (TypeError, ValueError) as exc2:
                    raise RuntimeError(f"Не удалось обновить DGP поле '{name}'") from exc2
            raise RuntimeError(f"Не удалось обновить DGP поле '{name}'") from exc

    raise RuntimeError(f"Не удалось обновить DGP поле '{name}'")


def _to_dict_safe(value: Any) -> Dict[str, Any]:
    """Безопасное приведение к Dict[str, Any]."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    try:
        return {str(k): v for k, v in dict(value).items()}
    except (TypeError, ValueError, AttributeError):
        return {}


def _mapping_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# Baseline initial helpers
# ---------------------------------------------------------------------------
def compute_initial_baseline(
    target_probability: float,
    time_horizon: float,
    family: str,
    shape: Optional[float],
) -> float:
    validate_open_probability(target_probability, "target_probability")
    validate_positive(time_horizon, "time_horizon")

    neg_log = -math.log1p(-float(target_probability))
    fam = str(family).lower()

    if fam == "exponential":
        return float(neg_log / time_horizon)

    if fam == "weibull":
        if shape is None:
            k = DEFAULT_WEIBULL_SHAPE
        else:
            k = float(shape)
        if not math.isfinite(k) or k <= 0.0:
            k = DEFAULT_WEIBULL_SHAPE
        # Weibull CDF: F(t) = 1 - exp(-lambda * t^k)
        # For target P: lambda = -ln(1-P) / t^k
        try:
            t_pow_k = float(time_horizon) ** k
            if t_pow_k <= 0.0 or not math.isfinite(t_pow_k):
                raise OverflowError
            return float(neg_log / t_pow_k)
        except OverflowError:
            logger.warning("Weibull initial baseline overflow, fallback к exponential-like initial")
            return float(neg_log / time_horizon)

    if fam == "gompertz":
        if shape is None:
            b = DEFAULT_GOMPERTZ_RATE
        else:
            b = float(shape)
        if not math.isfinite(b) or b <= 0.0:
            b = DEFAULT_GOMPERTZ_RATE

        x = b * float(time_horizon)
        if x > MAX_GOMPERTZ_EXP_ARGUMENT:
            logger.warning("Gompertz initial: b*time_horizon слишком велик, fallback")
            return float(neg_log / time_horizon)

        denom = math.expm1(x)
        if not math.isfinite(denom) or denom <= 0.0:
            return float(neg_log / time_horizon)
        return float(b * neg_log / denom)

    return float(neg_log / time_horizon)


# ---------------------------------------------------------------------------
# Baseline calibration helpers
# ---------------------------------------------------------------------------
def _simulate_failure_probability(
    baseline_hazard: float,
    time_horizon: float,
    dgp: DGPParameters,
    rng: np.random.Generator,
    n_sim: int = 5000,
    censoring_scale: float = 1e12,
    contamination: bool = False,
    contamination_probability: float = 1.0,
    instrument_strength: Optional[float] = None,
) -> float:
    df = generate_data_compat(
        n=n_sim,
        contamination=contamination,
        baseline_hazard=float(baseline_hazard),
        censoring_scale=float(censoring_scale),
        rng=rng,
        dgp=dgp,
        contamination_probability=contamination_probability,
        instrument_strength=instrument_strength,
    )
    validate_dataframe(df)
    col = "T_true" if "T_true" in df.columns else "time"
    return float((df[col] <= float(time_horizon)).mean())


def calibrate_baseline_by_target_probability(
    *,
    target_probability: float,
    time_horizon: float,
    dgp: DGPParameters,
    global_seed: int,
    n_calib: int = 5000,
    tolerance_abs: float = CALIBRATION_TOLERANCE_ABS,
    max_iter: int = 40,
    initial_hazard: Optional[float] = None,
    contamination: bool = False,
    contamination_probability: float = 1.0,
    instrument_strength: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    validate_open_probability(target_probability, "target_probability")
    validate_positive(time_horizon, "time_horizon")

    family = str(getattr(dgp, "baseline_family", "weibull")).lower()
    shape = getattr(dgp, "baseline_shape", None)

    if initial_hazard is None:
        initial_hazard = compute_initial_baseline(
            target_probability=target_probability,
            time_horizon=time_horizon,
            family=family,
            shape=shape,
        )

    initial_hazard_f = float(initial_hazard)
    validate_positive(initial_hazard_f, "initial_hazard")

    search_lo = MIN_BASELINE_SEARCH
    search_hi = min(
        max(0.1, initial_hazard_f * 10.0),
        MAX_BASELINE_SEARCH,
    )

    def sim(hazard: float, stream: int) -> float:
        return _simulate_failure_probability(
            baseline_hazard=float(hazard),
            time_horizon=time_horizon,
            dgp=dgp,
            rng=make_rng(global_seed, stream),
            n_sim=n_calib,
            contamination=contamination,
            contamination_probability=contamination_probability,
            instrument_strength=instrument_strength,
        )

    p_lo = sim(search_lo, 9000)
    p_hi = sim(search_hi, 9000)

    max_expansion = 20
    expansion = 0
    while not (min(p_lo, p_hi) <= target_probability <= max(p_lo, p_hi)):
        if expansion >= max_expansion:
            raise RuntimeError(
                f"Cannot bracket target probability {target_probability:.3f} "
                f"at time {time_horizon}. p_lo={p_lo:.4f}, p_hi={p_hi:.4f}."
            )

        if target_probability > max(p_lo, p_hi):
            search_hi = min(search_hi * 3.0, MAX_BASELINE_SEARCH)
            p_hi = sim(search_hi, 9000)
        else:
            search_lo = max(search_lo / 3.0, MIN_BASELINE_SEARCH)
            p_lo = sim(search_lo, 9000)
        expansion += 1

    best_h = initial_hazard_f
    best_error = float("inf")
    best_achieved = float("nan")
    actual_iterations = 0

    for i in range(max_iter):
        actual_iterations = i + 1
        mid = (search_lo + search_hi) / 2.0

        # Common random numbers: один и тот же stream для всех итераций.
        achieved = sim(mid, 9500)
        error = abs(achieved - target_probability)

        if error < best_error:
            best_h = mid
            best_error = error
            best_achieved = achieved

        if error <= tolerance_abs:
            break

        if achieved < target_probability:
            search_lo = mid
        else:
            search_hi = mid

    val_prob = sim(best_h, 9999)

    diagnostics = {
        "method": "target_probability",
        "target_probability": float(target_probability),
        "time_horizon": float(time_horizon),
        "time_unit": MODEL_TIME_UNIT,
        "achieved_probability": float(best_achieved)
        if np.isfinite(best_achieved)
        else float(val_prob),
        "final_baseline": float(best_h),
        "search_lo": float(search_lo),
        "search_hi": float(search_hi),
        "iterations": actual_iterations,
        "validation_achieved_probability": float(val_prob),
        "final_error": abs(val_prob - target_probability),
    }
    return float(best_h), diagnostics


def calibrate_baseline_by_quantile(
    *,
    target_time: float,
    target_quantile: float,
    dgp: DGPParameters,
    global_seed: int,
    n_calib: int = 3000,
    contamination: bool = False,
    contamination_probability: float = 1.0,
    instrument_strength: Optional[float] = None,
    tol_rel: float = 0.05,
    max_iter: int = 30,
) -> Tuple[float, Dict[str, Any]]:
    validate_positive(target_time, "target_time")
    validate_open_probability(target_quantile, "target_quantile")

    def simulate_quantile(hazard: float) -> float:
        # Common random numbers для всех hazard-кандидатов.
        rng = make_rng(global_seed, 1000)
        df = generate_data_compat(
            n=n_calib,
            contamination=contamination,
            baseline_hazard=float(hazard),
            censoring_scale=1e12,
            rng=rng,
            dgp=dgp,
            contamination_probability=contamination_probability,
            instrument_strength=instrument_strength,
        )
        validate_dataframe(df)
        col = "T_true" if "T_true" in df.columns else "time"
        q = float(df[col].quantile(target_quantile))
        if not math.isfinite(q):
            raise RuntimeError("simulate_quantile produced non-finite quantile")
        return q

    lo = MIN_BASELINE_SEARCH
    hi = 0.1

    q_lo = simulate_quantile(lo)
    q_hi = simulate_quantile(hi)

    expansion = 0
    max_expansion = 50
    while not (min(q_lo, q_hi) <= target_time <= max(q_lo, q_hi)):
        if expansion >= max_expansion:
            raise RuntimeError("Cannot bracket target quantile")

        if target_time < min(q_lo, q_hi):
            hi = min(hi * 2.0, MAX_BASELINE_SEARCH)
            q_hi = simulate_quantile(hi)
        else:
            lo = max(lo / 2.0, MIN_BASELINE_SEARCH)
            q_lo = simulate_quantile(lo)
        expansion += 1

    best_h = (lo + hi) / 2.0
    best_q = float("nan")
    best_error = float("inf")
    iterations = 0

    for _ in range(max_iter):
        iterations += 1
        mid = (lo + hi) / 2.0
        q_mid = simulate_quantile(mid)
        error = abs(q_mid - target_time) / max(abs(target_time), 1e-12)

        if error < best_error:
            best_h = mid
            best_q = q_mid
            best_error = error

        if error <= tol_rel:
            break

        if q_mid > target_time:
            lo = mid
        else:
            hi = mid

    diagnostics = {
        "method": "target_quantile",
        "iterations": iterations,
        "target_time": float(target_time),
        "time_unit": MODEL_TIME_UNIT,
        "target_quantile": float(target_quantile),
        "final_baseline": float(best_h),
        "obtained_quantile": float(best_q),
        "lower": float(lo),
        "upper": float(hi),
    }
    return float(best_h), diagnostics


# ---------------------------------------------------------------------------
# Bootstrap SE for generated regressors (Issue #4)
# ---------------------------------------------------------------------------
def _bootstrap_cox_se(
    *,
    data_mod: pd.DataFrame,
    v_hat_basis: str,
    v_hat_basis_params: Optional[Dict[str, Any]],
    cox_names: List[str],
    n_bootstrap: int,
    seed: int,
    n_jobs: int = 1,
) -> Dict[str, float]:
    """
    Compute bootstrap SE for CF Cox coefficients.

    This accounts for first-stage uncertainty in the generated regressor
    (v_hat). Each bootstrap iteration:
    1. Resample observations with replacement (cluster-aware if cluster_id exists)
    2. Refit first stage
    3. Refit CF Cox
    4. Collect coefficients

    Returns dict mapping coefficient name -> bootstrap SE.
    """
    from joblib import Parallel, delayed

    n = len(data_mod)
    has_clusters = "cluster_id" in data_mod.columns

    if has_clusters:
        clusters = data_mod["cluster_id"].unique()
        n_clusters = len(clusters)
        logger.info(
            "Cluster bootstrap: %d clusters, %d observations, n_jobs=%d",
            n_clusters,
            n,
            n_jobs,
        )

    def _single_bootstrap_iteration(b: int):
        """Одна итерация бутстрапа. Возвращает (dict_coefs, error_or_None)."""
        # ★ FIX: УНИКАЛЬНЫЙ seed для каждой итерации
        rng = make_rng(seed, 88888 + b)
        try:
            if has_clusters:
                boot_clusters = rng.choice(clusters, size=n_clusters, replace=True)
                n_unique_in_boot = len(np.unique(boot_clusters))
                if n_unique_in_boot < n_clusters * 0.5:
                    logger.debug(
                        "Bootstrap iter %d: только %d/%d уникальных кластеров",
                        b,
                        n_unique_in_boot,
                        n_clusters,
                    )
                boot_indices = []
                for bc in boot_clusters:
                    mask = data_mod["cluster_id"] == bc
                    boot_indices.extend(mask[mask].index.tolist())
            else:
                boot_indices = rng.choice(n, size=n, replace=True).tolist()

            boot_data = data_mod.iloc[boot_indices].copy()
            fs_result = _fit_first_stage_adapter(boot_data)
            cf_boot = fit_cf_cox_compat(
                data=boot_data,
                first_stage=fs_result.fs,
                v_hat_basis=v_hat_basis,
                v_hat_basis_params=v_hat_basis_params,
            )

            cph_boot = getattr(cf_boot, "cph", None)
            if cph_boot is None:
                return None, "no_cph"
            params_boot = getattr(cph_boot, "params_", None)
            if params_boot is None:
                return None, "no_params"

            coefs = {}
            for name in cox_names:
                val = _safe_series_scalar(params_boot, name, f"bootstrap[{b}]")
                if math.isfinite(val):
                    coefs[name] = val
            return coefs, None

        except Exception as exc:
            return None, str(exc)

    # ─── Параллельное выполнение ─────────────────────────────────
    if n_jobs <= 1:
        results = [_single_bootstrap_iteration(b) for b in range(n_bootstrap)]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=5)(
            delayed(_single_bootstrap_iteration)(b) for b in range(n_bootstrap)
        )

    # ─── Агрегация результатов ───────────────────────────────────
    bootstrap_coefs: Dict[str, List[float]] = {name: [] for name in cox_names}
    n_failures = 0

    for coefs, error in results:
        if error is not None or coefs is None:
            n_failures += 1
            continue
        for name in cox_names:
            if name in coefs:
                bootstrap_coefs[name].append(coefs[name])

    if n_failures > 0:
        logger.warning("Bootstrap: %d / %d итераций не удались", n_failures, n_bootstrap)

    result: Dict[str, float] = {}
    for name, values in bootstrap_coefs.items():
        if len(values) < 10:
            logger.warning(
                "Bootstrap SE for '%s': only %d valid iterations (< 10). Using naive SE.",
                name,
                len(values),
            )
            result[name] = float("nan")
            continue
        result[name] = float(np.std(values, ddof=1))
    return result


def run_calibration(
    *,
    calib_method: str,
    dgp: DGPParameters,
    seed: int,
    n: int,
    target_time: float,
    target_quantile: float,
    target_probability: Optional[float],
    target_event_rate: float,
    contamination: bool,
    contamination_probability: float,
    baseline_hazard_initial: Optional[float],
    instrument_strength: Optional[float] = None,
    instrument_source: str = "normal",
) -> Tuple[float, float, Dict[str, Any]]:
    if calib_method == "manual":
        baseline_h = _require_float(
            baseline_hazard_initial,
            "baseline_hazard_initial",
        )
        validate_positive(baseline_h, "baseline_hazard")

        censor = calibrate_censoring_scale_compat(
            target_event_rate=target_event_rate,
            n_calib=max(5000, n * 5),
            rng=make_rng(seed, 999),
            dgp=dgp,
            baseline_hazard=baseline_h,
            instrument_source=instrument_source,
            instrument_strength=instrument_strength,
            contamination=contamination,
            contamination_probability=contamination_probability,
        )

        diagnostics = {
            "method": "manual",
            "baseline_hazard": float(baseline_h),
            "censoring_scale": float(censor),
            "time_unit": MODEL_TIME_UNIT,
        }
        return baseline_h, censor, diagnostics

    if calib_method == "probability":
        target_probability_value = _require_float(
            target_probability,
            "target_probability",
        )

        initial_hazard: Optional[float] = None
        if baseline_hazard_initial is not None:
            initial_hazard = _require_float(
                baseline_hazard_initial,
                "baseline_hazard_initial",
            )

        baseline_h, baseline_diag = calibrate_baseline_by_target_probability(
            target_probability=target_probability_value,
            time_horizon=target_time,
            dgp=dgp,
            global_seed=seed,
            n_calib=max(5000, n),
            tolerance_abs=CALIBRATION_TOLERANCE_ABS,
            max_iter=30,
            initial_hazard=initial_hazard,
            contamination=contamination,
            contamination_probability=contamination_probability,
            instrument_strength=instrument_strength,
        )

    elif calib_method == "quantile":
        baseline_h, baseline_diag = calibrate_baseline_by_quantile(
            target_time=target_time,
            target_quantile=target_quantile,
            dgp=dgp,
            global_seed=seed,
            n_calib=max(5000, n),
            contamination=contamination,
            contamination_probability=contamination_probability,
            instrument_strength=instrument_strength,
        )

    else:
        raise ValueError(f"Unknown calibration method: {calib_method}")

    censor = calibrate_censoring_scale_compat(
        target_event_rate=target_event_rate,
        n_calib=max(5000, n * 5),
        rng=make_rng(seed, 999),
        dgp=dgp,
        baseline_hazard=baseline_h,
        instrument_source=instrument_source,
        instrument_strength=instrument_strength,
        contamination=contamination,
        contamination_probability=contamination_probability,
    )

    diagnostics = dict(baseline_diag)
    diagnostics["censoring_scale"] = float(censor)
    return float(baseline_h), float(censor), diagnostics


# ---------------------------------------------------------------------------
# Data generation and transformation
# ---------------------------------------------------------------------------
def generate_and_transform_data(
    *,
    n: int,
    contamination: bool,
    contamination_probability: float,
    baseline_hazard: float,
    censoring_scale: float,
    seed: int,
    dgp: DGPParameters,
    do_standardize: bool,
    do_center_only: bool,
    instrument_strength: Optional[float] = None,
    instrument_source: str = "normal",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    data = generate_data_compat(
        n=n,
        contamination=contamination,
        baseline_hazard=float(baseline_hazard),
        censoring_scale=float(censoring_scale),
        rng=make_rng(seed, 424242),
        dgp=dgp,
        contamination_probability=contamination_probability,
        instrument_strength=instrument_strength,
        instrument_source=instrument_source,
    )
    validate_dataframe(data)

    data_mod = data.copy()
    transform_info: Dict[str, Any] = {"type": "none"}

    if do_standardize:
        center = float(data_mod["PeakLoad"].mean())
        scale = float(data_mod["PeakLoad"].std(ddof=1))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        data_mod["PeakLoad"] = (data_mod["PeakLoad"] - center) / scale
        transform_info = {
            "type": "standardize",
            "center": center,
            "scale": scale,
            "ddof": 1,
        }
    elif do_center_only:
        center = float(data_mod["PeakLoad"].mean())
        data_mod["PeakLoad"] = data_mod["PeakLoad"] - center
        transform_info = {
            "type": "center",
            "center": center,
        }

    validate_dataframe(data_mod)
    return data, data_mod, transform_info


def summarize_data(
    data: pd.DataFrame,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    achieved_event_rate = float(data["event"].mean())

    peak_stats = {
        "min": float(data["PeakLoad"].min()),
        "max": float(data["PeakLoad"].max()),
        "p25": float(data["PeakLoad"].quantile(0.25)),
        "median": float(data["PeakLoad"].quantile(0.50)),
        "p75": float(data["PeakLoad"].quantile(0.75)),
    }

    training_meta_flat = {
        "peakload_min": peak_stats["min"],
        "peakload_max": peak_stats["max"],
        "peakload_p25": peak_stats["p25"],
        "peakload_median": peak_stats["median"],
        "peakload_p75": peak_stats["p75"],
    }

    return achieved_event_rate, peak_stats, training_meta_flat


# ---------------------------------------------------------------------------
# Clustered covariate generation (weather/soil real data)
# ---------------------------------------------------------------------------
def load_real_covariates_clustered_v2(
    n: int,
    rng: np.random.Generator,
    campaign: str = "sowing",
) -> Dict[str, Any]:
    """
    Кластерная генерация v2:
    - Z идентичен внутри кластера (без jitter)
    - Индивидуальная вариация через X и ε_D
    - Возвращает cluster_id для каждого трактора
    """
    result: Dict[str, Any] = {}

    # Загрузка кластеров (Region × Year × Campaign)
    rain_path = Path("data/processed/weather/rainfall_anomaly.csv")
    weather_path = Path("data/processed/weather/weather_windows.csv")
    soil_path = Path("data/processed/soil/soil_windows.csv")

    clusters: list = []  # list of dicts

    if rain_path.exists() and weather_path.exists() and soil_path.exists():
        rdf = pd.read_csv(rain_path)
        wdf = pd.read_csv(weather_path)
        sdf = pd.read_csv(soil_path)

        rdf = rdf[rdf["campaign"] == campaign].copy()
        wdf = wdf[wdf["campaign"] == campaign].copy()
        sdf = sdf[sdf["campaign"] == campaign].copy()

        # Нормализация x_climate
        wd_vals = pd.to_numeric(wdf["working_days_window"], errors="coerce")
        wd_min, wd_max = float(wd_vals.min()), float(wd_vals.max())

        # Нормализация x_soil
        soil_col = (
            "soil_index_normalized" if "soil_index_normalized" in sdf.columns else "soil_index"
        )
        soil_vals_all = pd.to_numeric(sdf[soil_col], errors="coerce")
        s_min: float = 0.0
        s_max: float = 0.0
        if soil_col == "soil_index":
            s_min, s_max = (
                float(soil_vals_all.min()),
                float(soil_vals_all.max()),
            )

        # Собираем кластеры
        for _, rrow in rdf.iterrows():
            region = rrow["region_code"]
            year = int(rrow["year"])

            # Z = rainfall anomaly (БЕЗ jitter!)
            z_val = float(rrow["rainfall_anomaly"])

            # x_climate для этого кластера
            w_match = wdf[(wdf["region_code"] == region) & (wdf["year"] == year)]
            if len(w_match) == 0:
                continue
            wd = float(pd.to_numeric(w_match["working_days_window"].iloc[0], errors="coerce"))
            x_clim = (wd - wd_min) / (wd_max - wd_min) if wd_max > wd_min else 0.5

            # x_soil для этого кластера
            s_match = sdf[(sdf["region_code"] == region) & (sdf["year"] == year)]
            if len(s_match) == 0:
                continue
            sv = float(pd.to_numeric(s_match[soil_col].iloc[0], errors="coerce"))
            if soil_col == "soil_index":
                x_soil_val = (sv - s_min) / (s_max - s_min) if s_max > s_min else 0.5
            else:
                x_soil_val = sv

            clusters.append(
                {
                    "cluster_id": f"{region}_{year}",
                    "Z": z_val,
                    "x_climate": x_clim,
                    "x_soil": x_soil_val,
                }
            )

    n_clusters = len(clusters)

    if n_clusters < 4:
        logger.warning(f"⚠️ Только {n_clusters} кластеров. Fallback к синтетическим.")
        result["Z"] = rng.normal(0, 1, size=n)
        result["x_climate"] = rng.beta(2.5, 1.5, size=n)
        result["x_soil"] = rng.beta(2.0, 2.5, size=n)
        result["cluster_id"] = np.arange(n)  # каждый трактор — свой кластер
        return result

    logger.info(
        f"✅ Кластерная генерация: {n_clusters} кластеров, ~{n // n_clusters} тракторов/кластер"
    )

    # Распределение тракторов по кластерам
    # Каждый кластер получает примерно равное количество тракторов
    tractors_per_cluster = n // n_clusters
    remainder = n % n_clusters

    cluster_assign = []
    for i, cluster in enumerate(clusters):
        n_tractors = tractors_per_cluster + (1 if i < remainder else 0)
        cluster_assign.extend([i] * n_tractors)

    cluster_assign = np.array(cluster_assign)
    rng.shuffle(cluster_assign)  # перемешиваем для случайности

    # Присваиваем Z, x_climate, x_soil (БЕЗ jitter для Z!)
    z_arr = np.array([clusters[c]["Z"] for c in cluster_assign])
    x_clim_arr = np.array([clusters[c]["x_climate"] for c in cluster_assign])
    x_soil_arr = np.array([clusters[c]["x_soil"] for c in cluster_assign])
    cluster_ids = np.array([clusters[c]["cluster_id"] for c in cluster_assign])

    # Стандартизация Z (глобальная, не внутри кластера)
    z_mean = float(np.mean(z_arr))
    z_std = float(np.std(z_arr, ddof=1))
    if z_std < 1e-9:
        z_std = 1.0
    result["Z"] = (z_arr - z_mean) / z_std

    # x_climate и x_soil — без jitter (они уже cluster-level)
    result["x_climate"] = x_clim_arr
    result["x_soil"] = x_soil_arr
    result["cluster_id"] = cluster_ids

    logger.info(
        f"Z = rainfall anomaly (cluster-level): mean={z_mean:.2f}, "
        f"std={z_std:.2f}, n_clusters={n_clusters}, n_tractors={n}"
    )

    return result


# ---------------------------------------------------------------------------
# First stage + IV diagnostics
# ---------------------------------------------------------------------------
def compute_iv_diagnostics(
    fitted_fs: Any,
    data_mod: pd.DataFrame,
    resid: np.ndarray,
    first_stage_fit: Optional[Any] = None,
    opts: Optional[CFFitOptions] = None,
) -> Dict[str, Any]:
    """
    IV-диагностика с защитой от None/uncallable функций.
    """
    iv: Dict[str, Any] = {
        "f_statistic": None,
        "f_pvalue": None,
        "f_statistic_weak": None,
        "cragg_donald_stat": None,
        "cragg_donald_weak": None,
        "anderson_rubin_stat": None,
        "anderson_rubin_pvalue": None,
        "endogenous": None,
        "instrument_adequate": None,
    }

    # ---------------------------------------------------------------
    # First-stage F statistic
    # ---------------------------------------------------------------
    f_stat_fn = classical_first_stage_f_statistic
    if callable(f_stat_fn):
        try:
            f_stat, f_pval = f_stat_fn(fitted_fs)
            f_stat_f = _as_float_or_none(f_stat)
            f_pval_f = _as_float_or_none(f_pval)

            iv["f_statistic"] = f_stat_f
            iv["f_pvalue"] = f_pval_f

            if f_stat_f is not None:
                iv["f_statistic_weak"] = bool(f_stat_f < AUTO_INSTRUMENT_THRESHOLD)
            else:
                iv["f_statistic_weak"] = None
        except (TypeError, ValueError, AttributeError, RuntimeError):
            iv["f_statistic_error"] = "classical_first_stage_f_statistic failed"
    else:
        iv["f_statistic_error"] = "classical_first_stage_f_statistic unavailable"

    # ---------------------------------------------------------------
    # Cragg-Donald
    # ---------------------------------------------------------------
    cd_fn = cragg_donald_stat
    if callable(cd_fn):
        try:
            cd_result = cd_fn(fitted_fs)

            cd_stat_raw: Any = None
            cd_weak_raw: Any = None

            if isinstance(cd_result, tuple):
                if len(cd_result) >= 1:
                    cd_stat_raw = cd_result[0]
                if len(cd_result) >= 3:
                    cd_weak_raw = cd_result[2]
            else:
                cd_stat_raw = _mapping_get(cd_result, "cd_stat", None)
                cd_weak_raw = _mapping_get(cd_result, "cd_weak", None)

            cd_stat_f = _as_float_or_none(cd_stat_raw)
            iv["cragg_donald_stat"] = cd_stat_f

            if cd_weak_raw is None:
                if cd_stat_f is not None:
                    iv["cragg_donald_weak"] = bool(cd_stat_f < AUTO_INSTRUMENT_THRESHOLD)
                else:
                    iv["cragg_donald_weak"] = None
            else:
                iv["cragg_donald_weak"] = bool(cd_weak_raw)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            iv["cragg_donald_error"] = "cragg_donald_stat failed"
    else:
        iv["cragg_donald_error"] = "cragg_donald_stat unavailable"

    # ---------------------------------------------------------------
    # Endogeneity LR test
    # ---------------------------------------------------------------
    endo_fn = endogeneity_lr_test
    if callable(endo_fn):
        try:
            if first_stage_fit is not None and opts is not None:
                lr_result = endo_fn(data_mod, first_stage_fit, opts)
            else:
                lr_result = endo_fn(data_mod, fitted_fs, resid)

            lr_stat = _as_float_or_none(_mapping_get(lr_result, "lr_stat", None))
            lr_pval = _as_float_or_none(_mapping_get(lr_result, "lr_pvalue", None))

            iv["anderson_rubin_stat"] = lr_stat
            iv["anderson_rubin_pvalue"] = lr_pval
            iv["endogenous"] = bool(_mapping_get(lr_result, "endogenous", False))
        except (TypeError, ValueError, AttributeError, RuntimeError):
            iv["endogeneity_error"] = "endogeneity_lr_test failed"
    else:
        iv["endogeneity_error"] = "endogeneity_lr_test unavailable"

    f_ok = iv["f_statistic_weak"] is False
    cd_ok = iv["cragg_donald_weak"] is False
    iv["instrument_adequate"] = bool(f_ok and cd_ok)

    return iv


def run_first_stage_and_iv(
    data_mod: pd.DataFrame,
) -> Tuple[Any, np.ndarray, List[str], Dict[str, float], Dict[str, Any]]:
    result = _fit_first_stage_adapter(data_mod)

    fitted_fs = result.fitted
    resid_obj = result.residuals
    if resid_obj is None:
        raise RuntimeError("Residuals are empty")

    model_obj = getattr(fitted_fs, "model", None)
    fs_names = _safe_exog_names(model_obj)
    if fs_names is None:
        params_obj = getattr(fitted_fs, "params", None)
        fs_names = [str(key) for key in _safe_keys(params_obj)]

    fs_params = _safe_params_to_float_dict(getattr(fitted_fs, "params", None))
    validate_finite_mapping("first_stage_params", fs_params)

    resid_arr = np.asarray(resid_obj, dtype=float)
    if resid_arr.size == 0:
        raise RuntimeError("Residuals are empty")
    if not np.all(np.isfinite(resid_arr)):
        raise RuntimeError("Residuals contain non-finite values")

    iv = compute_iv_diagnostics(
        fitted_fs,
        data_mod,
        resid_arr,
        first_stage_fit=result.fs,
        opts=_DEFAULT_CF_OPTIONS,
    )

    return result.fs, resid_arr, fs_names, fs_params, iv


# УДАЛЕНО: strengthen_instrument_coeff больше не используется.
# Автокоррекция слабого инструмента методологически некорректна.


def determine_iv_mode(
    iv_diagnostics: Dict[str, Any],
    weather_instrument_validated: bool = False,
) -> str:
    """
    Фаза 5.4: определить IV-режим.
    causal — если инструмент адекватен (F и Cragg-Donald в норме)
                 И (для production) погодный инструмент подтверждён.
    predictive — иначе.
    """
    instrument_adequate = bool(iv_diagnostics.get("instrument_adequate", False))
    endogenous = bool(iv_diagnostics.get("endogenous", False))

    # Инструмент должен быть и сильным, и должна быть эндогенность,
    # которую он корректирует.
    if instrument_adequate and endogenous and weather_instrument_validated:
        return IV_MODE_CAUSAL

    if instrument_adequate and endogenous:
        # Симуляционный инструмент силён, но production-инструмент
        # ещё не подтверждён реальными данными → conservatively predictive.
        logger.info(
            "Инструмент адекватен на симуляции, но production-инструмент "
            "не подтверждён. Устанавливаю iv_mode='predictive'."
        )
        return IV_MODE_PREDICTIVE

    return IV_MODE_PREDICTIVE


# ---------------------------------------------------------------------------
# Cox diagnostics
# ---------------------------------------------------------------------------
def run_ph_diagnostics(cf: Any, data_mod: pd.DataFrame) -> Dict[str, Any]:
    """
    P0-4: Запуск структурированной PH-диагностики.

    Возвращает структурированный PH-отчёт для сохранения в artifact.
    Не изменяет DGP, first stage или Cox estimation.
    """
    # ─── Импорт новой функции из Итог.py ─────────────────────────────
    try:
        from Итог import ph_diagnostics_report  # type: ignore[attr-defined]
    except ImportError:
        # Fallback на старую логику, если функция ещё не добавлена
        result: Dict[str, Any] = {"available": False}
        cph = getattr(cf, "cph", None)
        if cph is None:
            result["note"] = "cf.cph not available"
            return result
        check = getattr(cph, "check_assumptions", None)
        if check is None or not callable(check):
            result["note"] = "check_assumptions not available"
            return result
        params_obj = getattr(cph, "params_", None)
        cox_names = [str(name) for name in _safe_keys(params_obj)]
        cols = ["time", "event"]
        for col in cox_names:
            if col in data_mod.columns:
                cols.append(col)
        missing_model_cols = [col for col in cox_names if col not in data_mod.columns]
        if missing_model_cols:
            result["note"] = (
                f"PH diagnostics skipped: missing model columns in data_mod: {missing_model_cols}"
            )
            return result
        try:
            ph_data = data_mod[cols].astype(float)
        except (TypeError, ValueError) as exc:
            result["error"] = repr(exc)
            return result
        try:
            check(ph_data, p_value_threshold=0.05)
            result["available"] = True
            result["note"] = "check_assumptions executed"
        except TypeError:
            try:
                check(ph_data)
                result["available"] = True
                result["note"] = "check_assumptions executed"
            except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
                result["error"] = repr(exc)
        except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
            result["error"] = repr(exc)
        return result

    # ─── Новый путь: структурированный PH-отчёт ──────────────────────
    cph = getattr(cf, "cph", None)
    if cph is None:
        return {
            "available": False,
            "status": "ERROR",
            "error": "cf.cph not available",
        }

    report = ph_diagnostics_report(
        cph=cph,
        data=data_mod,
        alpha=0.05,
        time_transform="rank",
    )
    report["available"] = report.get("status") != "ERROR"
    return report


# ---------------------------------------------------------------------------
# Baseline serialization
# ---------------------------------------------------------------------------
def serialize_baseline(h_df: Any) -> Dict[str, List[float]]:
    if h_df is None:
        raise RuntimeError("baseline is None")

    try:
        length = len(h_df)
    except TypeError:
        raise RuntimeError("unsupported baseline type")

    if length == 0:
        raise RuntimeError("empty baseline")

    if isinstance(h_df, pd.Series):
        times = np.asarray(h_df.index, dtype=float)
        values = np.asarray(h_df.to_numpy(), dtype=float)
    elif isinstance(h_df, pd.DataFrame):
        if h_df.shape[1] != 1:
            raise RuntimeError("baseline DataFrame must contain exactly one column")
        times = np.asarray(h_df.index, dtype=float)
        values = np.asarray(h_df.iloc[:, 0].to_numpy(), dtype=float)
    else:
        raise RuntimeError("unsupported baseline type")

    if len(times) != len(values):
        raise RuntimeError("baseline length mismatch")
    if not np.isfinite(times).all():
        raise RuntimeError("baseline times contain non-finite values")
    if not np.isfinite(values).all():
        raise RuntimeError("baseline values contain non-finite values")
    if np.any(times < 0.0):
        raise RuntimeError("baseline times must be non-negative")

    order = np.argsort(times)
    times = times[order]
    values = values[order]

    # PATCH 3: Более надежный способ агрегации дубликатов через pandas groupby
    # np.maximum.reduceat может работать некорректно, если дубликаты не идут подряд
    # (хотя после argsort это маловероятно). pandas groupby более явный и надежный.
    df = pd.DataFrame({"times": times, "values": values})
    df = df.groupby("times", sort=True)["values"].max().reset_index()
    times = df["times"].values
    values = df["values"].values

    if len(times) == 0:
        raise RuntimeError("baseline became empty after aggregation")

    # Обязательно добавляем H(0) = 0.
    if times[0] > 0.0:
        times = np.concatenate(([0.0], times))
        values = np.concatenate(([0.0], values))
    else:
        times[0] = 0.0
        values[0] = 0.0

    values = np.clip(values, 0.0, None)
    values = np.maximum.accumulate(values)

    return {
        "times": times.tolist(),
        "values": values.tolist(),
    }


def validate_cox_baseline(baseline: Dict[str, Any]) -> bool:
    times = np.asarray(baseline["times"], dtype=float)
    values = np.asarray(baseline["values"], dtype=float)

    if len(times) != len(values):
        raise RuntimeError("Cox baseline length mismatch")
    if len(times) == 0:
        raise RuntimeError("Empty Cox baseline")
    if np.any(np.diff(times) < 0.0):
        raise RuntimeError("Cox times not sorted")
    if not np.isfinite(values).all():
        raise RuntimeError("Invalid Cox hazard")
    if np.any(values < 0.0):
        raise RuntimeError("Cox baseline values must be non-negative")
    if np.any(np.diff(values) < -1e-12):
        raise RuntimeError("Cox cumulative hazard must be non-decreasing")

    return True


# ---------------------------------------------------------------------------
# Template covariates
# ---------------------------------------------------------------------------
def build_raw_template_covariates(data: pd.DataFrame) -> Dict[str, float]:
    """
    Build raw template covariates for prediction_engine.
    Values are RAW, not standardized.
    """

    def safe_mean(column_name: str, default: float = 0.0) -> float:
        if column_name not in data.columns:
            logger.debug(
                "Template covariate column '%s' not found, using default %s",
                column_name,
                default,
            )
            return default

        try:
            values = data[column_name].astype(float)
        except (ValueError, TypeError):
            logger.warning(
                "Template covariate column '%s' cannot be converted to float",
                column_name,
            )
            return default

        values = values[np.isfinite(values)]
        if values.empty:
            return default
        return float(values.mean())

    brand_default = 4.0
    if "Brand" in data.columns:
        brand_mode = data["Brand"].mode()
        if not brand_mode.empty:
            try:
                brand_default = float(brand_mode.iloc[0])
            except (ValueError, TypeError):
                brand_default = 4.0

    return {
        "Z": safe_mean("Z", 0.0),
        "x_age": safe_mean("Age", 10.0),
        "x_hours": safe_mean("Hours", 1000.0),
        "x_climate": safe_mean("Climate", 0.5),
        "x_soil": safe_mean("Soil", 0.5),
        "x_brand": brand_default,
        "x_power": safe_mean("Power", 200.0),
        "x_age_hours": safe_mean("x_age_hours", 0.0),
    }


def _compute_major_failure_calibration(
    data: pd.DataFrame,
    prior_mean: float,
    prior_effective_n: float,
) -> Dict[str, Any]:
    """
    P0-5: Bayesian Beta-Binomial calibration для доли major failures.

    Prior: Beta(alpha_0, beta_0), где
        alpha_0 = prior_mean * prior_effective_n
        beta_0  = (1 - prior_mean) * prior_effective_n

    Posterior: Beta(alpha_0 + k, beta_0 + n - k), где
        k = число major events
        n = общее число events

    Parameters
    ----------
    data : pd.DataFrame
        Данные с колонками event и failure_type (или brand).
    prior_mean : float
        Prior mean (0.30).
    prior_effective_n : float
        Prior effective sample size (30.0).

    Returns
    -------
    dict
        Полная структура calibration:
        {
            "overall": {...},
            "by_brand": {...},
            "by_brand_observed": {...},
            "by_brand_posterior": {...},
        }
    """
    result: Dict[str, Any] = {}

    # ─── Overall calibration ─────────────────────────────────────────
    alpha_0 = float(prior_mean) * float(prior_effective_n)
    beta_0 = (1.0 - float(prior_mean)) * float(prior_effective_n)

    # Извлечь k и n из данных
    n_total = 0
    n_major = 0

    if "failure_type" in data.columns:
        ft = data["failure_type"].astype(str).to_numpy()
        # Считаем ВСЕ отказы: major + minor (не только event=True)
        n_major = int((ft == "major").sum())
        n_minor = int((ft == "minor_observed").sum()) + int((ft == "minor").sum())
        n_total = n_major + n_minor
    elif "event" in data.columns:
        # Fallback: если нет failure_type, используем все events как major
        n_total = int(data["event"].astype(int).sum())
        n_major = n_total  # консервативный fallback

    # Posterior
    alpha_post = alpha_0 + n_major
    beta_post = beta_0 + (n_total - n_major)
    posterior_mean = alpha_post / (alpha_post + beta_post)

    # 95% Credible Interval
    try:
        from scipy import stats as _st

        ci_low = float(_st.beta.ppf(0.025, alpha_post, beta_post))
        ci_high = float(_st.beta.ppf(0.975, alpha_post, beta_post))
    except Exception:
        sd = float(
            np.sqrt(posterior_mean * (1.0 - posterior_mean) / (alpha_post + beta_post + 1.0))
        )
        ci_low = max(0.0, posterior_mean - 1.96 * sd)
        ci_high = min(1.0, posterior_mean + 1.96 * sd)

    observed_share = n_major / n_total if n_total > 0 else float("nan")

    result["overall"] = {
        "prior_alpha": float(alpha_0),
        "prior_beta": float(beta_0),
        "prior_mean": float(prior_mean),
        "prior_effective_n": float(prior_effective_n),
        "n_events": int(n_total),
        "n_major": int(n_major),
        "observed_share": float(observed_share),
        "posterior_alpha": float(alpha_post),
        "posterior_beta": float(beta_post),
        "posterior_mean": float(posterior_mean),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }

    # ─── Brand-level calibration ─────────────────────────────────────
    by_brand: Dict[str, float] = {}
    by_brand_observed: Dict[str, float] = {}
    by_brand_posterior: Dict[str, Any] = {}

    brand_col = None
    if "Brand" in data.columns:
        brand_col = "Brand"
    elif "brand_code" in data.columns:
        brand_col = "brand_code"

    if brand_col is not None and "failure_type" in data.columns:
        events_mask = data["event"].astype(bool).to_numpy()
        events_data = data.loc[events_mask]

        for brand_val in events_data[brand_col].unique():
            brand_mask = events_data[brand_col] == brand_val
            brand_events = events_data.loc[brand_mask]
            brand_n = len(brand_events)
            brand_major = int((brand_events["failure_type"].astype(str) == "major").sum())

            # Raw share
            raw_share = brand_major / brand_n if brand_n > 0 else float("nan")
            brand_name = str(brand_val)
            by_brand_observed[brand_name] = float(raw_share)

            # Posterior (тот же global prior для каждого бренда)
            b_alpha_post = alpha_0 + brand_major
            b_beta_post = beta_0 + (brand_n - brand_major)
            b_posterior_mean = b_alpha_post / (b_alpha_post + b_beta_post)

            by_brand[brand_name] = float(b_posterior_mean)
            by_brand_posterior[brand_name] = {
                "n_events": int(brand_n),
                "n_major": int(brand_major),
                "observed_share": float(raw_share),
                "posterior_alpha": float(b_alpha_post),
                "posterior_beta": float(b_beta_post),
                "posterior_mean": float(b_posterior_mean),
            }

    result["by_brand"] = by_brand
    result["by_brand_observed"] = by_brand_observed
    result["by_brand_posterior"] = by_brand_posterior

    return result


# ---------------------------------------------------------------------------
# Partial-out helper
# ---------------------------------------------------------------------------
def compute_partial_out_fields(
    fitted_fs: Any,
    data_mod: pd.DataFrame,
    cf: Any,
) -> Tuple[Dict[str, float], Dict[str, float], float, float, float]:
    partial_betas_raw = _to_dict_safe(getattr(cf, "partial_out_all_betas", None))
    training_x_means_raw = _to_dict_safe(getattr(cf, "training_x_means", None))
    training_pl_hat_mean = float(getattr(cf, "training_pl_hat_mean", 0.0) or 0.0)

    fitted_model = getattr(fitted_fs, "fitted", None)
    if fitted_model is None:
        raise RuntimeError("First stage fitted model is empty")

    fitted_values_obj = getattr(fitted_model, "fittedvalues", None)
    if fitted_values_obj is None:
        raise RuntimeError("First stage fitted values are empty")

    pl_hat = np.asarray(fitted_values_obj, dtype=float)
    if pl_hat.size == 0:
        raise RuntimeError("First stage fitted values are empty")
    if not np.all(np.isfinite(pl_hat)):
        raise RuntimeError("First stage fitted values contain non-finite values")

    if not math.isfinite(training_pl_hat_mean):
        training_pl_hat_mean = float(np.mean(pl_hat))

    partial_betas = dict(partial_betas_raw)
    training_x_means = dict(training_x_means_raw)

    if not partial_betas:
        # Берём x_cols из first_stage.exog_names, чтобы включить дамми бренда
        # и все реальные регрессоры первой стадии.
        first_stage_model = getattr(fitted_fs, "fitted", None)
        fs_exog_names = _safe_exog_names(first_stage_model)

        # Конвенция: исключаем Z из partial-out согласно PL_HAT_EXOG_CONVENTION
        forbidden = {"const", "intercept", "PeakLoad", "time", "event"}
        if PL_HAT_EXOG_CONVENTION == "exclude_instrument":
            forbidden.add("Z")

        if fs_exog_names:
            x_cols = [
                str(col)
                for col in fs_exog_names
                if col not in forbidden and col in data_mod.columns
            ]
        else:
            # Fallback на X_STANDARDIZATION (legacy поведение).
            x_cols = [str(col) for col in X_STANDARDIZATION.keys() if col in data_mod.columns]

        if x_cols:
            params_obj = getattr(fitted_fs, "params", None)
            params_dict = _safe_params_to_float_dict(params_obj)

            partial_betas = {col: params_dict[col] for col in x_cols if col in params_dict}
            training_x_means = {col: float(data_mod[col].mean()) for col in x_cols}

        # Fallback, если first-stage params по какой-то причине недоступны.
        if not partial_betas:
            try:
                x_design = sm.add_constant(data_mod[x_cols], has_constant="add")
                ols = sm.OLS(pl_hat, x_design).fit()

                partial_betas = {
                    str(col): float(ols.params[col]) for col in x_cols if col in ols.params.index
                }
                training_x_means = {str(col): float(data_mod[col].mean()) for col in x_cols}
            except (ValueError, TypeError, RuntimeError):
                partial_betas = {}
                training_x_means = {}

    clean_partial_betas: Dict[str, float] = {}
    for key, value in partial_betas.items():
        try:
            value_f = float(value)
            if math.isfinite(value_f):
                clean_partial_betas[str(key)] = value_f
        except (TypeError, ValueError):
            continue

    clean_training_x_means: Dict[str, float] = {}
    for key, value in training_x_means.items():
        try:
            value_f = float(value)
            if math.isfinite(value_f):
                clean_training_x_means[str(key)] = value_f
        except (TypeError, ValueError):
            continue

    partial_betas = clean_partial_betas
    training_x_means = clean_training_x_means

    missing_means = [str(name) for name in partial_betas if str(name) not in training_x_means]
    if missing_means:
        logger.warning(
            "training_x_means missing for partial_out variables: %s. Filling.",
            sorted(missing_means),
        )
        for name in missing_means:
            if name in data_mod.columns:
                values = data_mod[name].astype(float).to_numpy()
                values = values[np.isfinite(values)]
                if values.size > 0:
                    training_x_means[name] = float(np.mean(values))
                else:
                    training_x_means[name] = 0.0
            else:
                logger.warning(
                    "Partial-out variable '%s' not found in data_mod; using mean 0.0",
                    name,
                )
                training_x_means[name] = 0.0

    partial_out_X_beta = 0.0
    if partial_betas:
        partial_out_X_beta = sum(
            float(beta) * float(training_x_means.get(name, 0.0))
            for name, beta in partial_betas.items()
        )

    if not math.isfinite(partial_out_X_beta):
        partial_out_X_beta = 0.0

    # Deprecated legacy field.
    # Ранее здесь использовалось среднее средних, что математически некорректно.
    training_x_mean = 0.0

    return (
        partial_betas,
        training_x_means,
        training_pl_hat_mean,
        float(partial_out_X_beta),
        training_x_mean,
    )


# ---------------------------------------------------------------------------
# CF metadata helper
# ---------------------------------------------------------------------------
def prepare_cf_basis_metadata(
    cf: Any,
    resid: np.ndarray,
    v_hat_basis: str,
    v_hat_basis_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    meta = _to_dict_safe(getattr(cf, "cf_basis_metadata", None))

    if resid is None:
        raise RuntimeError("Residuals are empty")

    residuals = np.asarray(resid, dtype=float)
    if residuals.size == 0:
        raise RuntimeError("Residuals are empty")
    if not np.all(np.isfinite(residuals)):
        raise RuntimeError("Residuals contain non-finite values")

    residuals_mean = float(np.mean(residuals))
    residuals_std = float(np.std(residuals, ddof=1))
    if not math.isfinite(residuals_std) or residuals_std <= 0.0:
        residuals_std = 1.0

    meta["residuals_mean"] = residuals_mean
    meta["residuals_std"] = residuals_std
    meta["training_residuals_std"] = residuals_std
    meta["training_residuals_mean"] = residuals_mean
    meta["requested_v_hat_basis"] = v_hat_basis

    if "v_hat_basis" not in meta:
        meta["v_hat_basis"] = getattr(cf, "v_hat_basis", "linear")

    if "v_hat_cols" not in meta:
        cph = getattr(cf, "cph", None)
        params_obj = getattr(cph, "params_", {}) if cph is not None else {}
        cox_names = [str(name) for name in _safe_keys(params_obj)]

        meta["v_hat_cols"] = [
            str(name) for name in cox_names if str(name).lower().startswith(("v_hat", "eps_d_hat"))
        ]

        if not meta.get("v_hat_cols"):
            logger.warning(
                "CF basis columns not found in Cox params. CF correction may be inactive."
            )

    if "linear_standardized" not in meta:
        has_old_linear = any(str(col).lower() == "eps_d_hat" for col in meta.get("v_hat_cols", []))
        meta["linear_standardized"] = not has_old_linear

    meta["cf_standardization_convention"] = "residual_mean_std"
    meta["residual_policy_production"] = "plug-in"
    meta["cox_peakload_convention"] = "observed_peakload"

    if v_hat_basis_params is not None:
        meta["requested_v_hat_basis_params"] = dict(v_hat_basis_params)

    basis_type = str(meta.get("v_hat_basis", "linear")).lower()
    if basis_type == "spline":
        if "spline_domain_min" not in meta or "spline_domain_max" not in meta:
            v_std = (residuals - residuals_mean) / residuals_std
            v_std = v_std[np.isfinite(v_std)]
            if v_std.size > 0:
                meta["spline_domain_min"] = float(np.quantile(v_std, 0.005))
                meta["spline_domain_max"] = float(np.quantile(v_std, 0.995))
            else:
                meta["spline_domain_min"] = -3.0
                meta["spline_domain_max"] = 3.0

    return meta


# ---------------------------------------------------------------------------
# Post-fit helpers
# ---------------------------------------------------------------------------
def _safe_row_float(row: pd.Series, key: str, default: float) -> float:
    if key in row.index:
        raw = row[key]
        if raw is not None:
            try:
                value = float(raw)
                if math.isfinite(value):
                    return value
            except (ValueError, TypeError):
                pass
    return float(default)


def _row_covariates(row: pd.Series, template_dict: Dict[str, float]) -> Dict[str, float]:
    return {
        "Z": _safe_row_float(row, "Z", template_dict.get("Z", 0.0)),
        "x_age": _safe_row_float(
            row,
            "Age",
            _safe_row_float(row, "x_age", template_dict.get("x_age", 10.0)),
        ),
        "x_hours": _safe_row_float(
            row,
            "Hours",
            _safe_row_float(row, "x_hours", template_dict.get("x_hours", 1000.0)),
        ),
        "x_climate": _safe_row_float(
            row,
            "Climate",
            _safe_row_float(row, "x_climate", template_dict.get("x_climate", 0.5)),
        ),
        "x_soil": _safe_row_float(
            row,
            "Soil",
            _safe_row_float(row, "x_soil", template_dict.get("x_soil", 0.5)),
        ),
        "x_brand": _safe_row_float(
            row,
            "Brand",
            _safe_row_float(row, "x_brand", template_dict.get("x_brand", 4.0)),
        ),
        "x_power": _safe_row_float(
            row,
            "Power",
            _safe_row_float(row, "x_power", template_dict.get("x_power", 200.0)),
        ),
        # ─── НОВОЕ: передаём уже вычисленный x_age_hours из данных,
        # чтобы prediction engine НЕ пересчитывал его из сырых Age × Hours
        "x_age_hours": _safe_row_float(row, "x_age_hours", 0.0),
    }


def estimate_post_fit_average_probability(
    model_params: Any,
    data: pd.DataFrame,
    target_time: float,
    template_dict: Dict[str, float],
    seed: int,
    max_rows: int = 500,
) -> Optional[float]:
    try:
        if data is None or data.empty:
            return None

        rng = make_rng(seed, 77777)
        if len(data) > max_rows:
            idx = rng.choice(len(data), size=max_rows, replace=False)
            sample = data.iloc[idx]
        else:
            sample = data

        probs: List[float] = []

        for _, row in sample.iterrows():
            peak = _safe_row_float(row, "PeakLoad", float("nan"))
            if not math.isfinite(peak):
                continue

            covariates = _row_covariates(row, template_dict)

            try:
                p = predict_probability(
                    model_params,
                    peak,
                    float(target_time),
                    residual_policy="plug-in",
                    covariates=covariates,
                    time_horizon_unit=MODEL_TIME_UNIT,
                    strict_covariates=True,
                )
                p = float(p)
                if math.isfinite(p) and 0.0 <= p <= 1.0:
                    probs.append(p)
            except Exception as exc:
                logger.debug("predict_probability failed for row: %s", exc)
                continue

        if not probs:
            return None

        return float(np.mean(probs))

    except (ValueError, TypeError, RuntimeError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# P-12: Kaplan-Meier validation
# ---------------------------------------------------------------------------
def run_kaplan_meier_validation(
    model_params: Any,
    data: pd.DataFrame,
    target_time: float,
) -> Optional[Dict[str, float]]:
    """
    P-12: Kaplan-Meier валидация baseline survival.
    Сравнивает модельную S0(t) = exp(-H0(t)) с эмпирической K-M оценкой.
    """
    try:
        # Поддержка обоих форматов данных:
        # - Симуляция: колонки "time" / "event"
        # - Claims: колонки "failure_time" / "event_flag"
        if "time" in data.columns:
            times = data["time"].astype(float).to_numpy()
        elif "failure_time" in data.columns:
            times = data["failure_time"].astype(float).to_numpy()
        else:
            logger.warning("K-M валидация: колонка времени не найдена")
            return None

        if "event" in data.columns:
            events = data["event"].astype(int).to_numpy()
        elif "event_flag" in data.columns:
            events = data["event_flag"].astype(int).to_numpy()
        else:
            logger.warning("K-M валидация: колонка события не найдена")
            return None

        # Удалить строки с невалидным временем
        valid = np.isfinite(times) & (times > 0)
        times = times[valid]
        events = events[valid]

        if len(times) == 0:
            logger.warning("K-M валидация: нет валидных наблюдений")
            return None

        result = kaplan_meier_check(
            params=model_params,
            times=times,
            events=events,
            eval_horizon=float(target_time),
        )
        return result
    except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
        logger.warning("Kaplan-Meier validation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Training pipeline functions
# ---------------------------------------------------------------------------
def collect_training_config() -> TrainingConfig:
    """
    Собрать все параметры обучения от пользователя.
    Возвращает заполненный TrainingConfig.
    """
    cfg = TrainingConfig(
        gamma=DEFAULT_GAMMA,
        delta=DEFAULT_DELTA,
        baseline_shape=DEFAULT_WEIBULL_SHAPE,
        target_time=CALIBRATION_HORIZON_ENGINE_HOURS,
        target_quantile=DEFAULT_TARGET_PROBABILITY,
        target_probability=DEFAULT_TARGET_PROBABILITY,
    )

    # ─── Фаза 3: автозагрузка TUM-статистик ────────────────────────────────
    tum_stats_candidates = [
        Path("data/processed/tum/tum_peakload_stats.json"),
        Path("tum_peakload_stats.json"),
        Path(__file__).parent / "tum_peakload_stats.json",
    ]

    for tum_stats_path in tum_stats_candidates:
        if tum_stats_path.exists():
            try:
                with open(tum_stats_path, "r", encoding="utf-8") as f:
                    tum_stats = json.load(f)

                cfg.tum_peakload_target_mean = _as_float_or_none(tum_stats.get("tum_peakload_mean"))
                cfg.tum_peakload_target_std = _as_float_or_none(tum_stats.get("tum_peakload_std"))

                if (
                    cfg.tum_peakload_target_mean is not None
                    and cfg.tum_peakload_target_std is not None
                ):
                    logger.info(
                        "TUM PeakLoad статистики загружены из %s: mean=%.4f, std=%.4f",
                        tum_stats_path,
                        cfg.tum_peakload_target_mean,
                        cfg.tum_peakload_target_std,
                    )
                else:
                    cfg.tum_peakload_target_mean = None
                    cfg.tum_peakload_target_std = None

                cfg.tum_stats_path = str(tum_stats_path)
                break
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Не удалось загрузить TUM-статистики из %s: %s",
                    tum_stats_path,
                    exc,
                )

    if not cfg.tum_stats_path:
        cfg.tum_stats_path = str(tum_stats_candidates[0])

    print()
    print("=" * 70)
    print("Обучение модели CF Cox / IV-Cox (v3.0 hardened + v0.2)")
    print("Согласовано с DGP v3.0 и обновлённым Итог.py")
    print("=" * 70)

    if cfg.tum_peakload_target_mean is not None and cfg.tum_peakload_target_std is not None:
        print(
            f"✅ TUM PeakLoad калибровка: mean={cfg.tum_peakload_target_mean:.4f}, "
            f"std={cfg.tum_peakload_target_std:.4f}"
        )
    else:
        print("⚠️  TUM PeakLoad калибровка НЕ применена (статистики не загружены)")

    # ─── ВЫБОР ИСТОЧНИКА ДАННЫХ (должен быть ПЕРВЫМ!) ─────────────
    print()
    print("Источник обучающих данных:")
    print("  1) Симуляция DGP (Monte Carlo)")
    print("  2) Литературные claims (claims_clean.csv)")
    print("  3) Гибрид: DGP + реальные weather/soil (рекомендуется)")
    data_source_choice = ask("Выбор источника данных", "1").strip() or "1"

    use_claims = data_source_choice == "2"
    use_hybrid = data_source_choice == "3"

    # УДАЛЕНО: allow_auto_instrument_correction больше не используется.
    # Автокоррекция инструмента удалена как методологически некорректная.
    # При слабом инструменте (F < 10.4) происходит переключение в predictive режим.

    if use_hybrid:
        # Гибридный режим: DGP + реальные weather/soil
        print()
        print("=" * 70)
        print("РЕЖИМ: ГИБРИДНЫЙ (DGP + реальные weather/soil)")
        print("=" * 70)
        print("Структурные данные генерируются через DGP (γ > 0 гарантирован).")
        print("Ковариаты x_climate, x_soil и инструмент Z берутся из")
        print("реальных спутниковых данных NASA POWER / GLDAS-2.1.")
        print("=" * 70)

        # Проверяем наличие файлов
        weather_path = Path("data/processed/weather/weather_windows.csv")
        soil_path = Path("data/processed/soil/soil_windows.csv")

        if weather_path.exists():
            logger.info("✅ weather_windows.csv найден")
        else:
            logger.warning("⚠️ weather_windows.csv не найден, будет синтетический fallback")

        if soil_path.exists():
            logger.info("✅ soil_windows.csv найден")
        else:
            logger.warning("⚠️ soil_windows.csv не найден, будет синтетический fallback")

    if use_claims:
        # Проверяем доступность claims
        claims_path_check = Path("data/processed/claims/claims_clean.csv")
        if not claims_path_check.exists():
            logger.warning(
                "Claims не найдены: %s. "
                "Запустите generate_literature_claims_v2.py и claims_validator.py",
                claims_path_check,
            )
            if ask_yesno("Переключиться на симуляцию?", True):
                use_claims = False
            else:
                raise RuntimeError("Claims не найдены")
        else:
            logger.info("Используются литературные claims из %s", claims_path_check)
            print()
            print("=" * 70)
            print("РЕЖИМ: ЛИТЕРАТУРНЫЕ CLAIMS")
            print("Параметры DGP не требуются — данные уже существуют.")
            print("=" * 70)

    # ─── Dataset size and seed ──────────────────────────────────────────────
    if not use_claims:
        # ─── ВЕТКА: СИМУЛЯЦИЯ DGP ──────────────────────────────────
        cfg.n = ask_int("Размер выборки", 40000)
        if cfg.n < 500:
            raise ValueError("Размер выборки должен быть >= 500")

        cfg.seed = ask_nonnegative_int("Сид", 12345)

        # ─── Contamination ──────────────────────────────────────────────────────
        cfg.contamination = ask_yesno(
            "Использовать загрязнение выборки (Student's t вместо normal)?",
            False,
        )

        if cfg.contamination:
            print()
            print("ВНИМАНИЕ: загрязнённые данные подходят только для stress-test.")
            print("Не используйте такие модели для продуктового ценообразования.")

            if not ask_yesno("Подтвердить stress-test режим?", False):
                cfg.contamination = False
                print("Отменено: используется чистая выборка.")

        if cfg.contamination:
            cfg.contamination_probability = ask_open_probability(
                "Вероятность загрязнённой компоненты (pi)",
                0.1,
            )
        else:
            cfg.contamination_probability = 0.0

        cfg.stress_test_mode = bool(cfg.contamination and cfg.contamination_probability > 0.0)

        if cfg.contamination and cfg.contamination_probability <= 0.0:
            logger.warning("contamination=True, но contamination_probability <= 0")

        if not cfg.contamination and cfg.contamination_probability > 0.0:
            logger.warning("contamination_probability игнорируется, потому что contamination=False")

        # ─── DGP parameters ─────────────────────────────────────────────────────
        print()
        print("Параметры DGP (v3.0)")
        print("-" * 70)

        cfg.gamma = ask_float(
            "gamma (эффект PeakLoad)", DEFAULT_GAMMA, min_value=0.0, max_value=10.0
        )
        validate_finite(cfg.gamma, "gamma")

        cfg.rho = ask_float("rho (корреляция ошибок)", 0.7, min_value=-0.999999, max_value=0.999999)
        validate_finite(cfg.rho, "rho")
        if not (-1.0 < cfg.rho < 1.0):
            raise ValueError("rho должно быть в открытом диапазоне (-1, 1)")

        cfg.delta = ask_float(
            "delta (латентная гетерогенность)",
            DEFAULT_DELTA,
            min_value=0.0,
            max_value=0.7,
        )
        validate_finite(cfg.delta, "delta")
        if cfg.delta > 0.5:
            print("ВНИМАНИЕ: delta > 0.5 может давать большой разброс hazard.")
            if not ask_yesno("Продолжить с delta > 0.5?", False):
                cfg.delta = ask_float(
                    "Введите delta (0.0–0.7, где 0 = нет гетерогенности)",
                    0.5,
                    min_value=0.0,
                    max_value=0.7,
                )

        cfg.beta_age_hours = ask_float(
            "beta_age_hours (синергия возраст×наработка, 0=нет)",
            0.15,
            min_value=0.0,
            max_value=1.0,
        )

        cfg.fs_intercept = ask_float(
            "fs_intercept — смещение для первой стадии модели (погрешность инструмента Z)", 10.0
        )
        validate_finite(cfg.fs_intercept, "fs_intercept")

        cfg.structural_intercept = ask_float(
            "structural_intercept (по умолчанию = fs_intercept)",
            cfg.fs_intercept,
        )
        validate_finite(cfg.structural_intercept, "structural_intercept")

        intercept_diff = abs(cfg.fs_intercept - cfg.structural_intercept)
        if intercept_diff > 1.0:
            raise ValueError("Разница intercept'ов слишком велика.")

        if intercept_diff > 0.01:
            print("ВНИМАНИЕ: fs_intercept != structural_intercept.")
            if ask_yesno(f"Установить structural_intercept = {cfg.fs_intercept}?", True):
                cfg.structural_intercept = cfg.fs_intercept
            else:
                raise ValueError(
                    "Для корректного обучения требуется fs_intercept == structural_intercept."
                )

        cfg.fs_z = ask_float("first_stage_z_coef (сила инструмента Z)", 0.5)
        validate_finite(cfg.fs_z, "first_stage_z_coef")
        if abs(cfg.fs_z) < 0.1:
            print("ВНИМАНИЕ: очень слабый инструмент Z.")

        cfg.fs_z_initial = float(cfg.fs_z)

        cfg.baseline_family = (
            ask(
                "baseline_family — семейство распределения: exponential / weibull / gompertz",
                "weibull",
            )
            .lower()
            .strip()
        )

        if cfg.baseline_family not in {"exponential", "weibull", "gompertz"}:
            raise ValueError("Неподдерживаемый baseline_family")

        cfg.baseline_shape = None
        if cfg.baseline_family == "weibull":
            cfg.baseline_shape = ask_float(
                "Weibull shape — параметр формы (>1 = риск растёт со временем)",
                DEFAULT_WEIBULL_SHAPE,
                min_value=0.05,
                max_value=10.0,
            )
        elif cfg.baseline_family == "gompertz":
            cfg.baseline_shape = ask_float(
                "Gompertz rate — параметр скорости роста риска (gompertz)",
                DEFAULT_GOMPERTZ_RATE,
                min_value=1e-6,
                max_value=0.4,
            )

        # ─── Параметрическая подгонка базового риска ──────────────────────────
        print()
        print("Параметрическая подгонка базового риска (для экстраполяции):")
        print("  1) weibull      — H0(t) = λ·t^k (рекомендуется)")
        print("  2) gompertz     — H0(t) = (λ/b)(e^{bt} - 1)")
        print("  3) exponential  — H0(t) = λ·t")
        print("  4) none         — без подгонки (только Breslow)")
        parametric_choice = ask("Семейство для подгонки базового риска", "1").strip() or "1"
        cfg.parametric_baseline_fit = {
            "1": "weibull",
            "2": "gompertz",
            "3": "exponential",
            "4": "none",
        }.get(parametric_choice, "weibull")

        cfg.target_event_rate = ask_open_probability(
            "Целевая доля событий (target event rate)",
            0.02,
        )

        # ─── P-03 / D1: event definition и competing risks ──────────────────────
        print()
        print("Определение страхового события:")
        print("  1) total_loss  — только полная гибель")
        print("  2) major_claim — крупный страховой случай (рекомендуется)")
        print("  3) any_failure — любой отказ")

        ed_choice = (
            ask("Выберите (1 = полная гибель, 2 = крупный случай, 3 = любой отказ)", "2").strip()
            or "2"
        )
        cfg.event_definition = {
            "1": "total_loss",
            "2": "major_claim",
            "3": "any_failure",
        }.get(ed_choice, "major_claim")
        cfg.event_definition = validate_event_definition(cfg.event_definition)

        cfg.competing_risks = ask_yesno(
            "Использовать конкурирующие риски (minor + major)?",
            True,
        )

        cfg.minor_failure_rate = 0.002
        if cfg.competing_risks:
            cfg.minor_failure_rate = ask_float(
                "λ_minor (интенсивность minor-отказов, 1/мч)",
                0.002,
                min_value=1e-6,
            )

        if (
            cfg.competing_risks
            and cfg.event_definition != "any_failure"
            and cfg.target_event_rate > 0.20
        ):
            print()
            print(
                f"⚠️  target_event_rate={cfg.target_event_rate:.3f} выглядит "
                f"слишком высоким для event_definition='{cfg.event_definition}'"
            )

            if ask_yesno("Установить target_event_rate = 0.01?", True):
                cfg.target_event_rate = 0.01
            else:
                cfg.target_event_rate = ask_open_probability(
                    "Введите целевую долю событий",
                    cfg.target_event_rate,
                )

        # ─── D2: сегмент парка ──────────────────────────────────────────────────
        cfg.segment = ask("Сегмент парка (light/heavy)", "light").strip().lower()
        if cfg.segment not in ("light", "heavy"):
            cfg.segment = "light"

        # ─── Calibration method ─────────────────────────────────────────────────
        print()
        print("Метод калибровки базового риска:")
        print("1) Целевая вероятность отказа")
        print("2) Квантильный метод (legacy)")
        print("3) Ручной ввод базового риска")

        calib_choice = ask("Выберите метод калибровки", "1").strip() or "1"
        cfg.calib_method = {
            "1": "probability",
            "2": "quantile",
            "3": "manual",
        }.get(calib_choice, "probability")

        if calib_choice not in ("1", "2", "3"):
            logger.warning(
                "Неизвестный метод калибровки '%s', используется probability",
                calib_choice,
            )

        cfg.target_time = CALIBRATION_HORIZON_ENGINE_HOURS
        cfg.target_quantile = DEFAULT_TARGET_PROBABILITY
        cfg.target_probability = None
        cfg.baseline_hazard_initial = None

        if cfg.calib_method == "probability":
            print(
                f"Горизонт калибровки: {CALIBRATION_HORIZON_ENGINE_HOURS:.0f} "
                f"мото-часов ({CALIBRATION_HORIZON_DAYS:.0f} дней × "
                f"{DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY:.0f} мч/день)"
            )

            cfg.target_time = ask_float(
                "Горизонт калибровки (мото-часы)",
                CALIBRATION_HORIZON_ENGINE_HOURS,
                min_value=1.0,
            )
            cfg.target_probability = ask_open_probability(
                "Целевая вероятность события до горизонта",
                DEFAULT_TARGET_PROBABILITY,
            )
            validate_open_probability(cfg.target_probability, "target_probability")
            sanity_check_target_probability(
                cfg.target_probability,
                cfg.event_definition,
            )

            use_auto_initial = ask_yesno(
                "Использовать автоматический initial baseline hazard?",
                True,
            )
            if not use_auto_initial:
                default_initial = compute_initial_baseline(
                    target_probability=cfg.target_probability,
                    time_horizon=cfg.target_time,
                    family=cfg.baseline_family,
                    shape=cfg.baseline_shape,
                )
                cfg.baseline_hazard_initial = ask_float(
                    "Начальный базовый риск (baseline_hazard_initial)",
                    default_initial,
                    min_value=1e-12,
                )

        elif cfg.calib_method == "quantile":
            cfg.target_time = ask_float(
                "Целевое время (мото-часы)",
                CALIBRATION_HORIZON_ENGINE_HOURS,
                min_value=1.0,
            )
            cfg.target_quantile = ask_open_probability(
                "Целевой квантиль (доля отказов до target_time)",
                DEFAULT_TARGET_PROBABILITY,
            )
            if cfg.target_quantile > 0.5:
                logger.warning(
                    "target_quantile > 0.5 означает, что большинство машин откажет до target_time."
                )

        else:  # manual
            cfg.target_time = ask_float(
                "Горизонт калибровки (мото-часы)",
                CALIBRATION_HORIZON_ENGINE_HOURS,
                min_value=1.0,
            )
            default_manual = compute_initial_baseline(
                target_probability=DEFAULT_TARGET_PROBABILITY,
                time_horizon=cfg.target_time,
                family=cfg.baseline_family,
                shape=cfg.baseline_shape,
            )
            cfg.baseline_hazard_initial = ask_float(
                "Базовый риск (baseline_hazard)",
                default_manual,
                min_value=1e-12,
            )

        cfg.do_standardize = ask_yesno(
            "Стандартизировать PeakLoad (привести к среднему 0 и стандартному откл. 1)?", True
        )
        cfg.do_center_only = False
        if not cfg.do_standardize:
            cfg.do_center_only = ask_yesno(
                "Только центрировать PeakLoad (вычесть среднее, без деления на отклонение)?", True
            )

    else:
        # ─── ВЕТКА: ЛИТЕРАТУРНЫЕ CLAIMS ────────────────────────────
        # Параметры DGP не нужны — данные уже существуют
        # Устанавливаем значения по умолчанию для совместимости

        cfg.n = 500  # Определяется данными
        cfg.seed = 12345
        cfg.contamination = False
        cfg.contamination_probability = 0.0
        cfg.stress_test_mode = False
        cfg.gamma = DEFAULT_GAMMA
        cfg.rho = 0.7
        cfg.delta = DEFAULT_DELTA
        cfg.fs_intercept = 10.0
        cfg.structural_intercept = 10.0
        cfg.fs_z = 0.5
        cfg.fs_z_initial = 0.5
        cfg.baseline_family = "breslow"  # Baseline оценивается из данных
        cfg.baseline_shape = None
        cfg.target_event_rate = 0.02  # Будет переопределён из данных
        cfg.event_definition = "major_claim"
        cfg.competing_risks = True
        cfg.minor_failure_rate = 0.002
        cfg.segment = "light"  # Будет переопределён из данных
        cfg.calib_method = "claims"  # Baseline из данных
        cfg.target_time = CALIBRATION_HORIZON_ENGINE_HOURS
        cfg.target_probability = None  # Определяется данными
        cfg.baseline_hazard_initial = None
        cfg.do_standardize = True
        cfg.do_center_only = False

        print()
        print("Параметры DGP установлены по умолчанию (не используются).")

    # ─── Путь сохранения (зависит от источника данных) ──────────────────────
    if use_claims:
        # Генерация имени файла по конвенции v1.0
        default_path = generate_model_filename("1.0", cfg.segment)
        cfg.out_path = ask(
            "Путь сохранения модели (рекомендуемый: model_params.json)", default_path
        )
    else:
        cfg.out_path = ask(
            "Путь сохранения модели (рекомендуемый: model_params.json)", "model_params.json"
        )

    # ─── CF basis choice ────────────────────────────────────────────────────
    print()
    print("Тип контрольной функции")
    print("1) linear")
    print("2) powers")
    print("3) spline")

    cf_choice = (
        ask(
            "Тип контрольной функции: 1) linear — линейная, 2) powers — степени, 3) spline — сплайн",
            "1",
        ).strip()
        or "1"
    )

    if cf_choice == "2":
        cfg.v_hat_basis = "powers"
        max_power = ask_int("max_power — максимальная степень контрольной функции (1–10)", 2)
        max_power = max(1, min(int(max_power), 10))
        cfg.v_hat_basis_params = {"max_power": max_power}
        if max_power > 4:
            logger.warning("max_power > 4 может привести к переобучению")
    elif cf_choice == "3":
        cfg.v_hat_basis = "spline"
        n_knots = ask_int("n_knots — количество узлов сплайна (1–10)", 2)
        n_knots = max(1, min(int(n_knots), 10))
        cfg.v_hat_basis_params = {"n_knots": n_knots}
        if n_knots > 5:
            logger.warning("n_knots > 5 может привести к переобучению")
    else:
        cfg.v_hat_basis = "linear"
        cfg.v_hat_basis_params = {}

    # ─── Фаза 6.6: выбор источника инструмента ─────────────────────
    # ─── Фаза 6.6: выбор источника инструмента ─────────────────────
    print()
    print("Источник инструмента Z:")
    print("  1) normal         — стандартный N(0,1)")
    print("  2) weather        — синтетический погодный Normal(45, 12)")
    print("  3) weather_real   — реальные данные NASA POWER")
    print("  4) price_bartik   — ценовой Bartik/Shift-Share IV [РЕКОМЕНДУЕТСЯ]")

    instrument_choice = (
        ask(
            "Источник инструмента Z (для устранения смещения PeakLoad): "
            "1) normal, 2) weather, 3) weather_real, 4) price_bartik",
            "4",  # ← Дефолт изменён на 4 (рекомендуемый)
        ).strip()
        or "4"
    )

    instrument_source = "normal"
    weather_campaign = "sowing"
    price_instrument_path = "instrument_z_bartik.csv"

    if instrument_choice == "2":
        instrument_source = "weather"

    elif instrument_choice == "3":
        instrument_source = "weather_real"
        weather_campaign = (
            ask("Кампания (sowing = посевная, harvest = уборочная)", "sowing").strip().lower()
        )
        if weather_campaign not in ("sowing", "harvest"):
            weather_campaign = "sowing"

        # Проверка доступности данных
        weather_path = Path("data/processed/weather/weather_windows.csv")
        if not weather_path.exists():
            logger.warning(
                "weather_windows.csv не найден. "
                "Запустите load_nasa_power.py и compute_working_days.py."
            )
            if ask_yesno("Нет реальных данных: использовать fallback на normal?", True):
                instrument_source = "normal"
            else:
                raise RuntimeError(
                    "Реальные погодные данные недоступны. "
                    "Запустите load_nasa_power.py и compute_working_days.py."
                )
        else:
            logger.info(
                "Используется реальный погодный инструмент (NASA POWER, campaign=%s)",
                weather_campaign,
            )

    # ─── НОВАЯ ВЕТКА: price_bartik ─────────────────────────────────
    elif instrument_choice == "4":
        instrument_source = "price_bartik"
        price_instrument_path_str = (
            ask(
                "Путь к ценовому инструменту",
                "instrument_z_bartik.csv",
            ).strip()
            or "instrument_z_bartik.csv"
        )

        price_instrument_path = price_instrument_path_str
        price_path_obj = Path(price_instrument_path)

        if not price_path_obj.exists():
            logger.warning(
                "Файл ценового инструмента не найден: %s. "
                "Запустите build_instrument_z.py для его создания.",
                price_instrument_path,
            )
            if ask_yesno(
                "Ценовой инструмент недоступен: использовать fallback на normal?",
                True,
            ):
                instrument_source = "normal"
            else:
                raise RuntimeError(
                    f"Ценовой инструмент недоступен: {price_instrument_path}. "
                    f"Запустите build_instrument_z.py."
                )
        else:
            # Валидация файла: проверяем наличие обязательных колонок
            try:
                import pandas as pd

                df_check = pd.read_csv(price_path_obj, encoding="utf-8-sig", nrows=5)
                required_cols = {"region_name", "year", "z_standardized"}
                missing = required_cols - set(df_check.columns)
                if missing:
                    raise ValueError(f"Отсутствуют колонки: {missing}")

                n_rows = len(pd.read_csv(price_path_obj, encoding="utf-8-sig"))
                logger.info(
                    "✅ Ценовой инструмент Bartik загружен: %s (%d записей)",
                    price_instrument_path,
                    n_rows,
                )
            except Exception as exc:
                logger.error("Ошибка валидации ценового инструмента: %s", exc)
                if ask_yesno("Использовать fallback на normal?", True):
                    instrument_source = "normal"
                else:
                    raise RuntimeError(f"Ценовой инструмент невалиден: {exc}") from exc

    else:
        if instrument_choice not in ("1", "2", "3", "4"):
            logger.warning("Неизвестный выбор '%s', используется normal", instrument_choice)

    cfg.instrument_source = instrument_source
    cfg.weather_campaign = weather_campaign
    cfg.price_instrument_path = price_instrument_path

    # ─── Фаза 6.6: источник данных о почве ────────────────────────
    soil_source = "claims"  # По умолчанию берём из claims
    # ─── Фаза 6.6: источник данных о почве ────────────────────────
    soil_source = "claims"  # По умолчанию берём из claims

    if use_claims:
        # Загружаем claims для проверки наличия soil
        claims_path = "data/processed/claims/claims_clean.csv"

        if Path(claims_path).exists():
            claims_df_for_check = pd.read_csv(claims_path, encoding="utf-8")

            if "soil" in claims_df_for_check.columns:
                soil_non_null = claims_df_for_check["soil"].notna().sum()
                logger.info(
                    "Soil в claims: %d непустых значений из %d",
                    soil_non_null,
                    len(claims_df_for_check),
                )
                if soil_non_null < len(claims_df_for_check) * 0.5:
                    logger.warning(
                        "Менее 50%% значений soil заполнены. "
                        "Рекомендуется запустить scripts/enrich_claims_with_soil.py"
                    )
            else:
                logger.warning(
                    "Колонка soil отсутствует в claims. "
                    "Будет использован fallback на Beta(2.0, 2.5). "
                    "Запустите scripts/enrich_claims_with_soil.py для реальных данных."
                )
                soil_source = "synthetic"

            # Освобождаем память
            del claims_df_for_check
        else:
            logger.warning(
                "Файл claims не найден: %s. Soil будет сгенерирован синтетически.",
                claims_path,
            )
            soil_source = "synthetic"
    else:
        # Для симуляции — спрашиваем пользователя
        print()
        print("Источник данных о почве (для симуляции):")
        print("  1) synthetic      — Beta(2.0, 2.5) распределение")
        print("  2) soil_real      — реальные данные GLDAS-2.1")
        soil_choice = (
            ask(
                "Источник soil-данных: 1) synthetic — Beta(2.0, 2.5), 2) soil_real — GLDAS-2.1", "1"
            ).strip()
            or "1"
        )
        if soil_choice == "2":
            soil_source = "soil_real"
            logger.info("Используется реальный soil из GLDAS-2.1 (только для симуляции)")
        else:
            soil_source = "synthetic"

    cfg.soil_source = soil_source

    # ─── Фаза 9: выбор формы модели ─────────────────────────────────────
    print()
    print("Форма эконометрической модели:")
    print("  1) control_function — CF Cox с 2SRI (попытка каузальной коррекции)")
    print("  2) reduced_form     — предиктивная модель (Z как ковариата, без v_hat)")
    print("     Рекомендуется при нарушении exclusion restriction.")
    form_choice = ask("Форма модели (1 = control_function, 2 = reduced_form)", "1").strip() or "1"
    cfg.model_form = {
        "1": "control_function",
        "2": "reduced_form",
    }.get(form_choice, "control_function")

    # Сохраним флаги в конфиг для использования в main()
    cfg._use_claims = use_claims  # type: ignore[attr-defined]
    cfg._claims_path = "data/processed/claims/claims_clean.csv" if use_claims else None  # type: ignore[attr-defined]
    cfg._use_hybrid = use_hybrid  # type: ignore[attr-defined]

    return cfg


def build_dgp_from_config(cfg: TrainingConfig, use_hybrid: bool = False) -> DGPParameters:
    """Создать DGPParameters из TrainingConfig."""
    return construct_dgp(
        {
            "gamma": cfg.gamma,
            "rho": cfg.rho,
            "delta": cfg.delta,
            "intercept": cfg.fs_intercept,
            "structural_intercept": cfg.structural_intercept,
            "first_stage_z_coef": cfg.fs_z,
            "fs_age_coef": 0.15,
            "fs_hours_coef": 0.10,
            "fs_climate_coef": 0.20,
            "fs_soil_coef": 0.15,
            "fs_brand_coef": 0.10,
            "fs_power_coef": 0.08,
            "beta_age": DEFAULT_BETA_AGE,
            "beta_hours": DEFAULT_BETA_HOURS,
            "beta_age_hours": cfg.beta_age_hours,
            "beta_climate": DEFAULT_BETA_CLIMATE,
            "beta_soil": DEFAULT_BETA_SOIL,
            "beta_brand": 0.06,
            "beta_power": DEFAULT_BETA_POWER,
            "baseline_family": cfg.baseline_family,
            "baseline_shape": cfg.baseline_shape,
            "brand_encoding": "dummies",
            "brand_prob_by_code": DEFAULT_BRAND_PROB_BY_CODE,
            "competing_risks": cfg.competing_risks,
            "minor_failure_rate": cfg.minor_failure_rate,
            "event_definition": cfg.event_definition,
            "segment": cfg.segment,
            "peakload_target_mean": cfg.tum_peakload_target_mean,
            "peakload_target_std": cfg.tum_peakload_target_std,
            "weather_campaign": cfg.weather_campaign,
            "soil_source": cfg.soil_source,
            "use_real_covariates": use_hybrid,
        }
    )


def calibrate_model(
    cfg: TrainingConfig,
    dgp: DGPParameters,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Выполнить калибровку baseline hazard и censoring scale.
    Возвращает (baseline_h, censoring_scale, diagnostics).
    """
    print()
    print("Калибровка")
    print("-" * 70)

    baseline_h, calibrated_censoring_scale, baseline_diag = run_calibration(
        calib_method=cfg.calib_method,
        dgp=dgp,
        seed=cfg.seed,
        n=cfg.n,
        target_time=cfg.target_time,
        target_quantile=cfg.target_quantile,
        target_probability=cfg.target_probability,
        target_event_rate=cfg.target_event_rate,
        contamination=cfg.contamination,
        contamination_probability=cfg.contamination_probability,
        baseline_hazard_initial=cfg.baseline_hazard_initial,
        instrument_strength=cfg.instrument_strength,
    )

    validate_positive(baseline_h, "baseline_hazard")
    validate_positive(calibrated_censoring_scale, "censoring_scale")

    print(f"baseline_hazard = {baseline_h}")
    print(f"censoring_scale = {calibrated_censoring_scale}")

    if cfg.calib_method == "probability" and cfg.target_probability is not None:
        final_error = baseline_diag.get("final_error")
        if final_error is not None and final_error > CALIBRATION_TOLERANCE_ABS:
            logger.warning(
                "Calibration final_error %.4f превышает tolerance %.4f",
                float(final_error),
                CALIBRATION_TOLERANCE_ABS,
            )

    return baseline_h, calibrated_censoring_scale, baseline_diag


def generate_training_data(
    cfg: TrainingConfig,
    dgp: DGPParameters,
    baseline_h: float,
    calibrated_censoring_scale: float,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
    float,
    Dict[str, float],
    Dict[str, float],
]:
    """
    Сгенерировать обучающие данные.
    Возвращает (data, data_mod, transform_info, achieved_event_rate,
    peak_stats, training_meta_flat).
    """
    print()
    print("Генерация обучающих данных")
    print("-" * 70)

    data, data_mod, transform_info = generate_and_transform_data(
        n=cfg.n,
        contamination=cfg.contamination,
        contamination_probability=cfg.contamination_probability,
        baseline_hazard=baseline_h,
        censoring_scale=calibrated_censoring_scale,
        seed=cfg.seed,
        dgp=dgp,
        do_standardize=cfg.do_standardize,
        do_center_only=cfg.do_center_only,
        instrument_strength=cfg.instrument_strength,
        instrument_source=getattr(cfg, "instrument_source", "normal"),
        price_instrument_path=getattr(cfg, "price_instrument_path", None),
    )

    achieved_event_rate, peak_stats, training_meta_flat = summarize_data(data)

    # ─── Диагностика инструмента ─────────────────────────────────────────
    print()
    print("Диагностика инструмента Z")
    print("-" * 70)

    Z = data_mod["Z"].values
    PeakLoad = data_mod["PeakLoad"].values
    x_climate = data_mod["x_climate"].values
    x_soil = data_mod["x_soil"].values

    corr_z_pl = np.corrcoef(Z, PeakLoad)[0, 1]
    corr_z_climate = np.corrcoef(Z, x_climate)[0, 1]
    corr_z_soil = np.corrcoef(Z, x_soil)[0, 1]

    print(f"Корреляция Z с PeakLoad:    {corr_z_pl:+.3f}  (должна быть высокой)")
    print(f"Корреляция Z с x_climate:   {corr_z_climate:+.3f}  (должна быть низкой)")
    print(f"Корреляция Z с x_soil:      {corr_z_soil:+.3f}  (должна быть низкой)")

    if abs(corr_z_climate) > 0.3:
        logger.warning(
            "⚠️ Z коррелирует с x_climate (|r|=%.3f > 0.3). "
            "Exclusion restriction может быть нарушен!",
            corr_z_climate,
        )
    else:
        print("ℹ️ Exclusion probe: Z не коррелирует с x_climate; это не тест exclusion restriction.")

    if abs(corr_z_pl) < 0.1:
        logger.warning(
            "⚠️ Z слабо коррелирует с PeakLoad (|r|=%.3f < 0.1). "
            "Instrument relevance может быть нарушен!",
            corr_z_pl,
        )
    else:
        print("✅ Instrument relevance: Z коррелирует с PeakLoad")

    print()
    print(f"Фактическая доля событий: {achieved_event_rate:.4f}")
    if abs(achieved_event_rate - cfg.target_event_rate) > 0.05:
        logger.warning(
            "Achieved event rate %.4f отличается от target_event_rate %.4f",
            achieved_event_rate,
            cfg.target_event_rate,
        )

    return (
        data,
        data_mod,
        transform_info,
        achieved_event_rate,
        peak_stats,
        training_meta_flat,
    )


def fit_first_stage_and_cf(
    cfg: TrainingConfig,
    dgp: DGPParameters,
    data: pd.DataFrame,
    data_mod: pd.DataFrame,
    baseline_h: float,
    calibrated_censoring_scale: float,
    baseline_diag: Dict[str, Any],
    transform_info: Dict[str, Any],
    achieved_event_rate: float,
    peak_stats: Dict[str, float],
    training_meta_flat: Dict[str, float],
) -> Dict[str, Any]:
    """
    Выполнить первую стадию, CF Cox оценку и все диагностики.
    Возвращает словарь со всеми результатами обучения.
    """
    # ─── Первая стадия + IV-диагностика ─────────────────────────────────────
    print()
    print("Первая стадия и IV-диагностика")
    print("-" * 70)

    fitted_fs, resid, fs_names, fs_params, iv_diagnostics = run_first_stage_and_iv(data_mod)

    print(f"F-statistic: {iv_diagnostics['f_statistic']}")
    print(f"Cragg-Donald: {iv_diagnostics['cragg_donald_stat']}")
    print(f"Endogenous: {iv_diagnostics['endogenous']}")
    print(f"Instrument adequate: {iv_diagnostics['instrument_adequate']}")

    # ─── Cluster-robust F-statistic для Z ────────────────────────────────
    try:
        cluster_col = "cluster_id" if "cluster_id" in data_mod.columns else None
        if cluster_col is not None:
            fitted_cr, cr_diag = fit_first_stage_cluster_robust(data_mod, cluster_col=cluster_col)
            iv_diagnostics["f_statistic_cluster_robust"] = cr_diag["f_statistic_cluster_robust"]
            iv_diagnostics["p_value_cluster_robust"] = cr_diag["p_value_cluster_robust"]
            iv_diagnostics["n_clusters"] = cr_diag["n_clusters"]
            iv_diagnostics["pi_z_cluster_robust"] = cr_diag["pi_z"]
            iv_diagnostics["se_pi_z_cluster_robust"] = cr_diag["se_pi_z_cluster"]
            print(f"\nCluster-robust диагностика (n_clusters={cr_diag['n_clusters']})")
            print(f"  F_stat_cluster_robust: {cr_diag['f_statistic_cluster_robust']:.2f}")
            print(f"  p_value_cluster_robust: {cr_diag['p_value_cluster_robust']:.4f}")
            print(
                f"  π_Z (cluster-robust SE): {cr_diag['pi_z']:.4f} ± {cr_diag['se_pi_z_cluster']:.4f}"
            )
            if cr_diag["f_statistic_cluster_robust"] < 10.0:
                logger.warning(
                    "⚠️ Cluster-robust F = %.2f < 10. Инструмент слабый "
                    "с учётом кластерной структуры.",
                    cr_diag["f_statistic_cluster_robust"],
                )
        else:
            logger.info("cluster_id не найден в данных, cluster-robust пропущен.")
            iv_diagnostics["f_statistic_cluster_robust"] = None
            iv_diagnostics["n_clusters"] = None
    except Exception:  # noqa: BLE001
        logger.exception("Cluster-robust диагностика не удалась")
        iv_diagnostics["f_statistic_cluster_robust"] = None
        iv_diagnostics["n_clusters"] = None

    # ─── Частичный F для Z после контроля всех X ────────────────────────────
    try:
        partial_f_z = partial_f_statistic_for_z(fitted_fs.fitted)
    except Exception:  # noqa: BLE001
        partial_f_z = float("nan")

    if math.isfinite(partial_f_z):
        iv_diagnostics["partial_f_z"] = partial_f_z
        logger.debug("Частичный F для Z (после контроля X): %.2f", partial_f_z)
        if partial_f_z < 10.0:
            logger.warning(
                "⚠️ Частичный F для Z = %.2f < 10 (Stock & Yogo, 2005). "
                "Инструмент слабый после контроля X. "
                "Каузальная интерпретация γ невозможна.",
                partial_f_z,
            )
    else:
        iv_diagnostics["partial_f_z"] = None
        logger.warning("Частичный F для Z не удалось вычислить.")

    # ─── Фаза 5.4: IV-режим ─────────────────────────────────────────────────
    iv_mode = determine_iv_mode(
        iv_diagnostics,
        cfg.weather_instrument_validated,
    )
    logger.info("IV-режим: %s", iv_mode)

    # ─── КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Удалена автокоррекция слабого инструмента
    # Автоусиление коэффициента fs_z искусственно изменяет DGP, что разрушает
    # свойства состоятельности IV-оценки. Если инструмент слабый (F < 10.4),
    # модель должна переключаться в predictive режим без попыток "лечения" данных.
    # См. Stock & Yogo (2005) для критических значений F-статистики.
    weak_instrument_fixed = False

    f_weak = iv_diagnostics.get("f_statistic_weak")
    cd_weak = iv_diagnostics.get("cragg_donald_weak")

    needs_strength = (f_weak is True) or (cd_weak is True)

    if needs_strength:
        f_stat_val = iv_diagnostics.get("f_statistic", 0.0)
        logger.warning(
            "⚠️ КРИТИЧЕСКАЯ ОШИБКА ЭКОНОМЕТРИКИ: Слабый инструмент обнаружен "
            "(F-statistic = %.2f < 10.4 по Stock-Yogo). "
            "Каузальная интерпретация коэффициента γ НЕВОЗМОЖНА. "
            "Автоматическая коррекция fs_z удалена как методологически некорректная.",
            f_stat_val,
        )

        # Принудительное переключение в predictive режим
        print()
        print("=" * 70)
        print("ПЕРЕКЛЮЧЕНИЕ В PREDICTIVE РЕЖИМ")
        print("=" * 70)
        print(f"Причина: F-statistic = {f_stat_val:.2f} < 10.4 (Stock-Yogo critical value)")
        print()
        print("ВНИМАНИЕ: Модель может использоваться ТОЛЬКО для предсказаний.")
        print("Каузальные утверждения о влиянии PeakLoad на отказы НЕКОРРЕКТНЫ.")
        print("=" * 70)
        print()

        # Переопределяем iv_mode в predictive
        iv_mode = IV_MODE_PREDICTIVE
        logger.info("IV-режим изменён на '%s' из-за слабого инструмента", iv_mode)

    # ─── CF Cox estimation ──────────────────────────────────────────────────
    print()
    print("Оценка CF Cox модели")
    print("-" * 70)

    cf = fit_cf_cox_compat(
        data=data_mod,
        first_stage=fitted_fs,
        v_hat_basis=cfg.v_hat_basis,
        v_hat_basis_params=cfg.v_hat_basis_params,
    )

    for warning in getattr(cf, "warnings", []) or []:
        logger.warning("[CF Cox WARNING] %s", warning)

    cph_obj = getattr(cf, "cph", None)
    if cph_obj is None:
        raise RuntimeError("fit_cf_cox did not return Cox model")

    params_obj = getattr(cph_obj, "params_", None)
    cox_names = [str(name) for name in _safe_keys(params_obj)]
    cox_coefs = _safe_params_to_float_dict(params_obj)

    std_obj = getattr(cph_obj, "standard_errors_", None)
    cox_ses = _safe_params_to_float_dict(std_obj)

    validate_finite_mapping("cox_coefs", cox_coefs)
    validate_finite_mapping("cox_standard_errors", cox_ses)

    # ─── PH diagnostics ─────────────────────────────────────────────────────
    # ★ v_hat создаётся внутри fit_cf_cox на локальной копии model_data,
    # поэтому data_mod не содержит его. Строим CF-колонки явно.
    from Итог import _build_cf_columns  # type: ignore[attr-defined]

    cf_build = _build_cf_columns(
        residuals=fitted_fs.residuals,
        v_hat_basis=cfg.v_hat_basis,
        v_hat_basis_params=cfg.v_hat_basis_params,
        base_df=data_mod,
    )
    data_mod_with_cf = cf_build.df_with_cf

    ph_diagnostics = run_ph_diagnostics(cf, data_mod_with_cf)
    if ph_diagnostics.get("error"):
        logger.warning("Cox PH diagnostics failed: %s", ph_diagnostics["error"])

    # ─── P0-4: Вывод структурированного PH-отчёта ────────────────────
    if ph_diagnostics.get("available", False):
        print("\n" + "=" * 70)
        print("PH DIAGNOSTICS — Schoenfeld residual test")
        print("=" * 70)
        print(f"\nMethod: {ph_diagnostics.get('method', 'N/A')}")
        print(f"Alpha:  {ph_diagnostics.get('alpha', 0.05)}")
        print(f"N:      {ph_diagnostics.get('n', 'N/A')}")
        print(f"Events: {ph_diagnostics.get('n_events', 'N/A')}")
        print(f"\n{'Variable':<28s} {'test_stat':>10s} {'p-value':>10s} {'verdict':>8s}")
        print("-" * 70)

        variables = ph_diagnostics.get("variables", {})
        for cov_name in sorted(variables.keys()):
            var_info = variables[cov_name]
            ts = var_info.get("test_statistic")
            pv = var_info.get("p_value")
            status = var_info.get("status", "SKIP")
            ts_str = f"{ts:.4f}" if ts is not None else "N/A"
            pv_str = f"{pv:.6f}" if pv is not None else "N/A"
            print(f"  {cov_name:<26s} {ts_str:>10s} {pv_str:>10s} {status:>8s}")

        global_test = ph_diagnostics.get("global_test")
        if global_test is not None:
            g_stat = global_test.get("test_statistic", float("nan"))
            g_p = global_test.get("p_value", float("nan"))
            g_reject = global_test.get("reject_at_alpha", False)
            g_verdict = "FAIL" if g_reject else "PASS"
            print(f"\nGlobal PH verdict: {g_verdict} (stat={g_stat:.4f}, p={g_p:.6f})")

        violations = ph_diagnostics.get("violations", [])
        if violations:
            print(f"Violations: {violations}")
        else:
            print("Violations: none")

        overall_status = ph_diagnostics.get("status", "UNKNOWN")
        print(f"\nOverall status: {overall_status}")
        print("=" * 70 + "\n")

    # ─── Bootstrap SE for generated regressors (Issue #4) ───────────────────
    # The naive SE from lifelines does not account for first-stage uncertainty
    # in the CF Cox model (v_hat is a generated regressor).
    # Bootstrap provides a consistent SE estimate under regularity conditions.
    # ★ FIX 2.3: Включён по умолчанию для корректного учёта
    # неопределённости первой стадии (generated regressor problem).

    bootstrap_se: Optional[Dict[str, float]] = None
    bootstrap_enabled = _dict_get_normalized(
        training_meta_flat,
        "bootstrap_se_enabled",
        True,  # ★ ИЗМЕНЕНО: True по умолчанию
    )

    if _as_bool(bootstrap_enabled, False):
        n_bootstrap = _try_int(_dict_get_normalized(training_meta_flat, "bootstrap_se_n", 50), 50)
        bootstrap_seed = _try_int(
            _dict_get_normalized(training_meta_flat, "bootstrap_se_seed", 42), 42
        )

        print()
        print("Bootstrap SE for generated regressors")
        print("-" * 70)
        print(f"n_bootstrap = {n_bootstrap}, seed = {bootstrap_seed}")

        # ─── Автоподбор n_jobs ───────────────────────────────────────
        import os as _os

        _n_cpus = _os.cpu_count() or 1
        _bootstrap_jobs = min(8, max(1, _n_cpus - 2))

        try:
            bootstrap_se = _bootstrap_cox_se(
                data_mod=data_mod,
                v_hat_basis=cfg.v_hat_basis,
                v_hat_basis_params=cfg.v_hat_basis_params,
                cox_names=cox_names,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
                n_jobs=_bootstrap_jobs,
            )
            print(f"✅ Bootstrap SE computed for {len(bootstrap_se)} coefficients")
        except Exception as exc:
            logger.warning("Bootstrap SE failed: %s. Using naive SE.", exc)
            bootstrap_se = None
    else:
        logger.info(
            "Bootstrap SE disabled. Set training_meta['bootstrap_se_enabled']=True "
            "and training_meta['bootstrap_se_n']=N to enable. "
            "Naive SE does not account for generated-regressor uncertainty."
        )

    # ─── Baseline serialization ─────────────────────────────────────────────
    baseline_hazard_obj = getattr(cph_obj, "baseline_cumulative_hazard_", None)
    baseline_cumulative_hazard = serialize_baseline(baseline_hazard_obj)
    validate_cox_baseline(baseline_cumulative_hazard)

    # ─── Параметрическая подгонка базового риска (v3.1) ─────────────────────
    baseline_spec_dict: Dict[str, Any] = {"family": "breslow"}
    parametric_family = str(getattr(cfg, "parametric_baseline_fit", "weibull")).lower()
    if HAS_PARAMETRIC_BASELINE and parametric_family in VALID_PARAMETRIC_FAMILIES:
        try:
            spec_obj = fit_parametric_baseline(
                breslow_times=np.asarray(baseline_cumulative_hazard["times"], dtype=float),
                breslow_values=np.asarray(baseline_cumulative_hazard["values"], dtype=float),
                family=parametric_family,
            )
            baseline_spec_dict = spec_obj.to_dict()
            logger.info(
                "✅ Параметрический базовый риск подогнан: family=%s, R²(log)=%.4f",
                baseline_spec_dict["family"],
                baseline_spec_dict.get("fit_r2", float("nan")),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Параметрическая подгонка базового риска не удалась (%s). Используется Breslow.",
                exc,
            )
            baseline_spec_dict = {"family": "breslow"}
    elif parametric_family not in ("none", "breslow", ""):
        logger.warning(
            "parametric_baseline_fit='%s' запрошен, но parametric_baseline.py "
            "недоступен. Используется Breslow.",
            parametric_family,
        )

    # ─── ДИАГНОСТИКА COX-МОДЕЛИ ─────────────────────────────────────────
    # После обучения Cox — выводим коэффициенты и честную интерпретацию.
    # УБРАНО: расчёт γ + λ как каузального эффекта (лишён смысла в нелинейной Cox).

    print("\n" + "=" * 70)
    print("ДИАГНОСТИКА COX-МОДЕЛИ")
    print("=" * 70)

    print("\nКоэффициенты Cox:")
    for name in sorted(cox_coefs.keys()):
        se = cox_ses.get(name, float("nan"))
        print(f"  {name:28s} = {cox_coefs[name]:+.6f}  (SE={se:.4f})")

    # Вместо γ + λ — честная интерпретация отдельных компонентов:
    print("\nИнтерпретация:")
    print("  γ (PeakLoad) — предсказательный эффект PeakLoad")
    print("  λ (v_hat) — эффект эндогенности (control function)")
    print("  В нелинейной модели Cox γ + λ НЕ является каузальным эффектом")
    print("  Для каузальной интерпретации требуется Monte Carlo recovery test")

    # ─── P0-3: LR test для Age × Hours interaction ────────────────────
    try:
        interaction_lr = interaction_lr_test(
            data=data_mod,
            first_stage=fitted_fs,
            opts=_DEFAULT_CF_OPTIONS,
            interaction_col="x_age_hours",
        )
    except Exception as exc:
        logger.warning("interaction_lr_test failed: %s", exc)
        interaction_lr = {"note": f"interaction_lr_test failed: {exc}"}

    print("=" * 70)
    print("LR TEST: Age × Hours Interaction (P0-3)")
    print("=" * 70)
    if interaction_lr.get("lr_stat") is not None:
        print(f"  LR statistic:   {interaction_lr['lr_stat']:.4f}")
        print(f"  df:             {interaction_lr['df']}")
        p_val = interaction_lr.get("p_value")
        if p_val is not None:
            print(f"  p-value:        {p_val:.6f}")
        else:
            print(f"  p-value:        N/A ({interaction_lr.get('note', '')})")
        beta_hat = interaction_lr.get("beta_hat")
        se_hat = interaction_lr.get("se_hat")
        if beta_hat is not None:
            print(f"  β_interaction:  {beta_hat:+.6f}")
        if se_hat is not None:
            print(f"  SE:             {se_hat:.6f}")
        aic_f = interaction_lr.get("aic_full")
        aic_r = interaction_lr.get("aic_restricted")
        if aic_f is not None and aic_r is not None:
            print(f"  AIC full:       {aic_f:.1f}")
            print(f"  AIC restricted: {aic_r:.1f}")
        print(f"  Penalized:      {interaction_lr.get('penalized', False)}")
    else:
        print(f"  ⚠️ LR test unavailable: {interaction_lr.get('note', 'unknown')}")
    print("=" * 70 + "\n")

    max_baseline_time = float(baseline_cumulative_hazard["times"][-1])
    if cfg.target_time > max_baseline_time:
        logger.warning(
            "target_time %.1f больше последней точки baseline %.1f.",
            float(cfg.target_time),
            max_baseline_time,
        )

    # ─── Template covariates ────────────────────────────────────────────────
    template_dict = build_raw_template_covariates(data)

    # ─── Partial-out fields ─────────────────────────────────────────────────
    (
        partial_out_all_betas,
        training_x_means,
        training_pl_hat_mean,
        partial_out_X_beta,
        training_x_mean,
    ) = compute_partial_out_fields(fitted_fs, data_mod, cf)

    # ─── CF metadata ────────────────────────────────────────────────────────
    cf_basis_metadata = prepare_cf_basis_metadata(
        cf,
        resid,
        cfg.v_hat_basis,
        cfg.v_hat_basis_params,
    )

    return {
        "fitted_fs": fitted_fs,
        "resid": resid,
        "fs_names": fs_names,
        "fs_params": fs_params,
        "iv_diagnostics": iv_diagnostics,
        "iv_mode": iv_mode,
        "weak_instrument_fixed": weak_instrument_fixed,
        "cf": cf,
        "cox_names": cox_names,
        "cox_coefs": cox_coefs,
        "cox_ses": cox_ses,
        "ph_diagnostics": ph_diagnostics,
        "baseline_cumulative_hazard": baseline_cumulative_hazard,
        "baseline_spec": baseline_spec_dict,  # ← НОВОЕ
        "template_dict": template_dict,
        "partial_out_all_betas": partial_out_all_betas,
        "training_x_means": training_x_means,
        "training_pl_hat_mean": training_pl_hat_mean,
        "partial_out_X_beta": partial_out_X_beta,
        "training_x_mean": training_x_mean,
        "cf_basis_metadata": cf_basis_metadata,
        # Обновлённые при автокоррекции
        "data": data,
        "data_mod": data_mod,
        "transform_info": transform_info,
        "achieved_event_rate": achieved_event_rate,
        "peak_stats": peak_stats,
        "training_meta_flat": training_meta_flat,
        "baseline_h": baseline_h,
        "calibrated_censoring_scale": calibrated_censoring_scale,
        "baseline_diag": baseline_diag,
        "interaction_lr": interaction_lr,
        "interaction_lr_test": interaction_lr,
        "bootstrap_se": bootstrap_se,
    }


def build_model_artifact(
    cfg: TrainingConfig,
    dgp: DGPParameters,
    fit: Dict[str, Any],
    transform_info: Dict[str, Any],
    achieved_event_rate: float,
    peak_stats: Dict[str, float],
    training_meta_flat: Dict[str, float],
    use_claims: bool = False,
    use_hybrid: bool = False,
    claims_path: Optional[str] = None,
) -> Any:
    """
    Собрать training_meta и ModelParameters.
    Возвращает ModelParameters.
    """
    dgp_meta = _dataclass_to_dict(dgp)

    calibration_target_probability: Optional[float] = (
        float(cfg.target_probability) if cfg.target_probability is not None else None
    )

    # Определить версию модели в зависимости от источника данных
    if use_claims:
        model_version_str = "1.0"  # v1.x = real claims
        model_semantic_version = "1.0"
    else:
        model_version_str = MODEL_FORMAT_VERSION  # v0.x = simulation
        model_semantic_version = MODEL_SEMANTIC_VERSION

    # ─── Interaction: простое центрирование (Aiken & West, 1991) ────────
    # Вычисляем ЗДЕСЬ, чтобы все 4 параметра были доступны ниже
    # и в training_meta, и в interaction_params.
    data_df = fit["data"]
    if "x_age" in data_df.columns and "x_hours" in data_df.columns:
        x_age_arr = data_df["x_age"].astype(float).to_numpy()
        x_hours_arr = data_df["x_hours"].astype(float).to_numpy()

        # Простое центрирование по выборочным средним
        x_age_mean = float(np.mean(x_age_arr))
        x_hours_mean = float(np.mean(x_hours_arr))
        x_age_hours_centered = (x_age_arr - x_age_mean) * (x_hours_arr - x_hours_mean)

        # Стандартизация interaction для численной стабильности
        x_age_hours_mean = float(np.mean(x_age_hours_centered))
        x_age_hours_std = float(np.std(x_age_hours_centered, ddof=1))
        if x_age_hours_std < 1e-9:
            x_age_hours_std = 1.0
    else:
        # Fallback, если колонок нет (не должно происходить)
        x_age_mean = 0.0
        x_hours_mean = 0.0
        x_age_hours_mean = 0.0
        x_age_hours_std = 1.0

    training_meta: Dict[str, Any] = {
        "model_version": model_version_str,
        "model_semantic_version": model_semantic_version,
        "data_source": (
            "literary_claims"
            if use_claims
            else "hybrid_dgp_real_covariates"
            if use_hybrid
            else "simulation"
        ),
        "claims_path": claims_path if use_claims else None,
        "use_real_covariates": use_hybrid,
        "weather_data_path": "data/processed/weather/weather_windows.csv" if use_hybrid else None,
        "soil_data_path": "data/processed/soil/soil_windows.csv" if use_hybrid else None,
        "engine_convention": ENGINE_CONVENTION,
        "n": int(cfg.n),
        "seed": int(cfg.seed),
        "created": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": get_dependency_versions(),
        "dgp": dgp_meta,
        "contamination": bool(cfg.contamination),
        "contamination_probability": float(cfg.contamination_probability),
        "stress_test_mode": bool(cfg.stress_test_mode),
        "baseline_family": cfg.baseline_family,
        "baseline_shape": cfg.baseline_shape,
        "baseline_hazard": (float(fit["baseline_h"]) if fit["baseline_h"] is not None else None),
        "censoring_scale": (
            float(fit["calibrated_censoring_scale"])
            if fit["calibrated_censoring_scale"] is not None
            else None
        ),
        "event_rate": achieved_event_rate,
        "target_event_rate": float(cfg.target_event_rate),
        "baseline_calibration": fit["baseline_diag"],
        "censoring_distortion": {
            "target_event_rate": float(cfg.target_event_rate),
            "achieved_event_rate": float(achieved_event_rate),
        },
        "peakload_min": training_meta_flat["peakload_min"],
        "peakload_max": training_meta_flat["peakload_max"],
        "peakload_p25": training_meta_flat["peakload_p25"],
        "peakload_median": training_meta_flat["peakload_median"],
        "peakload_p75": training_meta_flat["peakload_p75"],
        "transform": transform_info,
        "calibration_time_horizon": (
            float(cfg.target_time) if not use_claims else float(CALIBRATION_HORIZON_ENGINE_HOURS)
        ),
        "calibration_time_horizon_days": float(
            cfg.target_time / DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY
        ),
        "time_unit": MODEL_TIME_UNIT,
        "mtbf_input_unit": MTBF_INPUT_UNIT,
        "mtbf_to_model_time_factor": float(MTBF_TO_MODEL_TIME_FACTOR),
        "default_engine_hours_per_calendar_day": float(DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY),
        # ─── Форма модели (v3.1: Reduced Form) ────────────────────────
        "model_form": str(getattr(cfg, "model_form", "control_function")).lower(),
        "event_definition": cfg.event_definition,
        "competing_risks": bool(cfg.competing_risks),
        "minor_failure_rate": float(cfg.minor_failure_rate),
        "segment": cfg.segment,
        "segment_power_stats": None,  # Заполняется ниже
        "freq_shares": dict(FREQ_SHARES),
        "severity_weights": dict(SEVERITY_WEIGHTS),
        "rf_heavy_brand_catalog": RF_HEAVY_BRAND_CATALOG,
        "mtbf_baseline_hours": float(MTBF_BASELINE_HOURS),
        "downtime_model": DOWNTIME_MODEL,
        "power_segment_threshold": float(POWER_SEGMENT_THRESHOLD),
        "kaplan_meier_validator_enabled": bool(KAPLAN_MEIER_VALIDATOR_ENABLED),
        "tum_peakload_calibration": {
            "enabled": cfg.tum_peakload_target_mean is not None,
            "target_mean": cfg.tum_peakload_target_mean,
            "target_std": cfg.tum_peakload_target_std,
            "source": (str(cfg.tum_stats_path) if Path(cfg.tum_stats_path).exists() else None),
        },
        "calibration_target_probability": calibration_target_probability,
        "calibration_method": str(cfg.calib_method),
        "iv_diagnostics": fit["iv_diagnostics"],
        "iv_mode": fit["iv_mode"],
        "interaction_lr_test": fit.get("interaction_lr_test"),
        "iv_mode_candidates": sorted(VALID_IV_MODES),
        "weather_instrument_validated": bool(cfg.weather_instrument_validated),
        "iv_baseline": {
            "f_statistic": fit["iv_diagnostics"].get("f_statistic"),
            "partial_r2": None,
            "cragg_donald": fit["iv_diagnostics"].get("cragg_donald_stat"),
            "n": int(len(fit["data_mod"])),
            "monitored": True,
            "drift_relative_drop_threshold": 0.30,
        },
        "iv_baseline_partial_r2": None,
        "instrument_strength_initial": float(cfg.fs_z_initial),
        "instrument_strength_final": float(cfg.fs_z),
        "weak_instrument_auto_corrected": bool(fit["weak_instrument_fixed"]),
        "weak_instrument_fixed": bool(fit["weak_instrument_fixed"]),
        "n_events": int(fit["data_mod"]["event"].astype(int).sum()),
        "ph_diagnostics": fit["ph_diagnostics"],
        "x_standardization": {
            name: {
                "raw_col": info.get("raw_col"),
                "shift": info.get("shift"),
                "scale": info.get("scale"),
            }
            for name, info in X_STANDARDIZATION.items()
        },
        "brand_mapping": BRAND_TO_CODE,
        "brand_map": BRAND_MAP,
        "brand_encoding": dgp_meta.get("brand_encoding", "dummies"),
        "brand_prob_by_code": DEFAULT_BRAND_PROB_BY_CODE,
        "covariate_mapping": COVARIATE_MAPPING,
        "climate_index_reference": CLIMATE_INDEX_REFERENCE,
        "soil_index_reference": SOIL_INDEX_REFERENCE,
        "soil_source": cfg.soil_source,
        "soil_data_path": (
            "data/processed/soil/soil_windows.csv" if cfg.soil_source == "soil_real" else None
        ),

        # ─── НОВОЕ: метаданные ценового инструмента ────────────────────────
        "instrument_type": (
            "price_bartik"
            if cfg.instrument_source == "price_bartik"
            else "weather"
            if cfg.instrument_source in ("weather", "weather_real")
            else "synthetic"
        ),
        "price_instrument": (
            {
                "source_file": cfg.price_instrument_path,
                "basket_crops": ["wheat_total", "barley_total", "maize_grain", "sunflower_grain"],
                "baseline_period": "2015-2017",
                "price_lag_years": 1,
                "n_regions": 11,
                "excluded_regions": [
                    "Тверская область",
                    "Волгоградская область",
                    "Псковская область",
                    "Ленинградская область",
                    "Амурская область",
                ],
            }
            if cfg.instrument_source == "price_bartik"
            else None
        ),

        "regions_mis": [
            "Поволжье",
            "Северный Кавказ",
            "Северо-Запад",
            "Центрально-Чернозёмная зона",
            "Кубань",
            "Алтай",
            "Амурская область",
            "Владимирский регион",
        ],
        "region_usage": "stratification_only",
        "region_note": (
            "Не используется как трактор-уровневая ковариата до получения первичных протоколов МИС."
        ),
        "climate_soil_validation_status": "unvalidated_simulation",
        "climate_soil_note": ("Beta-распределения, не валидированы на реальных данных."),
        "cox_peakload_convention": "observed_peakload",
        "residual_policy_production": "plug-in",
        "allow_diagnostic_residual_policies": False,
        "linear_cf_standardized": bool(fit["cf_basis_metadata"].get("linear_standardized", True)),
        "cf_basis": fit["cf_basis_metadata"],
        # Production default: never extrapolate the fitted baseline hazard.
        # Research-only extrapolation must be enabled explicitly by a caller.
        "allow_baseline_extrapolation": False,
        # ─── Параметрический базовый риск (v3.1) ──────────────────────
        "baseline_spec": fit.get("baseline_spec", {"family": "breslow"}),
        # Issue #4: Generated regressors — naive SE does not account for
        # first-stage uncertainty. Bootstrap SE is computed if
        # fit.get("bootstrap_se") is not None (set during fit_first_stage_and_cf).
        "bootstrap_se_enabled": fit.get("bootstrap_se") is not None,
        "bootstrap_se_n": 200,
        # Safety net: allow_assumed_time_unit prevents InvalidInputError
        # when time_horizon_unit is not explicitly provided to the engine.
        # Default to False — callers must specify time_horizon_unit.
        "allow_assumed_time_unit": False,
        # Issue #7: γ (PeakLoad coefficient) refers to STANDARDIZED PeakLoad.
        # The model standardizes PeakLoad to mean=0, std=1 during training.
        # Therefore γ represents the effect of a ONE-STANDARD-DEVIATION
        # change in PeakLoad, NOT a one-unit change.
        # For interpretation: effect of Δ units in raw PeakLoad ≈
        #   γ * (Δ / training_std_of_PeakLoad)
        "peakload_interpretation_note": (
            "Cox coefficient for PeakLoad (γ) refers to STANDARDIZED "
            "PeakLoad (mean=0, std=1). It represents the log-hazard ratio "
            "for a one-standard-deviation increase in PeakLoad, not a "
            "one-unit increase. To compute the effect of Δ raw units: "
            "log_HR ≈ γ * (Δ / training_std)."
        ),
        "interaction_lr_test": fit.get("interaction_lr"),
        # ─── Interaction: параметры центрирования (Aiken & West, 1991) ───
        # Все 4 числа нужны prediction_engine для инференса.
        "interaction_params": {
            "x_age_mean": x_age_mean,
            "x_hours_mean": x_hours_mean,
            "x_age_hours_mean": x_age_hours_mean,
            "x_age_hours_std": x_age_hours_std,
        },
        # Дублируем для обратной совместимости со старым ключом
        "interaction_centering": {
            "x_age_mean": x_age_mean,
            "x_hours_mean": x_hours_mean,
            "x_age_hours_mean": x_age_hours_mean,
            "x_age_hours_std": x_age_hours_std,
        },
    }

    # Заполнить segment_power_stats
    data = fit["data"]
    if "Power" in data.columns:
        training_meta["segment_power_stats"] = {
            "segment": cfg.segment,
            "power_mean": float(np.mean(data["Power"])),
            "power_std": float(np.std(data["Power"])),
            "power_min": float(np.min(data["Power"])),
            "power_max": float(np.max(data["Power"])),
        }

    # ─── P0-5: Bayesian major failure calibration ──────────────────
    # Вычисляем ДО формирования training_meta, чтобы posterior был доступен.
    _major_calib = _compute_major_failure_calibration(
        data=data,
        prior_mean=MAJOR_FAILURE_SHARE_PRIOR["mean"],
        prior_effective_n=MAJOR_FAILURE_SHARE_PRIOR["effective_n"],
    )
    _major_overall = _major_calib["overall"]
    _major_posterior_mean = float(_major_overall["posterior_mean"])

    training_meta["major_failure_share"] = (
        _major_posterior_mean  # production value = posterior mean
    )
    training_meta["major_failure_share_observed"] = float(_major_overall["observed_share"])
    training_meta["major_failure_share_posterior"] = _major_overall
    training_meta["major_failure_share_by_brand"] = _major_calib["by_brand"]
    training_meta["major_failure_share_by_brand_observed"] = _major_calib["by_brand_observed"]
    training_meta["major_failure_share_by_brand_posterior"] = _major_calib["by_brand_posterior"]
    training_meta["major_failure_share_prior"] = MAJOR_FAILURE_SHARE_PRIOR

    # Сохранить partial R² для мониторинга дрейфа
    try:
        from Итог import partial_r2_from_fitted

        _pr2 = partial_r2_from_fitted(fit["fitted_fs"].fitted)
        if np.isfinite(_pr2):
            training_meta["iv_baseline"]["partial_r2"] = float(_pr2)
            training_meta["iv_baseline_partial_r2"] = float(_pr2)
    except Exception:
        pass

    # ─── ModelParameters construction ───────────────────────────────────────
    fitted_values_arr = np.asarray(
        fit["fitted_fs"].fitted.fittedvalues,
        dtype=float,
    )
    residuals_arr = np.asarray(fit["resid"], dtype=float)

    # ─── Reduced Form: обработка training_residuals_std ──────────────────
    model_form = str(getattr(cfg, "model_form", "control_function")).lower()
    if model_form == "reduced_form":
        # В Reduced Form нет остатков первой стадии → std = 1.0 (заглушка)
        training_residuals_std_val = 1.0
    else:
        training_residuals_std_val = max(float(np.std(residuals_arr, ddof=1)), 1e-12)

    model_kwargs: Dict[str, Any] = {
        "model_version": model_version_str,
        "model_semantic_version": model_semantic_version,
        "prediction_engine_version": ENGINE_CONVENTION,
        "metadata": default_metadata(),
        "transform_info": transform_info,
        "first_stage": {
            "exog_names": [str(name) for name in fit["fs_names"]],
            "params": fit["fs_params"],
        },
        "cox": {
            "exog_names": fit["cox_names"],
            "coefs": fit["cox_coefs"],
            "standard_errors": fit["cox_ses"],
            # Bootstrap SE for generated regressors (Issue #4).
            # If None, only naive SE is available.
            "bootstrap_se": fit.get("bootstrap_se"),
        },
        "baseline_cumulative_hazard": fit["baseline_cumulative_hazard"],
        "baseline_spec": fit.get("baseline_spec", {"family": "breslow"}),
        "template_covariates": {
            str(name): float(value) for name, value in fit["template_dict"].items()
        },
        "partial_out_X_beta": fit["partial_out_X_beta"],
        "training_pl_hat_mean": fit["training_pl_hat_mean"],
        "training_x_mean": fit["training_x_mean"],
        "training_x_means": fit["training_x_means"],
        "partial_out_all_betas": fit["partial_out_all_betas"],
        "training_residuals_mean": float(np.mean(residuals_arr)),
        "training_residuals_std": training_residuals_std_val,
        "training_first_stage_fitted": fitted_values_arr.tolist(),
        "training_residuals_arr": residuals_arr.tolist(),
        "training_meta": training_meta,
        "cf_basis_metadata": fit["cf_basis_metadata"],
        "calibration_time_horizon": float(cfg.target_time),
        "brand_mapping": BRAND_TO_CODE,
        "brand_effects": {},
        # ─── НОВОЕ: семантика события на верхнем уровне ──────────────
        # Унифицировано с training_meta для устранения рассогласования.
        "competing_risks": bool(cfg.competing_risks),
        "event_definition": str(cfg.event_definition),
        "minor_failure_rate": float(cfg.minor_failure_rate),
        "segment": str(cfg.segment),
    }

    model_params = construct_model_parameters(model_kwargs)
    validate_model(model_params)
    validate_event_semantic_consistency(model_params)  # ← НОВОЕ
    return model_params


def validate_save_and_smoke_test(
    cfg: TrainingConfig,
    model_params: Any,
    data: pd.DataFrame,
    template_dict: Dict[str, float],
    peak_stats: Dict[str, float],
) -> Any:
    """
    Post-fit проверка, сохранение модели, post-save валидация и smoke test.
    Возвращает загруженную модель (loaded_params).
    """
    # ─── Post-fit probability check ─────────────────────────────────────────
    post_fit_avg = estimate_post_fit_average_probability(
        model_params=model_params,
        data=data,
        target_time=cfg.target_time,
        template_dict=template_dict,
        seed=cfg.seed,
        max_rows=100,
    )

    if post_fit_avg is not None:
        model_params.training_meta["post_fit_average_probability"] = float(post_fit_avg)

        print()
        print("-" * 70)
        print("Marginal probability check:")
        print(f"  mean P = {float(post_fit_avg):.6f}")

        if cfg.target_probability is not None:
            post_fit_error = abs(float(post_fit_avg) - float(cfg.target_probability))
            model_params.training_meta["post_fit_target_error"] = float(post_fit_error)

            print(f"  target P = {float(cfg.target_probability):.6f}")
            print(f"  |error|  = {post_fit_error:.6f}")

            if post_fit_error > POST_FIT_FATAL_TOLERANCE:
                print(f"  ❌ FATAL: отклонение > {POST_FIT_FATAL_TOLERANCE}")
                return None

            if post_fit_error > POST_FIT_TIGHT_TOLERANCE:
                print(f"  ⚠️  WARNING: отклонение > {POST_FIT_TIGHT_TOLERANCE}")
            else:
                print(f"  ✅ OK: в пределах tolerance {POST_FIT_TIGHT_TOLERANCE}")
    else:
        logger.warning("Post-fit probability check unavailable")

    # ─── Save ───────────────────────────────────────────────────────────────
    # Save model artifact with campaign provenance
    try:
        save_model_params(cfg.out_path, model_params)
    except (OSError, TypeError, ValueError, AttributeError, RuntimeError) as exc:
        logger.error("Failed to save model: %s", exc)
        return None

    # Persist training_meta with weather_campaign provenance via json.dump
    try:
        meta_to_save = model_params.training_meta if hasattr(model_params, "training_meta") else {}
        if meta_to_save is None:
            meta_to_save = {}
        provenance_data = {"training_meta": meta_to_save}
        out_path_obj = Path(cfg.out_path)
        provenance_path = out_path_obj.with_name(out_path_obj.stem + "_training_meta.json")
        provenance_path.write_text(
            json.dumps(provenance_data, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        logger.info("training_meta saved to %s", provenance_path)

        # Apply campaign normalization if available
        if normalize_model_campaign_metadata is not None:
            try:
                model_json_for_norm = {
                    "training_meta": meta_to_save,
                    "metadata": getattr(model_params, "metadata", {}),
                    "weather_campaign": getattr(cfg, "weather_campaign", None),
                }
                normalize_model_campaign_metadata(model_json_for_norm)
            except Exception:
                logger.debug("normalize_model_campaign_metadata skipped")
    except Exception:
        logger.debug("training_meta provenance save skipped")

    print()
    print(f"Модель сохранена: {cfg.out_path}")

    # ─── Post-save validation + smoke test ──────────────────────────────────
    try:
        loaded_params = load_model_params(cfg.out_path)
        validate_model(loaded_params)
        print("Post-save validation: OK")

        test_peak = peak_stats["median"]
        test_prob = predict_probability(
            loaded_params,
            test_peak,
            float(cfg.target_time),
            residual_policy="plug-in",
            covariates=dict(template_dict),
            time_horizon_unit=MODEL_TIME_UNIT,
            strict_covariates=True,
        )

        if not math.isfinite(test_prob) or not (0.0 <= test_prob <= 1.0):
            raise RuntimeError("Smoke-test probability invalid")

        print(f"Smoke-test prediction: P(T <= {cfg.target_time:.0f} мч) = {test_prob:.6f}")

    except (OSError, TypeError, ValueError, AttributeError, RuntimeError) as exc:
        logger.error("Post-save validation/smoke-test failed: %s", exc)
        return None

    return loaded_params


def run_post_training_diagnostics(
    cfg: TrainingConfig,
    loaded_params: Any,
    data: pd.DataFrame,
    fit: Dict[str, Any],
) -> None:
    """K-M валидация и финальные предупреждения."""
    # ─── P-12: Kaplan-Meier validation ──────────────────────────────────────
    if KAPLAN_MEIER_VALIDATOR_ENABLED:
        print()
        print("-" * 70)
        print("Kaplan-Meier валидация baseline survival:")

        km_result = run_kaplan_meier_validation(
            model_params=loaded_params,
            data=data,
            target_time=cfg.target_time,
        )

        if km_result is not None:
            km_surv = km_result.get("km_survival", float("nan"))
            model_surv = km_result.get("model_survival", float("nan"))
            abs_diff = km_result.get("abs_diff", float("nan"))

            print(f"  K-M survival:    S({cfg.target_time:.0f}) = {km_surv:.6f}")
            print(f"  Model survival:  S0({cfg.target_time:.0f}) = {model_surv:.6f}")
            print(f"  |diff|:          {abs_diff:.6f}")
            print(f"  n_events:        {km_result.get('n_events', 0)}")

            loaded_params.training_meta["kaplan_meier_validation"] = km_result

            if abs_diff > 0.15:
                logger.warning("K-M abs_diff %.4f превышает порог 0.15", abs_diff)
            else:
                print(f"  ✅ OK: abs_diff < 0.15")
        else:
            print("  ⚠️  Kaplan-Meier валидация недоступна")
            loaded_params.training_meta["kaplan_meier_validation"] = None

    # ─── Final warnings ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("ИТОГОВЫЕ ПРЕДУПРЕЖДЕНИЯ")
    print("=" * 70)

    if cfg.stress_test_mode:
        print("🚨 Модель обучена в stress-test режиме.")
        print("   НЕ использовать для продуктового ценообразования.")

    model_form = str(fit.get("iv_diagnostics", {}).get("model_form", "control_function"))
    if model_form == "reduced_form":
        print("ℹ️  Reduced Form: IV-диагностика не применяется (нет первой стадии).")
    elif not fit["iv_diagnostics"].get("instrument_adequate", False):
        print("⚠️  Инструмент слабый или диагностика неполная.")

    brand_encoding_used = loaded_params.training_meta.get(
        "brand_encoding",
        "legacy_continuous",
    )
    if str(brand_encoding_used).lower() == "legacy_continuous":
        print("⚠️  Brand encoding: legacy_continuous.")
    else:
        print(f"✅ Brand encoding: {brand_encoding_used}")

    _sps = loaded_params.training_meta.get("segment_power_stats")
    if _sps:
        _seg = _sps.get("segment")
        _pmin, _pmax = _sps.get("power_min"), _sps.get("power_max")

        _ok = (
            (50.0 <= _pmin and _pmax <= 350.0)
            if _seg == "light"
            else (200.0 <= _pmin and _pmax <= 500.0)
        )
        _mark = "✅" if _ok else "⚠️ "

        print(f"{_mark} Сегмент '{_seg}': мощность mean={_sps.get('power_mean'):.1f}")

    # ─── P0-5: Bayesian major failure calibration ───────────────────
    _mfs_post = loaded_params.training_meta.get("major_failure_share")
    _mfs_obs = loaded_params.training_meta.get("major_failure_share_observed")
    _mfs_posterior_info = loaded_params.training_meta.get("major_failure_share_posterior", {})

    if _mfs_post is not None and _mfs_posterior_info:
        _ci_lo = _mfs_posterior_info.get("ci_low", float("nan"))
        _ci_hi = _mfs_posterior_info.get("ci_high", float("nan"))
        _n_ev = _mfs_posterior_info.get("n_events", 0)
        _n_maj = _mfs_posterior_info.get("n_major", 0)
        print(f"✅ MAJOR_FAILURE_SHARE = {_mfs_post:.4f} (Bayesian posterior)")
        print(f"   Prior: Beta(9, 21), mean=0.30, effective_n=30")
        print(f"   Data: {_n_maj} major / {_n_ev} total events")
        print(f"   Observed share: {_mfs_obs:.4f}")
        print(f"   Posterior 95% CrI: [{_ci_lo:.4f}, {_ci_hi:.4f}]")
    else:
        print(f"⚠️  MAJOR_FAILURE_SHARE = {MAJOR_FAILURE_SHARE} (🟡 expert assumption).")

    iv_mode = fit["iv_mode"]
    if iv_mode == IV_MODE_CAUSAL:
        print(f"✅ IV-режим: {iv_mode}")
    else:
        print(f"⚠️  IV-режим: {iv_mode}")
        print("   γ интерпретируется как предсказательный, не каузальный.")

    print()
    print("Обучение завершено успешно.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    setup_logging()
    validate_constants()

    # 1. Сбор параметров
    cfg = collect_training_config()

    # Получаем флаги использования claims и гибридного режима
    use_claims = getattr(cfg, "_use_claims", False)
    use_hybrid = getattr(cfg, "_use_hybrid", False)
    claims_path_val = getattr(cfg, "_claims_path", None)

    # 2. Создание DGP (только для симуляции или гибрида)
    dgp = build_dgp_from_config(cfg, use_hybrid=use_hybrid) if not use_claims else None

    # 3. Калибровка и генерация данных
    if use_claims and claims_path_val is not None:
        # ─── ВЕТКА: литературные claims ──────────────────────────
        print()
        print("Загрузка литературных claims")
        print("-" * 70)

        claims_df = load_claims_for_training(claims_path_val)
        data_mod = prepare_claims_for_cf(claims_df)
        data = claims_df  # Для совместимости с post-fit проверками

        # Transform info
        transform_info = {
            "type": "standardize" if cfg.do_standardize else "none",
            "mean": float(data_mod["PeakLoad"].mean()),
            "std": float(data_mod["PeakLoad"].std()),
        }

        # Метаданные из claims
        achieved_event_rate = float(data_mod["event"].mean())
        peak_stats = {
            "min": float(data_mod["PeakLoad"].min()),
            "max": float(data_mod["PeakLoad"].max()),
            "median": float(data_mod["PeakLoad"].median()),
            "p25": float(data_mod["PeakLoad"].quantile(0.25)),
            "p75": float(data_mod["PeakLoad"].quantile(0.75)),
        }
        training_meta_flat = {
            "peakload_min": peak_stats["min"],
            "peakload_max": peak_stats["max"],
            "peakload_p25": peak_stats["p25"],
            "peakload_median": peak_stats["median"],
            "peakload_p75": peak_stats["p75"],
        }

        # Baseline hazard не калибруется — будет оценён из данных
        baseline_h = None
        censoring_scale = None
        baseline_diag = {"method": "claims", "source": claims_path_val}

        print(f"Загружено наблюдений: {len(data_mod)}")
        print(f"Событий: {int(data_mod['event'].sum())}")
        print(f"Event rate: {achieved_event_rate:.4f}")

    else:
        # ─── ВЕТКА: симуляция DGP ────────────────────────────────
        baseline_h, censoring_scale, baseline_diag = calibrate_model(
            cfg,
            dgp,
        )

        (
            data,
            data_mod,
            transform_info,
            achieved_event_rate,
            peak_stats,
            training_meta_flat,
        ) = generate_training_data(
            cfg,
            dgp,
            baseline_h,
            censoring_scale,
        )

    # 5. Первая стадия + CF Cox
    if use_claims and claims_path_val is not None:
        # ─── ВЕТКА: claims ─────────────────────────────────────────
        print()
        print("Первая стадия на литературных claims")
        print("-" * 70)

        fitted_fs, resid, fs_names, fs_params, iv_diagnostics = run_first_stage_on_claims(data_mod)
        print(f"F-statistic: {iv_diagnostics['f_statistic']:.2f}")
        print(f"Partial R²: {iv_diagnostics['partial_r2']:.4f}")
        print(f"Instrument adequate: {iv_diagnostics['instrument_adequate']}")

        # IV-режим
        iv_mode = determine_iv_mode(
            iv_diagnostics,
            cfg.weather_instrument_validated,
        )
        logger.info("IV-режим: %s", iv_mode)

        # CF Cox на claims
        print()
        print("Оценка CF Cox на литературных claims")
        print("-" * 70)

        cf = fit_cf_cox_on_claims(
            data_mod,
            fitted_fs,
            cfg.v_hat_basis,
            cfg.v_hat_basis_params,
        )

        # Извлечь результаты
        cph_obj = cf.cph
        params_obj = cph_obj.params_
        cox_names = [str(name) for name in params_obj.index]
        cox_coefs = {str(k): float(v) for k, v in params_obj.items()}
        std_obj = cph_obj.standard_errors_
        cox_ses = {str(k): float(v) for k, v in std_obj.items()}

        # PH diagnostics
        # ★ Строим v_hat на data_mod для корректной PH-диагностики
        from Итог import _build_cf_columns  # type: ignore[attr-defined]

        cf_build2 = _build_cf_columns(
            residuals=fitted_fs.residuals,
            v_hat_basis=cfg.v_hat_basis,
            v_hat_basis_params=cfg.v_hat_basis_params,
            base_df=data_mod,
        )
        ph_diagnostics = run_ph_diagnostics(cf, cf_build2.df_with_cf)

        # Baseline serialization
        baseline_hazard_obj = cph_obj.baseline_cumulative_hazard_
        baseline_cumulative_hazard = serialize_baseline(baseline_hazard_obj)
        validate_cox_baseline(baseline_cumulative_hazard)

        # ─── Параметрическая подгонка базового риска (v3.1) ─────────────────────
        baseline_spec_dict: Dict[str, Any] = {"family": "breslow"}
        parametric_family = str(getattr(cfg, "parametric_baseline_fit", "weibull")).lower()
        if HAS_PARAMETRIC_BASELINE and parametric_family in VALID_PARAMETRIC_FAMILIES:
            try:
                spec_obj = fit_parametric_baseline(
                    breslow_times=np.asarray(baseline_cumulative_hazard["times"], dtype=float),
                    breslow_values=np.asarray(baseline_cumulative_hazard["values"], dtype=float),
                    family=parametric_family,
                )
                baseline_spec_dict = spec_obj.to_dict()
                logger.info(
                    "✅ Параметрический базовый риск подогнан: family=%s, R²(log)=%.4f",
                    baseline_spec_dict["family"],
                    baseline_spec_dict.get("fit_r2", float("nan")),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Параметрическая подгонка базового риска не удалась (%s). "
                    "Используется Breslow.",
                    exc,
                )
                baseline_spec_dict = {"family": "breslow"}
        elif parametric_family not in ("none", "breslow", ""):
            logger.warning(
                "parametric_baseline_fit='%s' запрошен, но parametric_baseline.py "
                "недоступен. Используется Breslow.",
                parametric_family,
            )

        # Template covariates
        template_dict = build_raw_template_covariates(data)

        # Partial-out fields
        (
            partial_out_all_betas,
            training_x_means,
            training_pl_hat_mean,
            partial_out_X_beta,
            training_x_mean,
        ) = compute_partial_out_fields(fitted_fs, data_mod, cf)

        # CF metadata
        cf_basis_metadata = prepare_cf_basis_metadata(
            cf,
            resid,
            cfg.v_hat_basis,
            cfg.v_hat_basis_params,
        )

        fit = {
            "fitted_fs": fitted_fs,
            "resid": resid,
            "fs_names": fs_names,
            "fs_params": fs_params,
            "iv_diagnostics": iv_diagnostics,
            "iv_mode": iv_mode,
            "weak_instrument_fixed": False,
            "cf": cf,
            "cox_names": cox_names,
            "cox_coefs": cox_coefs,
            "cox_ses": cox_ses,
            "ph_diagnostics": ph_diagnostics,
            "baseline_cumulative_hazard": baseline_cumulative_hazard,
            "baseline_spec": baseline_spec_dict,  # ← НОВОЕ
            "template_dict": template_dict,
            "partial_out_all_betas": partial_out_all_betas,
            "training_x_means": training_x_means,
            "training_pl_hat_mean": training_pl_hat_mean,
            "partial_out_X_beta": partial_out_X_beta,
            "training_x_mean": training_x_mean,
            "cf_basis_metadata": cf_basis_metadata,
            "data": data,
            "data_mod": data_mod,
            "baseline_h": baseline_h,
            "calibrated_censoring_scale": censoring_scale,
            "baseline_diag": baseline_diag,
            "transform_info": transform_info,
            "achieved_event_rate": achieved_event_rate,
            "peak_stats": peak_stats,
            "training_meta_flat": training_meta_flat,
            "interaction_lr_test": None,
        }
    else:
        # ─── ВЕТКА: симуляция ──────────────────────────────────────
        if str(getattr(cfg, "model_form", "control_function")).lower() == "reduced_form":
            fit = fit_reduced_form_pipeline(
                cfg,
                dgp,
                data,
                data_mod,
                baseline_h,
                censoring_scale,
                baseline_diag,
                transform_info,
                achieved_event_rate,
                peak_stats,
                training_meta_flat,
            )
        else:
            fit = fit_first_stage_and_cf(
                cfg,
                dgp,
                data,
                data_mod,
                baseline_h,
                censoring_scale,
                baseline_diag,
                transform_info,
                achieved_event_rate,
                peak_stats,
                training_meta_flat,
            )

    # Обновить ссылки (могли измениться при автокоррекции)
    data = fit["data"]
    data_mod = fit["data_mod"]
    transform_info = fit["transform_info"]
    achieved_event_rate = fit["achieved_event_rate"]
    peak_stats = fit["peak_stats"]
    training_meta_flat = fit["training_meta_flat"]
    baseline_h = fit["baseline_h"]
    censoring_scale = fit["calibrated_censoring_scale"]
    baseline_diag = fit["baseline_diag"]

    # 6. Сборка модели
    model_params = build_model_artifact(
        cfg,
        dgp if dgp is not None else build_dgp_from_config(cfg),
        fit,
        transform_info,
        achieved_event_rate,
        peak_stats,
        training_meta_flat,
        use_claims=use_claims,
        use_hybrid=use_hybrid,
        claims_path=claims_path_val,
    )

    # 7. Валидация, сохранение, smoke test
    loaded_params = validate_save_and_smoke_test(
        cfg,
        model_params,
        data,
        fit["template_dict"],
        peak_stats,
    )

    if loaded_params is None:
        return 1

    # 8. Пост-обучающая диагностика
    # Для K-M валидации используем data_mod (с колонками time/event),
    # а не оригинальный data (с колонками failure_time/event_flag)
    run_post_training_diagnostics(cfg, loaded_params, fit["data_mod"], fit)

    return 0


def validate_event_semantic_consistency(model_params: Any) -> None:
    """
    Критическая проверка: семантика события на верхнем уровне
    должна совпадать с training_meta. Любое рассогласование —
    блокирующая ошибка, потому что это создаёт две разные истины
    в одном артефакте модели.
    """
    top_cr = bool(getattr(model_params, "competing_risks", False))
    top_ed = str(getattr(model_params, "event_definition", "major_claim"))
    top_mfr = float(getattr(model_params, "minor_failure_rate", 0.002))
    top_seg = str(getattr(model_params, "segment", "light"))

    tm = getattr(model_params, "training_meta", {}) or {}
    meta_cr = bool(tm.get("competing_risks", top_cr))
    meta_ed = str(tm.get("event_definition", top_ed))
    meta_mfr = float(tm.get("minor_failure_rate", top_mfr))
    meta_seg = str(tm.get("segment", top_seg))

    errors = []
    if top_cr != meta_cr:
        errors.append(f"competing_risks mismatch: top={top_cr}, training_meta={meta_cr}")
    if top_ed != meta_ed:
        errors.append(f"event_definition mismatch: top={top_ed!r}, training_meta={meta_ed!r}")
    if abs(top_mfr - meta_mfr) > 1e-9:
        errors.append(f"minor_failure_rate mismatch: top={top_mfr}, training_meta={meta_mfr}")
    if top_seg != meta_seg:
        errors.append(f"segment mismatch: top={top_seg!r}, training_meta={meta_seg!r}")

    if errors:
        raise RuntimeError(
            "Event semantic consistency violation in model artifact:\n  " + "\n  ".join(errors)
        )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Прервано пользователем.")
        sys.exit(130)
