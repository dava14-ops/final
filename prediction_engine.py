"""
prediction_engine.py (v3.2.1-fixed)

Prediction engine for CF Cox / IV-Cox models.

Fixed and extended version:
- load_model_params() validates model by default;
- explicit time_horizon_unit control;
- baseline extrapolation is disabled unless explicitly allowed;
- strict covariate resolution;
- non-finite coefficients/values raise errors instead of silent zero fallback;
- CF policy "zero" sets CF contribution to zero;
- spline basis metadata is validated more strictly;
- dictionary keys and model column names are normalized;
- bool metadata is parsed safely;
- baseline validation is centralized;
- climate/soil resolution is centralized;
- static-analysis fixes:
    * safe string conversion for None / np.integer / np.floating;
    * lowercase local variables;
    * no float(list) hazard;
    * no direct relative import;
    * safe scipy BSpline access;
    * chained comparisons;
- v0.2:
    * P-01: competing risks metadata layer;
    * P-02: x_hours standardized around LogNormal median 1000;
    * P-03: event_definition control;
    * P-04: major_failure_share Beta-prior helpers;
    * P-05: FREQ_SHARES and SEVERITY_WEIGHTS;
    * P-06: Weibull shape 1.88 kept as default;
    * P-07: RF heavy brand catalog data layer;
    * P-08: MTBF baseline helper;
    * P-09: downtime by MTTR helper;
    * P-10: power segment helper;
    * P-12: Kaplan-Meier validator.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import math
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional scipy for spline basis evaluation
# ---------------------------------------------------------------------------
try:
    from scipy import interpolate as _scipy_interpolate

    HAS_SCIPY = hasattr(_scipy_interpolate, "BSpline")
except ImportError:
    _scipy_interpolate = None
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
# Fallback exception classes. If an external exceptions module is available,
# it will be loaded dynamically below. This avoids direct relative imports.
class InvalidInputError(ValueError):
    pass


class ModelLoadError(ValueError):
    pass


class ModelValidationError(ValueError):
    pass


class PredictionError(ValueError):
    pass


def _load_exception_module():
    """
    Dynamically load exceptions module without using direct relative import.
    """
    if __package__:
        try:
            return importlib.import_module(".exceptions", package=__package__)
        except ImportError:
            pass

    try:
        return importlib.import_module("exceptions")
    except ImportError:
        return None


_exception_module = _load_exception_module()

if _exception_module is not None:
    _loaded_invalid_input = getattr(_exception_module, "InvalidInputError", None)
    _loaded_model_load = getattr(_exception_module, "ModelLoadError", None)
    _loaded_model_validation = getattr(_exception_module, "ModelValidationError", None)
    _loaded_prediction = getattr(_exception_module, "PredictionError", None)

    if isinstance(_loaded_invalid_input, type) and issubclass(
        _loaded_invalid_input, BaseException
    ):
        InvalidInputError = _loaded_invalid_input

    if isinstance(_loaded_model_load, type) and issubclass(
        _loaded_model_load, BaseException
    ):
        ModelLoadError = _loaded_model_load

    if isinstance(_loaded_model_validation, type) and issubclass(
        _loaded_model_validation, BaseException
    ):
        ModelValidationError = _loaded_model_validation

    if isinstance(_loaded_prediction, type) and issubclass(
        _loaded_prediction, BaseException
    ):
        PredictionError = _loaded_prediction


logger = logging.getLogger(__name__)
__version__ = "3.2.1-fixed"

__all__ = [
    "__version__",
    "ModelParameters",
    "load_model_params",
    "save_model_params",
    "validate_model",
    "default_metadata",
    "transform_peak",
    "baseline_cumulative_hazard",
    "predict_first_stage",
    "compute_pl_hat_exog",
    "predict_probability",
    "predict_many",
    "build_cf_basis_at_prediction",
    "engine_hours_to_calendar_days",
    "calendar_days_to_engine_hours",
    "BRAND_MAP",
    "BRAND_TO_CODE",
    "CLIMATE_INDEX_REFERENCE",
    "SOIL_INDEX_REFERENCE",
    "VALID_EVENT_DEFINITIONS",
    "RF_HEAVY_BRAND_CATALOG",
    "FREQ_SHARES",
    "SEVERITY_WEIGHTS",
    "CRITICALITY_WEIGHTS",
    "get_criticality_weights",
    "DEFAULT_WEIBULL_SHAPE",
    "MTBF_BASELINE_HOURS",
    "POWER_SEGMENT_THRESHOLDS",
    "kaplan_meier_check",
    "get_mtbf_baseline_hours",
    "get_downtime_hours",
    "classify_power_segment",
    "get_freq_shares",
    "get_severity_weights",
]


# ---------------------------------------------------------------------------
# Constants (Imported from centralized constants.py)
# ---------------------------------------------------------------------------
from constants import (
    MODEL_TIME_UNIT,
    DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY,
    CALIBRATION_HORIZON_DAYS,
    CALIBRATION_HORIZON_ENGINE_HOURS,
    MAJOR_FAILURE_SHARE,
    DEFAULT_WEIBULL_SHAPE,
    VALID_EVENT_DEFINITIONS,
    RF_HEAVY_BRAND_CATALOG,
    FREQ_SHARES,
    CRITICALITY_WEIGHTS,
    SEVERITY_WEIGHTS,
    MTBF_BASELINE_HOURS,
    DEFAULT_MTTR_HOURS,
    DEFAULT_DOWNTIME_PER_MTTR_FACTOR,
    POWER_SEGMENT_THRESHOLDS,
    BRAND_MAP,
    BRAND_TO_CODE,
    BRAND_ALIASES,
    CLIMATE_INDEX_REFERENCE,
    SOIL_INDEX_REFERENCE,
)

# Локальные константы, специфичные только для prediction_engine
MAX_BATCH_SIZE = 10_000
MAX_LP = 700.0
MIN_LP = -700.0
MAX_LOG_CH = 700.0
MIN_LOG_CH = -37.0
PEAK_RANGE_ABS_TOLERANCE = 5.0
PEAK_RANGE_REL_TOLERANCE = 1e-6
ENFORCE_PEAK_RANGE = True
PROBABILITY_EPSILON = 1e-12
ALLOWED_PRODUCTION_RESIDUAL_POLICIES = {"plug-in"}
ALLOWED_DIAGNOSTIC_RESIDUAL_POLICIES = {"mean", "zero"}
DEFAULT_COX_PEAKLOAD_CONVENTION = "observed_peakload"
TIME_UNITS = {"days", "hours", "engine_hours"}
MTBF_INPUT_UNIT = "engine_hours"
MTBF_TO_MODEL_TIME_FACTOR = 1.0


# ---------------------------------------------------------------------------
# Fallback X standardization
# ---------------------------------------------------------------------------
X_STANDARDIZATION_FALLBACK: Dict[str, Dict[str, Any]] = {
    "x_age": {
        "raw_col": "Age",
        "shift": 10.0,
        "scale": 10.0,
    },
    "x_hours": {
        "raw_col": "Hours",
        "shift": 1000.0,  # P-02: медиана LogNormal(1000)
        "scale": 1000.0,
    },
    # FIX 1: PeakLoad normalized to [0, 1] range matching TUM CAN bus.
    "PeakLoad": {
        "raw_col": "PeakLoad",
        "shift": 0.55,
        "scale": 0.15,
    },
    "x_climate": {
        "raw_col": "Climate",
        "shift": None,
        "scale": None,
    },
    "x_soil": {
        "raw_col": "Soil",
        "shift": None,
        "scale": None,
    },
    "x_brand": {
        "raw_col": "Brand",
        "shift": 2.0,
        "scale": 2.0,
    },
    "x_power": {
        "raw_col": "Power",
        "shift": 200.0,
        "scale": 150.0,
    },
}

_RAW_ALIASES: Dict[str, List[str]] = {
    "x_age": [
        "Age",
        "age",
        "age_years",
        "x_age",
    ],
    "x_hours": [
        "Hours",
        "hours",
        "hours_annual",
        "annual_hours",
        "hours_per_year",
        "x_hours",
    ],
    "x_climate": [
        "Climate",
        "climate",
        "climate_index",
        "climate_factor",
        "x_climate",
    ],
    "x_soil": [
        "Soil",
        "soil",
        "soil_index",
        "soil_factor",
        "x_soil",
    ],
    "x_brand": [
        "Brand",
        "brand",
        "brand_code",
        "brand_name",
        "x_brand",
    ],
    "x_power": [
        "Power",
        "power",
        "power_hp",
        "x_power",
    ],
}


# ---------------------------------------------------------------------------
# Safe formatting / parsing helpers
# ---------------------------------------------------------------------------
def _to_text(value: Any) -> str:
    """
    Safe conversion to stripped string.

    Handles None, numpy scalars and unprintable objects without raising.
    """
    if value is None:
        return ""

    if isinstance(value, np.integer):
        return str(int(value))

    if isinstance(value, np.floating):
        return str(float(value))

    if isinstance(value, np.bool_):
        return str(bool(value))

    try:
        return str(value).strip()
    except (TypeError, ValueError):
        return ""


def _fmt(value: Any) -> str:
    """
    Safe formatting for error messages.
    Avoids static-analysis complaints about np.integer / np.floating / None.
    """
    if value is None:
        return "None"

    if isinstance(value, np.integer):
        return str(int(value))

    if isinstance(value, np.floating):
        return repr(float(value))

    if isinstance(value, np.bool_):
        return str(bool(value))

    try:
        return repr(value)
    except (TypeError, ValueError):
        return f"<unprintable {type(value).__name__}>"


def _normalize_name(name: Any) -> str:
    return _to_text(name).lower()


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return default
        value = value.item()

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}

    return default


def _try_float_optional(value: Any) -> Optional[float]:
    """
    Convert value to finite float if possible.
    Returns None instead of raising.
    """
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        return None

    if isinstance(value, (list, tuple, set, dict)):
        return None

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return None
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
    elif isinstance(value, str):
        try:
            out = float(value.strip())
        except (TypeError, ValueError):
            return None
    else:
        return None

    if not math.isfinite(out):
        return None

    return out


def _try_float(value: Any, default: float) -> float:
    """
    Convert value to finite float if possible, otherwise return default.
    """
    out = _try_float_optional(value)
    return default if out is None else out


def _try_int_optional(value: Any) -> Optional[int]:
    """
    Convert value to int if possible.
    Returns None instead of raising.
    """
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        return None

    if isinstance(value, (list, tuple, set, dict)):
        return None

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return None
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            return None

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        value_float = float(value)
        if not math.isfinite(value_float):
            return None
        if not value_float.is_integer():
            return None
        return int(value_float)

    if isinstance(value, str):
        try:
            return int(value.strip())
        except (TypeError, ValueError):
            return None

    return None


def _try_int(value: Any, default: int) -> int:
    """
    Convert value to int if possible, otherwise return default.
    """
    out = _try_int_optional(value)
    return default if out is None else out


def _dict_get_normalized(d: Any, key: Any, default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default

    target = _normalize_name(key)

    for k, v in d.items():
        if _normalize_name(k) == target:
            return v

    return default


def _get_field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return _dict_get_normalized(obj, name)
    return getattr(obj, name, None)


def _get_training_meta(params: Any) -> Dict[str, Any]:
    meta = _get_field(params, "training_meta")
    if isinstance(meta, dict):
        return meta
    return {}


def as_finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise InvalidInputError(f"{name} cannot be boolean")

    value_float = _try_float_optional(value)

    if value_float is None:
        raise InvalidInputError(
            f"{name} must be finite numeric, got {_fmt(value)}"
        )

    return value_float


_as_finite_float = as_finite_float
_coerce_float = as_finite_float


def _lower_dict_keys(d: Optional[Dict[Any, Any]]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}

    return {_to_text(k).lower(): v for k, v in d.items()}


def _get_source_value(
    params: Any,
    covariates: Optional[Dict[str, Any]],
    aliases: Sequence[str],
) -> Any:
    """
    Search covariates first, then template_covariates.
    Values explicitly set to None are ignored.
    """
    cov_lower = _lower_dict_keys(covariates)

    template = _get_field(params, "template_covariates")
    template_lower = _lower_dict_keys(template)

    for source in (cov_lower, template_lower):
        for alias in aliases:
            key = _normalize_name(alias)
            if key in source and source[key] is not None:
                return source[key]

    return None


def default_metadata() -> Dict[str, Any]:
    return {
        "created": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "prediction_engine_version": __version__,
    }


# ---------------------------------------------------------------------------
# Time unit helpers
# ---------------------------------------------------------------------------
def _normalize_time_unit(unit: Any) -> str:
    unit_text = _to_text(unit).lower()

    if unit_text in {"engine_hours", "engine_hour", "mch", "моточасы"}:
        return "engine_hours"

    if unit_text in {"hours", "hour", "h"}:
        return "hours"

    if unit_text in {"days", "day", "d", "calendar_days"}:
        return "days"

    raise InvalidInputError(
        f"Unknown time unit: {_fmt(unit)}. "
        "Expected one of: 'days', 'hours', 'engine_hours'."
    )


def _get_time_unit(params: Any) -> str:
    meta = _get_training_meta(params)
    unit_raw = _dict_get_normalized(meta, "time_unit", MODEL_TIME_UNIT)
    return _normalize_time_unit(unit_raw)


def _get_hours_per_day(params: Any) -> float:
    meta = _get_training_meta(params)

    raw = _dict_get_normalized(
        meta,
        "default_engine_hours_per_calendar_day",
        DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY,
    )

    hpd = _try_float(raw, DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY)

    if hpd > 0.0:
        return hpd

    return DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY


def _convert_time_horizon(
    value: float,
    from_unit: str,
    to_unit: str,
    hours_per_day: float,
) -> float:
    from_unit = _normalize_time_unit(from_unit)
    to_unit = _normalize_time_unit(to_unit)

    if from_unit == to_unit:
        return float(value)

    # Treat "hours" and "engine_hours" as equivalent model-time units.
    if from_unit in {"hours", "engine_hours"} and to_unit in {"hours", "engine_hours"}:
        return float(value)

    if from_unit == "days":
        return float(value) * float(hours_per_day)

    if to_unit == "days":
        return float(value) / float(hours_per_day)

    return float(value)


def _resolve_time_horizon(
    params: Any,
    time_horizon: Any,
    time_horizon_unit: Optional[str],
) -> float:
    horizon = _as_finite_float(time_horizon, "time_horizon")
    model_unit = _get_time_unit(params)

    if time_horizon_unit is None:
        meta = _get_training_meta(params)
        allow_assumed = _as_bool(
            _dict_get_normalized(meta, "allow_assumed_time_unit", False),
            False,
        )

        if allow_assumed:
            logger.warning(
                "time_horizon_unit was not specified; assuming model unit '%s'.",
                model_unit,
            )
            return horizon

        raise InvalidInputError(
            "time_horizon_unit must be specified explicitly. "
            "Expected one of: 'days', 'hours', 'engine_hours'."
        )

    from_unit = _normalize_time_unit(time_horizon_unit)
    hours_per_day = _get_hours_per_day(params)

    converted = _convert_time_horizon(
        horizon,
        from_unit,
        model_unit,
        hours_per_day,
    )

    if not math.isfinite(converted):
        raise InvalidInputError("Converted time_horizon is not finite")

    return float(converted)


def engine_hours_to_calendar_days(
    engine_hours: float,
    hours_per_day: float = DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY,
) -> float:
    if hours_per_day <= 0.0:
        return float(engine_hours)

    return float(engine_hours) / float(hours_per_day)


def calendar_days_to_engine_hours(
    calendar_days: float,
    hours_per_day: float = DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY,
) -> float:
    if hours_per_day <= 0.0:
        return float(calendar_days)

    return float(calendar_days) * float(hours_per_day)


# ---------------------------------------------------------------------------
# Model parameter container
# ---------------------------------------------------------------------------
@dataclass
class ModelParameters:
    """
    Runtime model container.
    """

    model_version: str = "3.0"
    model_semantic_version: str = "0.2"
    prediction_engine_version: str = __version__

    metadata: Dict[str, Any] = field(default_factory=dict)
    transform_info: Dict[str, Any] = field(default_factory=dict)
    first_stage: Dict[str, Any] = field(default_factory=dict)
    cox: Dict[str, Any] = field(default_factory=dict)
    baseline_cumulative_hazard: Dict[str, Any] = field(default_factory=dict)
    template_covariates: Dict[str, Any] = field(default_factory=dict)

    partial_out_X_beta: float = 0.0
    training_pl_hat_mean: float = 0.0
    training_x_mean: float = 0.0
    training_x_means: Dict[str, float] = field(default_factory=dict)
    partial_out_all_betas: Dict[str, float] = field(default_factory=dict)
    training_residuals_mean: float = 0.0
    training_residuals_std: float = 1.0
    training_first_stage_fitted: Any = field(default_factory=list)
    training_residuals_arr: Any = field(default_factory=list)
    training_meta: Dict[str, Any] = field(default_factory=dict)
    cf_basis_metadata: Optional[Dict[str, Any]] = None
    calibration_time_horizon: float = CALIBRATION_HORIZON_ENGINE_HOURS

    brand_mapping: Optional[Dict[Any, Any]] = None
    brand_effects: Optional[Dict[Any, float]] = None

    # P-03 / D1
    event_definition: str = "major_claim"
    competing_risks: bool = False
    minor_failure_rate: float = 0.002
    segment: str = "light"
    allow_baseline_extrapolation: bool = False

    def __post_init__(self) -> None:
        current = _try_float(self.calibration_time_horizon, 0.0)

        if current <= 0.0:
            meta = (
                self.training_meta
                if isinstance(self.training_meta, dict)
                else {}
            )

            meta_horizon = _dict_get_normalized(
                meta,
                "calibration_time_horizon",
            )

            self.calibration_time_horizon = _try_float(meta_horizon, 0.0)


# ---------------------------------------------------------------------------
# X standardization helpers
# ---------------------------------------------------------------------------
def _get_raw_aliases(name: str) -> List[str]:
    target = _normalize_name(name)

    for key, aliases in _RAW_ALIASES.items():
        if _normalize_name(key) == target:
            return list(aliases)

    return []


def _get_x_standardization_table(params: Any) -> Dict[str, Dict[str, Any]]:
    meta = _get_training_meta(params)
    x_std = _dict_get_normalized(meta, "x_standardization", {})

    if isinstance(x_std, dict):
        return x_std

    return {}


def _get_x_std_info(params: Any, name: str) -> Dict[str, Any]:
    table = _get_x_standardization_table(params)
    info = _dict_get_normalized(table, name)

    if isinstance(info, dict):
        return info

    fallback = _dict_get_normalized(X_STANDARDIZATION_FALLBACK, name)

    if isinstance(fallback, dict):
        # PATCH-14: предупреждение при использовании fallback
        logger.warning(
            "Используется FALLBACK стандартизация для '%s'. "
            "training_meta.x_standardization отсутствует. "
            "Возможен train-serve skew.",
            name,
        )
        return fallback

    return {}


def _standardize_x_value(params: Any, name: str, raw_value: float) -> float:
    info = _get_x_std_info(params, name)

    shift = info.get("shift")
    scale = info.get("scale")

    if shift is None or scale is None:
        return float(raw_value)

    shift_f = _as_finite_float(shift, f"x_standardization[{name}].shift")
    scale_f = _as_finite_float(scale, f"x_standardization[{name}].scale")

    if scale_f <= 0.0:
        raise ModelValidationError(
            f"x_standardization[{name}].scale must be positive, got {scale_f}"
        )

    return (float(raw_value) - shift_f) / scale_f


def validate_index_value(
    value: Any,
    name: str,
    reference: Dict[str, float],
) -> float:
    """
    Validate normalized index in [0, 1].
    Accepts numeric values or known categorical labels.
    """
    if isinstance(value, (bool, np.bool_)):
        raise InvalidInputError(f"{name} cannot be boolean")

    if isinstance(value, str):
        key = value.strip().lower()

        if key in reference:
            value = reference[key]
        else:
            try:
                value = float(key.replace(",", "."))
            except (TypeError, ValueError) as exc:
                raise InvalidInputError(
                    f"Unknown categorical value for {name}: {_fmt(value)}. "
                    "Use a normalized index in [0, 1]."
                ) from exc

    value = _as_finite_float(value, name)

    if not (-1e-9 <= value <= 1.0 + 1e-9):
        raise InvalidInputError(
            f"{name} must be a normalized index in [0, 1], got {value}"
        )

    return float(min(1.0, max(0.0, value)))


_validate_index_value = validate_index_value


# ---------------------------------------------------------------------------
# Brand helpers
# ---------------------------------------------------------------------------
def _get_model_brand_to_code(params: Any) -> Dict[str, int]:
    candidates: List[Any] = []

    brand_mapping = _get_field(params, "brand_mapping")
    if isinstance(brand_mapping, dict):
        candidates.append(brand_mapping)

    meta = _get_training_meta(params)

    meta_mapping = _dict_get_normalized(meta, "brand_mapping")
    meta_to_code = _dict_get_normalized(meta, "brand_to_code")

    if isinstance(meta_mapping, dict):
        candidates.append(meta_mapping)

    if isinstance(meta_to_code, dict):
        candidates.append(meta_to_code)

    result: Dict[str, int] = {}

    for mapping in candidates:
        if not isinstance(mapping, dict):
            continue

        for k, v in mapping.items():
            if isinstance(k, (bool, np.bool_)) or isinstance(v, (bool, np.bool_)):
                continue

            # code -> name
            if isinstance(k, (int, np.integer)):
                code = _try_int_optional(k)
                name_text = _to_text(v)

                if code is not None and name_text and 0 <= code <= 4:
                    result[name_text.lower()] = code
                    result[name_text] = code

                continue

            # name -> code
            code = _try_int_optional(v)

            if code is None:
                code_name = _to_text(v)
                code = BRAND_TO_CODE.get(
                    code_name,
                    BRAND_TO_CODE.get(code_name.lower()),
                )

            if code is None:
                continue

            if 0 <= code <= 4:
                key_text = _to_text(k)

                if key_text:
                    result[key_text.lower()] = int(code)
                    result[key_text] = int(code)

    return result


def coerce_brand_code(params: Any, value: Any) -> int:
    if value is None:
        raise InvalidInputError("Brand value is required but missing")

    if isinstance(value, (bool, np.bool_)):
        raise InvalidInputError("Brand cannot be boolean")

    if isinstance(value, (int, np.integer, float, np.floating)):
        code_float_opt = _try_float_optional(value)

        if code_float_opt is None:
            raise InvalidInputError("Brand code must be finite")

        code_float = code_float_opt
        code = int(round(code_float))

        if abs(code_float - code) > 1e-9:
            raise InvalidInputError(
                f"Brand code must be an integer in [0, 4], got {_fmt(value)}"
            )

        if not 0 <= code <= 4:
            raise InvalidInputError(
                f"Brand code must be in [0, 4], got {code}"
            )

        return code

    text = _to_text(value)

    if not text:
        raise InvalidInputError("Brand value is empty")

    model_mapping = _get_model_brand_to_code(params)

    if text in model_mapping:
        return int(model_mapping[text])

    if text.lower() in model_mapping:
        return int(model_mapping[text.lower()])

    code = _try_int_optional(text)

    if code is not None and 0 <= code <= 4:
        return code

    key = text.lower()

    if key in BRAND_ALIASES:
        return int(BRAND_ALIASES[key])

    key_compact = key.replace(" ", "").replace("-", "")

    for alias, alias_code in BRAND_ALIASES.items():
        alias_key = _to_text(alias).lower()
        alias_compact = alias_key.replace(" ", "").replace("-", "")

        if alias_code == code and alias_compact == key_compact:
            return int(alias_code)

    if text in BRAND_TO_CODE:
        return int(BRAND_TO_CODE[text])

    if text.lower() in BRAND_TO_CODE:
        return int(BRAND_TO_CODE[text.lower()])

    raise InvalidInputError(
        f"Unknown brand value: {_fmt(value)}. "
        "Provide a canonical brand code 0..4 or a known brand name."
    )


def name_is_brand_related(name: str) -> bool:
    lower = _normalize_name(name)

    if lower in {
        "brand",
        "x_brand",
        "brand_code",
        "brand_effect",
        "brand_effects",
    }:
        return True

    if lower.startswith("brand"):
        return True

    canonical_dummy_names = {
        _to_text(v).lower() for v in BRAND_MAP.values() if _to_text(v).lower() != "other"
    }

    if lower in canonical_dummy_names:
        return True

    return False


def _name_is_brand_related(name: str) -> bool:
    return name_is_brand_related(name)


def _brand_value_for_model_name(params: Any, name: str, brand_code: int) -> float:
    lower = _normalize_name(name)
    canonical = _to_text(BRAND_MAP.get(int(brand_code), "Other")).lower()

    # Brand-specific effect.
    if lower in {"brand_effect", "brand_effects"}:
        effects = _get_field(params, "brand_effects")

        if effects is None:
            meta = _get_training_meta(params)
            effects = _dict_get_normalized(meta, "brand_effects", {})

        if not isinstance(effects, dict):
            raise InvalidInputError(
                "brand_effects is required for brand_effect model column"
            )

        keys_to_try = [
            brand_code,
            str(brand_code),
            canonical,
            canonical.lower(),
            BRAND_MAP.get(int(brand_code), "Other"),
        ]

        for key in keys_to_try:
            val = _dict_get_normalized(effects, key)

            if val is not None:
                return _as_finite_float(val, f"brand_effects[{_fmt(key)}]")

        raise InvalidInputError(
            f"brand_effects does not contain an effect for brand_code={brand_code}"
        )

    # Explicit numeric code.
    if lower == "brand_code":
        return float(brand_code)

    # Dummy variables.
    if lower.startswith("brand_"):
        suffix = lower[len("brand_"):].strip()
        parts = [p for p in suffix.split("_") if p]

        if str(brand_code) in parts:
            return 1.0

        if canonical in parts:
            return 1.0

        suffix_compact = suffix.replace(" ", "").replace("-", "")

        for alias, alias_code in BRAND_ALIASES.items():
            if alias_code == brand_code:
                alias_key = _to_text(alias).lower()
                alias_compact = alias_key.replace(" ", "").replace("-", "")

                if alias_compact == suffix_compact:
                    return 1.0

        return 0.0

    # Dummy without prefix, e.g. MTZ82.
    if lower == canonical:
        return 1.0

    # Legacy continuous coding.
    if lower in {"brand", "x_brand"}:
        logger.warning(
            "Model uses legacy continuous brand coding. "
            "For production, prefer brand dummies or brand-specific effects."
        )

        if lower == "x_brand":
            return _standardize_x_value(params, "x_brand", float(brand_code))

        return float(brand_code)

    return 0.0


# ---------------------------------------------------------------------------
# Climate / Soil helpers
# ---------------------------------------------------------------------------
def _name_is_climate_related(name: str) -> bool:
    return _normalize_name(name) in {
        "climate",
        "x_climate",
        "climate_index",
        "climate_factor",
    }


def _name_is_soil_related(name: str) -> bool:
    return _normalize_name(name) in {
        "soil",
        "x_soil",
        "soil_index",
        "soil_factor",
    }


def _resolve_index_covariate(
    params: Any,
    covariates: Optional[Dict[str, Any]],
    x_key: str,
    display_name: str,
    reference: Dict[str, float],
) -> float:
    aliases = _get_raw_aliases(x_key)
    value = _get_source_value(params, covariates, aliases)

    if value is None:
        raise InvalidInputError(
            f"{display_name} index is required. "
            f"Provide {display_name} in [0, 1]."
        )

    return _validate_index_value(value, display_name, reference)


def _resolve_climate_value(
    params: Any,
    covariates: Optional[Dict[str, Any]],
) -> float:
    return _resolve_index_covariate(
        params,
        covariates,
        "x_climate",
        "Climate",
        CLIMATE_INDEX_REFERENCE,
    )


def _resolve_soil_value(
    params: Any,
    covariates: Optional[Dict[str, Any]],
) -> float:
    return _resolve_index_covariate(
        params,
        covariates,
        "x_soil",
        "Soil",
        SOIL_INDEX_REFERENCE,
    )
def _compute_age_hours_interaction(
    params: Any,
    covariates: Optional[Dict[str, Any]],
) -> float:
    """
    Interaction Age × Hours: воспроизводит generate_data().

    Шаги (должны совпадать с DGP):
    1. Стандартизация: x_age = (Age - 10) / 10, x_hours = (Hours - 1000) / 1000
    2. Центрирование: (x_age - x_age_mean) * (x_hours - x_hours_mean)
    3. Стандартизация результата: (interaction - mean) / std
    """
    if covariates is None:
        return 0.0

    # Если уже передан готовый — используем как есть
    ready = _get_source_value(
        params, covariates, ["x_age_hours", "Age_x_Hours", "age_hours"]
    )
    if ready is not None:
        return _coerce_float(ready, "x_age_hours")

    # Параметры из training_meta
    meta = _get_training_meta(params)
    ip = meta.get("interaction_params", {})
    x_age_mean = _try_float(ip.get("x_age_mean"), 0.0)
    x_hours_mean = _try_float(ip.get("x_hours_mean"), 0.0)
    x_age_hours_mean = _try_float(ip.get("x_age_hours_mean"), 0.0)
    x_age_hours_std = _try_float(ip.get("x_age_hours_std"), 1.0)
    if x_age_hours_std < 1e-9:
        x_age_hours_std = 1.0

    # Сырые Age и Hours
    age_aliases = _get_raw_aliases("x_age") + ["x_age", "Age", "age"]
    hours_aliases = _get_raw_aliases("x_hours") + ["x_hours", "Hours", "hours"]
    age_raw = _get_source_value(params, covariates, age_aliases)
    hours_raw = _get_source_value(params, covariates, hours_aliases)

    if age_raw is None or hours_raw is None:
        return 0.0

    # Шаг 1: стандартизация
    age_std = _standardize_x_value(params, "x_age", float(age_raw))
    hours_std = _standardize_x_value(params, "x_hours", float(hours_raw))

    # Шаг 2: центрирование и перемножение
    interaction = (age_std - x_age_mean) * (hours_std - x_hours_mean)

    # Шаг 3: стандартизация результата
    return float((interaction - x_age_hours_mean) / x_age_hours_std)

# ---------------------------------------------------------------------------
# Covariate resolution
# ---------------------------------------------------------------------------
def _resolve_standardized_x(
    params: Any,
    name: str,
    covariates: Optional[Dict[str, Any]],
    strict: bool = True,
) -> float:
    info = _get_x_std_info(params, name)
    aliases = _get_raw_aliases(name)

    raw_col = _to_text(info.get("raw_col"))

    if raw_col:
        aliases.append(raw_col)

    aliases.append(name)

    value = _get_source_value(params, covariates, aliases)

    if value is None:
        if strict:
            raise InvalidInputError(f"Required covariate '{name}' is missing")

        shift = info.get("shift")
        value = _try_float(shift, 0.0)

        logger.warning(
            "Covariate '%s' is missing; using fallback value %s in non-strict mode.",
            name,
            value,
        )

    value = _coerce_float(value, name)

    return _standardize_x_value(params, name, float(value))


def _get_z_default(params: Any) -> float:
    meta = _get_training_meta(params)
    raw = _dict_get_normalized(meta, "z_mean", 0.0)

    z = _try_float(raw, 0.0)

    if math.isfinite(z):
        return z

    return 0.0


def _resolve_covariate_value(
    params: Any,
    name: str,
    covariates: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> float:
    name_str = _to_text(name)
    lower = _normalize_name(name_str)

    if not name_str:
        if strict:
            raise InvalidInputError("Required covariate name is empty")

        logger.warning("Covariate name is empty; returning 0.0 in non-strict mode.")
        return 0.0

    if lower in {"const", "intercept"}:
        return 1.0

    if lower == "z":
        value = _get_source_value(params, covariates, ["Z", "z", "instrument"])

        if value is None:
            return _get_z_default(params)

        return _coerce_float(value, "Z")

    if _name_is_brand_related(name_str):
        brand_aliases = _get_raw_aliases("x_brand")
        brand_value = _get_source_value(params, covariates, brand_aliases)

        if brand_value is None:
            if strict:
                raise InvalidInputError(
                    f"Brand value is required for model column '{name_str}'."
                )

            logger.warning(
                "Brand value is missing for column '%s'; returning 0.0 in non-strict mode.",
                name_str,
            )
            return 0.0

        brand_code = coerce_brand_code(params, brand_value)

        return _brand_value_for_model_name(params, name_str, brand_code)

    if _name_is_climate_related(name_str):
        return _resolve_climate_value(params, covariates)

    if _name_is_soil_related(name_str):
        return _resolve_soil_value(params, covariates)

        # ─── НОВОЕ: Автоматический расчет interaction (СТРОГО ДО x_table!) ───
    if lower in {"x_age_hours", "age_hours", "age_x_hours"}:
        return _compute_age_hours_interaction(params, covariates)

    x_table = _get_x_standardization_table(params)

    if (
        _dict_get_normalized(X_STANDARDIZATION_FALLBACK, name_str) is not None
        or _dict_get_normalized(x_table, name_str) is not None
    ):
        return _resolve_standardized_x(
            params,
            name_str,
            covariates,
            strict=strict,
        )

    value = _get_source_value(params, covariates, [name_str])

    if value is None:
        if strict:
            raise InvalidInputError(f"Required covariate '{name_str}' is missing")

        logger.warning(
            "Covariate '%s' is missing; returning 0.0 in non-strict mode.",
            name_str,
        )
        return 0.0

    return _coerce_float(value, name_str)


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------
def _extract_baseline_arrays(
    baseline: Any,
    *,
    enforce_monotonic: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Common baseline extraction/validation logic.

    enforce_monotonic=False:
        strict validation only;
    enforce_monotonic=True:
        runtime numeric enforcement of non-decreasing cumulative hazard.
    """
    if not isinstance(baseline, dict):
        raise ModelValidationError(
            "baseline_cumulative_hazard must be a dictionary"
        )

    times_raw = _dict_get_normalized(baseline, "times")
    values_raw = _dict_get_normalized(baseline, "values")

    if times_raw is None or values_raw is None:
        raise ModelValidationError(
            "baseline_cumulative_hazard missing times/values"
        )

    try:
        times = np.asarray(times_raw, dtype=float)
        values = np.asarray(values_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(
            "baseline times/values must be numeric"
        ) from exc

    if times.ndim != 1 or values.ndim != 1:
        raise ModelValidationError(
            "baseline times/values must be one-dimensional"
        )

    if len(times) == 0:
        raise ModelValidationError("baseline_cumulative_hazard is empty")

    if len(times) != len(values):
        raise ModelValidationError("baseline times/values length mismatch")

    if not np.all(np.isfinite(times)):
        raise ModelValidationError("baseline times must be finite")

    if not np.all(np.isfinite(values)):
        raise ModelValidationError("baseline values must be finite")

    if np.any(np.diff(times) < 0):
        raise ModelValidationError("baseline times must be sorted")

    if np.any(values < -1e-12):
        raise ModelValidationError(
            "baseline cumulative hazard must be non-negative"
        )

    if np.any(np.diff(values) < -1e-12):
        raise ModelValidationError(
            "baseline cumulative hazard must be non-decreasing"
        )

    if enforce_monotonic:
        values = np.maximum.accumulate(values)

    return times, values


def _validate_baseline_dict(baseline: Any) -> None:
    _extract_baseline_arrays(baseline, enforce_monotonic=False)


def _baseline_arrays(params: Any) -> Tuple[np.ndarray, np.ndarray]:
    baseline = _get_field(params, "baseline_cumulative_hazard")
    return _extract_baseline_arrays(baseline, enforce_monotonic=True)


def _baseline_cumulative_hazard_array(
    params: Any,
    time_points: np.ndarray,
) -> np.ndarray:
    """
    Baseline cumulative hazard H0(t) via Breslow step-function.

    Standard non-parametric interpolation using searchsorted.
    H0(t) = 0 for t < t_(1) (first event time).
    H0(t) = H0(t_last) for t >= t_last (held constant).

    The step-function is the mathematically correct maximum-likelihood
    estimator for the cumulative baseline hazard in the Cox model.
    """
    time_points = np.asarray(time_points, dtype=float)

    if not np.all(np.isfinite(time_points)):
        raise InvalidInputError("time horizon must be finite")

    if np.any(time_points < 0.0):
        raise InvalidInputError("time horizon cannot be negative")

    times, values = _baseline_arrays(params)

    # Breslow step-function interpolation
    idx = np.searchsorted(times, time_points, side="right") - 1
    hazard = np.zeros_like(time_points, dtype=float)
    valid = idx >= 0
    hazard[valid] = values[idx[valid]]

    # Hold constant beyond last event time
    exceed = idx >= len(times)
    if np.any(exceed):
        hazard[exceed] = values[-1]

    return hazard


def _baseline_scalar(params: Any, time_point: float) -> float:
    """
    Return H0(t) for a single scalar time point.
    Avoids Union[float, List[float]] in internal calculation.
    """
    t = _as_finite_float(time_point, "time_horizon")

    if t < 0.0:
        raise InvalidInputError("time horizon cannot be negative")

    hazard = _baseline_cumulative_hazard_array(
        params,
        np.array([t], dtype=float),
    )

    return float(hazard[0])


def baseline_cumulative_hazard(
    params: Any,
    time_horizon: Union[float, Sequence[float]],
) -> Union[float, List[float]]:
    """
    Public helper: return H0(t).

    Scalar input returns float.
    Sequence input returns list.

    Important:
    time_horizon must already be in model baseline units.
    Use predict_probability(..., time_horizon_unit=...) for automatic conversion.
    """
    if isinstance(time_horizon, (str, bytes)):
        raise InvalidInputError("time_horizon must be numeric or sequence")

    if isinstance(time_horizon, np.ndarray):
        arr = np.asarray(time_horizon, dtype=float)

        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim != 1:
            raise InvalidInputError("time_horizon array must be one-dimensional")

        hazard_values = _baseline_cumulative_hazard_array(params, arr)
        return [float(x) for x in hazard_values]

    if isinstance(time_horizon, Sequence):
        arr = np.asarray(list(time_horizon), dtype=float)

        if arr.ndim != 1:
            raise InvalidInputError("time_horizon sequence must be one-dimensional")

        hazard_values = _baseline_cumulative_hazard_array(params, arr)
        return [float(x) for x in hazard_values]

    t = _as_finite_float(time_horizon, "time_horizon")
    return _baseline_scalar(params, t)


def _check_horizon_within_baseline(params: Any, time_horizon: float) -> None:
    meta = _get_training_meta(params)

    allow_extrapolation = _as_bool(
        _dict_get_normalized(meta, "allow_baseline_extrapolation", False),
        False,
    )

    times, _ = _baseline_arrays(params)
    max_t = float(times[-1])

    if time_horizon > max_t + 1e-9:
        msg = (
            f"time_horizon {time_horizon} exceeds max baseline time {max_t}. "
            "Baseline extrapolation is disabled."
        )

        if allow_extrapolation:
            logger.warning(msg)
        else:
            raise InvalidInputError(
                msg
                + " Set training_meta['allow_baseline_extrapolation']=True to allow."
            )


# ---------------------------------------------------------------------------
# Model validation helpers
# ---------------------------------------------------------------------------
def _validate_names_and_coefs(
    names: Any,
    coefs: Any,
    section: str,
) -> None:
    if not isinstance(names, list):
        raise ModelValidationError(f"{section}.exog_names must be a list")

    if not isinstance(coefs, dict):
        raise ModelValidationError(f"{section} coefficients must be a dictionary")

    seen = set()
    normalized_names: List[str] = []

    for raw_name in names:
        name = _to_text(raw_name)

        if not name:
            raise ModelValidationError(f"{section}.exog_names contains empty name")

        norm = _normalize_name(name)

        if norm in seen:
            raise ModelValidationError(
                f"{section}.exog_names contains duplicate column '{name}'"
            )

        seen.add(norm)
        normalized_names.append(norm)

    normalized_coefs = {}

    for raw_key, value in coefs.items():
        key = _to_text(raw_key)

        if not key:
            raise ModelValidationError(f"{section} coefficient key is empty")

        norm_key = _normalize_name(key)
        val = _as_finite_float(value, f"{section}.coefs[{key}]")
        normalized_coefs[norm_key] = val

    for norm_name in normalized_names:
        if norm_name not in normalized_coefs:
            raise ModelValidationError(
                f"{section} missing coefficient for column '{norm_name}'"
            )

    extra = set(normalized_coefs) - set(normalized_names)

    if extra:
        logger.warning(
            "%s contains extra coefficients not present in exog_names: %s",
            section,
            sorted(extra),
        )


def validate_model(params: Any) -> bool:
    if params is None:
        raise ModelValidationError("Model parameters required")

    training_meta = _get_field(params, "training_meta")

    if not isinstance(training_meta, dict):
        raise ModelValidationError("training_meta must be a dictionary")

    # Time unit.
    time_unit_raw = _dict_get_normalized(training_meta, "time_unit", MODEL_TIME_UNIT)

    try:
        _normalize_time_unit(time_unit_raw)
    except InvalidInputError as exc:
        raise ModelValidationError(
            f"Invalid training_meta.time_unit: {_fmt(time_unit_raw)}"
        ) from exc

    # ─── P-03: event_definition ─────────────────────────────────────────
    ed = _dict_get_normalized(training_meta, "event_definition")

    if ed is not None:
        ed_text = _to_text(ed).lower()

        if ed_text not in VALID_EVENT_DEFINITIONS:
            raise ModelValidationError(
                f"Unknown event_definition: '{_fmt(ed)}'. "
                f"Valid: {sorted(VALID_EVENT_DEFINITIONS)}."
            )

    ed_field = _get_field(params, "event_definition")

    if ed_field is not None:
        ed_field_text = _to_text(ed_field).lower()

        if ed_field_text not in VALID_EVENT_DEFINITIONS:
            raise ModelValidationError(
                f"Unknown event_definition field: '{_fmt(ed_field)}'. "
                f"Valid: {sorted(VALID_EVENT_DEFINITIONS)}."
            )

    # P-01: competing-risks semantics must have one canonical value.
    # A mismatch between the artifact root and training_meta used to allow
    # different consumers to interpret the same event definition differently.
    competing_root = _get_field(params, "competing_risks")
    competing_meta = _dict_get_normalized(training_meta, "competing_risks")
    if competing_root is not None and competing_meta is not None:
        root_bool = _as_bool(competing_root, False)
        meta_bool = _as_bool(competing_meta, False)
        if root_bool != meta_bool:
            raise ModelValidationError(
                "competing_risks mismatch: model root and training_meta disagree "
                f"({root_bool} != {meta_bool}). Rebuild the model artifact with "
                "one canonical value."
            )

    extrap_root = _get_field(params, "allow_baseline_extrapolation")
    extrap_meta = _dict_get_normalized(
        training_meta, "allow_baseline_extrapolation"
    )
    if extrap_root is not None and extrap_meta is not None:
        root_bool = _as_bool(extrap_root, False)
        meta_bool = _as_bool(extrap_meta, False)
        if root_bool != meta_bool:
            raise ModelValidationError(
                "allow_baseline_extrapolation mismatch: model root and "
                f"training_meta disagree ({root_bool} != {meta_bool})."
            )

    # First stage.
    first_stage = _get_field(params, "first_stage")

    if not isinstance(first_stage, dict):
        raise ModelValidationError("first_stage must be a dictionary")

    fs_names = _dict_get_normalized(first_stage, "exog_names")
    fs_params = _dict_get_normalized(first_stage, "params")

    _validate_names_and_coefs(fs_names, fs_params, "first_stage")

    forbidden_fs = {"peakload", "time", "event"}

    for raw_name in fs_names:
        name = _to_text(raw_name)

        if _normalize_name(name) in forbidden_fs:
            raise ModelValidationError(
                f"first_stage.exog_names must not contain '{name}'"
            )

    # Cox.
    cox = _get_field(params, "cox")

    if not isinstance(cox, dict):
        raise ModelValidationError("cox must be a dictionary")

    cox_names = _dict_get_normalized(cox, "exog_names")
    cox_coefs = _dict_get_normalized(cox, "coefs")

    _validate_names_and_coefs(cox_names, cox_coefs, "cox")

    # Baseline cumulative hazard.
    baseline = _get_field(params, "baseline_cumulative_hazard")
    _validate_baseline_dict(baseline)

    # Transform info.
    transform_info = _get_field(params, "transform_info")

    if transform_info is not None and not isinstance(transform_info, dict):
        raise ModelValidationError("transform_info must be a dictionary")

    if isinstance(transform_info, dict):
        t_type = _to_text(
            _dict_get_normalized(transform_info, "type", "none")
        ).lower()

        if t_type not in {"none", "center", "standardize"}:
            raise ModelValidationError(
                f"Unknown transform_info.type: '{_fmt(t_type)}'"
            )

        if t_type == "center":
            _as_finite_float(
                _dict_get_normalized(transform_info, "center", 0.0),
                "transform_info.center",
            )

        if t_type == "standardize":
            _as_finite_float(
                _dict_get_normalized(transform_info, "center", 0.0),
                "transform_info.center",
            )

            scale = _as_finite_float(
                _dict_get_normalized(transform_info, "scale", 1.0),
                "transform_info.scale",
            )

            if scale <= 0.0:
                raise ModelValidationError("transform_info.scale must be positive")

    # Residual standardization.
    residuals_mean = _get_field(params, "training_residuals_mean")
    residuals_std = _get_field(params, "training_residuals_std")

    if residuals_mean is not None:
        _as_finite_float(residuals_mean, "training_residuals_mean")

    if residuals_std is not None:
        std = _as_finite_float(residuals_std, "training_residuals_std")

        if std <= 0.0:
            raise ModelValidationError("training_residuals_std must be positive")

    # Calibration horizon.
    calibration = _get_field(params, "calibration_time_horizon")
    horizon = _try_float(calibration, 0.0)

    if horizon <= 0.0:
        meta_horizon = _dict_get_normalized(
            training_meta,
            "calibration_time_horizon",
        )
        horizon = _try_float(meta_horizon, 0.0)

    if horizon <= 0.0:
        raise ModelValidationError("calibration_time_horizon must be positive")

    # Baseline coverage.
    allow_extrapolation = _as_bool(
        _dict_get_normalized(training_meta, "allow_baseline_extrapolation", False),
        False,
    )

    times, _ = _baseline_arrays(params)
    max_baseline_time = float(times[-1])

    if horizon > max_baseline_time + 1e-9 and not allow_extrapolation:
        raise ModelValidationError(
            f"calibration_time_horizon {horizon} exceeds max baseline time "
            f"{max_baseline_time}. Baseline extrapolation is disabled. "
            "Set training_meta['allow_baseline_extrapolation']=True to allow extrapolation."
        )

    # Peak range metadata.
    pmin = _dict_get_normalized(training_meta, "peakload_min")
    pmax = _dict_get_normalized(training_meta, "peakload_max")

    if (pmin is None) != (pmax is None):
        raise ModelValidationError("Incomplete peakload training range")

    if pmin is not None and pmax is not None:
        pmin_f = _as_finite_float(pmin, "training_meta.peakload_min")
        pmax_f = _as_finite_float(pmax, "training_meta.peakload_max")

        if pmin_f > pmax_f:
            raise ModelValidationError("peakload_min exceeds peakload_max")

    # X standardization.
    x_std = _get_x_standardization_table(params)

    for raw_name, info in x_std.items():
        name = _to_text(raw_name)

        if not name:
            raise ModelValidationError(
                "x_standardization contains empty column name"
            )

        if not isinstance(info, dict):
            raise ModelValidationError(f"x_standardization[{name}] must be a dict")

        shift = info.get("shift")
        scale = info.get("scale")

        if shift is not None:
            _as_finite_float(shift, f"x_standardization[{name}].shift")

        if scale is not None:
            scale_f = _as_finite_float(scale, f"x_standardization[{name}].scale")

            if scale_f <= 0.0:
                raise ModelValidationError(
                    f"x_standardization[{name}].scale must be positive"
                )

    # CF basis metadata.
    cf_meta = _get_field(params, "cf_basis_metadata")

    if cf_meta is not None:
        if not isinstance(cf_meta, dict):
            raise ModelValidationError("cf_basis_metadata must be a dictionary")

        basis_type = _to_text(
            _dict_get_normalized(cf_meta, "v_hat_basis", "linear")
        ).lower()

        if basis_type not in {"linear", "spline", "powers"}:
            raise ModelValidationError(
                f"Unknown cf_basis_metadata.v_hat_basis: '{_fmt(basis_type)}'"
            )

        v_hat_cols = _dict_get_normalized(cf_meta, "v_hat_cols")

        if v_hat_cols is not None and not isinstance(v_hat_cols, list):
            raise ModelValidationError(
                "cf_basis_metadata.v_hat_cols must be a list"
            )

        if _dict_get_normalized(cf_meta, "residuals_mean") is not None:
            _as_finite_float(
                _dict_get_normalized(cf_meta, "residuals_mean"),
                "cf_basis_metadata.residuals_mean",
            )

        if _dict_get_normalized(cf_meta, "residuals_std") is not None:
            std = _as_finite_float(
                _dict_get_normalized(cf_meta, "residuals_std"),
                "cf_basis_metadata.residuals_std",
            )

            if std <= 0.0:
                raise ModelValidationError(
                    "cf_basis_metadata.residuals_std must be positive"
                )

        allow_heuristic = _as_bool(
            _dict_get_normalized(cf_meta, "allow_heuristic_spline", False),
            False,
        )

        if basis_type == "spline":
            knots = _dict_get_normalized(cf_meta, "knots")

            if not isinstance(knots, list) or not knots:
                raise ModelValidationError(
                    "cf_basis_metadata.knots is required for spline basis"
                )

            if not allow_heuristic:
                for required in (
                    "spline_degree",
                    "spline_domain_min",
                    "spline_domain_max",
                ):
                    if _dict_get_normalized(cf_meta, required) is None:
                        raise ModelValidationError(
                            f"cf_basis_metadata.{required} is required for spline basis "
                            "unless allow_heuristic_spline=True"
                        )

        if basis_type in {"spline", "powers"}:
            if (
                _get_field(params, "training_residuals_std") is None
                and _dict_get_normalized(cf_meta, "residuals_std") is None
            ):
                raise ModelValidationError(
                    "training_residuals_std or cf_basis_metadata.residuals_std "
                    "is required for nonlinear CF basis"
                )

    return True


# ---------------------------------------------------------------------------
# JSON serialization / deserialization
# ---------------------------------------------------------------------------
def _strip_json_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {_to_text(k): _strip_json_keys(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_strip_json_keys(v) for v in obj]

    return obj


def _normalize_model_names(params: ModelParameters) -> None:
    for section_name in ("first_stage", "cox"):
        section = _get_field(params, section_name)

        if not isinstance(section, dict):
            continue

        names = section.get("exog_names")

        if isinstance(names, list):
            section["exog_names"] = [_to_text(x) for x in names]

        coef_key = "params" if section_name == "first_stage" else "coefs"
        coefs = section.get(coef_key)

        if isinstance(coefs, dict):
            section[coef_key] = {_to_text(k): v for k, v in coefs.items()}

    cf_meta = _get_field(params, "cf_basis_metadata")

    if isinstance(cf_meta, dict):
        cols = cf_meta.get("v_hat_cols")

        if isinstance(cols, list):
            cf_meta["v_hat_cols"] = [_to_text(x) for x in cols]


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]

    if isinstance(obj, np.generic):
        return _to_jsonable(obj.item())

    if dataclasses.is_dataclass(obj):
        return _to_jsonable(dataclasses.asdict(obj))

    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ModelLoadError("Cannot serialize non-finite float value")
        return float(obj)

    if isinstance(obj, (int, bool, str)) or obj is None:
        return obj

    try:
        return str(obj)
    except (TypeError, ValueError) as exc:
        raise ModelLoadError("Cannot serialize model parameter object") from exc


def save_model_params(path: Union[str, Path], params: Any) -> None:
    try:
        path = Path(path)

        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        payload = _to_jsonable(params)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    except ModelLoadError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ModelLoadError(f"Unable to save model params to {path}") from exc


def load_model_params(
    path: Union[str, Path],
    validate: bool = True,
) -> ModelParameters:
    try:
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    except FileNotFoundError as exc:
        raise ModelLoadError(f"Model file not found: {path}") from exc
    except OSError as exc:
        raise ModelLoadError(f"Unable to read model file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"Malformed JSON in model file: {path}") from exc

    data = _strip_json_keys(data)

    if not isinstance(data, dict):
        raise ModelLoadError("Model file must contain a JSON object")

    field_names = {f.name for f in dataclasses.fields(ModelParameters)}
    unknown_fields = set(data.keys()) - field_names

    if unknown_fields:
        logger.warning("Unknown model fields ignored: %s", sorted(unknown_fields))

    if (
        "calibration_time_horizon" not in data
        or data.get("calibration_time_horizon") is None
    ):
        training_meta = data.get("training_meta", {})

        if isinstance(training_meta, dict):
            data["calibration_time_horizon"] = training_meta.get(
                "calibration_time_horizon",
                0.0,
            )
        else:
            data["calibration_time_horizon"] = 0.0

    filtered = {k: v for k, v in data.items() if k in field_names}

    try:
        params = ModelParameters(**filtered)
    except (TypeError, ValueError) as exc:
        raise ModelLoadError("Unable to construct ModelParameters") from exc

    _normalize_model_names(params)

    if validate:
        validate_model(params)

    return params


# ---------------------------------------------------------------------------
# PeakLoad transformation
# ---------------------------------------------------------------------------
def transform_peak(params: Any, peak_raw: float) -> float:
    peak = _as_finite_float(peak_raw, "PeakLoad")
    transform_info = _get_field(params, "transform_info")

    if not isinstance(transform_info, dict):
        return peak

    t_type = _to_text(
        _dict_get_normalized(transform_info, "type", "none")
    ).lower()

    if t_type == "none":
        return peak

    if t_type == "center":
        center = _as_finite_float(
            _dict_get_normalized(transform_info, "center", 0.0),
            "transform_info.center",
        )
        return peak - center

    if t_type == "standardize":
        center = _as_finite_float(
            _dict_get_normalized(transform_info, "center", 0.0),
            "transform_info.center",
        )
        scale = _as_finite_float(
            _dict_get_normalized(transform_info, "scale", 1.0),
            "transform_info.scale",
        )

        if scale <= 0.0:
            raise PredictionError("transform_info.scale must be positive")

        return (peak - center) / scale

    raise PredictionError(f"Unknown PeakLoad transform type: '{t_type}'")


def _validate_peak_range(params: Any, peak: float) -> None:
    if not ENFORCE_PEAK_RANGE:
        return

    meta = _get_training_meta(params)

    pmin = _dict_get_normalized(meta, "peakload_min")
    pmax = _dict_get_normalized(meta, "peakload_max")

    if pmin is None and pmax is None:
        return

    if pmin is None or pmax is None:
        raise ModelValidationError("Incomplete peakload training range")

    pmin_f = _as_finite_float(pmin, "peakload_min")
    pmax_f = _as_finite_float(pmax, "peakload_max")

    if pmin_f > pmax_f:
        raise ModelValidationError("peakload_min exceeds peakload_max")

    tolerance = PEAK_RANGE_ABS_TOLERANCE + PEAK_RANGE_REL_TOLERANCE * max(
        abs(pmin_f),
        abs(pmax_f),
        1.0,
    )

    if not (pmin_f - tolerance <= peak <= pmax_f + tolerance):
        raise InvalidInputError(
            f"PeakLoad {peak} outside training range [{pmin_f}, {pmax_f}] "
            f"with tolerance {tolerance}"
        )


# ---------------------------------------------------------------------------
# Coefficient helpers
# ---------------------------------------------------------------------------
def _get_coeff_map(coefs: Any, section: str) -> Dict[str, float]:
    if not isinstance(coefs, dict):
        raise ModelValidationError(f"{section} must be a dictionary")

    out: Dict[str, float] = {}

    for raw_key, value in coefs.items():
        key = _to_text(raw_key)

        if not key:
            raise ModelValidationError(f"{section} contains empty coefficient key")

        val = _as_finite_float(value, f"{section}[{key}]")
        out[_normalize_name(key)] = val

    return out


# ---------------------------------------------------------------------------
# First stage
# ---------------------------------------------------------------------------
def predict_first_stage(
    params: Any,
    covariates: Optional[Dict[str, Any]] = None,
    strict_covariates: bool = True,
) -> float:
    first_stage = _get_field(params, "first_stage")

    if not isinstance(first_stage, dict):
        raise ModelValidationError("Missing first_stage")

    exog_names = first_stage.get("exog_names", []) or []
    coefficients = first_stage.get("params", {}) or {}

    if not isinstance(exog_names, list):
        raise ModelValidationError("first_stage.exog_names must be a list")

    coef_map = _get_coeff_map(coefficients, "first_stage.params")

    eta = 0.0

    for raw_name in exog_names:
        name = _to_text(raw_name)
        norm = _normalize_name(name)

        if norm in {"peakload", "time", "event"}:
            continue

        if norm not in coef_map:
            raise ModelValidationError(
                f"first_stage.params missing coefficient for '{name}'"
            )

        value = _resolve_covariate_value(
            params,
            name,
            covariates,
            strict=strict_covariates,
        )

        if not math.isfinite(value):
            raise InvalidInputError(f"Covariate '{name}' is not finite")

        eta += coef_map[norm] * value

    if not math.isfinite(eta):
        raise PredictionError("First-stage linear predictor is not finite")

    return float(eta)


def compute_pl_hat_exog(
    params: Any,
    pl_hat: float,
    covariates: Optional[Dict[str, Any]] = None,
    strict_covariates: bool = True,
) -> float:
    pl_hat = _as_finite_float(pl_hat, "pl_hat")

    mean_raw = _get_field(params, "training_pl_hat_mean")

    if mean_raw is None:
        training_pl_hat_mean = 0.0
    else:
        training_pl_hat_mean = _as_finite_float(
            mean_raw,
            "training_pl_hat_mean",
        )

    partial_betas = _get_field(params, "partial_out_all_betas")

    if not isinstance(partial_betas, dict) or not partial_betas:
        result = pl_hat - training_pl_hat_mean

        if not math.isfinite(result):
            raise PredictionError("PL_hat_exog is not finite")

        return float(result)

    training_x_means = _get_field(params, "training_x_means")

    if not isinstance(training_x_means, dict):
        training_x_means = {}

    adjustment = 0.0

    for raw_name, beta in partial_betas.items():
        name = _to_text(raw_name)
        norm = _normalize_name(name)

        if norm in {"peakload", "time", "event"}:
            logger.warning(
                "partial_out_all_betas contains treatment/time variable '%s'; skipping.",
                name,
            )
            continue

        beta_f = _as_finite_float(beta, f"partial_out_all_betas[{name}]")

        x_current = _resolve_covariate_value(
            params,
            name,
            covariates,
            strict=strict_covariates,
        )

        if not math.isfinite(x_current):
            raise InvalidInputError(f"Covariate '{name}' is not finite")

        x_mean_raw = _dict_get_normalized(training_x_means, name, None)

        if x_mean_raw is None:
            if strict_covariates:
                raise ModelValidationError(
                    f"training_x_means missing mean for partial-out variable '{name}'"
                )

            x_mean = 0.0

            logger.warning(
                "training_x_means missing mean for '%s'; using 0.0 in non-strict mode.",
                name,
            )
        else:
            x_mean = _as_finite_float(x_mean_raw, f"training_x_means[{name}]")

        adjustment += beta_f * (x_current - x_mean)

    result = pl_hat - training_pl_hat_mean - adjustment

    if not math.isfinite(result):
        raise PredictionError("PL_hat_exog is not finite")

    return float(result)


# ---------------------------------------------------------------------------
# CF metadata helpers
# ---------------------------------------------------------------------------
def _get_cf_meta(params: Any) -> Dict[str, Any]:
    meta = _get_field(params, "cf_basis_metadata")

    if isinstance(meta, dict):
        return meta

    return {}


def name_is_cf(name: Any) -> bool:
    lower = _normalize_name(name)

    if lower in {"v_hat", "eps_d_hat"}:
        return True

    return lower.startswith(("v_hat", "eps_d_hat"))


def _name_is_cf(name: Any) -> bool:
    return name_is_cf(name)


def _is_linear_cf_name(name: Any) -> bool:
    return _normalize_name(name) in {"v_hat", "eps_d_hat"}


def is_cf_column_name(name: Any, cf_cols: Sequence[str]) -> bool:
    lower = _normalize_name(name)
    cf_cols_lower = {_normalize_name(c) for c in cf_cols}

    if lower in cf_cols_lower:
        return True

    if lower in {"v_hat", "eps_d_hat"}:
        return True

    if lower.startswith("v_hat") or lower.startswith("eps_d_hat_"):
        return True

    return False


def _get_cf_column_names(params: Any) -> List[str]:
    cox = _get_field(params, "cox")

    if isinstance(cox, dict):
        cox_names = cox.get("exog_names", []) or []

        if isinstance(cox_names, list):
            cf_from_cox: List[str] = []

            for raw_name in cox_names:
                name = _to_text(raw_name)

                if name and _name_is_cf(name):
                    cf_from_cox.append(name)

            if cf_from_cox:
                return cf_from_cox

    meta = _get_cf_meta(params)
    cols = _dict_get_normalized(meta, "v_hat_cols")

    if isinstance(cols, list) and cols:
        cleaned: List[str] = []

        for raw_col in cols:
            col = _to_text(raw_col)

            if col:
                cleaned.append(col)

        return cleaned

    return []


def _get_cf_basis_type(params: Any) -> str:
    meta = _get_cf_meta(params)
    basis_type_raw = _dict_get_normalized(meta, "v_hat_basis")
    basis_type = _to_text(basis_type_raw).lower()

    if basis_type:
        if basis_type in {"linear", "spline", "powers"}:
            return basis_type

        raise ModelValidationError(
            f"Unknown cf_basis_metadata.v_hat_basis: '{_fmt(basis_type_raw)}'"
        )

    cf_cols = _get_cf_column_names(params)

    for col in cf_cols:
        lower = _normalize_name(col)

        if lower.startswith("v_hat_s") or lower.startswith("eps_d_hat_s"):
            return "spline"

        if lower.startswith("v_hat_pow") or lower.startswith("eps_d_hat_pow"):
            return "powers"

    return "linear"


def _linear_cf_standardized(params: Any) -> bool:
    meta = _get_cf_meta(params)
    raw = _dict_get_normalized(meta, "linear_standardized")

    if raw is not None:
        return _as_bool(raw, True)

    cf_cols = _get_cf_column_names(params)

    for col in cf_cols:
        if _normalize_name(col) == "eps_d_hat":
            return False

    return True


def _get_cf_residual_mean_std(params: Any) -> Tuple[float, float]:
    meta = _get_cf_meta(params)

    mean_raw = _dict_get_normalized(meta, "residuals_mean")

    if mean_raw is None:
        mean_raw = _get_field(params, "training_residuals_mean")

    std_raw = _dict_get_normalized(meta, "residuals_std")

    if std_raw is None:
        std_raw = _get_field(params, "training_residuals_std")

    if mean_raw is None:
        mean = 0.0
    else:
        mean = _as_finite_float(mean_raw, "residuals_mean")

    if std_raw is None:
        std = 1.0
    else:
        std = _as_finite_float(std_raw, "residuals_std")

    if not math.isfinite(mean):
        raise ModelValidationError("residuals_mean is not finite")

    if not math.isfinite(std) or std <= 0.0:
        raise ModelValidationError("residuals_std must be positive")

    return float(mean), float(std)


def _standardized_residual_array(params: Any, raw_residuals: np.ndarray) -> np.ndarray:
    mean, std = _get_cf_residual_mean_std(params)
    return (np.asarray(raw_residuals, dtype=float) - mean) / std


def _get_cf_col_std_params(params: Any) -> Dict[str, Dict[str, Any]]:
    meta = _get_cf_meta(params)
    std_params = _dict_get_normalized(meta, "v_hat_col_std_params")

    if isinstance(std_params, dict):
        return std_params

    return {}


def _standardize_cf_array(
    name: str,
    values: np.ndarray,
    std_params: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    info = _dict_get_normalized(std_params, name, {})

    if not isinstance(info, dict):
        info = {}

    mean_raw = info.get("mean", 0.0)
    std_raw = info.get("std", 1.0)

    mean = _as_finite_float(mean_raw, f"v_hat_col_std_params[{name}].mean")
    std = _as_finite_float(std_raw, f"v_hat_col_std_params[{name}].std")

    if std <= 0.0:
        raise ModelValidationError(
            f"v_hat_col_std_params[{name}].std must be positive"
        )

    out = (np.asarray(values, dtype=float) - mean) / std

    if _as_bool(info.get("clip", False), False):
        clip_min = _try_float(info.get("clip_min", -10.0), -10.0)
        clip_max = _try_float(info.get("clip_max", 10.0), 10.0)
        out = np.clip(out, clip_min, clip_max)

    return out


# ---------------------------------------------------------------------------
# CF basis construction
# ---------------------------------------------------------------------------
def _get_spline_domain(params: Any, knots: List[float]) -> Tuple[float, float]:
    meta = _get_cf_meta(params)

    allow_heuristic = _as_bool(
        _dict_get_normalized(meta, "allow_heuristic_spline", False),
        False,
    )

    dmin_raw = _dict_get_normalized(meta, "spline_domain_min")
    dmax_raw = _dict_get_normalized(meta, "spline_domain_max")

    if dmin_raw is not None and dmax_raw is not None:
        dmin = _try_float_optional(dmin_raw)
        dmax = _try_float_optional(dmax_raw)

        if dmin is None or dmax is None or dmax <= dmin:
            raise ModelValidationError(
                "cf_basis_metadata.spline_domain_min/max are invalid"
            )

        return dmin, dmax

    if not allow_heuristic:
        raise ModelValidationError(
            "cf_basis_metadata.spline_domain_min/max are required for spline basis "
            "unless allow_heuristic_spline=True"
        )

    logger.warning("Using heuristic spline domain reconstruction.")

    residuals_arr = _get_field(params, "training_residuals_arr")

    if residuals_arr is not None:
        try:
            residuals = np.asarray(residuals_arr, dtype=float).ravel()
        except (TypeError, ValueError):
            residuals = np.array([], dtype=float)

        if residuals.size > 0:
            residuals = residuals[np.isfinite(residuals)]

            if residuals.size > 0:
                v_std = _standardized_residual_array(params, residuals)
                v_std = v_std[np.isfinite(v_std)]

                if v_std.size > 0:
                    dmin = float(np.min(v_std))
                    dmax = float(np.max(v_std))

                    if dmax <= dmin:
                        dmin -= 1.0
                        dmax += 1.0

                    dmin -= 1e-6
                    dmax += 1e-6

                    return dmin, dmax

    finite_knots = [k for k in knots if math.isfinite(k)]

    if finite_knots:
        dmin = min(finite_knots) - 1.0
        dmax = max(finite_knots) + 1.0
    else:
        dmin, dmax = -3.0, 3.0

    if dmax <= dmin:
        dmin, dmax = -3.0, 3.0

    return dmin, dmax


def _build_powers_basis(params: Any, raw_residuals: np.ndarray) -> Dict[str, np.ndarray]:
    meta = _get_cf_meta(params)
    cf_cols = _get_cf_column_names(params)
    std_params = _get_cf_col_std_params(params)

    max_power_raw = _dict_get_normalized(meta, "max_power")

    if max_power_raw is None:
        max_power = 2
        logger.warning("cf_basis_metadata.max_power missing; defaulting to 2.")
    else:
        max_power = _try_int_optional(max_power_raw)

        if max_power is None:
            raise ModelValidationError(
                "cf_basis_metadata.max_power must be an integer"
            )

        if max_power < 1:
            raise ModelValidationError("cf_basis_metadata.max_power must be >= 1")

    if cf_cols:
        max_power = max(max_power, len(cf_cols))

    v_std = _standardized_residual_array(params, raw_residuals)
    v_safe = np.clip(v_std, -10.0, 10.0)

    result: Dict[str, np.ndarray] = {}
    clip_powers = _as_bool(_dict_get_normalized(meta, "clip_powers", True), True)

    # PATCH-01: Apply SVD projection if available from training
    proj_raw = _dict_get_normalized(meta, "projection_matrix")
    proj_matrix = None
    if proj_raw is not None:
        try:
            proj_matrix = np.array(proj_raw, dtype=np.float64)
        except (ValueError, TypeError):
            logger.warning("projection_matrix in metadata is invalid; ignoring.")
            proj_matrix = None

    for p in range(1, max_power + 1):
        if p - 1 < len(cf_cols):
            name = cf_cols[p - 1]
        else:
            name = f"v_hat_pow{p}"

        if proj_matrix is not None and p <= proj_matrix.shape[0]:
            # Use projected basis from training
            raw_matrix = np.column_stack([v_safe**i for i in range(1, max_power + 1)])
            projected = raw_matrix @ proj_matrix
            values = projected[:, p - 1]
        else:
            values = v_safe ** p
        
        values = _standardize_cf_array(name, values, std_params)

        if clip_powers:
            values = np.clip(values, -10.0, 10.0)

        result[name] = values

    return result


def _build_spline_basis(params: Any, raw_residuals: np.ndarray) -> Dict[str, np.ndarray]:
    interpolate = _scipy_interpolate

    if interpolate is None:
        raise PredictionError("Spline CF basis requires scipy.interpolate.BSpline")

    bspline_class = getattr(interpolate, "BSpline", None)

    if bspline_class is None:
        raise PredictionError("Spline CF basis requires scipy.interpolate.BSpline")

    meta = _get_cf_meta(params)
    cf_cols = _get_cf_column_names(params)
    std_params = _get_cf_col_std_params(params)

    allow_heuristic = _as_bool(
        _dict_get_normalized(meta, "allow_heuristic_spline", False),
        False,
    )

    knots_raw = _dict_get_normalized(meta, "knots", []) or []

    if not isinstance(knots_raw, list):
        raise ModelValidationError("cf_basis_metadata.knots must be a list")

    knots: List[float] = []

    for k in knots_raw:
        k_float = _try_float_optional(k)

        if k_float is None:
            raise ModelValidationError(
                "cf_basis_metadata.knots must contain only numeric values"
            )

        knots.append(k_float)

    knots = sorted(set(knots))

    if not knots:
        raise ModelValidationError("Spline CF metadata requires non-empty knots")

    # FIX 5: Ensure float64 precision for spline basis to avoid
    # precision loss when knots are loaded from JSON as float32.
    knots = [float(np.float64(k)) for k in knots]

    domain_min, domain_max = _get_spline_domain(params, knots)
    domain_min = float(np.float64(domain_min))
    domain_max = float(np.float64(domain_max))

    interior_knots = sorted(
        {float(np.float64(k)) for k in knots if domain_min < float(np.float64(k)) < domain_max}
    )

    degree_raw = _dict_get_normalized(meta, "spline_degree")

    if degree_raw is None:
        if not allow_heuristic:
            raise ModelValidationError(
                "cf_basis_metadata.spline_degree is required for spline basis "
                "unless allow_heuristic_spline=True"
            )

        degree = 2
        logger.warning("cf_basis_metadata.spline_degree missing; defaulting to 2.")
    else:
        degree = _try_int_optional(degree_raw)

        if degree is None or degree < 0:
            raise ModelValidationError(
                "cf_basis_metadata.spline_degree must be an integer >= 0"
            )

    # FIX 5: Ensure float64 for knot_vector
    knot_vector = (
        [float(np.float64(domain_min))] * (degree + 1)
        + [float(np.float64(k)) for k in interior_knots]
        + [float(np.float64(domain_max))] * (degree + 1)
    )

    n_basis = len(knot_vector) - degree - 1

    if n_basis <= 0:
        raise PredictionError("Invalid spline knot vector")

    included_raw = _dict_get_normalized(meta, "spline_included_indices")
    dropped_raw = _dict_get_normalized(meta, "spline_dropped_column")

    if included_raw is not None:
        if not isinstance(included_raw, list):
            raise ModelValidationError(
                "cf_basis_metadata.spline_included_indices must be a list"
            )

        indices = []

        for item in included_raw:
            parsed = _try_int_optional(item)

            if parsed is None:
                raise ModelValidationError(
                    "cf_basis_metadata.spline_included_indices must contain integers"
                )

            indices.append(parsed)

    elif dropped_raw is not None:
        dropped = _try_int_optional(dropped_raw)

        if dropped is None:
            raise ModelValidationError(
                "cf_basis_metadata.spline_dropped_column must be an integer"
            )

        indices = [i for i in range(n_basis) if i != dropped]

    else:
        if len(cf_cols) == n_basis:
            indices = list(range(n_basis))
        elif len(cf_cols) == n_basis - 1:
            if not allow_heuristic:
                raise ModelValidationError(
                    "cf_basis_metadata.spline_included_indices or spline_dropped_column "
                    "is required to identify spline columns"
                )

            logger.warning(
                "Spline column matching is heuristic; defaulting to dropping first column."
            )
            indices = list(range(1, n_basis))
        else:
            raise ModelValidationError(
                "Cannot determine spline basis columns from cf_basis_metadata"
            )

    if not indices:
        raise ModelValidationError("Spline included column indices are empty")

    if any(i < 0 or i >= n_basis for i in indices):
        raise ModelValidationError("Spline included column index out of range")

    if len(set(indices)) != len(indices):
        raise ModelValidationError("Spline included column indices contain duplicates")

    v_std = _standardized_residual_array(params, raw_residuals)
    # FIX 5: Ensure float64 for spline evaluation
    v_std = np.asarray(v_std, dtype=np.float64)
    x = np.clip(v_std, float(np.float64(domain_min)), float(np.float64(domain_max))).astype(np.float64)

    try:
        eye = np.eye(n_basis)
        basis_cols = []

        for j in range(n_basis):
            bs = bspline_class(
                knot_vector,
                eye[j],
                k=degree,
                extrapolate=True,
            )
            basis_cols.append(bs(x))

        basis_matrix = np.column_stack(basis_cols)

    except (TypeError, ValueError) as exc:
        raise PredictionError("Unable to evaluate spline CF basis") from exc

    basis_matrix = np.nan_to_num(basis_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    if basis_matrix.ndim != 2 or basis_matrix.shape[1] == 0:
        raise PredictionError("Spline CF basis evaluation produced invalid matrix")

    # PATCH-01: Apply SVD projection for spline basis if available
    proj_raw = _dict_get_normalized(meta, "projection_matrix")
    if proj_raw is not None:
        try:
            proj_matrix = np.array(proj_raw, dtype=np.float64)
            if proj_matrix.shape[0] == basis_matrix.shape[1]:
                basis_matrix = basis_matrix @ proj_matrix
                logger.info("Applied SVD projection to spline CF basis")
        except (ValueError, TypeError) as e:
            logger.warning("projection_matrix for spline basis is invalid; ignoring: %s", e)

    basis_out = basis_matrix[:, indices]

    if cf_cols:
        if basis_out.shape[1] != len(cf_cols):
            if basis_out.shape[1] > len(cf_cols):
                if not allow_heuristic:
                    raise ModelValidationError(
                        "Spline basis has more columns than model expects and "
                        "cf_basis_metadata does not allow heuristic truncation"
                    )

                logger.warning(
                    "Spline CF basis has more columns than model expects; truncating."
                )
                basis_out = basis_out[:, : len(cf_cols)]
            else:
                raise PredictionError(
                    "Spline CF basis has fewer columns than model expects"
                )

        names = cf_cols
    else:
        names = [f"v_hat_s{i}" for i in range(basis_out.shape[1])]

    result: Dict[str, np.ndarray] = {}

    for i, name in enumerate(names):
        values = basis_out[:, i]
        values = _standardize_cf_array(name, values, std_params)
        result[name] = values

    return result


def _build_cf_basis_at_prediction(
    params: Any,
    residuals_arr: np.ndarray,
) -> Dict[str, np.ndarray]:
    residuals_arr = np.asarray(residuals_arr, dtype=float).ravel()

    if residuals_arr.size == 0:
        raise InvalidInputError("residuals_arr is empty")

    if not np.all(np.isfinite(residuals_arr)):
        raise InvalidInputError("residuals_arr contains non-finite values")

    basis_type = _get_cf_basis_type(params)
    cf_cols = _get_cf_column_names(params)

    if basis_type == "linear":
        name = cf_cols[0] if cf_cols else "v_hat"

        if _linear_cf_standardized(params):
            values = _standardized_residual_array(params, residuals_arr)
        else:
            mean, _ = _get_cf_residual_mean_std(params)
            values = residuals_arr - mean

        return {name: values}

    if basis_type == "powers":
        return _build_powers_basis(params, residuals_arr)

    if basis_type == "spline":
        return _build_spline_basis(params, residuals_arr)

    raise PredictionError(f"Unknown CF basis type: '{basis_type}'")


def build_cf_basis_at_prediction(
    params: Any,
    residuals_arr: np.ndarray,
) -> Dict[str, np.ndarray]:
    return _build_cf_basis_at_prediction(params, residuals_arr)


# ---------------------------------------------------------------------------
# Residual policy
# ---------------------------------------------------------------------------
def _validate_residual_policy(params: Any, policy: Any) -> str:
    if policy is None:
        policy = "plug-in"

    if hasattr(policy, "value"):
        policy = policy.value

    policy_text = _to_text(policy)

    if not policy_text:
        raise InvalidInputError("residual_policy must be a non-empty string or Enum value")

    policy = policy_text.strip().lower()

    if policy == "plug_in":
        policy = "plug-in"

    if policy == "bootstrap":
        raise InvalidInputError("bootstrap residual policy is not allowed")

    meta = _get_training_meta(params)

    allow_diagnostic = _as_bool(
        _dict_get_normalized(meta, "allow_diagnostic_residual_policies", False),
        False,
    )

    if policy in ALLOWED_PRODUCTION_RESIDUAL_POLICIES:
        return policy

    if policy in ALLOWED_DIAGNOSTIC_RESIDUAL_POLICIES:
        if not allow_diagnostic:
            raise InvalidInputError(
                f"Residual policy '{policy}' is diagnostic-only and is disabled. "
                "Set training_meta['allow_diagnostic_residual_policies']=True to enable."
            )

        # PATCH-17: явное предупреждение
        logger.warning(
            "⚠️ Диагностика политика '%s' активна. "
            "Коррекция эндогенности отключена. "
            "Результаты НЕ пригодны для ценообразования.",
            policy,
        )
        return policy

    raise InvalidInputError(
        f"Unknown residual policy: '{policy}'. Allowed production policy: 'plug-in'."
    )


def _raw_residual_for_policy(
    params: Any,
    policy: str,
    peak_transformed: float,
    pl_hat: float,
) -> float:
    if policy == "plug-in":
        residual = peak_transformed - pl_hat
    elif policy == "mean":
        mean, _ = _get_cf_residual_mean_std(params)
        residual = mean
    else:  # "zero"
        residual = 0.0
    if not math.isfinite(residual):
        raise PredictionError("Residual is not finite")
    return float(residual)


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------
def _probability_from_lp_h0(lp: float, h0: float) -> float:
    if not math.isfinite(lp):
        raise PredictionError("Cox linear predictor is not finite")

    h0 = float(h0)

    if not math.isfinite(h0):
        raise PredictionError("Baseline cumulative hazard is not finite")

    if h0 <= 0.0:
        return 0.0

    log_ch = math.log(h0) + lp

    if not math.isfinite(log_ch):
        if math.isinf(log_ch) and log_ch > 0.0:
            return 1.0

        if math.isinf(log_ch) and log_ch < 0.0:
            return 0.0

        raise PredictionError("Log cumulative hazard is not finite")

    if log_ch > MAX_LOG_CH:
        return 1.0

    if log_ch < MIN_LOG_CH:
        return 0.0

    ch = math.exp(log_ch)
    probability = -math.expm1(-ch)

    if not math.isfinite(probability):
        probability = 1.0 if ch > 0.0 else 0.0

    return float(min(1.0, max(0.0, probability)))


# ---------------------------------------------------------------------------
# Cox linear predictor
# ---------------------------------------------------------------------------
def _get_cox_peakload_convention(params: Any) -> str:
    meta = _get_training_meta(params)

    convention_raw = _dict_get_normalized(
        meta,
        "cox_peakload_convention",
        DEFAULT_COX_PEAKLOAD_CONVENTION,
    )

    convention = _to_text(convention_raw).lower()

    if not convention:
        convention = DEFAULT_COX_PEAKLOAD_CONVENTION

    if convention not in {"observed_peakload", "pl_hat_exog"}:
        raise ModelValidationError(
            f"Unknown cox_peakload_convention: {_fmt(convention_raw)}. "
            "Expected 'observed_peakload' or 'pl_hat_exog'."
        )

    return convention


def _cox_linear_predictor_details(
    params: Any,
    peak_raw: float,
    time_horizon: float,
    residual_policy: str,
    covariates: Optional[Dict[str, Any]] = None,
    time_horizon_unit: Optional[str] = None,
    strict_covariates: bool = True,
) -> Dict[str, Any]:
    peak_raw = _as_finite_float(peak_raw, "PeakLoad")

    time_horizon_model = _resolve_time_horizon(
        params,
        time_horizon,
        time_horizon_unit,
    )

    if time_horizon_model < 0.0:
        raise InvalidInputError("time horizon cannot be negative")

    _validate_peak_range(params, peak_raw)

    policy = _validate_residual_policy(params, residual_policy)
    peak_transformed = transform_peak(params, peak_raw)

    pl_hat = predict_first_stage(
        params,
        covariates,
        strict_covariates=strict_covariates,
    )

    pl_hat_exog = compute_pl_hat_exog(
        params,
        pl_hat,
        covariates,
        strict_covariates=strict_covariates,
    )

    raw_residual = _raw_residual_for_policy(
        params,
        policy,
        peak_transformed,
        pl_hat,
    )

    basis_type = _get_cf_basis_type(params)
    cf_cols = _get_cf_column_names(params)

    cox = _get_field(params, "cox")

    if not isinstance(cox, dict):
        raise ModelValidationError("Missing Cox model")

    cox_names = cox.get("exog_names", []) or []
    cox_coefs = cox.get("coefs", {}) or {}

    if not isinstance(cox_names, list):
        raise ModelValidationError("cox.exog_names must be a list")

    if not isinstance(cox_coefs, dict):
        raise ModelValidationError("cox.coefs must be a dictionary")

    cf_basis_values: Dict[str, np.ndarray] = {}

    if policy == "zero":
        v_hat_value = 0.0

        expected_cf = list(cf_cols)

        if not expected_cf:
            for raw_name in cox_names:
                name = _to_text(raw_name)

                if name and _name_is_cf(name):
                    expected_cf.append(name)

        for nm in expected_cf:
            cf_basis_values[nm] = np.array([0.0])

    else:
        if _linear_cf_standardized(params):
            v_hat_value = float(
                _standardized_residual_array(
                    params,
                    np.array([raw_residual], dtype=float),
                )[0]
            )
        else:
            mean, _ = _get_cf_residual_mean_std(params)
            v_hat_value = float(raw_residual - mean)

        if not math.isfinite(v_hat_value):
            raise PredictionError("v_hat is not finite")

        if basis_type in {"spline", "powers"}:
            cf_basis_values = _build_cf_basis_at_prediction(
                params,
                np.array([raw_residual], dtype=float),
            )
        else:
            linear_name = cf_cols[0] if cf_cols else "v_hat"
            cf_basis_values[linear_name] = np.array([v_hat_value], dtype=float)

    convention = _get_cox_peakload_convention(params)

    if convention == "pl_hat_exog":
        d_cox = pl_hat_exog
        # PATCH-08: В режиме 2SLS CF-коррекция НЕ нужна (двойная корректировка)
        for nm in list(cf_basis_values.keys()):
            cf_basis_values[nm] = np.array([0.0])
        logger.warning(
            "pl_hat_exog convention: CF correction disabled (2SLS mode)"
        )
    else:
        d_cox = peak_transformed

    if not math.isfinite(d_cox):
        raise PredictionError("Cox D variable is not finite")

    meta = _get_training_meta(params)

    peakload_column = _to_text(
        _dict_get_normalized(meta, "cox_peakload_column", "PeakLoad")
    )

    if not peakload_column:
        peakload_column = "PeakLoad"

    peakload_norm = _normalize_name(peakload_column)

    coef_map = _get_coeff_map(cox_coefs, "cox.coefs")

    lp = 0.0

    for raw_name in cox_names:
        name = _to_text(raw_name)

        if not name:
            raise ModelValidationError("cox.exog_names contains empty name")

        norm = _normalize_name(name)

        if norm == peakload_norm or norm == "peakload":
            val = d_cox

        elif _name_is_cf(name):
            if policy == "zero":
                val = 0.0
            else:
                val = None

                for k, arr in cf_basis_values.items():
                    if _normalize_name(k) == norm:
                        arr_flat = np.asarray(arr, dtype=float).ravel()

                        if arr_flat.size == 0:
                            raise PredictionError(f"CF column '{name}' is empty")

                        val = float(arr_flat[0])
                        break

                if val is None and basis_type == "linear" and _is_linear_cf_name(name):
                    val = v_hat_value

                if val is None:
                    raise PredictionError(
                        f"CF column '{name}' expected but not produced by basis builder"
                    )

        elif norm in {"const", "intercept"}:
            val = 1.0

        else:
            val = _resolve_covariate_value(
                params,
                name,
                covariates,
                strict=strict_covariates,
            )

        if not math.isfinite(val):
            raise InvalidInputError(f"Covariate value for '{name}' is not finite")

        if norm not in coef_map:
            raise ModelValidationError(f"cox.coefs missing coefficient for '{name}'")

        coef = coef_map[norm]
        lp += coef * val

    if not math.isfinite(lp):
        raise PredictionError("Cox linear predictor is not finite")

    _check_horizon_within_baseline(params, time_horizon_model)

    h0 = _baseline_scalar(params, time_horizon_model)
    probability = _probability_from_lp_h0(lp, h0)

    return {
        "peak_raw": peak_raw,
        "peak_transformed": peak_transformed,
        "pl_hat": pl_hat,
        "pl_hat_exog": pl_hat_exog,
        "raw_residual": raw_residual,
        "v_hat": v_hat_value,
        "lp": lp,
        "h0_t": h0,
        "probability": probability,
        "time_horizon": float(time_horizon_model),
        "time_horizon_unit": _get_time_unit(params),
        "residual_policy": policy,
        "cox_peakload_convention": convention,
        "cf_basis_type": basis_type,
    }


# ---------------------------------------------------------------------------
# Public prediction API
# ---------------------------------------------------------------------------
def predict_probability(
    params: Any,
    raw_peak: float,
    time_horizon: float,
    residual_policy: str = "plug-in",
    covariates: Optional[Dict[str, Any]] = None,
    time_horizon_unit: Optional[str] = None,
    strict_covariates: bool = True,
) -> float:
    # FIX 4: Warning when bootstrap SE is disabled
    training_meta = getattr(params, "training_meta", {}) or {}
    if not training_meta.get("bootstrap_se_enabled", False):
        logger.warning(
            "Bootstrap SE is disabled. Standard errors do not account for "
            "first-stage uncertainty in the generated regressor (v_hat). "
            "This leads to underestimated confidence intervals. "
            "Enable via training_meta['bootstrap_se_enabled'] = True."
        )

    if covariates is not None and not isinstance(covariates, dict):
        raise InvalidInputError("covariates must be a dictionary or None")

    details = _cox_linear_predictor_details(
        params=params,
        peak_raw=raw_peak,
        time_horizon=time_horizon,
        residual_policy=residual_policy,
        covariates=covariates,
        time_horizon_unit=time_horizon_unit,
        strict_covariates=strict_covariates,
    )

    probability = float(details["probability"])

    if probability < -PROBABILITY_EPSILON or probability > 1.0 + PROBABILITY_EPSILON:
        raise PredictionError(
            f"Predicted probability outside [0, 1]: {probability}"
        )

    return float(min(1.0, max(0.0, probability)))


def predict_many(
    params: Any,
    raw_peaks: Union[Sequence[float], np.ndarray],
    time_horizon: float,
    residual_policy: str = "plug-in",
    covariates: Optional[Dict[str, Any]] = None,
    time_horizon_unit: Optional[str] = None,
    strict_covariates: bool = True,
) -> Dict[str, Any]:
    if isinstance(raw_peaks, (str, bytes)):
        raise InvalidInputError("raw_peaks must not be a string")

    if isinstance(raw_peaks, np.ndarray):
        if raw_peaks.ndim != 1:
            raise InvalidInputError("raw_peaks must be one-dimensional")
        peaks_list = list(raw_peaks.tolist())
    elif isinstance(raw_peaks, Sequence):
        peaks_list = list(raw_peaks)
    else:
        raise InvalidInputError("raw_peaks must be a sequence or np.ndarray")

    if len(peaks_list) == 0:
        raise InvalidInputError("raw_peaks is empty")

    if len(peaks_list) > MAX_BATCH_SIZE:
        raise InvalidInputError(
            f"Batch size exceeds limit: {len(peaks_list)} > {MAX_BATCH_SIZE}"
        )

    if covariates is not None and not isinstance(covariates, dict):
        raise InvalidInputError("covariates must be a dictionary or None")

    policy = _validate_residual_policy(params, residual_policy)

    time_horizon_model = _resolve_time_horizon(
        params,
        time_horizon,
        time_horizon_unit,
    )

    if time_horizon_model < 0.0:
        raise InvalidInputError("time horizon cannot be negative")

    model_unit = _get_time_unit(params)

    probabilities: List[float] = []
    normalized_peaks: List[float] = []

    for peak in peaks_list:
        prob = predict_probability(
            params=params,
            raw_peak=peak,
            time_horizon=time_horizon_model,
            residual_policy=policy,
            covariates=covariates,
            time_horizon_unit=model_unit,
            strict_covariates=strict_covariates,
        )

        probabilities.append(float(prob))
        normalized_peaks.append(_as_finite_float(peak, "PeakLoad"))

    return {
        "probabilities": probabilities,
        "peaks": normalized_peaks,
        "time_horizon": float(time_horizon_model),
        "time_horizon_unit": model_unit,
        "residual_policy": policy,
    }


# ---------------------------------------------------------------------------
# v0.2 helpers: event definition / major failure share / Beta-prior
# ---------------------------------------------------------------------------
def _get_event_definition(params: Any) -> str:
    """
    P-03: определение события из training_meta.

    Допустимые значения: total_loss, major_claim, any_failure.
    При неизвестном значении возвращает major_claim.
    """
    meta = _get_training_meta(params)

    ed = _dict_get_normalized(meta, "event_definition", "major_claim")
    ed_text = _to_text(ed).lower()

    if ed_text not in VALID_EVENT_DEFINITIONS:
        return "major_claim"

    return ed_text


def _get_major_failure_share(params: Any) -> float:
    """
    P-04: доля major-отказов.

    Берёт значение из training_meta["major_failure_share"],
    иначе использует константу MAJOR_FAILURE_SHARE.
    """
    meta = _get_training_meta(params)

    raw = _dict_get_normalized(meta, "major_failure_share", MAJOR_FAILURE_SHARE)
    share = _try_float(raw, MAJOR_FAILURE_SHARE)

    if not (0.0 <= share <= 1.0):
        raise ModelValidationError(
            f"major_failure_share must be in [0, 1], got {share}"
        )

    return float(share)


def _get_major_failure_share_prior(params: Any) -> Dict[str, float]:
    """
    P-04: Beta-prior доли major-отказов.

    Ожидает training_meta["major_failure_share_prior"] вида:
        {
            "alpha": float,
            "beta": float,
            "mean": float,
        }

    Если prior не задан, строит fallback:
        mean = major_failure_share
        alpha = mean * 30
        beta = (1 - mean) * 30
    """
    meta = _get_training_meta(params)
    prior = _dict_get_normalized(meta, "major_failure_share_prior")

    if isinstance(prior, dict) and "mean" in prior:
        mean = _try_float_optional(prior.get("mean"))

        if mean is None:
            raise ModelValidationError(
                "major_failure_share_prior.mean must be finite numeric"
            )

        alpha_raw = prior.get("alpha")
        beta_raw = prior.get("beta")

        alpha = _try_float_optional(alpha_raw)
        beta = _try_float_optional(beta_raw)

        if alpha_raw is not None and alpha is None:
            raise ModelValidationError(
                "major_failure_share_prior.alpha must be finite numeric"
            )

        if beta_raw is not None and beta is None:
            raise ModelValidationError(
                "major_failure_share_prior.beta must be finite numeric"
            )

        if alpha is None:
            alpha = mean * 30.0

        if beta is None:
            beta = (1.0 - mean) * 30.0

        alpha = float(alpha)
        beta = float(beta)
        mean = float(mean)

        eps = 1e-12
        alpha = max(alpha, eps)
        beta = max(beta, eps)

        if not (0.0 <= mean <= 1.0):
            raise ModelValidationError(
                f"major_failure_share_prior.mean must be in [0, 1], got {mean}"
            )

        return {
            "alpha": alpha,
            "beta": beta,
            "mean": mean,
        }

    m = _get_major_failure_share(params)

    eps = 1e-12
    alpha = max(m * 30.0, eps)
    beta = max((1.0 - m) * 30.0, eps)

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "mean": float(m),
    }


# ---------------------------------------------------------------------------
# v0.2 helpers: P-05 shares / severity weights
# ---------------------------------------------------------------------------
def get_freq_shares(params: Any) -> Dict[str, float]:
    """
    P-05: вернуть доли отказов по системам.

    Если training_meta["freq_shares"] задан, он дополняет/переопределяет
    FREQ_SHARES. Итоговые значения нормализуются к сумме 1.
    """
    meta = _get_training_meta(params)
    raw = _dict_get_normalized(meta, "freq_shares", FREQ_SHARES)

    out = dict(FREQ_SHARES)

    if isinstance(raw, dict):
        for k, v in raw.items():
            key = _to_text(k).lower()

            if not key:
                continue

            out[key] = _as_finite_float(v, f"freq_shares[{key}]")

    for key, val in out.items():
        if val < 0.0:
            raise ModelValidationError(f"freq_shares[{key}] cannot be negative")

    total = float(sum(out.values()))

    if total <= 0.0:
        raise ModelValidationError("freq_shares sum must be positive")

    if abs(total - 1.0) > 1e-6:
        logger.warning(
            "freq_shares sum is %.10f, normalizing to 1.0.",
            total,
        )
        out = {k: float(v) / total for k, v in out.items()}

    return out


def get_severity_weights(params: Any) -> Dict[str, float]:
    """
    P-05: вернуть доли стоимости отказов по системам.
    """
    meta = _get_training_meta(params)
    raw = _dict_get_normalized(meta, "severity_weights", SEVERITY_WEIGHTS)
    out = dict(SEVERITY_WEIGHTS)
    if isinstance(raw, dict):
        for k, v in raw.items():
            key = _to_text(k).lower()
            if not key:
                continue
            out[key] = _as_finite_float(v, f"severity_weights[{key}]")
    for key, val in out.items():
        if val < 0.0:
            raise ModelValidationError(f"severity_weights[{key}] cannot be negative")
    return out


def get_criticality_weights(params: Any) -> Dict[str, float]:
    """
    P-05: вернуть веса критичности систем.
    Не путать с get_severity_weights (доли стоимости).
    """
    meta = _get_training_meta(params)
    raw = _dict_get_normalized(meta, "criticality_weights", CRITICALITY_WEIGHTS)
    out = dict(CRITICALITY_WEIGHTS)
    if isinstance(raw, dict):
        for k, v in raw.items():
            key = _to_text(k).lower()
            if not key:
                continue
            out[key] = _as_finite_float(v, f"criticality_weights[{key}]")
    for key, val in out.items():
        if val < 0.0:
            raise ModelValidationError(f"criticality_weights[{key}] cannot be negative")
    return out


# ---------------------------------------------------------------------------
# v0.2 helpers: P-08 MTBF baseline
# ---------------------------------------------------------------------------
def get_mtbf_baseline_hours(params: Any) -> float:
    """
    P-08: MTBF baseline в engine_hours.

    Берёт training_meta["mtbf_baseline_hours"], иначе MTBF_BASELINE_HOURS.
    """
    meta = _get_training_meta(params)

    raw = _dict_get_normalized(meta, "mtbf_baseline_hours", MTBF_BASELINE_HOURS)
    value = _try_float(raw, MTBF_BASELINE_HOURS)

    if value <= 0.0:
        raise ModelValidationError("mtbf_baseline_hours must be positive")

    return float(value)


# ---------------------------------------------------------------------------
# v0.2 helpers: P-09 downtime by MTTR
# ---------------------------------------------------------------------------
def get_downtime_hours(params: Any, mttr_hours: float) -> float:
    """
    P-09: downtime = MTTR * downtime_per_mttr_factor.

    Фактор берётся из training_meta["downtime_per_mttr_factor"],
    иначе DEFAULT_DOWNTIME_PER_MTTR_FACTOR.
    """
    mttr = _as_finite_float(mttr_hours, "mttr_hours")

    if mttr < 0.0:
        raise InvalidInputError("mttr_hours cannot be negative")

    meta = _get_training_meta(params)

    factor = _try_float(
        _dict_get_normalized(
            meta,
            "downtime_per_mttr_factor",
            DEFAULT_DOWNTIME_PER_MTTR_FACTOR,
        ),
        DEFAULT_DOWNTIME_PER_MTTR_FACTOR,
    )

    if factor < 0.0:
        raise ModelValidationError("downtime_per_mttr_factor cannot be negative")

    return float(mttr * factor)


# ---------------------------------------------------------------------------
# v0.2 helpers: P-10 power segments
# ---------------------------------------------------------------------------
def _get_power_segment_thresholds(params: Any) -> Dict[str, Tuple[float, float]]:
    """
    P-10: пороги сегментов мощности.

    Если training_meta["power_segment_thresholds"] задан и валиден,
    используется он. Иначе возвращается POWER_SEGMENT_THRESHOLDS.
    """
    if params is None:
        return POWER_SEGMENT_THRESHOLDS

    meta = _get_training_meta(params)
    raw = _dict_get_normalized(meta, "power_segment_thresholds")

    if not isinstance(raw, dict):
        return POWER_SEGMENT_THRESHOLDS

    out: Dict[str, Tuple[float, float]] = {}

    for segment in ("light", "medium", "heavy"):
        pair = _dict_get_normalized(raw, segment)

        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return POWER_SEGMENT_THRESHOLDS

        lower = _try_float_optional(pair[0])
        upper = _try_float_optional(pair[1])

        if lower is None or upper is None:
            return POWER_SEGMENT_THRESHOLDS

        if not (0.0 <= lower < upper):
            return POWER_SEGMENT_THRESHOLDS

        out[segment] = (float(lower), float(upper))

    return out


def classify_power_segment(power_hp: Any, params: Optional[Any] = None) -> str:
    """
    P-10: классифицировать мощность по сегментам:
        light / medium / heavy.
    """
    hp = _as_finite_float(power_hp, "power_hp")

    if hp < 0.0:
        raise InvalidInputError("power_hp cannot be negative")

    thresholds = _get_power_segment_thresholds(params)

    for segment in ("light", "medium", "heavy"):
        lower, upper = thresholds[segment]

        if lower <= hp < upper:
            return segment

    return "heavy"


# ---------------------------------------------------------------------------
# P-12: Kaplan–Meier validator
# ---------------------------------------------------------------------------
def kaplan_meier_check(
    params: Any,
    times: Sequence[float],
    events: Sequence[int],
    eval_horizon: Optional[float] = None,
) -> Dict[str, float]:
    """
    P-12: сравнить модельную baseline survival S0(t) с эмпирической Kaplan–Meier.

    Параметры
    ---------
    params:
        Модельные параметры.
    times:
        Наблюдаемые времена до события/цензурирования.
        Должны быть в тех же единицах времени, что и baseline.
    events:
        1 = событие, 0 = цензурирование.
    eval_horizon:
        Точка, в которой сравниваем выживание.
        Если None, используется calibration_time_horizon из params/training_meta.

    Возврат
    -------
    Dict[str, float]:
        eval_horizon, km_survival, model_survival, abs_diff, n_obs, n_events.

    Важно
    -----
    eval_horizon должен быть в model baseline units.
    Если модель в engine_hours, а вы хотите проверить календарные дни,
    сначала конвертируйте:
        calendar_days_to_engine_hours(days, hours_per_day)
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int).astype(bool)

    if t.size == 0:
        raise InvalidInputError("times/events must be non-empty")

    if t.size != e.size:
        raise InvalidInputError("times and events must have equal length")

    if np.any(t < 0.0):
        raise InvalidInputError("times cannot be negative")

    if eval_horizon is None:
        horizon = _try_float(
            _get_field(params, "calibration_time_horizon"),
            0.0,
        )
    else:
        horizon = _as_finite_float(eval_horizon, "eval_horizon")

    if horizon <= 0.0:
        raise InvalidInputError("eval horizon must be positive")

    # Запрещаем выход за пределы baseline, если extrapolation явно не разрешён.
    _check_horizon_within_baseline(params, horizon)

    # --- Эмпирическая Kaplan–Meier оценка выживания в точке horizon ---
    order = np.argsort(t, kind="mergesort")
    t_sorted = t[order]
    e_sorted = e[order]

    km = 1.0
    i = 0
    n_at_risk = t.size

    while i < t.size and t_sorted[i] <= horizon:
        j = i
        d = 0
        current_time = t_sorted[i]

        while j < t.size and t_sorted[j] == current_time:
            if e_sorted[j]:
                d += 1
            j += 1

        if n_at_risk > 0 and d > 0:
            km *= (1.0 - d / n_at_risk)

        n_at_risk -= (j - i)
        i = j

    km_survival = float(np.clip(km, 0.0, 1.0))

    # --- Модельное baseline выживание при lp=0: S0(t) = exp(-H0(t)) ---
    h0 = _baseline_scalar(params, horizon)

    if h0 < 0.0:
        raise ModelValidationError(
            f"baseline cumulative hazard must be non-negative, got {h0}"
        )

    # Защита от overflow/underflow при очень больших H0.
    if h0 > 700.0:
        model_survival = 0.0
    else:
        model_survival = float(math.exp(-h0))

    model_survival = float(np.clip(model_survival, 0.0, 1.0))

    return {
        "eval_horizon": float(horizon),
        "km_survival": km_survival,
        "model_survival": model_survival,
        "abs_diff": float(abs(km_survival - model_survival)),
        "n_obs": int(t.size),
        "n_events": int(e.sum()),
    }