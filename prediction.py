# -*- coding: utf-8 -*-
"""
prediction.py
Stable prediction & bootstrap API for CF Cox (Control Function) models.

Version: 3.1.0
- Corrected X_STANDARDIZATION to training values.
- PredictionTemplate is now picklable for joblib/loky.
- Robust event column handling in bootstrap.
- Strict validation for policy_horizon_days, theta, sum_insured.
- Removed unsafe silent fallbacks for missing first-stage covariates/coefficients.
- Bootstrap CI uses premium_engine when available.
- Added logging and stricter model-compatibility checks.
"""

from __future__ import annotations

import importlib.util as _ilu
import logging
import math
import numbers
import os as _os
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter

logger = logging.getLogger(__name__)

__all__ = [
    "PredictionTemplate",
    "ResidualPolicy",
    "TemplateMode",
    "ConvergenceInfo",
    "CFModelResult",
    "predict_first_stage",
    "predict_premiums",
    "bootstrap_premiums_ci",
    "BootstrapFailure",
    "BootstrapDiagnostics",
    "FitFirstStageFn",
    "FitCFFn",
]

# ---------------------------------------------------------------------------
# Robust import of shared dataclasses from Итог.py
# ---------------------------------------------------------------------------
try:
    from Итог import ConvergenceInfo, CFModelResult  # noqa: F401
except ImportError:
    try:
        _itog_path = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            "Итог.py",
        )
        if _os.path.exists(_itog_path):
            _spec = _ilu.spec_from_file_location("Итог", _itog_path)
            if _spec is None or _spec.loader is None:
                raise ImportError("Cannot create import spec for Итог.py")

            _itog_mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_itog_mod)

            ConvergenceInfo = _itog_mod.ConvergenceInfo
            CFModelResult = _itog_mod.CFModelResult
        else:
            raise ImportError("Итог.py not found")
    except Exception as exc:
        logger.warning(
            "Falling back to local ConvergenceInfo/CFModelResult definitions: %s",
            exc,
        )

        @dataclass
        class ConvergenceInfo:
            penalizer: float
            warning: Optional[str] = None
            attempted_penalizers: List[float] = field(default_factory=list)

        @dataclass
        class CFModelResult:
            gamma_hat: float
            se: float
            rho_hat: float
            cph: Any
            max_se: float
            penalizer: float
            convergence_info: Optional[ConvergenceInfo] = None
            warnings: List[str] = field(default_factory=list)
            n: Optional[int] = None
            n_events: Optional[int] = None
            v_hat_basis: str = "linear"
            partial_out_all_betas: Dict[str, float] = field(default_factory=dict)
            training_x_means: Dict[str, float] = field(default_factory=dict)
            training_pl_hat_mean: float = 0.0
            training_residuals_std: float = 0.0
            training_residuals_mean: float = 0.0
            vif_peakload: Optional[float] = None
            vif_vhat: Optional[float] = None
            cf_basis_metadata: Optional[Dict[str, Any]] = None
            rho_hat_signed: Optional[float] = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ProjectError(Exception):
    DEFAULT_MESSAGE = "Project error."

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        cause: Optional[Exception] = None,
    ) -> None:
        self.message = message if message is not None else self.DEFAULT_MESSAGE
        self.cause = cause
        super().__init__(self.message)
        if cause is not None:
            self.__cause__ = cause


class PredictionError(ProjectError):
    DEFAULT_MESSAGE = "Prediction error."


class BootstrapError(ProjectError):
    DEFAULT_MESSAGE = "Bootstrap failed."


class ModelCompatibilityError(ProjectError):
    DEFAULT_MESSAGE = "Model compatibility error."


class BootstrapFailure(ProjectError):
    DEFAULT_MESSAGE = "Bootstrap failed."

    def __init__(self, diagnostics: "BootstrapDiagnostics") -> None:
        msg = (
            f"Bootstrap failed "
            f"({diagnostics.bootstrap_successful}/"
            f"{diagnostics.bootstrap_requested})"
        )
        super().__init__(msg)
        self.diagnostics = diagnostics


# ---------------------------------------------------------------------------
# Constants and defaults
# ---------------------------------------------------------------------------
_TEMPLATE_SKIP = {
    "time",
    "event",
    "T_true",
    "C",
    "eps_D",
    "U",
}

_LATENT_SKIP = {
    "u_event",
    "u_censor",
    "clipped_lp",
    "lp_raw",
    "lp",
    "individual_hazard",
}

_BRAND_SKIP = {
    "brand_code",
    "brand_MTZ82",
    "brand_Versatile280",
    "brand_NewHollandT9",
    "brand_DT75",
    "brand_Other",
}

_ID_SKIP = {
    "id",
    "ID",
    "index",
    "Unnamed: 0",
}

_PREDICTION_SKIP = _TEMPLATE_SKIP | _LATENT_SKIP | _BRAND_SKIP | _ID_SKIP

DEFAULT_ALPHA = 0.05
MAX_BASELINE_H = 700.0
PEAK_RANGE_TOLERANCE = 5.0

# ---------------------------------------------------------------------------
# ВАЖНО: prediction.py и prediction_engine.py решают разные задачи и
# сознательно НЕ используют общую реализацию для predict_first_stage /
# baseline-hazard:
#
#   - prediction.py    — training-time bootstrap CI API. Работает
#     напрямую с только что подогнанными объектами statsmodels/lifelines
#     (fitted_first_stage, CoxPHFitter) внутри параллельного bootstrap-цикла.
#   - prediction_engine.py — serving-time движок. Работает с уже
#     сериализованным ModelParameters (dict-поля first_stage, cox,
#     baseline_cumulative_hazard и т.д.), без доступа к самим fitted-объектам.
#
# Раньше здесь был импорт функций/классов из prediction_engine с расчётом
# на "делегирование", но ни один из них фактически не вызывался — сигнатуры
# несовместимы (см. ниже), и импорт молча висел мёртвым грузом. Он удалён,
# чтобы код не утверждал то, чего не делает.
#
# TODO(train-serve parity): при необходимости реального единого источника
# истины для этой логики нужен отдельный адаптер, который на каждой
# bootstrap-итерации строит ModelParameters из fitted_first_stage/CoxPHFitter
# (или наоборот — прогоняет serving-логику через уже подогнанные объекты).
# Это отдельная задача, а не однострочная замена вызовов:
#   predict_first_stage(template, fitted_first_stage, covariates)             — здесь
#   prediction_engine.predict_first_stage(params: ModelParameters, ...)       — там
# принимают несовместимые типы первого аргумента.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ResidualPolicy(Enum):
    PLUG_IN = "plug-in"
    MEAN = "mean"
    ZERO = "zero"


class TemplateMode(Enum):
    BOOTSTRAP = "bootstrap"
    ORIGINAL = "original"


DEFAULT_RESIDUAL_POLICY: ResidualPolicy = ResidualPolicy.PLUG_IN
DEFAULT_TEMPLATE_MODE: TemplateMode = TemplateMode.BOOTSTRAP


# ---------------------------------------------------------------------------
# Protocols for injected functions
# ---------------------------------------------------------------------------
class FitFirstStageFn(Protocol):
    def __call__(
        self,
        data: pd.DataFrame,
        return_design: bool = True,
    ) -> Tuple[Any, np.ndarray, Any]:
        ...


class FitCFFn(Protocol):
    def __call__(
        self,
        data: pd.DataFrame,
        residuals: np.ndarray,
    ) -> Any:
        ...


# ---------------------------------------------------------------------------
# PredictionTemplate
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PredictionTemplate:
    """
    Immutable-by-convention prediction template.

    A plain dict is used instead of MappingProxyType because MappingProxyType
    is not reliably picklable by joblib/loky. Do not mutate covariates.
    """

    covariates: Dict[str, Any]

    @staticmethod
    def from_dataframe(
        fitted_first_stage: Any,
        data_ref: pd.DataFrame,
    ) -> "PredictionTemplate":
        """
        Build a template of covariates from data_ref using medians for
        continuous numeric columns and modes for categorical/binary columns.

        Excludes latent/simulation-only columns in _PREDICTION_SKIP.
        """
        del fitted_first_stage  # kept for API compatibility

        covs: Dict[str, Any] = {}

        for col in data_ref.columns:
            if col in _PREDICTION_SKIP:
                continue

            ser = data_ref[col]

            if pd.api.types.is_bool_dtype(ser):
                mode = ser.mode(dropna=True)
                covs[col] = bool(mode.iat[0]) if len(mode) > 0 else False
                continue

            if pd.api.types.is_numeric_dtype(ser):
                try:
                    n_unique = ser.nunique(dropna=True)
                except Exception:
                    n_unique = np.nan

                # Binary / low-cardinality numeric codes are safer as mode.
                if pd.notna(n_unique) and n_unique <= 2:
                    mode = ser.mode(dropna=True)
                    if len(mode) > 0:
                        covs[col] = float(mode.iat[0])
                    else:
                        covs[col] = 0.0
                else:
                    med = ser.median(skipna=True)
                    covs[col] = float(med) if pd.notna(med) else 0.0
                continue

            # Non-numeric categorical column.
            try:
                mode = ser.mode(dropna=True)
                covs[col] = mode.iat[0] if len(mode) > 0 else None
            except Exception:
                covs[col] = ser.iloc[0] if len(ser) > 0 else None

        return PredictionTemplate(covs)


# ---------------------------------------------------------------------------
# Dataclasses for bootstrap
# ---------------------------------------------------------------------------
@dataclass
class BootstrapWorkerResult:
    success: bool
    probabilities: Optional[np.ndarray]
    reason: Optional[str]
    exception_type: Optional[str]
    traceback: Optional[str]
    n_events: Optional[int]
    elapsed_seconds: float


@dataclass
class BootstrapDiagnostics:
    bootstrap_requested: int
    bootstrap_completed: int
    bootstrap_successful: int
    bootstrap_failed: int
    success_fraction: float
    fail_reasons: Dict[str, int]
    error_examples: List[str]
    elapsed_seconds: float
    median_runtime_per_rep: Optional[float] = None


# ---------------------------------------------------------------------------
# Small type/name helpers
# ---------------------------------------------------------------------------
def _is_real_number(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _is_integral_number(x: Any) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _norm_name(name: Any) -> str:
    return str(name).strip().lower()


def _normalized_lookup(mapping: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not mapping:
        return {}
    return {_norm_name(k): v for k, v in mapping.items()}


def _get_by_norm(
    mapping: Optional[Mapping[str, Any]],
    key: Any,
    default: Any = None,
) -> Any:
    if not mapping:
        return default
    norm = _normalized_lookup(mapping)
    return norm.get(_norm_name(key), default)


def _coerce_residual_policy(policy: Any) -> ResidualPolicy:
    if policy is None:
        return DEFAULT_RESIDUAL_POLICY
    if isinstance(policy, ResidualPolicy):
        return policy
    if isinstance(policy, str):
        return ResidualPolicy(policy.strip().lower())
    raise TypeError("residual_policy must be ResidualPolicy or string")


def _coerce_template_mode(mode: Any) -> TemplateMode:
    if mode is None:
        return DEFAULT_TEMPLATE_MODE
    if isinstance(mode, TemplateMode):
        return mode
    if isinstance(mode, str):
        return TemplateMode(mode.strip().lower())
    raise TypeError("template_mode must be TemplateMode or string")


def seed_sequence_to_int(ss: np.random.SeedSequence) -> int:
    return int(int(ss.generate_state(1)[0]) & 0x7FFFFFFF)


# ---------------------------------------------------------------------------
# Standardization / covariate resolution
# ---------------------------------------------------------------------------
def _get_x_standardization_info(name: str) -> Optional[Dict[str, Any]]:
    if name in X_STANDARDIZATION:
        return X_STANDARDIZATION[name]
    norm = _norm_name(name)
    for key, value in X_STANDARDIZATION.items():
        if _norm_name(key) == norm:
            return value
    return None


def _standardize_x_value(name: str, raw_value: float) -> float:
    """
    Apply X_STANDARDIZATION to a raw covariate value.

    If shift/scale are None, the value is treated as non-standardized
    categorical/ordinal encoding and returned unchanged.
    """
    info = _get_x_standardization_info(name)
    if info is None:
        return float(raw_value)

    shift = info.get("shift", None)
    scale = info.get("scale", None)

    if shift is None or scale is None:
        return float(raw_value)

    shift = float(shift)
    scale = float(scale)

    if not math.isfinite(shift) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"Invalid X_STANDARDIZATION shift/scale for '{name}': "
            f"shift={shift}, scale={scale}"
        )

    return (float(raw_value) - shift) / scale


def _resolve_model_value(
    model_name: str,
    norm_resolved: Dict[str, Any],
) -> float:
    """
    Resolve a model-scale covariate value.

    Resolution order:
    1. If the standardized/model column itself is present, use it as-is.
       This prevents double standardization.
    2. If raw_col from X_STANDARDIZATION is present, standardize it.
    3. Try _RAW_TO_STD aliases.

    Raises KeyError if the covariate cannot be resolved.
    """
    norm_model = _norm_name(model_name)

    # Direct model-scale value has priority.
    if norm_model in norm_resolved:
        return float(norm_resolved[norm_model])

    info = _get_x_standardization_info(model_name)

    # If this is a known standardized column, try its raw column.
    if info is not None:
        raw_col = str(info.get("raw_col", "")).strip()
        if raw_col:
            norm_raw = _norm_name(raw_col)
            if norm_raw in norm_resolved:
                raw_val = float(norm_resolved[norm_raw])
                return _standardize_x_value(model_name, raw_val)

    # Try aliases.
    for raw_name, std_name in _RAW_TO_STD.items():
        if _norm_name(std_name) == norm_model:
            norm_raw = _norm_name(raw_name)
            if norm_raw in norm_resolved:
                raw_val = float(norm_resolved[norm_raw])
                if info is not None:
                    return _standardize_x_value(model_name, raw_val)
                return raw_val

    raise KeyError(
        f"Covariate '{model_name}' not found in template or covariates."
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_peaks(
    peaks: List[float],
    training_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Validate peaks, including optional training range check.
    """
    if not isinstance(peaks, (list, tuple, np.ndarray)):
        raise TypeError("peaks must be a list/tuple/array of numeric values")

    if len(peaks) == 0:
        raise ValueError("peaks must contain at least one value")

    pmin = None
    pmax = None

    if training_meta is not None and isinstance(training_meta, dict):
        pmin = training_meta.get("peakload_min")
        pmax = training_meta.get("peakload_max")

        try:
            if pmin is not None and pmax is not None:
                pmin = float(pmin)
                pmax = float(pmax)
                if not (math.isfinite(pmin) and math.isfinite(pmax)):
                    pmin = pmax = None
                elif pmin > pmax:
                    logger.warning(
                        "Invalid training_meta peak range: min=%s > max=%s. "
                        "Range check skipped.",
                        pmin,
                        pmax,
                    )
                    pmin = pmax = None
        except Exception:
            pmin = pmax = None

    for p in peaks:
        if not _is_real_number(p):
            raise ValueError(f"PeakLoad value is not numeric: {p}")

        p_float = float(p)

        if not math.isfinite(p_float):
            raise ValueError(f"PeakLoad value is not finite: {p}")

        if pmin is not None and pmax is not None:
            if (
                p_float < pmin - PEAK_RANGE_TOLERANCE
                or p_float > pmax + PEAK_RANGE_TOLERANCE
            ):
                raise ValueError(
                    f"PeakLoad {p_float} outside training range "
                    f"[{pmin}, {pmax}] "
                    f"with tolerance {PEAK_RANGE_TOLERANCE}"
                )


def _validate_prediction_args(
    time_horizon: float,
    theta: float,
    sum_insured: float,
    alpha: float,
) -> None:
    if not (
        _is_real_number(time_horizon)
        and np.isfinite(time_horizon)
        and float(time_horizon) > 0.0
    ):
        raise ValueError("time_horizon must be a positive finite number")

    if not (
        _is_real_number(theta)
        and np.isfinite(theta)
        and float(theta) >= 0.0
    ):
        raise ValueError("theta must be a finite non-negative number")

    if not (
        _is_real_number(sum_insured)
        and np.isfinite(sum_insured)
        and float(sum_insured) > 0.0
    ):
        raise ValueError("sum_insured must be a positive finite number")

    if not (_is_real_number(alpha) and 0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1)")


def _validate_policy_horizon(policy_horizon_days: Optional[float]) -> None:
    if policy_horizon_days is None:
        return

    if not (
        _is_real_number(policy_horizon_days)
        and np.isfinite(policy_horizon_days)
        and float(policy_horizon_days) > 0.0
    ):
        raise ValueError("policy_horizon_days must be positive finite or None")


def _validate_discount_rate(discount_rate: float) -> None:
    if not (_is_real_number(discount_rate) and np.isfinite(discount_rate)):
        raise ValueError("discount_rate must be a finite number")

    if discount_rate < 0.0 or discount_rate >= 1.0:
        raise ValueError("discount_rate must be in [0, 1)")


def _validate_bootstrap_args(
    B: int,
    n_jobs: int,
    min_success_frac: float,
    min_events: int,
) -> None:
    if not (_is_integral_number(B) and int(B) >= 3):
        raise ValueError("B must be an integer >= 3 for meaningful bootstrap CI")

    if not (_is_integral_number(n_jobs) and int(n_jobs) >= 1):
        raise ValueError("n_jobs must be an integer >= 1")

    if not (_is_real_number(min_success_frac) and 0.0 <= float(min_success_frac) <= 1.0):
        raise ValueError("min_success_frac must be in [0, 1]")

    if not (_is_integral_number(min_events) and int(min_events) >= 1):
        raise ValueError("min_events must be an integer >= 1")


# ---------------------------------------------------------------------------
# CF result validation / compatibility
# ---------------------------------------------------------------------------
def _ensure_cf_result(cf_res: Any) -> Any:
    """
    Duck-typing validation for CFModelResult-like objects.

    This avoids hard dependence on a particular class identity, which is
    important when CFModelResult is dynamically imported or fallback-defined.
    """
    if cf_res is None:
        raise ModelCompatibilityError("CFModelResult is None")

    if not hasattr(cf_res, "cph"):
        raise ModelCompatibilityError("CFModelResult-like object missing 'cph'")

    if not hasattr(cf_res.cph, "predict_partial_hazard"):
        raise ModelCompatibilityError(
            "cph must implement predict_partial_hazard"
        )

    return cf_res


# ---------------------------------------------------------------------------
# Baseline survival
# ---------------------------------------------------------------------------
def baseline_survival_at_breslow(
    cph: CoxPHFitter,
    t: float,
    allow_extrapolation: bool = False,
) -> float:
    """
    Return S0(t) using baseline_cumulative_hazard_ (Breslow estimator).
    Uses step-function interpolation with searchsorted.
    Extrapolation beyond the last baseline time is disabled by default.
    """
    if (
        not hasattr(cph, "baseline_cumulative_hazard_")
        or cph.baseline_cumulative_hazard_ is None
    ):
        raise ModelCompatibilityError(
            "CoxPHFitter missing baseline_cumulative_hazard_; "
            "cannot compute S0(t)."
        )
    H0 = cph.baseline_cumulative_hazard_
    if H0.shape[1] == 0:
        raise ModelCompatibilityError("baseline_cumulative_hazard_ is empty")
    if H0.shape[1] > 1:
        raise NotImplementedError(
            "Stratified Cox models with multiple baseline hazards are not "
            "supported by baseline_survival_at_breslow()."
        )
    times = np.asarray(H0.index, dtype=float)
    if times.size == 0:
        raise ModelCompatibilityError("baseline_cumulative_hazard_ is empty")
    if not np.all(np.isfinite(times)):
        raise ModelCompatibilityError(
            "baseline_cumulative_hazard_ index contains non-finite times"
        )
    hvals = H0.iloc[:, 0].to_numpy(dtype=float)
    if not np.all(np.isfinite(hvals)):
        raise ModelCompatibilityError(
            "baseline_cumulative_hazard_ contains non-finite values"
        )
    idx = int(np.searchsorted(times, float(t), side="right") - 1)
    if idx < 0:
        Ht = 0.0
    elif idx >= len(hvals):
        if not allow_extrapolation:
            raise ModelCompatibilityError(
                f"time_horizon {t} exceeds max baseline time {times[-1]}. "
                "Baseline extrapolation is disabled. "
                "Set allow_baseline_extrapolation=True in training_meta to allow."
            )
        Ht = float(hvals[-1])
    else:
        Ht = float(hvals[idx])
    if not math.isfinite(Ht):
        raise ModelCompatibilityError("Non-finite baseline cumulative hazard")
    Ht = max(0.0, min(Ht, MAX_BASELINE_H))
    return float(math.exp(-Ht))


# ---------------------------------------------------------------------------
# First-stage prediction
# ---------------------------------------------------------------------------
def predict_first_stage(
    template: PredictionTemplate,
    fitted_first_stage: Any,
    covariates: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute predicted first-stage value using template covariates and
    fitted_first_stage, optionally overridden by user covariates.
    """
    resolved = dict(template.covariates)

    if covariates:
        resolved.update({str(k): v for k, v in covariates.items()})

    norm_resolved = _normalized_lookup(resolved)

    exog_names = getattr(
        getattr(fitted_first_stage, "model", None),
        "exog_names",
        None,
    )

    if exog_names is None:
        design_row = {
            "const": 1.0,
            "Z": _resolve_model_value("Z", norm_resolved),
            "X": _resolve_model_value("X", norm_resolved),
        }
    else:
        design_row = {}

        for n in exog_names:
            n_str = str(n)
            n_norm = _norm_name(n_str)

            if n_norm in {"const", "intercept"}:
                design_row[n_str] = 1.0
                continue

            design_row[n_str] = _resolve_model_value(n_str, norm_resolved)

    pred_df = pd.DataFrame([design_row])
    pred_raw = fitted_first_stage.predict(pred_df)

    pred_val = (
        float(np.asarray(pred_raw).reshape(-1)[0])
        if hasattr(pred_raw, "__len__")
        else float(pred_raw)
    )

    if not math.isfinite(pred_val):
        raise PredictionError("First-stage prediction produced non-finite value")

    return pred_val


# ---------------------------------------------------------------------------
# First-stage coefficients
# ---------------------------------------------------------------------------
def _get_first_stage_coefs(first_stage_model: Any, names: List[str]) -> np.ndarray:
    params = getattr(first_stage_model, "params", None)

    if params is None:
        raise ModelCompatibilityError(
            "first_stage_model has no 'params' attribute"
        )

    # pandas Series / array-like with index
    if hasattr(params, "index"):
        index_map = {_norm_name(str(idx)): idx for idx in list(params.index)}
        coefs: List[float] = []
        missing: List[str] = []

        for n in names:
            orig = index_map.get(_norm_name(str(n)))
            if orig is None:
                missing.append(str(n))
            else:
                coefs.append(float(params[orig]))

        if missing:
            raise ModelCompatibilityError(
                f"First-stage model params missing for columns: {missing}"
            )

        return np.asarray(coefs, dtype=float)

    # dict-like params
    if isinstance(params, dict):
        norm_params = {_norm_name(k): v for k, v in params.items()}
        coefs = []
        missing = []

        for n in names:
            key = _norm_name(n)
            if key not in norm_params:
                missing.append(str(n))
            else:
                coefs.append(float(norm_params[key]))

        if missing:
            raise ModelCompatibilityError(
                f"First-stage model params missing for columns: {missing}"
            )

        return np.asarray(coefs, dtype=float)

    # numpy array params: only safe if length matches names exactly
    if isinstance(params, np.ndarray):
        if params.size != len(names):
            raise ModelCompatibilityError(
                "First-stage params array length does not match exog_names"
            )
        return np.asarray(params, dtype=float).reshape(-1)

    raise ModelCompatibilityError(
        "Unsupported first_stage_model.params type; expected pandas Series, "
        "dict, or numpy array"
    )


# ---------------------------------------------------------------------------
# Transform helper
# ---------------------------------------------------------------------------
def _transform_values(
    values: np.ndarray,
    transform_info: Optional[Dict[str, Any]],
    require_transform: bool = True,
) -> Tuple[np.ndarray, str, float, float]:
    """
    Apply PeakLoad transform.

    Returns:
        transformed_values, transform_type, center, scale
    """
    values = np.asarray(values, dtype=float)

    if transform_info is None:
        if require_transform:
            raise ModelCompatibilityError(
                "transform_info is required for this model. "
                "Model was trained with PeakLoad transformation. "
                "Set require_transform=False to use raw values (not recommended)."
            )
        logger.warning(
            "transform_info is None, using raw values (may cause scale mismatch)"
        )
        return values, "none", 0.0, 1.0

    if not isinstance(transform_info, dict):
        raise ValueError("transform_info must be a dict or None")

    transform_type = str(transform_info.get("type", "none")).strip().lower()

    if transform_type == "none":
        return values, transform_type, 0.0, 1.0

    if transform_type == "standardize":
        center = float(transform_info.get("center", 0.0))
        scale = float(transform_info.get("scale", 1.0))

        if not math.isfinite(center):
            raise ValueError("transform_info.center must be finite")

        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("transform_info.scale must be positive finite")

        return (values - center) / scale, transform_type, center, scale

    if transform_type == "center":
        center = float(transform_info.get("center", 0.0))

        if not math.isfinite(center):
            raise ValueError("transform_info.center must be finite")

        return values - center, transform_type, center, 1.0

    raise ValueError(f"Unsupported transform_info.type: '{transform_type}'")


# ---------------------------------------------------------------------------
# CF basis construction at prediction time
# ---------------------------------------------------------------------------
def _get_cf_column_std_params(
    col_name: str,
    std_params_raw: Dict[str, Any],
) -> Tuple[float, float]:
    info = std_params_raw.get(col_name, {})

    if not isinstance(info, dict):
        info = {}

    try:
        col_mean = float(info.get("mean", 0.0))
    except Exception:
        col_mean = 0.0

    try:
        col_std = float(info.get("std", 1.0))
    except Exception:
        col_std = 1.0

    if not math.isfinite(col_mean):
        col_mean = 0.0

    if not math.isfinite(col_std) or col_std <= 0.0:
        col_std = 1.0

    return col_mean, col_std


def _build_cf_basis_at_prediction(
    residuals_arr: np.ndarray,
    cf_basis_metadata: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Build CF basis columns at prediction time using training metadata.

    Strict behaviour:
    - linear uses training residuals_mean/residuals_std when present;
      otherwise logs warning and uses 0/1, not batch-dependent statistics.
    - spline requires training knots/domain metadata.
    - spline does not silently fallback to powers.
    """
    result: Dict[str, np.ndarray] = {}
    residuals_arr = np.asarray(residuals_arr, dtype=float)

    basis_type = str(cf_basis_metadata.get("v_hat_basis", "linear")).strip().lower()

    r_mean_raw = cf_basis_metadata.get("residuals_mean", None)
    r_std_raw = cf_basis_metadata.get("residuals_std", None)

    if r_mean_raw is None or not np.isfinite(float(r_mean_raw)):
        logger.warning(
            "cf_basis_metadata missing valid residuals_mean; using 0.0"
        )
        r_mean = 0.0
    else:
        r_mean = float(r_mean_raw)

    if (
        r_std_raw is None
        or not np.isfinite(float(r_std_raw))
        or float(r_std_raw) <= 0.0
    ):
        logger.warning(
            "cf_basis_metadata missing valid residuals_std; using 1.0"
        )
        r_std = 1.0
    else:
        r_std = float(r_std_raw)

    linear_standardized = bool(cf_basis_metadata.get("linear_standardized", True))

    # ------------------------------------------------------------------
    # Linear
    # ------------------------------------------------------------------
    if basis_type == "linear":
        if linear_standardized:
            result["v_hat"] = (residuals_arr - r_mean) / r_std
        else:
            result["v_hat"] = residuals_arr - r_mean
        return result

    # ------------------------------------------------------------------
    # Spline basis
    # ------------------------------------------------------------------
    if basis_type == "spline":
        v_std = (residuals_arr - r_mean) / r_std

        try:
            degree = int(cf_basis_metadata.get("spline_degree", 2))
        except Exception:
            degree = 2

        if degree < 1:
            raise ModelCompatibilityError("spline_degree must be >= 1")

        try:
            interior_knots = sorted(
                float(k)
                for k in cf_basis_metadata.get("knots", [])
                if np.isfinite(float(k))
            )
        except Exception as exc:
            raise ModelCompatibilityError(
                "Invalid spline knots in cf_basis_metadata"
            ) from exc

        if not interior_knots:
            raise ModelCompatibilityError(
                "Spline basis requires training knots in cf_basis_metadata"
            )

        domain_min = cf_basis_metadata.get("spline_domain_min", None)
        domain_max = cf_basis_metadata.get("spline_domain_max", None)

        if domain_min is None or domain_max is None:
            raise ModelCompatibilityError(
                "Spline basis requires spline_domain_min and spline_domain_max"
            )

        try:
            dmin = float(domain_min)
            dmax = float(domain_max)
        except Exception as exc:
            raise ModelCompatibilityError(
                "Invalid spline domain in cf_basis_metadata"
            ) from exc

        if not (math.isfinite(dmin) and math.isfinite(dmax) and dmax > dmin):
            raise ModelCompatibilityError(
                "Invalid spline domain in cf_basis_metadata"
            )

        interior_knots = sorted(k for k in interior_knots if dmin < k < dmax)

        t = [dmin] * (degree + 1) + interior_knots + [dmax] * (degree + 1)
        n_basis = len(t) - degree - 1

        if n_basis <= 0:
            raise ModelCompatibilityError(
                "Invalid spline configuration: n_basis <= 0"
            )

        try:
            from scipy import interpolate as _scipy_interpolate
        except Exception as exc:
            raise ModelCompatibilityError(
                "scipy is required for spline CF basis"
            ) from exc

        try:
            eye = np.eye(n_basis)
            basis_funcs = []
            x_clipped = np.clip(v_std, dmin, dmax)

            for j in range(n_basis):
                bs = _scipy_interpolate.BSpline(
                    t,
                    eye[j],
                    k=degree,
                    extrapolate=True,
                )
                basis_funcs.append(bs(x_clipped))

            basis_vals = np.column_stack(basis_funcs)
            basis_vals = np.nan_to_num(
                basis_vals,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        except Exception as exc:
            raise ModelCompatibilityError(
                f"Spline basis evaluation failed: {exc}"
            ) from exc

        drop_first = bool(cf_basis_metadata.get("spline_drop_first", True))
        if drop_first and basis_vals.shape[1] > 1:
            basis_vals = basis_vals[:, 1:]

        if basis_vals.shape[1] == 0:
            raise ModelCompatibilityError(
                "Spline basis produced zero columns after intercept removal"
            )

        expected_columns = cf_basis_metadata.get("v_hat_column_names", None)
        if (
            expected_columns is not None
            and isinstance(expected_columns, (list, tuple))
            and len(expected_columns) == basis_vals.shape[1]
        ):
            col_names = [str(x) for x in expected_columns]
        else:
            if expected_columns is not None:
                logger.warning(
                    "cf_basis_metadata.v_hat_column_names does not match "
                    "computed spline basis size; using default names"
                )
            col_names = [f"v_hat_s{j}" for j in range(basis_vals.shape[1])]

        std_params_raw = cf_basis_metadata.get("v_hat_col_std_params", {})

        if isinstance(std_params_raw, list):
            std_params_raw = {
                f"v_hat_s{i}": p for i, p in enumerate(std_params_raw)
            }

        if not isinstance(std_params_raw, dict):
            std_params_raw = {}

        for j, col_name in enumerate(col_names):
            col_mean, col_std = _get_cf_column_std_params(
                col_name,
                std_params_raw,
            )
            vals = (basis_vals[:, j] - col_mean) / col_std
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            result[col_name] = vals

        return result

    # ------------------------------------------------------------------
    # Powers basis
    # ------------------------------------------------------------------
    if basis_type == "powers":
        v_std = (residuals_arr - r_mean) / r_std

        try:
            max_power = int(cf_basis_metadata.get("max_power", 2))
        except Exception:
            logger.warning("cf_basis_metadata.max_power invalid; using 2")
            max_power = 2

        max_power = max(1, max_power)

        # Numerical guard against overflow before exponentiation.
        v_std = np.clip(v_std, -10.0, 10.0)

        basis_vals = np.column_stack(
            [v_std**p for p in range(1, max_power + 1)]
        )

        expected_columns = cf_basis_metadata.get("v_hat_column_names", None)
        if (
            expected_columns is not None
            and isinstance(expected_columns, (list, tuple))
            and len(expected_columns) == basis_vals.shape[1]
        ):
            col_names = [str(x) for x in expected_columns]
        else:
            if expected_columns is not None:
                logger.warning(
                    "cf_basis_metadata.v_hat_column_names does not match "
                    "computed powers basis size; using default names"
                )
            col_names = [f"v_hat_pow{j + 1}" for j in range(basis_vals.shape[1])]

        std_params_raw = cf_basis_metadata.get("v_hat_col_std_params", {})

        if isinstance(std_params_raw, list):
            std_params_raw = {
                f"v_hat_pow{i + 1}": p for i, p in enumerate(std_params_raw)
            }

        if not isinstance(std_params_raw, dict):
            std_params_raw = {}

        for j, col_name in enumerate(col_names):
            col_mean, col_std = _get_cf_column_std_params(
                col_name,
                std_params_raw,
            )
            vals = (basis_vals[:, j] - col_mean) / col_std
            vals = np.clip(vals, -10.0, 10.0)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            result[col_name] = vals

        return result

    raise ValueError(f"Unknown CF basis type: '{basis_type}'")


# ---------------------------------------------------------------------------
# Vectorized probability computation
# ---------------------------------------------------------------------------
def _compute_probabilities_vectorized(
    cf_res: Any,
    template: PredictionTemplate,
    pred_fs_value: float,
    peaks: List[float],
    time_horizon: float,
    residual_policy: ResidualPolicy,
    params: Optional[Any] = None,
    partial_out_all_betas: Optional[Dict[str, float]] = None,
    training_x_means: Optional[Dict[str, float]] = None,
    training_pl_hat_mean: Optional[float] = None,
    first_stage_model: Any = None,
    transform_info: Optional[Dict[str, Any]] = None,
    covariates: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Compute probabilities 1 - S_i(t) for each peak.
    """
    del params  # kept for API compatibility

    cf_res = _ensure_cf_result(cf_res)
    residual_policy = _coerce_residual_policy(residual_policy)

    cf_metadata = getattr(cf_res, "cf_basis_metadata", None)

    training_meta = getattr(cf_res, "training_meta", None)
    if training_meta is None and isinstance(cf_metadata, dict):
        training_meta = cf_metadata

    validate_peaks(
        peaks,
        training_meta if isinstance(training_meta, dict) else None,
    )

    # Optional explicit time-unit contract.
    time_unit = getattr(cf_res, "time_unit", None)
    if time_unit is None and isinstance(cf_metadata, dict):
        time_unit = cf_metadata.get("time_unit", None)

    # Единая проверка time_unit — объединяет оба набора допустимых значений
    ALLOWED_TIME_UNITS = frozenset({
        "days", "hours", "engine_hours", "engine_hour", "mch", "моточасы",
    })
    if time_unit is not None and str(time_unit).strip().lower() not in ALLOWED_TIME_UNITS:
        raise NotImplementedError(
            f"Unsupported model time unit: '{time_unit}'. "
            "This prediction API expects time_horizon in days or engine_hours."
        )

    # time_horizon уже в единицах модели (например, моточасы).
    # Базовая опасность обучена в тех же единицах, поэтому конвертация не нужна.
    time_horizon_model = time_horizon

    n_peaks = len(peaks)

    resolved_covariates = dict(template.covariates)
    if covariates:
        resolved_covariates.update({str(k): v for k, v in covariates.items()})

    norm_resolved = _normalized_lookup(resolved_covariates)

    peaks_arr = np.asarray(peaks, dtype=float)
    peaks_std, transform_type, center, scale = _transform_values(
        peaks_arr,
        transform_info,
    )

    # ------------------------------------------------------------------
    # First-stage prediction vector
    # ------------------------------------------------------------------
    if first_stage_model is not None:
        exog_names = getattr(
            getattr(first_stage_model, "model", None),
            "exog_names",
            None,
        )

        if exog_names is None:
            raise ModelCompatibilityError(
                "first_stage_model.model.exog_names is required for "
                "vectorized first-stage design construction"
            )

        exog_names = [str(x) for x in exog_names]
        design_rows: Dict[str, np.ndarray] = {}

        for name in exog_names:
            name_norm = _norm_name(name)

            if name_norm in {"const", "intercept"}:
                design_rows[name] = np.ones(n_peaks, dtype=float)
                continue

            try:
                val = _resolve_model_value(name, norm_resolved)
            except KeyError as exc:
                raise KeyError(
                    f"First-stage covariate '{name}' required by model "
                    f"not found in template or covariates."
                ) from exc

            design_rows[name] = np.full(n_peaks, float(val), dtype=float)

        design_matrix = np.column_stack([design_rows[n] for n in exog_names])
        first_stage_coefs = _get_first_stage_coefs(
            first_stage_model,
            exog_names,
        )

        pl_hat_raw = design_matrix @ first_stage_coefs
    else:
        pl_hat_raw = np.full(n_peaks, float(pred_fs_value), dtype=float)

    if not np.all(np.isfinite(pl_hat_raw)):
        raise PredictionError("First-stage predicted values contain non-finite values")

    # Transform first-stage predictions to the same scale as peaks_std.
    if transform_type == "standardize":
        pl_hat_model = (pl_hat_raw - center) / scale
    elif transform_type == "center":
        pl_hat_model = pl_hat_raw - center
    else:
        pl_hat_model = pl_hat_raw

    if not np.all(np.isfinite(pl_hat_model)):
        raise PredictionError(
            "Transformed first-stage predicted values contain non-finite values"
        )

    # ------------------------------------------------------------------
    # Compute PL_hat_exog with ALL partial-out betas
    # ------------------------------------------------------------------
    pl_mean = (
        training_pl_hat_mean
        if training_pl_hat_mean is not None
        else getattr(cf_res, "training_pl_hat_mean", 0.0)
    )
    pl_mean = float(pl_mean)

    active_betas = dict(getattr(cf_res, "partial_out_all_betas", {}) or {})
    active_x_means = dict(getattr(cf_res, "training_x_means", {}) or {})

    if partial_out_all_betas is not None:
        active_betas = dict(partial_out_all_betas)

    if training_x_means is not None:
        active_x_means = dict(training_x_means)

    active_betas = {str(k): float(v) for k, v in active_betas.items()}
    active_x_means_norm = _normalized_lookup(active_x_means)

    pl_hat_exog_arr = pl_hat_model - pl_mean

    if active_betas:
        for col_name, beta_part in active_betas.items():
            col_norm = _norm_name(col_name)

            if col_norm in {"const", "intercept"}:
                continue

            if not math.isfinite(float(beta_part)):
                raise PredictionError(
                    f"partial_out_all_betas['{col_name}'] is not finite"
                )

            x_mean = float(active_x_means_norm.get(col_norm, 0.0))

            try:
                x_val = _resolve_model_value(col_name, norm_resolved)
            except KeyError:
                logger.warning(
                    "Partial-out covariate '%s' not found in template/covariates. "
                    "Using training mean %.6f, so this term contributes zero.",
                    col_name,
                    x_mean,
                )
                x_val = x_mean

            pl_hat_exog_arr -= float(beta_part) * (float(x_val) - x_mean)

    if not np.all(np.isfinite(pl_hat_exog_arr)):
        raise PredictionError("pl_hat_exog contains non-finite values")

    # ------------------------------------------------------------------
    # Compute residuals according to policy
    # ------------------------------------------------------------------
    if residual_policy == ResidualPolicy.PLUG_IN:
        residuals_arr = peaks_std - pl_hat_model
    elif residual_policy == ResidualPolicy.MEAN:
        r_mean = 0.0
        if isinstance(cf_metadata, dict):
            try:
                r_mean = float(cf_metadata.get("residuals_mean", 0.0))
            except Exception:
                r_mean = 0.0
        residuals_arr = np.full(n_peaks, r_mean, dtype=float)
    else:
        residuals_arr = np.zeros(n_peaks, dtype=float)

    if not np.all(np.isfinite(residuals_arr)):
        raise PredictionError("CF residuals contain non-finite values")

    # ------------------------------------------------------------------
    # Validate Cox model
    # ------------------------------------------------------------------
    if not hasattr(cf_res.cph, "predict_partial_hazard"):
        raise ModelCompatibilityError(
            "CoxPHFitter must implement predict_partial_hazard."
        )

    if not hasattr(cf_res.cph, "params_"):
        raise ModelCompatibilityError(
            "CoxPHFitter appears unfitted: missing params_"
        )

    cov_names = [str(x) for x in cf_res.cph.params_.index]

    if len(set(cov_names)) != len(cov_names):
        raise ModelCompatibilityError(
            "Duplicate covariate names found in cph.params_.index"
        )

    # ------------------------------------------------------------------
    # Build CF basis at prediction time
    # ------------------------------------------------------------------
    cf_basis_values: Dict[str, np.ndarray] = {}

    if isinstance(cf_metadata, dict) and residuals_arr.size > 0:
        cf_basis_values = _build_cf_basis_at_prediction(
            residuals_arr,
            cf_metadata,
        )

    # ------------------------------------------------------------------
    # Determine Cox PeakLoad convention
    # ------------------------------------------------------------------
    cox_convention = "observed_peakload"

    if isinstance(cf_metadata, dict):
        conv = str(
            cf_metadata.get("cox_peakload_convention", "observed_peakload")
        ).strip().lower()

        if conv in {"observed_peakload", "pl_hat_exog"}:
            cox_convention = conv
        else:
            logger.warning(
                "Unknown cox_peakload_convention '%s'; using observed_peakload",
                conv,
            )

    # ------------------------------------------------------------------
    # Build prediction rows
    # ------------------------------------------------------------------
    rows: List[Dict[str, Any]] = []

    for i in range(n_peaks):
        row: Dict[str, Any] = {}

        for cn in cov_names:
            cn_norm = _norm_name(cn)

            if cn == "PeakLoad" or cn_norm == "peakload":
                if cox_convention == "pl_hat_exog":
                    row[cn] = float(pl_hat_exog_arr[i])
                else:
                    row[cn] = float(peaks_std[i])
                continue

            if cn in cf_basis_values:
                row[cn] = float(cf_basis_values[cn][i])
                continue

            if cn == "v_hat":
                r_mean = 0.0
                r_std = 1.0
                lin_std = True

                if isinstance(cf_metadata, dict):
                    try:
                        r_mean = float(cf_metadata.get("residuals_mean", 0.0))
                    except Exception:
                        r_mean = 0.0

                    try:
                        r_std = float(cf_metadata.get("residuals_std", 1.0))
                    except Exception:
                        r_std = 1.0

                    if not math.isfinite(r_std) or r_std <= 0.0:
                        r_std = 1.0

                    lin_std = bool(cf_metadata.get("linear_standardized", True))

                if lin_std:
                    row[cn] = float((residuals_arr[i] - r_mean) / r_std)
                else:
                    row[cn] = float(residuals_arr[i] - r_mean)
                continue

            try:
                row[cn] = _resolve_model_value(cn, norm_resolved)
            except KeyError as exc:
                raise KeyError(
                    f"Covariate '{cn}' required by CF Cox model not present "
                    f"in template or covariates."
                ) from exc

        rows.append(row)

    df_rows = pd.DataFrame(rows)
    df_rows = df_rows[cov_names]

    if not np.all(np.isfinite(df_rows.to_numpy(dtype=float))):
        raise PredictionError(
            "Cox design matrix contains non-finite values"
        )

    ph_raw = cf_res.cph.predict_partial_hazard(df_rows)
    ph_arr = np.asarray(ph_raw, dtype=float).reshape(-1)

    if ph_arr.size != n_peaks:
        raise PredictionError(
            "predict_partial_hazard returned unexpected number of rows"
        )

    if not np.all(np.isfinite(ph_arr)):
        raise PredictionError("predict_partial_hazard returned non-finite values")

    # ------------------------------------------------------------------
    # Probability with expm1 for numerical stability
    # ------------------------------------------------------------------
    allow_extrapolation = False
    if isinstance(training_meta, dict):
        allow_extrapolation = bool(training_meta.get("allow_baseline_extrapolation", False))
    s0 = baseline_survival_at_breslow(
        cf_res.cph,
        float(time_horizon_model),
        allow_extrapolation=allow_extrapolation,
    )

    if s0 <= 0.0:
        probs = np.ones(n_peaks, dtype=float)
    else:
        log_s0 = math.log(max(s0, 1e-300))
        cumulative_hazard = ph_arr * (-log_s0)
        cumulative_hazard = np.clip(cumulative_hazard, 0.0, MAX_BASELINE_H)
        probs = -np.expm1(-cumulative_hazard)
        probs = np.clip(probs, 0.0, 1.0)

    if not np.all(np.isfinite(probs)):
        raise PredictionError("Non-finite probabilities produced.")

    return probs


# ---------------------------------------------------------------------------
# Public prediction
# ---------------------------------------------------------------------------
def predict_premiums(
    cf_res: Any,
    fitted_first_stage: Any,
    template: PredictionTemplate,
    peaks: List[float],
    time_horizon: float,
    sum_insured: float,
    theta: float,
    residual_policy: ResidualPolicy = DEFAULT_RESIDUAL_POLICY,
    alpha: float = DEFAULT_ALPHA,
    discount_rate: float = 0.0,
    policy_horizon_days: Optional[float] = None,
    partial_out_all_betas: Optional[Dict[str, float]] = None,
    training_x_means: Optional[Dict[str, float]] = None,
    training_pl_hat_mean: Optional[float] = None,
    transform_info: Optional[Dict[str, Any]] = None,
    covariates: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute predicted probabilities and premiums for the given peaks.
    """
    residual_policy = _coerce_residual_policy(residual_policy)

    _validate_prediction_args(time_horizon, theta, sum_insured, alpha)
    _validate_policy_horizon(policy_horizon_days)
    _validate_discount_rate(discount_rate)
    validate_peaks(peaks)

    pred_fs_val = predict_first_stage(
        template,
        fitted_first_stage,
        covariates=covariates,
    )

    probs = _compute_probabilities_vectorized(
        cf_res,
        template,
        pred_fs_val,
        peaks,
        time_horizon,
        residual_policy,
        partial_out_all_betas=partial_out_all_betas,
        training_x_means=training_x_means,
        training_pl_hat_mean=training_pl_hat_mean,
        first_stage_model=fitted_first_stage,
        transform_info=transform_info,
        covariates=covariates,
    )

    sum_insured = float(sum_insured)
    theta = float(theta)
    discount_rate = float(discount_rate)

    discount_horizon = (
        float(policy_horizon_days)
        if policy_horizon_days is not None
        else float(time_horizon)
    )

    try:
        from premium_engine import calculate_single_premium
        use_premium_engine = True
    except ImportError:
        calculate_single_premium = None
        use_premium_engine = False

    rows = []

    for i, p in enumerate(peaks):
        prob = float(probs[i])

        if not (0.0 <= prob <= 1.0):
            raise PredictionError(f"Probability outside [0, 1]: {prob}")

        if use_premium_engine:
            premium = calculate_single_premium(
                prob,
                sum_insured,
                theta,
                discount_rate=discount_rate,
                calibration_horizon_days=discount_horizon,
                policy_horizon_days=policy_horizon_days,
            )

            rows.append(
                {
                    "PeakLoad": float(p),
                    "probability": prob,
                    "net_undiscounted": premium["net_undiscounted"],
                    "net_discounted": premium["net_discounted"],
                    "gross_undiscounted": premium["gross_undiscounted"],
                    "gross_discounted": premium["gross_discounted"],
                    "net_premium": premium["net_discounted"],
                    "gross_premium": premium["gross_discounted"],
                    "tariff_pct": premium["tariff"],
                    "discount_factor": premium["discount_factor"],
                    "loading_amount": premium["loading_amount"],
                }
            )
        else:
            net_undiscounted = prob * sum_insured

            if discount_rate > 0.0:
                df = math.exp(-discount_rate * discount_horizon / 365.0)
                net_discounted = net_undiscounted * df
            else:
                df = 1.0
                net_discounted = net_undiscounted

            gross_undiscounted = net_undiscounted * (1.0 + theta)
            gross_discounted = net_discounted * (1.0 + theta)
            tariff = gross_discounted / sum_insured * 100.0

            rows.append(
                {
                    "PeakLoad": float(p),
                    "probability": prob,
                    "net_undiscounted": float(net_undiscounted),
                    "net_discounted": float(net_discounted),
                    "gross_undiscounted": float(gross_undiscounted),
                    "gross_discounted": float(gross_discounted),
                    "net_premium": float(net_discounted),
                    "gross_premium": float(gross_discounted),
                    "tariff_pct": float(tariff),
                    "discount_factor": float(df),
                    "loading_amount": float(gross_discounted - net_discounted),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap implementation
# ---------------------------------------------------------------------------
def _bootstrap_sample(
    data: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(data)

    if n == 0:
        return data.copy().reset_index(drop=True)

    idxs = rng.integers(0, n, size=n)
    return data.iloc[idxs].reset_index(drop=True)


def _fit_models_on_sample(
    data_b: pd.DataFrame,
    fit_first_stage_fn: FitFirstStageFn,
    fit_cf_fn: FitCFFn,
) -> Tuple[Any, Any]:
    fitted_fs_b, resid_b, _ = fit_first_stage_fn(
        data_b,
        return_design=True,
    )

    cf_res_b = fit_cf_fn(data_b, resid_b)
    _ensure_cf_result(cf_res_b)

    return fitted_fs_b, cf_res_b


def _worker_bootstrap(
    seed_int: int,
    data: pd.DataFrame,
    peaks: List[float],
    fit_first_stage_fn: FitFirstStageFn,
    fit_cf_fn: FitCFFn,
    template_mode: TemplateMode,
    template_original: PredictionTemplate,
    time_horizon: float,
    residual_policy: ResidualPolicy,
    min_events: int,
    transform_info: Optional[Dict[str, Any]] = None,
    covariates: Optional[Dict[str, float]] = None,
) -> BootstrapWorkerResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed_int))
    n_events: Optional[int] = None

    try:
        data_b = _bootstrap_sample(data, rng)

        if "event" not in data_b.columns:
            raise KeyError("Column 'event' not found in bootstrap sample")

        event_num = pd.to_numeric(data_b["event"], errors="coerce")

        if event_num.isna().any():
            raise ValueError(
                "Column 'event' contains NaN or non-numeric values"
            )

        n_events = int((event_num > 0).sum())

        if n_events < min_events:
            return BootstrapWorkerResult(
                success=False,
                probabilities=None,
                reason="too_few_events",
                exception_type="TooFewEvents",
                traceback=None,
                n_events=n_events,
                elapsed_seconds=time.perf_counter() - t0,
            )

        fitted_fs_b, cf_res_b = _fit_models_on_sample(
            data_b,
            fit_first_stage_fn,
            fit_cf_fn,
        )

        if template_mode == TemplateMode.BOOTSTRAP:
            template_b = PredictionTemplate.from_dataframe(
                fitted_fs_b,
                data_b,
            )
            pred_fs_val = predict_first_stage(
                template_b,
                fitted_fs_b,
                covariates=covariates,
            )

            probs = _compute_probabilities_vectorized(
                cf_res_b,
                template_b,
                pred_fs_val,
                peaks,
                time_horizon,
                residual_policy,
                partial_out_all_betas=getattr(
                    cf_res_b,
                    "partial_out_all_betas",
                    None,
                ),
                training_x_means=getattr(
                    cf_res_b,
                    "training_x_means",
                    None,
                ),
                training_pl_hat_mean=getattr(
                    cf_res_b,
                    "training_pl_hat_mean",
                    None,
                ),
                first_stage_model=fitted_fs_b,
                transform_info=transform_info,
                covariates=covariates,
            )
        else:
            pred_fs_val = predict_first_stage(
                template_original,
                fitted_fs_b,
                covariates=covariates,
            )

            probs = _compute_probabilities_vectorized(
                cf_res_b,
                template_original,
                pred_fs_val,
                peaks,
                time_horizon,
                residual_policy,
                partial_out_all_betas=getattr(
                    cf_res_b,
                    "partial_out_all_betas",
                    None,
                ),
                training_x_means=getattr(
                    cf_res_b,
                    "training_x_means",
                    None,
                ),
                training_pl_hat_mean=getattr(
                    cf_res_b,
                    "training_pl_hat_mean",
                    None,
                ),
                first_stage_model=fitted_fs_b,
                transform_info=transform_info,
                covariates=covariates,
            )

        return BootstrapWorkerResult(
            success=True,
            probabilities=probs,
            reason=None,
            exception_type=None,
            traceback=None,
            n_events=n_events,
            elapsed_seconds=time.perf_counter() - t0,
        )

    except Exception as exc:
        tb = traceback.format_exc()
        return BootstrapWorkerResult(
            success=False,
            probabilities=None,
            reason=str(exc),
            exception_type=type(exc).__name__,
            traceback=tb,
            n_events=n_events,
            elapsed_seconds=time.perf_counter() - t0,
        )


def bootstrap_premiums_ci(
    data: pd.DataFrame,
    peaks: List[float],
    fit_first_stage_fn: FitFirstStageFn,
    fit_cf_fn: FitCFFn,
    B: int = 1000,
    time_horizon: float = 214.0,
    sum_insured: float = 5e6,
    theta: float = 0.15,
    residual_policy: ResidualPolicy = DEFAULT_RESIDUAL_POLICY,
    seed: int = 12345,
    min_success_frac: float = 0.8,
    n_jobs: Optional[int] = 1,
    template_mode: TemplateMode = DEFAULT_TEMPLATE_MODE,
    min_events: int = 10,
    alpha: float = DEFAULT_ALPHA,
    bootstrap_method: str = "percentile",
    discount_rate: float = 0.0,
    policy_horizon_days: Optional[float] = None,
    transform_info: Optional[Dict[str, Any]] = None,
    covariates: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, BootstrapDiagnostics]:
    """
    Compute percentile bootstrap CIs for predicted premiums.
    """
    residual_policy = _coerce_residual_policy(residual_policy)
    template_mode = _coerce_template_mode(template_mode)

    if n_jobs is None:
        n_jobs = 1

    _validate_bootstrap_args(B, n_jobs, min_success_frac, min_events)
    _validate_prediction_args(time_horizon, theta, sum_insured, alpha)
    _validate_policy_horizon(policy_horizon_days)
    _validate_discount_rate(discount_rate)
    validate_peaks(peaks)

    if len(data) == 0:
        raise ValueError("data must contain at least one row")

    if bootstrap_method != "percentile":
        raise NotImplementedError(
            "Only 'percentile' bootstrap_method supported (TODO: BCa)"
        )

    peaks = [float(x) for x in peaks]

    t0 = time.perf_counter()

    fitted_fs_full, resid_full, _ = fit_first_stage_fn(
        data,
        return_design=True,
    )

    template_original = PredictionTemplate.from_dataframe(
        fitted_fs_full,
        data,
    )

    cf_res_full = fit_cf_fn(data, resid_full)
    _ensure_cf_result(cf_res_full)

    training_pl_hat_mean = getattr(cf_res_full, "training_pl_hat_mean", 0.0)

    point_df = predict_premiums(
        cf_res_full,
        fitted_fs_full,
        template_original,
        peaks,
        time_horizon,
        sum_insured,
        theta,
        residual_policy,
        alpha,
        discount_rate=discount_rate,
        policy_horizon_days=policy_horizon_days,
        partial_out_all_betas=getattr(
            cf_res_full,
            "partial_out_all_betas",
            None,
        ),
        training_x_means=getattr(
            cf_res_full,
            "training_x_means",
            None,
        ),
        training_pl_hat_mean=training_pl_hat_mean,
        transform_info=transform_info,
        covariates=covariates,
    )

    # Deterministic seeds
    base_ss = np.random.SeedSequence(int(seed))
    child_ss = base_ss.spawn(B)
    seeds = [seed_sequence_to_int(ss) for ss in child_ss]

    worker = partial(
        _worker_bootstrap,
        data=data,
        peaks=peaks,
        fit_first_stage_fn=fit_first_stage_fn,
        fit_cf_fn=fit_cf_fn,
        template_mode=template_mode,
        template_original=template_original,
        time_horizon=time_horizon,
        residual_policy=residual_policy,
        min_events=min_events,
        transform_info=transform_info,
        covariates=covariates,
    )

    if n_jobs == 1:
        results = [worker(s) for s in seeds]
    else:
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            prefer="processes",
            batch_size="auto",
        )(delayed(worker)(s) for s in seeds)

    # Aggregate
    runtimes: List[float] = []
    success_count = 0
    completed = len(results)
    fail_reasons: Dict[str, int] = {}
    error_examples: List[str] = []
    probs_list: List[np.ndarray] = []

    for r in results:
        runtimes.append(r.elapsed_seconds)

        if (
            r.success
            and r.probabilities is not None
            and r.probabilities.size == len(peaks)
            and np.all(np.isfinite(r.probabilities))
        ):
            success_count += 1
            probs_list.append(r.probabilities)
        else:
            key = r.reason or (r.exception_type or "unknown")

            if key.startswith("too_few_events"):
                key = "too_few_events"

            fail_reasons[key] = fail_reasons.get(key, 0) + 1

            if r.traceback and len(error_examples) < 5:
                error_examples.append(r.traceback)

    elapsed = time.perf_counter() - t0
    success_fraction = float(success_count) / float(B) if B > 0 else 0.0
    median_runtime = (
        float(np.median(np.array(runtimes))) if runtimes else None
    )

    diag = BootstrapDiagnostics(
        bootstrap_requested=B,
        bootstrap_completed=completed,
        bootstrap_successful=success_count,
        bootstrap_failed=B - success_count,
        success_fraction=success_fraction,
        fail_reasons=fail_reasons,
        error_examples=error_examples,
        elapsed_seconds=elapsed,
        median_runtime_per_rep=median_runtime,
    )

    min_threshold = max(1, min(B, int(math.ceil(min_success_frac * B))))

    if success_count < min_threshold:
        raise BootstrapFailure(diag)

    if len(probs_list) == 0:
        raise BootstrapFailure(diag)

    shapes = set(p.shape for p in probs_list)

    if len(shapes) != 1 or list(shapes)[0] != (len(peaks),):
        raise BootstrapFailure(diag)

    arr = np.vstack(probs_list)

    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    lo = np.percentile(arr, lower_q, axis=0)
    hi = np.percentile(arr, upper_q, axis=0)

    discount_horizon = (
        float(policy_horizon_days)
        if policy_horizon_days is not None
        else float(time_horizon)
    )

    try:
        from premium_engine import calculate_single_premium
        use_premium_engine = True
    except ImportError:
        calculate_single_premium = None
        use_premium_engine = False

    out_rows = []

    for idx, p in enumerate(peaks):
        prow = point_df.iloc[idx]

        prob_hat = float(prow["probability"])
        np_hat = float(prow["net_premium"])
        bp_hat = float(prow["gross_premium"])
        tariff = float(prow["tariff_pct"])

        prob_lo = float(lo[idx])
        prob_hi = float(hi[idx])

        if use_premium_engine:
            prem_lo = calculate_single_premium(
                prob_lo,
                float(sum_insured),
                float(theta),
                discount_rate=float(discount_rate),
                calibration_horizon_days=discount_horizon,
                policy_horizon_days=policy_horizon_days,
            )
            prem_hi = calculate_single_premium(
                prob_hi,
                float(sum_insured),
                float(theta),
                discount_rate=float(discount_rate),
                calibration_horizon_days=discount_horizon,
                policy_horizon_days=policy_horizon_days,
            )

            np_lo = float(prem_lo["net_discounted"])
            np_hi = float(prem_hi["net_discounted"])
            bp_lo = float(prem_lo["gross_discounted"])
            bp_hi = float(prem_hi["gross_discounted"])
            tariff_lo = float(prem_lo["tariff"])
            tariff_hi = float(prem_hi["tariff"])
        else:
            if discount_rate > 0.0:
                df = math.exp(-discount_rate * discount_horizon / 365.0)
            else:
                df = 1.0

            np_lo = prob_lo * float(sum_insured) * df
            np_hi = prob_hi * float(sum_insured) * df

            bp_lo = np_lo * (1.0 + float(theta))
            bp_hi = np_hi * (1.0 + float(theta))

            tariff_lo = bp_lo / float(sum_insured) * 100.0
            tariff_hi = bp_hi / float(sum_insured) * 100.0

        out_rows.append(
            {
                "PeakLoad": float(p),
                "probability": prob_hat,
                "prob_lo": prob_lo,
                "prob_hi": prob_hi,
                "net_premium": np_hat,
                "net_premium_lo": np_lo,
                "net_premium_hi": np_hi,
                "gross_premium": bp_hat,
                "gross_premium_lo": bp_lo,
                "gross_premium_hi": bp_hi,
                "tariff_pct": tariff,
                "tariff_lo_pct": tariff_lo,
                "tariff_hi_pct": tariff_hi,
            }
        )

    return pd.DataFrame(out_rows), diag