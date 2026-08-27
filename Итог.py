#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Итог.py (v3.2 / v0.2)
Monte Carlo IV Cox (Control Function / 2SRI) — simulation + estimation.

v0.2 additions:
• P-01: competing risks
• P-02: Hours -> LogNormal + segment priors
• P-03: event_definition actually affects event/time
• P-04: Beta-prior for major failure share
• P-05: FREQ_SHARES / SEVERITY_WEIGHTS
• P-06: Weibull shape 1.88 preserved
• P-07 data layer: RF heavy brand catalog / segments
• P-08: MTBF baseline 1500
• P-09: downtime by MTTR
• P-10: Power segments
• P-12: Kaplan-Meier validator
"""

from __future__ import annotations

import logging
import math
import os
import re
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from enterprise_quality import (
    generate_enterprise_quality,
    validate_enterprise_structure,
)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from lifelines import CoxPHFitter

# Import centralized constants
from constants import (
    BRAND_MAP,
    DEFAULT_BRAND_PROB_BY_CODE,
    SEGMENTS,
    RF_HEAVY_BRAND_CATALOG,
    FREQ_SHARES,
    SEVERITY_WEIGHTS,
    MTBF_BASELINE_HOURS,
    MODEL_TIME_UNIT,
    PL_HAT_EXOG_CONVENTION,
)

# Optional scipy.interpolate for spline basis.
try:
    from scipy import interpolate as _scipy_interpolate

    HAS_SCIPY_INTERPOLATE = _scipy_interpolate is not None and hasattr(
        _scipy_interpolate, "BSpline"
    )
except Exception:  # noqa: BLE001
    _scipy_interpolate = None
    HAS_SCIPY_INTERPOLATE = False

# Optional scipy.stats for copula contamination and LR p-values.
try:
    from scipy import stats as _scipy_stats

    HAS_SCIPY_STATS = _scipy_stats is not None
except Exception:  # noqa: BLE001
    _scipy_stats = None
    HAS_SCIPY_STATS = False


def _get_scipy_interpolate():
    """Return scipy.interpolate or raise if unavailable."""
    interp = _scipy_interpolate
    if interp is None or not HAS_SCIPY_INTERPOLATE:
        raise RuntimeError("scipy.interpolate.BSpline is unavailable")
    return interp


def _get_scipy_stats():
    """Return scipy.stats or raise if unavailable."""
    stats_mod = _scipy_stats
    if stats_mod is None or not HAS_SCIPY_STATS:
        raise RuntimeError("scipy.stats is unavailable")
    return stats_mod


# ─── Bootstrap parallelism ──────────────────────────────────────────
# REDUCED to prevent OOM with nested joblib (external MC sims × internal bootstrap)
_n_cpus = os.cpu_count() or 1
_bootstrap_jobs = min(2, max(1, _n_cpus // 4))  # Conservative: prevents OOM crashes

logger = logging.getLogger(__name__)


def _get_cox_log_likelihood(cph: CoxPHFitter) -> float:
    """Safely extract log-likelihood from lifelines CoxPHFitter."""
    ll = getattr(cph, "log_likelihood_", np.nan)
    if callable(ll):
        try:
            ll = ll()
        except Exception:  # noqa: BLE001
            return np.nan
    try:
        val = float(ll)
    except (TypeError, ValueError):
        return np.nan
    if not math.isfinite(val):
        return np.nan
    return val


def _scalar_from_array(value: Any) -> float:
    """Extract a scalar float from array-like objects."""
    arr = np.asarray(value)
    if arr.size == 0:
        return np.nan
    try:
        return float(arr.ravel()[0])
    except (TypeError, ValueError, IndexError):
        return np.nan


def _safe_keys(obj: Any) -> List[Any]:
    """
    Безопасно возвращает ключи из dict/Series/params-like объекта.

    Используется для извлечения имён ковариат из cph.params_ в
    ph_diagnostics_report и других диагностических функциях.
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


def _is_ambiguous_array_error(exc: BaseException) -> bool:
    """Проверка, что ошибка именно про неоднозначный булев массив."""
    return isinstance(exc, ValueError) and "truth value of an array" in str(exc).lower()


def _safe_series_scalar(series: Any, key: str, what: str) -> float:
    """
    Безопасно извлекает скаляр из pandas Series по имени коэффициента.
    Не использует .loc[] напрямую, чтобы избежать ошибок при дублях индекса.
    """
    if series is None:
        raise RuntimeError(f"{what}: series is None")

    try:
        mask = np.asarray(series.index == key, dtype=bool)
    except Exception:
        mask = np.array([idx == key for idx in series.index], dtype=bool)

    vals = np.asarray(series).ravel()[mask]

    if vals.size == 0:
        raise RuntimeError(f"{what}: coefficient '{key}' not found")

    if vals.size > 1:
        logger.warning(
            "%s: coefficient '%s' has %d duplicate entries; taking first",
            what,
            key,
            vals.size,
        )

    try:
        val = float(vals[0])
    except Exception as exc:
        raise RuntimeError(f"{what}: cannot convert coefficient '{key}' to float") from exc

    if not math.isfinite(val):
        raise RuntimeError(f"{what}: coefficient '{key}' is non-finite")

    return val


# ---------------------------------------------------------------------------
# Constants (Imported from centralized constants.py)
# ---------------------------------------------------------------------------
COX_SE_THRESHOLD_DEFAULT = 10.0
_VALID_BASELINE_FAMILIES = {"exponential", "weibull", "gompertz"}

FORBIDDEN_EXTRA_X_COLS = {
    "time",
    "event",
    "PeakLoad",
    "Z",
    "const",
    "eps_D",
    "U",
    "T_true",
    "C",
    "u_event",
    "u_censor",
    "lp",
    "lp_raw",
    "individual_hazard",
    "clipped_lp",
    # v0.2 outcome / competing risk columns
    "T_minor",
    "time_minor",
    "event_minor",
    "failure_type",
    "event_definition",
    "downtime_hours",
}

_ALL_PENALIZERS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

_BAD_KEYWORDS = (
    r"\bdid not converge\b",
    r"\bfailed to converge\b",
    r"\bnot converge\b",
    r"\bhessian is not positive definite\b",
    r"\bhessian inverted\b",
    r"\bperfect separation\b",
    r"\bquasi_separation\b",
    r"\bsingular matrix\b",
    r"\brank deficient\b",
    r"\bill-conditioned\b",
    r"\bill conditioned\b",
    r"\boverflow\b",
    r"\bunderflow\b",
    r"\bnan\b",
    r"\binf\b",
)
_COMPILE_BAD_KEYWORDS = [re.compile(p, re.IGNORECASE) for p in _BAD_KEYWORDS]

# ---------------------------------------------------------------------------
# P-03: event definitions
# Keep in sync with constants.VALID_EVENT_DEFINITIONS.
# ---------------------------------------------------------------------------
EVENT_DEFINITIONS: Tuple[str, ...] = (
    "total_loss",
    "major_claim",
    "any_failure",
)

# ---------------------------------------------------------------------------
# P-02: Hours prior by segment
# ---------------------------------------------------------------------------
HOURS_PRIOR: Dict[str, Dict[str, float]] = {
    "light": {
        "median": 900.0,
        "sigma": 0.60,
        "clip_min": 100.0,
        "clip_max": 3000.0,
    },
    "heavy": {
        "median": 1200.0,
        "sigma": 0.60,
        "clip_min": 200.0,
        "clip_max": 3000.0,
    },
}

# Фаза 4.4: Наработка по мониторингу РФ (общая за период, не годовая)
HOURS_PRIOR_RF: Dict[str, Dict[str, float]] = {
    "Fendt 930 Vario": {"min": 3782, "max": 5725, "years": 4},
    "John Deere 8430": {"min": 523, "max": 4000, "years": 3},
    "Кировец К-744Р": {"min": 59, "max": 3420, "years": 3},
    "New Holland T8040": {"min": 21, "max": 1192, "years": 2},
    "Bühler Versatile 2425": {"min": 460, "max": 4500, "years": 4},
}

# ---------------------------------------------------------------------------
# P-08: MTBF baseline (imported from constants.py)
# ---------------------------------------------------------------------------
DEFAULT_BASELINE_HAZARD: float = 1.0 / MTBF_BASELINE_HOURS

# ---------------------------------------------------------------------------
# P-09: downtime by MTTR
# ---------------------------------------------------------------------------
MTTR_HOURS: Dict[str, float] = {
    "minor": 8.0,
    "major": 48.0,
}


def downtime_hours_from_failure_type(failure_type: Any) -> np.ndarray:
    """
    P-09: convert failure_type array into downtime hours using MTTR.
    """
    ft = np.asarray(failure_type, dtype=str)
    out = np.zeros(ft.shape, dtype=float)
    out[ft == "minor"] = MTTR_HOURS["minor"]
    out[ft == "major"] = MTTR_HOURS["major"]
    return out


# ---------------------------------------------------------------------------
# P-04: Beta prior for major failure share
# ---------------------------------------------------------------------------
def major_failure_beta_prior(
    mean: float = 0.30,
    effective_n: float = 30.0,
) -> Dict[str, float]:
    """
    P-04: Beta-prior for major failure share instead of a point constant.
    """
    mean = float(np.clip(mean, 1e-3, 1.0 - 1e-3))
    effective_n = float(effective_n)

    if effective_n <= 0.0:
        raise ValueError("effective_n must be > 0")

    alpha = mean * effective_n
    beta = (1.0 - mean) * effective_n

    ci_low, ci_high = float("nan"), float("nan")
    try:
        from scipy import stats as _st

        ci_low = float(_st.beta.ppf(0.025, alpha, beta))
        ci_high = float(_st.beta.ppf(0.975, alpha, beta))
    except Exception:
        sd = float(np.sqrt(mean * (1.0 - mean) / (effective_n + 1.0)))
        ci_low = max(0.0, mean - 1.96 * sd)
        ci_high = min(1.0, mean + 1.96 * sd)

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "mean": float(mean),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "effective_n": float(effective_n),
    }


# ---------------------------------------------------------------------------
# Dataclasses: diagnostics / results
# ---------------------------------------------------------------------------
@dataclass
class ConvergenceInfo:
    penalizer: float
    warning: Optional[str] = None
    attempted_penalizers: List[float] = field(default_factory=list)


@dataclass
class FirstStageReport:
    n: int
    z_variance: float
    classical_f: float
    classical_f_pvalue: float
    robust_f: float
    robust_f_pvalue: float
    cluster_f: float
    cluster_f_pvalue: float
    partial_r2: float
    partial_f_z: float
    condition_number: float
    weak_instrument: bool
    min_f_threshold: float
    n_clusters: int = 0


@dataclass
class FirstStageFit:
    fitted: Any
    residuals: np.ndarray
    design: np.ndarray
    x_cols: List[str]
    report: FirstStageReport


@dataclass
class CFModelResult:
    gamma_hat: float = float("nan")
    naive_model_se: float = float("nan")
    bootstrap_se: Optional[float] = None
    se_type: str = "naive"
    cf_coef: float = float("nan")
    cf_coef_signed: Optional[float] = None
    cph: Optional[CoxPHFitter] = None
    max_se: float = float("nan")
    penalizer: float = 0.0
    is_penalized: bool = False
    convergence_info: Optional[ConvergenceInfo] = None
    warnings: List[str] = field(default_factory=list)
    n: Optional[int] = None
    n_events: Optional[int] = None
    v_hat_basis: str = "linear"
    first_stage_report: Optional[FirstStageReport] = None
    partial_out_all_betas: Dict[str, float] = field(default_factory=dict)
    training_x_means: Dict[str, float] = field(default_factory=dict)
    training_pl_hat_mean: float = 0.0
    training_residuals_std: float = 0.0
    training_residuals_mean: float = 0.0
    vif_peakload: Optional[float] = None
    vif_vhat: Optional[float] = None
    vif_max: Optional[float] = None
    cf_basis_metadata: Optional[Dict[str, Any]] = None


@dataclass
class NaiveCoxResult:
    gamma_hat: float
    naive_se: float
    max_se: float
    penalizer_used: float
    is_penalized: bool
    cph: CoxPHFitter
    convergence_info: ConvergenceInfo
    warnings: List[str] = field(default_factory=list)
    n: Optional[int] = None
    n_events: Optional[int] = None


@dataclass
class EndogeneityTestResult:
    lr_stat: Optional[float]
    lr_pvalue: Optional[float]
    endogenous: Optional[bool]
    penalized: bool
    df: int
    actual_cf_cols: List[str] = field(default_factory=list)
    note: Optional[str] = None


@dataclass
class CalibrationResult:
    censoring_scale: float
    target_event_rate: float
    achieved_event_rate: float
    converged: bool
    iterations: int
    post_check_passed: Optional[bool] = None
    post_check_rates: Dict[float, float] = field(default_factory=dict)


@dataclass
class BootstrapResult:
    bootstrap_se: Optional[float]
    n_successful: int
    n_failures: int
    success_rate: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    high_failure_rate: bool = False
    estimates: Optional[List[float]] = None
    error_examples: List[str] = field(default_factory=list)
    rejection_reason_summary: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DGP / simulation config
# ---------------------------------------------------------------------------
@dataclass
class DGPParameters:
    """Parameters for data generating process (DGP v3.2 / v0.2)."""

    gamma: float = 0.5
    rho: float = 0.7
    delta: float = 0.7
    # FIX 1: PeakLoad normalized to [0, 1] range to match TUM CAN bus data.
    # Original intercept was 10.0 (DGP scale); now 0.5 (midpoint of [0,1]).
    intercept: float = 0.5
    structural_intercept: Optional[float] = None
    first_stage_z_coef: float = 0.5

    # First stage coefficients — scaled for [0,1] range.
    # Original values were for DGP scale (~10.0); now kept for relative effects.
    fs_age_coef: float = 0.15
    fs_hours_coef: float = 0.10
    fs_climate_coef: float = 0.20
    fs_soil_coef: float = 0.15
    fs_brand_coef: float = 0.10
    fs_power_coef: float = 0.08

    # Structural equation coefficients
    beta_age: float = 0.20
    beta_hours: float = 0.10
    beta_climate: float = 0.20
    beta_soil: float = 0.12
    beta_brand: float = 0.06
    beta_power: float = -0.05

    # ─── Фаза 8: Interaction Age × Hours ───────────────────────────
    beta_age_hours: float = 0.15  # синергетический эффект

    clip_lp: Optional[float] = None
    corr_zu: float = 0.0

    # Baseline hazard
    baseline_shape: Optional[float] = 1.88
    baseline_family: str = "weibull"

    # ─── v0.2 / P-01 / P-03: competing risks and event definition ───────
    competing_risks: bool = False
    minor_failure_rate: float = 0.002
    event_definition: str = "major_claim"  # total_loss | major_claim | any_failure
    segment: str = "light"  # light | heavy

    # Brand encoding
    brand_encoding: str = "dummies"
    brand_reference_code: int = 0
    brand_prob_by_code: Optional[Dict[int, float]] = None
    fs_brand_coefs: Optional[Dict[int, float]] = None
    beta_brand_coefs: Optional[Dict[int, float]] = None

    # ─── Фаза 3: калибровка PeakLoad под TUM ───────────────────────────
    # FIX 1: PeakLoad теперь генерируется в [0, 1] диапазоне.
    # Эти значения соответствуют статистике TUM CAN bus данных.
    peakload_target_mean: float = 0.55
    peakload_target_std: float = 0.15

    # ─── Фаза 5.3 / 6.6: погодный инструмент ─────────────────────────────
    # Если instrument_source == "weather", Z генерируется как
    # working_days_window ~ Normal(weather_mean_days, weather_std_days),
    # клип в физически правдоподобный диапазон [5, 90].
    # Если instrument_source == "weather_real", используются реальные данные
    # из weather_windows.csv (NASA POWER API).
    weather_mean_days: float = 45.0
    weather_std_days: float = 12.0
    weather_campaign: str = "sowing"  # "sowing" или "harvest"

    # ─── Фаза 6.6: источник данных о почве ─────────────────────────────
    # "synthetic" — Beta(2.0, 2.5) распределение
    # "claims" — из claims_clean.csv (реальный GLDAS)
    # "soil_real" — реальные данные GLDAS-2.1 из soil_windows.csv
    soil_source: str = "synthetic"

    # ─── Фаза 6.6: гибридный режим (DGP + реальные weather/soil) ───────
    # Если True, ковариаты x_climate, x_soil и инструмент Z берутся
    # из реальных спутниковых данных NASA POWER / GLDAS-2.1
    use_real_covariates: bool = False

    # ─── Фаза EQI: Enterprise Quality Index ────────────────────────
    beta_eqi: float = 0.0
    n_enterprises: int = 500
    use_enterprise_quality: bool = False


@dataclass
class Scenario:
    pi: float
    rho: float
    delta: float
    clip_lp: Optional[float]
    corr_zu: float
    target_event_rate: Optional[float]
    experiment: str = "unspecified"

    def scenario_id(self) -> str:
        def sf(x: Any) -> str:
            if x is None:
                return "None"
            if isinstance(x, (float, np.floating)):
                return f"{float(x):.2f}".replace(".", "p")
            return str(x)

        parts = [
            self.experiment,
            f"pi{sf(self.pi)}",
            f"rho{sf(self.rho)}",
            f"delta{sf(self.delta)}",
            f"clip{sf(self.clip_lp)}",
            f"zu{sf(self.corr_zu)}",
            f"tgt{sf(self.target_event_rate)}",
        ]
        return "_".join(parts)


@dataclass(frozen=True)
class SimulationConfig:
    sims_per_scenario: int
    n_samples: int
    contamination: bool
    n_jobs: int
    seed: int
    n_bootstrap: int
    bootstrap_jobs: int
    baseline_hazard: float
    censoring_scale: float
    dgp: DGPParameters

    save_tracebacks: bool = True
    bootstrap_success_frac: float = 0.8
    bootstrap_method: str = "case"
    bootstrap_mode: str = "applied"
    wild_bootstrap_dist: str = "rademacher"
    cox_max_se_threshold: float = COX_SE_THRESHOLD_DEFAULT
    min_cox_events: int = 10
    min_events_per_covariate: int = 5
    var_z_threshold: float = 1e-8
    min_first_stage_f: float = 10.0
    fail_on_weak_instrument: bool = True
    max_failure_rate: float = 0.10
    allow_experimental_bootstrap: bool = False
    v_hat_basis: str = "linear"
    v_hat_basis_params: Optional[Dict[str, Any]] = field(default_factory=lambda: {"n_knots": 2})
    contamination_probability: float = 1.0
    extra_x_cols: Optional[List[str]] = None
    center_peakload: Optional[float] = None


@dataclass(frozen=True)
class CFFitOptions:
    cox_se_threshold: float
    v_hat_basis: str
    v_hat_basis_params: Optional[Dict[str, Any]]
    extra_x_cols: Optional[List[str]]
    center_peakload: Optional[float]
    brand_encoding: str
    brand_reference_code: int
    var_z_threshold: float
    min_first_stage_f: float
    fail_on_weak_instrument: bool
    min_cox_events: int
    min_events_per_covariate: int
    save_tracebacks: bool
    cluster_col: Optional[str] = None
    # ─── Bootstrap SE for generated regressors (FIX 4) ───
    # Bootstrap включён по умолчанию для корректного учёта
    # неопределённости первой стадии (generated regressor problem).
    # Наивные SE из lifelines занижают дисперсию.
    # PATCH-11: 200 итераций — минимально допустимое для
    # относительной ошибки SE ≈ 1/√(2·200) ≈ 5%.
    # Для публикации: 1000+.
    n_bootstrap: int = 200  # БЫЛО: 50


def fit_options_from_config(config: SimulationConfig) -> CFFitOptions:
    return CFFitOptions(
        cox_se_threshold=config.cox_max_se_threshold,
        v_hat_basis=config.v_hat_basis,
        v_hat_basis_params=config.v_hat_basis_params,
        extra_x_cols=config.extra_x_cols,
        center_peakload=config.center_peakload,
        brand_encoding=config.dgp.brand_encoding,
        brand_reference_code=config.dgp.brand_reference_code,
        var_z_threshold=config.var_z_threshold,
        min_first_stage_f=config.min_first_stage_f,
        fail_on_weak_instrument=config.fail_on_weak_instrument,
        min_cox_events=config.min_cox_events,
        min_events_per_covariate=config.min_events_per_covariate,
        save_tracebacks=config.save_tracebacks,
        n_bootstrap=config.n_bootstrap,  # Propagate n_bootstrap to workers
    )


# ... (Utilities: format_exception, _as_finite_float, etc.) ...
# ... (X standardization / design building) ...


# ---------------------------------------------------------------------------
# Simulation config validation
# ---------------------------------------------------------------------------


def validate_simulation_config(config: SimulationConfig) -> None:
    if config.sims_per_scenario <= 0:
        raise ValueError("sims_per_scenario must be > 0")
    if config.n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if config.n_bootstrap < 0:
        raise ValueError("n_bootstrap must be >= 0")
    if config.bootstrap_jobs is not None and config.bootstrap_jobs <= 0:
        raise ValueError("bootstrap_jobs must be positive or None")

    # ★ FIX: baseline_hazard и censoring_scale нужны только для mc_parametric,
    # так как case/applied_wild используют ресэмплинг существующих данных.
    if config.bootstrap_method.lower() == "mc_parametric":
        if config.baseline_hazard <= 0:
            raise ValueError("baseline_hazard must be > 0")
        if config.censoring_scale <= 0:
            raise ValueError("censoring_scale must be > 0")

    if not (0 <= config.bootstrap_success_frac <= 1):
        raise ValueError("bootstrap_success_frac must be in [0, 1]")
    if not (0 <= config.max_failure_rate <= 1):
        raise ValueError("max_failure_rate must be in [0, 1]")

    method = config.bootstrap_method.lower()
    if method not in {"case", "applied_wild", "mc_parametric"}:
        raise ValueError(f"Unknown bootstrap_method: {config.bootstrap_method}")

    if method == "mc_parametric" and config.bootstrap_mode != "mc":
        raise ValueError("mc_parametric bootstrap is allowed only in bootstrap_mode='mc'")

    if method == "applied_wild" and not config.allow_experimental_bootstrap:
        raise ValueError(
            "applied_wild bootstrap is experimental and disabled by default. "
            "Set allow_experimental_bootstrap=True to use it."
        )

    _validate_dgp(config.dgp)


# ---------------------------------------------------------------------------    # Utilities
# ---------------------------------------------------------------------------
def format_exception(exc: Optional[BaseException], save_traceback: bool) -> str:
    if exc is None:
        return "Unknown error"

    if save_traceback:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return tb.replace("\\", "/")

    return f"{type(exc).__name__}: {exc}"


def _as_finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} cannot be boolean")

    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")

    return value


def _validate_corr(value: Any, name: str) -> float:
    x = _as_finite_float(value, name)

    if not (-1.0 <= x <= 1.0):
        raise ValueError(f"{name} must be in [-1, 1], got {x}")

    if x >= 1.0:
        x = 1.0 - 1e-12
    elif x <= -1.0:
        x = -1.0 + 1e-12

    return float(x)


def _safe_std(arr: np.ndarray, ddof: int = 1, floor: float = 1e-10) -> float:
    arr = np.asarray(arr, dtype=float)

    if arr.size <= ddof:
        return floor

    val = float(np.std(arr, ddof=ddof))

    if (not np.isfinite(val)) or val < floor:
        return floor

    return val


def _standardize_array(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return x

    m = float(np.mean(x))
    s = float(np.std(x, ddof=0))

    if (not np.isfinite(s)) or s < floor:
        raise ValueError("Cannot standardize degenerate array")

    return (x - m) / s


def replace_nonfinite(obj: Any, visited: Optional[set] = None) -> Any:
    """JSON-safe recursive replacement of non-finite values."""
    if visited is None:
        visited = set()

    if obj is pd.NaT:
        return None

    if isinstance(obj, np.generic):
        obj = obj.item()

    obj_id = id(obj)

    if isinstance(obj, (dict, list, tuple)) and obj_id in visited:
        return None

    if isinstance(obj, dict):
        visited.add(obj_id)
        result = {k: replace_nonfinite(v, visited) for k, v in obj.items()}
        visited.discard(obj_id)
        return result

    if isinstance(obj, (list, tuple)):
        visited.add(obj_id)
        result = [replace_nonfinite(v, visited) for v in obj]
        visited.discard(obj_id)
        return result

    if isinstance(obj, np.ndarray):
        return replace_nonfinite(obj.tolist(), visited)

    if isinstance(obj, (pd.Series, pd.Index)):
        return replace_nonfinite(list(obj), visited)

    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return None if obj is pd.NaT else str(obj)

    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return float(obj)

    if isinstance(obj, (int, bool, str)) or obj is None:
        return obj

    try:
        from dataclasses import is_dataclass, asdict

        if is_dataclass(obj):
            return replace_nonfinite(asdict(obj), visited)
    except Exception:
        pass

    try:
        return str(obj)
    except Exception:
        return None


def _validate_survival_frame(
    df: pd.DataFrame,
    required_covs: List[str],
) -> None:
    required = {"time", "event"} | set(required_covs)
    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    time = df["time"].astype(float).to_numpy()
    event = df["event"].astype(int).to_numpy()

    if not np.all(np.isfinite(time)):
        raise ValueError("Survival time contains non-finite values")

    if not np.all(time > 0):
        raise ValueError("Survival time must be strictly positive")

    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("Event column must be binary 0/1")

    if required_covs:
        cov_values = df[list(required_covs)].astype(float).to_numpy()

        if not np.all(np.isfinite(cov_values)):
            raise ValueError("Covariates contain non-finite values")


def _validate_extra_x_cols(
    source_data: pd.DataFrame,
    extra_x_cols: Optional[List[str]],
) -> List[str]:
    if not extra_x_cols:
        return []

    seen = set()
    valid_cols: List[str] = []

    for col in extra_x_cols:
        col = str(col)

        if col in FORBIDDEN_EXTRA_X_COLS:
            raise ValueError(f"extra_x_cols contains forbidden column: {col}")

        if col in seen:
            raise ValueError(f"Duplicate extra_x_cols entry: {col}")

        # Skip x_age_hours if it's not in the data (it may be added later as interaction)
        if col == "x_age_hours" and col not in source_data.columns:
            continue

        if col not in source_data.columns:
            raise KeyError(f"extra_x_cols column not found: {col}")

        try:
            vals = source_data[col].astype(float).to_numpy()
        except Exception as exc:
            raise TypeError(f"extra_x_cols column {col} is not numeric") from exc

        if not np.all(np.isfinite(vals)):
            raise ValueError(f"extra_x_cols column contains non-finite values: {col}")

        seen.add(col)
        valid_cols.append(col)

    return valid_cols


def _summarize_warnings(
    caught_warnings: Optional[List[warnings.WarningMessage]],
) -> Optional[str]:
    if not caught_warnings:
        return None

    bad_messages: List[str] = []

    for w in caught_warnings:
        msg = str(w.message)

        if any(pat.search(msg) for pat in _COMPILE_BAD_KEYWORDS):
            bad_messages.append(msg)

    if not bad_messages:
        return None

    return "; ".join(bad_messages[:5])


def _validate_int_float_dict(
    value: Optional[Dict[Any, float]],
    name: str,
) -> Optional[Dict[int, float]]:
    if value is None:
        return None

    if not isinstance(value, dict):
        raise ValueError(f"{name} must be dict or None")

    out: Dict[int, float] = {}

    for k, v in value.items():
        out[int(k)] = _as_finite_float(v, f"{name}[{k}]")

    return out


def _validate_brand_probs(
    probs: Optional[Dict[int, float]],
) -> Dict[int, float]:
    if probs is None:
        probs = DEFAULT_BRAND_PROB_BY_CODE

    if not isinstance(probs, dict) or not probs:
        raise ValueError("brand_prob_by_code must be non-empty dict")

    out: Dict[int, float] = {}
    total = 0.0

    for code, p in probs.items():
        p_float = _as_finite_float(p, f"brand_prob_by_code[{code}]")

        if p_float < 0.0:
            raise ValueError(f"brand_prob_by_code[{code}] must be >= 0")

        out[int(code)] = p_float
        total += p_float

    if total <= 0.0:
        raise ValueError("Sum of brand probabilities must be positive")

    return {k: v / total for k, v in out.items()}


def _default_brand_coefs(
    scalar: float,
    reference_code: int,
) -> Dict[int, float]:
    """
    Compatibility mapping from legacy scalar brand effect to dummy coefficients
    relative to reference category.
    """
    scalar = float(scalar)
    reference_code = int(reference_code)

    out: Dict[int, float] = {}

    for code in BRAND_MAP.keys():
        if code == reference_code:
            continue

        out[int(code)] = scalar * ((code - reference_code) / 2.0)

    return out


# ---------------------------------------------------------------------------
# X standardization / design building
# ---------------------------------------------------------------------------
X_STANDARDIZATION: Dict[str, Dict[str, Any]] = {
    "x_age": {
        "raw_col": "Age",
        "standardize": True,
        "shift": 10.0,
        "scale": 10.0,
    },
    "x_hours": {
        "raw_col": "Hours",
        "standardize": True,
        "shift": 1000.0,  # P-02: медиана LogNormal(1000)
        "scale": 1000.0,
    },
    # FIX 1: PeakLoad normalized to [0, 1] range matching TUM CAN bus.
    "PeakLoad": {
        "raw_col": "PeakLoad",
        "standardize": True,
        "shift": 0.55,  # TUM data mean
        "scale": 0.15,  # TUM data std
    },
    "x_climate": {
        "raw_col": "Climate",
        "standardize": False,
    },
    "x_soil": {
        "raw_col": "Soil",
        "standardize": False,
    },
    "x_power": {
        "raw_col": "Power",
        "standardize": True,
        "shift": 200.0,
        "scale": 150.0,
    },
    "x_brand": {
        "raw_col": "Brand",
        "standardize": True,
        "shift": 2.0,
        "scale": 2.0,
    },
    "x_age_hours": {
        "raw_col": "Age_x_Hours",
        "standardize": False,  # Уже стандартизирован по конструкции (x_age * x_hours)
    },
}

CONTINUOUS_X_COLS = ["x_age", "x_hours", "x_climate", "x_soil", "x_power"]


def _standardize_x_column(std_col: str, values: np.ndarray) -> np.ndarray:
    if std_col not in X_STANDARDIZATION:
        raise KeyError(f"Unknown standardized X column: {std_col}")

    info = X_STANDARDIZATION[std_col]
    values = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError(f"Column {std_col} contains non-finite values")

    if not info.get("standardize", False):
        return values

    shift = _as_finite_float(info.get("shift"), f"{std_col}.shift")
    scale = _as_finite_float(info.get("scale"), f"{std_col}.scale")

    if scale <= 0.0:
        raise ValueError(f"{std_col}.scale must be > 0")

    out = (values - shift) / scale

    if not np.all(np.isfinite(out)):
        raise ValueError(f"Standardization produced non-finite values for {std_col}")

    return out


def _ensure_brand_dummies(
    model_data: pd.DataFrame,
    source_data: pd.DataFrame,
    reference_code: int,
) -> List[str]:
    reference_code = int(reference_code)
    ref_name = BRAND_MAP.get(reference_code, str(reference_code))
    ref_col = f"brand_{ref_name}"

    if "brand_code" in source_data.columns:
        codes = source_data["brand_code"].astype(int).to_numpy()
    elif "Brand" in source_data.columns:
        codes = source_data["Brand"].astype(int).to_numpy()
    else:
        existing = [c for c in source_data.columns if str(c).startswith("brand_") and c != ref_col]

        if not existing:
            raise KeyError("Brand/brand_code or brand dummy columns are missing")

        for c in existing:
            vals = source_data[c].astype(float).to_numpy()
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"Brand dummy column {c} contains non-finite values")

            model_data[c] = vals

        return sorted(existing)

    cols: List[str] = []

    for code, name in BRAND_MAP.items():
        if int(code) == reference_code:
            continue

        col = f"brand_{name}"
        model_data[col] = (codes == int(code)).astype(float)
        cols.append(col)

    return cols


def _add_design_x_columns(
    model_data: pd.DataFrame,
    source_data: pd.DataFrame,
    extra_x_cols: Optional[List[str]] = None,
    brand_encoding: str = "dummies",
    brand_reference_code: int = 0,
) -> List[str]:
    """Add covariate design columns and return their names."""
    x_cols: List[str] = []

    # Continuous X
    for std_col in CONTINUOUS_X_COLS:
        info = X_STANDARDIZATION[std_col]
        raw_col = info["raw_col"]

        if raw_col in source_data.columns:
            raw_vals = source_data[raw_col].astype(float).to_numpy()
            model_data[std_col] = _standardize_x_column(std_col, raw_vals)
            x_cols.append(std_col)
        elif std_col in source_data.columns:
            vals = source_data[std_col].astype(float).to_numpy()

            if not np.all(np.isfinite(vals)):
                raise ValueError(f"Existing standardized column {std_col} is non-finite")

            model_data[std_col] = vals
            x_cols.append(std_col)
        else:
            raise KeyError(f"Missing raw or standardized column for {std_col}")

    # Brand encoding
    brand_encoding = str(brand_encoding).lower()

    if brand_encoding == "legacy_continuous":
        if "Brand" in source_data.columns:
            model_data["x_brand"] = _standardize_x_column(
                "x_brand",
                source_data["Brand"].astype(float).to_numpy(),
            )
        elif "x_brand" in source_data.columns:
            vals = source_data["x_brand"].astype(float).to_numpy()

            if not np.all(np.isfinite(vals)):
                raise ValueError("Existing x_brand column is non-finite")

            model_data["x_brand"] = vals
        else:
            raise KeyError("Brand or x_brand is required for legacy_continuous encoding")

        x_cols.append("x_brand")

    elif brand_encoding == "dummies":
        dummy_cols = _ensure_brand_dummies(
            model_data=model_data,
            source_data=source_data,
            reference_code=brand_reference_code,
        )
        x_cols.extend(dummy_cols)

    else:
        raise ValueError("brand_encoding must be either 'dummies' or 'legacy_continuous'")

    # Extra X
    valid_extra = _validate_extra_x_cols(source_data, extra_x_cols)

    for col in valid_extra:
        if col not in x_cols:
            model_data[col] = source_data[col].astype(float).to_numpy()
            x_cols.append(col)

    # ─── Interaction Age × Hours ──────────────────────────────────────
    # ВАЖНО: используем готовую колонку из source_data, если она есть
    # (уже центрирована и стандартизирована в generate_data).
    # НЕ вычисляем x_age * x_hours — это нецентрированное произведение
    # с другим масштабом, которое разрушает коэффициенты Cox.
    if "x_age_hours" in source_data.columns:
        vals = source_data["x_age_hours"].astype(float).to_numpy()
        if np.all(np.isfinite(vals)):
            model_data["x_age_hours"] = vals
            if "x_age_hours" not in x_cols:
                x_cols.append("x_age_hours")
    elif "x_age" in model_data.columns and "x_hours" in model_data.columns:
        # Fallback: вычисляем и стандартизируем (для claims без готовой колонки)
        raw_interaction = model_data["x_age"].to_numpy() * model_data["x_hours"].to_numpy()
        m = float(np.mean(raw_interaction))
        s = float(np.std(raw_interaction, ddof=1))
        if s < 1e-9:
            s = 1.0
        model_data["x_age_hours"] = (raw_interaction - m) / s
        if "x_age_hours" not in x_cols:
            x_cols.append("x_age_hours")

    # ─── PATCH-10: Enterprise Quality Index ────────────────────────
    if "x_enterprise_quality" in source_data.columns:
        vals = source_data["x_enterprise_quality"].astype(float).to_numpy()
        if np.all(np.isfinite(vals)) and np.std(vals) > 1e-12:
            model_data["x_enterprise_quality"] = vals
            if "x_enterprise_quality" not in x_cols:
                x_cols.append("x_enterprise_quality")

    return x_cols


# ---------------------------------------------------------------------------
# Patch 6: Валидация corr_zu для каузального режима
# ---------------------------------------------------------------------------
def validate_dgp_for_causal_mode(dgp: DGPParameters, iv_mode: str = "causal") -> None:
    """Проверяет корректность DGP для каузального режима."""
    if iv_mode == "causal" and abs(dgp.corr_zu) > 1e-12:
        raise ValueError(
            f"corr_zu={dgp.corr_zu} нарушает экзогенность инструмента в causal-режиме. "
            "Установите corr_zu=0.0 или используйте 'predictive' режим."
        )

    if abs(dgp.corr_zu) > 1e-12:
        logger.warning(
            "corr_zu=%.4f != 0: инструмент эндогенен. "
            "Результаты следует интерпретировать как стресс-тест, а не каузальный эффект.",
            dgp.corr_zu,
        )


# ---------------------------------------------------------------------------
# DGP validation
# ---------------------------------------------------------------------------
def _validate_dgp(dgp: Optional[DGPParameters]) -> DGPParameters:
    if dgp is None:
        dgp = DGPParameters()

    if not isinstance(dgp, DGPParameters):
        raise TypeError("dgp must be DGPParameters")

    numeric_fields = [
        "gamma",
        "rho",
        "delta",
        "intercept",
        "first_stage_z_coef",
        "fs_age_coef",
        "fs_hours_coef",
        "fs_climate_coef",
        "fs_soil_coef",
        "fs_brand_coef",
        "fs_power_coef",
        "beta_age",
        "beta_hours",
        "beta_climate",
        "beta_soil",
        "beta_brand",
        "beta_power",
        "beta_age_hours",
        "corr_zu",
    ]

    for name in numeric_fields:
        value = getattr(dgp, name)
        _as_finite_float(value, f"dgp.{name}")

    dgp.rho = _validate_corr(dgp.rho, "dgp.rho")
    dgp.corr_zu = _validate_corr(dgp.corr_zu, "dgp.corr_zu")

    if abs(dgp.corr_zu) > 1e-12:
        logger.warning(
            "DGP: corr_zu=%.6f != 0. Instrument Z is correlated with structural error. "
            "Exclusion restriction violated. Causal interpretation invalid. "
            "Use this mode only for stress testing / invalid-instrument experiments.",
            dgp.corr_zu,
        )

    if dgp.structural_intercept is not None:
        dgp.structural_intercept = _as_finite_float(
            dgp.structural_intercept,
            "dgp.structural_intercept",
        )

    if dgp.clip_lp is not None:
        clip_val = _as_finite_float(dgp.clip_lp, "dgp.clip_lp")

        if clip_val < 0.0:
            raise ValueError("dgp.clip_lp must be non-negative")

        dgp.clip_lp = clip_val

    family = str(getattr(dgp, "baseline_family", "weibull")).lower()

    if family not in _VALID_BASELINE_FAMILIES:
        raise ValueError(
            f"Unknown baseline_family: '{family}'. "
            f"Valid values are: {sorted(_VALID_BASELINE_FAMILIES)}"
        )

    dgp.baseline_family = family

    if family in {"weibull", "gompertz"}:
        if dgp.baseline_shape is None:
            raise ValueError(f"baseline_shape is required for {family}")

        shape = _as_finite_float(dgp.baseline_shape, "dgp.baseline_shape")

        if shape <= 0.0:
            raise ValueError("dgp.baseline_shape must be > 0")

        dgp.baseline_shape = shape

    # ─── v0.2 / P-01 / P-03 validation ───────────────────────────────
    if not isinstance(getattr(dgp, "competing_risks", False), (bool, np.bool_)):
        raise ValueError("dgp.competing_risks must be boolean")

    dgp.competing_risks = bool(dgp.competing_risks)

    dgp.minor_failure_rate = _as_finite_float(
        getattr(dgp, "minor_failure_rate", 0.002),
        "dgp.minor_failure_rate",
    )

    if dgp.minor_failure_rate <= 0.0:
        raise ValueError("dgp.minor_failure_rate must be > 0")

    event_definition = str(getattr(dgp, "event_definition", "major_claim")).lower()

    if event_definition not in EVENT_DEFINITIONS:
        raise ValueError(
            f"Unknown event_definition: '{event_definition}'. "
            f"Valid values are: {sorted(EVENT_DEFINITIONS)}"
        )

    dgp.event_definition = event_definition

    segment = str(getattr(dgp, "segment", "light")).lower()

    if segment not in SEGMENTS:
        raise ValueError(f"Unknown segment: '{segment}'. Valid values are: {sorted(SEGMENTS)}")

    dgp.segment = segment

    # ─── Brand encoding validation ────────────────────────────────────
    brand_encoding = str(getattr(dgp, "brand_encoding", "dummies")).lower()

    if brand_encoding not in {"dummies", "legacy_continuous"}:
        raise ValueError("dgp.brand_encoding must be 'dummies' or 'legacy_continuous'")

    dgp.brand_encoding = brand_encoding

    ref_code = int(getattr(dgp, "brand_reference_code", 0))

    if ref_code not in BRAND_MAP:
        raise ValueError("dgp.brand_reference_code must be a valid brand code")

    dgp.brand_reference_code = ref_code

    dgp.brand_prob_by_code = _validate_brand_probs(dgp.brand_prob_by_code)

    dgp.fs_brand_coefs = _validate_int_float_dict(
        dgp.fs_brand_coefs,
        "dgp.fs_brand_coefs",
    )
    dgp.beta_brand_coefs = _validate_int_float_dict(
        dgp.beta_brand_coefs,
        "dgp.beta_brand_coefs",
    )

    # ─── Фаза 3: валидация TUM PeakLoad калибровки ──────────────────
    pl_target_mean = getattr(dgp, "peakload_target_mean", None)
    pl_target_std = getattr(dgp, "peakload_target_std", None)
    if pl_target_mean is not None:
        pl_target_mean = _as_finite_float(
            pl_target_mean,
            "dgp.peakload_target_mean",
        )
        if pl_target_mean <= 0.0:
            raise ValueError("dgp.peakload_target_mean must be positive")
        dgp.peakload_target_mean = pl_target_mean
    if pl_target_std is not None:
        pl_target_std = _as_finite_float(
            pl_target_std,
            "dgp.peakload_target_std",
        )
        if pl_target_std <= 0.0:
            raise ValueError("dgp.peakload_target_std must be positive")
        dgp.peakload_target_std = pl_target_std

    # ─── Фаза 6.6: валидация weather_campaign ───────────────────────
    weather_campaign = str(getattr(dgp, "weather_campaign", "sowing")).lower()
    if weather_campaign not in VALID_WEATHER_CAMPAIGNS:
        raise ValueError(
            f"dgp.weather_campaign должно быть одним из "
            f"{sorted(VALID_WEATHER_CAMPAIGNS)}, "
            f"получено: {weather_campaign!r}"
        )
    dgp.weather_campaign = weather_campaign

    return dgp


# ---------------------------------------------------------------------------
# Error generation
# ---------------------------------------------------------------------------
def _standardize_sample(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 2:
        raise ValueError("Need at least 2 observations to standardize errors")

    x = x - float(np.mean(x))
    sd = float(np.std(x, ddof=0))

    if sd < 1e-12 or not math.isfinite(sd):
        raise ValueError("Degenerate error vector")

    return x / sd


def _orthogonalize_against(x: np.ndarray, base: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)

    base_norm = float(np.linalg.norm(base))

    if base_norm < 1e-12 or not math.isfinite(base_norm):
        raise ValueError("Base vector is degenerate")

    base_unit = base / base_norm
    x = x - float(np.dot(x, base_unit)) * base_unit

    return _standardize_sample(x)


def _correlated_standard_errors(
    rng: np.random.Generator,
    n: int,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rho = _validate_corr(rho, "rho")

    eps = _standardize_sample(rng.normal(size=n))
    eta = _standardize_sample(rng.normal(size=n))
    eta = _orthogonalize_against(eta, eps)

    u = rho * eps + math.sqrt(max(0.0, 1.0 - rho * rho)) * eta
    u = _standardize_sample(u)

    actual_rho = float(np.mean(eps * u))

    if not math.isfinite(actual_rho) or abs(actual_rho - rho) > 1e-6:
        raise FloatingPointError("Failed to construct errors with target correlation")

    return eps, u


def _normal_correlated_errors(
    rng: np.random.Generator,
    n: int,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray]:
    return _correlated_standard_errors(rng, n, rho)


def _contaminated_errors(
    rng: np.random.Generator,
    n: int,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Heavy-tailed errors using Gaussian copula with t(3) marginals."""
    stats_mod = _scipy_stats

    if stats_mod is None or not HAS_SCIPY_STATS:
        raise RuntimeError(
            "scipy.stats is required for contaminated heavy-tailed errors. "
            "Install scipy or set contamination=False."
        )

    z1, z2 = _correlated_standard_errors(rng, n, rho)

    u1 = np.clip(stats_mod.norm.cdf(z1), 1e-12, 1.0 - 1e-12)
    u2 = np.clip(stats_mod.norm.cdf(z2), 1e-12, 1.0 - 1e-12)

    eps = stats_mod.t.ppf(u1, df=3)
    u = stats_mod.t.ppf(u2, df=3)

    return _standardize_sample(eps), _standardize_sample(u)


def _generate_errors(
    rng: np.random.Generator,
    n: int,
    rho: float,
    contamination: bool,
    contamination_probability: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rho = _validate_corr(rho, "rho")

    try:
        pi = float(contamination_probability)
    except Exception:
        pi = 1.0 if contamination else 0.0

    if not math.isfinite(pi):
        pi = 1.0 if contamination else 0.0

    pi = float(np.clip(pi, 0.0, 1.0))

    if (not contamination) or pi <= 0.0:
        eps_d, u = _normal_correlated_errors(rng, n, rho)
    elif pi >= 1.0:
        eps_d, u = _contaminated_errors(rng, n, rho)
    else:
        mask = rng.random(n) < pi
        eps_clean, u_clean = _normal_correlated_errors(rng, n, rho)
        eps_cont, u_cont = _contaminated_errors(rng, n, rho)

        eps_d = np.where(mask, eps_cont, eps_clean)
        u = np.where(mask, u_cont, u_clean)

        eps_d = _standardize_sample(eps_d)
        u = _standardize_sample(u)

    return eps_d, u


# ---------------------------------------------------------------------------
# Фаза 6.6: загрузка реальных погодных данных NASA POWER
# ---------------------------------------------------------------------------
WEATHER_DATA_PATH = Path("data/processed/weather/weather_windows.csv")
VALID_WEATHER_CAMPAIGNS = frozenset({"sowing", "harvest"})


def load_real_weather_windows(
    campaign: str = "sowing",
    path: Optional[Path] = None,
) -> np.ndarray:
    """
    Фаза 6.6: загрузить реальные working_days_window из
    weather_windows.csv и вернуть стандартизованный массив.

    Parameters
    ----------
    campaign : str
        "sowing" или "harvest".
    path : Path, optional
        Путь к weather_windows.csv. По умолчанию WEATHER_DATA_PATH.

    Returns
    -------
    np.ndarray
        Стандартизованные working_days_window (mean=0, std=1).

    Raises
    ------
    FileNotFoundError
        Если файл не найден.
    ValueError
        Если данных недостаточно или нет записей для кампании.
    """
    if path is None:
        path = WEATHER_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Реальные погодные данные не найдены: {path}. "
            f"Сначала запустите load_nasa_power.py и "
            f"compute_working_days.py"
        )

    campaign = str(campaign).strip().lower()
    if campaign not in VALID_WEATHER_CAMPAIGNS:
        raise ValueError(
            f"campaign должно быть одним из "
            f"{sorted(VALID_WEATHER_CAMPAIGNS)}, получено: {campaign!r}"
        )

    df = pd.read_csv(path)

    # Проверка обязательных колонок
    required = {"campaign", "working_days_window"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В {path} отсутствуют колонки: {sorted(missing)}")

    # Фильтр по кампании
    subset = df[df["campaign"] == campaign]["working_days_window"]
    vals = pd.to_numeric(subset, errors="coerce").dropna()

    if len(vals) < 5:
        raise ValueError(f"Недостаточно данных для campaign='{campaign}': {len(vals)} записей < 5")

    mean = float(vals.mean())
    std = float(vals.std(ddof=1))
    if std < 1e-9:
        raise ValueError(f"working_days_window для campaign='{campaign}' имеет нулевую дисперсию")

    standardized = ((vals - mean) / std).to_numpy(dtype=float)
    logger.info(
        "Загружено %d реальных погодных записей (campaign=%s): mean=%.1f дней, std=%.1f дней",
        len(standardized),
        campaign,
        mean,
        std,
    )
    return standardized


# ---------------------------------------------------------------------------
# Фаза 6.6: загрузка реальных ковариат для гибридного режима
# ---------------------------------------------------------------------------
def load_real_covariates_for_simulation(
    n: int,
    rng: np.random.Generator,
    campaign: str = "sowing",
    jitter_std: float = 0.05,
    z_scale_factor: float = 2.0,
) -> Dict[str, np.ndarray]:
    """
    Загрузить реальные weather/soil/rainfall данные и сэмплировать n значений.

    Z = rainfall_anomaly (NASA POWER) — инструмент для PeakLoad
    x_climate = working_days_window (NASA POWER) — структурная ковариата
    x_soil = soil_index (GLDAS-2.1) — структурная ковариата

    ВАЖНО: cluster_id — это индекс кластера (0..n_clusters-1),
    а не уникальный ID трактора. Все тракторы в одном кластере
    имеют одинаковые Z, x_climate, x_soil.
    """
    result: Dict[str, np.ndarray] = {}

    # ─── Загрузка rainfall anomaly для Z ────────────────────────────
    rain_path = Path("data/processed/weather/rainfall_anomaly.csv")
    if rain_path.exists():
        try:
            rdf = pd.read_csv(rain_path, encoding="utf-8")
            rdf = rdf[rdf["campaign"] == campaign]

            if len(rdf) >= 5 and "rainfall_anomaly" in rdf.columns:
                rain_anomaly = (
                    pd.to_numeric(rdf["rainfall_anomaly"], errors="coerce").dropna().values
                )
                n_clusters = len(rain_anomaly)

                # СТРУКТУРНАЯ выборка: каждый трактор → случайный кластер
                cluster_indices = rng.integers(0, n_clusters, size=n)

                # Z для каждого трактора = Z его кластера
                z_raw = rain_anomaly[cluster_indices]

                # Стандартизация Z (глобальная, не внутри кластера)
                z_mean = float(rain_anomaly.mean())
                z_std = float(rain_anomaly.std(ddof=1))
                if z_std < 1e-9:
                    z_std = 1.0

                result["Z"] = (z_raw - z_mean) / z_std * z_scale_factor
                # НЕ добавляем jitter к Z — он должен быть cluster-level

                result["cluster_indices"] = cluster_indices  # ← ВАЖНО!

                logger.info(
                    "Z = rainfall anomaly (NASA POWER): "
                    "mean=%.2f, std=%.2f, n_clusters=%d, n_tractors=%d",
                    z_mean,
                    z_std,
                    n_clusters,
                    n,
                )
            else:
                raise ValueError("Недостаточно данных в rainfall_anomaly.csv")
        except Exception as exc:
            logger.warning("Не удалось загрузить rainfall anomaly: %s", exc)

    # Fallback для Z
    if "Z" not in result:
        result["Z"] = rng.normal(0, 1, size=n) * z_scale_factor
        result["cluster_indices"] = np.arange(n)  # fallback: каждый трактор = свой кластер
        logger.warning("Z: синтетический fallback")

    # ─── Загрузка working_days для x_climate ────────────────────────
    weather_path = Path("data/processed/weather/weather_windows.csv")
    if weather_path.exists():
        try:
            wdf = pd.read_csv(weather_path, encoding="utf-8")
            wdf = wdf[wdf["campaign"] == campaign]

            if len(wdf) >= 5 and "working_days_window" in wdf.columns:
                working_days = (
                    pd.to_numeric(wdf["working_days_window"], errors="coerce").dropna().values
                )

                # Используем ТЕ ЖЕ cluster_indices, что и для Z
                cluster_indices = result["cluster_indices"]

                # x_climate для каждого трактора = x_climate его кластера
                climate_raw = working_days[cluster_indices % len(working_days)]

                # Нормализация в [0, 1]
                c_min = float(working_days.min())
                c_max = float(working_days.max())
                if c_max > c_min:
                    result["x_climate"] = (climate_raw - c_min) / (c_max - c_min)
                else:
                    result["x_climate"] = np.full(n, 0.5)

                # Добавляем jitter к x_climate (индивидуальная вариация)
                result["x_climate"] = np.clip(
                    result["x_climate"] + rng.normal(0, jitter_std, size=n), 0.0, 1.0
                )

                logger.info(
                    "x_climate = working_days (NASA POWER): range=[%.0f, %.0f] дней, n=%d записей",
                    c_min,
                    c_max,
                    len(working_days),
                )
            else:
                raise ValueError("Недостаточно данных в weather_windows.csv")
        except Exception as exc:
            logger.warning("Не удалось загрузить weather: %s", exc)

    # Fallback для x_climate
    if "x_climate" not in result:
        result["x_climate"] = rng.beta(2.5, 1.5, size=n)
        result["x_climate"] = np.clip(
            result["x_climate"] + rng.normal(0, jitter_std, size=n), 0.0, 1.0
        )
        logger.warning("x_climate: синтетический fallback")

    # ─── Загрузка soil moisture для x_soil ──────────────────────────
    soil_path = Path("data/processed/soil/soil_windows.csv")
    if soil_path.exists():
        try:
            sdf = pd.read_csv(soil_path, encoding="utf-8")
            sdf = sdf[sdf["campaign"] == campaign]

            soil_col = None
            if "soil_index_normalized" in sdf.columns:
                soil_col = "soil_index_normalized"
            elif "soil_index" in sdf.columns:
                soil_col = "soil_index"

            if soil_col and len(sdf) >= 5:
                soil_vals = pd.to_numeric(sdf[soil_col], errors="coerce").dropna().values

                # Используем ТЕ ЖЕ cluster_indices, что и для Z
                cluster_indices = result["cluster_indices"]

                # x_soil для каждого трактора = x_soil его кластера
                soil_raw = soil_vals[cluster_indices % len(soil_vals)]

                if soil_col == "soil_index":
                    s_min = float(soil_vals.min())
                    s_max = float(soil_vals.max())
                    if s_max > s_min:
                        soil_raw = (soil_raw - s_min) / (s_max - s_min)
                    else:
                        soil_raw = np.full(n, 0.5)

                result["x_soil"] = np.clip(soil_raw, 0.0, 1.0)

                # Добавляем jitter к x_soil (индивидуальная вариация)
                result["x_soil"] = np.clip(
                    result["x_soil"] + rng.normal(0, jitter_std, size=n), 0.0, 1.0
                )

                logger.info("x_soil = soil moisture (GLDAS-2.1): n=%d записей", len(soil_vals))
            else:
                raise ValueError("Недостаточно данных в soil_windows.csv")
        except Exception as exc:
            logger.warning("Не удалось загрузить soil: %s", exc)

    # Fallback для x_soil
    if "x_soil" not in result:
        result["x_soil"] = rng.beta(2.0, 2.5, size=n)
        result["x_soil"] = np.clip(result["x_soil"] + rng.normal(0, jitter_std, size=n), 0.0, 1.0)
        logger.warning("x_soil: синтетический fallback")

    return result


# ---------------------------------------------------------------------------
# Baseline survival simulation
# ---------------------------------------------------------------------------
def generate_base_instrument(
    rng: np.random.Generator,
    n: int,
    source: str = "normal",
    weather_mean_days: float = 45.0,
    weather_std_days: float = 12.0,
    weather_campaign: str = "sowing",
    weather_data_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Генерация базового инструмента Z.

    source:
    - "normal":       Z ~ N(0, 1)
    - "uniform":      Z ~ U(-2, 2)
    - "weather":      Z = синтетический Normal(mean, std)
    - "weather_real": Z = реальные данные из weather_windows.csv
    """
    source = str(source).lower()

    if source == "normal":
        z = rng.normal(0.0, 1.0, size=n)
    elif source == "uniform":
        z = rng.uniform(-2.0, 2.0, size=n)
    elif source == "weather":
        # Синтетический погодный инструмент (legacy)
        window_days = rng.normal(weather_mean_days, weather_std_days, size=n)
        window_days = np.clip(window_days, 5.0, 90.0)
        z = window_days
    elif source == "weather_real":
        # Фаза 6.6: реальные данные из weather_windows.csv
        try:
            real_z = load_real_weather_windows(
                path=weather_data_path,
                campaign=weather_campaign,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "Не удалось загрузить реальные погодные данные: %s. "
                "Fallback на синтетический weather.",
                exc,
            )
            window_days = rng.normal(weather_mean_days, weather_std_days, size=n)
            window_days = np.clip(window_days, 5.0, 90.0)
            z = window_days
        else:
            # Повторная выборка с возвращением для n наблюдений
            indices = rng.integers(0, len(real_z), size=n)
            z = real_z[indices]
    else:
        raise ValueError(f"Unknown instrument_source: '{source}'")

    std = float(np.std(z, ddof=0))
    if (not math.isfinite(std)) or std < 1e-12:
        raise ValueError("Instrument has zero variance.")
    z = (z - z.mean()) / std
    return z


def _simulate_event_times(
    safe_lp: np.ndarray,
    baseline_hazard: float,
    baseline_family: str,
    baseline_shape: Optional[float],
    u_event: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    family = str(baseline_family).lower()

    if family not in _VALID_BASELINE_FAMILIES:
        raise ValueError(f"Unknown baseline_family: '{family}'")

    if baseline_hazard <= 0.0:
        raise ValueError("baseline_hazard must be > 0")

    if np.any(np.isnan(safe_lp)):
        raise ValueError("Linear predictor contains NaN")

    if family == "weibull":
        if baseline_shape is None:
            raise ValueError("Weibull requires baseline_shape")

        k = float(baseline_shape)

        if not math.isfinite(k) or k <= 0.0:
            raise ValueError(f"Weibull baseline_shape must be > 0. Got {k}")

        lam = baseline_hazard
        denom = lam * np.exp(safe_lp)
        denom = np.clip(denom, 1e-300, 1e300)

        true_time = np.power(-np.log(u_event) / denom, 1.0 / k)
        true_time = np.nan_to_num(true_time, nan=1e-12, posinf=1e12, neginf=1e-12)
        true_time = np.maximum(true_time, 1e-12)

        individual_hazard = lam * k * np.power(true_time, k - 1.0) * np.exp(safe_lp)
        individual_hazard = np.nan_to_num(
            individual_hazard,
            nan=0.0,
            posinf=1e300,
            neginf=0.0,
        )

    elif family == "gompertz":
        if baseline_shape is None:
            raise ValueError("Gompertz requires baseline_shape")

        b = float(baseline_shape)

        if not math.isfinite(b) or b <= 0.0:
            raise ValueError(f"Gompertz baseline_shape must be > 0. Got {b}")

        lam = baseline_hazard
        denom = lam * np.exp(safe_lp)
        denom = np.clip(denom, 1e-300, 1e300)

        arg = 1.0 - (b * np.log(u_event)) / denom
        arg = np.where(np.isfinite(arg), arg, 1.0)
        arg = np.maximum(arg, 1e-12)

        true_time = (1.0 / b) * np.log(arg)
        true_time = np.nan_to_num(true_time, nan=1e-12, posinf=1e12, neginf=1e-12)
        true_time = np.maximum(true_time, 1e-12)

        b_t = b * true_time
        clipped_bt = b_t > 700.0

        if clipped_bt.any():
            frac = float(clipped_bt.mean())
            if frac > 1e-6:
                logger.warning(f"Gompertz b*t clipped in {frac:.6%} rows")
            b_t = np.where(clipped_bt, 700.0, b_t)

        individual_hazard = denom * np.exp(b_t)
        individual_hazard = np.nan_to_num(
            individual_hazard,
            nan=0.0,
            posinf=1e300,
            neginf=0.0,
        )

    else:  # exponential
        individual_hazard = baseline_hazard * np.exp(safe_lp)
        individual_hazard = np.clip(individual_hazard, 1e-300, 1e300)

        true_time = -np.log(u_event) / individual_hazard
        true_time = np.nan_to_num(true_time, nan=1e-12, posinf=1e12, neginf=1e-12)
        true_time = np.maximum(true_time, 1e-12)

    return true_time, individual_hazard


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def _generate_power_by_segment(
    segment: str,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """P-10 / Фаза 4.1: генерация мощности по сегменту парка.

    light: Normal(140, 50), клип [50, 350]
    heavy: выборка из RF_HEAVY_BRAND_CATALOG + шум Normal(0, 15), клип [200, 500]
    """
    segment = str(segment).lower()

    # ─── Heavy segment: мощный парк РФ ─────────────────────────────
    if segment == "heavy" and RF_HEAVY_BRAND_CATALOG:
        names = list(RF_HEAVY_BRAND_CATALOG.keys())
        shares = np.array(
            [RF_HEAVY_BRAND_CATALOG[x]["share"] for x in names],
            dtype=float,
        )
        shares = np.clip(shares, 0.0, None)
        total = float(shares.sum())

        if total > 0.0:
            probs = shares / total
            chosen = rng.choice(names, size=n, p=probs)
            power = np.array(
                [RF_HEAVY_BRAND_CATALOG[x]["power_hp"] for x in chosen],
                dtype=float,
            )
        else:
            # Защитный fallback: каталог есть, но доли некорректны.
            power = rng.normal(300.0, 60.0, size=n)

        power = power + rng.normal(0.0, 15.0, size=n)
        return np.clip(power, 200.0, 500.0)

    # ─── Light segment ─────────────────────────────────────────────
    power = rng.normal(140.0, 50.0, size=n)
    return np.clip(power, 50.0, 350.0)


def generate_production_year(
    rng: np.random.Generator,
    n: int,
    observation_year: int = 2009,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Фаза 4.3: генерация когорт выпуска и возраста на момент наблюдения.
    Когорты основаны на мониторинге РФ 2005–2009.
    """
    production_years = rng.choice(
        [2003, 2005, 2006, 2007, 2008, 2009],
        p=[0.10, 0.15, 0.20, 0.25, 0.20, 0.10],
        size=n,
    )
    age_at_observation = observation_year - production_years
    return production_years, age_at_observation.astype(float)


def generate_data(
    n: int,
    contamination: bool,
    baseline_hazard: float,
    censoring_scale: float,
    rng: np.random.Generator,
    instrument_strength: Optional[float] = None,
    dgp: Optional[DGPParameters] = None,
    instrument_source: str = "normal",
    contamination_probability: float = 1.0,
) -> pd.DataFrame:
    """Generate synthetic survival data with endogenous treatment."""
    n = int(n)

    if n <= 0:
        raise ValueError("n must be > 0")
    if baseline_hazard <= 0.0:
        raise ValueError("baseline_hazard must be > 0")
    if censoring_scale <= 0.0:
        raise ValueError("censoring_scale must be > 0")

    dgp = _validate_dgp(dgp)

    if instrument_strength is None:
        instrument_strength = float(dgp.first_stage_z_coef)
    else:
        instrument_strength = _as_finite_float(instrument_strength, "instrument_strength")

    # Raw covariates
    # Фаза 4.3: production_year когорты вместо непрерывного age
    observation_year = 2009
    production_year, age = generate_production_year(rng, n, observation_year)
    age = np.clip(age, 0.0, 30.0)

    # P-02 / Фаза 4.4: Hours -> LogNormal или мониторинг РФ
    segment = str(getattr(dgp, "segment", "light")).lower()
    if segment == "heavy" and HOURS_PRIOR_RF:
        # Фаза 4.4: используем диапазоны из мониторинга РФ
        # Генерируем общую наработку за период, затем делим на годы
        hours_annual = np.zeros(n)
        brand_names = list(HOURS_PRIOR_RF.keys())
        for i in range(n):
            brand = rng.choice(brand_names)
            info = HOURS_PRIOR_RF[brand]
            total_hours = rng.uniform(info["min"], info["max"])
            hours_annual[i] = total_hours / info["years"]
        hours = np.clip(hours_annual, 100.0, 3000.0)
    else:
        # Light segment: LogNormal prior
        _hp = HOURS_PRIOR.get(segment, HOURS_PRIOR["light"])
        hours = rng.lognormal(
            mean=float(np.log(_hp["median"])),
            sigma=float(_hp["sigma"]),
            size=n,
        )
        hours = np.clip(hours, _hp["clip_min"], _hp["clip_max"])

    # ─── Гибридный режим: реальные weather/soil ───────────────────
    if getattr(dgp, "use_real_covariates", False):
        real_covs = load_real_covariates_for_simulation(
            n,
            rng,
            campaign=getattr(dgp, "weather_campaign", "sowing"),
            jitter_std=getattr(dgp, "jitter_std", 0.05),
            z_scale_factor=getattr(dgp, "z_scale_factor", 2.0),
        )
        # real_covs уже содержит numpy arrays, не вызываем .to_numpy()
        z_base = np.asarray(real_covs["Z"], dtype=float)
        climate = np.asarray(real_covs["x_climate"], dtype=float)
        soil = np.asarray(real_covs["x_soil"], dtype=float)

        # ─── ВАЖНО: извлекаем cluster_indices для правильной кластеризации ──────
        # Если 32 реальных записи размножаются до 40000 строк,
        # то cluster_id должен быть 0..31, а не 0..39999
        cluster_id_from_real = np.asarray(real_covs.get("cluster_indices", np.arange(n)), dtype=int)

        logger.info(
            "Гибридный режим: Z, x_climate, x_soil из реальных данных, "
            "cluster_id=%d unique values (n=%d)",
            len(np.unique(cluster_id_from_real)),
            n,
        )
    else:
        # Оригинальная генерация (синтетика)
        climate = rng.beta(2.5, 1.5, size=n)
        soil = rng.beta(2.0, 2.5, size=n)
        cluster_id_from_real = None  # Будет использоваться стандартная кластеризация

    # P-10: Power by segment
    power = _generate_power_by_segment(getattr(dgp, "segment", "light"), n, rng)

    brand_probs = dgp.brand_prob_by_code
    codes = sorted(brand_probs.keys())
    probs = np.array([brand_probs[c] for c in codes], dtype=float)
    probs = probs / probs.sum()
    brand_code = rng.choice(codes, size=n, p=probs).astype(int)

    # Errors
    eps_d, u = _generate_errors(
        rng=rng,
        n=n,
        rho=dgp.rho,
        contamination=contamination,
        contamination_probability=contamination_probability,
    )
    actual_error_corr = float(np.mean(eps_d * u))

    # Instrument (skip if hybrid mode already set z_base)
    if not getattr(dgp, "use_real_covariates", False):
        z_base = generate_base_instrument(
            rng,
            n,
            source=instrument_source,
            weather_mean_days=getattr(dgp, "weather_mean_days", 45.0),
            weather_std_days=getattr(dgp, "weather_std_days", 12.0),
            weather_campaign=getattr(dgp, "weather_campaign", "sowing"),
            weather_data_path=None,
        )
    alpha = dgp.corr_zu
    u_std = _standardize_array(u)
    z = np.sqrt(max(0.0, 1.0 - alpha * alpha)) * z_base + alpha * u_std

    if not np.isfinite(np.var(z, ddof=0)):
        raise FloatingPointError("Instrument Z variance is non-finite")

        # Standardized continuous X
    x_age = _standardize_x_column("x_age", age)
    x_hours = _standardize_x_column("x_hours", hours)
    x_climate = _standardize_x_column("x_climate", climate)
    x_soil = _standardize_x_column("x_soil", soil)
    x_power = _standardize_x_column("x_power", power)

    # ─── Фаза EQI: Enterprise Quality Index ────────────────────────
    if getattr(dgp, "use_enterprise_quality", False):
        x_enterprise_quality, enterprise_ids = generate_enterprise_quality(
            rng=rng,
            n_tractors=n,
            n_enterprises=getattr(dgp, "n_enterprises", 500),
        )
        if not validate_enterprise_structure(x_enterprise_quality, enterprise_ids):
            raise ValueError("Enterprise quality structure validation failed")
    else:
        x_enterprise_quality = np.zeros(n, dtype=float)
        enterprise_ids = np.zeros(n, dtype=int)

    # ─── Interaction: центрирование по выборочным средним ─────────────
    # ДОЛЖНО БЫТЬ СТРОГО ПОСЛЕ строк выше!
    x_age_mean = float(np.mean(x_age))
    x_hours_mean = float(np.mean(x_hours))
    x_age_hours = (x_age - x_age_mean) * (x_hours - x_hours_mean)

    # Стандартизация interaction для численной стабильности
    x_age_hours_mean = float(np.mean(x_age_hours))
    x_age_hours_std = float(np.std(x_age_hours, ddof=1))
    if x_age_hours_std < 1e-9:
        x_age_hours_std = 1.0
    x_age_hours = (x_age_hours - x_age_hours_mean) / x_age_hours_std

    logger.info(
        "x_age_hours diagnostics: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        float(np.mean(x_age_hours)),
        float(np.std(x_age_hours)),
        float(np.min(x_age_hours)),
        float(np.max(x_age_hours)),
    )

    # Legacy brand continuous score (kept for diagnostics)
    x_brand_legacy = _standardize_x_column("x_brand", brand_code.astype(float))

    # Brand design terms
    ref_code = int(dgp.brand_reference_code)
    brand_encoding = dgp.brand_encoding.lower()

    # ─── Первая часть: базовый PeakLoad без бренда ─────────────────────
    peak_load_base = (
        dgp.intercept
        + instrument_strength * z
        + dgp.fs_age_coef * x_age
        + dgp.fs_hours_coef * x_hours
        + dgp.fs_climate_coef * x_climate
        + dgp.fs_soil_coef * x_soil
        + dgp.fs_power_coef * x_power
        + eps_d
    )
    struct_int = (
        float(dgp.structural_intercept)
        if dgp.structural_intercept is not None
        else float(dgp.intercept)
    )

    # ─── Фаза 3: калибровка маргинального распределения под TUM ──────
    # Линейная перенормировка a + b*(x - mean) сохраняет ранговую
    # корреляцию peak_load с eps_d (и, следовательно, эндогенность),
    # но подгоняет среднее и дисперсию под эмпирическое распределение TUM.
    # Применяем ДО добавления бренд-эффектов, чтобы они тоже масштабировались.
    if (
        getattr(dgp, "peakload_target_mean", None) is not None
        and getattr(dgp, "peakload_target_std", None) is not None
    ):
        _pl_target_mean = float(dgp.peakload_target_mean)
        _pl_target_std = float(dgp.peakload_target_std)
        if _pl_target_std > 1e-9:
            _pl_mean = float(np.mean(peak_load_base))
            _pl_std = float(np.std(peak_load_base, ddof=1))
            if _pl_std > 1e-9:
                peak_load_base = (
                    _pl_target_mean + (peak_load_base - _pl_mean) / _pl_std * _pl_target_std
                )
                struct_int = _pl_target_mean + (struct_int - _pl_mean) / _pl_std * _pl_target_std
                logger.info(
                    "PeakLoad перенормирован под TUM: mean=%.4f → %.4f, std=%.4f → %.4f",
                    _pl_mean,
                    _pl_target_mean,
                    _pl_std,
                    _pl_target_std,
                )

    # Теперь peak_load будет построен на основе перенормированного base
    peak_load = peak_load_base.copy()

    # Инициализируем lp_raw заранее, чтобы избежать UnboundLocalError
    lp_raw = None

    if brand_encoding == "legacy_continuous":
        peak_load += dgp.fs_brand_coef * x_brand_legacy
        lp_raw = dgp.beta_brand * x_brand_legacy

    elif brand_encoding == "dummies":
        fs_coefs = dgp.fs_brand_coefs
        if fs_coefs is None:
            fs_coefs = _default_brand_coefs(dgp.fs_brand_coef, ref_code)

        beta_coefs = dgp.beta_brand_coefs
        if beta_coefs is None:
            beta_coefs = _default_brand_coefs(dgp.beta_brand, ref_code)

        # Начинаем с нулевого вклада бренда в lp_raw
        lp_raw = np.zeros(n, dtype=float)
        for code in BRAND_MAP.keys():
            if int(code) == ref_code:
                continue

            dummy = (brand_code == int(code)).astype(float)
            peak_load += float(fs_coefs.get(int(code), 0.0)) * dummy
            lp_raw += float(beta_coefs.get(int(code), 0.0)) * dummy

    else:
        raise ValueError("Invalid brand_encoding after DGP validation")

    # PATCH-04: Transform PeakLogit to (0, 1) using sigmoid instead of hard clipping
    # Hard clipping creates Tobit-like censoring that breaks OLS assumptions in first stage.
    # Sigmoid transformation preserves continuity and differentiability.
    n_clipped_before = int(((peak_load < 0.0) | (peak_load > 1.0)).sum())
    if n_clipped_before > 0:
        logger.warning(
            "PeakLoad transformed via sigmoid for %d/%d observations (%.1f%%) to avoid Tobit censoring bias",
            n_clipped_before, n, 100*n_clipped_before/n,
        )
    # Apply sigmoid with clipping to prevent overflow: sigma(x) = 1/(1+exp(-x))
    peak_load = 1.0 / (1.0 + np.exp(-np.clip(peak_load, -10.0, 10.0)))

    # ─── Построение linear predictor ────────────────────────────────────
    lp_raw = (
        dgp.gamma * (peak_load - struct_int)
        + dgp.beta_age * x_age
        + dgp.beta_hours * x_hours
        + getattr(dgp, "beta_age_hours", 0.0) * x_age_hours
        + dgp.beta_climate * x_climate
        + dgp.beta_soil * x_soil
        + dgp.beta_power * x_power
        + getattr(dgp, "beta_eqi", 0.0) * x_enterprise_quality
        + dgp.delta * u
    )

    # Clip linear predictor if requested
    if dgp.clip_lp is not None:
        clip_val = abs(float(dgp.clip_lp))
        clipped_mask = (lp_raw > clip_val) | (lp_raw < -clip_val)
        lp = np.clip(lp_raw, -clip_val, clip_val)

        if clipped_mask.any():
            logger.warning(
                f"Linear predictor clipped in {clipped_mask.sum()} rows "
                f"({clipped_mask.mean() * 100:.1f}%)."
            )
    else:
        clipped_mask = np.zeros_like(lp_raw, dtype=bool)
        lp = lp_raw

    if np.any(np.isnan(lp)):
        raise ValueError("Linear predictor contains NaN after clipping")

    safe_lp = np.clip(lp, -600.0, 600.0)

    # Simulate times
    u_event = rng.uniform(low=np.nextafter(0.0, 1.0), high=1.0, size=n)
    u_censor = rng.uniform(low=np.nextafter(0.0, 1.0), high=1.0, size=n)

    true_time, individual_hazard = _simulate_event_times(
        safe_lp=safe_lp,
        baseline_hazard=baseline_hazard,
        baseline_family=dgp.baseline_family,
        baseline_shape=dgp.baseline_shape,
        u_event=u_event,
    )

    censoring_time = -np.log(u_censor) * censoring_scale
    censoring_time = np.nan_to_num(censoring_time, nan=1e12, posinf=1e12, neginf=1e-12)
    censoring_time = np.maximum(censoring_time, 1e-12)

    # P-01: Competing risks
    if getattr(dgp, "competing_risks", False):
        u_minor = rng.uniform(low=np.nextafter(0.0, 1.0), high=1.0, size=n)
        minor_hazard = dgp.minor_failure_rate * np.exp(safe_lp)
        minor_hazard = np.clip(minor_hazard, 1e-300, 1e300)

        true_minor_time = -np.log(u_minor) / minor_hazard
        true_minor_time = np.nan_to_num(true_minor_time, nan=1e12, posinf=1e12, neginf=1e-12)
        true_minor_time = np.maximum(true_minor_time, 1e-12)
        event_minor = true_minor_time <= censoring_time
    else:
        true_minor_time = np.full(n, 1e12, dtype=float)
        event_minor = np.zeros(n, dtype=bool)

    event_def = getattr(dgp, "event_definition", "major_claim")

    # P-03: event_definition affects event/time
    if getattr(dgp, "competing_risks", False) and event_def == "any_failure":
        observed_time = np.minimum(np.minimum(true_time, true_minor_time), censoring_time)

        major_first = true_time <= np.minimum(true_minor_time, censoring_time)
        minor_first = (true_minor_time < true_time) & (true_minor_time <= censoring_time)

        event = major_first | minor_first
        failure_type = np.where(
            major_first,
            "major",
            np.where(minor_first, "minor", "censored"),
        )
    else:
        observed_time = np.minimum(true_time, censoring_time)
        event = true_time <= censoring_time
        if getattr(dgp, "competing_risks", False):
            # ──────────────────────────────────────────────────────────
            # v3.2-fix: minor-отказ НЕ preempt-ит major для major_claim.
            # Minor-отказы — рецидивирующие события, а не конкурирующие
            # риски в строгом смысле. Трактор может иметь несколько
            # minor-отказов и всё равно получить major.
            #
            # Для total_loss / major_claim:
            #   event = major-отказ до цензуры (minor не влияет)
            #   failure_type = "major" если major, иначе "censored"
            #   time_minor записывается как доп. информация
            #
            # Для any_failure:
            #   event = major ИЛИ minor (первый из них)
            #   failure_type = "major" или "minor"
            # ──────────────────────────────────────────────────────────
            # Minor НЕ preempt-ит major. Просто записываем, был ли minor.
            failure_type = np.where(
                event,
                "major",
                np.where(
                    event_minor & (true_minor_time <= observed_time),
                    "minor_observed",  # minor был, но major не произошёл
                    "censored",
                ),
            )
        else:
            failure_type = np.where(event, "major", "censored")

    observed_time = np.nan_to_num(observed_time, nan=1e-12, posinf=1e12, neginf=1e-12)
    observed_time = np.maximum(observed_time, 1e-12)

    true_minor_time = np.nan_to_num(true_minor_time, nan=1e12, posinf=1e12, neginf=1e-12)
    time_minor = np.where(event_minor, true_minor_time, observed_time)
    downtime_hours = downtime_hours_from_failure_type(failure_type)

    # ─── Cluster IDs: корректная кластерная структура ──────────────
    # Приоритет:
    #   1. Гибридный режим → cluster_indices из реальных данных
    #   2. EQI режим → enterprise_id (внутрипредприятиенная корреляция)
    #   3. Стандартный → production_year × campaign_group
    if cluster_id_from_real is not None:
        cluster_id = cluster_id_from_real
        logger.info(
            "Cluster ID: cluster_indices из реальных данных (%d уникальных)",
            len(np.unique(cluster_id)),
        )
    elif getattr(dgp, "use_enterprise_quality", False):
        # PATCH-10: предприятие как кластер.
        # Все тракторы одного предприятия имеют одинаковый EQI,
        # что создаёт внутрипредприятиенную корреляцию ошибок.
        # Стандартная кластеризация (год × кампания) эту корреляцию
        # НЕ учитывает, что приводит к заниженным кластерным SE.
        cluster_id = enterprise_ids.astype(str)
        n_ent = len(np.unique(enterprise_ids))
        logger.info(
            "Cluster ID: enterprise_id (%d предприятий, EQI режим)",
            n_ent,
        )
    else:
        n_years = int(np.unique(production_year).size)
        n_campaign_groups = 4
        campaign_group = rng.integers(1, n_campaign_groups + 1, size=n)
        cluster_id = production_year.astype(str) + "_C" + campaign_group.astype(str)

    data = pd.DataFrame(
        {
            "time": observed_time,
            "event": event.astype(bool),
            "PeakLoad": peak_load,
            "Z": z,
            "Age": age,
            "production_year": production_year,
            "Hours": hours,
            "Climate": climate,
            "Soil": soil,
            "Brand": brand_code,
            "brand_code": brand_code,
            "Power": power,
            "eps_D": eps_d,
            "U": u,
            "T_true": true_time,
            "C": censoring_time,
            "u_event": u_event,
            "u_censor": u_censor,
            "clipped_lp": clipped_mask,
            "lp_raw": lp_raw,
            "lp": lp,
            "individual_hazard": individual_hazard,
            "x_age": x_age,
            "x_hours": x_hours,
            "x_climate": x_climate,
            "x_soil": x_soil,
            "x_power": x_power,
            "x_brand": x_brand_legacy,
            "x_age_hours": x_age_hours,
            # ─── Фаза EQI ──────────────────────────────────────────────────
            "x_enterprise_quality": x_enterprise_quality,
            "enterprise_id": enterprise_ids,
            # P-01 / v0.2 columns
            "T_minor": true_minor_time,
            "time_minor": time_minor,
            "event_minor": event_minor.astype(bool),
            "failure_type": failure_type,
            "event_definition": event_def,
            "downtime_hours": downtime_hours,
            "cluster_id": cluster_id,
        }
    )

    # Brand dummy columns
    for code, name in BRAND_MAP.items():
        data[f"brand_{name}"] = (brand_code == int(code)).astype(float)

    numeric = data.select_dtypes(include=[np.number])

    if not np.all(np.isfinite(numeric.to_numpy())):
        raise FloatingPointError("Generated data contain non-finite numeric values.")

    if not (data["time"] > 0).all():
        raise FloatingPointError("Survival times must be positive")

    data.attrs["error_diagnostics"] = {
        "target_rho": float(dgp.rho),
        "actual_pearson_rho": actual_error_corr,
        "contamination": bool(contamination),
        "contamination_probability": float(contamination_probability),
    }

    # ─── Патч 4: interaction_params для инференса ─────────────────────
    # prediction_engine.py и Real_calculator.py читают это из training_meta
    data.attrs["interaction_params"] = {
        "x_age_mean": float(x_age_mean),
        "x_hours_mean": float(x_hours_mean),
        "x_age_hours_mean": float(x_age_hours_mean),
        "x_age_hours_std": float(x_age_hours_std),
    }

    # ─── Патч 4.2: информация о валидности инструмента ──────────────
    data.attrs["instrument_exogeneity"] = {
        "corr_zu": float(dgp.corr_zu),
        "valid": abs(dgp.corr_zu) < 1e-12,
    }

    return data


# ---------------------------------------------------------------------------
# Фаза 7: обучение на литературных claims-данных
# ---------------------------------------------------------------------------
def load_claims_for_training(
    path: str = "data/processed/claims/claims_clean.csv",
) -> pd.DataFrame:
    """
    Загрузить литературные claims-данные для обучения.

    Parameters
    ----------
    path : str
        Путь к claims_clean.csv.

    Returns
    -------
    pd.DataFrame
        Claims с обязательными колонками.

    Raises
    ------
    FileNotFoundError
        Если файл не найден.
    ValueError
        Если отсутствуют обязательные колонки.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Claims не найдены: {path}. "
            f"Сначала запустите generate_literature_claims_v2.py и "
            f"claims_validator.py"
        )

    df = pd.read_csv(p, encoding="utf-8")

    required = {"failure_time", "event_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {sorted(missing)}")

    n_events = int(df["event_flag"].sum())
    logger.info(
        "Загружено claims для обучения: %d строк, %d событий",
        len(df),
        n_events,
    )
    return df


def prepare_claims_for_cf(
    claims_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Подготовить claims для CF Cox estimation.
    Создаёт PeakLoad, X ковариаты, brand dummies, Z инструмент.

    Parameters
    ----------
    claims_df : pd.DataFrame
        Claims с колонками failure_time, event_flag, peak_load_proxy, и т.д.

    Returns
    -------
    pd.DataFrame
        Готовый DataFrame для first stage и CF Cox.
    """
    data = pd.DataFrame()

    # Survival time и event
    data["time"] = pd.to_numeric(claims_df["failure_time"], errors="coerce")
    data["event"] = pd.to_numeric(claims_df["event_flag"], errors="coerce").astype(int)

    # Удалить строки с невалидным time
    valid = data["time"].notna() & (data["time"] > 0)
    data = data[valid].reset_index(drop=True)
    claims_df = claims_df[valid].reset_index(drop=True)

    # PeakLoad
    if "peak_load_proxy" in claims_df.columns:
        data["PeakLoad"] = pd.to_numeric(claims_df["peak_load_proxy"], errors="coerce").fillna(0.71)
    else:
        logger.warning("peak_load_proxy отсутствует, используется 0.71")
        data["PeakLoad"] = 0.71

    # X ковариаты
    x_map = {
        "age_at_event": "x_age",
        "hours_at_event": "x_hours",
        "power_hp": "x_power",
        "climate": "x_climate",
        "soil": "x_soil",
    }
    for src, dst in x_map.items():
        if src in claims_df.columns:
            vals = pd.to_numeric(claims_df[src], errors="coerce")
            data[dst] = vals.fillna(vals.median() if vals.notna().any() else 0.0)
        else:
            # Генерируем синтетические данные для отсутствующих
            rng = np.random.default_rng(hash(dst) % 2**31)
            if dst == "x_climate":
                data[dst] = rng.beta(2.5, 1.5, size=len(data))
            elif dst == "x_soil":
                data[dst] = rng.beta(2.0, 2.5, size=len(data))
            else:
                data[dst] = 0.0
            logger.warning("Колонка %s отсутствует, %s сгенерирован", src, dst)

    # Brand dummies
    if "brand" in claims_df.columns:
        brand_col = claims_df["brand"].astype(str)
        for code, name in BRAND_MAP.items():
            if int(code) == 0:
                continue  # MTZ82 — референсная
            data[f"brand_{name}"] = (brand_col == name).astype(float)
    else:
        for code, name in BRAND_MAP.items():
            if int(code) == 0:
                continue
            data[f"brand_{name}"] = 0.0
        logger.warning("Колонка brand отсутствует, dummies = 0")

    # Z инструмент (weather_instrument)
    if "weather_instrument" in claims_df.columns:
        z_raw = pd.to_numeric(claims_df["weather_instrument"], errors="coerce")
        z_mean = z_raw.mean() if z_raw.notna().any() else 45.0
        z_std = z_raw.std() if z_raw.notna().any() and z_raw.std() > 0 else 1.0
        data["Z"] = ((z_raw.fillna(z_mean)) - z_mean) / z_std
        logger.info(
            "Z инструмент из weather_instrument: mean=%.2f, std=%.2f",
            z_mean,
            z_std,
        )
    else:
        # Fallback: синтетический Z
        rng_z = np.random.default_rng(42)
        data["Z"] = rng_z.normal(0, 1, size=len(data))
        logger.warning("weather_instrument отсутствует, Z сгенерирован синтетически")

    logger.info(
        "Claims подготовлены для CF Cox: %d наблюдений, %d событий",
        len(data),
        int(data["event"].sum()),
    )

    # НОВОЕ: Interaction для claims (центрированный и стандартизированный)
    if "x_age" in data.columns and "x_hours" in data.columns:
        x_age_mean = float(data["x_age"].mean())
        x_hours_mean = float(data["x_hours"].mean())
        raw_interaction = (data["x_age"] - x_age_mean) * (data["x_hours"] - x_hours_mean)
        x_age_hours_mean = float(raw_interaction.mean())
        x_age_hours_std = float(raw_interaction.std(ddof=1))
        if x_age_hours_std < 1e-9:
            x_age_hours_std = 1.0
        data["x_age_hours"] = (raw_interaction - x_age_hours_mean) / x_age_hours_std
        data.attrs["interaction_params"] = {
            "x_age_mean": x_age_mean,
            "x_hours_mean": x_hours_mean,
            "x_age_hours_mean": x_age_hours_mean,
            "x_age_hours_std": x_age_hours_std,
        }

    return data


# ---------------------------------------------------------------------------
# Фаза 7: адаптер statsmodels OLS для совместимости с compute_partial_out_fields
# ---------------------------------------------------------------------------
class FirstStageAdapter:
    """
    Адаптер statsmodels OLS результата к структуре, ожидаемой
    compute_partial_out_fields() и другими downstream-функциями.

    В симуляционной ветке fitted_fs имеет структуру:
        fitted_fs.fitted.fittedvalues

    В statsmodels OLS:
        ols_result.fittedvalues

    Этот адаптер имитирует первую структуру.
    """

    def __init__(self, ols_model):
        self._ols_model = ols_model
        self.fitted = ols_model  # Алиас: .fitted = сам результат OLS
        self.resid = ols_model.resid
        self.params = ols_model.params
        self.bse = ols_model.bse
        self.tvalues = ols_model.tvalues
        self.pvalues = ols_model.pvalues
        self.rsquared = ols_model.rsquared
        self.nobs = ols_model.nobs
        self.model = ols_model.model

    @property
    def fittedvalues(self):
        """Совместимость с прямым доступом."""
        return self._ols_model.fittedvalues


def run_first_stage_on_claims(
    data_mod: pd.DataFrame,
) -> Tuple[Any, np.ndarray, List[str], Dict[str, float], Dict[str, Any]]:
    """
    Оценить первую стадию на claims-данных.
    PeakLoad ~ Z + X

    Returns
    -------
    Tuple
        (fitted_fs, residuals, fs_names, fs_params, iv_diagnostics)
    """
    # Формула первой стадии
    exog_cols = ["const", "Z", "x_age", "x_hours", "x_climate", "x_soil", "x_power"]
    # Добавить brand dummies
    brand_cols = [c for c in data_mod.columns if c.startswith("brand_")]
    exog_cols.extend(brand_cols)

    # Убедиться, что все колонки есть
    for col in exog_cols:
        if col not in data_mod.columns:
            if col == "const":
                data_mod["const"] = 1.0
            else:
                data_mod[col] = 0.0

    y = data_mod["PeakLoad"].values
    X = data_mod[exog_cols].values

    # OLS
    X_with_const = sm.add_constant(X) if "const" not in exog_cols else X
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*covariance of constraints.*")
        ols_model = sm.OLS(y, X_with_const).fit(cov_type="HC3")

    residuals = ols_model.resid

    # IV-диагностики
    f_stat = float(ols_model.fvalue)

    # Partial R² для Z
    r2_full = ols_model.rsquared
    exog_no_z = [c for c in exog_cols if c != "Z"]
    X_no_z = data_mod[exog_no_z].values
    X_no_z_const = sm.add_constant(X_no_z) if "const" not in exog_no_z else X_no_z
    ols_no_z = sm.OLS(y, X_no_z_const).fit(cov_type="HC3")
    r2_no_z = ols_no_z.rsquared
    partial_r2 = max(0.0, r2_full - r2_no_z)

    # F-statistic для Z (t-статистика в квадрате для одного инструмента)
    z_idx = exog_cols.index("Z") if "Z" in exog_cols else None
    if z_idx is not None:
        # +1 потому что sm.add_constant добавляет const в позицию 0
        z_tstat = float(ols_model.tvalues.iloc[z_idx + 1])
        f_stat_z = float(z_tstat**2)
    else:
        f_stat_z = f_stat

    iv_diagnostics = {
        "f_statistic": f_stat_z,
        "f_statistic_weak": f_stat_z < 10.0,
        "partial_r2": partial_r2,
        "cragg_donald_stat": f_stat_z,
        "cragg_donald_weak": f_stat_z < 10.0,
        "endogenous": True,
        "instrument_adequate": f_stat_z >= 10.0,
        "partial_f_z": f_stat_z,  # ← частичный F для Z
    }

    logger.info(
        "Первая стадия на claims: F(Z)=%.2f, partial R²=%.4f, partial_f_z=%.2f",
        f_stat_z,
        partial_r2,
        f_stat_z,
    )

    # Оборачиваем в адаптер ПОСЛЕ всех расчётов
    fitted_adapter = FirstStageAdapter(ols_model)

    return (
        fitted_adapter,
        residuals,
        exog_cols,
        dict(zip(exog_cols, ols_model.params)),
        iv_diagnostics,
    )


def fit_cf_cox_on_claims(
    data_mod: pd.DataFrame,
    first_stage: Any,
    v_hat_basis: str = "linear",
    v_hat_basis_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Оценить CF Cox модель на claims-данных.

    Parameters
    ----------
    data_mod : pd.DataFrame
        Данные с PeakLoad, X, Z, time, event.
    first_stage : Any
        Результат первой стадии (OLS model).
    v_hat_basis : str
        Тип CF basis: "linear", "powers", "spline".
    v_hat_basis_params : dict, optional
        Параметры CF basis.

    Returns
    -------
    Any
        CFModelResult с cph, warnings, etc.
    """
    # Вычислить residuals из первой стадии
    residuals = first_stage.resid

    # Стандартизация residuals
    resid_mean = np.mean(residuals)
    resid_std = np.std(residuals, ddof=1)
    if resid_std < 1e-12:
        resid_std = 1.0

    if v_hat_basis == "linear":
        v_hat = (residuals - resid_mean) / resid_std
        data_mod = data_mod.copy()
        data_mod["v_hat"] = v_hat
        cf_cols = ["v_hat"]
    elif v_hat_basis == "powers":
        max_power = (v_hat_basis_params or {}).get("max_power", 2)
        data_mod = data_mod.copy()
        cf_cols = []
        for p in range(1, max_power + 1):
            col_name = f"v_hat_pow{p}"
            data_mod[col_name] = ((residuals - resid_mean) / resid_std) ** p
            cf_cols.append(col_name)
    else:
        # Fallback на linear
        v_hat = (residuals - resid_mean) / resid_std
        data_mod = data_mod.copy()
        data_mod["v_hat"] = v_hat
        cf_cols = ["v_hat"]

    # Cox exog: PeakLoad + X + CF
    cox_exog = ["PeakLoad", "x_age", "x_hours", "x_climate", "x_soil", "x_power"]
    brand_cols = [c for c in data_mod.columns if c.startswith("brand_")]
    cox_exog.extend(brand_cols)
    # Добавить x_age_hours если есть (PATCH-D4)
    if "x_age_hours" in data_mod.columns:
        cox_exog.append("x_age_hours")
    cox_exog.extend(cf_cols)

    # Убедиться, что все колонки есть
    for col in cox_exog:
        if col not in data_mod.columns:
            data_mod[col] = 0.0

    # Удалить колонки с нулевой дисперсией
    cols_to_keep = ["time", "event"]
    for col in cox_exog:
        if data_mod[col].std() > 1e-9:
            cols_to_keep.append(col)
        else:
            logger.warning("Колонка '%s' имеет нулевую дисперсию, удаляем", col)

    cox_data = data_mod[cols_to_keep].copy()

    # Обучение Cox PH
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(cox_data, duration_col="time", event_col="event")
    except Exception as exc:
        logger.warning("Обучение с penalizer=0.1 не удалось: %s. Пробую без регуляризации.", exc)
        cph = CoxPHFitter(penalizer=0.0)
        cph.fit(cox_data, duration_col="time", event_col="event")

    # Собрать результат
    cf = CFModelResult()
    cf.cph = cph
    cf.residuals = residuals
    cf.cf_basis = v_hat_basis
    cf.cf_basis_params = v_hat_basis_params or {}
    cf.warnings = []

    # --- ИЗВЛЕЧЕНИЕ GAMMA_HAT И СТАНДАРТНЫХ ОШИБОК ---
    # Ищем колонку контрольной функции (обычно 'v_hat' или начинается с 'v_hat')
    cf_cols = [c for c in cph.params_.index if c == "v_hat" or c.startswith("v_hat") or c == "cf"]
    cf_col = cf_cols[0] if cf_cols else "v_hat"

    if cf_col in cph.params_.index:
        cf.gamma_hat = float(cph.params_[cf_col])
        cf.cf_coef_signed = cf.gamma_hat
        cf.cf_coef = cf.gamma_hat
        try:
            # В lifelines стандартная ошибка хранится в summary
            cf.naive_model_se = float(cph.summary.loc[cf_col, "se(coef)"])
        except Exception:
            cf.naive_model_se = np.nan
    else:
        cf.gamma_hat = np.nan
        cf.naive_model_se = np.nan
        cf.cf_coef_signed = np.nan
        cf.cf_coef = np.nan

    logger.info(
        "CF Cox на claims обучен: %d ковариат, %d событий",
        len(cox_exog),
        int(cox_data["event"].sum()),
    )

    return cf


# ---------------------------------------------------------------------------
# Censoring calibration
# ---------------------------------------------------------------------------
def calibrate_censoring_scale_deterministic(
    target_event_rate: float,
    n_calib: int,
    rng: np.random.Generator,
    instrument_strength: Optional[float] = None,
    dgp: Optional[DGPParameters] = None,
    baseline_hazard: Optional[float] = None,
    instrument_source: str = "normal",
    tol: float = 0.005,
    max_iters: int = 40,
    post_check: bool = True,
    post_check_n: int = 5000,
    post_check_strengths: Optional[List[float]] = None,
    contamination: bool = False,
    contamination_probability: float = 1.0,
) -> CalibrationResult:
    """Calibrate censoring scale to achieve target event rate."""
    if not (0.0 < target_event_rate < 1.0):
        raise ValueError("target_event_rate must be in (0, 1)")

    n_calib = int(n_calib)
    if n_calib <= 0:
        raise ValueError("n_calib must be > 0")

    tol = float(tol)
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and > 0")
    # Для редких событий абсолютный tol=0.005 слишком грубый.
    # Иначе калибровка может принять event_rate=0 как допустимый.
    tol = min(tol, max(1e-6, 0.1 * float(target_event_rate)))

    max_iters = int(max_iters)
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0")

    post_check_n = int(post_check_n)
    if post_check_n <= 0:
        raise ValueError("post_check_n must be > 0")

    if baseline_hazard is None:
        baseline_hazard = DEFAULT_BASELINE_HAZARD
    if baseline_hazard <= 0.0:
        raise ValueError("baseline_hazard must be > 0")

    base = generate_data(
        n=n_calib,
        contamination=contamination,
        baseline_hazard=baseline_hazard,
        censoring_scale=1.0,
        rng=rng,
        instrument_strength=instrument_strength,
        dgp=dgp,
        instrument_source=instrument_source,
        contamination_probability=contamination_probability,
    )

    u_censor = np.asarray(base["u_censor"], dtype=float)
    true_time = np.asarray(base["T_true"], dtype=float)
    T_minor = np.asarray(base["T_minor"], dtype=float)

    event_def = (
        str(getattr(dgp, "event_definition", "major_claim")).lower() if dgp else "major_claim"
    )
    competing = bool(getattr(dgp, "competing_risks", False)) if dgp else False

    def event_rate_for_scale(scale: float) -> float:
        censoring_time = -np.log(u_censor) * scale

        if competing and event_def == "any_failure":
            event = (true_time <= censoring_time) | (T_minor <= censoring_time)
        elif competing:
            # v3.2-fix: minor НЕ preempt-ит major для major_claim/total_loss.
            # Minor-отказы — рецидивирующие события, не конкурирующие риски.
            event = true_time <= censoring_time
        else:
            event = true_time <= censoring_time

        return float(event.mean())

    low, high = 1e-6, 1e6
    er_low = event_rate_for_scale(low)
    er_high = event_rate_for_scale(high)

    attempts = 0

    while not (er_low <= target_event_rate <= er_high) and attempts < 12:
        low /= 10.0
        high *= 10.0
        er_low = event_rate_for_scale(low)
        er_high = event_rate_for_scale(high)
        attempts += 1

    if not (er_low <= target_event_rate <= er_high):
        if target_event_rate > er_high:
            hint = (
                "target_event_rate is above the maximum achievable event rate. "
                "Reduce target_event_rate, disable competing_risks, "
                "set event_definition='any_failure', decrease minor_failure_rate, "
                "or increase baseline_hazard/target_probability."
            )
        else:
            hint = (
                "target_event_rate is below the minimum achievable event rate. "
                "Increase target_event_rate or check DGP/censoring generation."
            )

        raise RuntimeError(
            f"Calibration: couldn't bracket target {target_event_rate:.3f}. "
            f"Achievable rate range after {attempts} expansions is "
            f"[{er_low:.4f}, {er_high:.4f}] "
            f"(low={low:g}, high={high:g}; competing_risks={competing}, "
            f"event_definition='{event_def}'). {hint}"
        )

    mid = high
    er_mid = np.nan
    converged = False

    for _ in range(max_iters):
        mid = math.sqrt(low * high)
        er_mid = event_rate_for_scale(mid)

        if not math.isfinite(er_mid):
            high = mid
            continue

        if abs(er_mid - target_event_rate) <= tol:
            converged = True
            break

        if er_mid < target_event_rate:
            low = mid
            er_low = er_mid
        else:
            high = mid
            er_high = er_mid

    if not converged:
        logger.warning("Censoring calibration did not reach tolerance; returning last iterate.")

    post_check_rates: Dict[float, float] = {}
    post_check_passed: Optional[bool] = None

    if post_check:
        check_seed_base = int(rng.integers(0, 2**31 - 1))

        if post_check_strengths is None:
            strengths_to_check = [
                float(instrument_strength) if instrument_strength is not None else 0.0
            ]
        else:
            strengths_to_check = [float(x) for x in post_check_strengths]

        for i, str_val in enumerate(strengths_to_check):
            try:
                check_rng = np.random.default_rng([check_seed_base, i])

                check_data = generate_data(
                    n=post_check_n,
                    contamination=contamination,
                    baseline_hazard=baseline_hazard,
                    censoring_scale=mid,
                    rng=check_rng,
                    instrument_strength=str_val,
                    dgp=dgp,
                    instrument_source=instrument_source,
                    contamination_probability=contamination_probability,
                )

                post_check_rates[str_val] = float(check_data["event"].mean())
            except Exception:
                post_check_rates[str_val] = np.nan

        finite_rates = [r for r in post_check_rates.values() if np.isfinite(r)]

        if finite_rates:
            max_deviation = max(abs(r - target_event_rate) for r in finite_rates)
            post_check_passed = max_deviation <= 0.02

            if not post_check_passed:
                logger.warning(f"Calibration post-check FAILED: max deviation={max_deviation:.3f}")
        else:
            post_check_passed = None

    return CalibrationResult(
        censoring_scale=float(mid),
        target_event_rate=float(target_event_rate),
        achieved_event_rate=float(er_mid) if math.isfinite(er_mid) else np.nan,
        converged=converged,
        iterations=max_iters,
        post_check_passed=post_check_passed,
        post_check_rates=post_check_rates,
    )


# ---------------------------------------------------------------------------
# CF basis helpers
# ---------------------------------------------------------------------------
def _remove_constant_direction(basis: np.ndarray) -> np.ndarray:
    """
    Remove the constant direction from a basis matrix in a principled way.
    This replaces arbitrary drop-first column logic.
    """
    B = np.asarray(basis, dtype=float)

    if B.ndim != 2:
        raise ValueError("basis must be 2D")
    if B.shape[1] == 0:
        raise ValueError("basis has no columns")
    if not np.all(np.isfinite(B)):
        raise ValueError("basis contains non-finite values")

    B_centered = B - B.mean(axis=0)
    U, s, _ = np.linalg.svd(B_centered, full_matrices=False)

    if s.size == 0:
        raise ValueError("Basis has no non-degenerate directions")

    tol = 1e-10 * max(B_centered.shape) * float(s[0])
    rank = int(np.sum(s > tol))

    if rank == 0:
        raise ValueError("Basis is collinear after removing constant direction")

    return U[:, :rank] * s[:rank]


def _v_hat_powers(v: np.ndarray, max_power: int = 2) -> np.ndarray:
    """Polynomial basis without intercept: [v, v^2, ..., v^max_power]."""
    v = np.asarray(v, dtype=float).reshape(-1)
    max_power = max(1, int(max_power))

    cols = []

    for p in range(1, max_power + 1):
        cols.append(v**p)

    return np.column_stack(cols)


def _v_hat_spline_basis_with_type(
    v: np.ndarray,
    knots: Optional[List[float]] = None,
) -> Tuple[np.ndarray, str, Optional[float], Optional[float]]:
    """Build clamped quadratic B-spline basis."""
    v = np.asarray(v, dtype=float).reshape(-1)

    if v.size == 0:
        return _v_hat_powers(v, max_power=2), "powers", None, None

    if not np.all(np.isfinite(v)):
        raise ValueError("CF basis input contains non-finite values")

    if knots is None:
        knots = list(np.percentile(v, [25, 50, 75]))

    knots = sorted(set(float(k) for k in knots if math.isfinite(float(k))))

    vmin = float(np.min(v))
    vmax = float(np.max(v))

    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return _v_hat_powers(v, max_power=2), "powers", None, None

    interior_knots = sorted(set(float(k) for k in knots if vmin < float(k) < vmax))

    interp = _scipy_interpolate

    if interp is None or not HAS_SCIPY_INTERPOLATE:
        return _v_hat_powers(v, max_power=2), "powers", vmin, vmax

    degree = 2
    t = [vmin] * (degree + 1) + interior_knots + [vmax] * (degree + 1)
    n_basis = len(t) - degree - 1

    if n_basis <= 0:
        return _v_hat_powers(v, max_power=2), "powers", vmin, vmax

    try:
        eye = np.eye(n_basis)
        bases = []

        for j in range(n_basis):
            bs = interp.BSpline(t, eye[j], k=degree, extrapolate=True)
            bases.append(bs(v))

        basis = np.column_stack(bases)

        if basis.shape[1] == 0:
            return _v_hat_powers(v, max_power=2), "powers", vmin, vmax

        if not np.all(np.isfinite(basis)):
            return _v_hat_powers(v, max_power=2), "powers", vmin, vmax

        return basis, "spline", vmin, vmax

    except Exception:  # noqa: BLE001
        return _v_hat_powers(v, max_power=2), "powers", vmin, vmax


@dataclass
class CFBuildResult:
    df_with_cf: pd.DataFrame
    column_names: List[str]
    residuals_std: float
    residuals_mean: float = 0.0
    basis_std_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    actual_basis: str = "linear"
    basis_kwargs: Dict[str, Any] = field(default_factory=dict)
    linear_standardized: bool = True


def _build_cf_columns(
    residuals: np.ndarray,
    v_hat_basis: str,
    v_hat_basis_params: Optional[Dict[str, Any]],
    base_df: pd.DataFrame,
) -> CFBuildResult:
    """Build control-function columns from first-stage residuals."""
    residuals = np.asarray(residuals, dtype=float).ravel()

    if residuals.size == 0:
        raise ValueError("residuals are empty")
    if len(residuals) != len(base_df):
        raise ValueError("residuals length does not match base_df rows")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals contain non-finite values")

    if v_hat_basis_params is None:
        bp: Dict[str, Any] = {}
    elif isinstance(v_hat_basis_params, dict):
        bp = v_hat_basis_params
    else:
        raise TypeError("v_hat_basis_params must be dict or None")

    df = base_df.copy()

    r_mean = float(np.mean(residuals))
    r_std = _safe_std(residuals, ddof=1, floor=1e-10)

    v_hat_basis = str(v_hat_basis).lower()

    if v_hat_basis == "linear":
        col = "v_hat"
        standardized = (residuals - r_mean) / r_std
        standardized = np.nan_to_num(
            standardized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if not np.all(np.isfinite(standardized)):
            raise ValueError("Linear CF column is non-finite")

        df[col] = standardized

        basis_std_params = {
            col: {
                "mean": r_mean,
                "std": r_std,
                "standardize": True,
            }
        }

        return CFBuildResult(
            df_with_cf=df,
            column_names=[col],
            residuals_std=r_std,
            residuals_mean=r_mean,
            basis_std_params=basis_std_params,
            actual_basis="linear",
            basis_kwargs={},
            linear_standardized=True,
        )

    elif v_hat_basis == "spline":
        v_std = (residuals - r_mean) / r_std
        v_std = np.nan_to_num(v_std, nan=0.0, posinf=0.0, neginf=0.0)

        n_knots = max(1, int(bp.get("n_knots", 2)))

        if n_knots >= 2:
            percentiles = list(np.linspace(0, 100, n_knots + 2)[1:-1])
        else:
            percentiles = [50.0]

        interior_knots = sorted(set(np.percentile(v_std, percentiles)))

        basis_vals, actual_basis, domain_min, domain_max = _v_hat_spline_basis_with_type(
            v_std, knots=interior_knots
        )
        basis = _remove_constant_direction(basis_vals)
        # PATCH-01: Сохранить проекционную матрицу для воспроизведения при инференсе
        proj_matrix = np.linalg.lstsq(basis_vals, basis, rcond=None)[0]
        rank = basis.shape[1]

        if actual_basis == "spline":
            col_names = [f"v_hat_s{i}" for i in range(rank)]
            basis_kwargs = {
                "knots": interior_knots,
                "n_knots": n_knots,
                "spline_degree": 2,
                "spline_domain_min": domain_min,
                "spline_domain_max": domain_max,
                "basis_transform": "constant_direction_removed",
                "projection_matrix": proj_matrix.tolist(),  # ← ДОБАВЛЕНО
            }
        else:
            col_names = [f"v_hat_pow{i}" for i in range(rank)]
            basis_kwargs = {
                "max_power": 2,
                "fallback_from": "spline",
                "basis_transform": "constant_direction_removed",
            }

        basis_std_params = {}

        for i, col_name in enumerate(col_names):
            vals = np.asarray(basis[:, i], dtype=float)
            m = float(np.mean(vals)) if vals.size else 0.0
            s = _safe_std(vals, ddof=1, floor=1e-10)

            standardized = (vals - m) / s
            standardized = np.nan_to_num(
                standardized,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            if not np.all(np.isfinite(standardized)):
                raise ValueError(f"CF basis column {col_name} is non-finite")

            basis_std_params[col_name] = {
                "mean": m,
                "std": s,
                "standardize": True,
            }
            df[col_name] = standardized

        return CFBuildResult(
            df_with_cf=df,
            column_names=col_names,
            residuals_std=r_std,
            residuals_mean=r_mean,
            basis_std_params=basis_std_params,
            actual_basis=actual_basis,
            basis_kwargs=basis_kwargs,
            linear_standardized=True,
        )

    elif v_hat_basis == "powers":
        v_std = (residuals - r_mean) / r_std
        v_std = np.nan_to_num(v_std, nan=0.0, posinf=0.0, neginf=0.0)

        max_power = max(1, int(bp.get("max_power", 2)))
        basis_vals = _v_hat_powers(v_std, max_power=max_power)
        basis = _remove_constant_direction(basis_vals)
        rank = basis.shape[1]

        col_names = [f"v_hat_pow{i}" for i in range(rank)]

        basis_std_params = {}

        for i, col_name in enumerate(col_names):
            vals = np.asarray(basis[:, i], dtype=float)
            m = float(np.mean(vals)) if vals.size else 0.0
            s = _safe_std(vals, ddof=1, floor=1e-10)

            standardized = (vals - m) / s
            standardized = np.nan_to_num(
                standardized,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            if not np.all(np.isfinite(standardized)):
                raise ValueError(f"CF basis column {col_name} is non-finite")

            basis_std_params[col_name] = {
                "mean": m,
                "std": s,
                "standardize": True,
            }
            df[col_name] = standardized

        # PATCH-01: Сохранить проекционную матрицу для powers basis
        proj_matrix = np.linalg.lstsq(basis_vals, basis, rcond=None)[0]
        
        return CFBuildResult(
            df_with_cf=df,
            column_names=col_names,
            residuals_std=r_std,
            residuals_mean=r_mean,
            basis_std_params=basis_std_params,
            actual_basis="powers",
            basis_kwargs={
                "max_power": max_power,
                "basis_transform": "constant_direction_removed",
                "projection_matrix": proj_matrix.tolist(),  # ← ДОБАВЛЕНО
            },
            linear_standardized=True,
        )

    else:
        raise ValueError(f"Unknown v_hat_basis: '{v_hat_basis}'")


# ---------------------------------------------------------------------------
# First stage
# ---------------------------------------------------------------------------
def classical_first_stage_f_statistic(fitted: Any) -> Tuple[float, float]:
    parameter_names = list(fitted.model.exog_names)

    if "Z" not in parameter_names:
        raise KeyError("Z missing")

    zpos = parameter_names.index("Z")
    R = np.zeros((1, len(parameter_names)))
    R[0, zpos] = 1.0

    t_res = fitted.t_test(R)

    t_val = _scalar_from_array(t_res.tvalue)
    p_val = _scalar_from_array(t_res.pvalue)

    if not math.isfinite(t_val):
        return float("nan"), float("nan")

    return float(t_val * t_val), float(p_val)


def partial_f_statistic_for_z(fitted: Any) -> float:
    """
    Частичный F-statistic для инструмента Z после контроля всех X.

    Для одного инструмента частичный F равен квадрату t-статистики
    коэффициента Z в полной регрессии первой стадии (которая уже
    включает все X). Это стандартный тест слабой идентификации
    после контроля конфаундеров.

    Критерий Stock & Yogo (2005): F > 10 для одного эндогенного
    регрессора при 5% уровне максимального IV-size.

    Parameters
    ----------
    fitted : Any
        Результат OLS первой стадии (statsmodels fitted model).

    Returns
    -------
    float
        Частичный F-statistic. NaN если вычислить не удалось.
    """
    try:
        parameter_names = list(fitted.model.exog_names)
        if "Z" not in parameter_names:
            return float("nan")
        zpos = parameter_names.index("Z")
        # Для полной регрессии t²(Z) = частичный F для Z | X
        t_stat = float(np.asarray(fitted.tvalues).ravel()[zpos])
        p_val = float(np.asarray(fitted.pvalues).ravel()[zpos])
        if not math.isfinite(t_stat):
            return float("nan")
        return float(t_stat**2)
    except Exception:  # noqa: BLE001
        return float("nan")


def cragg_donald_stat(fitted: Any) -> Tuple[float, float, bool]:
    """Cragg-Donald statistic for weak instrument detection.
    Returns:
    (statistic, critical_value_10pct, is_weak)
    where is_weak is True if statistic < critical_value.
    """
    try:
        from statsmodels.stats.diagnostic import cochrane_orcutt

        HAS_COCHRANE_ORCUTT = True
    except ImportError:
        cochrane_orcutt = None
        HAS_COCHRANE_ORCUTT = False

    # F statistic for Z coefficient
    f_stat, _ = classical_first_stage_f_statistic(fitted)
    f_stat = float(f_stat)

    # Stock-Yogo critical values (10% max IV size)
    # k = number of endogenous regressors (excluding Z)
    # Usually k=1 for single endogenous
    try:
        k_endog = len(fitted.model.endog_names)
    except Exception:
        k_endog = 1

    try:
        nobs = int(fitted.nobs)
    except Exception:
        nobs = 500

    try:
        k_instr = len(fitted.model.exog_names) - 1  # minus constant
    except Exception:
        k_instr = 2

    # Simplified Stock-Yogo 10% critical value for k=1 endogenous
    # More instruments → lower critical value
    critical_10pct = 16.38  # default for k=1, many instruments

    if k_instr <= 5 and k_endog == 1:
        critical_10pct = {
            1: 16.38,
            2: 14.94,
            3: 13.88,
            4: 13.07,
            5: 12.40,
        }.get(k_instr, 12.40)

    is_weak = f_stat < critical_10pct

    return f_stat, critical_10pct, is_weak


def robust_first_stage_hc3_stat(fitted: Any) -> Tuple[float, float]:
    robust = fitted.get_robustcov_results(cov_type="HC3")
    parameter_names = list(fitted.model.exog_names)

    if "Z" not in parameter_names:
        raise KeyError("Z missing")

    zpos = parameter_names.index("Z")
    R = np.zeros((1, len(parameter_names)))
    R[0, zpos] = 1.0

    try:
        t_res = robust.t_test(R)

        t_val = _scalar_from_array(t_res.tvalue)
        p_val = _scalar_from_array(t_res.pvalue)

        if not math.isfinite(t_val):
            return float("nan"), float("nan")

        return float(t_val * t_val), float(p_val)

    except Exception:  # noqa: BLE001
        tr = robust.f_test(R)

        f_val = _scalar_from_array(tr.fvalue)
        p_val = _scalar_from_array(tr.pvalue)

        if not math.isfinite(f_val):
            return float("nan"), float("nan")

        return float(f_val), float(p_val)

    except Exception:  # noqa: BLE001
        tr = robust.f_test(R)
        f_val = _scalar_from_array(tr.fvalue)
        p_val = _scalar_from_array(tr.pvalue)

        return f_val, p_val


def robust_first_stage_cluster_stat(
    fitted: Any,
    groups: np.ndarray,
) -> Tuple[float, float]:
    """
    Cluster-robust F-statistic for Z using clustered covariance.

    Использует cov_type="cluster" (не "CR1"), который является
    стандартным типом в statsmodels для cluster-robust SE.
    """
    try:
        nobs = fitted.nobs
        if len(groups) != nobs:
            logger.warning(
                "robust_first_stage_cluster_stat: len(groups)=%d != nobs=%d",
                len(groups),
                nobs,
            )
            return float("nan"), float("nan")
    except Exception as exc:
        logger.warning("robust_first_stage_cluster_stat: nobs check failed: %s", exc)
        return float("nan"), float("nan")

    try:
        # ВАЖНО: используем cov_type="cluster", а не "CR1"
        robust = fitted.get_robustcov_results(cov_type="cluster", groups=groups)
    except Exception as exc:
        logger.warning("robust_first_stage_cluster_stat: get_robustcov_results failed: %s", exc)
        return float("nan"), float("nan")

    parameter_names = list(fitted.model.exog_names)
    if "Z" not in parameter_names:
        logger.warning("robust_first_stage_cluster_stat: Z not in parameter_names")
        return float("nan"), float("nan")

    zpos = parameter_names.index("Z")
    R = np.zeros((1, len(parameter_names)))
    R[0, zpos] = 1.0

    try:
        f_res = robust.f_test(R)
        f_val = _scalar_from_array(f_res.fvalue)
        p_val = _scalar_from_array(f_res.pvalue)
        if not math.isfinite(f_val):
            logger.warning("robust_first_stage_cluster_stat: f_val not finite: %s", f_val)
            return float("nan"), float("nan")
        return float(f_val), float(p_val)
    except Exception as exc:
        logger.warning("robust_first_stage_cluster_stat: f_test failed: %s, trying t_test", exc)
        try:
            t_res = robust.t_test(R)
            t_val = _scalar_from_array(t_res.tvalue)
            p_val = _scalar_from_array(t_res.pvalue)
            if not math.isfinite(t_val):
                return float("nan"), float("nan")
            return float(t_val**2), float(p_val)
        except Exception as exc2:
            logger.warning("robust_first_stage_cluster_stat: t_test also failed: %s", exc2)
            return float("nan"), float("nan")


def partial_r2_from_fitted(fitted: Any) -> float:
    try:
        r2_full = float(fitted.rsquared)
        dataX = fitted.model.exog
        param_names = fitted.model.exog_names

        if "Z" not in param_names:
            return np.nan

        zpos = param_names.index("Z")
        X_restricted = np.delete(dataX, zpos, axis=1)

        try:
            model_r = sm.OLS(fitted.model.endog, X_restricted).fit()
            r2_res = float(model_r.rsquared)
        except np.linalg.LinAlgError:
            return np.nan

        denom = 1.0 - r2_res

        if denom <= 0:
            return np.nan

        partial_r2 = (r2_full - r2_res) / denom

        if not math.isfinite(partial_r2):
            return np.nan

        return max(0.0, partial_r2)

    except Exception:
        return np.nan


def _build_first_stage_report(
    fitted: Any,
    design: np.ndarray,
    z_variance: float,
    min_f_threshold: float,
    cluster_groups: Optional[np.ndarray] = None,
) -> FirstStageReport:
    try:
        cond_number = float(np.linalg.cond(design))
    except Exception:
        cond_number = np.inf

    try:
        classical_f, classical_p = classical_first_stage_f_statistic(fitted)
    except Exception:
        classical_f, classical_p = np.nan, np.nan

    try:
        robust_f, robust_p = robust_first_stage_hc3_stat(fitted)
    except Exception:
        robust_f, robust_p = np.nan, np.nan

    cluster_f = np.nan
    cluster_p = np.nan
    n_clusters = 0
    if cluster_groups is not None and len(cluster_groups) > 0:
        try:
            cluster_f, cluster_p = robust_first_stage_cluster_stat(fitted, cluster_groups)
            n_clusters = int(len(np.unique(cluster_groups)))
        except Exception as exc:
            logger.warning("_build_first_stage_report: cluster stat failed: %s", exc)

    try:
        partial_r2 = partial_r2_from_fitted(fitted)
    except Exception:
        partial_r2 = np.nan

    try:
        partial_f_z = partial_f_statistic_for_z(fitted)
    except Exception:
        partial_f_z = np.nan

    # Приоритет: cluster > robust > classical
    f_used = (
        cluster_f
        if np.isfinite(cluster_f)
        else (robust_f if np.isfinite(robust_f) else classical_f)
    )
    weak = (not np.isfinite(f_used)) or (f_used < min_f_threshold)

    return FirstStageReport(
        n=int(fitted.nobs),
        z_variance=float(z_variance),
        classical_f=float(classical_f) if np.isfinite(classical_f) else np.nan,
        classical_f_pvalue=float(classical_p) if np.isfinite(classical_p) else np.nan,
        robust_f=float(robust_f) if np.isfinite(robust_f) else np.nan,
        robust_f_pvalue=float(robust_p) if np.isfinite(robust_p) else np.nan,
        cluster_f=float(cluster_f) if np.isfinite(cluster_f) else np.nan,
        cluster_f_pvalue=float(cluster_p) if np.isfinite(cluster_p) else np.nan,
        partial_r2=float(partial_r2) if np.isfinite(partial_r2) else np.nan,
        partial_f_z=float(partial_f_z) if np.isfinite(partial_f_z) else np.nan,
        condition_number=cond_number,
        weak_instrument=bool(weak),
        min_f_threshold=float(min_f_threshold),
        n_clusters=n_clusters,
    )


def fit_first_stage(
    data: pd.DataFrame,
    opts: CFFitOptions,
) -> FirstStageFit:
    """First stage OLS: PeakLoad ~ Z + X."""
    model_data = data.copy()

    x_cols = _add_design_x_columns(
        model_data=model_data,
        source_data=data,
        extra_x_cols=opts.extra_x_cols,
        brand_encoding=opts.brand_encoding,
        brand_reference_code=opts.brand_reference_code,
    )

    # ─── Interaction: только в Cox, НЕ в первой стадии ──────────────
    # Эконометрическое обоснование: interaction — структурный эффект,
    # а не конфаундер PeakLoad. Включение в первую стадию искажает v_hat.
    if "x_age_hours" in data.columns:
        model_data["x_age_hours"] = data["x_age_hours"].astype(float).to_numpy()
        if "x_age_hours" not in x_cols:
            x_cols.append("x_age_hours")

    # Защита от вырожденных и слишком редких фиктивных переменных.
    x_cols = _filter_cox_covariates(
        df=model_data,
        cols=x_cols,
        required=[],
        event=None,
        var_floor=1e-12,
        min_binary_obs=max(10, int(opts.min_events_per_covariate)),
        min_binary_events=0,
    )

    required = {"PeakLoad", "Z"} | set(x_cols)
    missing = required.difference(set(model_data.columns))

    if missing:
        raise KeyError(f"Missing first-stage columns: {sorted(missing)}")

    # ─── x_age_hours exclusion from first stage (econometric justification) ───
    # x_age_hours is an interaction term (Age × Hours) that affects the structural
    # equation (Cox model) but is NOT a confounder of PeakLoad selection.
    # Including it in the first stage would bias v_hat (the control function),
    # because the IV instrument Z would be regressed on a structural effect
    # rather than only on the endogenous confounder.
    # This creates an asymmetry: x_age_hours is in Cox but not in first_stage.
    # This is intentional and correct per the Control Function literature.
    first_stage_cols = [c for c in x_cols if c != "x_age_hours"]
    cols_for_fit = ["Z"] + first_stage_cols

    if not np.all(np.isfinite(model_data[cols_for_fit + ["PeakLoad"]].to_numpy(dtype=float))):
        raise ValueError("First-stage input contains non-finite values")

    z_values = model_data["Z"].astype(float).to_numpy()
    z_var = float(np.var(z_values, ddof=0))

    if (not math.isfinite(z_var)) or z_var < opts.var_z_threshold:
        raise ValueError(f"Z variance too small: {z_var}")

    design_df = model_data[cols_for_fit]
    design = sm.add_constant(design_df, has_constant="add")
    model = sm.OLS(model_data["PeakLoad"], design)
    fitted = model.fit()
    residuals = np.asarray(fitted.resid, dtype=float)

    # Cluster groups (если указан cluster_col в opts)
    cluster_groups: Optional[np.ndarray] = None
    if opts.cluster_col is not None and opts.cluster_col in model_data.columns:
        # НЕ преобразуем в str — statsmodels принимает числовые метки
        cluster_groups = model_data[opts.cluster_col].to_numpy()

    report = _build_first_stage_report(
        fitted=fitted,
        design=design.to_numpy(dtype=float),
        z_variance=z_var,
        min_f_threshold=opts.min_first_stage_f,
        cluster_groups=cluster_groups,
    )

    return FirstStageFit(
        fitted=fitted,
        residuals=residuals,
        design=design.to_numpy(dtype=float),
        x_cols=x_cols,
        report=report,
    )


# ---------------------------------------------------------------------------
# Cluster-robust first stage
# ---------------------------------------------------------------------------
def fit_first_stage_cluster_robust(
    data: pd.DataFrame,
    cluster_col: str = "cluster_id",
) -> Tuple[Any, Dict[str, Any]]:
    """
    First stage с cluster-robust стандартными ошибками.

    PeakLoad_i = π_Z Z_c + X_i β + ε_i

    где c = cluster_id (Region × Year × Campaign)

    Returns:
        fitted_model, diagnostics
    """
    # Подготовка данных
    y = data["PeakLoad"].values
    X_cols = ["Z", "x_age", "x_hours", "x_climate", "x_soil", "x_power"]

    # Добавить brand dummies
    brand_cols = [c for c in data.columns if c.startswith("brand_")]
    X_cols.extend(brand_cols)

    # Оставляем как DataFrame для сохранения имён столбцов
    X_df = data[X_cols]
    X_with_const = sm.add_constant(X_df, has_constant="add")

    # Cluster IDs
    clusters = data[cluster_col].values

    n_clusters = len(np.unique(clusters))
    if n_clusters < 30:
        raise ValueError(
            f"Кластерная структура содержит только {n_clusters} кластеров. "
            f"Минимум 30 для состоятельных кластерных стандартных ошибок. "
            f"Используйте wild cluster bootstrap или увеличьте число кластеров."
        )
    if n_clusters < 50:
        logger.warning(
            "⚠️ Только %d кластеров (< 50). Кластерные SE могут быть смещены. "
            "Рассмотрите wild cluster bootstrap или CR2/CR3 коррекцию.",
            n_clusters,
        )

    # OLS с cluster-robust covariance
    model = sm.OLS(y, X_with_const)
    fitted = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})

    # Wald test для Z (cluster-robust)
    # H0: π_Z = 0
    z_idx = list(fitted.model.exog_names).index("Z")
    R = np.zeros((1, len(fitted.model.exog_names)))
    R[0, z_idx] = 1.0

    wald_test = fitted.wald_test(R)

    # Правильное извлечение F-статистики из ContrastResults
    # В новых версиях statsmodels Wald test возвращает .statistic (не .fvalue)
    # https://github.com/statsmodels/statsmodels/issues/8937
    try:
        # Новый API: используем .statistic
        f_stat_cluster = float(np.asarray(wald_test.statistic).ravel()[0])
    except (AttributeError, IndexError, TypeError):
        # Fallback на старый API с .fvalue
        try:
            f_stat_cluster = float(np.asarray(wald_test.fvalue).ravel()[0])
        except (AttributeError, IndexError, TypeError):
            # Если ничего не работает, используем t-value в квадрате
            f_stat_cluster = float(np.asarray(wald_test.tvalue).ravel()[0]) ** 2

    p_value_cluster = float(np.asarray(wald_test.pvalue).ravel()[0])

    diagnostics = {
        "f_statistic_cluster_robust": f_stat_cluster,
        "p_value_cluster_robust": p_value_cluster,
        "n_clusters": len(np.unique(clusters)),
        "n_obs": len(data),
        "pi_z": float(fitted.params.iloc[z_idx]),
        "se_pi_z_cluster": float(fitted.bse.iloc[z_idx]),
    }

    return fitted, diagnostics


# ---------------------------------------------------------------------------
# Cox fit helpers
# ---------------------------------------------------------------------------
def _check_baseline_hazard(cph: CoxPHFitter) -> None:
    bl = cph.baseline_cumulative_hazard_

    if bl is None:
        return
    if len(bl) == 0:
        return

    bl_vals = bl.iloc[:, 0].to_numpy(dtype=float)

    if not np.all(np.isfinite(bl_vals)):
        raise ValueError("Non-finite baseline cumulative hazard")
    if np.any(bl_vals < -1e-12):
        raise ValueError("Negative baseline cumulative hazard")


def _fit_cox_model(
    model_data: pd.DataFrame,
    covariate_cols: List[str],
    penalizer: float,
    robust: bool = True,
    check_baseline: bool = True,
    cluster_col: Optional[str] = None,
) -> CoxPHFitter:
    """
    Fit Cox PH model with optional cluster-robust standard errors.
    Includes defensive sanitization and robust->non-robust fallback
    for lifelines ambiguous-array bugs.
    """
    subset = ["time", "event"] + list(covariate_cols)

    use_cluster = False
    if cluster_col is not None:
        if cluster_col in covariate_cols:
            raise ValueError(
                f"cluster_col='{cluster_col}' must NOT be in covariate_cols. "
                f"It is used only for variance estimation."
            )
        if cluster_col not in model_data.columns:
            raise ValueError(
                f"cluster_col='{cluster_col}' explicitly requested "
                f"but not found in model_data columns"
            )
        subset.append(cluster_col)
        use_cluster = True

    df = model_data.loc[:, subset].copy().reset_index(drop=True)

    # Жёсткая нормализация типов.
    df["time"] = pd.to_numeric(df["time"], errors="coerce").astype(float)
    df["event"] = pd.to_numeric(df["event"], errors="coerce").astype(int)

    for col in covariate_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    if use_cluster:
        if df[cluster_col].isna().any():
            raise ValueError(f"cluster_col='{cluster_col}' contains NaNs")
        if not np.issubdtype(df[cluster_col].dtype, np.number):
            codes, _ = pd.factorize(df[cluster_col], sort=False)
            df[cluster_col] = codes.astype(int)

    # Чистка времени и событий.
    time_vals = df["time"].to_numpy(dtype=float)
    event_vals = df["event"].to_numpy(dtype=int)

    if not np.all(np.isfinite(time_vals)):
        raise ValueError("Cox input: time contains non-finite values")

    if not np.all(time_vals > 0):
        raise ValueError("Cox input: time must be strictly positive")

    if not np.all(np.isin(event_vals, [0, 1])):
        raise ValueError("Cox input: event must be binary 0/1")

    if covariate_cols:
        X = df[covariate_cols].to_numpy(dtype=float)
        if not np.all(np.isfinite(X)):
            raise ValueError("Cox input: covariates contain non-finite values")

    # Небольшая защита от нестабильных tie-путей в некоторых версиях lifelines.
    df = df.sort_values(["time"], kind="mergesort").reset_index(drop=True)

    fit_kwargs: Dict[str, Any] = {
        "duration_col": "time",
        "event_col": "event",
        "show_progress": False,
    }

    if use_cluster:
        fit_kwargs["cluster_col"] = cluster_col
        logger.info(
            "Cox fit with cluster-robust SE: cluster_col='%s', n_clusters=%d",
            cluster_col,
            int(df[cluster_col].nunique()),
        )

    def _do_fit(use_robust: bool) -> CoxPHFitter:
        cph_local = CoxPHFitter(penalizer=float(penalizer))

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            cph_local.fit(df, robust=use_robust, **fit_kwargs)

        warn_summary = _summarize_warnings(caught_warnings)
        if warn_summary:
            raise RuntimeError(f"Cox convergence warnings: {warn_summary}")

        return cph_local

    try:
        cph = _do_fit(bool(robust))
    except ValueError as exc:
        if bool(robust) and _is_ambiguous_array_error(exc):
            logger.warning(
                "Cox robust=True fit failed with ambiguous-array ValueError. "
                "Retrying with robust=False. Original error: %s",
                exc,
            )
            cph = _do_fit(False)
        else:
            raise

    ll = _get_cox_log_likelihood(cph)
    if not math.isfinite(ll):
        raise RuntimeError("Cox non-finite log-likelihood")

    params = cph.params_.to_numpy(dtype=float)
    ses = cph.standard_errors_.to_numpy(dtype=float)

    if not np.all(np.isfinite(params)):
        raise RuntimeError("Cox non-finite coefficients")

    if not np.all(np.isfinite(ses)):
        raise RuntimeError("Cox non-finite standard errors")

    if check_baseline:
        _check_baseline_hazard(cph)

    return cph


def _fit_cox_model_cluster(
    model_data: pd.DataFrame,
    covariate_cols: List[str],
    penalizer: float,
    robust: bool = True,
    check_baseline: bool = True,
    cluster_col: Optional[str] = None,
) -> CoxPHFitter:
    """
    Wrapper around _fit_cox_model that forwards cluster_col.

    If P0-6 already added cluster_col support to _fit_cox_model(),
    this function acts as a passthrough. Otherwise it falls back
    to the original call (with a warning if cluster_col is requested).

    Fail-closed: if cluster_col is explicitly requested but missing
    from model_data, raise ValueError.
    """
    if cluster_col is not None and cluster_col not in model_data.columns:
        raise ValueError(
            f"cluster_col='{cluster_col}' explicitly requested but not found in model_data columns"
        )

    # If _fit_cox_model already accepts cluster_col (P0-6 applied):
    try:
        return _fit_cox_model(
            model_data=model_data,
            covariate_cols=covariate_cols,
            penalizer=penalizer,
            robust=robust,
            check_baseline=check_baseline,
            cluster_col=cluster_col,
        )
    except TypeError:
        # Fallback: _fit_cox_model doesn't accept cluster_col yet
        if cluster_col is not None:
            logger.warning(
                "_fit_cox_model does not support cluster_col; fitting without cluster-robust SE"
            )
        return _fit_cox_model(
            model_data=model_data,
            covariate_cols=covariate_cols,
            penalizer=penalizer,
            robust=robust,
            check_baseline=check_baseline,
        )


def _min_events_required_from_count(
    n_covariates: int,
    opts: CFFitOptions,
) -> int:
    return max(
        int(opts.min_cox_events),
        int(opts.min_events_per_covariate) * max(1, int(n_covariates)),
    )


def _filter_cox_covariates(
    df: pd.DataFrame,
    cols: List[str],
    required: Optional[List[str]] = None,
    event: Optional[np.ndarray] = None,
    var_floor: float = 1e-12,
    min_binary_obs: int = 10,
    min_binary_events: int = 0,
) -> List[str]:
    """
    Фильтрует ковариаты до подачи в Cox / first stage.

    Безопасно обрабатывает:
    - event=None;
    - пустые колонки;
    - неконечные значения;
    - вырожденные колонки;
    - редкие бинарные фиктивные переменные;
    - separation / мало событий в бинарных группах, если задан event.
    """
    required_set = set(required or [])
    kept: List[str] = []
    seen: set = set()

    if event is None:
        event_arr = None
    else:
        event_arr = np.asarray(event, dtype=bool).reshape(-1)
        if event_arr.size != len(df):
            raise ValueError("event vector length mismatch in covariate filter")

    for col in cols:
        if col in seen:
            continue
        seen.add(col)

        if col not in df.columns:
            if col in required_set:
                raise KeyError(f"Required covariate '{col}' not found")
            logger.warning("Covariate '%s' not found in df; dropped", col)
            continue

        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        if not np.all(np.isfinite(vals)):
            if col in required_set:
                raise ValueError(f"Required covariate '{col}' contains non-finite values")
            logger.warning("Covariate '%s' contains non-finite values; dropped", col)
            continue

        if vals.size == 0:
            if col in required_set:
                raise ValueError(f"Required covariate '{col}' is empty")
            logger.warning("Covariate '%s' is empty; dropped", col)
            continue

        std_val = float(np.std(vals, ddof=0))
        if (not math.isfinite(std_val)) or std_val <= var_floor:
            if col in required_set:
                raise ValueError(
                    f"Required covariate '{col}' is degenerate (std <= {var_floor})"
                )
            logger.warning(
                "Covariate '%s' is degenerate (std=%.3g); dropped",
                col,
                std_val,
            )
            continue

        uniq = np.unique(vals)

        # Почти бинарная колонка: проверяем редкость и события по группам.
        if uniq.size <= 2:
            if uniq.size == 2:
                mask_one = vals == uniq[-1]
            else:
                mask_one = np.zeros_like(vals, dtype=bool)

            n_one = int(mask_one.sum())
            n_zero = int(vals.size - n_one)

            if n_one < min_binary_obs or n_zero < min_binary_obs:
                if col in required_set:
                    raise ValueError(f"Required binary covariate '{col}' is too rare")
                logger.warning(
                    "Binary covariate '%s' is too rare: n_one=%d, n_zero=%d; dropped",
                    col,
                    n_one,
                    n_zero,
                )
                continue

            if event_arr is not None and min_binary_events > 0:
                e_one = int(np.logical_and(event_arr, mask_one).sum())
                e_zero = int(np.logical_and(event_arr, ~mask_one).sum())

                bad = (
                    e_one < min_binary_events
                    or e_zero < min_binary_events
                    or e_one == 0
                    or e_zero == 0
                )

                if bad:
                    if col in required_set:
                        raise ValueError(
                            f"Required binary covariate '{col}' causes separation "
                            f"or has too few events"
                        )
                    logger.warning(
                        "Binary covariate '%s' causes separation or has too few events: "
                        "e_one=%d/%d, e_zero=%d/%d; dropped",
                        col,
                        e_one,
                        n_one,
                        e_zero,
                        n_zero,
                    )
                    continue

        kept.append(col)

    return kept


def compute_vif(design_matrix: np.ndarray, names: Optional[List[str]] = None) -> Dict[str, float]:
    X = np.asarray(design_matrix, dtype=float)

    if X.shape[1] <= 1:
        return {}

    if np.allclose(X[:, 0], X[0, 0]):
        Xv = X[:, 1:]
        offset = 1
    else:
        Xv = X
        offset = 0

    vifs: Dict[str, float] = {}

    for j in range(Xv.shape[1]):
        try:
            y = Xv[:, j]
            others = np.delete(Xv, j, axis=1)
            r2 = sm.OLS(y, sm.add_constant(others)).fit().rsquared

            if not math.isfinite(r2):
                vif = np.inf
            else:
                r2 = min(max(r2, 0.0), 1.0 - 1e-12)
                vif = 1.0 / (1.0 - r2)

        except Exception:
            vif = np.inf

        if names and len(names) > j + offset:
            name = names[j + offset]
        else:
            name = f"x{j}"

        vifs[name] = float(vif)

    return vifs


# ---------------------------------------------------------------------------
# Naive Cox
# ---------------------------------------------------------------------------
def fit_naive_cox(
    data: pd.DataFrame,
    opts: CFFitOptions,
) -> NaiveCoxResult:
    """Naive Cox model: time ~ PeakLoad + X."""
    if opts.cox_se_threshold <= 0.0:
        raise ValueError("cox_se_threshold must be positive")

    required_cols = {"time", "event", "PeakLoad"}
    missing = required_cols - set(data.columns)

    if missing:
        raise KeyError(f"fit_naive_cox missing required columns: {sorted(missing)}")

    model_data = data[["time", "event", "PeakLoad"]].copy()

    # Сохраняем cluster_col для кластерных стандартных ошибок.
    if opts.cluster_col is not None and opts.cluster_col in data.columns:
        model_data[opts.cluster_col] = data[opts.cluster_col]

    x_cols = _add_design_x_columns(
        model_data=model_data,
        source_data=data,
        extra_x_cols=opts.extra_x_cols,
        brand_encoding=opts.brand_encoding,
        brand_reference_code=opts.brand_reference_code,
    )

    event_arr = model_data["event"].astype(bool).to_numpy()

    x_cols = _filter_cox_covariates(
        df=model_data,
        cols=x_cols,
        required=[],
        event=event_arr,
        var_floor=1e-12,
        min_binary_obs=max(10, int(opts.min_events_per_covariate)),
        min_binary_events=max(1, int(opts.min_events_per_covariate)),
    )

    covariate_cols = ["PeakLoad"] + x_cols

    _validate_survival_frame(model_data, covariate_cols)

    n_events = int(np.sum(model_data["event"].astype(int).to_numpy()))
    n = len(model_data)
    min_events = _min_events_required_from_count(len(covariate_cols), opts)

    if n_events < min_events:
        raise RuntimeError(f"Naive Cox: too few events {n_events} < required {min_events}")

    attempted: List[float] = []
    last_exc: Optional[BaseException] = None

    for pen in _ALL_PENALIZERS:
        attempted.append(float(pen))

        try:
            cph = _fit_cox_model(
                model_data=model_data,
                covariate_cols=covariate_cols,
                penalizer=pen,
                robust=True,
                check_baseline=True,
                cluster_col=opts.cluster_col,
            )

            if "PeakLoad" not in cph.params_.index:
                raise RuntimeError("PeakLoad coefficient not found")

            gamma_hat = _safe_series_scalar(
                cph.params_,
                "PeakLoad",
                "Naive Cox params",
            )
            standard_error = _safe_series_scalar(
                cph.standard_errors_,
                "PeakLoad",
                "Naive Cox standard errors",
            )

            ses_all = cph.standard_errors_.to_numpy(dtype=float)
            max_se = float(np.max(np.abs(ses_all)))

            if max_se > opts.cox_se_threshold:
                raise RuntimeError(f"Too-large SEs (max_se={max_se})")

            warnings_list: List[str] = []

            if pen > 0.0:
                warnings_list.append(f"Naive Cox estimate is regularized (penalizer={pen}).")

            convergence_info = ConvergenceInfo(
                penalizer=float(pen),
                warning=None,
                attempted_penalizers=attempted.copy(),
            )

            return NaiveCoxResult(
                gamma_hat=gamma_hat,
                naive_se=standard_error,
                max_se=max_se,
                penalizer_used=float(pen),
                is_penalized=bool(pen > 0.0),
                cph=cph,
                convergence_info=convergence_info,
                warnings=warnings_list,
                n=n,
                n_events=n_events,
            )

        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError(
        f"Naive Cox fit failed; last error: {format_exception(last_exc, opts.save_tracebacks)}"
    )


# ---------------------------------------------------------------------------
# Reduced Form Cox (v3.1)
# ---------------------------------------------------------------------------
def fit_reduced_form_cox(
    data: pd.DataFrame,
    opts: CFFitOptions,
) -> CFModelResult:
    """
    Reduced Form Cox: h(t|x) = h0(t)·exp(γ·D + π·Z + β·X).

    Инструмент Z входит как обычная предиктивная ковариата.
    Нет первой стадии, нет контрольной функции, нет бутстрапа
    для генерируемых регрессоров (проблема #4 исчезает).

    Модель ПРЕДИКТИВНАЯ. Каузальная интерпретация невозможна
    из-за нарушения exclusion restriction.
    """
    if opts.cox_se_threshold <= 0.0:
        raise ValueError("cox_se_threshold must be positive")

    required_cols = {"time", "event", "PeakLoad", "Z"}
    missing = required_cols - set(data.columns)
    if missing:
        raise KeyError(f"fit_reduced_form_cox missing required columns: {sorted(missing)}")

    model_data = data[["time", "event", "PeakLoad", "Z"]].copy()

    x_cols = _add_design_x_columns(
        model_data=model_data,
        source_data=data,
        extra_x_cols=opts.extra_x_cols,
        brand_encoding=opts.brand_encoding,
        brand_reference_code=opts.brand_reference_code,
    )

    # PeakLoad и Z обязаны быть невырожденными
    for col in ("PeakLoad", "Z"):
        std_val = float(np.std(model_data[col].astype(float).to_numpy(), ddof=0))
        if (not math.isfinite(std_val)) or std_val <= 1e-12:
            raise ValueError(f"fit_reduced_form_cox: {col} вырождена (std <= 1e-12)")

    # Кластерная структура
    if opts.cluster_col is not None:
        if opts.cluster_col in x_cols:
            raise ValueError(
                f"cluster_col='{opts.cluster_col}' must NOT be in x_cols"
            )
        if opts.cluster_col not in data.columns:
            raise ValueError(
                f"cluster_col='{opts.cluster_col}' explicitly requested "
                f"but not found in data columns"
            )
        model_data[opts.cluster_col] = data[opts.cluster_col].to_numpy()

    event_arr = model_data["event"].astype(bool).to_numpy()
    x_cols = _filter_cox_covariates(
        df=model_data,
        cols=x_cols,
        required=[],
        event=event_arr,
        var_floor=1e-12,
        min_binary_obs=max(10, int(opts.min_events_per_covariate)),
        min_binary_events=max(1, int(opts.min_events_per_covariate)),
    )

    covariate_cols = ["PeakLoad", "Z"] + x_cols
    covariate_cols = list(dict.fromkeys(covariate_cols))
    _validate_survival_frame(model_data, covariate_cols)

    n_events = int(np.sum(model_data["event"].astype(int).to_numpy()))
    n = len(model_data)
    min_events = _min_events_required_from_count(len(covariate_cols), opts)
    if n_events < min_events:
        raise RuntimeError(f"Reduced Form Cox: too few events {n_events} < required {min_events}")

    attempted_penalizers: List[float] = []
    last_exc: Optional[BaseException] = None
    for pen in _ALL_PENALIZERS:
        attempted_penalizers.append(float(pen))
        try:
            cph = _fit_cox_model(
                model_data=model_data,
                covariate_cols=covariate_cols,
                penalizer=pen,
                robust=True,
                check_baseline=True,
                cluster_col=opts.cluster_col,
            )
            params_index = list(cph.params_.index)
            if "PeakLoad" not in params_index:
                raise RuntimeError("Reduced Form Cox: no PeakLoad coefficient")
            if "Z" not in params_index:
                raise RuntimeError("Reduced Form Cox: no Z coefficient")

            gamma_hat = _safe_series_scalar(cph.params_, "PeakLoad", "RF Cox params")
            naive_model_se = _safe_series_scalar(
                cph.standard_errors_, "PeakLoad", "RF Cox standard errors"
            )
            ses_all = cph.standard_errors_.to_numpy(dtype=float)
            max_se = float(np.max(np.abs(ses_all)))
            if max_se > opts.cox_se_threshold:
                raise RuntimeError(f"Reduced Form Cox: too-large max_se={max_se}")

            convergence_info = ConvergenceInfo(
                penalizer=float(pen),
                warning=None,
                attempted_penalizers=attempted_penalizers.copy(),
            )
            warnings_list: List[str] = []
            if pen > 0.0:
                warnings_list.append(
                    f"Reduced Form Cox estimate is regularized (penalizer={pen})."
                )
            warnings_list.append(
                "Reduced-form model: PREDICTIVE only, no causal correction. "
                "Z is included directly as a covariate; exclusion restriction "
                "violation makes causal interpretation invalid."
            )

            basis_meta: Dict[str, Any] = {
                "requested_v_hat_basis": "none",
                "v_hat_basis": "none",
                "v_hat_cols": [],
                "residuals_mean": 0.0,
                "residuals_std": 1.0,
                "linear_standardized": True,
                "cf_standardization_convention": "reduced_form",
                "residual_policy_production": "plug-in",
                "cox_peakload_convention": "observed_peakload",
                "model_form": "reduced_form",
            }

            return CFModelResult(
                gamma_hat=gamma_hat,
                naive_model_se=naive_model_se,
                bootstrap_se=None,
                se_type="naive",
                cf_coef=float("nan"),
                cf_coef_signed=None,
                cph=cph,
                max_se=max_se,
                penalizer=float(pen),
                is_penalized=bool(pen > 0.0),
                convergence_info=convergence_info,
                warnings=warnings_list,
                n=n,
                n_events=n_events,
                v_hat_basis="none",
                first_stage_report=None,
                partial_out_all_betas={},
                training_x_means={},
                training_pl_hat_mean=0.0,
                training_residuals_std=1.0,
                training_residuals_mean=0.0,
                cf_basis_metadata=basis_meta,
            )
        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError(
        f"Reduced Form Cox fit failed. Last error: "
        f"{format_exception(last_exc, opts.save_tracebacks)}"
    )


# ---------------------------------------------------------------------------
# CF Cox
# ---------------------------------------------------------------------------
def fit_cf_cox(
    data: pd.DataFrame,
    first_stage: FirstStageFit,
    opts: CFFitOptions,
) -> CFModelResult:
    """
    Fit control-function Cox model:
    h(t|x) = h0(t) * exp(gamma * D + lambda * CF + beta * X)
    Returns CFModelResult. The field naive_model_se is NOT a valid generated-regressor
    standard error unless bootstrap_se is supplied separately.
    """
    if opts.cox_se_threshold <= 0.0:
        raise ValueError("cox_se_threshold must be positive")

    required_cols = {"time", "event", "PeakLoad"}
    missing = required_cols - set(data.columns)

    if missing:
        raise KeyError(f"fit_cf_cox missing required columns: {sorted(missing)}")

    # ─── Stock & Yogo (2005) partial F check for Z ───────────────────
    partial_f_z = first_stage.report.partial_f_z
    if np.isfinite(partial_f_z):
        print(f"Частичный F для Z (после контроля X): {partial_f_z:.2f}")
        if partial_f_z < 10.0:
            logger.warning(
                "⚠️ Частичный F для Z = %.2f < 10 (Stock & Yogo, 2005). "
                "Инструмент слабый после контроля X. Каузальная интерпретация γ невозможна.",
                partial_f_z,
            )

    if first_stage.report.weak_instrument and opts.fail_on_weak_instrument:
        raise RuntimeError(
            "Weak instrument detected in first stage. "
            f"Robust F={first_stage.report.robust_f}, "
            f"threshold={first_stage.report.min_f_threshold}."
        )

    residuals = np.asarray(first_stage.residuals, dtype=float).reshape(-1)

    if len(residuals) != len(data):
        raise ValueError("Residuals length mismatch.")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("Residuals contain NaN or Inf.")

    model_data = data[["time", "event", "PeakLoad"]].copy()
    original_peakload = data["PeakLoad"].astype(float).to_numpy()

    x_cols = _add_design_x_columns(
        model_data=model_data,
        source_data=data,
        extra_x_cols=opts.extra_x_cols,
        brand_encoding=opts.brand_encoding,
        brand_reference_code=opts.brand_reference_code,
    )

    # PeakLoad обязан быть невырожденным.
    peakload_vals = model_data["PeakLoad"].astype(float).to_numpy()
    peakload_std = float(np.std(peakload_vals, ddof=0))
    if (not math.isfinite(peakload_std)) or peakload_std <= 1e-12:
        raise ValueError("fit_cf_cox: PeakLoad is degenerate (std <= 1e-12)")

    event_arr = model_data["event"].astype(bool).to_numpy()

    # Фильтруем X-ковариаты до построения Cox-модели.
    x_cols = _filter_cox_covariates(
        df=model_data,
        cols=x_cols,
        required=[],
        event=event_arr,
        var_floor=1e-12,
        min_binary_obs=max(10, int(opts.min_events_per_covariate)),
        min_binary_events=max(1, int(opts.min_events_per_covariate)),
    )

    # ─── P0-6: копируем cluster_id в model_data для cluster-robust Cox ──
    # Fail-closed: если cluster_col явно запрошен, но отсутствует в data,
    # это ошибка pipeline, а не штатная ситуация.
    if opts.cluster_col is not None:
        if opts.cluster_col in x_cols:
            raise ValueError(
                f"cluster_col='{opts.cluster_col}' must NOT be in x_cols. "
                f"It is used only for variance estimation, not as a regressor."
            )
        if opts.cluster_col not in data.columns:
            raise ValueError(
                f"cluster_col='{opts.cluster_col}' explicitly requested "
                f"but not found in data columns"
            )
        model_data[opts.cluster_col] = data[opts.cluster_col].to_numpy()

    if opts.center_peakload is not None:
        cp = float(opts.center_peakload)

        if not math.isfinite(cp):
            raise ValueError("center_peakload must be finite")

        model_data["PeakLoad"] = model_data["PeakLoad"] - cp

    covariate_cols_before_cf = ["PeakLoad"] + x_cols
    _validate_survival_frame(model_data, covariate_cols_before_cf)

    # ─── ФИЛЬТРАЦИЯ КОЛОНОК С НУЛЕВОЙ ДИСПЕРСИЕЙ ───────────────────────
    # Это предотвращает баги lifelines и ошибки неоднозначности массивов
    valid_x_cols = []
    for c in x_cols:
        if model_data[c].std() > 1e-12:
            valid_x_cols.append(c)
        else:
            logger.warning(f"Column {c} has zero variance, removing from model.")
    x_cols = valid_x_cols

    if model_data["PeakLoad"].std() <= 1e-12:
        raise RuntimeError("CF Cox: PeakLoad has zero variance")
    # ───────────────────────────────────────────────────────────────────

    cf_cols = _build_cf_columns(
        residuals=residuals,
        v_hat_basis=opts.v_hat_basis,
        v_hat_basis_params=opts.v_hat_basis_params,
        base_df=model_data,
    )
    model_data = cf_cols.df_with_cf
    v_hat_cols = cf_cols.column_names

    event_arr = model_data["event"].astype(bool).to_numpy()

    # CF-колонки тоже защищаем от вырожденных случаев.
    v_hat_cols = _filter_cox_covariates(
        df=model_data,
        cols=v_hat_cols,
        required=[],
        event=event_arr,
        var_floor=1e-12,
        min_binary_obs=max(10, int(opts.min_events_per_covariate)),
        min_binary_events=max(1, int(opts.min_events_per_covariate)),
    )

    if not v_hat_cols:
        raise ValueError("fit_cf_cox: no valid CF columns created after filtering")

    covariate_cols = ["PeakLoad"] + x_cols + v_hat_cols

    # Убираем возможные дубли имён колонок, сохраняя порядок.
    covariate_cols = list(dict.fromkeys(covariate_cols))

    n_events = int(np.sum(model_data["event"].astype(int).to_numpy()))
    n = len(model_data)

    min_events = _min_events_required_from_count(len(covariate_cols), opts)

    if n_events < min_events:
        raise RuntimeError(f"CF Cox: too few events {n_events} < required {min_events}")

    # Исключаем cluster_col из проверки: он может содержать строковые метки
    _check_cols = [c for c in model_data.columns if c != opts.cluster_col]

    if not np.all(np.isfinite(model_data[_check_cols].to_numpy(dtype=float))):
        raise ValueError("fit_cf_cox: non-finite data after CF construction")

    # Partial-out diagnostic: PL_hat ~ X (без инструмента Z).
    # Конвенция PL_HAT_EXOG_CONVENTION определяет, включать ли Z:
    # "exclude_instrument" — Z исключается из экзогенной части (по умолчанию).
    forbidden_partial_out = {"const", "intercept"}
    if PL_HAT_EXOG_CONVENTION == "exclude_instrument":
        forbidden_partial_out.add("Z")

    pl_hat = np.asarray(first_stage.fitted.fittedvalues, dtype=float).reshape(-1)
    if len(pl_hat) != len(data):
        raise ValueError("fit_cf_cox: first-stage fitted values length mismatch")
    if not np.all(np.isfinite(pl_hat)):
        raise ValueError("fit_cf_cox: non-finite PL_hat")
    training_pl_hat_mean = float(np.mean(pl_hat))
    partial_out_all_betas: Dict[str, float] = {}
    training_x_means: Dict[str, float] = {}
    if x_cols:
        try:
            diag_df = model_data[x_cols].copy()
            # Исключаем Z согласно конвенции PL_HAT_EXOG_CONVENTION
            diag_df = diag_df[[c for c in diag_df.columns if c not in forbidden_partial_out]]
            X_design = sm.add_constant(diag_df, has_constant="add")
            ols = sm.OLS(pl_hat, X_design).fit()
            partial_out_all_betas = {
                str(col): float(ols.params[col])
                for col in diag_df.columns
                if col in ols.params.index
            }
            training_x_means = {str(col): float(diag_df[col].mean()) for col in diag_df.columns}
        except Exception:
            partial_out_all_betas = {}
            training_x_means = {}

    # Cox fit with penalizer fallback
    attempted_penalizers: List[float] = []
    last_exc: Optional[BaseException] = None
    for pen in _ALL_PENALIZERS:
        attempted_penalizers.append(float(pen))

        try:
            cph = _fit_cox_model(
                model_data=model_data,
                covariate_cols=covariate_cols,
                penalizer=pen,
                robust=True,
                check_baseline=True,
                cluster_col=opts.cluster_col,
            )

            # ─── БЕЗОПАСНОЕ ИЗВЛЕЧЕНИЕ КОЭФФИЦИЕНТОВ ───────────────────────
            # Используем фильтрацию по индексу вместо .loc[], чтобы избежать
            # ValueError при дублирующихся индексах или возврате массивов
            def _safe_extract(series, key):
                vals = series[series.index == key]
                if len(vals) == 0:
                    raise RuntimeError(f"Key '{key}' not found in model params")
                return float(vals.iloc[0])

            params_index = list(cph.params_.index)

            if "PeakLoad" not in params_index:
                raise RuntimeError("CF Cox: no PeakLoad coefficient")

            missing_vc = None

            for vc in v_hat_cols:
                if vc not in params_index:
                    missing_vc = vc
                    break

            if missing_vc is not None:
                raise RuntimeError(f"CF Cox: no '{missing_vc}' coefficient")

            gamma_hat = _safe_extract(cph.params_, "PeakLoad")
            naive_model_se = _safe_extract(cph.standard_errors_, "PeakLoad")

            if len(v_hat_cols) > 1:
                v_hat_coefs = [_safe_extract(cph.params_, vc) for vc in v_hat_cols]
                cf_coef = float(np.sqrt(sum(c * c for c in v_hat_coefs)))
                cf_coef_signed = float(np.mean(v_hat_coefs))
            else:
                cf_coef = _safe_extract(cph.params_, v_hat_cols[0])
                cf_coef_signed = cf_coef
            # ───────────────────────────────────────────────────────────────

            ses_all = cph.standard_errors_.to_numpy(dtype=float)
            max_se = float(np.max(np.abs(ses_all)))

            # Debug: check types
            import sys

            if not all(np.isfinite(x) for x in [gamma_hat, naive_model_se, cf_coef, max_se]):
                raise RuntimeError("CF Cox: non-finite estimates")

            if max_se > opts.cox_se_threshold:
                raise RuntimeError(f"CF Cox: too-large max_se={max_se}")

            convergence_info = ConvergenceInfo(
                penalizer=float(pen),
                warning=None,
                attempted_penalizers=attempted_penalizers.copy(),
            )

            warnings_list: List[str] = []

            if pen > 0.0:
                warnings_list.append(
                    f"CF Cox estimate is regularized (penalizer={pen}). "
                    "Treat structural interpretation with caution."
                )

            if first_stage.report.weak_instrument:
                warnings_list.append("Weak instrument detected, but fail_on_weak_instrument=False.")

            # VIF diagnostics
            vif_peakload = None
            vif_vhat = None
            vif_max = None

            try:
                vif_cols = ["PeakLoad"] + v_hat_cols + x_cols

                if len(vif_cols) <= 15 and len(model_data) >= 50:
                    vif_data = model_data[vif_cols].dropna()

                    if len(vif_data) >= 50:
                        vifs = compute_vif(
                            vif_data.to_numpy(dtype=float),
                            names=vif_cols,
                        )
                        vif_peakload = vifs.get("PeakLoad")

                        vhat_values = [v for k, v in vifs.items() if k in v_hat_cols]

                        if vhat_values:
                            vif_vhat = float(max(vhat_values))

                        if vifs:
                            vif_max = float(max(vifs.values()))

                        if (vif_peakload is not None and vif_peakload > 10.0) or (
                            vif_vhat is not None and vif_vhat > 10.0
                        ):
                            warnings_list.append(
                                f"WARNING: Multicollinearity. VIF(PeakLoad)={vif_peakload}, "
                                f"VIF(v_hat)={vif_vhat}, VIF(max)={vif_max}."
                            )

            except Exception:
                pass

            basis_meta: Dict[str, Any] = {
                "requested_v_hat_basis": opts.v_hat_basis,
                "v_hat_basis": cf_cols.actual_basis,
                "v_hat_cols": v_hat_cols,
                "residuals_mean": cf_cols.residuals_mean,
                "residuals_std": cf_cols.residuals_std,
                "v_hat_col_std_params": cf_cols.basis_std_params,
                "linear_standardized": True,
                "cf_standardization_convention": "residual_mean_std",
                "residual_policy_production": "plug-in",
                "cox_peakload_convention": "observed_peakload",
                "center_peakload": opts.center_peakload,
            }
            basis_meta.update(cf_cols.basis_kwargs)

            # ─── Патч 4: propagation interaction_params к CFModelResult ─────
            interaction_params = data.attrs.get("interaction_params", None)
            if interaction_params is not None:
                basis_meta["interaction_params"] = interaction_params

            # ─── Bootstrap SE for generated regressars (Murphy-Topel correction) ───
            # Generated regressor: v_hat (first-stage residuals) introduces
            # additional uncertainty not captured by naive SE from CoxPHFitter.
            # Bootstrap accounts for this uncertainty.
            bootstrap_se_value: Optional[float] = None
            if opts.n_bootstrap > 0:
                try:
                    logger.info(
                        "Running bootstrap SE for CF Cox (n_bootstrap=%d)...", opts.n_bootstrap
                    )
                    bootstrap_result = bootstrap_cf_standard_error(
                        data=data,
                        n_bootstrap=opts.n_bootstrap,
                        seed_sequence=np.random.SeedSequence(int(np.random.randint(0, 2**31))),
                        config=SimulationConfig(
                            sims_per_scenario=1,
                            n_samples=len(data),
                            contamination=False,
                            n_jobs=1,
                            seed=int(np.random.randint(0, 2**31)),
                            n_bootstrap=opts.n_bootstrap,
                            bootstrap_jobs=_bootstrap_jobs,
                            baseline_hazard=DEFAULT_BASELINE_HAZARD,
                            censoring_scale=1e12,
                            dgp=DGPParameters(
                                gamma=0.5,
                                rho=0.7,
                                delta=0.7,
                                intercept=0.5,
                                structural_intercept=0.5,
                                first_stage_z_coef=0.5,
                                clip_lp=None,
                                corr_zu=0.0,
                                baseline_family="weibull",
                                baseline_shape=1.88,
                                brand_encoding="dummies",
                                brand_reference_code=0,
                                beta_age_hours=0.15,
                                use_real_covariates=False,
                                beta_eqi=0.0,
                                n_enterprises=500,
                                use_enterprise_quality=False,
                            ),
                        ),
                        fit_opts=opts,
                    )
                    bootstrap_se_value = bootstrap_result.bootstrap_se
                    if bootstrap_se_value is not None:
                        logger.info(
                            "Bootstrap SE=%.6e, naive SE=%.6e, ratio=%.2f",
                            bootstrap_se_value,
                            naive_model_se,
                            bootstrap_se_value / naive_model_se if naive_model_se > 0 else float("inf"),
                        )
                        # Add note about generated regressors
                        warnings_list.append(
                            f"Bootstrap SE computed (n={opts.n_bootstrap}): "
                            f"SE={bootstrap_se_value:.6e} vs naive SE={naive_model_se:.6e} "
                            f"(ratio={bootstrap_se_value / naive_model_se:.2f} if naive > 0). "
                            "Naive SE underestimates uncertainty for generated regressors."
                        )
                    else:
                        logger.warning(
                            "Bootstrap SE computation failed (success rate %.1f%%), "
                            "falling back to naive SE.",
                            bootstrap_result.success_rate * 100,
                        )
                except Exception as bootstrap_exc:
                    logger.warning(
                        "Bootstrap SE computation raised exception: %s. "
                        "Falling back to naive SE.",
                        bootstrap_exc,
                    )

            se_type = "bootstrap" if bootstrap_se_value is not None else "naive"

            return CFModelResult(
                gamma_hat=gamma_hat,
                naive_model_se=naive_model_se,
                bootstrap_se=bootstrap_se_value,
                se_type=se_type,
                cf_coef=cf_coef,
                cf_coef_signed=cf_coef_signed,
                cph=cph,
                max_se=max_se,
                penalizer=float(pen),
                is_penalized=bool(pen > 0.0),
                convergence_info=convergence_info,
                warnings=warnings_list,
                n=n,
                n_events=n_events,
                v_hat_basis=cf_cols.actual_basis,
                first_stage_report=first_stage.report,
                partial_out_all_betas=partial_out_all_betas,
                training_x_means=training_x_means,
                training_pl_hat_mean=training_pl_hat_mean,
                training_residuals_std=cf_cols.residuals_std,
                training_residuals_mean=cf_cols.residuals_mean,
                vif_peakload=vif_peakload,
                vif_vhat=vif_vhat,
                vif_max=vif_max,
                cf_basis_metadata=basis_meta,
            )

        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError(
        f"CF Cox fit failed. Last error: {format_exception(last_exc, opts.save_tracebacks)}"
    )


# ---------------------------------------------------------------------------
# Endogeneity LR test
# ---------------------------------------------------------------------------
def endogeneity_lr_test(
    data: pd.DataFrame,
    first_stage: FirstStageFit,
    opts: CFFitOptions,
) -> EndogeneityTestResult:
    """Endogeneity LR test: H0: CF coefficients = 0."""
    try:
        required = {"time", "event", "PeakLoad"}

        if not required.issubset(data.columns):
            return EndogeneityTestResult(
                lr_stat=np.nan,
                lr_pvalue=np.nan,
                endogenous=None,
                penalized=False,
                df=0,
                note="Missing required columns",
            )

        residuals = np.asarray(first_stage.residuals, dtype=float).reshape(-1)

        if len(residuals) != len(data):
            return EndogeneityTestResult(
                lr_stat=np.nan,
                lr_pvalue=np.nan,
                endogenous=None,
                penalized=False,
                df=0,
                note="Residuals length mismatch",
            )

        if not np.all(np.isfinite(residuals)):
            return EndogeneityTestResult(
                lr_stat=np.nan,
                lr_pvalue=np.nan,
                endogenous=None,
                penalized=False,
                df=0,
                note="Non-finite residuals",
            )

        if int(np.sum(data["event"].astype(int))) < 1:
            return EndogeneityTestResult(
                lr_stat=np.nan,
                lr_pvalue=np.nan,
                endogenous=None,
                penalized=False,
                df=0,
                note="Zero events",
            )

        model_data = data[["time", "event", "PeakLoad"]].copy()
        x_cols = _add_design_x_columns(
            model_data=model_data,
            source_data=data,
            extra_x_cols=opts.extra_x_cols,
            brand_encoding=opts.brand_encoding,
            brand_reference_code=opts.brand_reference_code,
        )

        # ─── P0-6: копируем cluster_id для cluster-robust Cox ────────────
        if opts.cluster_col is not None:
            if opts.cluster_col in x_cols:
                raise ValueError(f"cluster_col='{opts.cluster_col}' must NOT be in x_cols")
            if opts.cluster_col not in data.columns:
                raise ValueError(
                    f"cluster_col='{opts.cluster_col}' explicitly requested "
                    f"but not found in data columns"
                )
            model_data[opts.cluster_col] = data[opts.cluster_col].to_numpy()

        covariate_cols = ["PeakLoad"] + x_cols

        _validate_survival_frame(model_data, covariate_cols)

        cf_result = _build_cf_columns(
            residuals=residuals,
            v_hat_basis=opts.v_hat_basis,
            v_hat_basis_params=opts.v_hat_basis_params,
            base_df=model_data,
        )
        model_data = cf_result.df_with_cf
        v_hat_cols = cf_result.column_names

        if not v_hat_cols:
            return EndogeneityTestResult(
                lr_stat=np.nan,
                lr_pvalue=np.nan,
                endogenous=None,
                penalized=False,
                df=0,
                note="No CF columns",
            )

        restricted_cols = covariate_cols
        full_cols = covariate_cols + v_hat_cols

        if not np.all(np.isfinite(model_data.to_numpy(dtype=float))):
            return EndogeneityTestResult(
                lr_stat=np.nan,
                lr_pvalue=np.nan,
                endogenous=None,
                penalized=False,
                df=0,
                note="Non-finite model data",
            )

        cph_rest = None
        cph_full = None
        used_pen = None

        for pen in _ALL_PENALIZERS:
            try:
                cph_r = _fit_cox_model(
                    model_data=model_data,
                    covariate_cols=restricted_cols,
                    penalizer=pen,
                    robust=False,
                    check_baseline=True,
                    cluster_col=None,
                )
                cph_f = _fit_cox_model(
                    model_data=model_data,
                    covariate_cols=full_cols,
                    penalizer=pen,
                    robust=False,
                    check_baseline=True,
                    cluster_col=None,
                )

                cph_rest = cph_r
                cph_full = cph_f
                used_pen = float(pen)
                break

            except Exception:
                continue

        if cph_rest is None or cph_full is None or used_pen is None:
            return EndogeneityTestResult(
                lr_stat=np.nan,
                lr_pvalue=np.nan,
                endogenous=None,
                penalized=False,
                df=0,
                note="Both restricted and full Cox models failed",
            )

        ll_rest = _get_cox_log_likelihood(cph_rest)
        ll_full = _get_cox_log_likelihood(cph_full)

        if not math.isfinite(ll_rest) or not math.isfinite(ll_full):
            return EndogeneityTestResult(
                lr_stat=np.nan,
                lr_pvalue=np.nan,
                endogenous=None,
                penalized=used_pen > 0.0,
                df=0,
                note="Non-finite log-likelihood",
            )

        lr_stat = 2.0 * (ll_full - ll_rest)

        if lr_stat < 0.0:
            lr_stat = 0.0

        actual_cf_cols = [c for c in v_hat_cols if c in cph_full.params_.index]
        df_cf = len(actual_cf_cols)
        penalized = used_pen > 0.0

        stats_mod = _scipy_stats if HAS_SCIPY_STATS else None

        if penalized or df_cf <= 0 or stats_mod is None:
            lr_pvalue = np.nan
            endogenous = None
            note = (
                "LR p-value is not reported because the model is penalized, "
                "CF degrees of freedom are zero, or scipy is unavailable."
            )
        else:
            lr_pvalue = float(stats_mod.chi2.sf(lr_stat, df_cf))
            endogenous = bool(lr_pvalue < 0.05)
            note = None

        return EndogeneityTestResult(
            lr_stat=float(lr_stat),
            lr_pvalue=lr_pvalue,
            endogenous=endogenous,
            penalized=penalized,
            df=df_cf,
            actual_cf_cols=actual_cf_cols,
            note=note,
        )

    except Exception as exc:
        return EndogeneityTestResult(
            lr_stat=np.nan,
            lr_pvalue=np.nan,
            endogenous=None,
            penalized=False,
            df=0,
            note=format_exception(exc, save_traceback=False),
        )


def interaction_lr_test(
    data: pd.DataFrame,
    first_stage: FirstStageFit,
    opts: CFFitOptions,
    interaction_col: str = "x_age_hours",
) -> Dict[str, Any]:
    """
    P0-3: LR test for interaction term in CF Cox model.

    H0: β_interaction = 0  (restricted model, no interaction)
    H1: β_interaction ≠ 0  (full model, with interaction)

    LR = 2·(ℓ_full − ℓ_restricted),  df = 1
    p  = χ².sf(LR, df)

    Both models are fit with the SAME penalizer. If penalized > 0,
    the LR p-value is NOT valid and is reported as None with a flag.
    """
    result: Dict[str, Any] = {
        "lr_stat": None,
        "df": 1,
        "p_value": None,
        "aic_full": None,
        "aic_restricted": None,
        "beta_hat": None,
        "se_hat": None,
        "penalized": False,
        "penalizer_used": None,
        "n": None,
        "n_events": None,
        "note": None,
    }

    try:
        # ─── Precondition checks ─────────────────────────────────────
        required = {"time", "event", "PeakLoad"}
        if not required.issubset(data.columns):
            result["note"] = f"Missing required columns: {sorted(required - set(data.columns))}"
            return result

        if interaction_col not in data.columns:
            result["note"] = f"Interaction column '{interaction_col}' not found in data"
            return result

        n_events = int(np.sum(data["event"].astype(int).to_numpy()))
        result["n"] = len(data)
        result["n_events"] = n_events

        if n_events < max(int(opts.min_cox_events), 2):
            result["note"] = f"Too few events: {n_events}"
            return result

        # ─── Build design matrices ───────────────────────────────────
        model_data = data[["time", "event", "PeakLoad"]].copy()

        x_cols = _add_design_x_columns(
            model_data=model_data,
            source_data=data,
            extra_x_cols=opts.extra_x_cols,
            brand_encoding=opts.brand_encoding,
            brand_reference_code=opts.brand_reference_code,
        )

        # ─── P0-6 FIX: копируем cluster_id в model_data ──────────────
        # Без этого _fit_cox_model(cluster_col=...) бросит ValueError.
        # Fail-closed: если cluster_col запрошен, но отсутствует — это баг.
        if opts.cluster_col is not None:
            if opts.cluster_col in x_cols:
                result["note"] = f"cluster_col='{opts.cluster_col}' must NOT be in x_cols"
                return result
            if opts.cluster_col not in data.columns:
                result["note"] = (
                    f"cluster_col='{opts.cluster_col}' explicitly requested "
                    f"but not found in data columns"
                )
                return result
            model_data[opts.cluster_col] = data[opts.cluster_col].to_numpy()

        # Ensure interaction column is present in model_data
        if interaction_col not in model_data.columns:
            model_data[interaction_col] = data[interaction_col].astype(float).to_numpy()
        if interaction_col not in x_cols:
            x_cols.append(interaction_col)

        # Full covariates: PeakLoad + all X (including interaction)
        full_covariate_cols = ["PeakLoad"] + list(x_cols)

        # Restricted covariates: PeakLoad + X WITHOUT interaction
        restricted_x_cols = [c for c in x_cols if c != interaction_col]
        restricted_covariate_cols = ["PeakLoad"] + restricted_x_cols

        # ─── Build CF columns (identical for both models) ────────────
        residuals = np.asarray(first_stage.residuals, dtype=float).reshape(-1)
        if len(residuals) != len(data):
            result["note"] = "Residuals length mismatch"
            return result
        if not np.all(np.isfinite(residuals)):
            result["note"] = "Non-finite residuals"
            return result

        cf_result = _build_cf_columns(
            residuals=residuals,
            v_hat_basis=opts.v_hat_basis,
            v_hat_basis_params=opts.v_hat_basis_params,
            base_df=model_data,
        )
        model_data = cf_result.df_with_cf
        v_hat_cols = cf_result.column_names

        if not v_hat_cols:
            result["note"] = "No CF columns created"
            return result

        # Append CF cols to both covariate sets
        full_covariate_cols = full_covariate_cols + list(v_hat_cols)
        restricted_covariate_cols = restricted_covariate_cols + list(v_hat_cols)

        # ─── Validate ────────────────────────────────────────────────
        # ← P0-6 FIX: исключаем cluster_col из проверки (может быть строкой)
        _check_cols = [c for c in model_data.columns if c != opts.cluster_col]
        if not np.all(np.isfinite(model_data[_check_cols].to_numpy(dtype=float))):
            result["note"] = "Non-finite model data"
            return result

        # ─── Fit both models with SAME penalizer ─────────────────────
        cph_full = None
        cph_restricted = None
        used_pen = None
        last_exc: Optional[BaseException] = None

        for pen in _ALL_PENALIZERS:
            try:
                cph_f = _fit_cox_model(
                    model_data=model_data,
                    covariate_cols=full_covariate_cols,
                    penalizer=pen,
                    robust=True,
                    check_baseline=True,
                    cluster_col=opts.cluster_col,  # ← P0-6 FIX
                )
                cph_r = _fit_cox_model(
                    model_data=model_data,
                    covariate_cols=restricted_covariate_cols,
                    penalizer=pen,
                    robust=True,
                    check_baseline=True,
                    cluster_col=opts.cluster_col,  # ← P0-6 FIX
                )
                cph_full = cph_f
                cph_restricted = cph_r
                used_pen = float(pen)
                break
            except Exception as exc:
                last_exc = exc
                continue

        if cph_full is None or cph_restricted is None or used_pen is None:
            result["note"] = (
                f"Both models failed to fit. Last error: {format_exception(last_exc, False)}"
            )
            return result

        result["penalizer_used"] = used_pen
        result["penalized"] = used_pen > 0.0

        # ─── Extract log-likelihoods ─────────────────────────────────
        ll_full = _get_cox_log_likelihood(cph_full)
        ll_restricted = _get_cox_log_likelihood(cph_restricted)

        if not math.isfinite(ll_full) or not math.isfinite(ll_restricted):
            result["note"] = "Non-finite log-likelihood"
            return result

        # ─── LR statistic ────────────────────────────────────────────
        lr_stat = 2.0 * (ll_full - ll_restricted)
        if lr_stat < 0.0:
            lr_stat = 0.0  # numerical noise

        result["lr_stat"] = float(lr_stat)

        # ─── p-value (only valid if NOT penalized) ───────────────────
        stats_mod = _scipy_stats if HAS_SCIPY_STATS else None

        if result["penalized"]:
            result["p_value"] = None
            result["note"] = (
                "LR p-value not reported: model is penalized "
                f"(penalizer={used_pen}). Penalized LR test is invalid."
            )
        elif stats_mod is None:
            result["p_value"] = None
            result["note"] = "scipy.stats unavailable; p-value not computed"
        else:
            result["p_value"] = float(stats_mod.chi2.sf(lr_stat, df=1))
            result["note"] = None

        # ─── AIC ─────────────────────────────────────────────────────
        k_full = len(cph_full.params_)
        k_restricted = len(cph_restricted.params_)
        result["aic_full"] = float(-2.0 * ll_full + 2.0 * k_full)
        result["aic_restricted"] = float(-2.0 * ll_restricted + 2.0 * k_restricted)

        # ─── Interaction coefficient from full model ─────────────────
        if interaction_col in cph_full.params_.index:
            result["beta_hat"] = float(cph_full.params_.loc[interaction_col])
            result["se_hat"] = float(cph_full.standard_errors_.loc[interaction_col])
        else:
            result["note"] = (
                f"Interaction column '{interaction_col}' not found in full model params"
            )

        return result

    except Exception as exc:
        result["note"] = format_exception(exc, save_traceback=False)
        return result


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def _compute_cf_basis_col_count(config: SimulationConfig) -> int:
    bp = config.v_hat_basis_params if config.v_hat_basis_params else {}

    if config.v_hat_basis == "linear":
        return 1
    elif config.v_hat_basis == "spline":
        n_knots = max(1, int(bp.get("n_knots", 2)))
        return max(2, n_knots + 2)
    elif config.v_hat_basis == "powers":
        return max(1, int(bp.get("max_power", 2)))

    return 1


def _worker_case(
    seed_int: int,
    data: pd.DataFrame,
    config: SimulationConfig,
    fit_opts: CFFitOptions,
):
    rng = np.random.default_rng(seed_int)
    n = len(data)
    idx = rng.integers(0, n, size=n)
    bdata = data.iloc[idx].reset_index(drop=True)

    # NEW PROTECTION: Check for degenerate bootstrap samples
    if bdata["PeakLoad"].std() < 1e-6:
        return np.nan, "degenerate_peakload_in_bootstrap"
    if bdata["Z"].std() < 1e-6:
        return np.nan, "degenerate_Z_in_bootstrap"

    if np.var(bdata["Z"].to_numpy(), ddof=0) < config.var_z_threshold:
        return np.nan, "var_z_too_small"

    n_events = int(np.sum(bdata["event"].to_numpy(dtype=int)))

    if n_events < config.min_cox_events:
        return np.nan, f"too_few_events ({n_events})"

    try:
        first_stage = fit_first_stage(bdata, fit_opts)
    except Exception as e:  # noqa: BLE001
        return (
            np.nan,
            f"first_stage_failure: {format_exception(e, config.save_tracebacks)}",
        )

    try:
        result = fit_cf_cox(bdata, first_stage, fit_opts)
        return float(result.gamma_hat), None
    except Exception as e:  # noqa: BLE001
        return np.nan, f"cf_cox_failure: {format_exception(e, config.save_tracebacks)}"


def _worker_applied_wild(
    seed_int: int,
    data: pd.DataFrame,
    config: SimulationConfig,
    fit_opts: CFFitOptions,
):
    """Experimental applied wild bootstrap."""
    if not config.allow_experimental_bootstrap:
        return np.nan, "applied_wild_disabled"

    rng = np.random.default_rng(seed_int)

    if config.wild_bootstrap_dist not in {"rademacher", "normal"}:
        return np.nan, f"unknown_wild_dist ({config.wild_bootstrap_dist})"

    if np.var(data["Z"].to_numpy(), ddof=0) < config.var_z_threshold:
        return np.nan, "var_z_too_small"

    try:
        first_stage = fit_first_stage(data, fit_opts)
    except Exception as e:  # noqa: BLE001
        return (
            np.nan,
            f"first_stage_failure: {format_exception(e, config.save_tracebacks)}",
        )

    n_events = int(np.sum(data["event"].to_numpy(dtype=int)))

    if n_events < config.min_cox_events:
        return np.nan, f"too_few_events ({n_events})"

    n = len(data)

    if config.wild_bootstrap_dist == "rademacher":
        weights = rng.choice([-1.0, 1.0], size=n)
    else:
        weights = rng.normal(0.0, 1.0, size=n)

    peak_fitted = np.asarray(first_stage.fitted.fittedvalues)
    peak_star = peak_fitted + weights * first_stage.residuals
    bdata = data.copy().reset_index(drop=True)
    bdata["PeakLoad"] = peak_star

    try:
        first_stage_b = fit_first_stage(bdata, fit_opts)
    except Exception as e:  # noqa: BLE001
        return (
            np.nan,
            f"first_stage_failure_boot: {format_exception(e, config.save_tracebacks)}",
        )

    try:
        result = fit_cf_cox(bdata, first_stage_b, fit_opts)
        return float(result.gamma_hat), None
    except Exception as e:  # noqa: BLE001
        return (
            np.nan,
            f"cf_cox_failure_boot: {format_exception(e, config.save_tracebacks)}",
        )


def _worker_mc_parametric(
    seed_int: int,
    data: pd.DataFrame,
    config: SimulationConfig,
    fit_opts: CFFitOptions,
):
    """Parametric MC worker. Uses the same generate_data() DGP."""
    rng = np.random.default_rng(seed_int)
    n = len(data)

    if n <= 0:
        return np.nan, "empty_data"
    if config.baseline_hazard <= 0.0 or config.censoring_scale <= 0.0:
        return np.nan, "invalid_baseline_or_censoring"

    try:
        bdata = generate_data(
            n=n,
            contamination=config.contamination,
            baseline_hazard=config.baseline_hazard,
            censoring_scale=config.censoring_scale,
            rng=rng,
            instrument_strength=None,
            dgp=config.dgp,
            instrument_source="normal",
            contamination_probability=config.contamination_probability,
        )
    except Exception as e:  # noqa: BLE001
        return np.nan, f"dgp_failure: {format_exception(e, config.save_tracebacks)}"

    if not np.all(np.isfinite(bdata[["time", "PeakLoad", "Z"]].to_numpy())):
        return np.nan, "nonfinite_bootstrap_data"

    if np.var(bdata["Z"].to_numpy(), ddof=0) < config.var_z_threshold:
        return np.nan, "var_z_too_small"

    n_events = int(np.sum(bdata["event"].to_numpy(dtype=int)))

    if n_events < config.min_cox_events:
        return np.nan, f"too_few_events_parametric ({n_events})"

    try:
        first_stage = fit_first_stage(bdata, fit_opts)
    except Exception as e:  # noqa: BLE001
        return (
            np.nan,
            f"first_stage_failure_parametric: {format_exception(e, config.save_tracebacks)}",
        )

    try:
        result = fit_cf_cox(bdata, first_stage, fit_opts)
        return float(result.gamma_hat), None
    except Exception as e:  # noqa: BLE001
        return (
            np.nan,
            f"cf_cox_failed_parametric: {format_exception(e, config.save_tracebacks)}",
        )


def _worker_oracle_cox(
    seed_int: int,
    data: pd.DataFrame,
    config: SimulationConfig,
    fit_opts: CFFitOptions,
):
    """Oracle Cox diagnostic: fits Cox with true confounder U observed."""
    if "U" not in data.columns:
        return np.nan, "oracle_U_missing"

    if np.var(data["Z"].to_numpy(), ddof=0) < config.var_z_threshold:
        return np.nan, "var_z_too_small"

    model_data = data[["time", "event", "PeakLoad", "U"]].copy()

    try:
        x_cols = _add_design_x_columns(
            model_data=model_data,
            source_data=data,
            extra_x_cols=fit_opts.extra_x_cols,
            brand_encoding=fit_opts.brand_encoding,
            brand_reference_code=fit_opts.brand_reference_code,
        )
    except Exception as e:
        return (
            np.nan,
            f"oracle_design_failure: {format_exception(e, config.save_tracebacks)}",
        )

    covariate_cols = ["PeakLoad", "U"] + x_cols

    try:
        _validate_survival_frame(model_data, covariate_cols)
    except Exception as e:
        return (
            np.nan,
            f"oracle_validation_failure: {format_exception(e, config.save_tracebacks)}",
        )

    n_events = int(np.sum(model_data["event"].to_numpy(dtype=int)))
    min_events = _min_events_required_from_count(len(covariate_cols), fit_opts)

    if n_events < min_events:
        return np.nan, f"too_few_events_oracle ({n_events})"

    last_exc: Optional[BaseException] = None

    for pen in _ALL_PENALIZERS:
        try:
            cph = _fit_cox_model(
                model_data=model_data,
                covariate_cols=covariate_cols,
                penalizer=pen,
                robust=True,
                check_baseline=True,
                cluster_col=None,
            )

            gamma_hat = float(cph.params_.loc["PeakLoad"])

            if not np.isfinite(gamma_hat):
                return np.nan, "oracle_nonfinite_gamma"

            return gamma_hat, None

        except Exception as exc:
            last_exc = exc
            continue

    return (
        np.nan,
        f"oracle_cox_failure: {format_exception(last_exc, config.save_tracebacks)}",
    )


def bootstrap_cf_standard_error(
    data: pd.DataFrame,
    n_bootstrap: int,
    seed_sequence: np.random.SeedSequence,
    config: SimulationConfig,
    fit_opts: Optional[CFFitOptions] = None,
) -> BootstrapResult:
    """Compute bootstrap SE using selected method."""
    validate_simulation_config(config)

    if fit_opts is None:
        fit_opts = fit_options_from_config(config)

    # ★ DISABLE NESTED BOOTSTRAP: workers call fit_cf_cox which would
    # otherwise re-enter bootstrap_cf_standard_error → exponential blow-up.
    import dataclasses
    worker_fit_opts = dataclasses.replace(fit_opts, n_bootstrap=0)

    try:
        n_bootstrap = int(n_bootstrap)
    except Exception:
        n_bootstrap = 0

    if n_bootstrap <= 0:
        return BootstrapResult(
            bootstrap_se=None,
            n_successful=0,
            n_failures=0,
            success_rate=0.0,
            error_examples=["n_bootstrap <= 0"],
        )

    if n_bootstrap <= 1:
        return BootstrapResult(
            bootstrap_se=None,
            n_successful=0,
            n_failures=0,
            success_rate=0.0,
            error_examples=["n_bootstrap <= 1"],
        )

    child_seqs = seed_sequence.spawn(n_bootstrap)
    seeds = [int(ss.generate_state(1, dtype=np.uint64)[0]) for ss in child_seqs]

    estimates: List[float] = []
    failures = 0
    rejection_reasons: List[str] = []
    error_examples: List[str] = []

    method = config.bootstrap_method.lower()

    if method == "case":
        worker = _worker_case
    elif method == "applied_wild":
        worker = _worker_applied_wild
    elif method == "mc_parametric":
        worker = _worker_mc_parametric
    else:
        raise ValueError(f"Unknown bootstrap method: {method}")

    def safe_worker(seed_int: int):
        try:
            est, reason = worker(seed_int, data, config, worker_fit_opts)

            if reason is None and np.isfinite(est):
                return float(est), ""

            return np.nan, reason or "nonfinite"

        except Exception as exc:  # noqa: BLE001
            return np.nan, format_exception(exc, config.save_tracebacks)

    try:
        if config.bootstrap_jobs is None or config.bootstrap_jobs == 1:
            raw_results = [safe_worker(s) for s in seeds]
        else:
            raw_results = Parallel(
                n_jobs=config.bootstrap_jobs,
                backend="loky",
                batch_size=10,
            )(delayed(safe_worker)(s) for s in seeds)

    except Exception as exc:
        return BootstrapResult(
            bootstrap_se=None,
            n_successful=0,
            n_failures=n_bootstrap,
            success_rate=0.0,
            error_examples=[format_exception(exc, config.save_tracebacks)],
        )

    for est, reason in raw_results:
        if reason:
            failures += 1
            rejection_reasons.append(reason)

            if len(error_examples) < 10:
                error_examples.append(reason)
        else:
            if np.isfinite(est):
                estimates.append(float(est))
            else:
                failures += 1
                rejection_reasons.append("nonfinite_estimate")

    n_successful = len(estimates)
    success_rate = float(n_successful / n_bootstrap) if n_bootstrap > 0 else 0.0
    failure_rate = float(failures / n_bootstrap) if n_bootstrap > 0 else 1.0

    reason_summary: Dict[str, int] = {}

    for r in rejection_reasons:
        key = r.split(":")[0].strip()
        reason_summary[key] = reason_summary.get(key, 0) + 1

    requested_min = int(math.ceil(config.bootstrap_success_frac * n_bootstrap))
    required_min_success = max(3, requested_min)
    required_min_success = min(required_min_success, n_bootstrap)

    high_failure_rate = failure_rate > config.max_failure_rate

    if n_successful < required_min_success or n_successful <= 1 or high_failure_rate:
        logger.error(
            "Bootstrap failed. Success rate: %.2f%%. Rejection reasons: %s",
            success_rate * 100,
            reason_summary  # Shows why iterations fail (degenerate samples, OOM, etc.)
        )
        return BootstrapResult(
            bootstrap_se=None,
            n_successful=n_successful,
            n_failures=failures,
            success_rate=success_rate,
            ci_lower=None,
            ci_upper=None,
            high_failure_rate=high_failure_rate,
            estimates=None,
            error_examples=error_examples,
            rejection_reason_summary=reason_summary,
        )

    bootstrap_se = float(np.std(estimates, ddof=1))

    if not np.isfinite(bootstrap_se) or bootstrap_se <= 0.0:
        return BootstrapResult(
            bootstrap_se=None,
            n_successful=n_successful,
            n_failures=failures,
            success_rate=success_rate,
            ci_lower=None,
            ci_upper=None,
            high_failure_rate=high_failure_rate,
            estimates=None,
            error_examples=error_examples,
            rejection_reason_summary=reason_summary,
        )

    lower_pct, upper_pct = np.percentile(estimates, [2.5, 97.5])

    return BootstrapResult(
        bootstrap_se=bootstrap_se,
        n_successful=n_successful,
        n_failures=failures,
        success_rate=success_rate,
        ci_lower=float(lower_pct),
        ci_upper=float(upper_pct),
        high_failure_rate=high_failure_rate,
        estimates=None,
        error_examples=error_examples,
        rejection_reason_summary=reason_summary,
    )


# ---------------------------------------------------------------------------
# PH diagnostics
# ---------------------------------------------------------------------------
def check_proportional_hazards(
    cph: CoxPHFitter,
    data: pd.DataFrame,
) -> Optional[str]:
    """
    High-level PH check using lifelines as primary diagnostic.
    Returns None if no problem detected, otherwise a string message.
    """
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            cph.check_assumptions(
                data,
                p_value_threshold=0.05,
                show_plots=False,
            )

        warn_summary = _summarize_warnings(caught_warnings)

        if warn_summary is not None:
            return f"PH check flagged warnings: {warn_summary}"

        return None

    except Exception as exc:  # noqa: BLE001
        return f"PH check failed or flagged: {format_exception(exc, False)}"


# ---------------------------------------------------------------------------
# P0-4: Structured PH diagnostics report
# ---------------------------------------------------------------------------
def ph_diagnostics_report(
    cph: CoxPHFitter,
    data: pd.DataFrame,
    alpha: float = 0.05,
    time_transform: str = "rank",
) -> Dict[str, Any]:
    """
    P0-4: Структурированный PH-отчёт на основе Schoenfeld residuals.

    Использует lifelines.statistics.proportional_hazard_test с fallback
    на ручное вычисление. Возвращает структурированный dict для audit trail.
    """
    report: Dict[str, Any] = {
        "method": "Schoenfeld residual test (Grambsch-Therneau)",
        "alpha": float(alpha),
        "time_transform": str(time_transform),
        "n": int(len(data)),
        "n_events": int(data["event"].astype(int).sum()),
        "global_test": None,
        "variables": {},
        "violations": [],
        "status": "PASS",
    }

    # ─── Проверка предусловий ────────────────────────────────────────
    cph_params = getattr(cph, "params_", None)
    if cph_params is None:
        report["status"] = "ERROR"
        report["error"] = "cph.params_ not available (model not fitted)"
        return report

    try:
        covariates = [str(name) for name in cph_params.index]
    except AttributeError:
        try:
            covariates = [str(name) for name in cph_params.keys()]
        except (AttributeError, TypeError):
            report["status"] = "ERROR"
            report["error"] = "Cannot extract covariate names from cph.params_"
            return report

    if not covariates:
        report["status"] = "ERROR"
        report["error"] = "No covariates in fitted model"
        return report

    model_cols_in_data = [c for c in covariates if c in data.columns]
    missing_cols = [c for c in covariates if c not in data.columns]
    if missing_cols:
        report["status"] = "ERROR"
        report["error"] = "Covariates missing in data: " + str(sorted(missing_cols))
        return report

    ph_data_cols = ["time", "event"] + model_cols_in_data

    # ★ FIX: lifelines требует cluster_col для вычисления Schoenfeld residuals
    cluster_col_name = getattr(cph, "cluster_col", None)
    if (
        cluster_col_name
        and cluster_col_name in data.columns
        and cluster_col_name not in ph_data_cols
    ):
        ph_data_cols.append(cluster_col_name)

    ph_data = data[ph_data_cols].copy()
    # Приводим к float только числовые ковариаты, cluster_id может быть строкой
    for col in ph_data_cols:
        if col != cluster_col_name:
            ph_data[col] = pd.to_numeric(ph_data[col], errors="coerce").astype(float)

    # ─── Попытка 1: proportional_hazard_test ─────────────────────────
    ph_result = None
    try:
        from lifelines.statistics import proportional_hazard_test

        try:
            ph_result = proportional_hazard_test(cph, ph_data, time_transform=time_transform)
        except TypeError:
            try:
                ph_result = proportional_hazard_test(cph, ph_data)
            except Exception:
                ph_result = None
    except (ImportError, AttributeError):
        ph_result = None

    # ─── Извлечение результатов ──────────────────────────────────────
    if ph_result is not None:
        summary = getattr(ph_result, "summary", None)
        if summary is not None and len(summary) > 0:
            stat_col = None
            p_col = None
            for candidate in ("test_statistic", "test_stat", "coef", "statistic", "chi2_stat"):
                if candidate in summary.columns:
                    stat_col = candidate
                    break
            for candidate in ("p", "p_value", "p-value", "pvalue"):
                if candidate in summary.columns:
                    p_col = candidate
                    break

            if stat_col is not None and p_col is not None:
                for cov in summary.index:
                    cov_name = str(cov)
                    try:
                        test_stat = float(summary.loc[cov, stat_col])
                        p_value = float(summary.loc[cov, p_col])
                    except (KeyError, TypeError, ValueError, IndexError):
                        continue

                    if not math.isfinite(test_stat) or not math.isfinite(p_value):
                        report["variables"][cov_name] = {
                            "test_statistic": None,
                            "p_value": None,
                            "reject_at_alpha": None,
                            "status": "SKIP",
                            "note": "Non-finite test statistic or p-value",
                        }
                        continue

                    reject = bool(p_value < alpha)
                    report["variables"][cov_name] = {
                        "test_statistic": test_stat,
                        "p_value": p_value,
                        "reject_at_alpha": reject,
                        "status": "FAIL" if reject else "PASS",
                    }
                    if reject:
                        report["violations"].append(cov_name)

    # ─── Попытка 2: ручное вычисление Schoenfeld residuals ───────────
    if not report["variables"]:
        try:
            schoenfeld_res = cph.compute_schoenfeld_residuals()
        except Exception as exc:
            report["status"] = "ERROR"
            report["error"] = (
                "Both proportional_hazard_test and "
                "compute_schoenfeld_residuals failed: " + format_exception(exc, False)
            )
            return report

        event_mask = ph_data["event"].astype(bool).to_numpy()
        event_times = ph_data.loc[event_mask, "time"].to_numpy(dtype=float)

        if time_transform == "rank":
            g_t = np.argsort(np.argsort(event_times)).astype(float)
        elif time_transform == "log":
            g_t = np.log(np.maximum(event_times, 1e-12))
        else:
            g_t = event_times.copy()

        stats_mod = _scipy_stats if HAS_SCIPY_STATS else None

        for cov in covariates:
            if cov not in schoenfeld_res.columns:
                continue

            residuals = schoenfeld_res[cov].to_numpy(dtype=float)

            if len(residuals) < 3:
                report["variables"][cov] = {
                    "test_statistic": None,
                    "p_value": None,
                    "reject_at_alpha": None,
                    "status": "SKIP",
                    "note": "Too few events for PH test",
                }
                continue

            if len(residuals) != len(g_t):
                continue

            g_centered = g_t - float(np.mean(g_t))
            r_centered = residuals - float(np.mean(residuals))

            g_var = float(np.sum(g_centered**2))
            if g_var < 1e-12:
                report["variables"][cov] = {
                    "test_statistic": None,
                    "p_value": None,
                    "reject_at_alpha": None,
                    "status": "SKIP",
                    "note": "Degenerate time transform",
                }
                continue

            slope = float(np.sum(g_centered * r_centered) / g_var)
            r_var = float(np.var(residuals, ddof=1))

            if r_var < 1e-12:
                report["variables"][cov] = {
                    "test_statistic": 0.0,
                    "p_value": 1.0,
                    "reject_at_alpha": False,
                    "status": "PASS",
                }
                continue

            n_events_local = len(residuals)
            se_slope = math.sqrt(r_var / g_var)
            if se_slope < 1e-12:
                se_slope = 1e-12

            z_stat = slope / se_slope
            test_stat = z_stat**2

            if stats_mod is not None:
                p_value = float(stats_mod.chi2.sf(test_stat, df=1))
            else:
                p_value = float(math.exp(-test_stat / 2.0))

            if not math.isfinite(test_stat) or not math.isfinite(p_value):
                report["variables"][cov] = {
                    "test_statistic": None,
                    "p_value": None,
                    "reject_at_alpha": None,
                    "status": "SKIP",
                    "note": "Non-finite test result",
                }
                continue

            reject = bool(p_value < alpha)
            report["variables"][cov] = {
                "test_statistic": float(test_stat),
                "p_value": float(p_value),
                "reject_at_alpha": reject,
                "status": "FAIL" if reject else "PASS",
            }
            if reject:
                report["violations"].append(cov)

    # ─── Глобальный тест ─────────────────────────────────────────────
    valid_vars = {
        k: v for k, v in report["variables"].items() if v.get("test_statistic") is not None
    }
    if valid_vars:
        all_stats = [v["test_statistic"] for v in valid_vars.values()]
        global_stat = float(sum(all_stats))
        df_global = len(all_stats)

        stats_mod = _scipy_stats if HAS_SCIPY_STATS else None
        if stats_mod is not None:
            global_p = float(stats_mod.chi2.sf(global_stat, df_global))
        else:
            all_p = [v["p_value"] for v in valid_vars.values() if v["p_value"] is not None]
            global_p = float(min(all_p)) if all_p else float("nan")

        report["global_test"] = {
            "test_statistic": global_stat,
            "p_value": global_p,
            "df": df_global,
            "reject_at_alpha": bool(global_p < alpha),
        }

    # ─── Определение общего статуса ──────────────────────────────────
    if report["violations"]:
        report["status"] = "WARN"
    else:
        report["status"] = "PASS"

    return report


# ---------------------------------------------------------------------------
# P-12: Kaplan-Meier validator
# ---------------------------------------------------------------------------
def kaplan_meier_validator(
    data: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
) -> Dict[str, Any]:
    """P-12: Kaplan-Meier validator."""
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        raise RuntimeError("lifelines is required for kaplan_meier_validator")

    if time_col not in data.columns or event_col not in data.columns:
        raise KeyError("KM validator missing required columns")

    df = data.dropna(subset=[time_col, event_col])

    if len(df) == 0:
        raise ValueError("KM validator: empty data")

    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])

    return {
        "n": int(len(df)),
        "events": int(df[event_col].astype(int).sum()),
        "event_rate": float(df[event_col].astype(int).mean()),
        "median_survival_time": float(kmf.median_survival_time_),
    }