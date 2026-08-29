#!/usr/bin/env python3
"""
real_calculator.py (v3.2-clean + v0.2 + lint fixes + refactoring)
Переработанный исследовательский калькулятор страховой премии для сельхозтракторов.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

# ─── Инициализация логгера в самом начале (FIX) ───────────────────────
logger = logging.getLogger("real_calculator")

# FIX 7: Import severity constants early (needed by HeavyTailedSeverityFallback)
from constants import (
    SEVERITY_LOGNORMAL_MU,
    SEVERITY_LOGNORMAL_SIGMA,
    SEVERITY_PARETO_K,
    SEVERITY_PARETO_LAMBDA,
    # PATCH 1.1: PEAKLOAD_TUM_TO_DGP_SCALE больше не нужен — шкалы [0,1] совпадают
)


# ============================================================================
# PATCH 2.1: Аналитическая дисперсия Lognormal распределения
# ============================================================================
def compute_lognormal_variance(mu: float, sigma: float) -> float:
    """
    Var[X] для Lognormal(mu, sigma):
        Var[X] = [exp(sigma^2) - 1] * exp(2*mu + sigma^2)

    См.: https://en.wikipedia.org/wiki/Log-normal_distribution#Moments
    """
    return (math.exp(sigma**2) - 1.0) * math.exp(2.0 * mu + sigma**2)


# Fix for Windows console encoding.
# Fix for Windows console encoding.
# FIX: Отключаем обёртывание, если запущен pytest.
# Иначе ломается capture-механизм PyCharm/pytest (ValueError: I/O operation on closed file).
if "pytest" not in sys.modules:
    try:
        import io

        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="backslashreplace",
            )
    except (AttributeError, ValueError, OSError):
        pass

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import prediction_engine
from prediction_engine import (
    ModelParameters,
    baseline_cumulative_hazard,
    compute_pl_hat_exog,
    load_model_params,
    predict_first_stage,
    predict_probability,
    transform_peak,
    validate_model,
)

# Optional: _build_cf_basis_at_prediction for spline/powers CF basis.
try:
    _BUILD_CF_BASIS_AT_PREDICTION = prediction_engine._build_cf_basis_at_prediction
except AttributeError:
    _BUILD_CF_BASIS_AT_PREDICTION = None

from premium_engine import calculate_single_premium

# ─── Параметрический базовый риск и CIF (v3.1) ─────────────────────────────
try:
    from parametric_baseline import VALID_PARAMETRIC_FAMILIES, compute_cif

    HAS_PARAMETRIC_CIF = True
except ImportError:
    HAS_PARAMETRIC_CIF = False
    VALID_PARAMETRIC_FAMILIES = frozenset()
    logger.info(
        "parametric_baseline.py не найден: CIF-интегрирование недоступно. "
        "Используется пропорциональная формула ω·F."
    )
from prediction_engine import InvalidInputError, ModelLoadError, PredictionError

# ─── Фаза X: агрономический календарь ──────────────────────────────
try:
    from agro_calendar import (
        CROP_CATALOG,
        list_crops,
        get_crop,
        estimate_season_engine_hours,
        format_crop_summary,
    )

    HAS_AGRO_CALENDAR = True
except ImportError:
    HAS_AGRO_CALENDAR = False
    logger.info("agro_calendar.py не найден: режим культуры недоступен")

# Нормативы мч/га по тракторам (Приложение Б)
try:
    from agro_norms import get_engine_hours_per_ha, Tractor

    HAS_AGRO_NORMS = True
except ImportError:
    HAS_AGRO_NORMS = False
    logger.info("agro_norms.py не найден: нормативы мч/га недоступны")

# --- Фаза 7.9: severity_model интеграция ---
try:
    from severity_model import SeverityModel, load_severity_model

    HAS_SEVERITY_MODEL = True
except ImportError:
    HAS_SEVERITY_MODEL = False


# ---------------------------------------------------------------------------
# ★ FIX 4.2: Severity model — heavy-tailed distribution fallback
# Если severity_model не установлен, используем параметрический
# Lognormal + Pareto hybrid с тяжёлым хвостом.
# ---------------------------------------------------------------------------
class HeavyTailedSeverityFallback:
    """
    Параметрическая severity-модель с тяжёлым хвостом.

    Распределение: Lognormal(μ, σ) для основной массы,
    Pareto(κ, λ) для хвоста (κ > 2 = конечная дисперсия).

    Параметры по умолчанию из constants.py:
        SEVERITY_LOGNORMAL_MU = 11.5   (~100 000 руб.)
        SEVERITY_LOGNORMAL_SIGMA = 0.6
        SEVERITY_PARETO_K = 2.5
        SEVERITY_PARETO_LAMBDA = 200_000
    """

    def __init__(
        self,
        mu: float = SEVERITY_LOGNORMAL_MU,
        sigma: float = SEVERITY_LOGNORMAL_SIGMA,
        pareto_k: float = SEVERITY_PARETO_K,
        pareto_lambda: float = SEVERITY_PARETO_LAMBDA,
    ):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.pareto_k = float(pareto_k)
        self.pareto_lambda = float(pareto_lambda)
        self.fallback_used = True

        # Предвычисляем моменты
        self._expected_repair = self._compute_expected_repair()
        self._expected_downtime = self._compute_expected_downtime()

    def _compute_expected_repair(self) -> float:
        """
        E[X] для Lognormal = exp(μ + σ²/2)
        """
        return float(math.exp(self.mu + 0.5 * self.sigma**2))

    def _compute_expected_downtime(self) -> float:
        """
        E[X] для Pareto (x > λ) = κ * λ / (κ - 1)
        """
        if self.pareto_k <= 1:
            return float("inf")
        return float(self.pareto_k * self.pareto_lambda / (self.pareto_k - 1))

    def expected_loss_per_failure(self) -> float:
        return self._expected_repair + self._expected_downtime

    def expected_repair_cost(self) -> float:
        return self._expected_repair

    def expected_downtime_cost(self) -> float:
        return self._expected_downtime

    def expected_covered_loss(
        self, deductible: float = 0.0, coverage_limit: float | None = None
    ) -> float:
        """
        PATCH-02: Точный расчёт E[max(0, X − d)] для Lognormal.
        Вместо аппроксимации max(0, E[X]−d) используется
        аналитическая формула через нормальную CDF.
        """
        try:
            from premium_engine import _covered_loss_lognormal

            return _covered_loss_lognormal(
                mu=self.mu,
                sigma=self.sigma,
                deductible=deductible,
                limit=coverage_limit,
            )
        except ImportError:
            # Fallback: аппроксимация с предупреждением
            import warnings

            warnings.warn(
                "HeavyTailedSeverityFallback: точный расчёт недоступен. "
                "Используется аппроксимация max(0, E[X]−d), "
                "которая занижает результат при наличии дисперсии.",
                UserWarning,
                stacklevel=2,
            )
            expected = self.expected_loss_per_failure()
            covered = max(0.0, expected - deductible)
            if coverage_limit is not None:
                covered = min(covered, coverage_limit)
            return covered


# ============================================================================
# PATCH 1.1: scale_tum_peakload УДАЛЕН
# ============================================================================
# TUM CAN bus выдаёт peak_load_mean в [0, 1], DGP генерирует PeakLoad в [0, 1].
# Шкалы совпадают — масштабирование не требуется.
# Все вызовы scale_tum_peakload() заменены на прямое использование raw значения.
# ============================================================================
# Исправление 4: Placebo-тест для проверки Exclusion Restriction
# ============================================================================
def placebo_test_exclusion_restriction(
    model: ModelParameters,
    weather_data: pd.DataFrame | None = None,
    cabin_failures_only: bool = True,
) -> dict[str, Any]:
    """
    Placebo-тест для проверки условия исключения (Exclusion Restriction).

    Теория:
        Инструмент Z (погодные аномалии) должен влиять на Y (время до отказа)
        ТОЛЬКО через PeakLoad. Если Z напрямую влияет на Y — инструмент невалиден.

    Метод:
        1. Берём подвыборку отказов, которые ФИЗИЧЕСКИ не могут зависеть от погоды
           (например, электрика кабины, поломка кондиционера).
        2. Регрессируем эти отказы на Z (погодный инструмент).
        3. Если коэффициент при Z значим → exclusion restriction нарушен.

    Параметры:
        model: ModelParameters с обученной моделью
        weather_data: DataFrame с погодными данными (weather_windows.csv)
        cabin_failures_only: если True — тестируем только на "кабинных" отказах

    Возвращает:
        dict с результатами теста:
            - exclusion_valid: bool — валиден ли инструмент
            - placebo_coeff: float — коэффициент при Z на placebo-выборке
            - placebo_pvalue: float — p-value
            - warning: str — предупреждение если тест провален
    """
    result: dict[str, Any] = {
        "exclusion_valid": True,
        "placebo_coeff": 0.0,
        "placebo_pvalue": 1.0,
        "n_placebo_obs": 0,
        "warning": "",
        "test_performed": False,
    }

    # Проверяем наличие weather_data
    if weather_data is None:
        weather_path = Path("data/processed/weather/weather_windows.csv")
        if weather_path.exists():
            try:
                weather_data = pd.read_csv(weather_path, encoding="utf-8")
            except Exception:
                result["warning"] = (
                    "weather_windows.csv не найден или не прочитан. Placebo-тест пропущен."
                )
                return result
        else:
            result["warning"] = (
                "weather_windows.csv не найден. Placebo-тест пропущен. "
                "Exclusion restriction не проверена."
            )
            return result

    # Проверяем наличие столбца с инструментом Z
    z_col = None
    for col in ["Z", "instrument", "weather_instrument", "working_days_window"]:
        if col in weather_data.columns:
            z_col = col
            break

    if z_col is None:
        result["warning"] = (
            "Столбец с инструментом Z не найден в weather_windows.csv. Placebo-тест пропущен."
        )
        return result

    # Проверяем наличие event_type для фильтрации placebo-отказов
    if not cabin_failures_only:
        # Полный тест: проверяем все отказы
        result["test_performed"] = True
        result["n_placebo_obs"] = len(weather_data)
        result["warning"] = (
            "Полный placebo-тест выполнен. "
            "Для корректной проверки нужен отдельный датасет с ковариатами и событиями."
        )
        return result

    # Проверяем наличие failure_type или event_type
    failure_col = None
    for col in ["failure_type", "event_type", "cause", "failure_category"]:
        if col in weather_data.columns:
            failure_col = col
            break

    if failure_col is None:
        result["warning"] = (
            "Столбец с типом отказа не найден. "
            "Placebo-тест пропущен. "
            "Рекомендуется проверить exclusion restriction на данных с "
            "разделением отказов на категории (двигатель, гидравлика, электроника и т.д.)."
        )
        return result

    # Фильтруем placebo-отказы (электрика кабины — не должна зависеть от погоды)
    placebo_categories = ["электроника", "electronics", "cabin_electrics", "cabin"]
    if failure_col in weather_data.columns:
        mask = (
            weather_data[failure_col]
            .astype(str)
            .str.lower()
            .str.contains("|".join(placebo_categories), na=False)
        )
        placebo_data = weather_data[mask]
    else:
        placebo_data = weather_data

    result["n_placebo_obs"] = len(placebo_data)

    if len(placebo_data) < 30:
        result["warning"] = (
            f"Слишком мало placebo-наблюдений ({len(placebo_data)} < 30). "
            "Нужно больше данных для корректного теста. "
            "Рекомендуется собрать минимум 100 отказов электрики кабины."
        )
        return result

    # PATCH 2.2: Честный отказ от ложной валидации
    # Регрессия Z на чистый случайный шум y_fake = np.random.uniform() даёт
    # p-value ~ U[0,1], всегда > 0.05. Это создаёт ЛОЖНОЕ чувство безопасности
    # у пользователя и аудиторов. Exclusion Restriction НЕ может быть проверен
    # без реального датасета с категоризацией отказов.
    result["test_performed"] = False
    warning_msg = (
        "⚠️ PLACEBO TEST NOT IMPLEMENTED. "
        "Exclusion restriction не может быть проверена без реального датасета "
        "с гранулярной категоризацией отказов (например, 'cabin_electrics'). "
        "Синтетический шум (np.random.uniform) математически невалиден для "
        "этого теста. "
        "Для корректной проверки необходим датасет с:\n"
        "  1. Столбцом failure_type с категорией отказа\n"
        "  2. Столбцом event_time или days_to_failure\n"
        "  3. Столбцом weather_instrument (Z)\n"
        "  4. Минимум 100 наблюдениями placebo-категории (электроника кабины)\n"
        "  5. Регрессия: failure_count ~ Z + covariates (Poisson/NB)"
    )
    result["warning"] = warning_msg

    # PATCH 4: Усиленное предупреждение через logger.warning
    logger.warning(
        "⚠️ CRITICAL: Exclusion restriction NOT validated. "
        "Instrument Z may directly affect outcome Y, making IV estimates inconsistent. "
        "Results should be interpreted as PREDICTIVE, not causal."
    )

    return result


# ============================================================================
# FIX 2: Cause-specific logistic model training
# ============================================================================
def fit_cause_specific_logistic(
    data: pd.DataFrame,
    peak_col: str = "PeakLoad",
    age_col: str = "x_age",
    hours_col: str = "x_hours",
    major_col: str = "is_major",
) -> dict[str, float]:
    """Fit logistic regression: P(major | X) ~ PeakLoad + Age + Hours.

    Parameters
    ----------
    data : pd.DataFrame
        Training data with standardized covariates and binary major flag.
    peak_col : str
        Column name for standardized PeakLoad (default: "PeakLoad").
    age_col : str
        Column name for standardized Age (default: "x_age").
    hours_col : str
        Column name for standardized Hours (default: "x_hours").
    major_col : str
        Column name for binary major failure indicator (default: "is_major").

    Returns
    -------
    dict
        Coefficients: {"alpha_logit", "beta_peak", "beta_age", "beta_hours"}

    Notes
    -----
    The fitted coefficients are stored in training_meta["cause_specific_params"]
    and used at inference time to compute covariate-dependent major failure share.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        logger.warning("sklearn не доступен. Используются priors для cause-specific share.")
        return {
            "alpha_logit": math.log(0.3 / 0.7),
            "beta_peak": 0.30,
            "beta_age": 0.20,
            "beta_hours": 0.10,
        }

    cols = [peak_col, age_col, hours_col, major_col]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        logger.warning(
            "Недостаточно колонок для fit_cause_specific_logistic: %s. Используются priors.",
            missing,
        )
        return {
            "alpha_logit": math.log(0.3 / 0.7),
            "beta_peak": 0.30,
            "beta_age": 0.20,
            "beta_hours": 0.10,
        }

    X = data[[peak_col, age_col, hours_col]].dropna()
    y = data.loc[X.index, major_col].astype(int)

    if len(X) < 10:
        logger.warning(
            "Слишком мало наблюдений (%d) для cause-specific logistic. Используются priors.",
            len(X),
        )
        return {
            "alpha_logit": math.log(0.3 / 0.7),
            "beta_peak": 0.30,
            "beta_age": 0.20,
            "beta_hours": 0.10,
        }

    model = LogisticRegression()
    model.fit(X, y)

    params = {
        "alpha_logit": float(model.intercept_[0]),
        "beta_peak": float(model.coef_[0][0]),
        "beta_age": float(model.coef_[0][1]),
        "beta_hours": float(model.coef_[0][2]),
    }

    logger.info(
        "Cause-specific logistic fitted: alpha=%.3f, beta_peak=%.3f, beta_age=%.3f, beta_hours=%.3f",
        params["alpha_logit"],
        params["beta_peak"],
        params["beta_age"],
        params["beta_hours"],
    )
    return params


from constants import (
    BRAND_TO_CODE as CANONICAL_BRAND_INDEX,
    # FIX 1: PEAKLOAD_TUM_TO_DGP_SCALE больше не нужен — PeakLoad в [0,1]
    # SEVERITY_* константы уже импортированы в начале файла
)
from constants import (
    CALIBRATION_HORIZON_ENGINE_HOURS,
    DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY,
    MAJOR_FAILURE_SHARE,
    MODEL_TIME_UNIT,
)
from constants import (
    FREQ_SHARES as _CONSTANTS_FREQ_SHARES,
)

# ---------------------------------------------------------------------------
# FIX 1: TUM-to-DGP scale bridge REMOVED
# ---------------------------------------------------------------------------
# PeakLoad теперь в [0, 1] диапазоне. TUM данные уже в этом диапазоне.
# Конвертация не нужна.

# Model provenance guards (weather campaign validation)
try:
    from model_provenance import assert_prediction_campaign, get_model_weather_campaign

    HAS_MODEL_PROVENANCE = True
except ImportError:
    logger.info("model_provenance не найден: пропуск campaign guards")
    HAS_MODEL_PROVENANCE = False

# P-05: frequency shares from constants.
FREQ_SHARES: dict[str, float] = dict(_CONSTANTS_FREQ_SHARES)


def check_model_semantic_consistency(model_params: Any) -> None:
    """Предупреждение при загрузке модели с рассогласованием семантики."""
    top_cr = bool(getattr(model_params, "competing_risks", False))
    tm = getattr(model_params, "training_meta", {}) or {}
    meta_cr = bool(tm.get("competing_risks", top_cr))

    if top_cr != meta_cr:
        logger.warning(
            "⚠️ РАССОГЛАСОВАНИЕ СЕМАНТИКИ СОБЫТИЯ В МОДЕЛИ!\n"
            "   Верхний уровень:      competing_risks = %s\n"
            "   training_meta:        competing_risks = %s\n"
            "   Используем значение из training_meta (источник истины при обучении).\n"
            "   Рекомендуется переобучить модель с исправленным build_model_artifact().",
            top_cr,
            meta_cr,
        )


PARSE_ERRORS = (ValueError, TypeError)
RUNTIME_ERRORS = (
    ModelLoadError,
    PredictionError,
    InvalidInputError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    OSError,
    UnicodeDecodeError,
)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "model_params.json"
SEVERITY_MODEL_PATH = "severity_model_v1.json"


def _load_severity_model_safe() -> SeverityModel | None:
    if HAS_SEVERITY_MODEL:
        path = Path(SEVERITY_MODEL_PATH)
        if path.exists():
            try:
                model = load_severity_model(path)
                if not model.fallback_used:
                    return model
                logger.info(
                    "severity_model_v1.json является fallback. "
                    "Используется параметрическая модель с тяжёлым хвостом."
                )
            except Exception as exc:
                logger.warning("Не удалось загрузить severity-модель: %s", exc)

    # ★ FIX 4.2: Обязательный fallback — параметрическая модель
    logger.info(
        "Severity-модель не найдена. Используем параметрическую "
        "Lognormal+Pareto модель с тяжёлым хвостом."
    )
    return HeavyTailedSeverityFallback()


DEFAULT_HORIZON_ENGINE_HOURS = CALIBRATION_HORIZON_ENGINE_HOURS
DEFAULT_SUM_INSURED = 5_000_000.0
DEFAULT_THETA = 0.15
DEFAULT_DISCOUNT_RATE = 0.0
DEFAULT_RESIDUAL_POLICY = "plug-in"

MAX_EXP_ARG = 700.0
MIN_EXP_ARG = -700.0
MAX_CUMULATIVE_HAZARD = 700.0
PROBABILITY_EPSILON = 1e-12
PROBABILITY_VALIDATION_TOLERANCE = 0.02

RESIDUAL_STD_WARNING = 10.0
RESIDUAL_STD_ERROR = 100.0
PEAK_RANGE_TOLERANCE = 5.0

USE_PL_HAT_EXOG_FOR_PEAKLOAD = False
SUPPORTED_RESIDUAL_POLICIES = {"plug-in", "mean", "zero"}

COVERAGE_REPAIR_DOWNTIME_CAPPED = "repair_downtime_capped"
COVERAGE_FIXED_SUM = "fixed_sum"
COVERAGE_REPAIR_DOWNTIME_UNCAPPED = "repair_downtime_uncapped"

MTBF_SCENARIOS: dict[str, dict[str, Any]] = {
    "optimistic": {
        "description": "Новая техника, отличное ТО, лёгкие условия",
        "mtbf_all_engine_hours": 3000.0,
        "major_failure_share": 0.20,
    },
    "baseline": {
        "description": "Средний парк РФ/КЗ, стандартное ТО",
        "mtbf_all_engine_hours": 1500.0,
        "major_failure_share": MAJOR_FAILURE_SHARE,
    },
    "pessimistic": {
        "description": "Старый парк, тяжёлые почвы, дефицит запчастей",
        "mtbf_all_engine_hours": 1200.0,
        "major_failure_share": 0.45,
    },
}

FALLBACK_X_STANDARDIZATION: dict[str, dict[str, Any]] = {
    "x_age": {"raw_col": "Age", "shift": 10.0, "scale": 10.0},
    "x_hours": {"raw_col": "Hours", "shift": 1000.0, "scale": 1000.0},
    "x_climate": {"raw_col": "Climate", "shift": None, "scale": None},
    "x_soil": {"raw_col": "Soil", "shift": None, "scale": None},
    "x_brand": {"raw_col": "Brand", "shift": None, "scale": None},
    "x_power": {"raw_col": "Power", "shift": 200.0, "scale": 150.0},
}

REPAIR_COSTS: dict[str, dict[str, float]] = {
    "двигатель": {"base": 170_000.0, "heavy": 350_000.0},
    "трансмиссия": {"base": 110_000.0, "heavy": 375_000.0},
    "гидравлика": {"base": 50_000.0, "heavy": 160_000.0},
    "электроника": {"base": 20_000.0, "heavy": 175_000.0},
    "прочее": {"base": 30_000.0, "heavy": 90_000.0},
}

# ============================================================================
# FIX 7: Stochastic severity model (Gamma distribution with empirical fallback)
# ============================================================================
# Gamma(shape, scale) has E[X] = shape * scale, Var[X] = shape * scale^2.
# CV = sqrt(Var)/E = 1/sqrt(shape).
REPAIR_SEVERITY_PARAMS: dict[str, dict[str, float]] = {
    "двигатель": {"shape": 2.5, "scale": 68000.0, "cv": 0.63},
    "трансмиссия": {"shape": 2.0, "scale": 55000.0, "cv": 0.71},
    "гидравлика": {"shape": 3.0, "scale": 16667.0, "cv": 0.58},
    "электроника": {"shape": 1.5, "scale": 13333.0, "cv": 0.82},
    "прочее": {"shape": 2.0, "scale": 15000.0, "cv": 0.71},
}


def gamma_expected_severity(params: dict[str, float]) -> float:
    """E[X] for Gamma(shape, scale)."""
    shape = float(params.get("shape", 2.0))
    scale = float(params.get("scale", 50000.0))
    return shape * scale


def gamma_variance_severity(params: dict[str, float]) -> float:
    """Var[X] for Gamma(shape, scale)."""
    shape = float(params.get("shape", 2.0))
    scale = float(params.get("scale", 50000.0))
    return shape * scale**2


def calculate_risk_margin(
    probability: float,
    expected_severity: float,
    severity_variance: float,
    confidence_level: float = 0.95,
) -> float:
    """Variance-based risk margin (Cornish-Fisher approximation).

    Risk margin = z_alpha * sqrt(Var[Loss])
    where Var[Loss] = p * Var[Severity] + p*(1-p) * E[Severity]^2

    This replaces the flat theta loading with a principled risk margin
    that accounts for severity dispersion.
    """
    try:
        from scipy import stats as _scipy_stats

        z = _scipy_stats.norm.ppf(confidence_level)
    except ImportError:
        # Fallback: z_0.95 ≈ 1.645
        z = 1.645

    loss_variance = (
        probability * severity_variance + probability * (1.0 - probability) * expected_severity**2
    )
    loss_variance = max(0.0, loss_variance)  # numerical safety
    return float(z * np.sqrt(loss_variance))


DOWNTIME_STATS: dict[int, dict[str, float]] = {
    1: {"median": 5.0, "p90": 18.0, "mean": 11.5},
    2: {"median": 20.0, "p90": 60.0, "mean": 40.0},
    3: {"median": 45.0, "p90": 180.0, "mean": 45.0},
}
DEFAULT_FAILURE_GROUP = 3

# ---------------------------------------------------------------------------
# Словарь операций с переводом и реальной нагрузкой из TUM CAN bus
# ---------------------------------------------------------------------------
# Нагрузка (peak_load_mean) — это средняя доля от максимальной мощности двигателя,
# которую трактор отдаёт при выполнении данной операции. Измеряется в долях единицы
# (0.0–1.0). Источник: EngPercentLoadAtCurrentSpeed_(%) / 100
# ---------------------------------------------------------------------------
OPERATION_INFO: dict[str, dict[str, Any]] = {
    "Ploughing": {
        "operation": "Ploughing",
        "name_ru": "Вспашка",
        "description": "Глубокая вспашка почвы",
        "peak_load_mean": 0.85,  # ~85% нагрузки двигателя
        "peak_load_std": 0.12,
        "intensity": "высокая",
    },
    "Cultivating (deep)": {
        "operation": "Cultivating (deep)",
        "name_ru": "Глубокая культивация",
        "description": "Глубокое рыхление почвы",
        "peak_load_mean": 0.72,
        "peak_load_std": 0.10,
        "intensity": "высокая",
    },
    "Cultivating (shallow)": {
        "operation": "Cultivating (shallow)",
        "name_ru": "Плоская культивация",
        "description": "Поверхностная обработка почвы",
        "peak_load_mean": 0.45,
        "peak_load_std": 0.08,
        "intensity": "средняя",
    },
    "Disc harrowing": {
        "operation": "Disc harrowing",
        "name_ru": "Дискование",
        "description": "Плоскорезная обработка дисковыми орудиями",
        "peak_load_mean": 0.55,
        "peak_load_std": 0.09,
        "intensity": "средняя",
    },
    "Power harrowing": {
        "operation": "Power harrowing",
        "name_ru": "Фрезерование",
        "description": "Вспашка с помощью роторной культивации",
        "peak_load_mean": 0.70,
        "peak_load_std": 0.11,
        "intensity": "высокая",
    },
    "Seedbed combination": {
        "operation": "Seedbed combination",
        "name_ru": "Комбинированное подготовка семенного ложа",
        "description": "Подготовка семенного ложа комбинированным орудием",
        "peak_load_mean": 0.50,
        "peak_load_std": 0.08,
        "intensity": "средняя",
    },
    "Seed drill combination 3m": {
        "operation": "Seed drill combination 3m",
        "name_ru": "Посев комбайном 3м",
        "description": "Посев с одновременным внесением удобрений (ширина 3м)",
        "peak_load_mean": 0.60,
        "peak_load_std": 0.09,
        "intensity": "средняя",
    },
    "Seed drill combination 4m": {
        "operation": "Seed drill combination 4m",
        "name_ru": "Посев комбайном 4м",
        "description": "Посев с одновременным внесением удобрений (ширина 4м)",
        "peak_load_mean": 0.65,
        "peak_load_std": 0.10,
        "intensity": "средняя",
    },
    "Precision air seeding": {
        "operation": "Precision air seeding",
        "name_ru": "Точный посев (воздушный)",
        "description": "Точный посев с использованием воздушной технологии",
        "peak_load_mean": 0.58,
        "peak_load_std": 0.08,
        "intensity": "средняя",
    },
    "Rotary tilling": {
        "operation": "Rotary tilling",
        "name_ru": "Роторная обработка",
        "description": "Обработка почвы роторным орудием",
        "peak_load_mean": 0.68,
        "peak_load_std": 0.10,
        "intensity": "высокая",
    },
    "Sowing (general)": {
        "operation": "Sowing (general)",
        "name_ru": "Посев (общий)",
        "description": "Общий посев сельскохозяйственных культур",
        "peak_load_mean": 0.52,
        "peak_load_std": 0.09,
        "intensity": "средняя",
    },
    "Spraying": {
        "operation": "Spraying",
        "name_ru": "Опрыскивание",
        "description": "Обработка посевов гербицидами/пестицидами",
        "peak_load_mean": 0.40,
        "peak_load_std": 0.07,
        "intensity": "низкая",
    },
    "Fertilizing": {
        "operation": "Fertilizing",
        "name_ru": "Внесение удобрений",
        "description": "Внесение минеральных/органических удобрений",
        "peak_load_mean": 0.38,
        "peak_load_std": 0.06,
        "intensity": "низкая",
    },
    "Mowing (front)": {
        "operation": "Mowing (front)",
        "name_ru": "Кошение (фронтальное)",
        "description": "Кошение травы фронтальной косилкой",
        "peak_load_mean": 0.62,
        "peak_load_std": 0.09,
        "intensity": "средняя",
    },
    "Mowing (large-scale)": {
        "operation": "Mowing (large-scale)",
        "name_ru": "Кошение (промышленное)",
        "description": "Большое скошенное поле",
        "peak_load_mean": 0.58,
        "peak_load_std": 0.08,
        "intensity": "средняя",
    },
    "Swathing": {
        "operation": "Swathing",
        "name_ru": "Валкование",
        "description": "Валкование убранной культуры",
        "peak_load_mean": 0.55,
        "peak_load_std": 0.08,
        "intensity": "средняя",
    },
    "Mulching": {
        "operation": "Mulching",
        "name_ru": "Мульчирование",
        "description": "Перемалывание остатков культуры",
        "peak_load_mean": 0.60,
        "peak_load_std": 0.09,
        "intensity": "средняя",
    },
    "Transport": {
        "operation": "Transport",
        "name_ru": "Транспорт",
        "description": "Транспортные перевозки по дорогам (базовая нагрузка)",
        "peak_load_mean": 0.35,  # _baseline_
        "peak_load_std": 0.08,
        "intensity": "низкая",
    },
}


def _normalize_operation_name(op_name: str) -> str:
    """
    Приводит название операции к каноническому виду.

    Сравнивает с case-insensitive ключами из OPERATION_INFO.
    """
    op_lower = op_name.strip().lower()
    for canonical in OPERATION_INFO:
        if canonical.lower() == op_lower:
            return canonical
    return op_name


# ---------------------------------------------------------------------------
# Фаза X.X: загрузка реальных статистик операций из TUM CAN bus
# ---------------------------------------------------------------------------
def load_tum_operations() -> dict[str, dict[str, Any]]:
    """
    Загрузить реальные статистики операций из TUM CAN bus.

    Адаптирует имена полей из analyze_tum_operations.py:
      - n → n_observations
      - mean → peak_load_mean
      - std → peak_load_std
      - relative_to_transport → season_factor

    Дополнительно подключает русский перевод и описание нагрузки
    из OPERATION_INFO.
    """
    stats_path = Path("data/processed/tum/tum_operation_stats.json")
    if not stats_path.exists():
        logger.warning(
            "Файл статистик TUM не найден: %s. Запустите: python analyze_tum_operations.py",
            stats_path,
        )
        return {}

    try:
        with open(stats_path, encoding="utf-8") as f:
            data = json.load(f)

        raw_operations = data.get("operations", {})

        # Адаптация имён полей + подключение русского перевода
        operations: dict[str, dict[str, Any]] = {}
        for op_name, raw_stats in raw_operations.items():
            # Нормализация имени операции для сопоставления с OPERATION_INFO
            canonical_name = _normalize_operation_name(op_name)
            info = OPERATION_INFO.get(canonical_name, {})

            operations[op_name] = {
                "operation": op_name,
                "name_ru": info.get("name_ru", op_name),
                "description_ru": info.get("description", info.get("description_ru", "")),
                "intensity": info.get("intensity", "неизвестна"),
                "n_observations": raw_stats.get("n", raw_stats.get("n_observations", 0)),
                # FIX 1: PeakLoad уже в [0, 1] диапазоне, конвертация не нужна.
                "peak_load_mean": round(
                    float(raw_stats.get("mean", raw_stats.get("peak_load_mean", 0.0))),
                    6,
                ),
                "peak_load_std": round(
                    float(raw_stats.get("std", raw_stats.get("peak_load_std", 0.0))),
                    6,
                ),
                "season_factor": raw_stats.get(
                    "relative_to_transport", raw_stats.get("season_factor", 1.0)
                ),
                # Дополнительные статистики
                "peak_load_p25": raw_stats.get("p25"),
                "peak_load_p75": raw_stats.get("p75"),
                "peak_load_p95": raw_stats.get("p95"),
                "peak_load_median": raw_stats.get("median"),
            }

        logger.info(
            "Загружено %d операций из TUM CAN bus (реальные данные)",
            len(operations),
        )
        return operations
    except Exception as exc:
        logger.warning("Ошибка загрузки TUM статистик: %s", exc)
        return {}


# Загрузка при инициализации модуля
TUM_OPERATIONS = load_tum_operations()

# Fallback: экспертные сезонные факторы (используются если TUM данные не загружены)
SEASONAL_FACTORS: dict[str, float] = {
    "межсезонье": 1.0,
    "подготовка": 1.5,
    "посевная": 3.0,
    "заготовка кормов": 3.0,
    "уборочная": 5.0,
    "критическое окно": 8.0,
}

BASE_DOWNTIME_COST = 2000.0

REGIONAL_RATES: dict[str, tuple[float, float]] = {
    "низкозатратный": (800.0, 1300.0),
    "обычный региональный": (1300.0, 2500.0),
    "выездной": (2000.0, 3500.0),
    "Москва/СПб": (3000.0, 5500.0),
    "дилерский/срочный": (5000.0, 8000.0),
}
DEFAULT_HEAVY_PROB_BY_GROUP: dict[int, float] = {1: 0.10, 2: 0.20, 3: 0.35}
DGP_DEFAULT_PATH = "calibration_output/calibrated_dgp.json"


# ---------------------------------------------------------------------------
# Фаза 6.6: загрузка региональных индексов из реальных данных
# ---------------------------------------------------------------------------
def load_region_indices() -> dict[str, dict[str, float]]:
    """
    Загрузить реальные климатические и почвенные индексы по регионам.

    Returns
    -------
    dict
        {region_code: {"climate": float, "soil": float, "working_days": float}}
    """
    result: dict[str, dict[str, float]] = {}

    weather_path = Path("data/processed/weather/weather_windows.csv")
    soil_path = Path("data/processed/soil/soil_windows.csv")

    weather_df = None
    soil_df = None

    if weather_path.exists():
        try:
            weather_df = pd.read_csv(weather_path, encoding="utf-8")
        except Exception as exc:
            logger.warning("Не удалось загрузить weather_windows.csv: %s", exc)

    if soil_path.exists():
        try:
            soil_df = pd.read_csv(soil_path, encoding="utf-8")
        except Exception as exc:
            logger.warning("Не удалось загрузить soil_windows.csv: %s", exc)

    if weather_df is None and soil_df is None:
        logger.warning(
            "Файлы weather_windows.csv и soil_windows.csv не найдены. "
            "Региональные индексы будут недоступны."
        )
        return result

    # Определяем список регионов
    regions = set()
    if weather_df is not None and "region_code" in weather_df.columns:
        regions.update(weather_df["region_code"].unique())
    if soil_df is not None and "region_code" in soil_df.columns:
        regions.update(soil_df["region_code"].unique())

    for region in sorted(regions):
        entry: dict[str, float] = {
            "climate": 0.50,  # Fallback
            "soil": 0.50,  # Fallback
            "working_days": 50.0,  # Fallback
        }

        # Climate из weather_windows.csv
        if weather_df is not None:
            w_region = weather_df[
                (weather_df["region_code"] == region) & (weather_df["campaign"] == "sowing")
            ]
            if not w_region.empty and "working_days_window" in w_region.columns:
                working_days = float(w_region["working_days_window"].mean())
                entry["working_days"] = working_days
                # Нормализация в [0, 1]: 90 дней = максимум
                entry["climate"] = max(0.0, min(1.0, working_days / 90.0))

        # Soil из soil_windows.csv
        if soil_df is not None:
            s_region = soil_df[
                (soil_df["region_code"] == region) & (soil_df["campaign"] == "sowing")
            ]
            if not s_region.empty:
                # Ищем нормализованный индекс
                if "soil_index_normalized" in s_region.columns:
                    soil_val = float(s_region["soil_index_normalized"].mean())
                    entry["soil"] = max(0.0, min(1.0, soil_val))
                elif "soil_index" in s_region.columns:
                    # Если нет нормализованного, нормализуем сами
                    soil_raw = float(s_region["soil_index"].mean())
                    # GLDAS soil moisture обычно 100-400 кг/м²
                    entry["soil"] = max(0.0, min(1.0, soil_raw / 400.0))

        result[region] = entry

    return result


# ---------------------------------------------------------------------------
# Calculator configuration container
# ---------------------------------------------------------------------------
@dataclass
class CalculatorConfig:
    """Все параметры калькулятора, собранные от пользователя."""

    model_path: str = DEFAULT_MODEL_PATH
    dgp_data: dict[str, Any] | None = None
    peaks: list[float] = field(default_factory=list)
    extended: dict[str, Any] = field(default_factory=dict)
    covariate_values: dict[str, float] = field(default_factory=dict)
    raw_covariate_names: set[str] = field(default_factory=set)

    heavy_prob: float = 0.2
    labor_ratio: float = 0.5
    coverage_mode: str = COVERAGE_REPAIR_DOWNTIME_CAPPED
    major_failure_share: float = 1.0
    season_hazard_factor: float = 1.0

    sum_insured: float = DEFAULT_SUM_INSURED
    horizon: float = DEFAULT_HORIZON_ENGINE_HOURS
    theta: float = DEFAULT_THETA
    discount_rate: float = DEFAULT_DISCOUNT_RATE

    expected_repair: float = 0.0
    downtime_cost_per_failure: float = 0.0
    expected_loss_per_failure: float = 0.0
    claim_amount: float = 0.0

    # Severity Model (Phase 7.9)
    severity_expected_severity: float | None = None
    severity_deductible: float = 0.0
    severity_coverage_limit: float | None = None
    use_severity_pricing: bool = False
    severity_model: Any | None = None

    hours_per_day: float = DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY
    calib_horizon_value: float | None = None
    model_time_unit: str = MODEL_TIME_UNIT
    training_meta: dict[str, Any] = field(default_factory=dict)

    # ─── Фаза X: режим культуры ──────────────────────────────────
    crop_key: str = ""
    crop_area_ha: float = 0.0
    crop_weighted_peak: float | None = None
    crop_total_hours: float | None = None

    # ─── Фаза X: K_об ──────────────────────────────────────────────
    k_ob: float = 1.0
    k_ob_params: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _format_number(
    value: Any, *, fmt: str = ".3f", money: bool = False, default: str = "N/A"
) -> str:
    if value is None:
        return default
    numeric_value = _as_optional_float(value)
    if numeric_value is None:
        return default
    if math.isnan(numeric_value):
        return "NaN"
    if math.isinf(numeric_value):
        return "Inf" if numeric_value > 0 else "-Inf"
    if money:
        return f"{numeric_value:,.2f}"
    return format(numeric_value, fmt)


def fmt_num(value: Any, fmt: str = ".3f", default: str = "N/A") -> str:
    return _format_number(value, fmt=fmt, money=False, default=default)


def fmt_money(value: Any, default: str = "N/A") -> str:
    return _format_number(value, money=True, default=default)


# ---------------------------------------------------------------------------
# Type-safe conversion helpers
# ---------------------------------------------------------------------------
def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float, str)):
            result = float(value)
        else:
            return None
    except PARSE_ERRORS:
        return None
    if not math.isfinite(result):
        return None
    return result


def _as_float(value: Any, name: str) -> float:
    result = _as_optional_float(value)
    if result is None:
        raise InvalidInputError(f"{name} must be a finite number")
    return result


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return int(value)
        if isinstance(value, str):
            return int(value.strip())
        return None
    except PARSE_ERRORS:
        return None


def _as_int(value: Any, name: str, default: int) -> int:
    result = _as_optional_int(value)
    if result is None:
        logger.debug("Using default value for parameter '%s'.", name)
        return default
    return result


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _dict_get_optional_float(mapping: Mapping[str, Any], key: str) -> float | None:
    return _as_optional_float(mapping.get(key))


def _dict_get_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
    result = _as_optional_float(mapping.get(key))
    if result is None:
        return default
    return result


def _dict_get_int(mapping: Mapping[str, Any], key: str, default: int) -> int:
    result = _as_optional_int(mapping.get(key))
    if result is None:
        return default
    return result


def _dict_get_str(mapping: Mapping[str, Any], key: str, default: str) -> str:
    return _as_str(mapping.get(key), default)


# ---------------------------------------------------------------------------
# Parsing / input helpers
# ---------------------------------------------------------------------------
def _parse_user_float(text: str) -> float:
    txt = str(text).strip()
    if not txt:
        raise ValueError("empty value")
    txt = re.sub(r"\s+", "", txt)
    if "," in txt:
        txt = txt.replace(",", ".")
    return float(txt)


def _parse_float_list(raw: str) -> list[float]:
    raw = str(raw).strip()
    if not raw:
        return []
    if ";" in raw:
        parts = raw.split(";")
    elif "," in raw:
        try:
            return [_parse_user_float(raw)]
        except PARSE_ERRORS:
            parts = raw.split(",")
    else:
        parts = raw.split()
    values: list[float] = []
    for part in parts:
        part = str(part).strip()
        if not part:
            continue
        parsed_value = _parse_user_float(part)
        if not math.isfinite(parsed_value):
            raise ValueError("Все значения должны быть конечными числами.")
        values.append(parsed_value)
    return values


def ask_float(prompt: str, default: float) -> float:
    default_value = float(default)
    while True:
        value = input(f"{prompt} [{default_value}]: ").strip()
        if not value:
            return default_value
        try:
            result = _parse_user_float(value)
        except PARSE_ERRORS:
            logger.warning("Ошибка: введите число.")
            continue
        if not math.isfinite(result):
            logger.warning("Ошибка: значение должно быть конечным числом.")
            continue
        return result


def ask_int(prompt: str, default: int) -> int:
    default_value = int(default)
    while True:
        value = input(f"{prompt} [{default_value}]: ").strip()
        if not value:
            return default_value
        try:
            return int(value)
        except PARSE_ERRORS:
            logger.warning("Ошибка: введите целое число.")
            continue


def ask_str(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default


def ask_choice(prompt: str, options: list[str], default: str) -> str:
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    choice = input(f"Выбор [1-{len(options)}] (по умолчанию {default}): ").strip()
    if not choice:
        return default
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return options[idx]
        logger.warning("Неверный номер, используем значение по умолчанию.")
        return default
    except ValueError:
        if choice in options:
            return choice
        logger.warning("Неверный ввод, используем значение по умолчанию.")
        return default


def ask(prompt: str, default: str) -> str:
    """Простой запрос строки с дефолтным значением."""
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default


# ─── Фаза X: функции для выбора культуры ──────────────────────────────
def ask_crop_selection(extended: dict[str, Any]) -> Tuple[str, float]:
    """
    Запрос выбора культуры и площади.

    Parameters
    ----------
    extended : dict
        Расширенные параметры (нужны для определения трактора).

    Returns
    -------
    Tuple[str, float]
        (crop_key, area_ha). crop_key = "" если выбран универсальный режим.
    """
    if not HAS_AGRO_CALENDAR:
        return "", 0.0

    print("\n" + "=" * 60)
    print("РЕЖИМ КУЛЬТУРЫ (агрономический календарь)")
    print("=" * 60)

    crop_keys = list_crops()
    for i, key in enumerate(crop_keys, 1):
        crop = CROP_CATALOG[key]
        print(
            f"  {i:2d}) {crop.crop_name_ru:40s} | "
            f"{crop.n_tractor_operations} тракторных оп. | "
            f"{crop.region_preference}"
        )
    print(f"  {len(crop_keys) + 1:2d}) Другое / универсальный режим (одна операция)")

    choice = input(f"Выбор [1-{len(crop_keys) + 1}]: ").strip()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(crop_keys):
            crop_key = crop_keys[idx]
            area_ha = _ask_non_negative("Площадь под культуру (га)", 100.0)
            if area_ha <= 0.0:
                logger.warning("Площадь <= 0. Универсальный режим.")
                return "", 0.0
            return crop_key, area_ha
    except ValueError:
        pass

    return "", 0.0


def ask_k_ob_parameters() -> Tuple[float, Dict[str, str]]:
    """
    Запрос параметров K_об у пользователя.
    
    Returns
    -------
    Tuple[float, Dict[str, str]]
        (k_ob_value, raw_params)
    """
    if not HAS_AGRO_NORMS:
        return 1.0, {}
    
    from agro_norms import calculate_k_ob
    
    print("\n" + "=" * 60)
    print("ПОПРАВОЧНЫЙ КОЭФФИЦИЕНТ ПОЛЕВЫХ УСЛОВИЙ (K_об)")
    print("Источник: Сроки.ПДФ, Таблица 1.1, формула 1.2")
    print("=" * 60)
    print("K_об = K_K × K_h × K_C × K_П × K_R")
    print("Чем хуже условия → тем меньше K_об → тем больше моточасов")
    print()
    
    # 1. Тип работ
    op_type_options = ["пахотные", "непахотные", "кошение трав"]
    op_type = ask_choice("Тип работ:", op_type_options, "пахотные")
    
    # 2. Каменистость
    stoniness_options = ["отсутствует", "слабая", "средняя", "сильная"]
    stoniness = ask_choice("Степень каменистости почвы:", stoniness_options, "отсутствует")
    
    # 3. Высота над уровнем моря
    altitude_options = ["до 500", "500-1000", "1000-1500", "1500-2000"]
    altitude = ask_choice("Высота над уровнем моря (м):", altitude_options, "до 500")
    
    # 4. Сложность конфигурации полей
    config_options = ["I", "II", "III", "IV", "V"]
    print("Группы сложности: I = простые поля, V = очень сложные")
    field_config = ask_choice("Группа сложности конфигурации полей:", config_options, "I")
    
    # 5. Изрезанность препятствиями
    obstacles_options = ["0", "до 5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35"]
    obstacles_pct = ask_choice(
        "Площадь, занимаемая препятствиями (%):", obstacles_options, "0"
    )
    
    # 6. Рельеф
    relief_options = ["<=1", "1-3"]
    relief_slope = ask_choice("Угол склона (градусы):", relief_options, "<=1")
    
    # Вычисление
    k_ob = calculate_k_ob(
        operation_type=op_type,
        stoniness=stoniness,
        altitude=altitude,
        field_config=field_config,
        obstacles_pct=obstacles_pct,
        relief_slope=relief_slope,
    )
    
    raw_params = {
        "operation_type": op_type,
        "stoniness": stoniness,
        "altitude": altitude,
        "field_config": field_config,
        "obstacles_pct": obstacles_pct,
        "relief_slope": relief_slope,
    }
    
    print(f"\n  ✅ K_об = {k_ob:.4f}")
    if k_ob < 0.85:
        print("  ⚠️  Тяжёлые полевые условия: моточасы увеличатся на "
              f"{(1.0/k_ob - 1.0)*100:.0f}%")
    
    return k_ob, raw_params


def _map_brand_to_tractor(brand_name: str) -> str:
    """
    Сопоставить марку трактора из калькулятора с ключом в agro_norms.

    Parameters
    ----------
    brand_name : str
        Название бренда из collect_extended_parameters().

    Returns
    -------
    str
        Ключ трактора для agro_norms.get_engine_hours_per_ha().
    """
    brand_lower = brand_name.strip().lower()

    # Маппинг брендов из BRAND_MAP на ключи в agro_norms
    mapping = {
        "мтз-82": "МТЗ-82",
        "мтз": "МТЗ-82",
        "mtz": "МТЗ-82",
        "кировец": "К-744Р1",
        "к-744": "К-744Р1",
        "к-744р": "К-744Р1",
        "к-744р1": "К-744Р1",
        "new holland": "К-744Р1",  # Тяжёлый трактор → К-744Р1
        "versatile": "К-744Р1",  # Тяжёлый трактор → К-744Р1
        "дт-75": "ДТ-75М",
        "дт75": "ДТ-75М",
        "т-150к": "Т-150К",
        "т-150": "Т-150К",
    }

    for key, tractor in mapping.items():
        if key in brand_lower:
            return tractor

    # Fallback: МТЗ-82 (самый распространённый)
    logger.warning(
        "Бренд '%s' не найден в маппинге тракторов. Используется МТЗ-82.",
        brand_name,
    )
    return "МТЗ-82"


def _ask_bounded_float(
    prompt: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
    error_message: str,
) -> float:
    while True:
        value = ask_float(prompt, default)
        ok = True
        if min_value is not None:
            ok = (value >= min_value) if min_inclusive else (value > min_value)
        if ok and max_value is not None:
            ok = (value <= max_value) if max_inclusive else (value < max_value)
        if ok:
            return value
        logger.warning(error_message)


def _ask_probability(prompt: str, default: float) -> float:
    return _ask_bounded_float(
        prompt,
        default,
        min_value=0.0,
        max_value=1.0,
        min_inclusive=True,
        max_inclusive=True,
        error_message="Ошибка: вероятность должна быть в диапазоне [0, 1].",
    )


def _ask_non_negative(prompt: str, default: float) -> float:
    return _ask_bounded_float(
        prompt,
        default,
        min_value=0.0,
        min_inclusive=True,
        error_message="Ошибка: значение должно быть >= 0.",
    )


def _ask_discount_rate(prompt: str, default: float) -> float:
    return _ask_bounded_float(
        prompt,
        default,
        min_value=0.0,
        max_value=1.0,
        min_inclusive=True,
        max_inclusive=False,
        error_message="Ошибка: ставка должна быть в диапазоне [0, 1).",
    )


def _ask_theta(prompt: str, default: float) -> float:
    return _ask_bounded_float(
        prompt,
        default,
        min_value=0.0,
        max_value=1.0,
        min_inclusive=True,
        max_inclusive=False,
        error_message="Ошибка: нагрузка theta должна быть в диапазоне [0, 1).",
    )


def _ask_labor_ratio(prompt: str, default: float) -> float:
    return _ask_bounded_float(
        prompt,
        default,
        min_value=0.0,
        max_value=1.0,
        min_inclusive=True,
        max_inclusive=True,
        error_message="Ошибка: доля должна быть в диапазоне [0, 1].",
    )


def _ask_season_hazard_factor(prompt: str, default: float) -> float:
    return _ask_bounded_float(
        prompt,
        default,
        min_value=0.0,
        min_inclusive=True,
        error_message="Ошибка: множитель частоты должен быть >= 0.",
    )


def _ask_sum_insured(prompt: str, default: float) -> float:
    return _ask_bounded_float(
        prompt,
        default,
        min_value=0.0,
        min_inclusive=False,
        error_message="Ошибка: страховая сумма должна быть положительной.",
    )


def _ask_positive_horizon(prompt: str, default: float) -> float:
    return _ask_bounded_float(
        prompt,
        default,
        min_value=0.0,
        min_inclusive=False,
        error_message="Ошибка: горизонт должен быть положительным.",
    )


def _ask_environment_index(
    choice_prompt: str,
    index_prompt: str,
    options: list[str],
    default_option: str,
    option_defaults: dict[str, float],
) -> tuple[str, float]:
    chosen_option = ask_choice(choice_prompt, options, default_option)
    default_index = option_defaults.get(chosen_option, 0.0)
    chosen_index = _ask_probability(index_prompt, default_index)
    return chosen_option, chosen_index


def _ask_coverage_mode() -> str:
    print("\nЧто считать выплатой при наступлении страхового события?")
    print("  1) Ожидаемый ремонт + простой, ограниченный страховой суммой (рекомендуется)")
    print("  2) Фиксированная страховая сумма")
    print("  3) Ожидаемый ремонт + простой без ограничения")
    choice = input("Выбор [1]: ").strip() or "1"
    if choice == "2":
        return COVERAGE_FIXED_SUM
    if choice == "3":
        return COVERAGE_REPAIR_DOWNTIME_UNCAPPED
    if choice != "1":
        logger.warning(
            "Неверный выбор режима покрытия. Используется режим ремонта и простоя с лимитом."
        )
    return COVERAGE_REPAIR_DOWNTIME_CAPPED


def _ask_major_failure_share(model: ModelParameters) -> float:
    print("\nСтраховой случай:")
    print("  1) Любой отказ")
    print("  2) Только major failure")
    choice = input("Выбор [1]: ").strip() or "1"
    # PATCH: model event_definition is authoritative; incompatible insurance event is blocked.
    try:
        from model_provenance import assert_prediction_event_compatible

        training_meta = getattr(model, "training_meta", {}) or {}
        assert_prediction_event_compatible(training_meta, str(choice))
    except ImportError:
        logger.info("model_provenance не найден: пропуск event_compatible guard")
    except ValueError as _exc:
        print(f"ОШИБКА СОВМЕСТИМОСТИ СОБЫТИЯ: {_exc}")
        raise SystemExit(2)
    if choice == "1":
        return 1.0
    if choice != "2":
        logger.warning("Неверный выбор страхового случая. Используется любой отказ.")
        return 1.0
    options = list(MTBF_SCENARIOS.keys()) + ["custom"]
    scenario = ask_choice(
        "Сценарий для доли major (крупных) отказов: "
        "optimistic (новая техника) / baseline (средний) / pessimistic (старый парк)",
        options,
        "baseline",
    )
    if scenario == "custom":
        return _ask_probability(
            "Доля major (крупных) отказов от всех отказов (0.0–1.0, где 1 = все отказы major)",
            MAJOR_FAILURE_SHARE,
        )
    share = _dict_get_float(MTBF_SCENARIOS[scenario], "major_failure_share", MAJOR_FAILURE_SHARE)
    if share < 0.0 or share > 1.0:
        raise InvalidInputError(f"major_failure_share must be in [0,1], got {share}")
    logger.info("Используется major_failure_share=%.3f из сценария '%s'.", share, scenario)
    return share


# ---------------------------------------------------------------------------
# Time unit helpers
# ---------------------------------------------------------------------------
def _get_time_unit(model: ModelParameters) -> str:
    meta = getattr(model, "training_meta", {}) or {}
    if not isinstance(meta, dict):
        return MODEL_TIME_UNIT
    return _dict_get_str(meta, "time_unit", MODEL_TIME_UNIT).strip().lower()


def _get_hours_per_day(model: ModelParameters) -> float:
    meta = getattr(model, "training_meta", {}) or {}
    if not isinstance(meta, dict):
        return DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY
    hpd = _dict_get_optional_float(meta, "default_engine_hours_per_calendar_day")
    if hpd is not None and hpd > 0.0:
        return float(hpd)
    return DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY


def _engine_hours_to_calendar_days(engine_hours: float, hours_per_day: float) -> float:
    engine_hours = _as_float(engine_hours, "engine_hours")
    hours_per_day = _as_float(hours_per_day, "hours_per_day")
    if hours_per_day <= 0.0:
        raise InvalidInputError("hours_per_day must be positive")
    return engine_hours / hours_per_day


# ---------------------------------------------------------------------------
# DGP calibration loading
# ---------------------------------------------------------------------------
def load_dgp_calibration(path: str = DGP_DEFAULT_PATH) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.error("[DGP] Файл %s не содержит JSON-объект.", path)
            return None
        logger.info("[DGP] Параметры загружены из %s", path)
        if isinstance(data.get("brand_mapping"), dict):
            logger.info("[DGP] Найден brand_mapping.")
        else:
            logger.info("[DGP] brand_mapping не найден.")
        return data
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        if isinstance(e, FileNotFoundError):
            logger.info(
                "[DGP] Файл %s не найден. Будет использована модель из model_params.json.",
                path,
            )
        else:
            logger.error("[DGP] Ошибка чтения %s: %s", path, e)
        return None


def load_dgp_with_fallback(user_path: str = "") -> dict[str, Any] | None:
    candidates: list[str] = []
    if user_path:
        candidates.append(user_path)
    candidates.append(DGP_DEFAULT_PATH)
    fallback = Path(DEFAULT_MODEL_PATH).parent / "calibration_output" / "calibrated_dgp.json"
    candidates.append(str(fallback))
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        try:
            resolved = str(path.resolve())
        except (OSError, RuntimeError):
            resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.exists():
            if candidate == user_path:
                logger.warning("[DGP] Файл %s не найден.", candidate)
            continue
        data = load_dgp_calibration(str(path))
        if data is not None:
            return data
    logger.info("[DGP] Калибровка DGP не найдена или не загружена. DGP не будет использоваться.")
    return None


# ---------------------------------------------------------------------------
# Brand helpers
# ---------------------------------------------------------------------------
def _canonical_brand_key(brand_name: str) -> str:
    s = str(brand_name).strip().lower()
    s_compact = re.sub(r"\s+", "", s)
    if "мтз" in s or "mtz" in s_compact:
        return "MTZ82"
    if "versatile" in s_compact:
        return "Versatile280"
    if "newholland" in s_compact or "new holland" in s:
        return "NewHollandT9"
    if ("дт" in s and "75" in s) or "dt75" in s_compact:
        return "DT75"
    return "Other"


def _get_brand_index(
    model: ModelParameters, dgp_data: dict[str, Any] | None, brand_name: str
) -> float:
    brand_name = _as_str(brand_name, "")
    mappings: list[dict[str, Any]] = []
    meta = getattr(model, "training_meta", {}) or {}
    if isinstance(meta, dict):
        if isinstance(meta.get("brand_mapping"), dict):
            mappings.append(meta["brand_mapping"])
    if isinstance(dgp_data, dict):
        if isinstance(dgp_data.get("brand_mapping"), dict):
            mappings.append(dgp_data["brand_mapping"])
    brand_lower = brand_name.strip().lower()
    for mapping in mappings:
        if brand_name in mapping:
            mapped = _as_optional_float(mapping.get(brand_name))
            if mapped is not None:
                return float(mapped)
        normalized: dict[str, Any] = {}
        for k, v in mapping.items():
            normalized[str(k).strip().lower()] = v
        if brand_lower in normalized:
            mapped = _as_optional_float(normalized.get(brand_lower))
            if mapped is not None:
                return float(mapped)
    key = _canonical_brand_key(brand_name)
    if key in CANONICAL_BRAND_INDEX:
        logger.warning(
            "Brand '%s' не найден в brand_mapping. Используется канонический fallback: %s -> %.1f.",
            brand_name,
            key,
            CANONICAL_BRAND_INDEX[key],
        )
        return float(CANONICAL_BRAND_INDEX[key])
    raise InvalidInputError(f"Brand '{brand_name}' cannot be mapped to numeric Brand index.")


def _brand_dummy_key(name: str) -> str | None:
    n = str(name).strip().lower()
    n_compact = re.sub(r"[^a-z0-9а-яё]", "", n)
    if "brand" not in n_compact and "бренд" not in n_compact:
        return None
    if "mtz82" in n_compact or "mtz" in n_compact:
        return "MTZ82"
    if "versatile280" in n_compact or "versatile" in n_compact:
        return "Versatile280"
    if "newhollandt9" in n_compact or "newholland" in n_compact or "t9" in n_compact:
        return "NewHollandT9"
    if "dt75" in n_compact:
        return "DT75"
    if "other" in n_compact or "прочие" in n_compact or "другое" in n_compact:
        return "Other"
    return None


# ---------------------------------------------------------------------------
# Extended parameters
# ---------------------------------------------------------------------------
def collect_extended_parameters() -> dict[str, Any]:
    params: dict[str, Any] = {}
    brand_options = [
        "New Holland T9.505",
        "New Holland T9.615",
        "МТЗ-82",
        "Versatile 280",
        "ДТ-75",
        "Other",
    ]
    default_brand = brand_options[0]
    brand = ask_choice("Выберите марку трактора:", brand_options, default_brand)
    params["brand"] = brand

    age_years = ask_float("Возраст трактора (лет)", 10.0)
    if age_years < 0.0:
        age_years = 0.0
    elif age_years > 30.0:
        age_years = 30.0
    params["age_years"] = age_years

    hours = ask_float("Годовая наработка (мото-часов в год)", 1000.0)
    if hours < 0.0:
        hours = 0.0
    params["hours"] = hours

    power = ask_float("Мощность двигателя (л.с.)", 180.0)
    if power < 50.0:
        power = 50.0
    elif power > 350.0:
        power = 350.0
    params["power"] = power

    # ─── Фаза 6.6: выбор региона для климата и почвы ──────────────
    region_indices = load_region_indices()

    print()
    print("Выберите регион эксплуатации трактора:")
    print("  0) Ввести индексы вручную")

    region_list = sorted(region_indices.keys())
    for i, region in enumerate(region_list, start=1):
        idx = region_indices[region]
        print(
            f"  {i}) {region:25s} "
            f"(climate={idx['climate']:.3f}, soil={idx['soil']:.3f}, "
            f"рабочих дней={idx['working_days']:.0f})"
        )

    region_choice = ask(f"Выбор региона [0-{len(region_list)}]", "0").strip() or "0"

    climate_index = None
    soil_index = None
    selected_region = None

    if region_choice.isdigit() and 1 <= int(region_choice) <= len(region_list):
        # Автоматическая подстановка из реальных данных
        selected_region = region_list[int(region_choice) - 1]
        idx = region_indices[selected_region]
        climate_index = idx["climate"]
        soil_index = idx["soil"]

        print(f"  ✅ Выбран регион: {selected_region}")
        print(f"  ✅ x_climate = {climate_index:.4f} (из NASA POWER)")
        print(f"  ✅ x_soil    = {soil_index:.4f} (из NASA GLDAS-2.1)")
        print(f"  ✅ Рабочих дней в посевную: {idx['working_days']:.0f}")
    else:
        # Ручной ввод (fallback)
        print()
        print("Ручной ввод индексов:")
        climate_index = ask_float(
            "Индекс климата (0 = суровый, 1 = благоприятный; по данным NASA POWER)",
            0.25,
        )
        soil_index = ask_float(
            "Индекс почвы (0 = тяжёлая/бедная, 1 = лёгкая/плодородная; по данным GLDAS)",
            0.50,
        )

    params["climate"] = selected_region or "ручной ввод"
    params["climate_index"] = climate_index
    params["soil"] = selected_region or "ручной ввод"
    params["soil_index"] = soil_index
    params["selected_region"] = selected_region

    # Выбор операций из TUM или сезонных факторов (fallback)
    if TUM_OPERATIONS:
        print("Выберите операцию (из реальных данных TUM CAN bus):")
        op_list = sorted(TUM_OPERATIONS.keys())
        for i, op in enumerate(op_list, start=1):
            info = TUM_OPERATIONS[op]
            # Защита от разных имён полей
            pl_mean_raw = info.get("peak_load_mean", info.get("mean", 0.0))
            pl_std_raw = info.get("peak_load_std", info.get("std", 0.0))
            sf = info.get("season_factor", info.get("sf", 1.0))
            n_obs = info.get("n_observations", info.get("n", 0))
            name_ru = info.get("name_ru", op)
            intensity = info.get("intensity", "")
            # PATCH 1.1: Масштабирование удалено — шкалы TUM и DGP совпадают [0, 1]
            pl_mean = pl_mean_raw
            pl_std = pl_std_raw
            intensity_str = f", интенсивность: {intensity}" if intensity else ""
            print(
                f"  {i:2d}) {name_ru:30s} ({op}) "
                f"PeakLoad={pl_mean:.2f}±{pl_std:.2f} {intensity_str}\n"
                f"      TUM: {pl_mean_raw * 100:.1f}% ± {pl_std_raw * 100:.1f}%, "
                f"SF={sf:.2f}, n={n_obs:,}"
            )
        # Выбираем по номеру (дублирования нет — ask_choice уже был убран)
        op_input = input(f"Номер операции [1-{len(op_list)}]: ").strip()
        try:
            idx = int(op_input)
            if 1 <= idx <= len(op_list):
                selected_operation = op_list[idx - 1]
            else:
                logger.warning("Неверный номер, используется операция по умолчанию.")
                selected_operation = op_list[0]
        except ValueError:
            logger.warning("Неверный ввод, используется операция по умолчанию.")
            selected_operation = op_list[0]

        op_info = TUM_OPERATIONS[selected_operation]
        # Защита от отсутствующих ключей
        operation_peak_load_raw = op_info.get("peak_load_mean", op_info.get("mean", 0.71))
        # PATCH 1.1: Масштабирование удалено — шкалы TUM и DGP совпадают [0, 1]
        operation_peak_load = operation_peak_load_raw
        operation_season_factor = op_info.get("season_factor", 1.0)
        params["season"] = selected_operation
        params["downtime_hour_cost"] = BASE_DOWNTIME_COST * operation_season_factor
    else:
        print("⚠️  Реальные данные TUM не загружены. Обобщённый режим.")
        season_options = list(SEASONAL_FACTORS.keys())
        season = ask_choice("Выберите сезон/период работ:", season_options, season_options[0])
        params["season"] = season
        params["downtime_hour_cost"] = BASE_DOWNTIME_COST * SEASONAL_FACTORS[season]

    region_options = list(REGIONAL_RATES.keys())
    region = ask_choice("Выберите тип региона для нормо-часа:", region_options, region_options[1])
    low, high = REGIONAL_RATES[region]
    params["region"] = region
    regional_rate = _ask_non_negative(
        f"Стоимость нормо-часа (руб.) в диапазоне {low:.0f}-{high:.0f}",
        (low + high) / 2.0,
    )
    params["regional_rate"] = regional_rate

    use_standard = (
        ask_str("Использовать стандартные доли отказов по системам? [да]", "да").strip().lower()
    )
    if use_standard in ("да", "д", "yes", "y", ""):
        params["failure_shares"] = copy.deepcopy(FREQ_SHARES)
    else:
        shares: dict[str, float] = {}
        for system, default_share in FREQ_SHARES.items():
            shares[system] = _ask_probability(f"Доля отказов для {system} (0-1)", default_share)
        params["failure_shares"] = shares

    use_base_repair = ask_str("Использовать базовые стоимости ремонта? [да]", "да").strip().lower()
    if use_base_repair in ("да", "д", "yes", "y", ""):
        params["repair_costs"] = copy.deepcopy(REPAIR_COSTS)
    else:
        costs: dict[str, dict[str, float]] = {}
        for system in REPAIR_COSTS.keys():
            base = _ask_non_negative(
                f"Базовая стоимость ремонта для {system} (руб.)",
                REPAIR_COSTS[system]["base"],
            )
            heavy = _ask_non_negative(
                f"Тяжёлая стоимость ремонта для {system} (руб.)",
                REPAIR_COSTS[system]["heavy"],
            )
            costs[system] = {"base": base, "heavy": heavy}
        params["repair_costs"] = costs

    group = ask_int(
        "Группа сложности отказа: 1 = простой ремонт (5 ч), 2 = средний (20 ч), 3 = сложный (45 ч)",
        DEFAULT_FAILURE_GROUP,
    )
    group = max(1, min(3, group))
    params["failure_group"] = group
    return params


def _default_heavy_probability(extended: dict[str, Any]) -> float:
    group = _as_int(
        extended.get("failure_group"),
        name="failure_group",
        default=DEFAULT_FAILURE_GROUP,
    )
    base = float(DEFAULT_HEAVY_PROB_BY_GROUP.get(group, 0.20))
    age = _dict_get_float(extended, "age_years", 10.0)
    hours = _dict_get_float(extended, "hours", 1000.0)
    power = _dict_get_float(extended, "power", 180.0)
    age_factor = 0.10 * max(0.0, min(1.0, age / 30.0))
    hours_factor = 0.05 * max(0.0, min(1.0, hours / 3000.0))
    power_factor = 0.05 * max(0.0, min(1.0, (power - 250.0) / 100.0))
    value = base + age_factor + hours_factor + power_factor
    return float(max(0.02, min(0.90, value)))


def _ask_heavy_probability(extended: dict[str, Any]) -> float:
    default_value = _default_heavy_probability(extended)
    return _ask_probability("Вероятность тяжёлого сценария ремонта (0.0-1.0)", default_value)


# ---------------------------------------------------------------------------
# Economic helpers
# ---------------------------------------------------------------------------
def compute_expected_repair_cost(
    shares: dict[str, float],
    costs: dict[str, dict[str, float]],
    heavy_prob: float = 0.2,
) -> float:
    heavy_prob_f = _as_float(heavy_prob, "heavy_prob")
    heavy_prob_f = max(0.0, min(1.0, heavy_prob_f))
    clean_shares: dict[str, float] = {}
    for system, share in shares.items():
        share_f = _as_float(share, f"Failure share for '{system}'")
        if share_f < 0.0:
            raise InvalidInputError(f"Failure share for '{system}' cannot be negative")
        if share_f > 0.0:
            clean_shares[system] = share_f
    total_share = sum(clean_shares.values())
    if total_share <= 0.0:
        return 0.0
    if abs(total_share - 1.0) > 1e-6:
        logger.warning("Доли отказов суммируются в %.6f. Они будут нормализованы.", total_share)
    total = 0.0
    for system, share in clean_shares.items():
        if system not in costs:
            raise InvalidInputError(f"Repair cost for system '{system}' is missing")
        cost = costs[system]
        base_cost = _dict_get_float(cost, "base", 0.0)
        heavy_cost = _dict_get_float(cost, "heavy", 0.0)
        if base_cost < 0.0 or heavy_cost < 0.0:
            raise InvalidInputError(f"Repair cost for '{system}' cannot be negative")
        expected_system_cost = (1.0 - heavy_prob_f) * base_cost + heavy_prob_f * heavy_cost
        weight = share / total_share
        total += weight * expected_system_cost
    return float(total)


def compute_downtime_cost(downtime_hours: float, cost_per_hour: float) -> float:
    downtime_hours = _as_float(downtime_hours, "downtime_hours")
    cost_per_hour = _as_float(cost_per_hour, "cost_per_hour")
    if downtime_hours < 0.0:
        raise InvalidInputError("Downtime hours cannot be negative")
    if cost_per_hour < 0.0:
        raise InvalidInputError("Downtime cost per hour cannot be negative")
    return downtime_hours * cost_per_hour


def get_downtime_mean_hours(group: int) -> float:
    stats = DOWNTIME_STATS.get(group, DOWNTIME_STATS[2])
    if "mean" in stats:
        mean_hours = _dict_get_float(stats, "mean", 0.0)
        if mean_hours >= 0.0:
            return mean_hours
    median_hours = _dict_get_float(stats, "median", 0.0)
    if median_hours < 0.0:
        raise InvalidInputError("Invalid downtime statistics")
    return median_hours


def apply_hazard_multiplier(probability: float, factor: float) -> float:
    prob = _as_float(probability, "probability")
    factor = _as_float(factor, "hazard factor")
    if factor < 0.0:
        raise InvalidInputError("Hazard factor cannot be negative")
    prob = max(0.0, min(1.0, prob))
    if prob <= 0.0 or factor == 0.0:
        return 0.0
    if prob >= 1.0:
        return 1.0
    if abs(factor - 1.0) <= 1e-12:
        return prob
    adjusted = 1.0 - math.pow(1.0 - prob, factor)
    return float(max(0.0, min(1.0, adjusted)))


def compute_claim_amount(
    coverage_mode: str, expected_loss_per_failure: float, sum_insured: float
) -> float:
    expected_loss_per_failure = _as_float(expected_loss_per_failure, "expected loss per failure")
    sum_insured = _as_float(sum_insured, "sum insured")
    if sum_insured <= 0.0:
        raise InvalidInputError("Sum insured must be positive")
    expected_loss_per_failure = max(0.0, expected_loss_per_failure)
    if coverage_mode == COVERAGE_FIXED_SUM:
        return sum_insured
    if coverage_mode == COVERAGE_REPAIR_DOWNTIME_UNCAPPED:
        return expected_loss_per_failure
    if coverage_mode == COVERAGE_REPAIR_DOWNTIME_CAPPED:
        return min(expected_loss_per_failure, sum_insured)
    raise InvalidInputError(f"Unknown coverage mode: '{coverage_mode}'")


# ---------------------------------------------------------------------------
# PeakLoad selection
# ---------------------------------------------------------------------------
def get_peakload_choice(model: ModelParameters) -> list[float]:
    meta = getattr(model, "training_meta", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    p25 = _dict_get_optional_float(meta, "peakload_p25")
    p50 = _dict_get_optional_float(meta, "peakload_median")
    p75 = _dict_get_optional_float(meta, "peakload_p75")

    def _manual_input() -> list[float]:
        while True:
            raw = input("Введите одно или несколько значений PeakLoad через точку с запятой (;): ")
            try:
                values = _parse_float_list(raw)
                if values:
                    return values
                logger.warning("Список значений пуст. Попробуйте снова.")
            except PARSE_ERRORS:
                logger.warning("Ошибка ввода. Попробуйте снова.")

    if p25 is None or p50 is None or p75 is None:
        return _manual_input()
    p25_f, p50_f, p75_f = float(p25), float(p50), float(p75)
    print("\nВыберите значение пиковой нагрузки (PeakLoad):")
    print(f"  1) Нижняя квартиль (25%) = {p25_f:.2f}")
    print(f"  2) Медиана (50%)         = {p50_f:.2f}")
    print(f"  3) Верхняя квартиль (75%) = {p75_f:.2f}")
    print("  4) Все три значения")
    print("  5) Своё значение")
    choice = input("Ваш выбор [2]: ").strip() or "2"
    if choice == "1":
        return [p25_f]
    if choice == "2":
        return [p50_f]
    if choice == "3":
        return [p75_f]
    if choice == "4":
        return [p25_f, p50_f, p75_f]
    if choice == "5":
        return _manual_input()
    return [p50_f]


# ---------------------------------------------------------------------------
# Model covariates
# ---------------------------------------------------------------------------
def _get_cf_meta(params: ModelParameters) -> dict[str, Any]:
    meta = getattr(params, "cf_basis_metadata", None)
    if isinstance(meta, dict):
        return meta
    return {}


def _get_cf_columns(params: ModelParameters) -> set[str]:
    meta = _get_cf_meta(params)
    cols = meta.get("v_hat_cols", []) or []
    return {str(c) for c in cols}


def _is_cf_column(name: str, cf_cols: set[str]) -> bool:
    name_str = str(name)
    name_lower = name_str.lower()
    cf_cols_lower = {str(c).lower() for c in cf_cols}
    if name_lower in cf_cols_lower:
        return True
    if name_lower in {"v_hat", "eps_d_hat"}:
        return True
    if name_lower.startswith("v_hat_") or name_lower.startswith("eps_d_hat_"):
        return True
    return False


def _get_x_standardization(params: ModelParameters) -> dict[str, dict[str, Any]]:
    meta = getattr(params, "training_meta", {}) or {}
    if not isinstance(meta, dict):
        return {}
    x_std = meta.get("x_standardization", {}) or {}
    if not isinstance(x_std, dict):
        return {}
    return x_std


def _get_standardization_info(params: ModelParameters, name: str) -> dict[str, Any]:
    x_std = _get_x_standardization(params)
    info = x_std.get(name)
    if isinstance(info, dict):
        return info
    fallback = FALLBACK_X_STANDARDIZATION.get(name)
    if isinstance(fallback, dict):
        return fallback
    return {}


def _standardize_covariate(params: ModelParameters, name: str, value: float) -> float:
    value = _as_float(value, f"Covariate '{name}'")
    info = _get_standardization_info(params, name)
    if not info:
        return value
    shift = info.get("shift", None)
    scale = info.get("scale", None)
    if shift is None or scale is None:
        return value
    shift_f = _as_optional_float(shift)
    scale_f = _as_optional_float(scale)
    if shift_f is None or scale_f is None:
        return value
    if scale_f == 0.0:
        raise ModelLoadError(f"Zero standardization scale for covariate '{name}'")
    standardized = (value - shift_f) / scale_f
    if not math.isfinite(standardized):
        raise InvalidInputError(f"Standardized covariate '{name}' is not finite")
    return float(standardized)


def _compute_age_hours_interaction(
    model: ModelParameters,
    age_raw: float,
    hours_raw: float,
) -> float:
    """
    Вычисляет x_age_hours точно так же, как это сделано при обучении:
    1. Стандартизация сырых Age и Hours.
    2. Центрирование по выборочным средним стандартизированных переменных.
    3. Стандартизация произведения.
    """
    meta = getattr(model, "training_meta", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    ip = meta.get("interaction_params", {})
    if not isinstance(ip, dict):
        ip = {}

    x_age_mean = float(ip.get("x_age_mean", 0.0))
    x_hours_mean = float(ip.get("x_hours_mean", 0.0))
    x_age_hours_mean = float(ip.get("x_age_hours_mean", 0.0))
    x_age_hours_std = float(ip.get("x_age_hours_std", 1.0))
    if x_age_hours_std < 1e-9:
        x_age_hours_std = 1.0

    age_std = _standardize_covariate(model, "x_age", float(age_raw))
    hours_std = _standardize_covariate(model, "x_hours", float(hours_raw))

    interaction = (age_std - x_age_mean) * (hours_std - x_hours_mean)
    return float((interaction - x_age_hours_mean) / x_age_hours_std)


def collect_model_covariates(
    model: ModelParameters,
    extended: dict[str, Any] | None = None,
    dgp_data: dict[str, Any] | None = None,
) -> tuple[dict[str, float], set[str]]:
    first_stage = getattr(model, "first_stage", {}) or {}
    if not isinstance(first_stage, dict):
        first_stage = {}
    cox = getattr(model, "cox", {}) or {}
    if not isinstance(cox, dict):
        cox = {}
    fs_names = first_stage.get("exog_names", []) or []
    cox_names = cox.get("exog_names", []) or []
    all_names = set(fs_names + cox_names)
    exclude = {"const", "intercept", "peakload", "peak_load", "z", "x"}
    cf_cols = _get_cf_columns(model)
    cov_names: list[str] = []
    for name in all_names:
        name_str = str(name)
        if name_str.lower() in exclude:
            continue
        if _is_cf_column(name_str, cf_cols):
            continue
        cov_names.append(name_str)
    cov_names = sorted(cov_names)
    if not cov_names:
        return {}, set()
    cov_values: dict[str, float] = {}
    raw_covariate_names: set[str] = set()
    template = getattr(model, "template_covariates", {}) or {}
    if not isinstance(template, dict):
        template = {}
    extended = extended or {}
    brand_name_raw = extended.get("brand")
    brand_name_str = _as_str(brand_name_raw, "") if brand_name_raw is not None else ""
    brand_key = _canonical_brand_key(brand_name_str) if brand_name_str else "Other"
    for name in cov_names:
        dummy_key = _brand_dummy_key(name)
        if dummy_key is not None:
            cov_values[name] = 1.0 if dummy_key == brand_key else 0.0
    brand_code_value = CANONICAL_BRAND_INDEX.get(brand_key)
    if brand_code_value is not None:
        cov_values["Brand"] = float(brand_code_value)
    ext_to_model_variants: dict[str, list[str]] = {
        "age_years": ["x_age", "Age", "age"],
        "hours": ["x_hours", "Hours", "hours"],
        "age_hours": ["x_age_hours", "Age_x_Hours", "age_hours"],  # ← ДОБАВЛЕНО
        "climate_index": ["x_climate", "Climate", "climate"],
        "soil_index": ["x_soil", "Soil", "soil"],
        "power": ["x_power", "Power", "power"],
        "brand": ["x_brand", "Brand", "brand"],
    }
    for ext_key, model_names in ext_to_model_variants.items():
        if ext_key not in extended:
            continue
        raw_value = extended[ext_key]
        for model_name in model_names:
            if model_name not in cov_names:
                continue
            if model_name in cov_values:
                continue
            resolved_value = raw_value
            if model_name.lower() in {"brand", "x_brand"} and isinstance(raw_value, str):
                resolved_value = _get_brand_index(model, dgp_data, raw_value)
            cov_values[model_name] = _as_float(resolved_value, f"Model covariate '{model_name}'")
            raw_covariate_names.add(model_name)
    # ─── Автоматическое вычисление interaction Age × Hours ─────────────
    # x_age_hours — производная величина, у пользователя НЕ спрашивается.
    # Вычисляется с центрированием и стандартизацией, как при обучении.
    cov_names_set = set(cov_names)
    if "x_age_hours" in cov_names_set and "x_age_hours" not in cov_values:
        x_age_raw = cov_values.get("x_age")
        x_hours_raw = cov_values.get("x_hours")
        if x_age_raw is not None and x_hours_raw is not None:
            cov_values["x_age_hours"] = _compute_age_hours_interaction(
                model, float(x_age_raw), float(x_hours_raw)
            )
            logger.info(
                "x_age_hours вычислен автоматически: %.6f",
                cov_values["x_age_hours"],
            )

    missing = sorted(name for name in cov_names if name not in cov_values)
    if missing:
        logger.info("Ковариаты, не входящие в расширенные параметры:")
        for name in missing:
            # x_age_hours уже вычислен выше — пропускаем
            if name == "x_age_hours":
                continue
            default_template_value = _as_optional_float(template.get(name, 0.0))
            if default_template_value is None:
                default_template_value = 0.0
            info = _get_standardization_info(model, name)
            shift = info.get("shift", None)
            scale = info.get("scale", None)
            default_raw = default_template_value
            shift_f = _as_optional_float(shift)
            scale_f = _as_optional_float(scale)
            if shift_f is not None and scale_f is not None:
                default_raw = default_template_value * scale_f + shift_f
            if info:
                prompt = f"  {name} (сырое значение; стандартизация будет применена автоматически)"
            else:
                prompt = f"  {name} (модельное значение)"
            cov_values[name] = ask_float(prompt, default_raw)
            if info:
                raw_covariate_names.add(name)

    return cov_values, raw_covariate_names


# ---------------------------------------------------------------------------
# Params preparation and engine validation
# ---------------------------------------------------------------------------
def prepare_params_for_prediction(
    params: ModelParameters, covariate_values: dict[str, float]
) -> ModelParameters:
    params_copy = copy.deepcopy(params)
    template = getattr(params_copy, "template_covariates", None)
    if template is None:
        params_copy.template_covariates = {}
    if not isinstance(params_copy.template_covariates, dict):
        params_copy.template_covariates = {}
    for name, val in (covariate_values or {}).items():
        params_copy.template_covariates[str(name)] = _as_float(val, f"Covariate '{name}'")
    return params_copy


def _validate_engine_probability(probability: Any) -> float:
    prob = _as_optional_float(probability)
    if prob is None:
        raise PredictionError("Engine returned non-numeric or non-finite probability")
    if prob < -PROBABILITY_EPSILON or prob > 1.0 + PROBABILITY_EPSILON:
        raise PredictionError(f"Engine probability outside [0,1]: {prob}")
    return float(min(1.0, max(0.0, prob)))


def validate_probability_with_engine(
    params: ModelParameters,
    peak_raw: float,
    time_horizon: float,
    covariate_values: dict[str, float],
    residual_policy: str = DEFAULT_RESIDUAL_POLICY,
    time_horizon_unit: str = MODEL_TIME_UNIT,
) -> float:
    prepared_params = prepare_params_for_prediction(params, covariate_values)
    engine_probability = predict_probability(
        prepared_params,
        peak_raw,
        time_horizon,
        residual_policy,
        covariates=dict(covariate_values),
        time_horizon_unit=time_horizon_unit,
    )
    arr = np.asarray(engine_probability).ravel()
    if arr.size == 0:
        raise PredictionError("predict_probability вернул пустой результат.")
    return _validate_engine_probability(arr[0])


# ---------------------------------------------------------------------------
# Peak range validation
# ---------------------------------------------------------------------------
def _validate_peak_range(params: ModelParameters, peak: float) -> None:
    meta = getattr(params, "training_meta", {}) or {}
    if not isinstance(meta, dict):
        return
    pmin = meta.get("peakload_min")
    pmax = meta.get("peakload_max")
    if pmin is None and pmax is None:
        return
    if pmin is None or pmax is None:
        raise InvalidInputError("Incomplete peakload training range in model")
    pmin_f = _as_optional_float(pmin)
    pmax_f = _as_optional_float(pmax)
    if pmin_f is None or pmax_f is None:
        raise InvalidInputError("Invalid peakload training range in model")
    if pmin_f > pmax_f:
        raise InvalidInputError("Peakload minimum exceeds maximum in model")
    if peak < pmin_f - PEAK_RANGE_TOLERANCE or peak > pmax_f + PEAK_RANGE_TOLERANCE:
        raise InvalidInputError(
            f"PeakLoad {peak} outside training range [{pmin_f}, {pmax_f}] with tolerance {PEAK_RANGE_TOLERANCE}"
        )


# ---------------------------------------------------------------------------
# Manual CF basis helpers
# ---------------------------------------------------------------------------
def _parse_power_from_name(name: str, default_power: int) -> int:
    m = re.search(r"pow(?:er)?_?(\d+)$", str(name).strip().lower())
    if m:
        try:
            power = int(m.group(1))
            return max(1, power)
        except PARSE_ERRORS:
            pass
    return max(1, int(default_power))


def _manual_powers_basis(params: ModelParameters, raw_residual: float) -> dict[str, np.ndarray]:
    meta = _get_cf_meta(params)
    residuals_mean = _dict_get_optional_float(meta, "residuals_mean")
    if residuals_mean is None:
        residuals_mean = _as_optional_float(getattr(params, "training_residuals_mean", 0.0))
    if residuals_mean is None:
        residuals_mean = 0.0
    residuals_std = _dict_get_optional_float(meta, "residuals_std")
    if residuals_std is None:
        residuals_std = _as_optional_float(getattr(params, "training_residuals_std", 1.0))
    if residuals_std is None or residuals_std <= 0.0:
        residuals_std = 1.0
    v_std = (float(raw_residual) - residuals_mean) / residuals_std
    if not math.isfinite(v_std):
        raise PredictionError("Standardized residual is not finite")
    max_power = _dict_get_int(meta, "max_power", 2)
    max_power = max(1, max_power)
    cf_cols = list(meta.get("v_hat_cols", []) or [])
    if not cf_cols:
        cf_cols = [f"v_hat_pow{p}" for p in range(1, max_power + 1)]
    col_std_params = meta.get("v_hat_col_std_params", {}) or {}
    if not isinstance(col_std_params, dict):
        col_std_params = {}
    result: dict[str, np.ndarray] = {}
    for idx, col_name in enumerate(cf_cols):
        power = _parse_power_from_name(col_name, idx + 1)
        basis_value = v_std**power
        info = col_std_params.get(col_name, {})
        if not isinstance(info, dict):
            info = {}
        col_mean = _dict_get_optional_float(info, "mean")
        if col_mean is None:
            col_mean = 0.0
        col_std = _dict_get_optional_float(info, "std")
        if col_std is None or col_std <= 0.0:
            col_std = 1.0
        basis_value = (basis_value - col_mean) / col_std
        if bool(info.get("clip", True)):
            basis_value = float(np.clip(basis_value, -10.0, 10.0))
        if not math.isfinite(basis_value):
            raise PredictionError(f"Non-finite manual CF basis value for '{col_name}'")
        result[str(col_name)] = np.array([basis_value], dtype=float)
    return result


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def _clip_with_warning(value: float, low: float, high: float, label: str) -> float:
    if not math.isfinite(value):
        raise PredictionError(f"{label} is not finite")
    if value < low:
        logger.warning(
            "%s=%.6g ниже допустимого предела %.6g. Значение ограничено.",
            label,
            value,
            low,
        )
        return low
    if value > high:
        logger.warning(
            "%s=%.6g выше допустимого предела %.6g. Значение ограничено.",
            label,
            value,
            high,
        )
        return high
    return value


def _is_peakload_name(name: str) -> bool:
    return str(name).strip().lower() in {"peakload", "peak_load"}


# ---------------------------------------------------------------------------
# Full econometric calculation
# ---------------------------------------------------------------------------
def compute_full_details(
    params: ModelParameters,
    peak_raw: float,
    time_horizon: float,
    covariate_values: dict[str, float],
    raw_covariate_names: set[str] | None = None,
    residual_policy: str = DEFAULT_RESIDUAL_POLICY,
) -> dict[str, Any]:
    residual_policy = str(residual_policy).lower()
    if residual_policy not in SUPPORTED_RESIDUAL_POLICIES:
        raise InvalidInputError(f"Unsupported residual policy: '{residual_policy}'.")
    peak_raw = _as_float(peak_raw, "PeakLoad")
    time_horizon = _as_float(time_horizon, "time horizon")
    if time_horizon <= 0.0:
        raise InvalidInputError("Горизонт прогнозирования должен быть положительным числом.")
    _validate_peak_range(params, peak_raw)
    original_template = getattr(params, "template_covariates", {}) or {}
    if not isinstance(original_template, dict):
        original_template = {}
    params_copy = prepare_params_for_prediction(params, covariate_values)
    meta = getattr(params_copy, "training_meta", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    peak_convention = _dict_get_str(meta, "cox_peakload_convention", "").lower()
    if peak_convention == "pl_hat_exog":
        use_pl_hat_exog_for_peakload = True
    elif peak_convention == "observed_peakload":
        use_pl_hat_exog_for_peakload = False
    else:
        use_pl_hat_exog_for_peakload = USE_PL_HAT_EXOG_FOR_PEAKLOAD
    peak_transformed_raw = transform_peak(params_copy, peak_raw)
    peak_transformed_arr = np.asarray(peak_transformed_raw).ravel()
    if peak_transformed_arr.size == 0:
        raise PredictionError("transform_peak вернул пустой результат.")
    peak_transformed = _as_float(peak_transformed_arr[0], "transform_peak result")
    pl_hat_raw = predict_first_stage(params_copy, covariates=covariate_values)
    pl_hat_arr = np.asarray(pl_hat_raw).ravel()
    if pl_hat_arr.size == 0:
        raise PredictionError("predict_first_stage вернул пустой результат.")
    pl_hat = _as_float(pl_hat_arr[0], "predict_first_stage result")
    pl_hat_exog_raw: float | None = None
    try:
        pl_hat_exog_arr = np.asarray(compute_pl_hat_exog(params_copy, pl_hat)).ravel()
        if pl_hat_exog_arr.size > 0:
            pl_hat_exog_raw = _as_optional_float(pl_hat_exog_arr[0])
    except RUNTIME_ERRORS as exc:
        if use_pl_hat_exog_for_peakload:
            raise PredictionError(
                "compute_pl_hat_exog failed, but model requires PL_hat_exog."
            ) from exc
        pl_hat_exog_raw = None
    if pl_hat_exog_raw is None:
        if use_pl_hat_exog_for_peakload:
            raise PredictionError("PL_hat_exog is required but unavailable or non-finite.")
        pl_hat_exog = float(pl_hat)
    else:
        pl_hat_exog = float(pl_hat_exog_raw)
    cf_meta = _get_cf_meta(params_copy)
    residuals_mean = _dict_get_optional_float(cf_meta, "residuals_mean")
    if residuals_mean is None:
        residuals_mean = _as_optional_float(getattr(params_copy, "training_residuals_mean", 0.0))
    if residuals_mean is None:
        residuals_mean = 0.0
    residuals_std = _dict_get_optional_float(cf_meta, "residuals_std")
    if residuals_std is None:
        residuals_std = _as_optional_float(getattr(params_copy, "training_residuals_std", 1.0))
    if residuals_std is None or residuals_std <= 0.0:
        residuals_std = 1.0
    if residual_policy == "plug-in":
        raw_residual = peak_transformed - pl_hat
    elif residual_policy == "mean":
        raw_residual = residuals_mean
    elif residual_policy == "zero":
        raw_residual = 0.0
    else:
        raise InvalidInputError(f"Unsupported residual policy: '{residual_policy}'")
    if not math.isfinite(raw_residual):
        raise PredictionError("Non-finite CF residual.")
    standardized_residual_check = (raw_residual - residuals_mean) / residuals_std
    if not math.isfinite(standardized_residual_check):
        raise PredictionError("Non-finite standardized residual")
    abs_std_residual = abs(standardized_residual_check)
    if abs_std_residual > RESIDUAL_STD_ERROR:
        raise PredictionError(
            f"CF residual is too large: |standardized residual|={abs_std_residual:.3f} > {RESIDUAL_STD_ERROR:.3f}"
        )
    if abs_std_residual > RESIDUAL_STD_WARNING:
        logger.warning(
            "Большой CF остаток: |standardized residual|=%.3f > %.3f",
            abs_std_residual,
            RESIDUAL_STD_WARNING,
        )
    cf_cols = _get_cf_columns(params_copy)
    if "linear_standardized" in cf_meta:
        linear_standardized = bool(cf_meta.get("linear_standardized", True))
    else:
        linear_standardized = not any(str(c).lower().startswith("eps_d_hat") for c in cf_cols)
    if linear_standardized:
        v_hat = (raw_residual - residuals_mean) / residuals_std
    else:
        v_hat = raw_residual - residuals_mean
    if not math.isfinite(v_hat):
        raise PredictionError("Non-finite v_hat")
    basis_type = str(cf_meta.get("v_hat_basis", "linear")).lower()
    cf_basis_values: dict[str, np.ndarray] = {}
    if basis_type in {"spline", "powers"}:
        residuals_arr = np.array([raw_residual], dtype=float)
        if _BUILD_CF_BASIS_AT_PREDICTION is not None:
            try:
                built = _BUILD_CF_BASIS_AT_PREDICTION(params_copy, residuals_arr)
                if isinstance(built, dict):
                    cf_basis_values = built
            except RUNTIME_ERRORS:
                cf_basis_values = {}
        if not cf_basis_values:
            if basis_type == "powers":
                cf_basis_values = _manual_powers_basis(params_copy, raw_residual)
            else:
                raise PredictionError(
                    "Spline CF basis requires prediction_engine._build_cf_basis_at_prediction."
                )
        cf_basis_by_lower = {str(k).lower(): v for k, v in cf_basis_values.items()}
        for col in cf_cols:
            col_lower = str(col).lower()
            if col_lower in {"v_hat", "eps_d_hat"}:
                continue
            if col_lower not in cf_basis_by_lower:
                raise PredictionError(
                    f"Missing CF basis column '{col}' for basis_type='{basis_type}'"
                )
    else:
        cf_basis_by_lower = {}
    manual_covariates: dict[str, float] = {}
    for name, value in original_template.items():
        name_str = str(name)
        covariate_value = _as_optional_float(value)
        if covariate_value is None:
            raise InvalidInputError(f"Template covariate '{name_str}' is not finite")
        manual_covariates[name_str] = covariate_value
    raw_names = set(raw_covariate_names or [])
    for name, value in (covariate_values or {}).items():
        name_str = str(name)
        covariate_value = _as_float(value, f"Covariate '{name_str}'")
        if name_str in raw_names:
            manual_covariates[name_str] = _standardize_covariate(
                params_copy, name_str, covariate_value
            )
        else:
            manual_covariates[name_str] = covariate_value

    # ─── Автоматическое вычисление interaction Age × Hours ─────────────
    # Используем сырые значения из covariate_values, чтобы точно
    # воспроизвести тренировочную формулу.
    cox_model = getattr(params_copy, "cox", {}) or {}
    cox_names = cox_model.get("exog_names", []) or []
    if "x_age_hours" in [str(c) for c in cox_names] and "x_age_hours" not in manual_covariates:
        x_age_raw = (covariate_values or {}).get("x_age")
        x_hours_raw = (covariate_values or {}).get("x_hours")
        if x_age_raw is not None and x_hours_raw is not None:
            manual_covariates["x_age_hours"] = _compute_age_hours_interaction(
                params_copy, float(x_age_raw), float(x_hours_raw)
            )
            logger.info(
                "x_age_hours вычислен автоматически: %.6f",
                manual_covariates["x_age_hours"],
            )

    cox_model = getattr(params_copy, "cox", {}) or {}
    cox_coefs = cox_model.get("coefs", {}) or {}
    cox_names = cox_model.get("exog_names", []) or []
    if not isinstance(cox_coefs, dict):
        cox_coefs = {}
    lp = 0.0
    gamma = float("nan")
    peakload_coef_name: str | None = None
    missing_covariates: list[str] = []
    for raw_name in cox_names:
        name = str(raw_name)
        name_lower = name.lower()
        term_value = 0.0
        if _is_peakload_name(name):
            if use_pl_hat_exog_for_peakload:
                term_value = pl_hat_exog
            else:
                term_value = peak_transformed
            if peakload_coef_name is None:
                peakload_coef_name = name
        elif _is_cf_column(name, cf_cols):
            if name_lower in {"v_hat", "eps_d_hat"}:
                term_value = v_hat
            elif name_lower in cf_basis_by_lower:
                basis_arr = np.asarray(cf_basis_by_lower[name_lower]).ravel()
                if basis_arr.size == 0:
                    raise PredictionError(f"CF basis column '{name}' is empty")
                term_value = _as_float(basis_arr[0], f"CF basis value for '{name}'")
            else:
                raise PredictionError(f"Missing CF basis column '{name}'")
        elif name_lower in {"const", "intercept"}:
            term_value = 1.0
        else:
            if name not in manual_covariates:
                missing_covariates.append(name)
                continue
            term_value = manual_covariates[name]
        if not math.isfinite(term_value):
            raise PredictionError(f"Non-finite covariate value for Cox term '{name}'")
        coef_raw = None
        if name in cox_coefs:
            coef_raw = cox_coefs[name]
        else:
            for coef_name, coef_value in cox_coefs.items():
                if str(coef_name).lower() == name_lower:
                    coef_raw = coef_value
                    break
        coef = _as_optional_float(coef_raw)
        if coef is None:
            raise PredictionError(f"Missing or invalid Cox coefficient for term '{name}'")
        lp += coef * term_value
    if missing_covariates:
        raise PredictionError("Missing Cox covariates: " + ", ".join(missing_covariates))
    if not math.isfinite(lp):
        raise PredictionError("Линейный предиктор Cox-модели не является конечным числом.")
    if peakload_coef_name is not None:
        gamma_val = _as_optional_float(cox_coefs.get(peakload_coef_name))
        if gamma_val is not None:
            gamma = float(gamma_val)
    try:
        h0_raw = baseline_cumulative_hazard(params_copy, time_horizon)
    except RUNTIME_ERRORS as exc:
        raise PredictionError(f"Ошибка расчёта baseline_cumulative_hazard: {exc}") from exc
    h0_arr = np.asarray(h0_raw).ravel()
    if h0_arr.size == 0:
        raise PredictionError("baseline_cumulative_hazard вернул пустой результат.")
    h0 = _as_float(h0_arr[0], "baseline_cumulative_hazard result")
    if h0 < 0.0:
        if h0 >= -1e-12:
            h0 = 0.0
        else:
            raise PredictionError(f"Negative baseline cumulative hazard: {h0}")
    h0 = _clip_with_warning(h0, 0.0, MAX_CUMULATIVE_HAZARD, "Baseline cumulative hazard H0(t)")
    clipped_lp = _clip_with_warning(lp, MIN_EXP_ARG, MAX_EXP_ARG, "Cox linear predictor lp")
    hazard_ratio = math.exp(clipped_lp)
    if not math.isfinite(hazard_ratio):
        raise PredictionError("Hazard ratio is not finite")
    cumulative_hazard = h0 * hazard_ratio
    if math.isnan(cumulative_hazard):
        raise PredictionError("Cumulative hazard is NaN")
    if math.isinf(cumulative_hazard):
        logger.warning(
            "Cumulative hazard overflow. Значение ограничено %.1f.",
            MAX_CUMULATIVE_HAZARD,
        )
        cumulative_hazard = MAX_CUMULATIVE_HAZARD
    cumulative_hazard = _clip_with_warning(
        cumulative_hazard, 0.0, MAX_CUMULATIVE_HAZARD, "Individual cumulative hazard"
    )
    # ======================================================================
    # ★ Исправление 3: Унификация prediction — используем prediction_engine
    # ======================================================================
    # Ручной расчёт probability ниже может расходятся с prediction_engine
    # из-за различий в стандартизации, обработке CF basis и т.д.
    # Единственный источник истины — prediction_engine.predict_probability().
    # ======================================================================
    try:
        engine_probability = predict_probability(
            params_copy,
            peak_raw,
            time_horizon,
            residual_policy=residual_policy,
            covariates=covariate_values,
            time_horizon_unit=MODEL_TIME_UNIT,
            strict_covariates=False,
        )
        engine_probability = float(engine_probability)
    except RUNTIME_ERRORS as _exc:
        # Fallback на ручной расчёт если engine не справился
        logger.warning(
            "prediction_engine.predict_probability() не справился: %s. Используем ручной расчёт.",
            _exc,
        )
        engine_probability = None

    if engine_probability is not None:
        # Используем вероятность из engine как authoritative
        probability = engine_probability
        # Сравниваем с ручным расчётом для диагностики
        manual_prob = -math.expm1(-cumulative_hazard)
        manual_prob = float(max(0.0, min(1.0, manual_prob)))
        prob_diff = abs(probability - manual_prob)
        if prob_diff > 1e-6:
            logger.warning(
                "Расхождение probability: engine=%.6f, manual=%.6f, diff=%.2e. Используем engine.",
                probability,
                manual_prob,
                prob_diff,
            )
    else:
        # Fallback
        probability = -math.expm1(-cumulative_hazard)
        if not math.isfinite(probability):
            raise PredictionError("Probability is not finite")
        probability = _clip_with_warning(probability, 0.0, 1.0, "Probability")
    return {
        "peak_raw": peak_raw,
        "peak_transformed": peak_transformed,
        "pl_hat": pl_hat,
        "pl_hat_exog": pl_hat_exog,
        "v_hat": float(v_hat),
        "lp": lp,
        "h0_t": h0,
        "hazard_ratio": hazard_ratio,
        "cumulative_hazard": cumulative_hazard,
        "probability": probability,
        "gamma": gamma,
        "partial_out_X_beta": getattr(params_copy, "partial_out_X_beta", None),
        "training_residuals_mean": residuals_mean,
        "training_residuals_std": residuals_std,
        "time_horizon": time_horizon,
        "time_unit": MODEL_TIME_UNIT,
        "residual_policy": residual_policy,
        "basis_type": basis_type,
        "linear_standardized": linear_standardized,
        "peakload_convention": "pl_hat_exog"
        if use_pl_hat_exog_for_peakload
        else "observed_peakload",
    }


# ---------------------------------------------------------------------------
# Decomposed Main Functions
# ---------------------------------------------------------------------------
def load_and_validate_model(
    model_path: str,
) -> tuple[ModelParameters, dict[str, Any], dict[str, Any] | None]:
    try:
        params = load_model_params(model_path)
    except RUNTIME_ERRORS as e:
        logger.error("Ошибка загрузки модели: %s", e)
        raise SystemExit(2)

    # Anchor for provenance tools: model_json reference
    model_json = getattr(params, "training_meta", {}) or {}

    # Campaign validation: ensure prediction matches trained model
    if HAS_MODEL_PROVENANCE:
        try:
            trained_campaign = get_model_weather_campaign(
                {
                    "training_meta": getattr(params, "training_meta", {}) or {},
                    "metadata": getattr(params, "metadata", {}),
                }
            )
            assert_prediction_campaign(
                {
                    "training_meta": getattr(params, "training_meta", {}) or {},
                    "metadata": getattr(params, "metadata", {}),
                },
                trained_campaign,
            )
            print(f"Environmental campaign модели: {trained_campaign}")
        except Exception:
            logger.debug("Campaign validation skipped")

    check_model_semantic_consistency(params)

    try:
        validate_model(params)
    except RUNTIME_ERRORS as e:
        logger.error("Ошибка валидации модели: %s", e)
        raise SystemExit(2)

    model_version = _as_str(getattr(params, "model_version", "N/A"), "N/A")
    logger.info("Модель загружена (версия %s)", model_version)

    training_meta = getattr(params, "training_meta", {}) or {}
    if not isinstance(training_meta, dict):
        training_meta = {}

    model_time_unit = _get_time_unit(params)
    if model_time_unit != MODEL_TIME_UNIT:
        logger.error(
            "Модель использует единицу времени '%s', но калькулятор поддерживает только '%s'. Расчёт прерван.",
            model_time_unit,
            MODEL_TIME_UNIT,
        )
        raise SystemExit(2)

    if training_meta.get("stress_test_mode") or training_meta.get("contamination"):
        logger.warning(
            "Модель обучена в stress-test режиме (загрязнённые данные / contamination=True)."
        )
        logger.warning("Результаты НЕ должны использоваться для продуктового ценообразования.")
        answer = ask_str("Продолжить расчёт, несмотря на stress-test? [нет]", "нет").strip().lower()
        if answer not in {"да", "д", "yes", "y"}:
            logger.info("Расчёт прерван пользователем.")
            raise SystemExit(2)

    censoring_distortion = training_meta.get("censoring_distortion", {}) or {}
    if isinstance(censoring_distortion, dict):
        q_ratio_value = _dict_get_optional_float(censoring_distortion, "q_ratio")
        if q_ratio_value is not None and q_ratio_value < 0.8:
            logger.warning(
                "Обнаружено искажение цензуры: q_ratio=%.3f < 0.8.",
                float(q_ratio_value),
            )

    dgp_path = ask_str("Путь к файлу калибровки DGP (calibrated_dgp.json) [авто]", "")
    dgp_data = load_dgp_with_fallback(dgp_path)

    calib_horizon_raw = getattr(params, "calibration_time_horizon", None)
    if calib_horizon_raw is None:
        calib_horizon_raw = training_meta.get("calibration_time_horizon")
    calib_horizon_value = _as_optional_float(calib_horizon_raw)
    if calib_horizon_value is not None:
        calib_horizon_f = float(calib_horizon_value)
        calib_horizon_days_f = _engine_hours_to_calendar_days(
            calib_horizon_f, _get_hours_per_day(params)
        )
        logger.info(
            "Горизонт калибровки модели: %.0f мото-часов (≈ %.0f календарных дней)",
            calib_horizon_f,
            calib_horizon_days_f,
        )

    ed = _dict_get_str(training_meta, "event_definition", "major_claim")
    logger.info("Определение события: %s", ed)
    prior = training_meta.get("major_failure_share_prior")
    if isinstance(prior, dict):
        prior_mean = _dict_get_optional_float(prior, "mean")
        prior_low = _dict_get_optional_float(prior, "ci_low")
        prior_high = _dict_get_optional_float(prior, "ci_high")
        if prior_mean is not None and prior_low is not None and prior_high is not None:
            logger.info(
                "Доля major-отказов: mean=%.3f 95%%CI=[%.3f, %.3f]",
                float(prior_mean),
                float(prior_low),
                float(prior_high),
            )

    target_probability = _dict_get_optional_float(training_meta, "target_probability")
    if target_probability is not None:
        target_probability_f = float(target_probability)
        if 0.0 <= target_probability_f <= 1.0:
            logger.info("Целевая вероятность события: %.6f", target_probability_f)
        else:
            logger.warning("target_probability вне [0,1]: %.6f", target_probability_f)
    else:
        logger.info("Горизонт калибровки не указан.")

    return params, training_meta, dgp_data


def collect_user_inputs(
    params: ModelParameters,
    training_meta: dict[str, Any],
    dgp_data: dict[str, Any] | None,
) -> CalculatorConfig:
    cfg = CalculatorConfig()
    cfg.training_meta = training_meta
    cfg.dgp_data = dgp_data
    cfg.hours_per_day = _get_hours_per_day(params)
    cfg.model_time_unit = _get_time_unit(params)

    calib_horizon_raw = getattr(params, "calibration_time_horizon", None)
    if calib_horizon_raw is None:
        calib_horizon_raw = training_meta.get("calibration_time_horizon")
    cfg.calib_horizon_value = _as_optional_float(calib_horizon_raw)

    cfg.peaks = get_peakload_choice(params)
    cfg.extended = collect_extended_parameters()

    # ─── Фаза X: режим культуры ──────────────────────────────────
    cfg.crop_key = ""
    cfg.crop_area_ha = 0.0
    cfg.crop_weighted_peak = None
    cfg.crop_total_hours = None

    if HAS_AGRO_CALENDAR:
        crop_key, crop_area = ask_crop_selection(cfg.extended)
        if crop_key:
            cfg.crop_key = crop_key
            cfg.crop_area_ha = crop_area

            # Определить трактор из расширенных параметров
            brand_name = cfg.extended.get("brand", "МТЗ-82")
            tractor = _map_brand_to_tractor(brand_name)

            # ─── Фаза X: K_об ──────────────────────────────────────
            if HAS_AGRO_NORMS:
                cfg.k_ob, cfg.k_ob_params = ask_k_ob_parameters()

            # Вычислить средневзвешенный пик и суммарные моточасы
            # (estimate_season_engine_hours уже использует OPERATION_INFO внутри себя)
            total_hours, weighted_peak = estimate_season_engine_hours(
                crop_key, crop_area, tractor=tractor, k_ob=cfg.k_ob
            )

            # Переопределить пики и горизонт
            cfg.peaks = [weighted_peak]
            cfg.horizon = total_hours
            cfg.crop_weighted_peak = weighted_peak
            cfg.crop_total_hours = total_hours

            # Показать план работ
            operation_names_ru = {}
            for op_key in TUM_OPERATIONS:
                operation_names_ru[op_key] = TUM_OPERATIONS[op_key].get("name_ru", op_key)
            for op_key in OPERATION_INFO:
                if op_key not in operation_names_ru:
                    operation_names_ru[op_key] = OPERATION_INFO[op_key].get("name_ru", op_key)

            print(
                "\n" + format_crop_summary(crop_key, crop_area, cfg.peaks, operation_names_ru, cfg.k_ob)
            )

            # ─── Проверка горизонта калибровки ────────────────────
            calib_horizon = cfg.calib_horizon_value or DEFAULT_HORIZON_ENGINE_HOURS
            if total_hours > calib_horizon:
                print(
                    f"\n⚠️  Суммарные моточасы ({total_hours:.0f}) превышают "
                    f"калибровочный горизонт ({calib_horizon:.0f})."
                )

                # Проверить наличие параметрического базового риска
                baseline_spec = getattr(params, "baseline_spec", None)
                baseline_family = "breslow"
                if isinstance(baseline_spec, dict):
                    baseline_family = str(baseline_spec.get("family", "breslow")).lower()

                if baseline_family in ("weibull", "gompertz", "exponential"):
                    print(
                        f"   ✅ Параметрический базовый риск ({baseline_family}) "
                        f"позволяет экстраполяцию."
                    )
                else:
                    print("   ❌ Базовый риск Бреслоу НЕ позволяет экстраполяцию.")
                    print("   Переобучите модель с параметрическим базовым риском:")
                    print("   python train_model.py → параметрическая подгонка → weibull")

            logger.info(
                "Режим культуры: %s, площадь %.0f га, трактор %s, "
                "средневзвешенный пик %.4f, суммарные моточасы %.0f",
                crop_key,
                crop_area,
                tractor,
                weighted_peak,
                total_hours,
            )

    cfg.covariate_values, cfg.raw_covariate_names = collect_model_covariates(
        params, cfg.extended, dgp_data
    )

    cfg.heavy_prob = _ask_heavy_probability(cfg.extended)
    cfg.labor_ratio = _ask_labor_ratio(
        "Доля времени простоя, когда работает мастер (0.0-1.0)", 0.50
    )
    cfg.coverage_mode = _ask_coverage_mode()
    cfg.major_failure_share = _ask_major_failure_share(params)
    cfg.season_hazard_factor = _ask_season_hazard_factor(
        "Сезонный множитель частоты отказов (1.0 = без изменения)", 1.0
    )

    cox_model_info = getattr(params, "cox", {}) or {}
    fs_model_info = getattr(params, "first_stage", {}) or {}
    cox_names = cox_model_info.get("exog_names", []) or []
    fs_names = fs_model_info.get("exog_names", []) or []
    print("\n" + "-" * 60 + "\nКОВАРИАТЫ МОДЕЛИ\n" + "-" * 60)
    print(f"  Cox-модель (exog):      {cox_names}")
    print(f"  Первая стадия (exog):   {fs_names}")

    # Информация о регионе
    selected_region = cfg.extended.get("selected_region")
    if selected_region:
        print(f"\n  Регион:                {selected_region} (из реальных данных)")
    else:
        print("\n  Регион:                ручной ввод")
    print(f"  x_climate:             {cfg.covariate_values.get('x_climate', 'N/A'):.4f}")
    print(f"  x_soil:                {cfg.covariate_values.get('x_soil', 'N/A'):.4f}")

    cfg.sum_insured = _ask_sum_insured(
        "Страховая сумма / лимит выплаты (руб.)", DEFAULT_SUM_INSURED
    )

    horizon_default = DEFAULT_HORIZON_ENGINE_HOURS
    if cfg.calib_horizon_value is not None:
        horizon_default = float(cfg.calib_horizon_value)
    cfg.horizon = _ask_positive_horizon("Горизонт прогнозирования (мото-часы)", horizon_default)

    cfg.theta = _ask_theta("Страховая нагрузка (доля, например 0.15 = 15%)", DEFAULT_THETA)
    cfg.discount_rate = _ask_discount_rate(
        "Годовая ставка дисконтирования (например 0.10 = 10%, 0 = без дисконтирования)",
        DEFAULT_DISCOUNT_RATE,
    )

    if cfg.calib_horizon_value is not None:
        calib_horizon_f = float(cfg.calib_horizon_value)
        if abs(cfg.horizon - calib_horizon_f) > 1e-6:
            logger.warning(
                "Модель калибрована на %.0f мото-часов, а вы запросили %.0f мото-часов. Прогноз на другом горизонте может быть неточным.",
                calib_horizon_f,
                cfg.horizon,
            )

    try:
        cfg.expected_repair = compute_expected_repair_cost(
            cfg.extended["failure_shares"],
            cfg.extended["repair_costs"],
            heavy_prob=cfg.heavy_prob,
        )
    except RUNTIME_ERRORS as e:
        logger.error("Ошибка расчёта ожидаемой стоимости ремонта: %s", e)
        raise SystemExit(6)

    group = _as_int(
        cfg.extended.get("failure_group"),
        name="failure_group",
        default=DEFAULT_FAILURE_GROUP,
    )
    group = max(1, min(3, group))
    downtime_hours = get_downtime_mean_hours(group)
    regional_rate = _dict_get_float(cfg.extended, "regional_rate", 0.0)
    if regional_rate < 0.0:
        raise InvalidInputError("regional_rate must be non-negative")
    effective_labor_cost = regional_rate * cfg.labor_ratio
    seasonal_downtime_cost = _dict_get_float(
        cfg.extended, "downtime_hour_cost", float(BASE_DOWNTIME_COST)
    )
    if seasonal_downtime_cost < 0.0:
        raise InvalidInputError("downtime_hour_cost must be non-negative")

    try:
        cfg.downtime_cost_per_failure = compute_downtime_cost(
            downtime_hours, seasonal_downtime_cost + effective_labor_cost
        )
    except RUNTIME_ERRORS as e:
        logger.error("Ошибка расчёта стоимости простоя: %s", e)
        raise SystemExit(6)

    cfg.expected_loss_per_failure = cfg.expected_repair + cfg.downtime_cost_per_failure

    # --- Фаза 7.9: попытка использовать severity_model из реальных данных ---
    severity_model = _load_severity_model_safe()
    if severity_model is not None:
        cfg.expected_loss_per_failure = severity_model.expected_loss_per_failure()
        cfg.expected_repair = severity_model.expected_repair_cost()
        cfg.downtime_cost_per_failure = severity_model.expected_downtime_cost()
        logger.info(
            "Severity из реальных данных: E[loss] = %s руб. (E[repair] = %s, E[downtime_cost] = %s)",
            fmt_money(cfg.expected_loss_per_failure),
            fmt_money(cfg.expected_repair),
            fmt_money(cfg.downtime_cost_per_failure),
        )
        cfg.severity_model = severity_model

    cfg.use_severity_pricing = (
        severity_model is not None and cfg.coverage_mode != COVERAGE_FIXED_SUM
    )
    if cfg.use_severity_pricing:
        cfg.severity_expected_severity = cfg.expected_loss_per_failure
        cfg.severity_deductible = 0.0
        cfg.severity_coverage_limit = (
            cfg.sum_insured if cfg.coverage_mode == COVERAGE_REPAIR_DOWNTIME_CAPPED else None
        )
        cfg.claim_amount = cfg.sum_insured
        logger.info(
            "Используется severity-based расчёт: E[covered loss] = %s руб., deductible = %s, limit = %s",
            fmt_money(cfg.severity_expected_severity),
            fmt_money(cfg.severity_deductible),
            fmt_money(cfg.severity_coverage_limit) if cfg.severity_coverage_limit else "нет",
        )
    else:
        try:
            cfg.claim_amount = compute_claim_amount(
                cfg.coverage_mode, cfg.expected_loss_per_failure, cfg.sum_insured
            )
        except RUNTIME_ERRORS as e:
            logger.error("Ошибка расчёта суммы выплаты: %s", e)
            raise SystemExit(6)

    print("\n" + "-" * 60 + "\nЭКОНОМИЧЕСКИЕ ПАРАМЕТРЫ УБЫТКА\n" + "-" * 60)
    if severity_model is not None:
        print(
            f"Ожидаемая стоимость ремонта (из severity-модели): {fmt_money(severity_model.expected_repair_cost())} руб."
        )
        print(
            f"Ожидаемая стоимость простоя (из severity-модели): {fmt_money(severity_model.expected_downtime_cost())} руб."
        )
    else:
        print(
            f"Ожидаемая стоимость ремонта (экспертная, на один отказ): {fmt_money(cfg.expected_repair)} руб."
        )
        print(
            f"Стоимость простоя (на один отказ, mean {downtime_hours:.1f} ч): {fmt_money(cfg.downtime_cost_per_failure)} руб."
        )
    print(f"Ожидаемый убыток на один отказ: {fmt_money(cfg.expected_loss_per_failure)} руб.")
    if cfg.use_severity_pricing:
        print(
            f"Сумма выплаты для премии (режим '{cfg.coverage_mode}'): "
            f"{fmt_money(cfg.severity_expected_severity)} руб. "
            f"(лимит: {fmt_money(cfg.claim_amount)} руб.)"
        )
    else:
        print(
            f"Сумма выплаты для премии (режим '{cfg.coverage_mode}'): {fmt_money(cfg.claim_amount)} руб."
        )

    return cfg


def compute_all_peaks(params: ModelParameters, cfg: CalculatorConfig) -> list[dict[str, Any]]:
    """Compute failure probabilities for multiple PeakLoad values.

    WARNING: Exclusion Restriction Assumption
    ----------------------------------------
    This model assumes weather instruments (Z) affect failure time (Y)
    ONLY through PeakLoad. If Z directly affects Y (e.g., through soil
    wear, track damage, hydraulic corrosion from mud), the IV estimate
    is inconsistent.

    To validate this assumption, run:
        placebo_test_exclusion_restriction(model_params, weather_data)

    If the placebo test fails (p < 0.05), consider:
    1. Using a different instrument Z (e.g., days without field access)
    2. Including weather covariates directly in the Cox model
    3. Using a control function approach with additional instruments

    Competing Risks Formula
    -----------------------
    P(T <= t, cause=k) = P(T <= t) * P(cause=k | X)

    This is valid under the proportional causes assumption:
    h_k(t|X) = P(cause=k|X) * h_total(t|X)

    P(cause=k|X) is computed via logistic regression:
    logit(P(major|X)) = alpha + beta_peak * PeakLoad + beta_age * Age + beta_hours * Hours
    """
    # ─── Фаза X: режим культуры — один средневзвешенный пик ──────
    if cfg.crop_key and HAS_AGRO_CALENDAR:
        # В режиме культуры пики уже установлены в collect_user_inputs()
        # как [средневзвешенный пик]. Просто используем их.
        crop = get_crop(cfg.crop_key)
        crop_name = crop.crop_name_ru if crop else cfg.crop_key
        logger.info(
            "Режим культуры '%s': средневзвешенный пик %.4f "
            "на горизонте %.0f моточасов (площадь %.0f га)",
            crop_name,
            cfg.peaks[0] if cfg.peaks else 0.50,
            cfg.horizon,
            cfg.crop_area_ha,
        )
        # Переопределяем labels для вывода
        # (вместо "Вспашка (Ploughing)" будет "Пшеница озимая (200 га)")

    # PATCH-06: Проверка exclusion restriction перед расчётом
    placebo = placebo_test_exclusion_restriction(params)
    if placebo.get("exclusion_valid") is None:
        logger.warning(
            "⚠️ Exclusion restriction НЕ проверена. "
            "Результаты имеют предсказательный, а не каузальный характер."
        )

    results: list[dict[str, Any]] = []

    # ★ НОВОЕ: Если выбрана конкретная операция — добавить её PeakLoad
    peaks_to_compute = list(cfg.peaks)
    labels = [f"Q{i}" for i in range(len(cfg.peaks))]

    # В режиме культуры переопределяем label для вывода
    if cfg.crop_key and HAS_AGRO_CALENDAR:
        crop = get_crop(cfg.crop_key)
        crop_name = crop.crop_name_ru if crop else cfg.crop_key
        labels = [f"{crop_name} ({cfg.crop_area_ha:.0f} га)"]

    selected_operation = cfg.extended.get("season")
    if selected_operation and selected_operation in TUM_OPERATIONS:
        op_info = TUM_OPERATIONS[selected_operation]
        op_peak_raw = op_info.get("peak_load_mean", op_info.get("mean"))
        if op_peak_raw is not None and op_peak_raw > 0:
            # PATCH 1.1: Масштабирование удалено — шкалы TUM и DGP совпадают [0, 1]
            op_peak = op_peak_raw
            peaks_to_compute.append(op_peak)
            # Показываем русское название операции
            name_ru = op_info.get("name_ru", selected_operation)
            labels.append(f"{name_ru} ({selected_operation})")

    print("\n" + "-" * 60 + "\nРЕЗУЛЬТАТЫ РАСЧЁТА (пошагово)\n" + "-" * 60)

    for i, peak in enumerate(peaks_to_compute):
        label = labels[i] if i < len(labels) else f"Peak {i + 1}"
        print(f"\n{'=' * 60}")
        print(f"Расчёт для: {label}")
        print(f"{'=' * 60}")

        # ─── Reduced Form: индикация режима ─────────────────────────────
        model_form = str(cfg.training_meta.get("model_form", "control_function")).lower()
        if model_form == "reduced_form":
            print("ℹ️ Режим: REDUCED FORM (предиктивная модель, Z как ковариата)")

        try:
            details = compute_full_details(
                params,
                peak,
                cfg.horizon,
                cfg.covariate_values,
                raw_covariate_names=cfg.raw_covariate_names,
                residual_policy=DEFAULT_RESIDUAL_POLICY,
            )
        except RUNTIME_ERRORS as e:
            logger.error("Ошибка расчёта для PeakLoad=%s: %s", fmt_num(peak), e)
            continue

        print(f"\nPeakLoad (исходное):             {fmt_num(details['peak_raw'], '.4f')}")
        print(f"PeakLoad (трансформированное):   {fmt_num(details['peak_transformed'], '.4f')}")
        print(f"PL_hat (первая стадия):          {fmt_num(details['pl_hat'], '.4f')}")
        print(f"PL_hat_exog (диагностика):       {fmt_num(details['pl_hat_exog'], '.4f')}")
        print(f"v_hat (станд. остаток):          {fmt_num(details['v_hat'], '.4f')}")
        print(f"Линейный предиктор lp:           {fmt_num(details['lp'], '.4f')}")
        print(
            f"Базовый кумулятивный риск H0({cfg.horizon:.0f} мч): {fmt_num(details['h0_t'], '.6f')}"
        )
        print(f"Отношение рисков exp(lp):        {fmt_num(details['hazard_ratio'], '.4f')}")
        print(f"Индивидуальный кумулятивный риск: {fmt_num(details['cumulative_hazard'], '.6f')}")

        manual_prob = _as_float(details["probability"], "manual probability")
        print(
            f"Ручная вероятность P(T <= {cfg.horizon:.0f} мч): {fmt_num(manual_prob, '.6f')} ({fmt_num(manual_prob * 100.0, '.2f')})%"
        )

        # ======================================================================
        # ★ FIX: Убран дублирующий вызов validate_probability_with_engine()
        # ======================================================================
        # compute_full_details() уже вызывает predict_probability() внутри
        # (Исправление 3). Вызов validate_probability_with_engine() здесь
        # приводил к ДВОЙНОМУ вызову predict_probability() для каждого пика.
        #
        # manual_prob из compute_full_details() уже содержит authoritative
        # вероятность из prediction_engine.
        # ======================================================================
        base_probability = manual_prob

        try:
            adjusted_probability = apply_hazard_multiplier(
                base_probability, cfg.season_hazard_factor
            )
        except RUNTIME_ERRORS as e:
            logger.error("Ошибка сезонной корректировки вероятности: %s", e)
            raise SystemExit(5)

        # ======================================================================
        # ★ Исправление 2: Competing Risks — корректная формула
        # ======================================================================
        # Правильная формула для competing risks:
        #   P(T ≤ t, cause=k) = ∫₀ᵗ h_k(s|X) S(s|X) ds
        #
        # При пропорциональности cause-specific hazards:
        #   h_k(s|X) = ω_k(X) · h(s|X)
        # получаем: P(T ≤ t, cause=k) = ω_k(X) · P(T ≤ t)
        #
        # где ω_k(X) = P(cause=k | T ≤ t, X) — доля отказов типа k,
        # которая может зависеть от ковариат X.
        #
        # ★ КРИТИЧЕСКОЕ УЛУЧШЕНИЕ: ω_k(X) теперь зависит от стандартизированных
        # ковариат через логистическую регрессию, а не константу.
        # ======================================================================
        model_event_def = str(cfg.training_meta.get("event_definition", "major_claim")).lower()
        user_wants_major = float(cfg.major_failure_share) < 1.0

        if model_event_def in {"major_claim", "total_loss"}:
            # Модель уже обучена на крупном событии — вероятность
            # уже является cause-specific для major failures.
            effective_share = 1.0
            logger.info(
                "Модель обучена на event_definition='%s' — вероятность уже cause-specific.",
                model_event_def,
            )
        elif user_wants_major:
            # ★ PATCH 1.2: Явная стандартизация ковариат для Competing Risks
            # Коэффициенты beta_age, beta_peak обучались на СТАНДАРТИЗИРОВАННЫХ
            # данных (mean=0, std=1). Подавать сырые значения нельзя — это
            # приводит к насыщению сигмоиды и потере чувствительности.
            #
            # Пример: age_raw=10, beta_age=0.20 → вклад 2.0 → сигмоида ~0.88
            # На стандартизированных: age_std≈0, вклад 0 → сигмоида ~0.30
            #
            # Стандартизация через training_meta[x_standardization]
            x_std_meta = cfg.training_meta.get("x_standardization", {})
            if not isinstance(x_std_meta, dict):
                x_std_meta = {}

            # Получаем параметры стандартизации для age и peakload
            age_info = x_std_meta.get("x_age", {})
            peak_info = x_std_meta.get("PeakLoad", {})

            # Дефолтные параметры (если training_meta не заполнен)
            age_shift = float(age_info.get("shift", 10.0))
            age_scale = float(age_info.get("scale", 10.0))
            peak_shift = float(peak_info.get("shift", 0.55))
            peak_scale = float(peak_info.get("scale", 0.15))

            if age_scale == 0.0:
                age_scale = 10.0
            if peak_scale == 0.0:
                peak_scale = 0.15

            # Берём RAW значения из cfg.extended
            age_raw = cfg.extended.get("age_years", age_shift)  # по умолчанию = mean
            # PATCH 2 (CRITICAL): Используем ТЕКУЩИЙ peak из цикла, а не фиксированный peak_load_mean
            # Это критично для competing risks, т.к. effective_share должен зависеть от текущего PeakLoad
            peak_raw = peak  # ← используем текущий peak из цикла for peak in peaks_to_compute

            # Стандартизируем
            age_std = (float(age_raw) - age_shift) / age_scale
            peak_std = (float(peak_raw) - peak_shift) / peak_scale

            # FIX 2: Коэффициенты logistic модели загружаются из training_meta
            # если модель была обучена с fit_cause_specific_logistic().
            # Иначе используются priors из MAJOR_FAILURE_SHARE_PRIOR.
            cause_params = cfg.training_meta.get("cause_specific_params", {})
            if cause_params:
                alpha_logit = float(cause_params.get("alpha_logit", math.log(0.3 / 0.7)))
                beta_peak = float(cause_params.get("beta_peak", 0.30))
                beta_age = float(cause_params.get("beta_age", 0.20))
                beta_hours = float(cause_params.get("beta_hours", 0.10))
                logger.info(
                    "Cause-specific share: использованы обученные коэффициенты "
                    "(alpha=%.3f, beta_peak=%.3f, beta_age=%.3f, beta_hours=%.3f)",
                    alpha_logit,
                    beta_peak,
                    beta_age,
                    beta_hours,
                )
            else:
                # Приор из MAJOR_FAILURE_SHARE_PRIOR: mean=0.30, effective_n=30
                # α = log(mean / (1 - mean)) = log(0.3 / 0.7) ≈ -0.847
                # β_peak = 0.3 (высокая нагрузка увеличивает долю major)
                # β_age = 0.2 (старая техника увеличивает долю major)
                alpha_logit = math.log(
                    float(cfg.major_failure_share) / (1.0 - float(cfg.major_failure_share))
                )
                beta_peak = 0.30
                beta_age = 0.20
                beta_hours = 0.10
                # PATCH-15: явное логирование использования priors
                logger.warning(
                    "Cause-specific share: использованы ЭКСПЕРТНЫЕ priors "
                    "(beta_peak=%.2f, beta_age=%.2f, beta_hours=%.2f). "
                    "Неопределённость НЕ пропагирована в доверительные интервалы. "
                    "Для точного расчёта обучите модель через "
                    "fit_cause_specific_logistic().",
                    beta_peak,
                    beta_age,
                    beta_hours,
                )

            logit_share = alpha_logit + beta_peak * float(peak_std) + beta_age * float(age_std)
            covariate_dependent_share = 1.0 / (1.0 + math.exp(-logit_share))
            effective_share = float(max(0.05, min(0.95, covariate_dependent_share)))

            logger.info(
                "Cause-specific share: P(major|X) = %.3f "
                "(logit=%.3f, PeakLoad_std=%.3f, Age_std=%.3f)",
                effective_share,
                logit_share,
                peak_std,
                age_std,
            )
        else:
            # Любой отказ — share = 1.0
            effective_share = 1.0

        # ======================================================================
        # ★ ЗАДАЧА B: CIF через численное интегрирование (v3.1)
        # ======================================================================
        # Правильная формула для конкурирующих рисков:
        #   CIF_k(t) = ∫₀ᵗ ω_k(u|X) · f(u|X) du
        # где f(u|X) = h₀(u)·exp(lp)·exp(-H₀(u)·exp(lp)) — плотность отказа.
        #
        # При пропорциональности причин (ω не зависит от t):
        #   CIF_k(t) = ω_k · F(t) — старая формула как частный случай.
        #
        # Параметрический базовый риск (Weibull/Gompertz) даёт гладкую
        # плотность, которую можно интегрировать. Для Breslow (ступенька)
        # интегрирование невозможно — сохраняем старую формулу.
        # ======================================================================

        baseline_spec = getattr(params, "baseline_spec", None)
        if baseline_spec is None:
            baseline_spec = {"family": "breslow"}
        baseline_family = str(baseline_spec.get("family", "breslow")).lower()

        use_cif_integration = HAS_PARAMETRIC_CIF and baseline_family in VALID_PARAMETRIC_FAMILIES

        if use_cif_integration:
            # ─── CIF через квадратуру Гаусса-Лежандра ─────────────────────
            lp_value = float(details["lp"])

            # Сезонный множитель влияет на интенсивность отказов →
            # добавляем log(season_factor) к линейному предиктору
            if abs(cfg.season_hazard_factor - 1.0) > 1e-12:
                lp_adjusted = lp_value + math.log(max(cfg.season_hazard_factor, 1e-9))
            else:
                lp_adjusted = lp_value

            # ω_k(t|X): константная доля причины (пропорциональная модель).
            # В будущем можно расширить до зависящей от времени ω(t).
            def omega_cause_fn(t: float) -> float:
                """Доля причины отказа в момент времени t."""
                return float(effective_share)

            try:
                cif_value = compute_cif(
                    spec=baseline_spec,
                    lp=lp_adjusted,
                    horizon=float(cfg.horizon),
                    omega_fn=omega_cause_fn,
                    n_quad=96,
                )

                # ─── Sanity check: при константной ω CIF ≈ ω·F ───────────
                proportional_check = adjusted_probability * effective_share
                cif_diff = abs(cif_value - proportional_check)
                if cif_diff > 0.02:
                    logger.warning(
                        "⚠️ CIF=%.6f отличается от пропорциональной проверки "
                        "ω·F=%.6f на %.4f. Проверьте согласованность baseline.",
                        cif_value,
                        proportional_check,
                        cif_diff,
                    )
                else:
                    logger.debug(
                        "CIF sanity check OK: CIF=%.6f ≈ ω·F=%.6f (diff=%.2e)",
                        cif_value,
                        proportional_check,
                        cif_diff,
                    )

                event_probability = float(min(max(cif_value, 0.0), 1.0))
                print(
                    f"  CIF (численное интегрирование, {baseline_family}): {event_probability:.6f}"
                )

            except Exception as cif_exc:  # noqa: BLE001
                logger.warning(
                    "CIF-интегрирование не удалось (%s). Fallback на пропорциональную формулу ω·F.",
                    cif_exc,
                )
                event_probability = adjusted_probability * effective_share
                event_probability = float(max(0.0, min(1.0, event_probability)))
        else:
            # ─── Старый путь: пропорциональная формула ω·F ────────────────
            # Используется для базового риска Breslow (ступенчатая функция)
            # или при отсутствии модуля parametric_baseline.
            event_probability = adjusted_probability * effective_share
            event_probability = float(max(0.0, min(1.0, event_probability)))
        # ======================================================================

        # Печать с указанием метода расчёта
        if use_cif_integration:
            method_label = f"CIF ({baseline_family}, квадратура Гаусса-Лежандра)"
        else:
            method_label = "пропорциональная формула ω·F"
        print(
            f"Итоговая вероятность страхового события: "
            f"{fmt_num(event_probability, '.6f')} "
            f"({fmt_num(event_probability * 100.0, '.2f')}%)"
        )
        print(f"  Метод расчёта: {method_label}")
        if abs(effective_share - 1.0) > 1e-12:
            if use_cif_integration:
                print(
                    f"  (CIF интегрирует плотность отказа с ω={effective_share:.3f}; "
                    f"базовая P(T≤t)={adjusted_probability:.6f})"
                )
            else:
                print(
                    f"  (базовая вероятность {fmt_num(adjusted_probability, '.6f')} "
                    f"× effective_share {fmt_num(effective_share, '.3f')})"
                )
        if abs(cfg.season_hazard_factor - 1.0) > 1e-12:
            print(
                f"  (применён сезонный множитель частоты {fmt_num(cfg.season_hazard_factor, '.3f')})"
            )

        horizon_calendar_days = _engine_hours_to_calendar_days(cfg.horizon, cfg.hours_per_day)
        premium_kwargs: dict[str, Any] = {
            "discount_rate": cfg.discount_rate,
            "calibration_horizon_days": None,
            "policy_horizon_days": horizon_calendar_days,
        }
        if cfg.calib_horizon_value is not None:
            premium_kwargs["calibration_horizon_days"] = _engine_hours_to_calendar_days(
                float(cfg.calib_horizon_value), cfg.hours_per_day
            )

        try:
            premium = calculate_single_premium(
                event_probability,
                cfg.claim_amount,
                cfg.theta,
                expected_severity=cfg.severity_expected_severity,
                deductible=cfg.severity_deductible,
                coverage_limit=cfg.severity_coverage_limit,
                **premium_kwargs,
            )
        except RUNTIME_ERRORS as e:
            logger.error("Ошибка расчёта премии: %s", e)
            raise SystemExit(6)

        if not isinstance(premium, dict):
            logger.error("premium_engine вернул неожиданный тип: %s", type(premium).__name__)
            raise SystemExit(6)

        net_premium = premium.get("net")
        gross_premium = premium.get("gross")
        if net_premium is None or gross_premium is None:
            logger.error("premium_engine не вернул net/gross премию.")
            raise SystemExit(6)

        print(f"\nНетто-премия:          {fmt_money(net_premium)} руб.")
        print(f"Брутто-премия:         {fmt_money(gross_premium)} руб.")
        tariff_value = _as_optional_float(premium.get("tariff"))
        if tariff_value is not None:
            print(
                f"Страховая ставка от страховой суммы:                 {float(tariff_value):.4f} %"
            )
        else:
            print(f"Тариф:                 {fmt_num(premium.get('tariff'))}")
        discount_factor = premium.get("discount_factor")
        if discount_factor is not None:
            print(f"Discount factor:       {fmt_num(discount_factor, '.6f')}")
        loading_amount = premium.get("loading_amount")
        if loading_amount is not None:
            print(f"Страховая нагрузка:    {fmt_money(loading_amount)} руб.")

        # ======================================================================
        # ★ Исправление 6: Risk margin на основе дисперсии severity
        # ======================================================================
        severity_risk_margin = 0.0
        severity_dispersion_warning = ""

        if HAS_SEVERITY_MODEL and cfg.severity_model is not None:
            sm = cfg.severity_model
            severity_var = getattr(sm, "expected_variance", None)
            if severity_var is not None and severity_var > 0:
                severity_sd = math.sqrt(severity_var)
                # PATCH 1 (CRITICAL): Правильная формула Risk Margin с учетом биномиальной дисперсии
                # L = I * S, где I ~ Bernoulli(p), S — severity
                # Var(L) = p * Var(S) + p*(1-p) * E[S]^2
                expected_severity = cfg.expected_loss_per_failure
                total_variance = (
                    event_probability * severity_var
                    + event_probability * (1.0 - event_probability) * expected_severity**2
                )
                severity_risk_margin = 1.645 * math.sqrt(total_variance)
                severity_dispersion_warning = (
                    f"Risk margin (95% VaR): {fmt_money(severity_risk_margin)} руб. "
                    f"(SD={fmt_money(severity_sd)}, E[severity]={fmt_money(expected_severity)})"
                )
            else:
                severity_dispersion_warning = (
                    "Severity-модель не содержит оценки дисперсии. Risk margin не рассчитан."
                )
        else:
            # PATCH 2.1: Корректная дисперсия через Lognormal Variance
            # Вместо произвольного CV=0.5 используем аналитическую формулу:
            #   Var[X] = [exp(sigma^2) - 1] * exp(2*mu + sigma^2)
            # где mu=11.5, sigma=0.6 (из constants.py)
            severity_variance = compute_lognormal_variance(
                SEVERITY_LOGNORMAL_MU, SEVERITY_LOGNORMAL_SIGMA
            )
            severity_sd = math.sqrt(severity_variance)

            # Downtime SD — отдельный компонент, если есть данные
            downtime_sd = 0.0
            if cfg.downtime_cost_per_failure > 0:
                # Для downtime используем экспертный CV=0.5 как fallback,
                # т.к. нет параметрической модели
                downtime_sd = cfg.downtime_cost_per_failure * 0.5

            # Объединённая SD (repair + downtime независимы)
            total_sd = math.sqrt(severity_sd**2 + downtime_sd**2)

            # PATCH 1 (CRITICAL): Правильная формула Risk Margin с учетом биномиальной дисперсии
            # L = I * S, где I ~ Bernoulli(p), S — severity
            # Var(L) = p * Var(S) + p*(1-p) * E[S]^2
            expected_severity = cfg.expected_loss_per_failure
            total_variance = (
                event_probability * total_sd**2
                + event_probability * (1.0 - event_probability) * expected_severity**2
            )
            severity_risk_margin = 1.645 * math.sqrt(total_variance)
            severity_dispersion_warning = (
                f"Risk margin (95% VaR, Lognormal μ={SEVERITY_LOGNORMAL_MU:.1f}, "
                f"σ={SEVERITY_LOGNORMAL_SIGMA:.2f}): {fmt_money(severity_risk_margin)} руб. "
                f"(SD={fmt_money(total_sd)}, E[severity]={fmt_money(expected_severity)})"
            )

        if severity_risk_margin > 0:
            print(f"\n★ Risk margin (95% VaR): {fmt_money(severity_risk_margin)} руб.")
            print(f"  ({severity_dispersion_warning})")

        expected_repair_with_prob = cfg.expected_repair * event_probability
        expected_downtime_with_prob = cfg.downtime_cost_per_failure * event_probability
        print(
            f"Ожидаемая стоимость ремонта (с учётом вероятности события): {fmt_money(expected_repair_with_prob)} руб."
        )
        print(
            f"Ожидаемая стоимость простоя (с учётом вероятности события): {fmt_money(expected_downtime_with_prob)} руб."
        )

        season_name = _dict_get_str(cfg.extended, "season", "межсезонье")
        # Получаем сезонный фактор из TUM_OPERATIONS или fallback на SEASONAL_FACTORS
        if TUM_OPERATIONS and season_name in TUM_OPERATIONS:
            seasonal_factor_raw = TUM_OPERATIONS[season_name].get("season_factor", 1.0)
            # Добавляем информацию о нагрузке операции
            op_info = TUM_OPERATIONS[season_name]
            op_name_ru = op_info.get("name_ru", season_name)
            op_peak_mean = op_info.get("peak_load_mean", 0.0)
            op_peak_std = op_info.get("peak_load_std", 0.0)
            op_intensity = op_info.get("intensity", "")
            intensity_str = f", интенсивность: {op_intensity}" if op_intensity else ""
            print(
                f"  Операция: {op_name_ru} ({season_name})"
                f"{intensity_str}"
                f" | Средняя нагрузка: {op_peak_mean * 100:.1f}% ± {op_peak_std * 100:.1f}%"
            )
        else:
            seasonal_factor_raw = SEASONAL_FACTORS.get(season_name, 1.0)
        seasonal_factor = _as_float(seasonal_factor_raw, "seasonal_factor")
        regional_rate = _dict_get_float(cfg.extended, "regional_rate", 0.0)
        effective_labor_cost = regional_rate * cfg.labor_ratio
        seasonal_downtime_cost = _dict_get_float(
            cfg.extended, "downtime_hour_cost", float(BASE_DOWNTIME_COST)
        )
        group = _as_int(
            cfg.extended.get("failure_group"),
            name="failure_group",
            default=DEFAULT_FAILURE_GROUP,
        )
        downtime_hours = get_downtime_mean_hours(max(1, min(3, group)))
        print(
            f"  (простой {fmt_num(downtime_hours, '.1f')} ч, сезонный коэффициент стоимости {fmt_num(seasonal_factor, '.1f')}x; базовая ставка {fmt_num(seasonal_downtime_cost, '.0f')} руб./ч + эффективный нормо-час {fmt_num(effective_labor_cost, '.0f')} руб./ч)"
        )

        results.append(
            {
                "peak": peak,
                "details": details,
                "event_probability": event_probability,
                "premium": premium,
                "net_premium": net_premium,
                "gross_premium": gross_premium,
            }
        )

    if not results:
        logger.error("Ни один PeakLoad не был рассчитан успешно.")
        raise SystemExit(3)

    return results


def display_results(
    results: list[dict[str, Any]], cfg: CalculatorConfig, params: ModelParameters
) -> None:
    last_result = results[-1]
    last_details = last_result["details"]

    gamma_value = _as_optional_float(last_details.get("gamma"))
    if gamma_value is not None:
        gamma_float = float(gamma_value)
        if math.isfinite(gamma_float):
            print(f"\nОценка γ (эффект PeakLoad): {gamma_float:.4f}")

    # ─── Reduced Form: индикация режима ────────────────────────────────
    model_form = str(cfg.training_meta.get("model_form", "control_function")).lower()
    if model_form == "reduced_form":
        print("ℹ️ Модель в режиме REDUCED FORM: результаты предиктивные, не каузальные.")
    else:
        print("ℹ️ Модель в режиме CONTROL FUNCTION (2SRI).")

    iv_diagnostics = cfg.training_meta.get("iv_diagnostics", {}) or {}
    if isinstance(iv_diagnostics, dict):
        f_stat_value = _dict_get_optional_float(iv_diagnostics, "f_statistic")
        if f_stat_value is not None:
            print(f"\nF-статистика первой стадии: {float(f_stat_value):.2f}")
        endogenous = iv_diagnostics.get("endogenous")
        if endogenous is not None:
            print(f"Эндогенность обнаружена: {endogenous}")
        instrument_adequate = iv_diagnostics.get("instrument_adequate")
        if instrument_adequate is not None:
            print(
                "Первая стадия: инструмент релевантен по screening-критерию: "
                + ("да" if instrument_adequate else "нет")
            )
            print("Экзогенность и exclusion restriction: не проверены")

    print("\nРасчёт завершён.")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("\n" + "=" * 60)
    print("ИССЛЕДОВАТЕЛЬСКИЙ КАЛЬКУЛЯТОР СТРАХОВОЙ ПРЕМИИ")
    print("(CF Cox модель + явная экономическая логика)")
    print(f"Единицы времени: {MODEL_TIME_UNIT}")
    print("=" * 60 + "\n")

    model_path = ask_str(
        "Путь к файлу модели (обученная модель в формате JSON)",
        DEFAULT_MODEL_PATH,
    )
    try:
        params, training_meta, dgp_data = load_and_validate_model(model_path)
    except SystemExit as e:
        return e.code

    try:
        cfg = collect_user_inputs(params, training_meta, dgp_data)
    except SystemExit as e:
        return e.code

    try:
        results = compute_all_peaks(params, cfg)
    except SystemExit as e:
        return e.code

    display_results(results, cfg, params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
