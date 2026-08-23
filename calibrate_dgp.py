#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_dgp.py (v3.1.1)
══════════════════════════════════════════════════════════════════════
Исправленная версия semi-synthetic DGP calibration.

Основные изменения v3.1.1:
• Устранены типовые ошибки:
  - Optional значения не передаются напрямую в float/format;
  - добавлены safe-float helpers;
  - minimize вызывается через обычную callable-обёртку.
• Убраны слишком широкие except Exception.
• Дублирующая логика генерации survival/metrics вынесена в helper.
• Локальные переменные в функциях переведены в lowercase.
• JSON сериализуется без NaN.
• Аргументы командной строки обрабатываются argparse.
• Other включён в целевые MTBF по умолчанию.
• Цензурирование по умолчанию согласовано с Hours:
  C_i = Hours_i * CALIBRATION_HORIZON_DAYS / DAYS_PER_YEAR.
• Калибруются только baseline_hazard и brand effects.
  beta_* и gamma считаются фиксированными экспертными параметрами.
══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


# ═══════════════════════════════════════════════════════════════════════
# 1. ГЛОБАЛЬНЫЕ НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════════════

SEED = 12345

# Размеры выборок
N_CALIB = 20_000
MC_REPS = 80
MC_N = 2_000
N_REPORT = 50_000

# Weibull
WEIBULL_SHAPE = 1.88

# Эндогенность: фиксированные экспертные параметры
DEFAULT_RHO = 0.7
DEFAULT_DELTA = 0.7

# Клиппинг линейного предиктора
CLIP_MIN = -10.0
CLIP_MAX = 10.0
CLIP_THRESHOLD = 0.01
CLIP_PENALTY = 10.0

# Веса целевой функции
PROBABILITY_WEIGHT = 1.0
FLEET_WEIGHT = 1.0
BRAND_RIDGE = 1e-4

# Потеря приемлема только ниже этого порога
MAX_ACCEPTABLE_LOSS = 1e6

# ─── Единицы времени ───────────────────────────────────────────────────
MODEL_TIME_UNIT = "engine_hours"
MTBF_INPUT_UNIT = "engine_hours"
MTBF_TO_MODEL_TIME_FACTOR = 1.0

# ─── Календарная конверсия ─────────────────────────────────────────────
DAYS_PER_YEAR = 365.0
DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY = 8.0  # только для legacy-диагностики

# ─── Горизонт калибровки ──────────────────────────────────────────────
CALIBRATION_HORIZON_DAYS = 214.0
CALIBRATION_HORIZON_ENGINE_HOURS = (
    CALIBRATION_HORIZON_DAYS * DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY
)

# ─── Режим цензурирования ──────────────────────────────────────────────
# individual_calendar:
#   C_i = Hours_i * CALIBRATION_HORIZON_DAYS / DAYS_PER_YEAR
# constant_engine_hours:
#   C_i = CALIBRATION_HORIZON_ENGINE_HOURS для всех
CENSORING_MODE = "individual_calendar"

# ─── Ограничение годовой наработки ─────────────────────────────────────
HOURS_CLIP_MAX = 6_000.0

# ─── MAJOR_FAILURE_SHARE ──────────────────────────────────────────────
MAJOR_FAILURE_SHARE = 0.30

# ─── Сценарий по умолчанию ─────────────────────────────────────────────
DEFAULT_SCENARIO = "baseline"


# ═══════════════════════════════════════════════════════════════════════
# 2. СЦЕНАРИИ MTBF
# ═══════════════════════════════════════════════════════════════════════

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "optimistic": {
        "description": "Новая техника, отличное ТО, лёгкие условия",
        "mtbf_all_engine_hours": 6000.0,
        "major_failure_share": 0.20,
    },
    "baseline": {
        "description": "Средний парк РФ/КЗ, стандартное ТО",
        "mtbf_all_engine_hours": 3000.0,
        "major_failure_share": 0.30,
    },
    "pessimistic": {
        "description": "Старый парк, тяжёлые почвы, дефицит запчастей",
        "mtbf_all_engine_hours": 1200.0,
        "major_failure_share": 0.45,
    },
}


def get_scenario(scenario_name: str) -> Dict[str, Any]:
    """Возвращает параметры сценария по имени."""
    name = scenario_name.strip().lower()
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario: '{scenario_name}'. "
            f"Valid: {sorted(SCENARIOS.keys())}"
        )
    return SCENARIOS[name]


def compute_mtbf_major(
    mtbf_all: float,
    major_failure_share: float,
) -> float:
    """MTBF_major = MTBF_all / MAJOR_FAILURE_SHARE."""
    mtbf_all_float = _finite_float_or_none(mtbf_all)
    share_float = _finite_float_or_none(major_failure_share)

    if mtbf_all_float is None or mtbf_all_float <= 0.0:
        raise ValueError("MTBF must be positive")
    if share_float is None or share_float <= 0.0:
        raise ValueError("major_failure_share must be positive")

    return mtbf_all_float / share_float


# ═══════════════════════════════════════════════════════════════════════
# 3. BRAND MAPPING
# ═══════════════════════════════════════════════════════════════════════

BRAND_MAP: Dict[int, str] = {
    0: "MTZ82",
    1: "Versatile280",
    2: "NewHollandT9",
    3: "DT75",
    4: "Other",
}

BRAND_TO_CODE: Dict[str, int] = {v: k for k, v in BRAND_MAP.items()}

BRAND_ALIASES: Dict[str, int] = {
    "mtz82": 0,
    "mtz-82": 0,
    "мтз-82": 0,
    "мтз 82": 0,
    "мтз82": 0,
    "versatile280": 1,
    "versatile 280": 1,
    "версатиль 280": 1,
    "newhollandt9": 2,
    "new holland t9": 2,
    "new holland t9.505": 2,
    "new holland t9.615": 2,
    "нью холланд t9": 2,
    "dt75": 3,
    "dt-75": 3,
    "дт-75": 3,
    "дт75": 3,
    "other": 4,
    "прочее": 4,
    "другой": 4,
    "unknown": 4,
}

BRAND_PROB_RU: Dict[int, float] = {
    0: 0.35,
    1: 0.08,
    2: 0.07,
    3: 0.10,
    4: 0.40,
}

DEFAULT_BRAND_PROB = BRAND_PROB_RU


# ═══════════════════════════════════════════════════════════════════════
# 4. X STANDARDIZATION
# ═══════════════════════════════════════════════════════════════════════

# v3.1:
# • x_brand удалён из first stage, чтобы не смешивать ordinal brand
#   с отдельными brand effects.
# • x_power стандартизован ближе к фактическому распределению.
X_STANDARDIZATION: Dict[str, Dict[str, Any]] = {
    "x_age": {
        "raw_col": "Age",
        "shift": 10.0,
        "scale": 10.0,
    },
    "x_hours": {
        "raw_col": "Hours",
        "shift": 1350.0,
        "scale": 1350.0,
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
    "x_power": {
        "raw_col": "Power",
        "shift": 180.0,
        "scale": 80.0,
    },
}


def standardize_x(name: str, raw_values: np.ndarray) -> np.ndarray:
    """Применяет фиксированную стандартизацию X."""
    info = X_STANDARDIZATION.get(name)
    if info is None:
        raise KeyError(f"Unknown standardized X column: {name}")

    raw_values = np.asarray(raw_values, dtype=float)
    shift = info.get("shift", None)
    scale = info.get("scale", None)

    if shift is None or scale is None:
        return raw_values

    shift_float = _finite_float_or_none(shift)
    scale_float = _finite_float_or_none(scale)

    if shift_float is None or scale_float is None:
        raise ValueError(f"Invalid standardization constants for {name}")
    if scale_float == 0.0:
        raise ValueError(f"Zero standardization scale for {name}")

    return (raw_values - shift_float) / scale_float


# ═══════════════════════════════════════════════════════════════════════
# 5. КОЭФФИЦИЕНТЫ DGP
# ═══════════════════════════════════════════════════════════════════════

FIRST_STAGE_INTERCEPT = 10.0
STRUCTURAL_INTERCEPT = 10.0

FIRST_STAGE_Z_COEF = 0.5
FIRST_STAGE_AGE_COEF = 0.15
FIRST_STAGE_HOURS_COEF = 0.10
FIRST_STAGE_CLIMATE_COEF = 0.20
FIRST_STAGE_SOIL_COEF = 0.15
FIRST_STAGE_BRAND_COEF = 0.0  # v3.1: brand убран из first stage
FIRST_STAGE_POWER_COEF = 0.08

# Фиксированные экспертные структурные коэффициенты.
# В v3.1 они НЕ оптимизируются, чтобы убрать недоопределённость.
FIXED_BETA_AGE = 0.20
FIXED_BETA_HOURS = 0.10
FIXED_BETA_CLIMATE = 0.20
FIXED_BETA_SOIL = 0.12
FIXED_BETA_POWER = -0.05
FIXED_GAMMA = 0.5

GAMMA_FACTOR = math.gamma(1.0 + 1.0 / float(WEIBULL_SHAPE))


# ═══════════════════════════════════════════════════════════════════════
# 6. ПАРАМЕТРЫ ОПТИМИЗАЦИИ
# ═══════════════════════════════════════════════════════════════════════

# v3.1: оптимизируются только baseline hazard и brand effects.
# Порядок:
#   0 log_baseline_hazard
#   1 brand_effect Versatile280
#   2 brand_effect NewHollandT9
#   3 brand_effect DT75
#   4 brand_effect Other
BOUNDS: List[Tuple[float, float]] = [
    (-25.0, 5.0),   # log_baseline_hazard
    (-6.0, 6.0),    # Versatile280
    (-6.0, 6.0),    # NewHollandT9
    (-6.0, 6.0),    # DT75
    (-6.0, 6.0),    # Other
]

FREE_PARAMETER_NAMES = [
    "log_baseline_hazard",
    "brand_effect_Versatile280",
    "brand_effect_NewHollandT9",
    "brand_effect_DT75",
    "brand_effect_Other",
]


def _clip_to_bounds(x0: np.ndarray) -> np.ndarray:
    lower = np.array([low for low, _ in BOUNDS], dtype=float)
    upper = np.array([high for _, high in BOUNDS], dtype=float)
    return np.clip(x0, lower, upper)


# ═══════════════════════════════════════════════════════════════════════
# 7. УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════════════

def _finite_float_or_none(value: Any) -> Optional[float]:
    """Возвращает float только если значение конечно, иначе None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None

    if math.isfinite(out):
        return out

    return None


def _fmt(value: Any, decimals: int = 6) -> str:
    """Безопасное форматирование Optional[float]."""
    out = _finite_float_or_none(value)
    if out is None:
        return "NA"
    return f"{out:.{decimals}f}"


def _to_native(obj: Any) -> Any:
    """Рекурсивная конвертация numpy-типов и NaN/inf для json.dump."""
    if obj is None:
        return None

    if isinstance(obj, dict):
        out: Dict[Any, Any] = {}
        for k, v in obj.items():
            if isinstance(k, np.integer):
                key: Any = int(k)
            else:
                key = k
            out[key] = _to_native(v)
        return out

    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj]

    return obj


def _standardize_array(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x

    m = float(np.mean(x))
    s = float(np.std(x, ddof=0))

    if (not math.isfinite(s)) or s < floor:
        s = floor

    return (x - m) / s


def normalize_probabilities(
    prob_dict: Dict[int, float],
    name: str = "probabilities",
) -> Dict[int, float]:
    """Нормализует неотрицательные вероятности."""
    if not prob_dict:
        raise ValueError(f"{name} is empty")

    clean: Dict[int, float] = {}

    for code, p in prob_dict.items():
        p_float = _finite_float_or_none(p)
        if p_float is None or p_float < 0.0:
            raise ValueError(f"{name} for code {code} is invalid: {p}")

        try:
            code_int = int(code)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} has invalid code: {code}") from exc

        clean[code_int] = p_float

    total = sum(clean.values())
    if total <= 0.0:
        raise ValueError(f"{name} sum is zero")

    return {code: p / total for code, p in clean.items()}


def resolve_brand_code(value: Any, warn: bool = True) -> int:
    """Конвертирует brand-значение в канонический код 0..4."""
    if value is None:
        if warn:
            logger.warning("Missing brand value mapped to Other (code 4)")
        return 4

    if isinstance(value, (int, np.integer)):
        code = int(value)
        if 0 <= code <= 4:
            return code
        if warn:
            logger.warning(
                "Numeric brand code %s outside [0, 4], mapped to Other",
                value,
            )
        return 4

    if isinstance(value, (float, np.floating)):
        value_float = _finite_float_or_none(value)
        if value_float is None:
            if warn:
                logger.warning(
                    "Non-finite numeric brand value %s mapped to Other",
                    value,
                )
            return 4

        rounded = round(value_float)
        if not math.isclose(value_float, rounded, abs_tol=1e-6) and warn:
            logger.warning(
                "Non-integer numeric brand value %s rounded to %s",
                value,
                rounded,
            )

        code = int(rounded)
        if 0 <= code <= 4:
            return code

        if warn:
            logger.warning(
                "Numeric brand code %s outside [0, 4], mapped to Other",
                value,
            )
        return 4

    try:
        text = str(value).strip().lower()
    except (TypeError, ValueError):
        if warn:
            logger.warning(
                "Unconvertible brand value %r mapped to Other",
                value,
            )
        return 4

    text = " ".join(text.split())

    if not text:
        if warn:
            logger.warning("Empty brand value mapped to Other (code 4)")
        return 4

    if text in BRAND_ALIASES:
        return BRAND_ALIASES[text]

    if text in BRAND_TO_CODE:
        return BRAND_TO_CODE[text]

    if warn:
        logger.warning("Unknown brand value '%s' mapped to Other", value)
    return 4


def _weighted_mean(
    values: Dict[int, float],
    weights: Dict[int, float],
) -> Optional[float]:
    """Взвешенное среднее по кодам брендов с пропуском NaN."""
    numerator = 0.0
    denominator = 0.0

    for code, w in weights.items():
        weight_float = _finite_float_or_none(w)
        if weight_float is None or weight_float <= 0.0:
            continue

        value_float = _finite_float_or_none(values.get(code, np.nan))
        if value_float is None:
            continue

        numerator += value_float * weight_float
        denominator += weight_float

    if denominator <= 0.0:
        return None

    return numerator / denominator


def _stats(values: List[float]) -> Dict[str, Any]:
    """Безопасная статистика для Monte Carlo значений."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "n_replications": 0,
            "mc_mean": None,
            "mc_sd": None,
            "mc_q025": None,
            "mc_q975": None,
        }

    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else None

    if arr.size > 1:
        q025, q975 = np.percentile(arr, [2.5, 97.5])
        q025 = float(q025)
        q975 = float(q975)
    else:
        q025 = q975 = mean

    return {
        "n_replications": int(arr.size),
        "mc_mean": mean,
        "mc_sd": sd,
        "mc_q025": q025,
        "mc_q975": q975,
    }


def _map_code_float_to_name(values: Dict[int, float]) -> Dict[str, float]:
    """Конвертирует dict[code -> float] в dict[brand_name -> float]."""
    out: Dict[str, float] = {}

    for code, value in values.items():
        value_float = _finite_float_or_none(value)
        if value_float is None:
            continue

        try:
            code_int = int(code)
        except (TypeError, ValueError):
            continue

        out[BRAND_MAP.get(code_int, str(code_int))] = value_float

    return out


# ═══════════════════════════════════════════════════════════════════════
# 8. CENSORING
# ═══════════════════════════════════════════════════════════════════════

def make_censoring_times(df: pd.DataFrame) -> np.ndarray:
    """
    Возвращает индивидуальное или постоянное время цензурирования.

    individual_calendar:
        C_i = Hours_i * CALIBRATION_HORIZON_DAYS / DAYS_PER_YEAR

    constant_engine_hours:
        C_i = CALIBRATION_HORIZON_ENGINE_HOURS
    """
    n = len(df)

    if CENSORING_MODE == "constant_engine_hours":
        return np.full(n, float(CALIBRATION_HORIZON_ENGINE_HOURS), dtype=float)

    if CENSORING_MODE == "individual_calendar":
        if "Hours" not in df.columns:
            raise ValueError(
                "CENSORING_MODE=individual_calendar requires Hours column"
            )

        hours = np.asarray(df["Hours"], dtype=float)
        hours = np.nan_to_num(
            hours,
            nan=0.0,
            posinf=HOURS_CLIP_MAX,
            neginf=0.0,
        )
        hours = np.clip(hours, 0.0, HOURS_CLIP_MAX)

        censoring = (
            hours
            * float(CALIBRATION_HORIZON_DAYS)
            / float(DAYS_PER_YEAR)
        )

        censoring = np.nan_to_num(
            censoring,
            nan=1e-6,
            posinf=1e12,
            neginf=1e-6,
        )
        return np.maximum(censoring, 1e-6)

    raise ValueError(f"Unknown CENSORING_MODE: {CENSORING_MODE}")


# ═══════════════════════════════════════════════════════════════════════
# 9. ОПЦИОНАЛЬНЫЙ XLSX LOADER
# ═══════════════════════════════════════════════════════════════════════

def load_targets_from_xlsx(
    path: Optional[PathLike],
) -> Optional[Dict[str, Any]]:
    """Загрузка реальных MTBF и долей брендов из xlsx."""
    if path is None:
        return None

    xlsx_path = Path(path)

    if not xlsx_path.exists():
        logger.warning("xlsx file not found: %s", xlsx_path)
        return None

    try:
        df = pd.read_excel(xlsx_path)
    except (OSError, ImportError, ValueError) as exc:
        logger.warning("Unable to read xlsx: %s", exc)
        return None

    if df.empty:
        logger.warning("xlsx file is empty: %s", xlsx_path)
        return None

    columns = {str(c).strip().lower(): c for c in df.columns}

    def find_column(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in columns:
                return columns[c]
        return None

    brand_col = find_column(
        ["brand", "brand_name", "марка", "tractor"]
    )
    mtbf_col = find_column(
        [
            "mtbf",
            "mtbf_target",
            "mtbf_major",
            "mtbf_major_engine_hours",
            "наработка на отказ",
        ]
    )
    prob_col = find_column(
        ["probability", "share", "доля"]
    )

    if brand_col is None or mtbf_col is None:
        logger.warning(
            "xlsx must contain brand and MTBF columns; found columns: %s",
            sorted(columns.keys()),
        )
        return None

    targets: Dict[int, float] = {}
    probs: Dict[int, float] = {}

    for _, row in df.iterrows():
        mtbf_cell = row[mtbf_col]

        try:
            if pd.isna(mtbf_cell):
                continue
        except (TypeError, ValueError):
            pass

        mtbf_raw = _finite_float_or_none(mtbf_cell)
        if mtbf_raw is None or mtbf_raw <= 0.0:
            logger.warning("Skipping xlsx row with invalid MTBF")
            continue

        brand_cell = row[brand_col]

        try:
            if pd.isna(brand_cell):
                code = resolve_brand_code(None)
            else:
                code = resolve_brand_code(brand_cell)
        except (TypeError, ValueError):
            code = resolve_brand_code(brand_cell)

        if code in targets:
            logger.warning(
                "Duplicate xlsx row for brand code %s; overwriting MTBF",
                code,
            )

        targets[code] = mtbf_raw

        if prob_col is not None:
            prob_raw = _finite_float_or_none(row[prob_col])
            if prob_raw is not None and prob_raw > 0.0:
                probs[code] = prob_raw

    if not targets:
        logger.warning("No valid MTBF targets found in xlsx")
        return None

    if probs:
        try:
            probs = normalize_probabilities(probs, name="xlsx probabilities")
        except ValueError as exc:
            logger.warning("Invalid xlsx probabilities: %s", exc)
            probs = {}

    return {
        "targets_by_code_raw": targets,
        "brand_prob_by_code": probs,
        "source": str(xlsx_path),
    }


# ═══════════════════════════════════════════════════════════════════════
# 10. ГЕНЕРАЦИЯ КОВАРИАТ
# ═══════════════════════════════════════════════════════════════════════

def generate_covariates(
    n: int,
    rng: np.random.Generator,
    brand_prob_by_code: Dict[int, float],
) -> pd.DataFrame:
    """
    Генерация ковариат.

    Age     ~ Lognormal(μ=2.30, σ=0.60), clip [0, 30]
    Hours   ~ Exponential(1350), clip [0, HOURS_CLIP_MAX]
    Climate ~ Beta(2.5, 1.5)
    Soil    ~ Beta(2.0, 2.5)
    Power   ~ Normal(180, 80), clip [50, 350]
    Brand   ~ Categorical(p)
    """
    if n <= 0:
        return pd.DataFrame(
            columns=["Age", "Hours", "Climate", "Soil", "Power", "Brand"]
        )

    if not brand_prob_by_code:
        logger.warning("brand_prob_by_code is empty; using DEFAULT_BRAND_PROB")
        brand_prob_by_code = DEFAULT_BRAND_PROB

    brand_prob_by_code = normalize_probabilities(
        brand_prob_by_code,
        name="brand_prob_by_code",
    )

    age = rng.lognormal(mean=2.30, sigma=0.60, size=n)
    age = np.clip(age, 0.0, 30.0)

    hours = rng.exponential(1350.0, size=n)
    hours = np.clip(hours, 0.0, HOURS_CLIP_MAX)

    climate = rng.beta(2.5, 1.5, size=n)
    soil = rng.beta(2.0, 2.5, size=n)

    power = rng.normal(180.0, 80.0, size=n)
    power = np.clip(power, 50.0, 350.0)

    codes = sorted(brand_prob_by_code.keys())
    probs = np.array([brand_prob_by_code[c] for c in codes], dtype=float)

    if probs.size == 0:
        raise ValueError("Brand probabilities are empty")

    if np.any(~np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("Invalid brand probabilities")

    probs = probs / probs.sum()
    brand_code = rng.choice(codes, size=n, p=probs)

    return pd.DataFrame(
        {
            "Age": age,
            "Hours": hours,
            "Climate": climate,
            "Soil": soil,
            "Power": power,
            "Brand": brand_code.astype(int),
        }
    )


# ═══════════════════════════════════════════════════════════════════════
# 11. ГЕНЕРАЦИЯ ОШИБОК
# ═══════════════════════════════════════════════════════════════════════

def generate_errors(
    n: int,
    rng: np.random.Generator,
    rho: float = DEFAULT_RHO,
) -> Tuple[np.ndarray, np.ndarray]:
    """Коррелированные eps_D и U."""
    rho_float = _finite_float_or_none(rho)
    if rho_float is None:
        rho_float = DEFAULT_RHO

    rho_float = float(np.clip(rho_float, -0.999, 0.999))
    cov = np.array([[1.0, rho_float], [rho_float, 1.0]], dtype=float)

    try:
        draws = rng.multivariate_normal(
            mean=[0.0, 0.0],
            cov=cov,
            size=n,
        )
        eps_d, u = draws[:, 0], draws[:, 1]
    except ValueError as exc:
        logger.error(
            "multivariate_normal failed, fallback to independent errors: %s",
            exc,
        )
        draws = rng.normal(0.0, 1.0, size=(n, 2))
        eps_d, u = draws[:, 0], draws[:, 1]

    eps_d = _standardize_array(eps_d)
    u = _standardize_array(u)

    return eps_d, u


# ═══════════════════════════════════════════════════════════════════════
# 12. ГЕНЕРАЦИЯ PEAKLOAD
# ═══════════════════════════════════════════════════════════════════════

def generate_peakload(
    df: pd.DataFrame,
    rng: np.random.Generator,
    rho: float = DEFAULT_RHO,
) -> Dict[str, Any]:
    """
    Генерация инструмента Z, ошибок и PeakLoad.

    v3.1: brand исключён из first stage.
    """
    z = rng.normal(0.0, 1.0, size=len(df))
    z = _standardize_array(z)

    eps_d, u = generate_errors(len(df), rng, rho=rho)

    x_age = standardize_x("x_age", df["Age"].to_numpy())
    x_hours = standardize_x("x_hours", df["Hours"].to_numpy())
    x_climate = standardize_x("x_climate", df["Climate"].to_numpy())
    x_soil = standardize_x("x_soil", df["Soil"].to_numpy())
    x_power = standardize_x("x_power", df["Power"].to_numpy())

    peakload = (
        FIRST_STAGE_INTERCEPT
        + FIRST_STAGE_Z_COEF * z
        + FIRST_STAGE_AGE_COEF * x_age
        + FIRST_STAGE_HOURS_COEF * x_hours
        + FIRST_STAGE_CLIMATE_COEF * x_climate
        + FIRST_STAGE_SOIL_COEF * x_soil
        + FIRST_STAGE_POWER_COEF * x_power
        + eps_d
    )

    peakload_c = peakload - STRUCTURAL_INTERCEPT

    return {
        "Z": z,
        "eps_D": eps_d,
        "U": u,
        "PeakLoad": peakload,
        "PeakLoad_c": peakload_c,
        "x_age": x_age,
        "x_hours": x_hours,
        "x_climate": x_climate,
        "x_soil": x_soil,
        "x_power": x_power,
    }


# ═══════════════════════════════════════════════════════════════════════
# 13. ГЕНЕРАЦИЯ ВРЕМЕНИ ВЫЖИВАНИЯ (Weibull PH)
# ═══════════════════════════════════════════════════════════════════════

def generate_survival_time(
    df: pd.DataFrame,
    peakload_c: np.ndarray,
    u: np.ndarray,
    beta_age: float,
    beta_hours: float,
    beta_climate: float,
    beta_soil: float,
    beta_power: float,
    gamma: float,
    baseline_hazard: float,
    brand_effects_by_code: Dict[int, float],
    delta: float = DEFAULT_DELTA,
    censoring_time: Optional[Union[float, np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Генерация времени выживания (Weibull PH).

    T = (-log(U_event) / (baseline_hazard * exp(lp)))^(1/shape)

    Единицы: мото-часы.
    """
    if rng is None:
        rng = np.random.default_rng()

    baseline_hazard_float = _finite_float_or_none(baseline_hazard)
    if baseline_hazard_float is None or baseline_hazard_float <= 0.0:
        raise ValueError("baseline_hazard must be positive")

    n = len(df)

    x_age = standardize_x("x_age", df["Age"].to_numpy())
    x_hours = standardize_x("x_hours", df["Hours"].to_numpy())
    x_climate = standardize_x("x_climate", df["Climate"].to_numpy())
    x_soil = standardize_x("x_soil", df["Soil"].to_numpy())
    x_power = standardize_x("x_power", df["Power"].to_numpy())

    brand_lp = np.array(
        [
            float(brand_effects_by_code.get(int(code), 0.0))
            for code in df["Brand"].to_numpy()
        ],
        dtype=float,
    )

    lp = (
        gamma * peakload_c
        + beta_age * x_age
        + beta_hours * x_hours
        + beta_climate * x_climate
        + beta_soil * x_soil
        + beta_power * x_power
        + brand_lp
        + delta * u
    )

    lp_original = lp.copy()
    lp = np.clip(lp, CLIP_MIN, CLIP_MAX)

    clipped_fraction = float(
        np.mean((lp_original < CLIP_MIN) | (lp_original > CLIP_MAX))
    )

    u_event = rng.uniform(
        low=np.nextafter(0.0, 1.0),
        high=1.0,
        size=n,
    )

    denom = baseline_hazard_float * np.exp(lp)
    denom = np.clip(denom, 1e-300, 1e300)

    shape = float(WEIBULL_SHAPE)

    time_true = np.power(-np.log(u_event) / denom, 1.0 / shape)
    time_true = np.nan_to_num(
        time_true,
        nan=1e-12,
        posinf=1e12,
        neginf=1e-12,
    )
    time_true = np.maximum(time_true, 1e-12)

    if censoring_time is None:
        censoring = make_censoring_times(df)
    elif np.isscalar(censoring_time):
        censoring_scalar = _finite_float_or_none(censoring_time)
        if censoring_scalar is None:
            censoring_scalar = float(CALIBRATION_HORIZON_ENGINE_HOURS)
        censoring = np.full(n, censoring_scalar, dtype=float)
    else:
        censoring = np.asarray(censoring_time, dtype=float)
        if censoring.size != n:
            raise ValueError("censoring_time array length mismatch")

    censoring = np.nan_to_num(
        censoring,
        nan=1e-6,
        posinf=1e12,
        neginf=1e-6,
    )
    censoring = np.maximum(censoring, 1e-6)

    t_observed = np.minimum(time_true, censoring)
    event = (time_true <= censoring).astype(int)

    return {
        "T": t_observed,
        "T_true": time_true,
        "C": censoring,
        "event": event,
        "clipped_fraction": clipped_fraction,
    }


# ═══════════════════════════════════════════════════════════════════════
# 14. РАСПАКОВКА ВЕКТОРА ПАРАМЕТРОВ
# ═══════════════════════════════════════════════════════════════════════

def unpack_reduced(theta: np.ndarray) -> Dict[str, Any]:
    """
    Распаковка уменьшенного вектора параметров v3.1.

    theta:
        0 log_baseline_hazard
        1 brand_effect Versatile280
        2 brand_effect NewHollandT9
        3 brand_effect DT75
        4 brand_effect Other
    """
    theta = np.asarray(theta, dtype=float)

    baseline_hazard = float(math.exp(theta[0]))

    brand_effects_by_code = {
        0: 0.0,              # MTZ82 reference
        1: float(theta[1]),  # Versatile280
        2: float(theta[2]),  # NewHollandT9
        3: float(theta[3]),  # DT75
        4: float(theta[4]),  # Other
    }

    return {
        "beta_age": float(FIXED_BETA_AGE),
        "beta_hours": float(FIXED_BETA_HOURS),
        "beta_climate": float(FIXED_BETA_CLIMATE),
        "beta_soil": float(FIXED_BETA_SOIL),
        "beta_power": float(FIXED_BETA_POWER),
        "gamma": float(FIXED_GAMMA),
        "baseline_hazard": baseline_hazard,
        "brand_effects_by_code": brand_effects_by_code,
    }


# ═══════════════════════════════════════════════════════════════════════
# 15. MTBF / FAILURE PROBABILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════

def calculate_mtbf(
    df: pd.DataFrame,
    time_true: np.ndarray,
    target_codes: List[int],
) -> Dict[int, float]:
    """MTBF из истинных (не цензурированных) времён."""
    tmp = df.copy()
    tmp["T_true"] = time_true

    result: Dict[int, float] = {}

    for code in target_codes:
        subset = tmp[tmp["Brand"] == code]
        if len(subset) < 10:
            result[code] = np.nan
        else:
            result[code] = float(subset["T_true"].mean())

    return result


def calculate_failure_probability(
    df: pd.DataFrame,
    event: np.ndarray,
    target_codes: List[int],
) -> Dict[int, float]:
    """
    Доля отказов к горизонту цензурирования.

    Это cumulative incidence / failure probability, а не hazard rate.
    """
    tmp = df.copy()
    tmp["event"] = event

    result: Dict[int, float] = {}

    for code in target_codes:
        subset = tmp[tmp["Brand"] == code]
        if len(subset) < 10:
            result[code] = np.nan
        else:
            result[code] = float(subset["event"].mean())

    return result


def compute_target_event_rates_for_df(
    df: pd.DataFrame,
    targets_by_code_model_time: Dict[int, float],
) -> Dict[int, float]:
    """
    Целевая failure probability без ковариат для каждого бренда.

    Использует target MTBF и фактическое цензурирование df.
    """
    censoring = make_censoring_times(df)
    brand = df["Brand"].to_numpy()
    shape = float(WEIBULL_SHAPE)

    result: Dict[int, float] = {}

    for code, target in targets_by_code_model_time.items():
        target_float = _finite_float_or_none(target)
        if target_float is None or target_float <= 0.0:
            result[code] = np.nan
            continue

        scale = target_float / GAMMA_FACTOR
        if scale <= 0.0:
            result[code] = np.nan
            continue

        x = (censoring / scale) ** shape
        prob = -np.expm1(-x)

        mask = brand == code
        if mask.sum() == 0:
            result[code] = np.nan
        else:
            result[code] = float(prob[mask].mean())

    return result


def simulate_survival_from_peak(
    df: pd.DataFrame,
    peak_data: Dict[str, Any],
    params: Dict[str, Any],
    rng: np.random.Generator,
    censoring_time: Optional[Union[float, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Единая точка вызова generate_survival_time для готового peak_data."""
    return generate_survival_time(
        df,
        peak_data["PeakLoad_c"],
        peak_data["U"],
        beta_age=float(params["beta_age"]),
        beta_hours=float(params["beta_hours"]),
        beta_climate=float(params["beta_climate"]),
        beta_soil=float(params["beta_soil"]),
        beta_power=float(params["beta_power"]),
        gamma=float(params["gamma"]),
        baseline_hazard=float(params["baseline_hazard"]),
        brand_effects_by_code=params["brand_effects_by_code"],
        delta=DEFAULT_DELTA,
        censoring_time=censoring_time,
        rng=rng,
    )


def simulate_metrics_from_peak(
    df: pd.DataFrame,
    peak_data: Dict[str, Any],
    params: Dict[str, Any],
    target_codes: List[int],
    rng: np.random.Generator,
    censoring_time: Optional[Union[float, np.ndarray]] = None,
) -> Tuple[Dict[str, Any], Dict[int, float], Dict[int, float]]:
    """Генерация survival и расчёт MTBF / failure probability."""
    surv = simulate_survival_from_peak(
        df=df,
        peak_data=peak_data,
        params=params,
        rng=rng,
        censoring_time=censoring_time,
    )

    mtbf = calculate_mtbf(df, surv["T_true"], target_codes)
    failure_probability = calculate_failure_probability(
        df,
        surv["event"],
        target_codes,
    )

    return surv, mtbf, failure_probability


# ═══════════════════════════════════════════════════════════════════════
# 16. ЦЕЛЕВАЯ ФУНКЦИЯ ОПТИМИЗАЦИИ
# ═══════════════════════════════════════════════════════════════════════

class CalibrationObjective:
    """
    Common-random-numbers objective для калибровки v3.1.

    Целевые моменты:
    • MTBF по брендам;
    • failure probability по брендам;
    • fleet MTBF;
    • fleet failure probability.
    """

    def __init__(
        self,
        targets_by_code_model_time: Dict[int, float],
        brand_prob_by_code: Dict[int, float],
    ) -> None:
        self.targets = dict(targets_by_code_model_time)
        self.target_codes = list(self.targets.keys())

        self.brand_prob = normalize_probabilities(
            brand_prob_by_code,
            name="brand_prob_by_code",
        )

        self.rng = np.random.default_rng(SEED)

        self.df = generate_covariates(
            N_CALIB,
            self.rng,
            self.brand_prob,
        )

        self.censoring = make_censoring_times(self.df)

        self.peak_data = generate_peakload(
            self.df,
            self.rng,
            rho=DEFAULT_RHO,
        )

        self.target_event_rates = self._compute_target_event_rates()
        self.target_fleet_mtbf = self._compute_target_fleet_mtbf()
        self.target_fleet_probability = self._compute_target_fleet_probability()

        self._call_count = 0

    def _compute_target_event_rates(self) -> Dict[int, float]:
        return compute_target_event_rates_for_df(self.df, self.targets)

    def _compute_target_fleet_mtbf(self) -> Optional[float]:
        return _weighted_mean(self.targets, self.brand_prob)

    def _compute_target_fleet_probability(self) -> Optional[float]:
        return _weighted_mean(self.target_event_rates, self.brand_prob)

    def __call__(self, theta: np.ndarray) -> float:
        params = unpack_reduced(theta)

        local_rng = np.random.default_rng(SEED + 999)

        surv, mtbf, failure_probability = simulate_metrics_from_peak(
            df=self.df,
            peak_data=self.peak_data,
            params=params,
            target_codes=self.target_codes,
            rng=local_rng,
            censoring_time=self.censoring,
        )

        self._call_count += 1

        if self._call_count % 500 == 0:
            logger.info(
                "Calibration iter %d: clipped=%.6f",
                self._call_count,
                surv.get("clipped_fraction", 0.0),
            )

        loss = 0.0

        # ─── Бренд-специфичные MTBF и failure probability ─────────────
        for code, target in self.targets.items():
            target_float = _finite_float_or_none(target)
            if target_float is None or target_float <= 0.0:
                return 1e12

            predicted_mtbf = _finite_float_or_none(mtbf.get(code, np.nan))
            if predicted_mtbf is None or predicted_mtbf <= 0.0:
                return 1e12

            predicted_prob = _finite_float_or_none(
                failure_probability.get(code, np.nan)
            )
            if predicted_prob is None:
                return 1e12

            log_rel_error = math.log(predicted_mtbf / target_float)
            loss += log_rel_error ** 2

            target_prob = _finite_float_or_none(
                self.target_event_rates.get(code, np.nan)
            )

            if target_prob is not None and target_prob > 0.0:
                relative_prob_error = (
                    (predicted_prob - target_prob)
                    / max(target_prob, 1e-6)
                )
                loss += PROBABILITY_WEIGHT * relative_prob_error ** 2

        # ─── Fleet MTBF ────────────────────────────────────────────────
        predicted_fleet_mtbf = _weighted_mean(mtbf, self.brand_prob)

        if self.target_fleet_mtbf is not None:
            if (
                predicted_fleet_mtbf is None
                or predicted_fleet_mtbf <= 0.0
                or not math.isfinite(predicted_fleet_mtbf)
            ):
                return 1e12

            fleet_log_error = math.log(
                predicted_fleet_mtbf / self.target_fleet_mtbf
            )
            loss += FLEET_WEIGHT * fleet_log_error ** 2

        # ─── Fleet failure probability ─────────────────────────────────
        predicted_fleet_prob = _weighted_mean(
            failure_probability,
            self.brand_prob,
        )

        if self.target_fleet_probability is not None:
            if predicted_fleet_prob is None or not math.isfinite(predicted_fleet_prob):
                return 1e12

            if self.target_fleet_probability > 0.0:
                fleet_prob_error = (
                    (predicted_fleet_prob - self.target_fleet_probability)
                    / max(self.target_fleet_probability, 1e-6)
                )
                loss += FLEET_WEIGHT * fleet_prob_error ** 2

        # ─── Клиппинг ──────────────────────────────────────────────────
        clipped_fraction = _finite_float_or_none(
            surv.get("clipped_fraction", 0.0)
        )
        if clipped_fraction is not None and clipped_fraction > CLIP_THRESHOLD:
            loss += CLIP_PENALTY * (clipped_fraction - CLIP_THRESHOLD) ** 2

        # ─── Минимальный ridge на brand effects ────────────────────────
        loss += BRAND_RIDGE * float(np.sum(np.asarray(theta[1:]) ** 2))

        return float(loss)


# ═══════════════════════════════════════════════════════════════════════
# 17. НАЧАЛЬНАЯ ТОЧКА
# ═══════════════════════════════════════════════════════════════════════

def _make_initial_point(
    targets_by_code_model_time: Dict[int, float],
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Аналитическая начальная точка из формулы Weibull mean.

    E[T | lp=0] = Γ(1 + 1/b) · λ^(-1/b)
    => λ = (Γ(1 + 1/b) / MTBF)^b
    """
    if not targets_by_code_model_time:
        raise ValueError("targets_by_code_model_time is empty")

    reference_code = 0
    if reference_code not in targets_by_code_model_time:
        reference_code = list(targets_by_code_model_time.keys())[0]

    reference_mtbf = _finite_float_or_none(
        targets_by_code_model_time[reference_code]
    )
    if reference_mtbf is None or reference_mtbf <= 0.0:
        reference_mtbf = 1000.0

    baseline_hazard_initial = (
        GAMMA_FACTOR / reference_mtbf
    ) ** float(WEIBULL_SHAPE)

    log_baseline_hazard_initial = math.log(baseline_hazard_initial)

    brand_effects_initial: Dict[int, float] = {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
    }

    shape = float(WEIBULL_SHAPE)

    for code, target in targets_by_code_model_time.items():
        if code == reference_code:
            continue

        target_float = _finite_float_or_none(target)
        if target_float is None or target_float <= 0.0:
            continue

        brand_effects_initial[code] = -shape * math.log(
            target_float / reference_mtbf
        )

    x0 = np.array(
        [
            log_baseline_hazard_initial,
            brand_effects_initial[1],
            brand_effects_initial[2],
            brand_effects_initial[3],
            brand_effects_initial[4],
        ],
        dtype=float,
    )

    x0 = _clip_to_bounds(x0)

    return x0, brand_effects_initial


def _make_x0(
    rng: np.random.Generator,
    targets_by_code_model_time: Dict[int, float],
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Случайная пертурбация аналитической начальной точки."""
    x0, _ = _make_initial_point(targets_by_code_model_time)

    x0 = x0 + rng.normal(0.0, 0.05, size=x0.shape)
    x0 = _clip_to_bounds(x0)

    brand_effects_initial = {
        0: 0.0,
        1: float(x0[1]),
        2: float(x0[2]),
        3: float(x0[3]),
        4: float(x0[4]),
    }

    return x0, brand_effects_initial


# ═══════════════════════════════════════════════════════════════════════
# 18. КАЛИБРОВКА
# ═══════════════════════════════════════════════════════════════════════

def calibrate(
    targets_by_code_model_time: Dict[int, float],
    brand_prob_by_code: Dict[int, float],
) -> Tuple[Any, Dict[str, Any], CalibrationObjective]:
    """Multi-start калибровка v3.1."""
    objective = CalibrationObjective(
        targets_by_code_model_time=targets_by_code_model_time,
        brand_prob_by_code=brand_prob_by_code,
    )

    n_starts = 3
    best_loss = math.inf
    best_result = None

    rng = np.random.default_rng(SEED + 5000)

    print("\nStarting calibration (multi-start, reduced parameter set)...")

    def objective_fun(theta: np.ndarray) -> float:
        return float(objective(theta))

    options: Dict[str, Union[int, float]] = {
        "maxiter": 300,
        "ftol": 1e-10,
        "gtol": 1e-8,
        "maxls": 50,
    }

    for i in range(n_starts):
        x0, _ = _make_x0(rng, targets_by_code_model_time)

        try:
            res = minimize(
                fun=objective_fun,
                x0=[float(v) for v in np.asarray(x0, dtype=float)],
                method="L-BFGS-B",
                bounds=BOUNDS,
                options=cast(Any, options),
            )

            loss_value = _finite_float_or_none(res.fun)
            if loss_value is None:
                loss_value = math.inf

            if loss_value < best_loss:
                best_loss = loss_value
                best_result = res

            print(f"  Start {i + 1:2d}: loss={loss_value:.6f}")

        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"  Start {i + 1:2d}: FAILED ({exc})")
            continue

    if best_result is None:
        raise RuntimeError("All calibration starts failed.")

    print(f"\nBest loss: {best_loss:.6f}")

    if not best_result.success:
        print("NOTE:", best_result.message)

    calibrated = unpack_reduced(best_result.x)

    calibrated["optimizer_success"] = bool(best_result.success)
    calibrated["optimizer_message"] = str(best_result.message)
    calibrated["loss"] = float(best_loss)
    calibrated["n_starts"] = int(n_starts)
    calibrated["free_parameters"] = FREE_PARAMETER_NAMES

    # Финальная диагностика на калибровочной выборке
    local_rng = np.random.default_rng(SEED + 999)

    surv, _, _ = simulate_metrics_from_peak(
        df=objective.df,
        peak_data=objective.peak_data,
        params=calibrated,
        target_codes=objective.target_codes,
        rng=local_rng,
        censoring_time=objective.censoring,
    )

    calibrated["clipped_fraction"] = _finite_float_or_none(
        surv.get("clipped_fraction", 0.0)
    )
    calibrated["target_event_rates"] = objective.target_event_rates
    calibrated["target_fleet_mtbf"] = objective.target_fleet_mtbf
    calibrated["target_fleet_probability"] = objective.target_fleet_probability

    return best_result, calibrated, objective


# ═══════════════════════════════════════════════════════════════════════
# 19. MTBF / PROBABILITY CONSISTENCY CHECK
# ═══════════════════════════════════════════════════════════════════════

def build_target_consistency(
    targets_by_code_model_time: Dict[int, float],
    brand_prob_by_code: Dict[int, float],
    n: int = 20_000,
) -> Dict[str, Any]:
    """
    Диагностика целевых MTBF и failure probability.

    Для каждого целевого MTBF считает Weibull scale и ожидаемую
    failure probability при текущем режиме цензурирования.
    """
    rng = np.random.default_rng(SEED + 3000)

    hours = rng.exponential(1350.0, size=n)
    hours = np.clip(hours, 0.0, HOURS_CLIP_MAX)

    df = pd.DataFrame({"Hours": hours})
    censoring = make_censoring_times(df)

    shape = float(WEIBULL_SHAPE)

    result: Dict[str, Any] = {
        "censoring_mode": CENSORING_MODE,
        "calibration_horizon_days": float(CALIBRATION_HORIZON_DAYS),
        "days_per_year": float(DAYS_PER_YEAR),
        "constant_engine_hours": float(CALIBRATION_HORIZON_ENGINE_HOURS),
        "weibull_shape": shape,
        "gamma_factor": float(GAMMA_FACTOR),
        "by_brand": {},
    }

    brand_prob_by_code = normalize_probabilities(
        brand_prob_by_code,
        name="brand_prob_by_code",
    )

    fleet_probs: Dict[int, float] = {}

    for code, mtbf in targets_by_code_model_time.items():
        mtbf_float = _finite_float_or_none(mtbf)
        if mtbf_float is None or mtbf_float <= 0.0:
            continue

        scale = mtbf_float / GAMMA_FACTOR
        if scale <= 0.0:
            continue

        x = (censoring / scale) ** shape
        prob = -np.expm1(-x)
        prob_mean = float(np.mean(prob))

        result["by_brand"][BRAND_MAP.get(code, str(code))] = {
            "brand_code": int(code),
            "mtbf_major_engine_hours": float(mtbf_float),
            "weibull_scale": float(scale),
            "target_failure_probability": prob_mean,
        }

        fleet_probs[code] = prob_mean

    fleet_probability = _weighted_mean(fleet_probs, brand_prob_by_code)
    result["fleet_target_failure_probability"] = fleet_probability

    return result


# ═══════════════════════════════════════════════════════════════════════
# 20. MONTE CARLO ВАЛИДАЦИЯ
# ═══════════════════════════════════════════════════════════════════════

def validate_calibration(
    calibrated: Dict[str, Any],
    targets_by_code_model_time: Dict[int, float],
    brand_prob_by_code: Dict[int, float],
) -> Dict[str, Any]:
    """Независимая MC валидация по всем целевым брендам и fleet."""
    brand_prob_by_code = normalize_probabilities(
        brand_prob_by_code,
        name="brand_prob_by_code",
    )

    target_codes = list(targets_by_code_model_time.keys())

    mtbf_values: Dict[int, List[float]] = {code: [] for code in target_codes}
    prob_values: Dict[int, List[float]] = {code: [] for code in target_codes}

    fleet_mtbf_values: List[float] = []
    fleet_prob_values: List[float] = []

    print("\nMonte Carlo validation: all target brands")

    for rep in range(MC_REPS):
        rng = np.random.default_rng(SEED + 10_000 + rep)

        df = generate_covariates(MC_N, rng, brand_prob_by_code)
        peak_data = generate_peakload(df, rng, rho=DEFAULT_RHO)

        _, mtbf, failure_probability = simulate_metrics_from_peak(
            df=df,
            peak_data=peak_data,
            params=calibrated,
            target_codes=target_codes,
            rng=rng,
            censoring_time=None,
        )

        for code in target_codes:
            mtbf_value = _finite_float_or_none(mtbf.get(code, np.nan))
            prob_value = _finite_float_or_none(
                failure_probability.get(code, np.nan)
            )

            if mtbf_value is not None:
                mtbf_values[code].append(mtbf_value)

            if prob_value is not None:
                prob_values[code].append(prob_value)

        fleet_mtbf = _weighted_mean(mtbf, brand_prob_by_code)
        fleet_prob = _weighted_mean(failure_probability, brand_prob_by_code)

        if fleet_mtbf is not None and math.isfinite(fleet_mtbf):
            fleet_mtbf_values.append(float(fleet_mtbf))

        if fleet_prob is not None and math.isfinite(fleet_prob):
            fleet_prob_values.append(float(fleet_prob))

    by_brand: Dict[str, Any] = {}

    for code in target_codes:
        stats = _stats(mtbf_values[code])
        prob_stats = _stats(prob_values[code])

        target_float = float(targets_by_code_model_time[code])
        mc_mean = _finite_float_or_none(stats.get("mc_mean"))

        relative_bias: Optional[float] = None
        if mc_mean is not None and target_float > 0.0:
            relative_bias = (mc_mean - target_float) / target_float

        by_brand[BRAND_MAP.get(code, str(code))] = {
            "brand_code": int(code),
            "target_mtbf": target_float,
            "mc_mean": mc_mean,
            "mc_sd": _finite_float_or_none(stats.get("mc_sd")),
            "mc_q025": _finite_float_or_none(stats.get("mc_q025")),
            "mc_q975": _finite_float_or_none(stats.get("mc_q975")),
            "relative_bias": relative_bias,
            "n_replications": int(stats.get("n_replications", 0)),
            "failure_probability_mean": _finite_float_or_none(
                prob_stats.get("mc_mean")
            ),
            "failure_probability_sd": _finite_float_or_none(
                prob_stats.get("mc_sd")
            ),
        }

    fleet_stats = _stats(fleet_mtbf_values)
    fleet_prob_stats = _stats(fleet_prob_values)

    target_fleet_mtbf = _weighted_mean(
        targets_by_code_model_time,
        brand_prob_by_code,
    )

    fleet_relative_bias: Optional[float] = None
    fleet_mc_mean = _finite_float_or_none(fleet_stats.get("mc_mean"))

    if (
        fleet_mc_mean is not None
        and target_fleet_mtbf is not None
        and target_fleet_mtbf > 0.0
    ):
        fleet_relative_bias = (
            fleet_mc_mean - target_fleet_mtbf
        ) / target_fleet_mtbf

    return {
        "censoring_mode": CENSORING_MODE,
        "mc_reps": int(MC_REPS),
        "mc_n": int(MC_N),
        "by_brand": by_brand,
        "fleet": {
            "target_mtbf": _finite_float_or_none(target_fleet_mtbf),
            "mc_mean": fleet_mc_mean,
            "mc_sd": _finite_float_or_none(fleet_stats.get("mc_sd")),
            "mc_q025": _finite_float_or_none(fleet_stats.get("mc_q025")),
            "mc_q975": _finite_float_or_none(fleet_stats.get("mc_q975")),
            "relative_bias": fleet_relative_bias,
            "n_replications": int(fleet_stats.get("n_replications", 0)),
            "failure_probability_mean": _finite_float_or_none(
                fleet_prob_stats.get("mc_mean")
            ),
            "failure_probability_sd": _finite_float_or_none(
                fleet_prob_stats.get("mc_sd")
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# 21. ОТЧЁТ О КАЛИБРОВКЕ
# ═══════════════════════════════════════════════════════════════════════

def create_calibration_report(
    calibrated: Dict[str, Any],
    targets_by_code_model_time: Dict[int, float],
    brand_prob_by_code: Dict[int, float],
) -> pd.DataFrame:
    """Финальный отчёт на большой выборке."""
    rng = np.random.default_rng(SEED + 777)

    df = generate_covariates(N_REPORT, rng, brand_prob_by_code)
    peak_data = generate_peakload(df, rng, rho=DEFAULT_RHO)

    target_codes = list(targets_by_code_model_time.keys())

    _, mtbf, failure_probability = simulate_metrics_from_peak(
        df=df,
        peak_data=peak_data,
        params=calibrated,
        target_codes=target_codes,
        rng=rng,
        censoring_time=None,
    )

    target_event_rates = compute_target_event_rates_for_df(
        df,
        targets_by_code_model_time,
    )

    rows = []

    for code, target in targets_by_code_model_time.items():
        target_float = _finite_float_or_none(target)
        predicted_mtbf = _finite_float_or_none(mtbf.get(code, np.nan))
        predicted_prob = _finite_float_or_none(
            failure_probability.get(code, np.nan)
        )
        target_prob = _finite_float_or_none(
            target_event_rates.get(code, np.nan)
        )

        absolute_error: Optional[float] = None
        relative_error: Optional[float] = None

        if (
            predicted_mtbf is not None
            and target_float is not None
            and target_float > 0.0
        ):
            absolute_error = predicted_mtbf - target_float
            relative_error = absolute_error / target_float

        rows.append(
            {
                "brand_code": int(code),
                "brand": BRAND_MAP.get(code, str(code)),
                "target_MTBF_major": target_float,
                "calibrated_MTBF_major": predicted_mtbf,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "target_failure_probability": target_prob,
                "calibrated_failure_probability": predicted_prob,
            }
        )

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 22. ЭКСПОРТ calibrated_dgp.json
# ═══════════════════════════════════════════════════════════════════════

def export_model(
    calibrated: Dict[str, Any],
    validation: Dict[str, Any],
    report: pd.DataFrame,
    targets_by_code_model_time: Dict[int, float],
    targets_by_code_raw: Dict[int, float],
    brand_prob_by_code: Dict[int, float],
    consistency: Dict[str, Any],
    scenario_name: str,
    scenario_params: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Экспорт самодостаточного calibrated_dgp.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_hazard = float(calibrated["baseline_hazard"])
    shape = float(WEIBULL_SHAPE)
    baseline_scale = baseline_hazard ** (-1.0 / shape)

    brand_effects_by_name = _map_code_float_to_name(
        calibrated["brand_effects_by_code"]
    )

    brand_effects_by_code: Dict[int, float] = {}
    for code, effect in calibrated["brand_effects_by_code"].items():
        effect_float = _finite_float_or_none(effect)
        if effect_float is not None:
            brand_effects_by_code[int(code)] = effect_float

    targets_by_name_model_time = _map_code_float_to_name(
        targets_by_code_model_time
    )
    targets_by_name_raw = _map_code_float_to_name(targets_by_code_raw)
    brand_prob_by_name = _map_code_float_to_name(brand_prob_by_code)

    target_event_rates_by_name = _map_code_float_to_name(
        calibrated.get("target_event_rates", {})
    )

    failure_probability_by_brand: Dict[str, Optional[float]] = {}

    if "calibrated_failure_probability" in report.columns:
        for _, row in report.iterrows():
            brand = str(row.get("brand", ""))
            failure_probability_by_brand[brand] = _finite_float_or_none(
                row.get("calibrated_failure_probability")
            )

    scenario_mtbf_all = _finite_float_or_none(
        scenario_params.get("mtbf_all_engine_hours")
    )
    if scenario_mtbf_all is None:
        scenario_mtbf_all = 0.0

    scenario_major_share = _finite_float_or_none(
        scenario_params.get("major_failure_share")
    )
    if scenario_major_share is None:
        scenario_major_share = float(MAJOR_FAILURE_SHARE)

    model = {
        "model_type": "semi_synthetic_weibull_ph",
        "version": "3.1.1",
        "seed": SEED,

        # ─── Единицы времени ───────────────────────────────────────
        "time_unit": MODEL_TIME_UNIT,
        "mtbf_input_unit": MTBF_INPUT_UNIT,
        "mtbf_to_model_time_factor": float(MTBF_TO_MODEL_TIME_FACTOR),

        # ─── Цензурирование ────────────────────────────────────────
        "censoring": {
            "mode": CENSORING_MODE,
            "calibration_horizon_days": float(CALIBRATION_HORIZON_DAYS),
            "days_per_year": float(DAYS_PER_YEAR),
            "constant_engine_hours": float(CALIBRATION_HORIZON_ENGINE_HOURS),
            "hours_clip_max": float(HOURS_CLIP_MAX),
        },

        # ─── Сценарий ──────────────────────────────────────────────
        "scenario": {
            "name": scenario_name,
            "description": str(scenario_params.get("description", "")),
            "mtbf_all_engine_hours": float(scenario_mtbf_all),
            "major_failure_share": float(scenario_major_share),
        },

        # ─── Weibull ───────────────────────────────────────────────
        "weibull": {
            "shape": shape,
            "baseline_hazard": baseline_hazard,
            "baseline_scale": baseline_scale,
        },

        # ─── Структурные коэффициенты ──────────────────────────────
        "coefficients": {
            "beta_age": float(calibrated["beta_age"]),
            "beta_hours": float(calibrated["beta_hours"]),
            "beta_climate": float(calibrated["beta_climate"]),
            "beta_soil": float(calibrated["beta_soil"]),
            "beta_power": float(calibrated["beta_power"]),
            "gamma": float(calibrated["gamma"]),
            "baseline_hazard": baseline_hazard,
            "baseline_scale": baseline_scale,
        },

        # ─── First stage ───────────────────────────────────────────
        "first_stage": {
            "intercept": float(FIRST_STAGE_INTERCEPT),
            "z_coef": float(FIRST_STAGE_Z_COEF),
            "age_coef": float(FIRST_STAGE_AGE_COEF),
            "hours_coef": float(FIRST_STAGE_HOURS_COEF),
            "climate_coef": float(FIRST_STAGE_CLIMATE_COEF),
            "soil_coef": float(FIRST_STAGE_SOIL_COEF),
            "brand_coef": float(FIRST_STAGE_BRAND_COEF),
            "power_coef": float(FIRST_STAGE_POWER_COEF),
        },

        # ─── Structural / clipping ─────────────────────────────────
        "structural": {
            "structural_intercept": float(STRUCTURAL_INTERCEPT),
            "clip_min": float(CLIP_MIN),
            "clip_max": float(CLIP_MAX),
            "u_policy": "set_u_zero_for_individual_prediction",
            "note": (
                "For population-level metrics, integrate over U by Monte "
                "Carlo. For individual prediction, U is usually set to 0, "
                "which yields conditional, not marginal, expectations."
            ),
        },

        # ─── Brand effects ─────────────────────────────────────────
        "brand_effects": brand_effects_by_name,
        "brand_effects_by_code": brand_effects_by_code,
        "brand_mapping": BRAND_MAP,
        "brand_to_code": BRAND_TO_CODE,

        # ─── Цели ──────────────────────────────────────────────────
        "targets": targets_by_name_model_time,
        "targets_raw": targets_by_name_raw,
        "target_event_rates": target_event_rates_by_name,
        "target_fleet_mtbf": _finite_float_or_none(
            calibrated.get("target_fleet_mtbf")
        ),
        "target_fleet_failure_probability": _finite_float_or_none(
            calibrated.get("target_fleet_probability")
        ),
        "brand_probabilities": brand_prob_by_name,

        # ─── Распределения ковариат ────────────────────────────────
        "distributions": {
            "Age": {
                "distribution": "lognormal",
                "mu": 2.30,
                "sigma": 0.60,
                "clip_min": 0.0,
                "clip_max": 30.0,
                "note": "Медиана≈10 лет; P(Age>10)≈0.50",
            },
            "Hours": {
                "distribution": "exponential",
                "scale": 1350.0,
                "clip_min": 0.0,
                "clip_max": float(HOURS_CLIP_MAX),
                "note": (
                    "Годовая наработка мч/год. Используется для "
                    "индивидуального цензурирования, если "
                    "CENSORING_MODE=individual_calendar."
                ),
            },
            "Climate": {
                "distribution": "beta",
                "alpha": 2.5,
                "beta": 1.5,
                "note": "ГТК Селянинова, нормировано на [0, 1]",
            },
            "Soil": {
                "distribution": "beta",
                "alpha": 2.0,
                "beta": 2.5,
                "note": "Почвенные карты, нормировано на [0, 1]",
            },
            "Power": {
                "distribution": "normal",
                "mu": 180.0,
                "sigma": 80.0,
                "clip_min": 50.0,
                "clip_max": 350.0,
            },
            "Brand": {
                "distribution": "categorical",
                "probabilities": brand_prob_by_name,
            },
        },

        # ─── X standardization ─────────────────────────────────────
        "x_standardization": X_STANDARDIZATION,

        # ─── Эндогенность ──────────────────────────────────────────
        "endogeneity": {
            "rho": float(DEFAULT_RHO),
            "delta": float(DEFAULT_DELTA),
        },

        # ─── Калибровка ────────────────────────────────────────────
        "calibration": {
            "optimizer_success": bool(calibrated.get("optimizer_success", False)),
            "optimizer_message": str(calibrated.get("optimizer_message", "")),
            "loss": _finite_float_or_none(calibrated.get("loss")),
            "n_starts": int(calibrated.get("n_starts", 0)),
            "clipped_fraction": _finite_float_or_none(
                calibrated.get("clipped_fraction")
            ),
            "free_parameters": calibrated.get("free_parameters", []),
            "fixed_parameters": {
                "beta_age": float(FIXED_BETA_AGE),
                "beta_hours": float(FIXED_BETA_HOURS),
                "beta_climate": float(FIXED_BETA_CLIMATE),
                "beta_soil": float(FIXED_BETA_SOIL),
                "beta_power": float(FIXED_BETA_POWER),
                "gamma": float(FIXED_GAMMA),
                "rho": float(DEFAULT_RHO),
                "delta": float(DEFAULT_DELTA),
            },
            "bounds": BOUNDS,
        },

        # ─── Валидация и отчёты ────────────────────────────────────
        "validation": validation,
        "failure_probability_by_brand": failure_probability_by_brand,
        "mtbf_probability_consistency": consistency,

        # ─── Примечания ────────────────────────────────────────────
        "notes": [
            "v3.1.1: beta_* and gamma are fixed expert assumptions to avoid underidentification.",
            "Calibrated parameters: baseline_hazard and brand effects only.",
            "MTBF_major = MTBF_all / major_failure_share.",
            f"major_failure_share used = {scenario_major_share}.",
            "Time unit: engine_hours.",
            "Weibull parameterization: H(t) = baseline_hazard * exp(lp) * t^shape.",
            "Brand effects relative to MTZ82 code 0.",
            "Failure probability is cumulative incidence at censoring horizon.",
            f"Censoring mode: {CENSORING_MODE}.",
            "Semi-synthetic calibration, not individual-level estimation.",
        ],
    }

    json_path = output_dir / "calibrated_dgp.json"
    csv_path = output_dir / "calibration_results.csv"

    model_native = _to_native(model)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_native, f, ensure_ascii=False, indent=2, allow_nan=False)

    report.to_csv(csv_path, index=False)

    print("\nExported:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
# 23. MAIN
# ═══════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate semi-synthetic Weibull PH DGP (v3.1.1)",
    )

    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS.keys()),
        default=DEFAULT_SCENARIO,
        help="MTBF scenario name",
    )

    parser.add_argument(
        "--xlsx",
        type=Path,
        default=None,
        help="Optional xlsx file with brand MTBF targets and probabilities",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("calibration_output"),
        help="Output directory for JSON/CSV artifacts",
    )

    parser.add_argument(
        "--censoring-mode",
        choices=["individual_calendar", "constant_engine_hours"],
        default=CENSORING_MODE,
        help=(
            "individual_calendar: C_i = Hours_i * horizon_days / 365; "
            "constant_engine_hours: constant C_i = 1712 hours"
        ),
    )

    parser.add_argument(
        "--allow-failed",
        action="store_true",
        help="Do not exit with error if optimizer fails",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    global CENSORING_MODE
    CENSORING_MODE = args.censoring_mode

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    print("=" * 70)
    print("SEMI-SYNTHETIC DGP CALIBRATION (v3.1.1, engine_hours)")
    print("=" * 70)

    # ─── Сценарий ──────────────────────────────────────────────────
    scenario = get_scenario(args.scenario)

    mtbf_all = float(scenario["mtbf_all_engine_hours"])
    major_share = float(
        scenario.get("major_failure_share", MAJOR_FAILURE_SHARE)
    )
    mtbf_major_ref = compute_mtbf_major(mtbf_all, major_share)

    print(f"\nScenario: {args.scenario}")
    print(f"  Description: {scenario['description']}")
    print(f"  MTBF_all:    {mtbf_all:.0f} мч")
    print(f"  MAJOR_SHARE: {major_share:.2f}")
    print(f"  MTBF_major:  {mtbf_major_ref:.0f} мч (reference)")
    print(f"  Censoring:   {CENSORING_MODE}")

    # ─── Целевые MTBF ──────────────────────────────────────────────
    # v3.1: Other включён по умолчанию.
    targets_by_code_raw: Dict[int, float] = {
        0: mtbf_major_ref,  # MTZ82
        1: mtbf_major_ref,  # Versatile280
        2: mtbf_major_ref,  # NewHollandT9
        3: mtbf_major_ref,  # DT75
        4: mtbf_major_ref,  # Other
    }

    brand_prob_by_code = dict(DEFAULT_BRAND_PROB)
    source = f"scenario:{args.scenario}"

    # ─── Опциональный xlsx override ────────────────────────────────
    xlsx_data = load_targets_from_xlsx(args.xlsx)

    if xlsx_data is not None:
        loaded_targets = xlsx_data.get("targets_by_code_raw", {})
        loaded_probs = xlsx_data.get("brand_prob_by_code", {})

        if loaded_targets:
            targets_by_code_raw.update(loaded_targets)
            source = str(xlsx_data.get("source", "xlsx"))

        if loaded_probs:
            merged = dict(DEFAULT_BRAND_PROB)
            merged.update(loaded_probs)

            try:
                brand_prob_by_code = normalize_probabilities(
                    merged,
                    name="merged brand probabilities",
                )
                if source.startswith("scenario:"):
                    source += " + xlsx probabilities"
            except ValueError as exc:
                logger.warning(
                    "Invalid merged brand probabilities: %s; using defaults",
                    exc,
                )

    print(f"\nMTBF source: {source}")
    print(f"MTBF_TO_MODEL_TIME_FACTOR: {MTBF_TO_MODEL_TIME_FACTOR}")

    # ─── Конвертация в model time ─────────────────────────────────
    targets_by_code_model_time: Dict[int, float] = {}

    for code, mtbf_raw in targets_by_code_raw.items():
        mtbf_model = _finite_float_or_none(mtbf_raw)
        if mtbf_model is None:
            print(f"WARNING: invalid MTBF for code {code}")
            continue

        try:
            code_int = int(code)
        except (TypeError, ValueError):
            print(f"WARNING: invalid brand code {code}")
            continue

        targets_by_code_model_time[code_int] = (
            mtbf_model * MTBF_TO_MODEL_TIME_FACTOR
        )

    if not targets_by_code_model_time:
        print("ERROR: no valid MTBF targets.")
        return 1

    try:
        brand_prob_by_code = normalize_probabilities(
            brand_prob_by_code,
            name="brand_prob_by_code",
        )
    except (TypeError, ValueError) as exc:
        print(f"ERROR: invalid brand probabilities: {exc}")
        return 1

    # Проверка, что все целевые бренды присутствуют в парке
    for code in targets_by_code_model_time.keys():
        if brand_prob_by_code.get(code, 0.0) <= 0.0:
            print(
                f"ERROR: target brand code {code} has zero probability "
                "in brand_prob_by_code"
            )
            return 1

    print("\nTargets (MTBF_major, engine hours):")
    for code, mtbf_model in targets_by_code_model_time.items():
        print(
            f"  {BRAND_MAP.get(code, str(code)):20s}: "
            f"{mtbf_model:.1f} мч"
        )

    # ─── Consistency check ─────────────────────────────────────────
    consistency = build_target_consistency(
        targets_by_code_model_time=targets_by_code_model_time,
        brand_prob_by_code=brand_prob_by_code,
    )

    print("\n" + "=" * 70)
    print("MTBF vs P(T<=C) CONSISTENCY CHECK")
    print("=" * 70)

    print(f"\nCensoring mode: {CENSORING_MODE}")
    print(f"Horizon days:   {CALIBRATION_HORIZON_DAYS:.0f}")
    print(f"Weibull shape:  {WEIBULL_SHAPE}")
    print(f"Γ(1 + 1/b):     {consistency['gamma_factor']:.6f}")

    by_brand = consistency.get("by_brand", {})

    for brand_name, item in by_brand.items():
        if not isinstance(item, dict):
            continue

        mtbf_val = _finite_float_or_none(
            item.get("mtbf_major_engine_hours")
        )
        scale_val = _finite_float_or_none(item.get("weibull_scale"))
        prob_val = _finite_float_or_none(
            item.get("target_failure_probability")
        )

        if mtbf_val is None or scale_val is None or prob_val is None:
            continue

        print(
            f"  {str(brand_name):20s}: "
            f"MTBF_major={mtbf_val:.0f}, "
            f"scale={scale_val:.0f}, "
            f"P(failure)={prob_val:.4f}"
        )

    fleet_prob = _finite_float_or_none(
        consistency.get("fleet_target_failure_probability")
    )
    if fleet_prob is not None:
        print(f"  {'Fleet':20s}: P(failure)={fleet_prob:.4f}")

    # ─── Калибровка ────────────────────────────────────────────────
    try:
        _, calibrated, _ = calibrate(
            targets_by_code_model_time=targets_by_code_model_time,
            brand_prob_by_code=brand_prob_by_code,
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        logger.error("Calibration failed with exception: %s", exc)
        return 1

    print("\nOptimization result:")
    print(f"  success: {calibrated.get('optimizer_success')}")
    print(f"  message: {calibrated.get('optimizer_message')}")
    print(f"  loss:    {_fmt(calibrated.get('loss'), 6)}")
    print(f"  clipped: {_fmt(calibrated.get('clipped_fraction'), 6)}")

    if not calibrated.get("optimizer_success", False) and not args.allow_failed:
        logger.error(
            "Optimization did not succeed. Use --allow-failed to export anyway."
        )
        return 1

    loss_value = _finite_float_or_none(calibrated.get("loss"))
    if loss_value is None or loss_value > MAX_ACCEPTABLE_LOSS:
        logger.error(
            "Calibration loss is too large: %s. "
            "Use --allow-failed to export anyway.",
            _fmt(loss_value, 6),
        )
        if not args.allow_failed:
            return 1

    print("\nCalibrated coefficients:")
    print(f"  beta_age        = {calibrated['beta_age']:.8f} (fixed)")
    print(f"  beta_hours      = {calibrated['beta_hours']:.8f} (fixed)")
    print(f"  beta_climate    = {calibrated['beta_climate']:.8f} (fixed)")
    print(f"  beta_soil       = {calibrated['beta_soil']:.8f} (fixed)")
    print(f"  beta_power      = {calibrated['beta_power']:.8f} (fixed)")
    print(f"  gamma           = {calibrated['gamma']:.8f} (fixed)")
    print(f"  baseline_hazard = {calibrated['baseline_hazard']:.8e}")

    print("\nBrand effects (reference = MTZ82 code 0):")
    for code, effect in calibrated["brand_effects_by_code"].items():
        effect_float = _finite_float_or_none(effect)
        if effect_float is None:
            continue

        print(
            f"  {BRAND_MAP.get(code, str(code)):20s} "
            f"(code {code}) = {effect_float:.8f}"
        )

    # ─── Отчёт ─────────────────────────────────────────────────────
    report = create_calibration_report(
        calibrated=calibrated,
        targets_by_code_model_time=targets_by_code_model_time,
        brand_prob_by_code=brand_prob_by_code,
    )

    print("\nCalibration report:")
    print(report.to_string(index=False))

    # ─── Валидация ─────────────────────────────────────────────────
    validation = validate_calibration(
        calibrated=calibrated,
        targets_by_code_model_time=targets_by_code_model_time,
        brand_prob_by_code=brand_prob_by_code,
    )

    # ─── Экспорт ───────────────────────────────────────────────────
    export_model(
        calibrated=calibrated,
        validation=validation,
        report=report,
        targets_by_code_model_time=targets_by_code_model_time,
        targets_by_code_raw=targets_by_code_raw,
        brand_prob_by_code=brand_prob_by_code,
        consistency=consistency,
        scenario_name=args.scenario,
        scenario_params=scenario,
        output_dir=args.output_dir,
    )

    print("\nCalibration finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())