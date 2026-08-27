# -*- coding: utf-8 -*-
"""
constants.py
Единый реестр констант проекта.
Фаза 8.3 / поперечный рефакторинг.

Здесь собраны доменные константы, которые ранее дублировались в
prediction_engine.py, Итог.py, train_model.py и Real_calculator.py.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
# Time conventions (Конвенции времени)
# ---------------------------------------------------------------------------
MODEL_TIME_UNIT: str = "engine_hours"
DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY: float = 8.0
CALIBRATION_HORIZON_DAYS: float = 214.0
CALIBRATION_HORIZON_ENGINE_HOURS: float = (
    CALIBRATION_HORIZON_DAYS * DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY
)
ENGINE_CONVENTION: str = "cf_cox_v3"

# ---------------------------------------------------------------------------
# FIX 1: PeakLoad normalization (DGP ↔ TUM CAN bus scale)
# ---------------------------------------------------------------------------
# PeakLoad теперь генерируется в [0, 1] диапазоне, соответствующем
# TUM CAN bus данным (доли от максимальной мощности двигателя).
PEAKLOAD_NORMALIZED_MEAN: float = 0.55
PEAKLOAD_NORMALIZED_STD: float = 0.15
# ---------------------------------------------------------------------------
# Event definitions & segments (Определения событий и сегменты)
# ---------------------------------------------------------------------------
VALID_EVENT_DEFINITIONS = frozenset({"total_loss", "major_claim", "any_failure"})
SEGMENTS = ("light", "heavy")

# ---------------------------------------------------------------------------
# Expert assumptions (Экспертные допущения и priors)
# ---------------------------------------------------------------------------
MAJOR_FAILURE_SHARE_PRIOR_MEAN = 0.30
MAJOR_FAILURE_SHARE_PRIOR_EFFECTIVE_N = 30.0

# Backward-compatible alias: существующий код, читающий
# MAJOR_FAILURE_SHARE, продолжит работать.
MAJOR_FAILURE_SHARE = MAJOR_FAILURE_SHARE_PRIOR_MEAN
DEFAULT_WEIBULL_SHAPE: float = 1.88
MTBF_BASELINE_HOURS: float = 1500.0
DEFAULT_GOMPERTZ_RATE: float = 0.01

# ---------------------------------------------------------------------------
# P-09: Downtime by MTTR (Простой по среднему времени восстановления)
# ---------------------------------------------------------------------------
DEFAULT_MTTR_HOURS: float = 8.0
DEFAULT_DOWNTIME_PER_MTTR_FACTOR: float = 1.0
MTTR_HOURS: Dict[str, float] = {
    "minor": 8.0,
    "major": 48.0,
}

# ---------------------------------------------------------------------------
# Brand canonical coding (Канонические коды брендов)
# ---------------------------------------------------------------------------
BRAND_MAP: Dict[int, str] = {
    0: "MTZ82",
    1: "Versatile280",
    2: "NewHollandT9",
    3: "DT75",
    4: "Other",
}
BRAND_TO_CODE: Dict[str, int] = {v: k for k, v in BRAND_MAP.items()}

BRAND_ALIASES: Dict[str, int] = {
    # MTZ82
    "mtz82": 0, "mtz-82": 0, "мтз-82": 0, "мтз 82": 0, "мтз82": 0,
    # Versatile 280
    "versatile280": 1, "versatile 280": 1, "versatile-280": 1, "версатиль 280": 1,
    # New Holland T9
    "newhollandt9": 2, "new holland t9": 2, "newholland t9": 2,
    "new holland t9.505": 2, "new holland t9.615": 2,
    "newhollandt9.505": 2, "newhollandt9.615": 2, "нью холланд t9": 2,
    # DT75
    "dt75": 3, "dt-75": 3, "дт-75": 3, "дт75": 3,
    # Other
    "other": 4, "other_brand": 4, "прочее": 4, "другой": 4, "unknown": 4,
}

DEFAULT_BRAND_PROB_BY_CODE: Dict[int, float] = {
    0: 0.35,  # MTZ82
    1: 0.08,  # Versatile280
    2: 0.07,  # NewHollandT9
    3: 0.10,  # DT75
    4: 0.40,  # Other
}

# ---------------------------------------------------------------------------
# P-05: Failure frequency, severity, and criticality shares
# (Частота, тяжесть и критичность отказов по системам)
# ---------------------------------------------------------------------------
FREQ_SHARES: Dict[str, float] = {
    "гидравлика": 0.30,
    "электроника": 0.30,
    "двигатель": 0.12,
    "трансмиссия": 0.20,
    "прочее": 0.08,
}

SEVERITY_WEIGHTS: Dict[str, float] = {
    "двигатель": 0.43,
    "трансмиссия": 0.28,
    "гидравлика": 0.17,
    "электроника": 0.12,
    "прочее": 0.00,
}

CRITICALITY_WEIGHTS: Dict[str, float] = {
    "гидравлика": 1.00,
    "электроника": 1.20,
    "двигатель": 1.50,
    "трансмиссия": 1.30,
    "прочее": 0.80,
}

# ---------------------------------------------------------------------------
# P-07: RF heavy fleet catalog (Каталог мощного парка РФ)
# ---------------------------------------------------------------------------
RF_HEAVY_BRAND_CATALOG: Dict[str, Dict[str, Any]] = {
    "Кировец К-744Р": {"power_hp": 340.0, "share": 0.425, "years": "2006-2008"},
    "John Deere 8430": {"power_hp": 295.0, "share": 0.248, "years": "2007"},
    "New Holland T8/T9": {"power_hp": 303.0, "share": 0.124, "years": "2006-2008"},
    "Bühler Versatile 2425": {"power_hp": 425.0, "share": 0.106, "years": "2007"},
    "АТМ-5280": {"power_hp": 280.0, "share": 0.053, "years": "2005-2009"},
    "Fendt 930 Vario": {"power_hp": 300.0, "share": 0.044, "years": "2005"},
}

# ---------------------------------------------------------------------------
# P-10: Power segments (Сегменты мощности)
# ---------------------------------------------------------------------------
POWER_SEGMENT_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "light": (0.0, 200.0),
    "medium": (200.0, 320.0),
    "heavy": (320.0, 1_000_000.0),
}

# ---------------------------------------------------------------------------
# Climate / Soil reference indices (Референсные индексы климата и почв)
# ---------------------------------------------------------------------------
CLIMATE_INDEX_REFERENCE: Dict[str, float] = {
    "минимальная": 0.0, "минимальный": 0.0,
    "умеренный": 0.25, "умеренная": 0.25,
    "холодный": 0.60, "холодная": 0.60,
    "тропический": 0.85, "тропическая": 0.85,
    "максимальная": 1.0, "максимальный": 1.0,
}

SOIL_INDEX_REFERENCE: Dict[str, float] = {
    "минимальная": 0.0, "минимальный": 0.0,
    "глинистая": 0.35,
    "суглинистая": 0.50,
    "песчаная": 1.00,
    "максимальная": 1.00, "максимальный": 1.00,
}

# ---------------------------------------------------------------------------
# IV / partial-out conventions (Конвенции IV и partial-out)
# ---------------------------------------------------------------------------
# Конвенция для partial_out_all_betas:
# "exclude_instrument" — убираем только экзогенные X, Z остаётся в остатке
# "include_instrument" — убираем всю объяснённую часть, включая Z
PL_HAT_EXOG_CONVENTION: str = "exclude_instrument"

# ---------------------------------------------------------------------------
# FIX 1 (PATCH 1.1): PeakLoad scale bridge — ОТКАТ от масштабирования
# ---------------------------------------------------------------------------
# DGP обновлён: intercept=0.5, peakload_target_mean=0.55, диапазон [0, 1].
# TUM CAN bus также выдаёт peak_load_mean в долях [0, 1].
# Шкалы совпадают — масштабирование НЕ требуется.
#
# СТАРЫЙ DGP (удалён): intercept=10.0, диапазон ~[6, 14]
# НОВЫЙ DGP (текущий): intercept=0.5, диапазон [0, 1]
#
# PEAKLOAD_TUM_TO_DGP_SCALE оставлен для обратной совместимости,
# но должен устанавливаться в 1.0 при использовании нового DGP.
PEAKLOAD_TUM_TO_DGP_SCALE: float = 1.0  # PATCH 1.1: откат до 1.0 (шкалы совпадают)

# ---------------------------------------------------------------------------
# Severity model: heavy-tailed distribution parameters
# ---------------------------------------------------------------------------
# По умолчанию используется Lognormal с тяжёлым хвостом.
# mu и sigma — параметры логарифма стоимости ремонта.
SEVERITY_LOGNORMAL_MU: float = 11.5   # ~100 000 руб.
SEVERITY_LOGNORMAL_SIGMA: float = 0.6  # дисперсия
SEVERITY_PARETO_K: float = 2.5         # форма хвоста (k > 2 = конечная дисперсия)
SEVERITY_PARETO_LAMBDA: float = 200_000.0  # масштаб