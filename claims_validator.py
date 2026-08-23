#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
claims_validator.py
Фаза 6.4: очистка и валидация claims-данных.

Загружает claims_pilot_v1.csv, валидирует против docs/data_contract.md,
используя константы из constants.py, и сохраняет очищенные данные.

Использование:
    python claims_validator.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from constants import (
    BRAND_MAP,
    FREQ_SHARES,
    VALID_EVENT_DEFINITIONS,
    MODEL_TIME_UNIT,
    CALIBRATION_HORIZON_ENGINE_HOURS,
)

# BRAND_ALIASES опционален
try:
    from constants import BRAND_ALIASES
except ImportError:
    BRAND_ALIASES = {}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
CLAIMS_PATH = Path("data/raw/claims/claims_pilot_v3.csv")
OUTPUT_DIR = Path("data/processed/claims")
OUTPUT_PATH = OUTPUT_DIR / "claims_clean.csv"
REPORT_PATH = OUTPUT_DIR / "validation_report.json"

# ---------------------------------------------------------------------------
# Обязательные поля согласно data_contract.md
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {
    "machine_id",
    "brand",
    "power_hp",
    "production_year",
    "age_at_event",
    "hours_at_event",
    "failure_time",
    "event_flag",
    "failure_system",
    "major_failure_flag",
}

# Опциональные колонки (v2)
OPTIONAL_COLUMNS = {
    "peak_load_proxy",
    "weather_instrument",
    "climate",
    "soil",
    "region",
    "campaign",
    "repair_cost",
    "downtime_hours",
}

# Допустимые системы отказов (из FREQ_SHARES)
VALID_FAILURE_SYSTEMS = set(FREQ_SHARES.keys())

# Допустимые имена брендов (из BRAND_MAP)
VALID_BRAND_NAMES = set(BRAND_MAP.values())

# ---------------------------------------------------------------------------
# Пороги покрытия
# ---------------------------------------------------------------------------
MIN_MACHINES = 100
MIN_EVENTS = 200
MIN_EVENTS_PER_BRAND = 30
MIN_HORIZON = CALIBRATION_HORIZON_ENGINE_HOURS

# ---------------------------------------------------------------------------
# Диапазоны
# ---------------------------------------------------------------------------
POWER_RANGE = (50.0, 500.0)
PRODUCTION_YEAR_RANGE = (1990, 2025)
AGE_RANGE = (0.0, 30.0)
HOURS_RANGE = (0.0, 50000.0)


def load_claims(path: Path) -> pd.DataFrame:
    """Загрузить claims CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Файл claims не найден: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    logger.info("Загружено строк: %d", len(df))
    return df


def validate_schema(df: pd.DataFrame) -> List[str]:
    """Проверить наличие обязательных колонок."""
    errors: List[str] = []
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Отсутствуют обязательные колонки: {sorted(missing)}")

    # Проверка опциональных колонок (v2)
    optional_present = OPTIONAL_COLUMNS & set(df.columns)
    optional_missing = OPTIONAL_COLUMNS - set(df.columns)
    if optional_missing:
        logger.info(
            "Опциональные колонки отсутствуют (будут использованы fallback): %s",
            sorted(optional_missing),
        )
    if optional_present:
        logger.info(
            "Обнаружены опциональные колонки v2: %s",
            sorted(optional_present),
        )

    return errors


def normalize_brand(brand_raw: Any) -> str:
    """Нормализовать имя бренда к каноническому виду из BRAND_MAP."""
    if brand_raw is None or (
        isinstance(brand_raw, float) and np.isnan(brand_raw)
    ):
        return "Other"
    s = str(brand_raw).strip().lower()
    if not s:
        return "Other"
    # Прямое совпадение с каноническими именами
    for canonical in VALID_BRAND_NAMES:
        if s == canonical.lower():
            return canonical
    # Поиск по алиасам
    s_compact = s.replace(" ", "").replace("-", "")
    for alias, code in BRAND_ALIASES.items():
        alias_compact = alias.replace(" ", "").replace("-", "")
        if s_compact == alias_compact:
            return BRAND_MAP.get(code, "Other")
    return "Other"


def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Валидация типов, диапазонов и очистка."""
    errors: List[str] = []
    df = df.copy()

    # Числовые поля с диапазонами
    numeric_fields: Dict[str, Tuple[Any, Any]] = {
        "power_hp": POWER_RANGE,
        "production_year": PRODUCTION_YEAR_RANGE,
        "age_at_event": AGE_RANGE,
        "hours_at_event": HOURS_RANGE,
        "failure_time": (0.0, None),
        "repair_cost": (0.0, None),
        "downtime_hours": (0.0, None),
        "claim_amount": (0.0, None),
    }
    for field_name, (low, high) in numeric_fields.items():
        if field_name not in df.columns:
            continue
        df[field_name] = pd.to_numeric(df[field_name], errors="coerce")
        n_bad = int(df[field_name].isna().sum())
        if n_bad > 0:
            errors.append(f"{field_name}: {n_bad} нечисловых значений")
        if low is not None:
            neg = df[field_name] < low
            if neg.any():
                errors.append(
                    f"{field_name}: {int(neg.sum())} значений < {low}"
                )
        if high is not None:
            over = df[field_name] > high
            if over.any():
                errors.append(
                    f"{field_name}: {int(over.sum())} значений > {high}"
                )

    # event_flag и major_failure_flag бинарные
    for field_name in ("event_flag", "major_failure_flag"):
        if field_name not in df.columns:
            continue
        df[field_name] = pd.to_numeric(df[field_name], errors="coerce")
        invalid = ~df[field_name].isin([0, 1, 0.0, 1.0, np.nan])
        if invalid.any():
            errors.append(
                f"{field_name}: {int(invalid.sum())} значений вне {{0,1}}"
            )

    # Нормализация брендов
    if "brand" in df.columns:
        df["brand"] = df["brand"].map(normalize_brand)
        unknown = ~df["brand"].isin(VALID_BRAND_NAMES)
        if unknown.any():
            errors.append(
                f"brand: {int(unknown.sum())} неизвестных брендов"
            )
            df.loc[unknown, "brand"] = "Other"

    # Нормализация failure_system
    if "failure_system" in df.columns:
        df["failure_system"] = (
            df["failure_system"].astype(str).str.strip().str.lower()
        )
        invalid = ~df["failure_system"].isin(VALID_FAILURE_SYSTEMS)
        if invalid.any():
            errors.append(
                f"failure_system: {int(invalid.sum())} неизвестных систем"
            )
            df.loc[invalid, "failure_system"] = "прочее"
    # Валидация опциональных колонок v2
    optional_numeric_fields = {
        "peak_load_proxy": (0.0, 2.0),
        "weather_instrument": (0.0, 100.0),
        "climate": (0.0, 1.0),
        "soil": (0.0, 1.0),
    }
    for field_name, (low, high) in optional_numeric_fields.items():
        if field_name in df.columns:
            df[field_name] = pd.to_numeric(df[field_name], errors="coerce")
            n_bad = int(df[field_name].isna().sum())
            if n_bad > 0:
                logger.warning(
                    "%s: %d нечисловых значений (заменены медианой)",
                    field_name,
                    n_bad,
                )
                median_val = df[field_name].median()
                if pd.isna(median_val):
                    median_val = 0.5
                df[field_name] = df[field_name].fillna(median_val)

            if low is not None:
                neg = df[field_name] < low
                if neg.any():
                    df.loc[neg, field_name] = low
            if high is not None:
                over = df[field_name] > high
                if over.any():
                    df.loc[over, field_name] = high

    # event_definition (опционально)
    if "event_definition" in df.columns:
        df["event_definition"] = (
            df["event_definition"].astype(str).str.strip().str.lower()
        )
        invalid = ~df["event_definition"].isin(VALID_EVENT_DEFINITIONS)
        if invalid.any():
            errors.append(
                f"event_definition: {int(invalid.sum())} неизвестных значений"
            )

    # Согласованность event_flag и failure_time
    if "event_flag" in df.columns and "failure_time" in df.columns:
        bad = (df["event_flag"] == 1) & (df["failure_time"] <= 0)
        if bad.any():
            errors.append(
                f"failure_time <= 0 при event_flag=1: {int(bad.sum())}"
            )

    return df, errors


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Удалить дубликаты по machine_id + failure_time + event_flag."""
    n_before = len(df)
    df = df.drop_duplicates(
        subset=["machine_id", "failure_time", "event_flag"]
    )
    n_removed = n_before - len(df)
    return df, n_removed


def check_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """Проверить покрытие данных."""
    events = df[df["event_flag"] == 1]
    n_machines = int(df["machine_id"].nunique())
    n_events = int(len(events))
    brand_coverage = events["brand"].value_counts().to_dict()
    max_obs_time = (
        float(df["failure_time"].max()) if len(df) > 0 else 0.0
    )

    return {
        "n_machines": n_machines,
        "n_events": n_events,
        "max_observation_time": max_obs_time,
        "brand_coverage": brand_coverage,
        "machines_ok": n_machines >= MIN_MACHINES,
        "events_ok": n_events >= MIN_EVENTS,
        "horizon_ok": max_obs_time >= MIN_HORIZON,
        "all_ok": (
            n_machines >= MIN_MACHINES
            and n_events >= MIN_EVENTS
            and max_obs_time >= MIN_HORIZON
        ),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(message)s"
    )

    print("=" * 70)
    print("Фаза 6.4: Валидация claims-данных")
    print("=" * 70)

    try:
        df = load_claims(CLAIMS_PATH)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        print(
            "Создайте файл claims_pilot_v1.csv "
            "по схеме docs/data_contract.md"
        )
        return 1

    all_errors: List[str] = []

    # Схема
    schema_errors = validate_schema(df)
    all_errors.extend(schema_errors)
    if schema_errors:
        for e in schema_errors:
            logger.error(e)
        return 1

    # Валидация и очистка
    df, clean_errors = validate_and_clean(df)
    all_errors.extend(clean_errors)

    # Дубликаты
    df, n_duplicates = remove_duplicates(df)
    if n_duplicates > 0:
        logger.info("Удалено дубликатов: %d", n_duplicates)

    # Удаление строк с NaN в обязательных полях
    df = df.dropna(subset=list(REQUIRED_COLUMNS))

    # Покрытие
    coverage = check_coverage(df)

    # Вывод
    print()
    print("-" * 70)
    print("Результаты валидации:")
    print(
        f"  Машин:          {coverage['n_machines']} "
        f"(мин. {MIN_MACHINES})"
    )
    print(
        f"  Событий:        {coverage['n_events']} "
        f"(мин. {MIN_EVENTS})"
    )
    print(
        f"  Макс. время:    {coverage['max_observation_time']:.0f} мч "
        f"(мин. {MIN_HORIZON:.0f})"
    )
    print(f"  Покрытие брендов: {coverage['brand_coverage']}")
    print()

    if all_errors:
        print(f"⚠️  Обнаружено ошибок: {len(all_errors)}")
        for e in all_errors[:20]:
            print(f"  - {e}")

    if coverage["all_ok"]:
        print("✅ Данные достаточны для пилотного переобучения.")
    else:
        print("❌ Данные НЕДОСТАТОЧНЫ. Соберите больше claims.")

    # Сохранение
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Очищенные данные: {OUTPUT_PATH}")

    report = {
        "errors": all_errors,
        "coverage": coverage,
        "n_duplicates_removed": n_duplicates,
        "n_rows_final": len(df),
        "time_unit": MODEL_TIME_UNIT,
        "calibration_horizon_engine_hours": (
            CALIBRATION_HORIZON_ENGINE_HOURS
        ),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {REPORT_PATH}")

    return 0 if coverage["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())