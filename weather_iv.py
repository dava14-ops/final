#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weather_iv.py
Фаза 5.2: инфраструктура погодного инструмента working_days_window.

Задачи:
1. Загрузка таблицы weather_windows.csv.
2. Валидация покрытия (>= 5 регионов, >= 3 лет).
3. Стандартизация working_days_window для использования как Z.
4. Генерация синтетических данных для Фазы 5.3 (симуляция).

Реальные данные появятся в Фазе 6. До тех пор скрипт работает
с синтетическими данными для проверки концепции.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("weather_iv")

# ---------------------------------------------------------------------------
# Пути и константы
# ---------------------------------------------------------------------------
WEATHER_DATA_DIR = Path("data/processed/weather")
WEATHER_DATA_PATH = WEATHER_DATA_DIR / "weather_windows.csv"

# Минимальные требования к покрытию (критерий Фазы 5.2)
MIN_REGIONS = 5
MIN_YEARS = 3

# Допустимые кампании
VALID_CAMPAIGNS = frozenset({"sowing", "harvest"})

# Коды регионов (согласовано с regions_mis Фазы 4.5)
REGION_CODES: List[str] = [
    "volga",
    "north_caucasus",
    "northwest",
    "central_chernozem",
    "kuban",
    "altai",
    "amur",
    "vladimir",
]

REQUIRED_COLUMNS = frozenset({
    "region_code",
    "year",
    "campaign",
    "working_days_window",
    "total_window_days",
    "source",
})


# ---------------------------------------------------------------------------
# Структура данных
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeatherWindow:
    """Одна запись погодного окна."""
    region_code: str
    year: int
    campaign: str
    working_days_window: float
    total_window_days: float
    source: str


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------
def load_weather_windows(
    path: Path = WEATHER_DATA_PATH,
) -> pd.DataFrame:
    """
    Загрузить таблицу погодных окон.

    Raises
    ------
    FileNotFoundError
        Если файл не найден.
    ValueError
        Если отсутствуют обязательные колонки или данные некорректны.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл погодных данных не найден: {path}")

    df = pd.read_csv(path)

    # Проверка колонок
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {sorted(missing)}")

    # Приведение типов
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["working_days_window"] = pd.to_numeric(
        df["working_days_window"], errors="coerce"
    )
    df["total_window_days"] = pd.to_numeric(
        df["total_window_days"], errors="coerce"
    )
    df["campaign"] = df["campaign"].astype(str).str.strip().str.lower()
    df["region_code"] = df["region_code"].astype(str).str.strip().str.lower()

    # Валидация campaign
    invalid_campaign = ~df["campaign"].isin(VALID_CAMPAIGNS)
    if invalid_campaign.any():
        bad = df.loc[invalid_campaign, "campaign"].unique()
        raise ValueError(f"Недопустимые значения campaign: {list(bad)}")

    # Валидация конечности
    for col in ("working_days_window", "total_window_days"):
        if df[col].isna().any():
            n_bad = int(df[col].isna().sum())
            raise ValueError(f"Колонка {col} содержит {n_bad} нечисловых значений")
        if (df[col] < 0).any():
            raise ValueError(f"Колонка {col} содержит отрицательные значения")

    # working_days_window не может превышать total_window_days
    if (df["working_days_window"] > df["total_window_days"]).any():
        n_bad = int((df["working_days_window"] > df["total_window_days"]).sum())
        logger.warning(
            "%d строк: working_days_window > total_window_days", n_bad
        )

    return df


# ---------------------------------------------------------------------------
# Валидация покрытия
# ---------------------------------------------------------------------------
def validate_weather_coverage(
    df: pd.DataFrame,
    min_regions: int = MIN_REGIONS,
    min_years: int = MIN_YEARS,
) -> dict:
    """
    Проверить критерий Фазы 5.2:
    данные есть для >= min_regions регионов и >= min_years лет.

    Returns
    -------
    dict с полями:
        n_regions, n_years, region_ok, year_ok,
        full_coverage_regions, coverage_ok
    """
    regions = df["region_code"].dropna().unique()
    years = df["year"].dropna().unique()
    region_ok = len(regions) >= min_regions
    year_ok = len(years) >= min_years

    # Полнота: для каждого региона должны быть все годы
    coverage = df.groupby("region_code")["year"].nunique()
    full_regions = int((coverage >= min_years).sum())

    coverage_ok = bool(region_ok and year_ok and full_regions >= min_regions)

    return {
        "n_regions": int(len(regions)),
        "n_years": int(len(years)),
        "region_ok": bool(region_ok),
        "year_ok": bool(year_ok),
        "full_coverage_regions": full_regions,
        "coverage_ok": coverage_ok,
        "required_regions": min_regions,
        "required_years": min_years,
    }


# ---------------------------------------------------------------------------
# Стандартизация
# ---------------------------------------------------------------------------
def standardize_working_days(
    df: pd.DataFrame,
    campaign: str = "sowing",
) -> pd.Series:
    """
    Стандартизовать working_days_window для использования как Z.
    Возвращает Series с mean=0, std=1.

    Parameters
    ----------
    df : DataFrame с колонками campaign, working_days_window
    campaign : "sowing" или "harvest"

    Raises
    ------
    ValueError
        Если нет данных для выбранной кампании или нулевая дисперсия.
    """
    if campaign not in VALID_CAMPAIGNS:
        raise ValueError(f"campaign должно быть одним из {sorted(VALID_CAMPAIGNS)}")

    subset = df[df["campaign"] == campaign]["working_days_window"]
    vals = pd.to_numeric(subset, errors="coerce").dropna()

    if vals.empty:
        raise ValueError(f"Нет данных для campaign='{campaign}'")

    mean = float(vals.mean())
    std = float(vals.std(ddof=1))
    if std <= 1e-9:
        raise ValueError("working_days_window имеет нулевую дисперсию")

    standardized = (subset - mean) / std
    return standardized


# ---------------------------------------------------------------------------
# Генерация синтетических данных (для Фазы 5.3)
# ---------------------------------------------------------------------------
def generate_synthetic_weather(
    regions: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    seed: int = 20260501,
) -> pd.DataFrame:
    """
    Генерация синтетических погодных данных для проверки концепции.
    Используется в Фазе 5.3, пока реальные данные не собраны.

    Параметры модели:
    - sowing:  Normal(45, 12) дней, клип [5, 60]
    - harvest: Normal(55, 15) дней, клип [5, 75]
    """
    if regions is None:
        regions = REGION_CODES
    if years is None:
        years = [2022, 2023, 2024, 2025]

    rng = np.random.default_rng(seed)
    rows: List[WeatherWindow] = []

    for region in regions:
        for year in years:
            # Sowing
            sowing_days = rng.normal(45.0, 12.0)
            sowing_days = float(np.clip(sowing_days, 5.0, 60.0))
            rows.append(WeatherWindow(
                region_code=region,
                year=year,
                campaign="sowing",
                working_days_window=sowing_days,
                total_window_days=60.0,
                source="synthetic",
            ))
            # Harvest
            harvest_days = rng.normal(55.0, 15.0)
            harvest_days = float(np.clip(harvest_days, 5.0, 75.0))
            rows.append(WeatherWindow(
                region_code=region,
                year=year,
                campaign="harvest",
                working_days_window=harvest_days,
                total_window_days=75.0,
                source="synthetic",
            ))

    df = pd.DataFrame([w.__dict__ for w in rows])
    return df


def save_synthetic_weather(
    path: Path = WEATHER_DATA_PATH,
    seed: int = 20260501,
) -> Path:
    """Сгенерировать и сохранить синтетические данные."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_weather(seed=seed)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Синтетические данные сохранены: %s (%d строк)", path, len(df))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    print("=" * 70)
    print("Фаза 5.2: Weather IV — проверка покрытия")
    print("=" * 70)

    # Если файла нет — предлагаем сгенерировать синтетические данные
    if not WEATHER_DATA_PATH.exists():
        print(f"⚠️  Файл не найден: {WEATHER_DATA_PATH}")
        print("   Это ожидаемо до Фазы 6 (реальные данные).")
        answer = input("   Сгенерировать синтетические данные для теста? [да]: ").strip().lower()
        if answer in ("", "да", "д", "yes", "y"):
            save_synthetic_weather()
            print("✅ Синтетические данные созданы.")
        else:
            print("   Пропущено. Создайте файл по схеме docs/weather_data_schema.md")
            return 1

    # Загрузка
    try:
        df = load_weather_windows()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Ошибка загрузки: %s", exc)
        return 1

    print(f"Загружено строк: {len(df):,}")
    print(f"Регионы: {sorted(df['region_code'].unique())}")
    print(f"Годы: {sorted(df['year'].dropna().unique())}")
    print(f"Кампании: {sorted(df['campaign'].unique())}")
    print()

    # Валидация покрытия
    report = validate_weather_coverage(df)
    print("-" * 70)
    print(f"Регионов:                    {report['n_regions']} "
          f"(требуется >= {report['required_regions']})")
    print(f"Лет:                         {report['n_years']} "
          f"(требуется >= {report['required_years']})")
    print(f"Регионов с полным покрытием: {report['full_coverage_regions']}")
    status = "✅" if report["coverage_ok"] else "❌"
    print(f"{status} Покрытие: {'ДОСТАТОЧНО' if report['coverage_ok'] else 'НЕДОСТАТОЧНО'}")
    print("-" * 70)

    # Стандартизация (демо для sowing)
    if report["coverage_ok"]:
        try:
            z_sowing = standardize_working_days(df, campaign="sowing")
            print(f"\nСтандартизация sowing: n={len(z_sowing)}, "
                  f"mean={z_sowing.mean():.4f}, std={z_sowing.std():.4f}")
            print("✅ working_days_window готов к использованию как Z")
        except ValueError as exc:
            logger.warning("Стандартизация не выполнена: %s", exc)

    return 0 if report["coverage_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())