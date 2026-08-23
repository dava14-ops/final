#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_weather_data.py

Фаза 6.6 — валидация реальных погодных данных.

Проверяет:

1. RAW NASA POWER:
   data/raw/weather/nasa_power_daily.csv

2. PROCESSED weather windows:
   data/processed/weather/weather_windows.csv

3. Покрытие:
   - минимум 5 регионов;
   - минимум 3 года;
   - обе кампании: sowing / harvest.

4. Источник:
   - raw source должен быть nasa_power;
   - processed source не должен быть synthetic.

5. Качество данных:
   - даты корректны;
   - нет дубликатов region/year/date;
   - погодные показатели числовые;
   - working_days_window >= 0;
   - working_days_window <= total_window_days;
   - рабочие дни физически правдоподобны.

6. Совместимость с weather_iv.py:
   - обязательные поля присутствуют;
   - region_code совпадает с ожидаемыми регионами;
   - campaign содержит ожидаемые значения;
   - source содержит реальные источники.

Результат:
    data/processed/weather/weather_validation_report.json

Важно:
    Скрипт НЕ генерирует синтетические данные.
    Скрипт только проверяет уже загруженные/рассчитанные данные.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT ROOT
# ============================================================================

# Этот файл находится в:
#
#   project/
#       test/
#           check_weather_data.py
#
# Поэтому project root = parent от test/
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# PATHS
# ============================================================================

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "weather" / "nasa_power_daily.csv"

WINDOWS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather"
    / "weather_windows.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather"
    / "weather_validation_report.json"
)


# ============================================================================
# EXPECTED REGIONS
# ============================================================================

REGION_CODES = [
    "volga",
    "north_caucasus",
    "northwest",
    "central_chernozem",
    "kuban",
    "altai",
    "amur",
    "vladimir",
]


# ============================================================================
# EXPECTED CAMPAIGNS
# ============================================================================

EXPECTED_CAMPAIGNS = {
    "sowing",
    "harvest",
}


# ============================================================================
# REQUIREMENTS
# ============================================================================

MIN_REGIONS = 5
MIN_YEARS = 3

# Минимально допустимое количество рабочих дней.
# 0 допускается на уровне отдельных кампаний/регионов,
# но предупреждение будет выдано.
MIN_REASONABLE_WORKING_DAYS = 5

# Максимально физически разумное окно для текущей конфигурации
# sowing: 04-15 ... 06-15
# harvest: 08-01 ... 10-15
MAX_REASONABLE_WORKING_DAYS = 90


# ============================================================================
# RAW REQUIRED COLUMNS
# ============================================================================

RAW_REQUIRED_COLUMNS = {
    "date",
    "region_code",
    "year",
    "PRECTOTCORR",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "WS2M",
}


# ============================================================================
# WINDOWS REQUIRED COLUMNS
# ============================================================================

WINDOW_REQUIRED_COLUMNS = {
    "region_code",
    "year",
    "campaign",
    "working_days_window",
    "total_window_days",
    "source",
}


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# HELPERS
# ============================================================================

def _safe_int(value: Any) -> int | None:
    """Безопасно преобразовать значение в int."""
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    """Безопасно преобразовать значение в float."""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _print_status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ============================================================================
# RAW VALIDATION
# ============================================================================

def validate_raw_weather() -> tuple[bool, dict[str, Any], list[str]]:
    """
    Проверить data/raw/weather/nasa_power_daily.csv.
    """

    errors: list[str] = []
    warnings: list[str] = []

    result: dict[str, Any] = {
        "path": str(RAW_PATH),
        "exists": RAW_PATH.exists(),
    }

    if not RAW_PATH.exists():
        errors.append(f"Raw weather file not found: {RAW_PATH}")
        return False, result, errors

    try:
        df = pd.read_csv(RAW_PATH)
    except Exception as exc:
        errors.append(f"Cannot read raw weather file: {exc}")
        return False, result, errors

    result["records"] = int(len(df))
    result["columns"] = list(df.columns)

    # ------------------------------------------------------------------
    # Columns
    # ------------------------------------------------------------------

    missing_columns = sorted(RAW_REQUIRED_COLUMNS - set(df.columns))

    if missing_columns:
        errors.append(
            "Raw weather file is missing columns: "
            + ", ".join(missing_columns)
        )

    if errors:
        return False, result, errors

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")

    invalid_dates = int(parsed_dates.isna().sum())

    result["invalid_dates"] = invalid_dates

    if invalid_dates > 0:
        errors.append(
            f"Raw weather contains {invalid_dates} invalid dates"
        )

    # ------------------------------------------------------------------
    # Year
    # ------------------------------------------------------------------

    numeric_year = pd.to_numeric(df["year"], errors="coerce")

    invalid_years = int(numeric_year.isna().sum())

    result["invalid_years"] = invalid_years

    if invalid_years > 0:
        errors.append(
            f"Raw weather contains {invalid_years} invalid year values"
        )

    # ------------------------------------------------------------------
    # Regions
    # ------------------------------------------------------------------

    actual_regions = sorted(
        str(x)
        for x in df["region_code"].dropna().unique()
    )

    result["regions"] = actual_regions
    result["n_regions"] = len(actual_regions)

    missing_regions = sorted(
        set(REGION_CODES) - set(actual_regions)
    )

    result["missing_expected_regions"] = missing_regions

    if len(actual_regions) < MIN_REGIONS:
        errors.append(
            f"Only {len(actual_regions)} regions found; "
            f"minimum required is {MIN_REGIONS}"
        )

    if missing_regions:
        warnings.append(
            "Expected regions missing: "
            + ", ".join(missing_regions)
        )

    # ------------------------------------------------------------------
    # Years
    # ------------------------------------------------------------------

    actual_years = sorted(
        int(x)
        for x in numeric_year.dropna().unique()
    )

    result["years"] = actual_years
    result["n_years"] = len(actual_years)

    if len(actual_years) < MIN_YEARS:
        errors.append(
            f"Only {len(actual_years)} years found; "
            f"minimum required is {MIN_YEARS}"
        )

    # ------------------------------------------------------------------
    # Source
    # ------------------------------------------------------------------

    if "source" in df.columns:
        sources = sorted(
            str(x)
            for x in df["source"].dropna().unique()
        )
    else:
        # NASA POWER loader may not need source column in every version.
        # The validation therefore does not make it mandatory in RAW.
        sources = []

    result["sources"] = sources

    if sources and "synthetic" in {
        s.lower() for s in sources
    }:
        errors.append(
            "Raw weather data contains source='synthetic'"
        )

    # ------------------------------------------------------------------
    # Numeric weather fields
    # ------------------------------------------------------------------

    numeric_columns = [
        "PRECTOTCORR",
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "WS2M",
    ]

    numeric_report: dict[str, Any] = {}

    for column in numeric_columns:
        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_count = int(converted.isna().sum())

        numeric_report[column] = {
            "invalid": invalid_count,
            "min": _safe_float(converted.min()),
            "max": _safe_float(converted.max()),
            "mean": _safe_float(converted.mean()),
        }

        if invalid_count > 0:
            warnings.append(
                f"{column}: {invalid_count} non-numeric/missing values"
            )

    result["numeric_columns"] = numeric_report

    # ------------------------------------------------------------------
    # Missingness
    # ------------------------------------------------------------------

    missing_report = {
        column: int(df[column].isna().sum())
        for column in numeric_columns
    }

    result["missing_values"] = missing_report

    # ------------------------------------------------------------------
    # Duplicate observations
    # ------------------------------------------------------------------

    duplicate_count = int(
        df.duplicated(
            subset=["region_code", "year", "date"]
        ).sum()
    )

    result["duplicate_region_year_date"] = duplicate_count

    if duplicate_count > 0:
        errors.append(
            "Raw weather contains "
            f"{duplicate_count} duplicate region/year/date records"
        )

    # ------------------------------------------------------------------
    # Date range
    # ------------------------------------------------------------------

    valid_dates = parsed_dates.dropna()

    if not valid_dates.empty:
        result["first_date"] = str(valid_dates.min().date())
        result["last_date"] = str(valid_dates.max().date())

    # ------------------------------------------------------------------
    # Expected 2022-2025 coverage
    # ------------------------------------------------------------------

    expected_years = {2022, 2023, 2024, 2025}

    missing_years = sorted(
        expected_years - set(actual_years)
    )

    result["missing_expected_years"] = missing_years

    if missing_years:
        warnings.append(
            "Expected years missing: "
            + ", ".join(map(str, missing_years))
        )

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------

    result["warnings"] = warnings
    result["status"] = "PASS" if not errors else "FAIL"

    return not errors, result, errors


# ============================================================================
# WEATHER WINDOWS VALIDATION
# ============================================================================

def validate_weather_windows() -> tuple[bool, dict[str, Any], list[str]]:
    """
    Проверить data/processed/weather/weather_windows.csv.
    """

    errors: list[str] = []
    warnings: list[str] = []

    result: dict[str, Any] = {
        "path": str(WINDOWS_PATH),
        "exists": WINDOWS_PATH.exists(),
    }

    if not WINDOWS_PATH.exists():
        errors.append(
            f"Weather windows file not found: {WINDOWS_PATH}"
        )
        return False, result, errors

    try:
        df = pd.read_csv(WINDOWS_PATH)
    except Exception as exc:
        errors.append(
            f"Cannot read weather windows file: {exc}"
        )
        return False, result, errors

    result["records"] = int(len(df))
    result["columns"] = list(df.columns)

    # ------------------------------------------------------------------
    # Columns
    # ------------------------------------------------------------------

    missing_columns = sorted(
        WINDOW_REQUIRED_COLUMNS - set(df.columns)
    )

    if missing_columns:
        errors.append(
            "Weather windows is missing columns: "
            + ", ".join(missing_columns)
        )

    if errors:
        return False, result, errors

    # ------------------------------------------------------------------
    # Basic values
    # ------------------------------------------------------------------

    numeric_columns = [
        "year",
        "working_days_window",
        "total_window_days",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_count = int(df[column].isna().sum())

        if invalid_count > 0:
            errors.append(
                f"{column}: {invalid_count} invalid values"
            )

    # ------------------------------------------------------------------
    # Regions
    # ------------------------------------------------------------------

    actual_regions = sorted(
        str(x)
        for x in df["region_code"].dropna().unique()
    )

    result["regions"] = actual_regions
    result["n_regions"] = len(actual_regions)

    missing_regions = sorted(
        set(REGION_CODES) - set(actual_regions)
    )

    result["missing_expected_regions"] = missing_regions

    if len(actual_regions) < MIN_REGIONS:
        errors.append(
            f"Only {len(actual_regions)} regions found in "
            f"weather_windows; minimum required is {MIN_REGIONS}"
        )

    if missing_regions:
        warnings.append(
            "Weather windows missing expected regions: "
            + ", ".join(missing_regions)
        )

    # ------------------------------------------------------------------
    # Years
    # ------------------------------------------------------------------

    valid_years = sorted(
        int(x)
        for x in df["year"].dropna().unique()
    )

    result["years"] = valid_years
    result["n_years"] = len(valid_years)

    if len(valid_years) < MIN_YEARS:
        errors.append(
            f"Only {len(valid_years)} years found in weather_windows; "
            f"minimum required is {MIN_YEARS}"
        )

    # ------------------------------------------------------------------
    # Campaigns
    # ------------------------------------------------------------------

    campaigns = sorted(
        str(x)
        for x in df["campaign"].dropna().unique()
    )

    result["campaigns"] = campaigns

    missing_campaigns = sorted(
        EXPECTED_CAMPAIGNS - set(campaigns)
    )

    result["missing_expected_campaigns"] = missing_campaigns

    if missing_campaigns:
        errors.append(
            "Missing expected campaigns: "
            + ", ".join(missing_campaigns)
        )

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    sources = sorted(
        str(x)
        for x in df["source"].dropna().unique()
    )

    result["sources"] = sources

    synthetic_sources = [
        source
        for source in sources
        if source.lower() == "synthetic"
    ]

    result["synthetic_sources"] = synthetic_sources

    if synthetic_sources:
        errors.append(
            "Weather windows still contains source='synthetic'"
        )

    # For the real-data phase we expect NASA POWER.
    nasa_rows = int(
        (
            df["source"]
            .astype(str)
            .str.lower()
            == "nasa_power"
        ).sum()
    )

    result["nasa_power_rows"] = nasa_rows

    if nasa_rows == 0:
        errors.append(
            "No rows with source='nasa_power' found"
        )

    # ------------------------------------------------------------------
    # Working days
    # ------------------------------------------------------------------

    negative_working_days = int(
        (df["working_days_window"] < 0).sum()
    )

    result["negative_working_days"] = negative_working_days

    if negative_working_days > 0:
        errors.append(
            f"{negative_working_days} rows have "
            "working_days_window < 0"
        )

    # ------------------------------------------------------------------
    # Working days cannot exceed window
    # ------------------------------------------------------------------

    exceeds_window = int(
        (
            df["working_days_window"]
            > df["total_window_days"]
        ).sum()
    )

    result["working_days_exceed_window"] = exceeds_window

    if exceeds_window > 0:
        errors.append(
            f"{exceeds_window} rows have working_days_window "
            "> total_window_days"
        )

    # ------------------------------------------------------------------
    # Window must be positive
    # ------------------------------------------------------------------

    invalid_windows = int(
        (df["total_window_days"] <= 0).sum()
    )

    result["invalid_total_windows"] = invalid_windows

    if invalid_windows > 0:
        errors.append(
            f"{invalid_windows} rows have "
            "total_window_days <= 0"
        )

    # ------------------------------------------------------------------
    # Physical plausibility
    # ------------------------------------------------------------------

    too_short = int(
        (
            df["working_days_window"]
            < MIN_REASONABLE_WORKING_DAYS
        ).sum()
    )

    too_long = int(
        (
            df["working_days_window"]
            > MAX_REASONABLE_WORKING_DAYS
        ).sum()
    )

    result["working_days_below_reasonable_min"] = too_short
    result["working_days_above_reasonable_max"] = too_long

    if too_short > 0:
        warnings.append(
            f"{too_short} rows have fewer than "
            f"{MIN_REASONABLE_WORKING_DAYS} working days"
        )

    if too_long > 0:
        errors.append(
            f"{too_long} rows have more than "
            f"{MAX_REASONABLE_WORKING_DAYS} working days"
        )

    # ------------------------------------------------------------------
    # Duplicate region/year/campaign
    # ------------------------------------------------------------------

    duplicates = int(
        df.duplicated(
            subset=[
                "region_code",
                "year",
                "campaign",
            ]
        ).sum()
    )

    result["duplicate_region_year_campaign"] = duplicates

    if duplicates > 0:
        errors.append(
            "Weather windows contains "
            f"{duplicates} duplicate "
            "region/year/campaign records"
        )

    # ------------------------------------------------------------------
    # Expected row count
    # ------------------------------------------------------------------

    expected_min_rows = (
        MIN_REGIONS
        * MIN_YEARS
        * len(EXPECTED_CAMPAIGNS)
    )

    result["expected_min_rows"] = expected_min_rows

    if len(df) < expected_min_rows:
        errors.append(
            f"Only {len(df)} weather-window rows found; "
            f"minimum expected is {expected_min_rows}"
        )

    # ------------------------------------------------------------------
    # Distribution statistics
    # ------------------------------------------------------------------

    result["working_days_statistics"] = {
        "min": _safe_float(df["working_days_window"].min()),
        "median": _safe_float(df["working_days_window"].median()),
        "mean": _safe_float(df["working_days_window"].mean()),
        "p25": _safe_float(
            df["working_days_window"].quantile(0.25)
        ),
        "p75": _safe_float(
            df["working_days_window"].quantile(0.75)
        ),
        "max": _safe_float(df["working_days_window"].max()),
    }

    # ------------------------------------------------------------------
    # Coverage matrix
    # ------------------------------------------------------------------

    coverage = (
        df.groupby(
            ["region_code", "year"]
        )["campaign"]
        .nunique()
        .reset_index(name="campaigns")
    )

    result["coverage_pairs"] = int(len(coverage))

    incomplete_pairs = int(
        (coverage["campaigns"] < len(EXPECTED_CAMPAIGNS)).sum()
    )

    result["incomplete_region_year_pairs"] = incomplete_pairs

    if incomplete_pairs > 0:
        warnings.append(
            f"{incomplete_pairs} region/year pairs do not contain "
            "both expected campaigns"
        )

    result["warnings"] = warnings
    result["status"] = "PASS" if not errors else "FAIL"

    return not errors, result, errors


# ============================================================================
# CROSS-CHECK RAW VS WINDOWS
# ============================================================================

def cross_check_datasets(
    raw_result: dict[str, Any],
    windows_result: dict[str, Any],
) -> tuple[bool, list[str], list[str]]:
    """
    Проверить логическую согласованность RAW и processed данных.
    """

    errors: list[str] = []
    warnings: list[str] = []

    raw_regions = set(
        raw_result.get("regions", [])
    )

    window_regions = set(
        windows_result.get("regions", [])
    )

    unknown_window_regions = sorted(
        window_regions - raw_regions
    )

    if unknown_window_regions:
        errors.append(
            "Weather windows contains regions absent from "
            "raw NASA POWER data: "
            + ", ".join(unknown_window_regions)
        )

    raw_years = set(
        raw_result.get("years", [])
    )

    window_years = set(
        windows_result.get("years", [])
    )

    unknown_window_years = sorted(
        window_years - raw_years
    )

    if unknown_window_years:
        errors.append(
            "Weather windows contains years absent from "
            "raw NASA POWER data: "
            + ", ".join(map(str, unknown_window_years))
        )

    return not errors, errors, warnings


# ============================================================================
# REPORT
# ============================================================================

def build_report(
    raw_ok: bool,
    raw_result: dict[str, Any],
    raw_errors: list[str],
    windows_ok: bool,
    windows_result: dict[str, Any],
    windows_errors: list[str],
    cross_ok: bool,
    cross_errors: list[str],
) -> dict[str, Any]:

    all_errors = (
        raw_errors
        + windows_errors
        + cross_errors
    )

    all_warnings = (
        raw_result.get("warnings", [])
        + windows_result.get("warnings", [])
    )

    overall_ok = (
        raw_ok
        and windows_ok
        and cross_ok
    )

    return {
        "phase": "6.6",
        "status": "PASS" if overall_ok else "FAIL",
        "project_root": str(PROJECT_ROOT),
        "raw": {
            "status": "PASS" if raw_ok else "FAIL",
            "errors": raw_errors,
            "details": raw_result,
        },
        "weather_windows": {
            "status": "PASS" if windows_ok else "FAIL",
            "errors": windows_errors,
            "details": windows_result,
        },
        "cross_check": {
            "status": "PASS" if cross_ok else "FAIL",
            "errors": cross_errors,
        },
        "errors": all_errors,
        "warnings": all_warnings,
        "criteria": {
            "min_regions": MIN_REGIONS,
            "min_years": MIN_YEARS,
            "expected_campaigns": sorted(
                EXPECTED_CAMPAIGNS
            ),
            "max_reasonable_working_days":
                MAX_REASONABLE_WORKING_DAYS,
            "synthetic_source_forbidden": True,
        },
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    print("=" * 70)
    print("ФАЗА 6.6 — WEATHER DATA VALIDATION")
    print("=" * 70)
    print()

    print(f"Project root: {PROJECT_ROOT}")
    print()

    # ------------------------------------------------------------------
    # RAW
    # ------------------------------------------------------------------

    raw_ok, raw_result, raw_errors = (
        validate_raw_weather()
    )

    print("RAW NASA POWER DATA")
    print(f"  Status:   {_print_status(raw_ok)}")

    if raw_result.get("records") is not None:
        print(
            f"  Records:  "
            f"{raw_result['records']:,}"
        )

    if raw_result.get("n_regions") is not None:
        print(
            f"  Regions:  "
            f"{raw_result['n_regions']}"
        )

    if raw_result.get("n_years") is not None:
        print(
            f"  Years:    "
            f"{raw_result['n_years']}"
        )

    if raw_result.get("first_date"):
        print(
            f"  Dates:    "
            f"{raw_result['first_date']} — "
            f"{raw_result['last_date']}"
        )

    if raw_errors:
        for error in raw_errors:
            print(f"  ❌ {error}")

    print()

    # ------------------------------------------------------------------
    # WEATHER WINDOWS
    # ------------------------------------------------------------------

    windows_ok, windows_result, windows_errors = (
        validate_weather_windows()
    )

    print("WEATHER WINDOWS")
    print(f"  Status:   {_print_status(windows_ok)}")

    if windows_result.get("records") is not None:
        print(
            f"  Records:  "
            f"{windows_result['records']:,}"
        )

    print(
        f"  Regions:  "
        f"{windows_result.get('n_regions', 0)}"
    )

    print(
        f"  Years:    "
        f"{windows_result.get('n_years', 0)}"
    )

    print(
        f"  Campaigns: "
        f"{windows_result.get('campaigns', [])}"
    )

    print(
        f"  Sources:   "
        f"{windows_result.get('sources', [])}"
    )

    if windows_errors:
        for error in windows_errors:
            print(f"  ❌ {error}")

    print()

    # ------------------------------------------------------------------
    # CROSS-CHECK
    # ------------------------------------------------------------------

    cross_ok, cross_errors, _ = cross_check_datasets(
        raw_result,
        windows_result,
    )

    print("RAW ↔ WEATHER WINDOWS CROSS-CHECK")
    print(
        f"  Status:   "
        f"{_print_status(cross_ok)}"
    )

    if cross_errors:
        for error in cross_errors:
            print(f"  ❌ {error}")
    else:
        print("  Regions/years are consistent.")

    print()

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------

    report = build_report(
        raw_ok=raw_ok,
        raw_result=raw_result,
        raw_errors=raw_errors,
        windows_ok=windows_ok,
        windows_result=windows_result,
        windows_errors=windows_errors,
        cross_ok=cross_ok,
        cross_errors=cross_errors,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    overall_ok = report["status"] == "PASS"

    print("=" * 70)

    if overall_ok:
        print("✅ WEATHER DATA VALIDATION PASSED")
        print()
        print("Фаза 6.6: данные готовы к использованию.")
        print(
            "Источник погодных данных: NASA POWER "
            "(не synthetic)."
        )
    else:
        print("❌ WEATHER DATA VALIDATION FAILED")
        print()

        if report["errors"]:
            print("Ошибки:")

            for index, error in enumerate(
                report["errors"],
                start=1,
            ):
                print(f" {index}. {error}")

    print()
    print(f"Report: {REPORT_PATH}")
    print("=" * 70)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())