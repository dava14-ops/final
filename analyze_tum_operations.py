#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_tum_operations.py

TUM CAN BUS — REAL OPERATION ANALYSIS

Назначение
----------
Извлекает реальный Engine Load из TUM CAN BUS telemetry и строит
статистику PeakLoad по сельскохозяйственным операциям.

Источник PeakLoad
-----------------
    EngPercentLoadAtCurrentSpeed_(%)

Преобразование
--------------
    PeakLoad = EngPercentLoadAtCurrentSpeed_(%) / 100

Например:
    45.7 % -> 0.457

Baseline
--------
Transport используется как реальная базовая нагрузка трактора:

    baseline = mean(PeakLoad | Transport)

Для каждой операции дополнительно рассчитывается:

    relative_to_transport = operation_mean / transport_mean

Важно
-----
- Никаких синтетических PeakLoad.
- Никаких случайных значений.
- Никакой замены Engine Load на RearDraft.
- Нулевая нагрузка считается валидным наблюдением.
- NaN / +/-inf удаляются.
- Повреждённый CSV не останавливает весь анализ.
- Отдельная диагностика показывает, почему файл не был использован.

Ожидаемая структура:

data/
└── raw/
    └── tum/
        ├── Fendt 211/
        │   ├── Power harrowing/
        │   │   ├── Field_1.csv
        │   │   └── Field_2.csv
        │   └── ...
        ├── Fendt 314/
        ├── Fendt 722/
        ├── Fendt 724/
        └── Fendt 820/

Результат:

data/processed/tum/tum_operation_stats.json
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

TUM_ROOT = SCRIPT_DIR / "data" / "raw" / "tum"

OUTPUT_DIR = SCRIPT_DIR / "data" / "processed" / "tum"
OUTPUT_FILE = OUTPUT_DIR / "tum_operation_stats.json"

BASELINE_OPERATION = "Transport"

PEAKLOAD_COLUMN = "EngPercentLoadAtCurrentSpeed_(%)"

# Допустимый физический диапазон исходного значения в процентах.
# 0..100 — нормальный диапазон Engine Load.
RAW_LOAD_MIN = 0.0
RAW_LOAD_MAX = 100.0

# Минимальное число валидных наблюдений для статистики операции.
MIN_VALID_OBSERVATIONS = 1

# Ограничение памяти при чтении CSV.
# При необходимости pandas автоматически будет использовать chunks.
CSV_CHUNK_SIZE = 250_000

# Операции, которые гарантированно должны быть представлены в JSON,
# если они реально обнаружены в TUM.
EXPECTED_OPERATION_ORDER = [
    "Cultivating (deep)",
    "Cultivating (shallow)",
    "Disc harrowing",
    "Fertilizing",
    "Mowing (front)",
    "Mowing (large-scale)",
    "Mulching",
    "Ploughing",
    "Power harrowing",
    "Precision air seeding",
    "Rotary tilling",
    "Seed drill combination",
    "Seed drill combination 3m",
    "Seed drill combination 4m",
    "Seedbed combination",
    "Spraying",
    "Swathing",
    "Transport",
]

# Возможные кодировки CSV.
CSV_ENCODINGS = [
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",
]

# Разделители, которые могут встретиться в TUM CSV.
CSV_SEPARATORS = [",", ";", "\t"]

TRACTOR_NAME_PATTERN = re.compile(r"^Fendt\s+\d+$", re.IGNORECASE)


# ============================================================================
# LOGGING
# ============================================================================

def configure_logging() -> None:
    """
    Настраивает UTF-8 logging для Windows/Linux.
    """

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class TUMFile:
    """
    Один TUM Field CSV.
    """

    path: Path
    tractor: str
    operation: str


@dataclass
class LoadExtraction:
    """
    Результат извлечения Engine Load из одного CSV.
    """

    values: np.ndarray
    raw_count: int
    valid_count: int
    invalid_count: int
    outside_range_count: int


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def safe_float(value: Any) -> float | None:
    """
    Безопасное преобразование в float.
    """

    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def percentile(values: np.ndarray, q: float) -> float | None:
    """
    Безопасный percentile.
    """

    if values.size == 0:
        return None

    return float(np.percentile(values, q))


def mean(values: np.ndarray) -> float | None:
    """
    Безопасное среднее.
    """

    if values.size == 0:
        return None

    return float(np.mean(values))


def median(values: np.ndarray) -> float | None:
    """
    Безопасная медиана.
    """

    if values.size == 0:
        return None

    return float(np.median(values))


def std(values: np.ndarray) -> float | None:
    """
    Безопасное стандартное отклонение.
    """

    if values.size <= 1:
        return 0.0 if values.size == 1 else None

    return float(np.std(values, ddof=1))


def min_value(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None

    return float(np.min(values))


def max_value(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None

    return float(np.max(values))


def round_or_none(value: float | None, digits: int = 8) -> float | None:
    if value is None:
        return None

    return round(float(value), digits)


# ============================================================================
# PATH DISCOVERY
# ============================================================================

def find_tum_root() -> Path:
    """
    Находит корневую директорию TUM.

    В первую очередь используется:
        data/raw/tum

    Также поддерживаются несколько типичных вариантов.
    """

    candidates = [
        TUM_ROOT,
        Path.cwd() / "data" / "raw" / "tum",
        SCRIPT_DIR / "tum",
        Path.cwd() / "tum",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            logger.info("Found TUM root: %s", candidate.resolve())
            return candidate.resolve()

    raise FileNotFoundError(
        "TUM root not found. Expected: "
        f"{TUM_ROOT}"
    )


def is_tractor_directory(path: Path) -> bool:
    """
    Проверяет, является ли директория директорией трактора.
    """

    return path.is_dir() and bool(
        TRACTOR_NAME_PATTERN.match(path.name.strip())
    )


# ============================================================================
# OPERATION DISCOVERY
# ============================================================================

def discover_tum_files(tum_root: Path) -> list[TUMFile]:
    """
    Находит все Field_*.csv внутри:

        TUM root / Fendt XXX / operation / Field_*.csv
    """

    discovered: list[TUMFile] = []

    tractor_dirs = sorted(
        [
            p
            for p in tum_root.iterdir()
            if is_tractor_directory(p)
        ],
        key=lambda p: p.name.lower(),
    )

    logger.info(
        "Found tractor directories: %d",
        len(tractor_dirs),
    )

    if not tractor_dirs:
        raise RuntimeError(
            f"No Fendt tractor directories found in {tum_root}"
        )

    for tractor_dir in tractor_dirs:
        tractor_name = tractor_dir.name.strip()

        logger.info(
            "Scanning tractor: %s",
            tractor_name,
        )

        operation_dirs = sorted(
            [
                p
                for p in tractor_dir.iterdir()
                if p.is_dir()
            ],
            key=lambda p: p.name.lower(),
        )

        for operation_dir in operation_dirs:
            operation_name = operation_dir.name.strip()

            csv_files = sorted(
                operation_dir.glob("*.csv"),
                key=lambda p: p.name.lower(),
            )

            if not csv_files:
                continue

            logger.info(
                "  %-38s %5d files",
                operation_name,
                len(csv_files),
            )

            for csv_path in csv_files:
                discovered.append(
                    TUMFile(
                        path=csv_path,
                        tractor=tractor_name,
                        operation=operation_name,
                    )
                )

    return discovered


def group_by_operation(
    files: list[TUMFile],
) -> dict[str, list[TUMFile]]:
    """
    Группирует CSV по операциям.
    """

    result: dict[str, list[TUMFile]] = defaultdict(list)

    for item in files:
        result[item.operation].append(item)

    return dict(
        sorted(
            result.items(),
            key=lambda item: item[0].lower(),
        )
    )


def group_by_tractor(
    files: list[TUMFile],
) -> dict[str, dict[str, list[TUMFile]]]:
    """
    Группирует:

        tractor -> operation -> files
    """

    result: dict[str, dict[str, list[TUMFile]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for item in files:
        result[item.tractor][item.operation].append(item)

    return {
        tractor: dict(
            sorted(
                operations.items(),
                key=lambda item: item[0].lower(),
            )
        )
        for tractor, operations in sorted(
            result.items(),
            key=lambda item: item[0].lower(),
        )
    }


# ============================================================================
# CSV READING
# ============================================================================

def detect_separator(path: Path) -> str:
    """
    Определяет separator CSV.

    TUM обычно использует ','.
    Для надёжности используется csv.Sniffer.
    """

    for encoding in CSV_ENCODINGS:
        try:
            with path.open(
                "r",
                encoding=encoding,
                errors="replace",
                newline="",
            ) as fh:
                sample = fh.read(8192)

            if not sample.strip():
                return ","

            try:
                dialect = csv.Sniffer().sniff(
                    sample,
                    delimiters="," + ";" + "\t",
                )
                return dialect.delimiter
            except csv.Error:
                pass

            # Fallback по количеству разделителей.
            counts = {
                sep: sample.count(sep)
                for sep in CSV_SEPARATORS
            }

            return max(
                counts,
                key=counts.get,
            )

        except OSError:
            continue

    return ","


def read_csv_columns(path: Path) -> list[str]:
    """
    Читает только header.
    """

    separator = detect_separator(path)

    last_error: Exception | None = None

    for encoding in CSV_ENCODINGS:
        try:
            frame = pd.read_csv(
                path,
                sep=separator,
                encoding=encoding,
                nrows=0,
                engine="python",
            )

            return [
                str(column).strip()
                for column in frame.columns
            ]

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Unable to read CSV header: {path}: {last_error}"
    )


def find_peakload_column(columns: list[str]) -> str | None:
    """
    Находит реальный Engine Load column.

    Приоритет:
        1. точное имя
        2. нормализованное сравнение
    """

    if PEAKLOAD_COLUMN in columns:
        return PEAKLOAD_COLUMN

    normalized_target = re.sub(
        r"[^a-z0-9]",
        "",
        PEAKLOAD_COLUMN.lower(),
    )

    for column in columns:
        normalized = re.sub(
            r"[^a-z0-9]",
            "",
            str(column).lower(),
        )

        if normalized == normalized_target:
            return str(column)

    return None


def read_peakload_from_csv(
    path: Path,
) -> LoadExtraction:
    """
    Извлекает Engine Load из CSV.

    ВАЖНО:
    Нулевые значения считаются валидными.

    Исходный диапазон:
        0..100 %

    Результат:
        0..1
    """

    separator = detect_separator(path)

    peak_column: str | None = None

    # ------------------------------------------------------------------
    # First try: exact expected column
    # ------------------------------------------------------------------

    columns = read_csv_columns(path)

    peak_column = find_peakload_column(columns)

    if peak_column is None:
        raise KeyError(
            f"PeakLoad column not found. "
            f"Expected: {PEAKLOAD_COLUMN}. "
            f"Available columns: {columns}"
        )

    chunks: list[np.ndarray] = []

    raw_count = 0
    invalid_count = 0
    outside_range_count = 0

    for encoding in CSV_ENCODINGS:
        try:
            chunks.clear()
            raw_count = 0
            invalid_count = 0
            outside_range_count = 0

            for chunk in pd.read_csv(
                path,
                sep=separator,
                encoding=encoding,
                usecols=[peak_column],
                chunksize=CSV_CHUNK_SIZE,
                engine="python",
            ):
                series = chunk[peak_column]

                raw_count += int(series.shape[0])

                numeric = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                invalid_mask = numeric.isna()
                invalid_count += int(invalid_mask.sum())

                numeric = numeric.dropna()

                if numeric.empty:
                    continue

                # Проверяем физический диапазон ДО деления на 100.
                outside_mask = (
                    (numeric < RAW_LOAD_MIN)
                    | (numeric > RAW_LOAD_MAX)
                )

                outside_range_count += int(
                    outside_mask.sum()
                )

                numeric = numeric[
                    ~outside_mask
                ]

                if numeric.empty:
                    continue

                values = (
                    numeric.to_numpy(dtype=np.float64)
                    / 100.0
                )

                # Дополнительная защита.
                values = values[
                    np.isfinite(values)
                ]

                if values.size:
                    chunks.append(values)

            break

        except (
            UnicodeDecodeError,
            UnicodeError,
            pd.errors.ParserError,
        ) as exc:
            chunks.clear()
            raw_count = 0
            invalid_count = 0
            outside_range_count = 0

            last_error = exc
            continue

    if not chunks:
        values = np.empty(
            0,
            dtype=np.float64,
        )
    else:
        values = np.concatenate(chunks)

    return LoadExtraction(
        values=values,
        raw_count=raw_count,
        valid_count=int(values.size),
        invalid_count=invalid_count,
        outside_range_count=outside_range_count,
    )


# ============================================================================
# STATISTICS
# ============================================================================

def calculate_statistics(
    values: np.ndarray,
) -> dict[str, Any]:
    """
    Рассчитывает статистику PeakLoad.
    """

    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    return {
        "n": int(values.size),
        "mean": round_or_none(mean(values)),
        "median": round_or_none(median(values)),
        "std": round_or_none(std(values)),
        "min": round_or_none(min_value(values)),
        "p01": round_or_none(percentile(values, 1)),
        "p05": round_or_none(percentile(values, 5)),
        "p10": round_or_none(percentile(values, 10)),
        "p25": round_or_none(percentile(values, 25)),
        "p50": round_or_none(percentile(values, 50)),
        "p75": round_or_none(percentile(values, 75)),
        "p90": round_or_none(percentile(values, 90)),
        "p95": round_or_none(percentile(values, 95)),
        "p99": round_or_none(percentile(values, 99)),
        "max": round_or_none(max_value(values)),
    }


def calculate_relative_load(
    operation_mean: float | None,
    baseline_mean: float,
) -> float | None:
    """
    Отношение средней нагрузки операции к Transport baseline.
    """

    if operation_mean is None:
        return None

    if not math.isfinite(baseline_mean):
        return None

    if baseline_mean <= 0:
        return None

    return round(
        operation_mean / baseline_mean,
        8,
    )


# ============================================================================
# BASELINE
# ============================================================================

def calculate_transport_baseline(
    transport_files: list[TUMFile],
) -> tuple[float, dict[str, Any]]:
    """
    Рассчитывает Transport baseline.

    Важный момент:
    нулевые Engine Load наблюдения НЕ удаляются.
    """

    logger.info(
        "Calculating baseline from '%s'...",
        BASELINE_OPERATION,
    )

    all_values: list[np.ndarray] = []

    total_raw = 0
    total_valid = 0
    total_invalid = 0
    total_outside = 0

    files_used = 0
    files_failed = 0

    for item in transport_files:
        try:
            extraction = read_peakload_from_csv(
                item.path
            )

            total_raw += extraction.raw_count
            total_valid += extraction.valid_count
            total_invalid += extraction.invalid_count
            total_outside += extraction.outside_range_count

            if extraction.values.size:
                all_values.append(
                    extraction.values
                )
                files_used += 1

        except Exception as exc:
            files_failed += 1

            logger.warning(
                "Transport file skipped: %s | %s",
                item.path,
                exc,
            )

    if not all_values:
        raise RuntimeError(
            "Transport baseline contains no valid "
            f"{PEAKLOAD_COLUMN} observations."
        )

    values = np.concatenate(all_values)

    baseline = float(np.mean(values))

    if not math.isfinite(baseline):
        raise RuntimeError(
            "Transport baseline is not finite."
        )

    if baseline <= 0:
        raise RuntimeError(
            "Transport baseline is <= 0. "
            "Cannot calculate relative operation load."
        )

    stats = calculate_statistics(values)

    stats.update(
        {
            "operation": BASELINE_OPERATION,
            "files_total": len(transport_files),
            "files_used": files_used,
            "files_failed": files_failed,
            "raw_observations": total_raw,
            "valid_observations": total_valid,
            "invalid_observations": total_invalid,
            "outside_range_observations": total_outside,
            "peakload_definition": (
                f"{PEAKLOAD_COLUMN} / 100"
            ),
            "units": "fraction_of_engine_load",
        }
    )

    logger.info(
        "Transport baseline mean PeakLoad = %.6f (n=%d)",
        baseline,
        values.size,
    )

    return baseline, stats


# ============================================================================
# OPERATION ANALYSIS
# ============================================================================

def analyze_operation(
    operation: str,
    files: list[TUMFile],
    baseline: float,
) -> tuple[dict[str, Any] | None, np.ndarray]:
    """
    Анализирует одну операцию.
    """

    logger.info(
        "Analyzing operation: %s (%d files)",
        operation,
        len(files),
    )

    all_values: list[np.ndarray] = []

    files_used = 0
    files_failed = 0

    raw_observations = 0
    valid_observations = 0
    invalid_observations = 0
    outside_range_observations = 0

    tractor_counts: dict[str, int] = defaultdict(int)

    for item in files:
        try:
            extraction = read_peakload_from_csv(
                item.path
            )

            raw_observations += extraction.raw_count
            valid_observations += extraction.valid_count
            invalid_observations += extraction.invalid_count
            outside_range_observations += (
                extraction.outside_range_count
            )

            if extraction.values.size == 0:
                logger.warning(
                    "No valid PeakLoad observations: %s",
                    item.path,
                )
                continue

            all_values.append(
                extraction.values
            )

            files_used += 1
            tractor_counts[item.tractor] += int(
                extraction.values.size
            )

        except KeyError as exc:
            files_failed += 1

            logger.warning(
                "%s: %s",
                item.path,
                exc,
            )

        except Exception as exc:
            files_failed += 1

            logger.warning(
                "Failed reading %s: %s",
                item.path,
                exc,
            )

    if not all_values:
        logger.warning(
            "Operation '%s': no valid PeakLoad observations.",
            operation,
        )

        return None, np.empty(
            0,
            dtype=np.float64,
        )

    values = np.concatenate(all_values)

    stats = calculate_statistics(values)

    operation_mean = (
        float(np.mean(values))
        if values.size
        else None
    )

    relative = calculate_relative_load(
        operation_mean,
        baseline,
    )

    stats.update(
        {
            "operation": operation,
            "files_total": len(files),
            "files_used": files_used,
            "files_failed": files_failed,
            "raw_observations": raw_observations,
            "valid_observations": valid_observations,
            "invalid_observations": invalid_observations,
            "outside_range_observations": (
                outside_range_observations
            ),
            "relative_to_transport": relative,
            "baseline_transport_mean": round(
                baseline,
                8,
            ),
            "peakload_source": PEAKLOAD_COLUMN,
            "peakload_transformation": (
                "EngPercentLoadAtCurrentSpeed_(%) / 100"
            ),
            "peakload_units": "fraction",
            "tractors": dict(
                sorted(
                    tractor_counts.items()
                )
            ),
        }
    )

    return stats, values


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def diagnose_file(path: Path) -> dict[str, Any]:
    """
    Возвращает диагностическую информацию для CSV.

    Полезно для случаев, когда конкретный файл не попал в статистику.
    """

    result: dict[str, Any] = {
        "file": str(path),
        "exists": path.exists(),
        "columns": [],
        "peakload_column_found": False,
        "raw_observations": 0,
        "valid_observations": 0,
        "invalid_observations": 0,
        "outside_range_observations": 0,
    }

    if not path.exists():
        return result

    try:
        columns = read_csv_columns(path)

        result["columns"] = columns

        peak_column = find_peakload_column(
            columns
        )

        result["peakload_column_found"] = (
            peak_column is not None
        )

        if peak_column is not None:
            extraction = read_peakload_from_csv(
                path
            )

            result.update(
                {
                    "raw_observations": extraction.raw_count,
                    "valid_observations": (
                        extraction.valid_count
                    ),
                    "invalid_observations": (
                        extraction.invalid_count
                    ),
                    "outside_range_observations": (
                        extraction.outside_range_count
                    ),
                }
            )

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ============================================================================
# JSON SERIALIZATION
# ============================================================================

def build_output(
    tum_root: Path,
    discovered_files: list[TUMFile],
    operation_stats: dict[str, dict[str, Any]],
    baseline: float,
    baseline_stats: dict[str, Any],
) -> dict[str, Any]:
    """
    Формирует итоговый JSON.
    """

    grouped = group_by_operation(
        discovered_files
    )

    tractors = sorted(
        {
            item.tractor
            for item in discovered_files
        },
        key=str.lower,
    )

    operation_file_counts = {
        operation: len(files)
        for operation, files in grouped.items()
    }

    return {
        "schema_version": "1.0",
        "analysis_version": "1.0",
        "source": {
            "dataset": "TUM CAN BUS",
            "root": str(tum_root),
            "synthetic_data": False,
        },
        "peakload": {
            "column": PEAKLOAD_COLUMN,
            "raw_units": "percent",
            "model_units": "fraction",
            "transformation": (
                "EngPercentLoadAtCurrentSpeed_(%) / 100"
            ),
            "valid_raw_range": [
                RAW_LOAD_MIN,
                RAW_LOAD_MAX,
            ],
            "zero_is_valid": True,
        },
        "baseline": {
            "operation": BASELINE_OPERATION,
            "mean_peakload": round(
                baseline,
                8,
            ),
            "statistics": baseline_stats,
        },
        "dataset": {
            "tractors": tractors,
            "tractor_count": len(tractors),
            "operations_discovered": len(grouped),
            "field_csv_files": len(discovered_files),
            "operation_file_counts": (
                operation_file_counts
            ),
        },
        "operations": operation_stats,
    }


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """
    Сохраняет JSON UTF-8.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fh:
        json.dump(
            data,
            fh,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    os.replace(
        temporary_path,
        path,
    )


# ============================================================================
# CONSOLE REPORT
# ============================================================================

def print_header(tum_root: Path) -> None:
    print()
    print("=" * 100)
    print("TUM CAN BUS - REAL OPERATION ANALYSIS")
    print("=" * 100)
    print(f"TUM root:  {tum_root}")
    print(f"Output:    {OUTPUT_FILE}")
    print(f"Baseline:  {BASELINE_OPERATION}")
    print(
        "PeakLoad:  "
        f"{PEAKLOAD_COLUMN} -> percent / 100"
    )
    print("=" * 100)


def print_discovery(
    files: list[TUMFile],
) -> None:
    grouped = group_by_operation(files)

    print()
    print("-" * 80)
    print("STEP 1 - DISCOVER REAL TUM OPERATIONS")
    print("-" * 80)

    print(
        f"Operations discovered: {len(grouped)}"
    )
    print(
        f"Field CSV files: {len(files)}"
    )

    ordered_operations: list[str] = []

    for operation in EXPECTED_OPERATION_ORDER:
        if operation in grouped:
            ordered_operations.append(
                operation
            )

    for operation in grouped:
        if operation not in ordered_operations:
            ordered_operations.append(
                operation
            )

    for operation in ordered_operations:
        print(
            f"  {operation:<38} "
            f"{len(grouped[operation]):>5} files"
        )


def print_baseline(
    baseline: float,
    baseline_stats: dict[str, Any],
) -> None:
    print()
    print("-" * 80)
    print("STEP 2 - REAL TRANSPORT BASELINE")
    print("-" * 80)

    print(
        "Transport baseline mean PeakLoad = "
        f"{baseline:.6f}"
    )

    print(
        f"Valid observations: "
        f"{baseline_stats['valid_observations']:,}"
    )

    print(
        f"Files used: "
        f"{baseline_stats['files_used']}"
        f"/{baseline_stats['files_total']}"
    )

    print(
        f"P25={baseline_stats['p25']:.6f}  "
        f"Median={baseline_stats['p50']:.6f}  "
        f"P75={baseline_stats['p75']:.6f}"
    )


def print_operation_results(
    stats: dict[str, dict[str, Any]],
    baseline: float,
) -> None:
    print()
    print("-" * 100)
    print("STEP 3 - REAL PEAKLOAD BY OPERATION")
    print("-" * 100)

    header = (
        f"{'Operation':<38}"
        f"{'N':>12}"
        f"{'Mean':>12}"
        f"{'P50':>12}"
        f"{'P95':>12}"
        f"{'vs Transport':>16}"
    )

    print(header)
    print("-" * 100)

    for operation in sorted(
        stats,
        key=str.lower,
    ):
        item = stats[operation]

        n = item["n"]
        operation_mean = item["mean"]
        p50 = item["p50"]
        p95 = item["p95"]
        relative = item[
            "relative_to_transport"
        ]

        mean_text = (
            f"{operation_mean:.6f}"
            if operation_mean is not None
            else "N/A"
        )

        p50_text = (
            f"{p50:.6f}"
            if p50 is not None
            else "N/A"
        )

        p95_text = (
            f"{p95:.6f}"
            if p95 is not None
            else "N/A"
        )

        relative_text = (
            f"{relative:.3f}x"
            if relative is not None
            else "N/A"
        )

        print(
            f"{operation:<38}"
            f"{n:>12,}"
            f"{mean_text:>12}"
            f"{p50_text:>12}"
            f"{p95_text:>12}"
            f"{relative_text:>16}"
        )

    print()
    print(
        "Interpretation:"
    )
    print(
        "  relative_to_transport = "
        "operation mean PeakLoad / Transport mean PeakLoad"
    )
    print(
        f"  Transport baseline = {baseline:.6f}"
    )


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis() -> dict[str, Any]:
    """
    Основной pipeline анализа.
    """

    tum_root = find_tum_root()

    discovered_files = discover_tum_files(
        tum_root
    )

    if not discovered_files:
        raise RuntimeError(
            f"No CSV files found under {tum_root}"
        )

    print_header(tum_root)
    print_discovery(
        discovered_files
    )

    grouped = group_by_operation(
        discovered_files
    )

    if BASELINE_OPERATION not in grouped:
        raise RuntimeError(
            f"Baseline operation '{BASELINE_OPERATION}' "
            "was not found in TUM data."
        )

    # ------------------------------------------------------------------
    # STEP 2
    # ------------------------------------------------------------------

    baseline, baseline_stats = (
        calculate_transport_baseline(
            grouped[BASELINE_OPERATION]
        )
    )

    print_baseline(
        baseline,
        baseline_stats,
    )

    # ------------------------------------------------------------------
    # STEP 3
    # ------------------------------------------------------------------

    print()
    print("-" * 80)
    print("STEP 3 - REAL PEAKLOAD BY OPERATION")
    print("-" * 80)

    operation_stats: dict[
        str,
        dict[str, Any],
    ] = {}

    # Transport уже рассчитан отдельно.
    operation_stats[
        BASELINE_OPERATION
    ] = baseline_stats.copy()

    # Добавляем relative_to_transport для Transport.
    operation_stats[
        BASELINE_OPERATION
    ]["relative_to_transport"] = 1.0

    for operation in sorted(
        grouped,
        key=str.lower,
    ):
        if operation == BASELINE_OPERATION:
            continue

        stats, _ = analyze_operation(
            operation,
            grouped[operation],
            baseline,
        )

        if stats is None:
            logger.warning(
                "Skipping operation: %s",
                operation,
            )
            continue

        operation_stats[
            operation
        ] = stats

    # ------------------------------------------------------------------
    # BUILD OUTPUT
    # ------------------------------------------------------------------

    result = build_output(
        tum_root=tum_root,
        discovered_files=discovered_files,
        operation_stats=operation_stats,
        baseline=baseline,
        baseline_stats=baseline_stats,
    )

    save_json(
        result,
        OUTPUT_FILE,
    )

    print_operation_results(
        operation_stats,
        baseline,
    )

    print()
    print("=" * 100)
    print("ANALYSIS COMPLETED")
    print("=" * 100)
    print(
        f"Operations with valid PeakLoad: "
        f"{len(operation_stats)}"
    )
    print(
        f"Transport baseline: "
        f"{baseline:.6f}"
    )
    print(
        f"Output: {OUTPUT_FILE}"
    )
    print("=" * 100)

    return result


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> int:
    configure_logging()

    try:
        run_analysis()
        return 0

    except KeyboardInterrupt:
        logger.error(
            "Analysis interrupted by user."
        )
        return 130

    except FileNotFoundError as exc:
        logger.error(
            "TUM data not found: %s",
            exc,
        )
        return 1

    except Exception as exc:
        logger.error(
            "TUM analysis failed: %s",
            exc,
        )

        # Подробный traceback сохраняем для диагностики,
        # но основной пользовательский вывод остаётся читаемым.
        logger.debug(
            "Full traceback:\n%s",
            traceback.format_exc(),
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )