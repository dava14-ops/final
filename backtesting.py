#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtesting.py
Фаза 7, Итерация 3: Бэктестинг и калибровка модели v1.0.
Задачи:
7.7: Temporal split (обучение/тест по году выпуска)
7.8: Калибровка по децилям (O/E ratio, Brier Score)

Использование:
    python backtesting.py

Вход:
    data/processed/claims/claims_clean.csv
    model_params_v1.json (или последняя версия из model_registry)

Выход:
    data/processed/claims/backtest_report.json
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
    BRAND_TO_CODE,
    MODEL_TIME_UNIT,
    CALIBRATION_HORIZON_ENGINE_HOURS,
)
from prediction_engine import (
    load_model_params,
    predict_probability,
)
from model_registry import get_latest_model_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
CLAIMS_PATH = Path("data/processed/claims/claims_clean.csv")
OUTPUT_DIR = Path("data/processed/claims")
OUTPUT_REPORT_PATH = OUTPUT_DIR / "backtest_report.json"


def _resolve_model_path() -> Path:
    """Определить путь к модели: сначала registry, потом fallback."""
    latest = get_latest_model_path()
    if latest is not None and latest.exists():
        logger.info("Используется последняя модель из registry: %s", latest)
        return latest

    fallback = Path("model_params_v1.json")
    if fallback.exists():
        logger.warning("Registry пуст, используется fallback: %s", fallback)
        return fallback

    raise FileNotFoundError(f"Модель не найдена. Проверены: {latest}, {fallback}")


def _determine_split_year(df: pd.DataFrame) -> int:
    """
    Определить год разделения автоматически на основе данных.
    Использует медианный год, чтобы получить примерно 50/50 split.
    """
    if "production_year" not in df.columns:
        logger.warning("production_year отсутствует, используется 2020")
        return 2020

    years = df["production_year"].dropna()
    if len(years) == 0:
        logger.warning("production_year пуст, используется 2020")
        return 2020

    median_year = int(years.median())
    logger.info(
        "Автоматический temporal split: медианный год = %d",
        median_year,
    )
    return median_year


# ---------------------------------------------------------------------------
# Константы бэктеста
# ---------------------------------------------------------------------------
N_DECILES = 10


def load_data() -> Tuple[pd.DataFrame, Any, Path]:
    """Загрузить claims и модель."""
    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(f"Claims не найдены: {CLAIMS_PATH}")

    df = pd.read_csv(CLAIMS_PATH, encoding="utf-8")
    logger.info("Загружено claims: %d строк", len(df))

    model_path = _resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Модель v1.0 не найдена: {model_path}")

    model = load_model_params(model_path)
    logger.info("Модель v1.0 загружена из %s", model_path)

    return df, model, model_path


def temporal_split(
    df: pd.DataFrame,
    split_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разделить данные на train/test по году выпуска."""
    if "production_year" not in df.columns:
        logger.warning("production_year отсутствует. Случайный сплит 70/30.")
        mask = np.random.rand(len(df)) < 0.7
        return df[mask], df[~mask]

    train_mask = df["production_year"] < split_year
    test_mask = ~train_mask

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    logger.info(
        "Temporal split (%d): train=%d, test=%d",
        split_year,
        len(df_train),
        len(df_test),
    )

    return df_train, df_test


def build_covariates(row: pd.Series) -> Dict[str, float]:
    """Собрать словарь ковариат для prediction_engine из строки claims."""
    covs: Dict[str, float] = {}

    # PeakLoad proxy
    if "peak_load_proxy" in row.index and pd.notna(row["peak_load_proxy"]):
        covs["PeakLoad"] = float(row["peak_load_proxy"])
    else:
        covs["PeakLoad"] = 0.71  # Fallback на среднее TUM

    # Непрерывные ковариаты
    if "age_at_event" in row.index:
        covs["x_age"] = float(row["age_at_event"])
    if "hours_at_event" in row.index:
        covs["x_hours"] = float(row["hours_at_event"])
    if "power_hp" in row.index:
        covs["x_power"] = float(row["power_hp"])

    # ★ НОВОЕ: climate и soil (Фаза 6.6)
    if "climate" in row.index and pd.notna(row["climate"]):
        covs["x_climate"] = float(row["climate"])
    if "soil" in row.index and pd.notna(row["soil"]):
        covs["x_soil"] = float(row["soil"])

    # Brand dummies (согласовано с retrain_v1.py)
    brand_name = str(row.get("brand", "Other"))
    for code, name in BRAND_MAP.items():
        if int(code) == 0:
            continue  # MTZ82 — референсная категория
        covs[f"brand_{name}"] = 1.0 if brand_name == name else 0.0

    return covs


def predict_probabilities(
        model: Any,
        df_test: pd.DataFrame,
) -> np.ndarray:
    """Рассчитать вероятности отказа для тестовой выборки."""
    probs = []
    horizon = CALIBRATION_HORIZON_ENGINE_HOURS

    for idx, row in df_test.iterrows():
        covs = build_covariates(row)
        try:
            p = predict_probability(
                params=model,
                raw_peak=covs.get("PeakLoad", 0.71),
                time_horizon=horizon,
                residual_policy="plug-in",
                covariates=covs,
                time_horizon_unit=MODEL_TIME_UNIT,
                strict_covariates=False,
            )
            probs.append(float(p))
        except Exception as exc:
            logger.warning(
                "Ошибка предсказания для строки %s: %s. P=0.0", idx, exc
            )
            probs.append(0.0)

    return np.array(probs, dtype=float)


def compute_calibration_table(
        probs: np.ndarray,
        events: np.ndarray,
        n_bins: int = N_DECILES,
) -> pd.DataFrame:
    """Построить калибровочную таблицу (децили предсказаний vs факт)."""
    df = pd.DataFrame({"prob": probs, "event": events})

    # Бины по квантилям предсказанной вероятности
    try:
        df["decile"] = pd.qcut(df["prob"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        # Если уникальных значений мало, qcut может упасть
        df["decile"] = pd.cut(df["prob"], bins=n_bins, labels=False, include_lowest=True)

    table = (
        df.groupby("decile")
        .agg(
            n_obs=("prob", "count"),
            n_events=("event", "sum"),
            mean_pred_prob=("prob", "mean"),
            min_pred_prob=("prob", "min"),
            max_pred_prob=("prob", "max"),
        )
        .reset_index()
    )

    table["actual_freq"] = table["n_events"] / table["n_obs"]
    table["OE_ratio"] = table["actual_freq"] / table["mean_pred_prob"].replace(0, np.nan)

    return table


def compute_metrics(probs: np.ndarray, events: np.ndarray) -> Dict[str, float]:
    """Рассчитать метрики качества бинарной классификации."""
    # Brier Score: mean((p - y)^2)
    brier_score = float(np.mean((probs - events) ** 2))

    # Calibration-in-the-large: mean(p) vs mean(y)
    mean_pred = float(np.mean(probs))
    mean_obs = float(np.mean(events))
    citl = mean_pred - mean_obs

    # Null model Brier score (предсказываем среднюю частоту)
    null_brier = float(np.mean((mean_obs - events) ** 2))
    brier_skill_score = 1.0 - (brier_score / null_brier) if null_brier > 0 else 0.0

    return {
        "brier_score": brier_score,
        "null_brier_score": null_brier,
        "brier_skill_score": brier_skill_score,
        "calibration_in_the_large": citl,
        "mean_predicted_probability": mean_pred,
        "mean_observed_frequency": mean_obs,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("=" * 70)
    print("Фаза 7, Итерация 3: Бэктестинг и калибровка v1.0")
    print("=" * 70)

    try:
        df, model, model_path = load_data()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    # 1. Определение года разделения
    split_year = _determine_split_year(df)

    # 2. Temporal split
    df_train, df_test = temporal_split(df, split_year)

    if len(df_test) == 0:
        logger.error("Тестовая выборка пуста. Бэктест невозможен.")
        return 1

    # 3. Предсказание на тесте
    print("\nРасчёт вероятностей на тестовой выборке...")
    probs = predict_probabilities(model, df_test)
    events = df_test["event_flag"].astype(int).to_numpy()

    # 4. Метрики
    metrics = compute_metrics(probs, events)

    # 5. Калибровочная таблица
    cal_table = compute_calibration_table(probs, events, N_DECILES)

    # Вывод результатов
    print()
    print("-" * 70)
    print("МЕТРИКИ КАЧЕСТВА (Out-of-Sample)")
    print("-" * 70)
    print(f"  Модель:                 {model_path.name}")
    print(f"  Temporal split:         production_year < {split_year}")
    print(f"  Тестовая выборка:       {len(df_test)} машин")
    print(f"  Фактических событий:    {int(events.sum())}")
    print(f"  Средняя предсказ. P:    {metrics['mean_predicted_probability']:.4f}")
    print(f"  Фактическая частота:    {metrics['mean_observed_frequency']:.4f}")
    print(f"  Calibration-in-large:   {metrics['calibration_in_the_large']:+.4f}")
    print(f"  Brier Score:            {metrics['brier_score']:.6f}")
    print(f"  Null Brier Score:       {metrics['null_brier_score']:.6f}")
    print(f"  Brier Skill Score:      {metrics['brier_skill_score']:.4f}")

    print()
    print("-" * 70)
    print(f"КАЛИБРОВОЧНАЯ ТАБЛИЦА ({N_DECILES} децилей)")
    print("-" * 70)
    print(
        f"{'Бин':>4} | {'N':>5} | {'Событий':>7} | {'Предсказ. P':>11} | "
        f"{'Факт. част.':>11} | {'O/E ratio':>9}"
    )
    print("-" * 70)

    for _, row in cal_table.iterrows():
        oe = f"{row['OE_ratio']:.3f}" if pd.notna(row["OE_ratio"]) else "n/a"
        print(
            f"{int(row['decile']):>4} | {int(row['n_obs']):>5} | "
            f"{int(row['n_events']):>7} | {row['mean_pred_prob']:>11.4f} | "
            f"{row['actual_freq']:>11.4f} | {oe:>9}"
        )

    # Сохранение отчёта
    report = {
        "model_version": getattr(model, "model_version", "1.0"),
        "model_path": str(model_path),
        "split_method": f"production_year < {split_year}",
        "split_year": split_year,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "n_test_events": int(events.sum()),
        "metrics": metrics,
        "calibration_table": cal_table.to_dict(orient="records"),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print()
    print(f"Отчёт сохранён: {OUTPUT_REPORT_PATH}")
    print("=" * 70)
    print("Итерация 3 Фазы 7 завершена. Фаза 7 ЗАКРЫТА.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())