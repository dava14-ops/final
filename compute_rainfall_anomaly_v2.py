#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ФАЗА 6.6 v2 — Вычисление rainfall anomaly с корректным baseline.

IV-дизайн:
  BASELINE_YEARS = 2015-2021 (историческая норма)
  TARGET_YEARS   = 2022-2025 (целевой период)

Для каждого region × campaign:
  1. Рассчитать baseline_mean и baseline_std по BASELINE_YEARS
  2. Для каждого года из TARGET_YEARS:
     anomaly = (P_campaign - baseline_mean) / baseline_std

Защита от data leakage:
  - BASELINE и TARGET не пересекаются
  - Все BASELINE годы должны присутствовать в данных
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── КОНФИГУРАЦИЯ ПЕРИОДОВ ────────────────────────────────────────
BASELINE_YEARS = list(range(2015, 2022))  # 2015, 2016, ..., 2021
TARGET_YEARS = list(range(2022, 2026))    # 2022, 2023, 2024, 2025

# Проверка отсутствия пересечения
if set(BASELINE_YEARS) & set(TARGET_YEARS):
    raise ValueError(
        f"CRITICAL: Baseline years {BASELINE_YEARS} and target years "
        f"{TARGET_YEARS} overlap! This would cause data leakage."
    )

# ─── КОНФИГУРАЦИЯ КАМПАНИЙ ────────────────────────────────────────
# Примечание: harvest до 31 октября (стандартный аграрный цикл)
CAMPAIGNS = {
    "sowing": {"start": "04-01", "end": "05-31"},
    "harvest": {"start": "08-01", "end": "10-31"},  # Исправлено: до 31 октября
}


def main():
    raw_path = Path("data/raw/weather/nasa_power_daily.csv")
    if not raw_path.exists():
        logger.error(
            "Файл не найден: %s. Сначала запустите load_nasa_power.py "
            "с YEARS = list(range(2015, 2026))",
            raw_path,
        )
        return

    logger.info("Загрузка ежедневных данных из %s", raw_path)
    df = pd.read_csv(raw_path, parse_dates=["date"])

    if "PRECTOTCORR" not in df.columns:
        logger.error("Колонка PRECTOTCORR не найдена в данных.")
        return

    # Проверка наличия всех baseline лет
    available_years = set(df["year"].unique())
    missing_baseline = set(BASELINE_YEARS) - available_years
    if missing_baseline:
        logger.error(
            "CRITICAL: Отсутствуют baseline годы: %s. "
            "Загрузите данные за 2015-2021 через load_nasa_power.py.",
            sorted(missing_baseline),
        )
        return

    missing_target = set(TARGET_YEARS) - available_years
    if missing_target:
        logger.warning(
            "Отсутствуют target годы: %s. Будут обработаны только доступные.",
            sorted(missing_target),
        )

    records = []

    for region in df["region_code"].unique():
        region_df = df[df["region_code"] == region].copy()

        for campaign_name, dates in CAMPAIGNS.items():
            # Фильтр по кампании
            def in_campaign(date_str):
                md = date_str[5:]  # MM-DD
                return dates["start"] <= md <= dates["end"]

            region_df["in_campaign"] = (
                region_df["date"].astype(str).apply(in_campaign)
            )

            # Группируем по году и суммируем осадки за кампанию
            yearly_precip = (
                region_df[region_df["in_campaign"]]
                .groupby("year")["PRECTOTCORR"]
                .sum()
                .reset_index()
            )

            # ─── РАСЧЁТ BASELINE ──────────────────────────────────────
            historical = yearly_precip[yearly_precip["year"].isin(BASELINE_YEARS)]

            if len(historical) < 3:
                logger.warning(
                    "Недостаточно baseline данных для %s / %s "
                    "(всего %d лет из %d нужных)",
                    region,
                    campaign_name,
                    len(historical),
                    len(BASELINE_YEARS),
                )
                continue

            baseline_mean = float(historical["PRECTOTCORR"].mean())
            baseline_std = float(historical["PRECTOTCORR"].std(ddof=1))

            if baseline_std < 1e-6:
                logger.warning(
                    "Baseline std слишком мал для %s / %s: %.6f. "
                    "Используется fallback std=1.0",
                    region,
                    campaign_name,
                    baseline_std,
                )
                baseline_std = 1.0

            # ─── РАСЧЁТ ANOMALY ДЛЯ TARGET YEARS ──────────────────────
            target = yearly_precip[yearly_precip["year"].isin(TARGET_YEARS)]

            for _, row in target.iterrows():
                year = int(row["year"])
                p_val = float(row["PRECTOTCORR"])
                anomaly = (p_val - baseline_mean) / baseline_std

                records.append(
                    {
                        "region_code": region,
                        "year": year,
                        "campaign": campaign_name,
                        "precipitation_mm": p_val,
                        "baseline_mean_mm": baseline_mean,
                        "baseline_std_mm": baseline_std,
                        "baseline_start_year": min(BASELINE_YEARS),
                        "baseline_end_year": max(BASELINE_YEARS),
                        "baseline_n_years": len(historical),
                        "rainfall_anomaly": anomaly,
                    }
                )

    if not records:
        logger.error("Не удалось вычислить аномалии. Проверьте данные.")
        return

    result_df = pd.DataFrame(records)

    # Сохраняем результат
    out_dir = Path("data/processed/weather")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rainfall_anomaly.csv"

    result_df.to_csv(out_path, index=False, encoding="utf-8")

    logger.info("=" * 70)
    logger.info("РАСЧЁТ RAINFALL ANOMALY v2 ЗАВЕРШЁН")
    logger.info("=" * 70)
    logger.info("Baseline: %d-%d (%d лет)", min(BASELINE_YEARS), max(BASELINE_YEARS), len(BASELINE_YEARS))
    logger.info("Target: %d-%d (%d лет)", min(TARGET_YEARS), max(TARGET_YEARS), len(TARGET_YEARS))
    logger.info("Записей: %d", len(result_df))
    logger.info("Регионов: %d", result_df["region_code"].nunique())
    logger.info("Кампаний: %s", list(result_df["campaign"].unique()))
    logger.info("Сохранено в: %s", out_path)
    logger.info("\nПример данных:\n%s", result_df.head(10).to_string(index=False))

    # ─── ДИАГНОСТИКА ────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("ДИАГНОСТИКА ANOMALY")
    logger.info("=" * 70)
    for region in result_df["region_code"].unique():
        region_data = result_df[result_df["region_code"] == region]
        for campaign in region_data["campaign"].unique():
            camp_data = region_data[region_data["campaign"] == campaign]
            anomalies = camp_data["rainfall_anomaly"].values
            logger.info(
                "%s / %s: mean_anomaly=%.3f, std_anomaly=%.3f, n=%d",
                region,
                campaign,
                np.mean(anomalies),
                np.std(anomalies, ddof=1),
                len(anomalies),
            )


if __name__ == "__main__":
    main()