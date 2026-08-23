#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ФАЗА 6.6 — Вычисление rainfall anomaly (аномалии осадков) для IV.

Загружает ежедневные данные NASA POWER и вычисляет аномалию осадков
для посевной (апрель-май) и уборочной (август-октябрь) кампаний.

Z = (P_campaign - mean_P_campaign) / std_P_campaign
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Определение кампаний (месяц-день)
CAMPAIGNS = {
    "sowing": {"start": "04-01", "end": "05-31"},
    "harvest": {"start": "08-01", "end": "10-15"},
}


def main():
    raw_path = Path("data/raw/weather/nasa_power_daily.csv")
    if not raw_path.exists():
        logger.error(
            "Файл не найден: %s. Сначала запустите load_nasa_power.py", raw_path
        )
        return

    logger.info("Загрузка ежедневных данных из %s", raw_path)
    df = pd.read_csv(raw_path, parse_dates=["date"])

    # Убедимся, что PRECTOTCORR (осадки, мм/день) существует
    if "PRECTOTCORR" not in df.columns:
        logger.error("Колонка PRECTOTCORR не найдена в данных.")
        return

    records = []

    for region in df["region_code"].unique():
        region_df = df[df["region_code"] == region].copy()

        for campaign_name, dates in CAMPAIGNS.items():
            # Функция для проверки попадания даты в кампанию
            def in_campaign(date_str):
                # date_str format: YYYY-MM-DD
                md = date_str[5:]  # MM-DD
                return dates["start"] <= md <= dates["end"]

            region_df["in_campaign"] = region_df["date"].astype(str).apply(in_campaign)

            # Группируем по году и суммируем осадки за кампанию
            yearly_precip = (
                region_df[region_df["in_campaign"]]
                .groupby("year")["PRECTOTCORR"]
                .sum()
                .reset_index()
            )

            if len(yearly_precip) < 2:
                logger.warning(
                    "Недостаточно данных для %s / %s (всего %d лет)",
                    region,
                    campaign_name,
                    len(yearly_precip),
                )
                continue

            # Вычисляем mean и std по годам для этого региона и кампании
            mean_p = float(yearly_precip["PRECTOTCORR"].mean())
            std_p = float(yearly_precip["PRECTOTCORR"].std(ddof=1))

            if std_p < 1e-6:
                std_p = 1.0  # Защита от деления на ноль

            # Вычисляем аномалию для каждого года
            for _, row in yearly_precip.iterrows():
                year = int(row["year"])
                p_val = float(row["PRECTOTCORR"])
                anomaly = (p_val - mean_p) / std_p

                records.append(
                    {
                        "region_code": region,
                        "year": year,
                        "campaign": campaign_name,
                        "precipitation_mm": p_val,
                        "baseline_mean_mm": mean_p,
                        "baseline_std_mm": std_p,
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
    logger.info("РАСЧЁТ RAINFALL ANOMALY ЗАВЕРШЁН")
    logger.info("=" * 70)
    logger.info("Записей: %d", len(result_df))
    logger.info("Регионов: %d", result_df["region_code"].nunique())
    logger.info("Кампаний: %s", list(result_df["campaign"].unique()))
    logger.info("Сохранено в: %s", out_path)
    logger.info("\nПример данных:\n%s", result_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
