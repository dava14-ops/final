#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_soil_index.py
Агрегация ежедневных данных GLDAS в почвенные индексы для кампаний.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("data/raw/soil/gldas_soil_daily.csv")
OUTPUT_DIR = Path("data/processed/soil")
OUTPUT_PATH = OUTPUT_DIR / "soil_windows.csv"


def main():
    print("Чтение gldas_soil_daily.csv...")
    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Определяем кампанию
    def get_campaign(row):
        m, d = row["month"], row["day"]
        if (m == 4 and d >= 15) or (m == 5) or (m == 6 and d <= 15):
            return "sowing"
        if (m == 8) or (m == 9) or (m == 10 and d <= 15):
            return "harvest"
        return np.nan

    df["campaign"] = df.apply(get_campaign, axis=1)
    df = df.dropna(subset=["campaign"])

    # Композитный индекс (верхние слои важнее для техники)
    df["soil_raw"] = (
        0.5 * df["soil_moisture_0_10cm"].fillna(0)
        + 0.3 * df["soil_moisture_10_40cm"].fillna(0)
        + 0.2 * df["soil_moisture_40_100cm"].fillna(0)
    )

    # Среднее по региону, году и кампании
    agg = (
        df.groupby(["region_code", "year", "campaign"])["soil_raw"].mean().reset_index()
    )
    agg.rename(columns={"soil_raw": "soil_index"}, inplace=True)

    # Нормализация в [0, 1]
    vmin, vmax = agg["soil_index"].min(), agg["soil_index"].max()
    if vmax > vmin:
        agg["soil_index_normalized"] = (agg["soil_index"] - vmin) / (vmax - vmin)
    else:
        agg["soil_index_normalized"] = 0.5

    agg["source"] = "gldas_2.1_real"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Сохранено {len(agg)} записей в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
