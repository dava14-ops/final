#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_crop_audit.py

Stage 2.1: Crop Inventory Analysis

Анализирует rosstat_prices_raw_long_v2.csv и создает полный инвентарь культур:
- Временной охват (first_year, last_year)
- Доступность национальных/региональных цен
- Количество наблюдаемых значений vs пропусков
- Маппинг OKPD2 → project_crop_code для релевантных культур

Использование:
    python stage2_crop_audit.py
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )


# ---------------------------------------------------------------------------
# Маппинг OKPD2 → project_crop_code
# ---------------------------------------------------------------------------

# Культуры, релевантные для тракторных работ (на основе baseline_weights_2016_benchmark.csv)
RELEVANT_CROPS = {
    "01.11.1": "wheat_total",  # Пшеница (агрегированная)
    "01.11.11": "wheat_durum",  # Пшеница твердая
    "01.11.12": "wheat_soft",  # Пшеница мягкая (агрегированная)
    "01.11.12.004.АГ": "wheat_soft_1class",  # Пшеница мягкая 1 класса
    "01.11.12.005.АГ": "wheat_soft_2class",  # Пшеница мягкая 2 класса
    "01.11.12.006.АГ": "wheat_soft_3class",  # Пшеница мягкая 3 класса
    "01.11.12.007.АГ": "wheat_soft_4class",  # Пшеница мягкая 4 класса
    "01.11.12.008.АГ": "wheat_soft_5class",  # Пшеница мягкая 5 класса
    "01.11.20": "maize_grain",  # Кукуруза на зерно
    "01.11.31": "barley_total",  # Ячмень
    "01.11.32": "rye",  # Рожь
    "01.11.33": "oats",  # Овес
    "01.11.42": "millet",  # Просо
    "01.11.49.110": "buckwheat",  # Гречиха
    "01.11.49.120": "triticale",  # Тритикале
    "01.11.7": "pulses_dried",  # Овощи бобовые сушеные
    "01.11.75": "peas_dried",  # Горох сушеный
    "01.11.81": "soybean",  # Бобы соевые
    "01.11.91": "flax_seed",  # Семена льна
    "01.11.92": "mustard_seed",  # Семена горчицы
    "01.11.93": "rapeseed_total",  # Семена рапса
    "01.11.95": "sunflower_grain",  # Семена подсолнечника
    "01.12.10": "rice_unhusked",  # Рис нешелушеный
}

# Агрегированные категории (могут дублировать детальные)
AGGREGATE_CATEGORIES = {
    "01.11.12.001.АГ": "grains_pulses_total",  # Зерновые и зернобобовые культуры
    "01.11.12.003.АГ": "grains_crops",  # Культуры зерновые
}

# ---------------------------------------------------------------------------
# Основной анализ
# ---------------------------------------------------------------------------


def analyze_crop_inventory(input_csv: str, output_csv: str):
    """
    Построить полный инвентарь культур из rosstat_prices_raw_long_v2.csv
    """
    logger = logging.getLogger(__name__)

    # Загрузка данных
    logger.info(f"Загрузка {input_csv}...")
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    logger.info(f"Загружено {len(df):,} записей")

    # Фильтрация: только строки с OKPD2
    df_with_okpd = df[df["okpd2_raw"].notna() & (df["okpd2_raw"] != "")].copy()
    logger.info(
        f"Записей с OKPD2: {len(df_with_okpd):,} ({100 * len(df_with_okpd) / len(df):.1f}%)"
    )

    # Группировка по OKPD2 и названию
    inventory = []

    for (okpd2, product_name), group in df_with_okpd.groupby(["okpd2_raw", "product_name_raw"]):
        # Временной охват
        years = group["year"].dropna()
        if len(years) > 0:
            first_year = int(years.min())
            last_year = int(years.max())
        else:
            first_year = None
            last_year = None

        # Типы данных
        has_monthly = (group["frequency"] == "monthly").any()
        has_annual = (group["frequency"] == "annual").any()

        # Уровни географии
        has_national = (group["geography_level"] == "national").any()
        has_regional = (group["geography_level"].isin(["subject", "federal_district"])).any()

        # Статусы значений
        n_observed = (group["value_status"] == "observed").sum()
        n_confidential = (group["value_status"] == "confidential").sum()
        n_unavailable = (group["value_status"] == "unavailable").sum()
        n_structurally_missing = (group["value_status"] == "structurally_missing").sum()

        # Project crop code
        project_crop = RELEVANT_CROPS.get(okpd2, AGGREGATE_CATEGORIES.get(okpd2, None))

        inventory.append(
            {
                "okpd2_raw": okpd2,
                "product_name_raw": product_name,
                "project_crop_code": project_crop,
                "first_year": first_year,
                "last_year": last_year,
                "has_monthly": has_monthly,
                "has_annual": has_annual,
                "has_national": has_national,
                "has_regional": has_regional,
                "n_observations": len(group),
                "n_observed": n_observed,
                "n_confidential": n_confidential,
                "n_unavailable": n_unavailable,
                "n_structurally_missing": n_structurally_missing,
                "pct_observed": 100 * n_observed / len(group) if len(group) > 0 else 0,
            }
        )

    df_inventory = pd.DataFrame(inventory)
    df_inventory = df_inventory.sort_values("okpd2_raw")

    # Сохранение
    df_inventory.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Инвентарь сохранен: {output_csv} ({len(df_inventory)} культур)")

    # Вывод сводки
    print("\n" + "=" * 80)
    print("СВОДКА ПО КУЛЬТУРАМ")
    print("=" * 80)

    print(f"\nВсего уникальных OKPD2: {len(df_inventory)}")
    print(f"С project_crop_code: {df_inventory['project_crop_code'].notna().sum()}")
    print(f"Без project_crop_code: {df_inventory['project_crop_code'].isna().sum()}")

    # Релевантные культуры для тракторных работ
    print("\n" + "-" * 80)
    print("РЕЛЕВАНТНЫЕ КУЛЬТУРЫ ДЛЯ ТРАКТОРНЫХ РАБОТ")
    print("-" * 80)

    relevant_df = df_inventory[df_inventory["project_crop_code"].notna()].copy()
    relevant_df = relevant_df.sort_values("project_crop_code")

    print(
        f"\n{'OKPD2':<20} {'Project Code':<25} {'Years':<12} {'Monthly':<8} {'Annual':<8} {'National':<9} {'Regional':<9} {'Observed%':<10}"
    )
    print("-" * 110)

    for _, row in relevant_df.iterrows():
        years_str = (
            f"{int(row['first_year'])}-{int(row['last_year'])}"
            if pd.notna(row["first_year"])
            else "N/A"
        )
        print(
            f"{row['okpd2_raw']:<20} {row['project_crop_code']:<25} {years_str:<12} "
            f"{'✓' if row['has_monthly'] else '✗':<8} "
            f"{'✓' if row['has_annual'] else '✗':<8} "
            f"{'✓' if row['has_national'] else '✗':<9} "
            f"{'✓' if row['has_regional'] else '✗':<9} "
            f"{row['pct_observed']:<10.1f}"
        )

    # Культуры для Bartik IV (основной basket)
    print("\n" + "-" * 80)
    print("КУЛЬТУРЫ ДЛЯ BARTIK IV (ОСНОВНОЙ BASKET)")
    print("-" * 80)

    bartik_crops = [
        "wheat_total",
        "barley_total",
        "maize_grain",
        "sunflower_grain",
        "soybean",
        "rapeseed_total",
    ]

    bartik_df = relevant_df[relevant_df["project_crop_code"].isin(bartik_crops)].copy()

    print(
        f"\n{'Project Code':<25} {'OKPD2':<20} {'National':<9} {'Regional':<9} {'Years':<15} {'Observed%':<10}"
    )
    print("-" * 100)

    for _, row in bartik_df.iterrows():
        years_str = (
            f"{int(row['first_year'])}-{int(row['last_year'])}"
            if pd.notna(row["first_year"])
            else "N/A"
        )
        print(
            f"{row['project_crop_code']:<25} {row['okpd2_raw']:<20} "
            f"{'✓' if row['has_national'] else '✗':<9} "
            f"{'✓' if row['has_regional'] else '✗':<9} "
            f"{years_str:<15} "
            f"{row['pct_observed']:<10.1f}"
        )

    # Проверка: все ли культуры Bartik имеют национальные цены?
    missing_national = bartik_df[~bartik_df["has_national"]]["project_crop_code"].tolist()
    if missing_national:
        print(f"\n⚠️ ВНИМАНИЕ: Следующие культуры Bartik НЕ имеют национальных цен:")
        for crop in missing_national:
            print(f"   - {crop}")
        print("\n   Это критическая проблема для Dataset A (основной инструмент)!")
    else:
        print(f"\n✅ Все {len(bartik_crops)} культур Bartik имеют национальные цены.")

    # Агрегированные категории (потенциальные дубликаты)
    print("\n" + "-" * 80)
    print("АГРЕГИРОВАННЫЕ КАТЕГОРИИ (ПОТЕНЦИАЛЬНЫЕ ДУБЛИКАТЫ)")
    print("-" * 80)

    aggregate_df = df_inventory[df_inventory["okpd2_raw"].isin(AGGREGATE_CATEGORIES.keys())].copy()

    if len(aggregate_df) > 0:
        print(f"\n{'OKPD2':<25} {'Project Code':<30} {'Product Name':<50}")
        print("-" * 110)

        for _, row in aggregate_df.iterrows():
            name = (
                row["product_name_raw"][:47] + "..."
                if len(row["product_name_raw"]) > 50
                else row["product_name_raw"]
            )
            print(f"{row['okpd2_raw']:<25} {str(row['project_crop_code']):<30} {name:<50}")

        print("\n⚠️ Эти категории могут дублировать детальные культуры (wheat, barley и т.д.).")
        print("   Рекомендуется использовать детальные коды для Bartik IV.")

    # Статистика по статусам пропусков
    print("\n" + "-" * 80)
    print("СТАТИСТИКА ПРОПУСКОВ (ВСЕ КУЛЬТУРЫ)")
    print("-" * 80)

    total_observed = df_inventory["n_observed"].sum()
    total_confidential = df_inventory["n_confidential"].sum()
    total_unavailable = df_inventory["n_unavailable"].sum()
    total_structurally_missing = df_inventory["n_structurally_missing"].sum()
    total_records = df_inventory["n_observations"].sum()

    print(f"\nВсего записей: {total_records:,}")
    print(
        f"  Observed (наблюдаемые):              {total_observed:>10,} ({100 * total_observed / total_records:.1f}%)"
    )
    print(
        f"  Confidential (конфиденциальные):     {total_confidential:>10,} ({100 * total_confidential / total_records:.1f}%)"
    )
    print(
        f"  Unavailable (отсутствуют):           {total_unavailable:>10,} ({100 * total_unavailable / total_records:.1f}%)"
    )
    print(
        f"  Structurally missing:                {total_structurally_missing:>10,} ({100 * total_structurally_missing / total_records:.1f}%)"
    )

    print("\n" + "=" * 80)
    print("АУДИТ ЗАВЕРШЕН")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    setup_logging()

    input_csv = "rosstat_prices_raw_long_v2.csv"
    output_csv = "crop_inventory_full.csv"

    if not Path(input_csv).exists():
        logging.error(f"Файл не найден: {input_csv}")
        sys.exit(1)

    analyze_crop_inventory(input_csv, output_csv)


if __name__ == "__main__":
    main()
