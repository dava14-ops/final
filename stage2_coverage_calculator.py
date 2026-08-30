#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_coverage_calculator.py

Рассчитывает покрытие 16 целевых регионов на основе:
- baseline_weights_2016_benchmark.csv (веса из VSHP-2016)
- crop_inventory_full.csv (доступность национальных цен)

Определяет, какие регионы можно включить в анализ.
"""

import logging
import sys
from pathlib import Path

import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )


def calculate_coverage():
    logger = logging.getLogger(__name__)

    # 16 целевых регионов (из предыдущих обсуждений)
    TARGET_REGIONS = [
        "Алтайский край",
        "Амурская область",
        "Белгородская область",
        "Владимирская область",
        "Волгоградская область",
        "Воронежская область",
        "Краснодарский край",
        "Курская область",
        "Ленинградская область",
        "Липецкая область",
        "Псковская область",
        "Ростовская область",
        "Самарская область",
        "Саратовская область",
        "Ставропольский край",
        "Тверская область",
    ]

    # Вариант A: 4 культуры (полное покрытие 2010-2026)
    BASKET_A = ["wheat_total", "barley_total", "maize_grain", "sunflower_grain"]

    # Вариант B: 6 культур (только 2021-2026)
    BASKET_B = [
        "wheat_total",
        "barley_total",
        "maize_grain",
        "sunflower_grain",
        "soybean",
        "rapeseed_total",
    ]

    # Загрузка весов
    weights_path = Path("baseline_weights_2016_benchmark.csv")
    if not weights_path.exists():
        logger.error(f"Файл не найден: {weights_path}")
        sys.exit(1)

    logger.info(f"Загрузка {weights_path}...")
    df_weights = pd.read_csv(weights_path, encoding="utf-8-sig")
    logger.info(f"Загружено {len(df_weights)} записей весов")

    # Фильтрация по целевым регионам
    df_weights = df_weights[df_weights["region_name_project"].isin(TARGET_REGIONS)]
    logger.info(f"Записей для 16 целевых регионов: {len(df_weights)}")

    # Расчёт покрытия для каждого региона и basket
    results = []

    for region in TARGET_REGIONS:
        region_weights = df_weights[df_weights["region_name_project"] == region]
        total_area = region_weights["total"].iloc[0] if len(region_weights) > 0 else 0

        # Вариант A: 4 культуры
        coverage_a = region_weights[region_weights["model_crop"].isin(BASKET_A)][
            "weight_2016_total_sown"
        ].sum()
        crops_a = region_weights[region_weights["model_crop"].isin(BASKET_A)]["model_crop"].tolist()

        # Вариант B: 6 культур
        coverage_b = region_weights[region_weights["model_crop"].isin(BASKET_B)][
            "weight_2016_total_sown"
        ].sum()
        crops_b = region_weights[region_weights["model_crop"].isin(BASKET_B)]["model_crop"].tolist()

        results.append(
            {
                "region": region,
                "total_area_ha": total_area * 1000,  # тыс. га -> га
                "coverage_4crops": coverage_a,
                "coverage_6crops": coverage_b,
                "n_crops_4": len(crops_a),
                "n_crops_6": len(crops_b),
                "crops_4": ", ".join(sorted(set(crops_a))),
                "crops_6": ", ".join(sorted(set(crops_b))),
            }
        )

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("coverage_6crops", ascending=False)

    # Сохранение
    output_path = "region_coverage_analysis.csv"
    df_results.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Результаты сохранены: {output_path}")

    # Вывод таблицы
    print("\n" + "=" * 120)
    print("ПОКРЫТИЕ 16 ЦЕЛЕВЫХ РЕГИОНОВ")
    print("=" * 120)
    print(
        f"\n{'Регион':<25} {'Площадь (га)':>12} {'4 культуры':>12} {'6 культур':>12} {'Статус 4':<10} {'Статус 6':<10}"
    )
    print("-" * 120)

    for _, row in df_results.iterrows():
        status_4 = (
            "✅ OK"
            if row["coverage_4crops"] >= 0.70
            else ("⚠️ Средне" if row["coverage_4crops"] >= 0.50 else "❌ Низкое")
        )
        status_6 = (
            "✅ OK"
            if row["coverage_6crops"] >= 0.70
            else ("⚠️ Средне" if row["coverage_6crops"] >= 0.50 else "❌ Низкое")
        )

        print(
            f"{row['region']:<25} {row['total_area_ha']:>12,.0f} "
            f"{row['coverage_4crops']:>11.1%} {row['coverage_6crops']:>11.1%} "
            f"{status_4:<10} {status_6:<10}"
        )

    # Статистика
    print("\n" + "-" * 120)
    print("СТАТИСТИКА ПОКРЫТИЯ")
    print("-" * 120)

    regions_ok_4 = (df_results["coverage_4crops"] >= 0.70).sum()
    regions_ok_6 = (df_results["coverage_6crops"] >= 0.70).sum()

    print(f"\nВариант A (4 культуры: wheat, barley, maize, sunflower):")
    print(f"  Регионов с покрытием ≥ 70%: {regions_ok_4} из {len(df_results)}")
    print(f"  Среднее покрытие: {df_results['coverage_4crops'].mean():.1%}")
    print(
        f"  Минимум: {df_results['coverage_4crops'].min():.1%} ({df_results.loc[df_results['coverage_4crops'].idxmin(), 'region']})"
    )
    print(
        f"  Максимум: {df_results['coverage_4crops'].max():.1%} ({df_results.loc[df_results['coverage_4crops'].idxmax(), 'region']})"
    )

    print(f"\nВариант B (6 культур: + soybean, rapeseed):")
    print(f"  Регионов с покрытием ≥ 70%: {regions_ok_6} из {len(df_results)}")
    print(f"  Среднее покрытие: {df_results['coverage_6crops'].mean():.1%}")
    print(
        f"  Минимум: {df_results['coverage_6crops'].min():.1%} ({df_results.loc[df_results['coverage_6crops'].idxmin(), 'region']})"
    )
    print(
        f"  Максимум: {df_results['coverage_6crops'].max():.1%} ({df_results.loc[df_results['coverage_6crops'].idxmax(), 'region']})"
    )

    # Рекомендация
    print("\n" + "=" * 120)
    print("РЕКОМЕНДАЦИЯ")
    print("=" * 120)

    if regions_ok_4 >= 12:
        print(f"\n✅ Вариант A (4 культуры) рекомендуется:")
        print(f"   - {regions_ok_4} регионов имеют покрытие ≥ 70%")
        print(f"   - Полное покрытие 2010-2026")
        print(f"   - Базовый период весов: 2015-2017")
        print(f"   - Культуры: wheat_total, barley_total, maize_grain, sunflower_grain")
    else:
        print(f"\n⚠️ Вариант A имеет недостаточное покрытие ({regions_ok_4} регионов ≥ 70%)")
        print(f"   Рассмотрите Вариант B или расширение basket")

    print("\n" + "=" * 120)


def main():
    setup_logging()
    calculate_coverage()


if __name__ == "__main__":
    main()
