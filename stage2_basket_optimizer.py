#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_basket_optimizer.py (v2 - FIXED)

Оптимизация crop basket для Bartik IV.
ИСПРАВЛЕНО: Использование ОКПД2 кодов вместо поиска по названию.
"""

import logging
import sys
from pathlib import Path
from itertools import combinations

import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )


# Маппинг project_crop_code → OKPD2
CROP_TO_OKPD2 = {
    "wheat_total": "01.11.1",
    "barley_total": "01.11.31",
    "maize_grain": "01.11.20",
    "sunflower_grain": "01.11.95",
    "soybean": "01.11.81",
    "rapeseed_total": "01.11.93",
    "oats": "01.11.33",
    "rye": "01.11.32",
    "millet": "01.11.42",
    "buckwheat": "01.11.49.110",
    "peas_dried": "01.11.75",
    "pulses_dried": "01.11.7",
}

# Обратный маппинг
OKPD2_TO_CROP = {v: k for k, v in CROP_TO_OKPD2.items()}


def optimize_basket():
    logger = logging.getLogger(__name__)

    # 16 целевых регионов
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

    # Загрузка данных
    logger.info("Загрузка данных...")
    df_prices = pd.read_csv("rosstat_prices_raw_long_v2.csv", encoding="utf-8-sig")
    df_weights = pd.read_csv("baseline_weights_2016_benchmark.csv", encoding="utf-8-sig")

    # Фильтрация весов по целевым регионам
    df_weights = df_weights[df_weights["region_name_project"].isin(TARGET_REGIONS)]

    # Кандидаты для basket
    candidate_crops = list(CROP_TO_OKPD2.keys())

    # Проверка доступности национальных цен по ОКПД2
    logger.info("Проверка доступности национальных цен по ОКПД2...")
    crop_availability = {}

    for crop_code, okpd2_code in CROP_TO_OKPD2.items():
        # Фильтрация по ОКПД2 и национальному уровню
        crop_data = df_prices[
            (df_prices["okpd2_raw"] == okpd2_code)
            & (df_prices["geography_level"] == "national")
            & (df_prices["value_status"] == "observed")
        ]

        if len(crop_data) > 0:
            years = crop_data["year"].dropna()
            if len(years) > 0:
                first_year = int(years.min())
                last_year = int(years.max())
                # Проверяем полный охват 2010-2026
                has_full_coverage = (first_year <= 2010) and (last_year >= 2026)

                crop_availability[crop_code] = {
                    "okpd2": okpd2_code,
                    "first_year": first_year,
                    "last_year": last_year,
                    "has_full_coverage": has_full_coverage,
                    "n_observations": len(crop_data),
                }

    # Вывод доступности культур
    print("\n" + "=" * 100)
    print("ДОСТУПНОСТЬ НАЦИОНАЛЬНЫХ ЦЕН ДЛЯ КУЛЬТУР-КАНДИДАТОВ (ПО ОКПД2)")
    print("=" * 100)
    print(
        f"\n{'Культура':<25} {'ОКПД2':<20} {'Первый год':<12} {'Последний год':<14} {'Полный охват':<14} {'Наблюдений':<12}"
    )
    print("-" * 100)

    for crop, info in sorted(
        crop_availability.items(), key=lambda x: x[1]["has_full_coverage"], reverse=True
    ):
        full_coverage = "✅ Да" if info["has_full_coverage"] else "❌ Нет"
        print(
            f"{crop:<25} {info['okpd2']:<20} {info['first_year']:<12} {info['last_year']:<14} {full_coverage:<14} {info['n_observations']:<12}"
        )

    # Культуры без данных
    missing_crops = [c for c in candidate_crops if c not in crop_availability]
    if missing_crops:
        print(f"\n⚠️ Культуры без национальных цен: {', '.join(missing_crops)}")

    # Фильтрация культур с полным охватом
    crops_with_full_coverage = [
        crop for crop, info in crop_availability.items() if info["has_full_coverage"]
    ]

    print(f"\n✅ Культуры с полным охватом 2010-2026: {len(crops_with_full_coverage)}")
    if crops_with_full_coverage:
        print(f"   {', '.join(crops_with_full_coverage)}")

    if not crops_with_full_coverage:
        print("\n⚠️ Нет культур с полным охватом 2010-2026.")
        print("   Используем культуры с частичным охватом (2010-2021)...")

        # Альтернатива: культуры с охватом 2010-2021 (до разрыва)
        crops_with_partial_coverage = [
            crop for crop, info in crop_availability.items() if info["first_year"] <= 2010
        ]

        if crops_with_partial_coverage:
            print(f"\n✅ Культуры с охватом с 2010: {len(crops_with_partial_coverage)}")
            print(f"   {', '.join(crops_with_partial_coverage)}")
            crops_with_full_coverage = crops_with_partial_coverage
        else:
            print("\n❌ Невозможно найти культуры с национальными ценами. Выход.")
            return

    # Расчёт покрытия для разных размеров basket
    print("\n" + "=" * 100)
    print("ПОИСК ОПТИМАЛЬНОГО BASKET")
    print("=" * 100)

    best_basket = None
    best_score = 0

    max_basket_size = min(len(crops_with_full_coverage), 8)

    for basket_size in range(4, max_basket_size + 1):
        print(f"\nТестирование basket размера {basket_size}...")

        for combo in combinations(crops_with_full_coverage, basket_size):
            basket = list(combo)

            # Расчёт покрытия для каждого региона
            region_coverages = []
            for region in TARGET_REGIONS:
                region_weights = df_weights[df_weights["region_name_project"] == region]
                coverage = region_weights[region_weights["model_crop"].isin(basket)][
                    "weight_2016_total_sown"
                ].sum()
                region_coverages.append(coverage)

            # Метрика качества basket
            n_regions_ok = sum(1 for c in region_coverages if c >= 0.70)
            avg_coverage = sum(region_coverages) / len(region_coverages) if region_coverages else 0
            min_coverage = min(region_coverages) if region_coverages else 0

            # Score: приоритет количеству регионов с покрытием ≥ 70%
            score = n_regions_ok * 100 + avg_coverage * 10 + min_coverage

            if score > best_score:
                best_score = score
                best_basket = {
                    "crops": basket,
                    "size": basket_size,
                    "n_regions_ok": n_regions_ok,
                    "avg_coverage": avg_coverage,
                    "min_coverage": min_coverage,
                    "region_coverages": dict(zip(TARGET_REGIONS, region_coverages)),
                }

    # Вывод лучшего basket
    print("\n" + "=" * 100)
    print("ОПТИМАЛЬНЫЙ BASKET")
    print("=" * 100)

    if best_basket:
        print(f"\nРазмер basket: {best_basket['size']} культур")
        print(f"Культуры: {', '.join(best_basket['crops'])}")
        print(
            f"\nРегионов с покрытием ≥ 70%: {best_basket['n_regions_ok']} из {len(TARGET_REGIONS)}"
        )
        print(f"Среднее покрытие: {best_basket['avg_coverage']:.1%}")
        print(f"Минимальное покрытие: {best_basket['min_coverage']:.1%}")

        print(f"\nПокрытие по регионам:")
        print(f"{'Регион':<30} {'Покрытие':<12} {'Статус':<10}")
        print("-" * 60)

        for region, coverage in sorted(
            best_basket["region_coverages"].items(), key=lambda x: x[1], reverse=True
        ):
            status = (
                "✅ OK" if coverage >= 0.70 else ("⚠️ Средне" if coverage >= 0.50 else "❌ Низкое")
            )
            print(f"{region:<30} {coverage:<11.1%} {status:<10}")

        # Рекомендация по регионам
        regions_to_exclude = [r for r, c in best_basket["region_coverages"].items() if c < 0.50]

        print("\n" + "=" * 100)
        print("РЕКОМЕНДАЦИЯ")
        print("=" * 100)

        if best_basket["n_regions_ok"] >= 12:
            print(f"\n✅ Оптимальный basket найден!")
            print(f"   - {best_basket['n_regions_ok']} регионов имеют покрытие ≥ 70%")
            print(f"   - Базовый период весов: 2015-2017")
        else:
            print(f"\n⚠️ Оптимальный basket не достигает порога 12 регионов ≥ 70%")
            print(
                f"   Рекомендуется исключить {len(regions_to_exclude)} регионов с покрытием < 50%:"
            )
            for r in regions_to_exclude:
                print(f"   - {r} ({best_basket['region_coverages'][r]:.1%})")

        # Сохранение финальной конфигурации
        config = {
            "basket_crops": best_basket["crops"],
            "basket_size": best_basket["size"],
            "regions_included": [
                r for r, c in best_basket["region_coverages"].items() if c >= 0.50
            ],
            "regions_excluded": regions_to_exclude,
            "baseline_period": "2015-2017",
            "n_regions_ge_70pct": best_basket["n_regions_ok"],
            "avg_coverage": best_basket["avg_coverage"],
        }

        config_df = pd.DataFrame([config])
        config_df.to_csv("final_basket_config.csv", index=False, encoding="utf-8-sig")
        logger.info(f"Конфигурация сохранена: final_basket_config.csv")
    else:
        print("\n❌ Не удалось найти оптимальный basket.")

    print("\n" + "=" * 100)


def main():
    setup_logging()
    optimize_basket()


if __name__ == "__main__":
    main()
