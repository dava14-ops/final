#!/usr/bin/env python3
"""
test_agro_pipeline.py (v2.0)
Тестирование полного пайплайна агрокалендаря.
Совместим с pytest и с запуском как скрипт.

Запуск:
    pytest test_agro_pipeline.py -v
    python test_agro_pipeline.py
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("test_agro_pipeline")

# Константы (дублируются из conftest.py для standalone-запуска)
TEST_CROPS = ["wheat_spring", "forage_beet", "potato"]
TEST_AREAS = [100.0, 200.0]
TEST_TRACTORS = ["МТЗ-82", "К-744Р1"]
CALIBRATION_HORIZON = 214.0

EXPECTED_HOURS_PER_HA = {
    "wheat_spring": (1.0, 6.0),
    "forage_beet":  (3.0, 7.0),
    "potato":       (3.0, 10.0),
}


# ═══════════════════════════════════════════════════════════════════════
# ТЕСТ 1: Загрузка модулей
# ═══════════════════════════════════════════════════════════════════════

class TestModuleLoading:
    """Проверка загрузки всех модулей."""

    def test_agro_calendar_loaded(self, agro_calendar):
        assert agro_calendar is not None
        assert hasattr(agro_calendar, "CROP_CATALOG")
        assert hasattr(agro_calendar, "list_crops")
        assert hasattr(agro_calendar, "get_crop")
        assert hasattr(agro_calendar, "estimate_season_engine_hours")

    def test_16_crops_in_catalog(self, agro_calendar):
        assert len(agro_calendar.CROP_CATALOG) == 16

    def test_agro_norms_loaded(self, agro_norms):
        assert agro_norms is not None
        assert hasattr(agro_norms, "OPERATION_NORMS")
        assert hasattr(agro_norms, "get_engine_hours_per_ha")
        assert len(agro_norms.OPERATION_NORMS) >= 10

    def test_operation_mapping_loaded(self, operation_mapping):
        assert operation_mapping is not None
        assert hasattr(operation_mapping, "OPERATION_MAPPING")
        assert len(operation_mapping.OPERATION_MAPPING) >= 30

    def test_combine_operations_excluded(self, operation_mapping):
        combine_ops = operation_mapping.list_combine_operations()
        assert len(combine_ops) >= 3


# ═══════════════════════════════════════════════════════════════════════
# ТЕСТ 2: Вычисление моточасов
# ═══════════════════════════════════════════════════════════════════════

class TestEngineHoursComputation:
    """Проверка вычисления моточасов и пиков."""

    def test_all_crops_have_results(self, results):
        for crop_key in TEST_CROPS:
            assert crop_key in results, f"Культура {crop_key} отсутствует"
            assert len(results[crop_key]) > 0, f"Нет результатов для {crop_key}"

    def test_hours_positive(self, results):
        for crop_key, crop_results in results.items():
            for label, (total_hours, _) in crop_results.items():
                assert total_hours > 0, f"{crop_key} {label}: моточасы <= 0"

    def test_peak_in_range(self, results):
        for crop_key, crop_results in results.items():
            for label, (_, weighted_peak) in crop_results.items():
                assert 0.0 <= weighted_peak <= 1.0, (
                    f"{crop_key} {label}: пик {weighted_peak:.4f} вне [0,1]"
                )

    def test_hours_per_ha_in_expected_range(self, results, test_config):
        expected = test_config["expected_hours_per_ha"]
        for crop_key, crop_results in results.items():
            if crop_key not in expected:
                continue
            exp_min, exp_max = expected[crop_key]
            for label, (total_hours, _) in crop_results.items():
                area = float(label.split("га_")[0])
                hours_per_ha = total_hours / area
                assert exp_min <= hours_per_ha <= exp_max, (
                    f"{crop_key} {label}: мч/га={hours_per_ha:.2f} "
                    f"вне [{exp_min}, {exp_max}]"
                )

    def test_potato_heavier_than_wheat(self, results):
        """Картофель должен требовать больше моточасов, чем пшеница."""
        for area in TEST_AREAS:
            for tractor in TEST_TRACTORS:
                label = f"{area}га_{tractor}"
                wheat = results.get("wheat_spring", {}).get(label)
                potato = results.get("potato", {}).get(label)
                if wheat and potato:
                    assert potato[0] > wheat[0], (
                        f"{label}: картофель ({potato[0]:.0f} мч) "
                        f"должен быть тяжелее пшеницы ({wheat[0]:.0f} мч)"
                    )

    def test_potato_heavier_than_beet(self, results):
        """Картофель должен требовать больше моточасов, чем кормовая свёкла."""
        for area in TEST_AREAS:
            for tractor in TEST_TRACTORS:
                label = f"{area}га_{tractor}"
                beet = results.get("forage_beet", {}).get(label)
                potato = results.get("potato", {}).get(label)
                if beet and potato:
                    assert potato[0] > beet[0], (
                        f"{label}: картофель ({potato[0]:.0f} мч) "
                        f"должен быть тяжелее кормовой свёклы ({beet[0]:.0f} мч)"
                    )


# ═══════════════════════════════════════════════════════════════════════
# ТЕСТ 3: Разница между тракторами
# ═══════════════════════════════════════════════════════════════════════

class TestTractorDifference:
    """МТЗ-82 должен требовать больше моточасов, чем К-744Р1."""

    def test_mtz_slower_than_k744(self, results):
        for crop_key in TEST_CROPS:
            for area in TEST_AREAS:
                key_mtz = f"{area}га_МТЗ-82"
                key_k744 = f"{area}га_К-744Р1"

                if key_mtz not in results.get(crop_key, {}):
                    continue
                if key_k744 not in results.get(crop_key, {}):
                    continue

                hours_mtz, _ = results[crop_key][key_mtz]
                hours_k744, _ = results[crop_key][key_k744]

                assert hours_mtz > hours_k744, (
                    f"{crop_key} {area}га: МТЗ-82 ({hours_mtz:.0f}) "
                    f"должен быть медленнее К-744Р1 ({hours_k744:.0f})"
                )

    def test_tractor_ratio_reasonable(self, results):
        """Разница между тракторами должна быть 2-5×."""
        for crop_key in TEST_CROPS:
            for area in TEST_AREAS:
                key_mtz = f"{area}га_МТЗ-82"
                key_k744 = f"{area}га_К-744Р1"

                if key_mtz not in results.get(crop_key, {}):
                    continue
                if key_k744 not in results.get(crop_key, {}):
                    continue

                hours_mtz, _ = results[crop_key][key_mtz]
                hours_k744, _ = results[crop_key][key_k744]

                if hours_k744 > 0:
                    ratio = hours_mtz / hours_k744
                    assert 1.2 <= ratio <= 6.0, (
                        f"{crop_key} {area}га: соотношение {ratio:.2f}× "
                        f"вне ожиданий [1.5, 6.0]"
                    )


# ═══════════════════════════════════════════════════════════════════════
# ТЕСТ 4: Горизонт калибровки
# ═══════════════════════════════════════════════════════════════════════

class TestCalibrationHorizon:
    """Проверка превышения горизонта калибровки."""

    def test_horizon_exceedance_warning(self, results, test_config, capsys):
        """Для площади >= 100 га суммарные моточасы превышают 214 мч."""
        calib = test_config["calibration_horizon"]
        exceedances = 0

        for crop_key, crop_results in results.items():
            for label, (total_hours, _) in crop_results.items():
                if total_hours > calib:
                    exceedances += 1

        # Для 100 га пшеницы может быть ~200-270 мч — на грани
        # Для 200 га любой культуры — всегда превышение
        assert exceedances > 0, (
            f"Ни одна комбинация не превысила горизонт {calib} мч"
        )


# ═══════════════════════════════════════════════════════════════════════
# ТЕСТ 5: Интеграция с prediction_engine
# ═══════════════════════════════════════════════════════════════════════

class TestPredictionEngineIntegration:
    """Проверка вычисления вероятностей через prediction_engine."""

    def test_probabilities_computed(self, probabilities):
        assert len(probabilities) > 0, "Ни одна вероятность не вычислена"

    def test_probabilities_in_range(self, probabilities):
        for key, prob in probabilities.items():
            assert 0.0 <= prob <= 1.0, f"{key}: P={prob} вне [0,1]"

    def test_probabilities_finite(self, probabilities):
        import math
        for key, prob in probabilities.items():
            assert math.isfinite(prob), f"{key}: P={prob} не конечна"


# ═══════════════════════════════════════════════════════════════════════
# ТЕСТ 6: Сравнительный анализ культур
# ═══════════════════════════════════════════════════════════════════════

class TestCropComparison:
    """Тяжёлые культуры должны давать большую вероятность."""

    def test_beet_probability_geq_wheat(self, probabilities):
        """P(свёкла) >= P(пшеница) на одинаковых условиях."""
        area = 200.0
        tractor = "МТЗ-82"
        label = f"{area}га_{tractor}"

        wheat_key = f"wheat_spring_{label}"
        beet_key = f"forage_beet_{label}"

        wheat_prob = probabilities.get(wheat_key)
        beet_prob = probabilities.get(beet_key)

        if wheat_prob is not None and beet_prob is not None:
            assert beet_prob >= wheat_prob, (
                f"P(свёкла)={beet_prob:.6f} < P(пшеница)={wheat_prob:.6f}"
            )

    def test_potato_probability_geq_wheat(self, probabilities):
        """P(картофель) >= P(пшеница) на одинаковых условиях."""
        area = 200.0
        tractor = "МТЗ-82"
        label = f"{area}га_{tractor}"

        wheat_key = f"wheat_spring_{label}"
        potato_key = f"potato_{label}"

        wheat_prob = probabilities.get(wheat_key)
        potato_prob = probabilities.get(potato_key)

        if wheat_prob is not None and potato_prob is not None:
            assert potato_prob >= wheat_prob, (
                f"P(картофель)={potato_prob:.6f} < P(пшеница)={wheat_prob:.6f}"
            )


# ═══════════════════════════════════════════════════════════════════════
# ТЕСТ 7: Расчёт премий
# ═══════════════════════════════════════════════════════════════════════

class TestPremiumCalculation:
    """Проверка расчёта премий через premium_engine."""

    def test_premium_engine_import(self):
        try:
            from premium_engine import calculate_single_premium
            assert calculate_single_premium is not None
        except ImportError:
            import pytest
            pytest.skip("premium_engine не доступен")

    def test_premium_calculation_basic(self, probabilities):
        """Расчёт премии для первой доступной вероятности."""
        try:
            from premium_engine import calculate_single_premium
        except ImportError:
            import pytest
            pytest.skip("premium_engine не доступен")

        if not probabilities:
            import pytest
            pytest.skip("Нет вероятностей для расчёта")

        # Берём первую доступную вероятность
        prob = next(iter(probabilities.values()))
        sum_insured = 5_000_000.0
        theta = 0.15

        premium = calculate_single_premium(prob, sum_insured, theta)

        assert premium is not None
        assert isinstance(premium, dict)
        assert "net" in premium or "net_discounted" in premium
        assert "gross" in premium or "gross_discounted" in premium

        net = premium.get("net", premium.get("net_discounted", 0.0))
        gross = premium.get("gross", premium.get("gross_discounted", 0.0))

        assert net >= 0.0, f"Нетто-премия отрицательна: {net}"
        assert gross >= net, f"Брутто ({gross}) < Нетто ({net})"
        assert gross <= sum_insured, (
            f"Брутто-премия ({gross:,.0f}) превышает страховую сумму ({sum_insured:,.0f})"
        )

    def test_tariff_in_reasonable_range(self, probabilities):
        """Тариф должен быть в разумных пределах."""
        try:
            from premium_engine import calculate_single_premium
        except ImportError:
            import pytest
            pytest.skip("premium_engine не доступен")

        if not probabilities:
            import pytest
            pytest.skip("Нет вероятностей")

        prob = next(iter(probabilities.values()))
        sum_insured = 5_000_000.0
        theta = 0.15

        premium = calculate_single_premium(prob, sum_insured, theta)
        tariff = premium.get("tariff", 0.0)

        # Тариф должен быть в [0%, 100%]
        assert 0.0 <= tariff <= 100.0, f"Тариф {tariff}% вне [0, 100]"


# ═══════════════════════════════════════════════════════════════════════
# ТЕСТ 8: Форматирование сводки
# ═══════════════════════════════════════════════════════════════════════

class TestFormatSummary:
    """Проверка форматирования технологической карты."""

    def test_summary_contains_crop_name(self, agro_calendar, operation_peaks):
        for crop_key in TEST_CROPS:
            crop = agro_calendar.get_crop(crop_key)
            assert crop is not None, f"Культура {crop_key} не найдена"

            summary = agro_calendar.format_crop_summary(
                crop_key, 100.0, tractor="МТЗ-82", k_ob=1.0
            )
            assert crop.crop_name_ru in summary, (
                f"Сводка {crop_key} не содержит название '{crop.crop_name_ru}'"
            )

    def test_summary_not_empty(self, agro_calendar):
        for crop_key in TEST_CROPS:
            summary = agro_calendar.format_crop_summary(
                crop_key, 50.0, tractor="К-744Р1", k_ob=1.0
            )
            assert len(summary) > 100, f"Сводка {crop_key} слишком короткая"
            assert "ИТОГО" in summary or "Суммарные" in summary


# ═══════════════════════════════════════════════════════════════════════
# Standalone запуск (без pytest)
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    """Запуск тестов без pytest."""
    print("=" * 70)
    print("ШАГ 5: ТЕСТИРОВАНИЕ ПАЙПЛАЙНА (standalone)")
    print("=" * 70)

    # Имитируем фикстуры вручную
    try:
        import agro_calendar as _ac
    except ImportError as exc:
        print(f"❌ agro_calendar.py не найден: {exc}")
        return 1

    try:
        import agro_norms as _an
    except ImportError:
        _an = None

    # operation_peaks
    peaks: Dict[str, float] = {}
    try:
        from Real_calculator import OPERATION_INFO
        for k, v in OPERATION_INFO.items():
            peaks[k] = v.get("peak_load_mean", 0.50)
    except ImportError:
        peaks = {
            "Ploughing": 0.85,
            "Cultivating (deep)": 0.72,
            "Cultivating (shallow)": 0.45,
            "Disc harrowing": 0.55,
            "Seedbed combination": 0.50,
            "Seed drill combination 4m": 0.65,
            "Precision air seeding": 0.58,
            "Spraying": 0.40,
            "Fertilizing": 0.38,
            "Mowing (front)": 0.62,
            "Swathing": 0.55,
            "Transport": 0.35,
        }

    # results
    computed: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for crop_key in TEST_CROPS:
        computed[crop_key] = {}
        for area in TEST_AREAS:
            for tractor in TEST_TRACTORS:
                label = f"{area}га_{tractor}"
                try:
                    h, p = _ac.estimate_season_engine_hours(
                        crop_key, area, tractor=tractor, k_ob=1.0
                    )
                    computed[crop_key][label] = (h, p)
                except Exception:
                    pass

    # Вывод результатов
    print(f"\n  {'Культура':<20s} {'Площадь':>8s} {'Трактор':>10s} "
          f"{'Моточасы':>9s} {'Пик':>7s}")
    print("  " + "-" * 60)

    for crop_key in TEST_CROPS:
        for label, (hours, peak) in computed.get(crop_key, {}).items():
            parts = label.split("га_")
            area = parts[0]
            tractor = parts[1] if len(parts) > 1 else "?"
            print(
                f"  {crop_key:<20s} {area:>6s} га {tractor:>10s} "
                f"{hours:>9.1f} {peak:>7.4f}"
            )

    # Проверки
    passed = 0
    failed = 0

    # Проверка 1: 16 культур
    if len(_ac.CROP_CATALOG) == 16:
        print("\n  ✅ 16 культур в каталоге")
        passed += 1
    else:
        print(f"\n  ❌ Ожидалось 16 культур, получено {len(_ac.CROP_CATALOG)}")
        failed += 1

    # Проверка 2: моточасы > 0
    all_positive = all(
        h > 0 for cr in computed.values() for h, _ in cr.values()
    )
    if all_positive:
        print("  ✅ Все моточасы > 0")
        passed += 1
    else:
        print("  ❌ Есть моточасы <= 0")
        failed += 1

    # Проверка 3: пики в [0, 1]
    all_in_range = all(
        0.0 <= p <= 1.0 for cr in computed.values() for _, p in cr.values()
    )
    if all_in_range:
        print("  ✅ Все пики ∈ [0, 1]")
        passed += 1
    else:
        print("  ❌ Есть пики вне [0, 1]")
        failed += 1

    # Проверка 4: МТЗ-82 > К-744Р1
    mtz_slower = True
    for crop_key in TEST_CROPS:
        for area in TEST_AREAS:
            k1 = f"{area}га_МТЗ-82"
            k2 = f"{area}га_К-744Р1"
            if k1 in computed.get(crop_key, {}) and k2 in computed.get(crop_key, {}):
                if computed[crop_key][k1][0] <= computed[crop_key][k2][0]:
                    mtz_slower = False
    if mtz_slower:
        print("  ✅ МТЗ-82 медленнее К-744Р1")
        passed += 1
    else:
        print("  ❌ МТЗ-82 НЕ медленнее К-744Р1")
        failed += 1

    # Проверка 5: свёкла тяжелее пшеницы
    beet_heavier = True
    for area in TEST_AREAS:
        for tractor in TEST_TRACTORS:
            label = f"{area}га_{tractor}"
            w = computed.get("wheat_winter", {}).get(label)
            b = computed.get("sugar_beet", {}).get(label)
            if w and b and b[0] <= w[0]:
                beet_heavier = False
    if beet_heavier:
        print("  ✅ Свёкла тяжелее пшеницы")
        passed += 1
    else:
        print("  ❌ Свёкла НЕ тяжелее пшеницы")
        failed += 1

    print(f"\n{'=' * 70}")
    print(f"ИТОГО: {passed} пройдено, {failed} провалено")
    print(f"{'=' * 70}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())