# -*- coding: utf-8 -*-
"""
tests/test_constants.py
Инварианты единого реестра констант (constants.py).

Эти тесты фиксируют контракт constants.py ДО и ПОСЛЕ любого рефакторинга.
Self-contained: не требуют внешних данных, только импорта constants.
"""
from __future__ import annotations

import math

import pytest

from constants import (
    BRAND_MAP,
    BRAND_TO_CODE,
    BRAND_ALIASES,
    DEFAULT_BRAND_PROB_BY_CODE,
    FREQ_SHARES,
    SEVERITY_WEIGHTS,
    CRITICALITY_WEIGHTS,
    VALID_EVENT_DEFINITIONS,
    SEGMENTS,
    MODEL_TIME_UNIT,
    DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY,
    CALIBRATION_HORIZON_DAYS,
    CALIBRATION_HORIZON_ENGINE_HOURS,
    MAJOR_FAILURE_SHARE,
    DEFAULT_WEIBULL_SHAPE,
    MTBF_BASELINE_HOURS,
    RF_HEAVY_BRAND_CATALOG,
    POWER_SEGMENT_THRESHOLDS,
    CLIMATE_INDEX_REFERENCE,
    SOIL_INDEX_REFERENCE,
)


# ---------------------------------------------------------------------------
# Brand mappings
# ---------------------------------------------------------------------------
class TestBrandConstants:
    def test_brand_map_is_non_empty(self):
        assert len(BRAND_MAP) > 0

    def test_brand_map_codes_are_unique(self):
        codes = list(BRAND_MAP.keys())
        assert len(codes) == len(set(codes))

    def test_brand_map_names_are_unique(self):
        names = list(BRAND_MAP.values())
        assert len(names) == len(set(names))

    def test_brand_to_code_is_inverse_of_brand_map(self):
        """BRAND_TO_CODE — точная инверсия BRAND_MAP."""
        for code, name in BRAND_MAP.items():
            assert BRAND_TO_CODE[name] == code
        for name, code in BRAND_TO_CODE.items():
            assert BRAND_MAP[code] == name

    def test_brand_aliases_point_to_valid_codes(self):
        """Каждый алиас указывает на валидный код из BRAND_MAP."""
        valid_codes = set(BRAND_MAP.keys())
        for alias, code in BRAND_ALIASES.items():
            assert code in valid_codes, (
                f"BRAND_ALIASES[{alias!r}] = {code} не в BRAND_MAP"
            )

    def test_brand_aliases_canonical_names_are_aliases(self):
        """Канонические имена из BRAND_MAP должны быть в BRAND_ALIASES."""
        for code, name in BRAND_MAP.items():
            assert name.lower() in BRAND_ALIASES or name in BRAND_ALIASES

    def test_default_brand_prob_sums_to_one(self):
        """Вероятности брендов должны суммироваться в 1.0."""
        total = sum(DEFAULT_BRAND_PROB_BY_CODE.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_default_brand_prob_keys_match_brand_map(self):
        """Ключи вероятностей должны совпадать с кодами BRAND_MAP."""
        assert set(DEFAULT_BRAND_PROB_BY_CODE.keys()) == set(BRAND_MAP.keys())

    def test_default_brand_prob_values_non_negative(self):
        for code, p in DEFAULT_BRAND_PROB_BY_CODE.items():
            assert 0.0 <= p <= 1.0, (
                f"DEFAULT_BRAND_PROB_BY_CODE[{code}] = {p} вне [0, 1]"
            )


# ---------------------------------------------------------------------------
# Failure frequency / severity / criticality
# ---------------------------------------------------------------------------
class TestFailureShares:
    def test_freq_shares_sum_to_one(self):
        total = sum(FREQ_SHARES.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_freq_shares_non_negative(self):
        for system, share in FREQ_SHARES.items():
            assert 0.0 <= share <= 1.0, (
                f"FREQ_SHARES[{system!r}] = {share} вне [0, 1]"
            )

    def test_freq_shares_keys_match_severity_weights(self):
        """Ключи частот и тяжести должны совпадать (одни и те же системы)."""
        assert set(FREQ_SHARES.keys()) == set(SEVERITY_WEIGHTS.keys())

    def test_freq_shares_keys_match_criticality_weights(self):
        assert set(FREQ_SHARES.keys()) == set(CRITICALITY_WEIGHTS.keys())


class TestSeverityAndCriticality:
    def test_severity_weights_sum_to_one(self):
        total = sum(SEVERITY_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_severity_weights_non_negative(self):
        for system, w in SEVERITY_WEIGHTS.items():
            assert w >= 0.0, (
                f"SEVERITY_WEIGHTS[{system!r}] = {w} отрицательный"
            )

    def test_criticality_weights_positive(self):
        for system, w in CRITICALITY_WEIGHTS.items():
            assert w > 0.0, (
                f"CRITICALITY_WEIGHTS[{system!r}] = {w} неположительный"
            )


# ---------------------------------------------------------------------------
# Event definitions / segments
# ---------------------------------------------------------------------------
class TestEventAndSegmentConstants:
    def test_valid_event_definitions_non_empty(self):
        assert len(VALID_EVENT_DEFINITIONS) > 0

    def test_valid_event_definitions_contains_required(self):
        required = {"total_loss", "major_claim", "any_failure"}
        assert required.issubset(VALID_EVENT_DEFINITIONS)

    def test_valid_event_definitions_is_frozenset(self):
        assert isinstance(VALID_EVENT_DEFINITIONS, frozenset)

    def test_segments_contains_light_and_heavy(self):
        assert "light" in SEGMENTS
        assert "heavy" in SEGMENTS

    def test_segments_non_empty(self):
        assert len(SEGMENTS) > 0


# ---------------------------------------------------------------------------
# Time conventions
# ---------------------------------------------------------------------------
class TestTimeConventions:
    def test_model_time_unit_is_engine_hours(self):
        assert MODEL_TIME_UNIT == "engine_hours"

    def test_engine_hours_per_day_positive(self):
        assert DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY > 0.0

    def test_calibration_horizon_days_positive(self):
        assert CALIBRATION_HORIZON_DAYS > 0.0

    def test_calibration_horizon_engine_hours_consistent(self):
        """Горизонт в мч = дни × мч/день."""
        expected = (
            CALIBRATION_HORIZON_DAYS * DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY
        )
        assert CALIBRATION_HORIZON_ENGINE_HOURS == pytest.approx(expected)

    def test_calibration_horizon_engine_hours_is_1712(self):
        assert CALIBRATION_HORIZON_ENGINE_HOURS == pytest.approx(1712.0)


# ---------------------------------------------------------------------------
# Expert assumptions
# ---------------------------------------------------------------------------
class TestExpertAssumptions:
    def test_major_failure_share_in_unit_interval(self):
        assert 0.0 < MAJOR_FAILURE_SHARE < 1.0

    def test_weibull_shape_positive(self):
        assert DEFAULT_WEIBULL_SHAPE > 0.0

    def test_mtbf_baseline_positive(self):
        assert MTBF_BASELINE_HOURS > 0.0


# ---------------------------------------------------------------------------
# RF heavy brand catalog
# ---------------------------------------------------------------------------
class TestRfHeavyCatalog:
    def test_rf_heavy_catalog_non_empty(self):
        assert len(RF_HEAVY_BRAND_CATALOG) > 0

    def test_rf_heavy_catalog_power_in_range(self):
        """Все мощности в каталоге — в допустимом диапазоне [50, 500]."""
        for brand, info in RF_HEAVY_BRAND_CATALOG.items():
            power = info["power_hp"]
            assert 50.0 <= power <= 500.0, (
                f"RF_HEAVY_BRAND_CATALOG[{brand!r}].power_hp = {power} "
                f"вне [50, 500]"
            )

    def test_rf_heavy_catalog_shares_sum_approximately_one(self):
        total = sum(info["share"] for info in RF_HEAVY_BRAND_CATALOG.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_rf_heavy_catalog_shares_non_negative(self):
        for brand, info in RF_HEAVY_BRAND_CATALOG.items():
            assert info["share"] >= 0.0


# ---------------------------------------------------------------------------
# Power segments
# ---------------------------------------------------------------------------
class TestPowerSegments:
    def test_power_segments_cover_required(self):
        assert "light" in POWER_SEGMENT_THRESHOLDS
        assert "heavy" in POWER_SEGMENT_THRESHOLDS

    def test_power_segment_lower_lt_upper(self):
        for segment, (lower, upper) in POWER_SEGMENT_THRESHOLDS.items():
            assert lower < upper, (
                f"POWER_SEGMENT_THRESHOLDS[{segment!r}]: "
                f"lower={lower} >= upper={upper}"
            )

    def test_power_segment_bounds_non_negative(self):
        for segment, (lower, upper) in POWER_SEGMENT_THRESHOLDS.items():
            assert lower >= 0.0
            assert upper > 0.0


# ---------------------------------------------------------------------------
# Climate / Soil references
# ---------------------------------------------------------------------------
class TestClimateAndSoilReferences:
    def test_climate_index_in_unit_interval(self):
        for label, value in CLIMATE_INDEX_REFERENCE.items():
            assert 0.0 <= value <= 1.0, (
                f"CLIMATE_INDEX_REFERENCE[{label!r}] = {value} вне [0, 1]"
            )

    def test_soil_index_in_unit_interval(self):
        for label, value in SOIL_INDEX_REFERENCE.items():
            assert 0.0 <= value <= 1.0, (
                f"SOIL_INDEX_REFERENCE[{label!r}] = {value} вне [0, 1]"
            )

    def test_climate_index_has_extremes(self):
        """Должны быть точки 0.0 и 1.0 для полной шкалы."""
        values = set(CLIMATE_INDEX_REFERENCE.values())
        assert 0.0 in values
        assert 1.0 in values

    def test_soil_index_has_extremes(self):
        values = set(SOIL_INDEX_REFERENCE.values())
        assert 0.0 in values
        assert 1.0 in values


# ---------------------------------------------------------------------------
# Global consistency
# ---------------------------------------------------------------------------
class TestGlobalConsistency:
    def test_no_nan_or_inf_in_constants(self):
        """Ни одна числовая константа не должна быть NaN или Inf."""
        checked = {
            "MAJOR_FAILURE_SHARE": MAJOR_FAILURE_SHARE,
            "DEFAULT_WEIBULL_SHAPE": DEFAULT_WEIBULL_SHAPE,
            "MTBF_BASELINE_HOURS": MTBF_BASELINE_HOURS,
            "DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY": (
                DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY
            ),
            "CALIBRATION_HORIZON_DAYS": CALIBRATION_HORIZON_DAYS,
            "CALIBRATION_HORIZON_ENGINE_HOURS": (
                CALIBRATION_HORIZON_ENGINE_HOURS
            ),
        }
        for name, value in checked.items():
            assert math.isfinite(value), f"{name} = {value} не конечное"

    def test_all_share_dictionaries_sum_to_one(self):
        """Все словари долей суммируются в 1.0 (глобальный инвариант)."""
        for name, mapping in [
            ("DEFAULT_BRAND_PROB_BY_CODE", DEFAULT_BRAND_PROB_BY_CODE),
            ("FREQ_SHARES", FREQ_SHARES),
            ("SEVERITY_WEIGHTS", SEVERITY_WEIGHTS),
        ]:
            total = sum(mapping.values())
            assert total == pytest.approx(1.0, abs=1e-9), (
                f"{name}: сумма = {total}, ожидается 1.0"
            )