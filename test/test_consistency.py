#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_consistency.py
Проверка согласованности между модулями проекта.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pytest

from constants import (
    BRAND_MAP,
    BRAND_TO_CODE,
    CALIBRATION_HORIZON_DAYS,
    CALIBRATION_HORIZON_ENGINE_HOURS,
    DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY,
    FREQ_SHARES,
    MAJOR_FAILURE_SHARE,
    MODEL_TIME_UNIT,
    MTBF_BASELINE_HOURS,
    MTTR_HOURS,
    POWER_SEGMENT_THRESHOLDS,
    SEVERITY_WEIGHTS,
    CRITICALITY_WEIGHTS,
    VALID_EVENT_DEFINITIONS,
    SEGMENTS,
)


# ===========================================================================
# 1. Согласованность constants.py
# ===========================================================================
class TestConstantsConsistency:
    """Внутренняя согласованность constants.py."""

    def test_calibration_horizon_arithmetic(self):
        """CALIBRATION_HORIZON = DAYS × HOURS_PER_DAY."""
        expected = CALIBRATION_HORIZON_DAYS * DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY
        assert abs(CALIBRATION_HORIZON_ENGINE_HOURS - expected) < 1e-10

    def test_calibration_horizon_value(self):
        """Конкретное значение: 214 × 8 = 1712."""
        assert CALIBRATION_HORIZON_ENGINE_HOURS == 1712.0

    def test_model_time_unit(self):
        """Единица времени модели."""
        assert MODEL_TIME_UNIT == "engine_hours"

    def test_freq_shares_sum_to_one(self):
        """FREQ_SHARES должны суммироваться в 1.0."""
        total = sum(FREQ_SHARES.values())
        assert abs(total - 1.0) < 1e-10

    def test_freq_shares_non_negative(self):
        """Все доли частот должны быть неотрицательными."""
        for key, val in FREQ_SHARES.items():
            assert val >= 0.0, f"FREQ_SHARES[{key}] = {val} < 0"

    def test_severity_weights_non_negative(self):
        """Все веса severity должны быть неотрицательными."""
        for key, val in SEVERITY_WEIGHTS.items():
            assert val >= 0.0, f"SEVERITY_WEIGHTS[{key}] = {val} < 0"

    def test_criticality_weights_non_negative(self):
        """Все веса критичности должны быть неотрицательными."""
        for key, val in CRITICALITY_WEIGHTS.items():
            assert val >= 0.0, f"CRITICALITY_WEIGHTS[{key}] = {val} < 0"

    def test_freq_and_severity_same_systems(self):
        """FREQ_SHARES и SEVERITY_WEIGHTS должны содержать одни и те же системы."""
        assert set(FREQ_SHARES.keys()) == set(SEVERITY_WEIGHTS.keys())

    def test_criticality_same_systems(self):
        """CRITICALITY_WEIGHTS должны содержать те же системы."""
        assert set(CRITICALITY_WEIGHTS.keys()) == set(FREQ_SHARES.keys())

    def test_major_failure_share_in_range(self):
        """MAJOR_FAILURE_SHARE в [0, 1]."""
        assert 0.0 <= MAJOR_FAILURE_SHARE <= 1.0

    def test_mtbf_positive(self):
        """MTBF должен быть положительным."""
        assert MTBF_BASELINE_HOURS > 0.0

    def test_mttr_positive(self):
        """MTTR должен быть положительным."""
        for key, val in MTTR_HOURS.items():
            assert val > 0.0, f"MTTR_HOURS[{key}] = {val} <= 0"

    def test_valid_event_definitions(self):
        """Определения событий должны содержать expected set."""
        expected = {"total_loss", "major_claim", "any_failure"}
        assert set(VALID_EVENT_DEFINITIONS) == expected

    def test_segments(self):
        """SEGMENTS должны быть light и heavy."""
        assert SEGMENTS == ("light", "heavy")

    def test_power_segment_thresholds_coverage(self):
        """POWER_SEGMENT_THRESHOLDS должны покрывать весь диапазон."""
        light_lo, light_hi = POWER_SEGMENT_THRESHOLDS["light"]
        medium_lo, medium_hi = POWER_SEGMENT_THRESHOLDS["medium"]
        heavy_lo, heavy_hi = POWER_SEGMENT_THRESHOLDS["heavy"]

        assert light_lo == 0.0
        assert light_hi == medium_lo
        assert medium_hi == heavy_lo
        assert heavy_hi > heavy_lo


# ===========================================================================
# 2. Согласованность BRAND_MAP и BRAND_TO_CODE
# ===========================================================================
class TestBrandConsistency:
    """Согласованность маппингов брендов."""

    def test_brand_map_and_to_code_inverse(self):
        """BRAND_TO_CODE должен быть инверсией BRAND_MAP."""
        for code, name in BRAND_MAP.items():
            assert name in BRAND_TO_CODE
            assert BRAND_TO_CODE[name] == code

    def test_brand_codes_range(self):
        """Коды брендов должны быть в [0, 4]."""
        for code in BRAND_MAP.keys():
            assert 0 <= code <= 4

    def test_brand_names_unique(self):
        """Имена брендов должны быть уникальными."""
        names = list(BRAND_MAP.values())
        assert len(names) == len(set(names))

    def test_five_brands(self):
        """Должно быть ровно 5 брендов."""
        assert len(BRAND_MAP) == 5


# ===========================================================================
# 3. Согласованность prediction_engine и constants
# ===========================================================================
class TestPredictionEngineConsistency:
    """Согласованность prediction_engine с constants.py."""

    def test_get_freq_shares_matches_constants(self, minimal_model):
        """get_freq_shares() должен возвращать FREQ_SHARES из constants."""
        from prediction_engine import get_freq_shares
        result = get_freq_shares(minimal_model)
        for key in FREQ_SHARES:
            assert key in result
            assert abs(result[key] - FREQ_SHARES[key]) < 1e-10

    def test_get_severity_weights_matches_constants(self, minimal_model):
        """get_severity_weights() должен возвращать SEVERITY_WEIGHTS."""
        from prediction_engine import get_severity_weights
        result = get_severity_weights(minimal_model)
        for key in SEVERITY_WEIGHTS:
            assert key in result
            assert abs(result[key] - SEVERITY_WEIGHTS[key]) < 1e-10

    def test_get_mtbf_matches_constants(self, minimal_model):
        """get_mtbf_baseline_hours() должен возвращать MTBF_BASELINE_HOURS."""
        from prediction_engine import get_mtbf_baseline_hours
        result = get_mtbf_baseline_hours(minimal_model)
        assert abs(result - MTBF_BASELINE_HOURS) < 1e-10

    def test_get_downtime_hours(self, minimal_model):
        """get_downtime_hours() = MTTR × factor."""
        from prediction_engine import get_downtime_hours
        result = get_downtime_hours(minimal_model, 48.0)
        assert result == 48.0  # factor = 1.0 по умолчанию


# ===========================================================================
# 4. Согласованность X_STANDARDIZATION
# ===========================================================================
class TestXStandardizationConsistency:
    """Согласованность X_STANDARDIZATION между модулями."""

    def test_itog_and_engine_x_standardization(self):
        """X_STANDARDIZATION в Итог.py и prediction_engine.py должны совпадать."""
        from Итог import X_STANDARDIZATION as itog_std
        from prediction_engine import X_STANDARDIZATION_FALLBACK as engine_std

        common_keys = set(itog_std.keys()) & set(engine_std.keys())
        for key in common_keys:
            itog_info = itog_std[key]
            engine_info = engine_std[key]
            # shift и scale должны совпадать
            itog_shift = itog_info.get("shift")
            engine_shift = engine_info.get("shift")
            itog_scale = itog_info.get("scale")
            engine_scale = engine_info.get("scale")

            if itog_shift is not None and engine_shift is not None:
                assert abs(itog_shift - engine_shift) < 1e-10, (
                    f"x_standardization[{key}].shift: "
                    f"Итог={itog_shift}, engine={engine_shift}"
                )
            if itog_scale is not None and engine_scale is not None:
                assert abs(itog_scale - engine_scale) < 1e-10, (
                    f"x_standardization[{key}].scale: "
                    f"Итог={itog_scale}, engine={engine_scale}"
                )

    def test_x_hours_shift_is_1000(self):
        """P-02: x_hours shift/scale = 1000."""
        from Итог import X_STANDARDIZATION as itog_std
        info = itog_std.get("x_hours", {})
        assert info.get("shift") == 1000.0
        assert info.get("scale") == 1000.0


# ===========================================================================
# 5. Согласованность premium_engine
# ===========================================================================
class TestPremiumEngineConsistency:
    """Внутренняя согласованность premium_engine."""

    def test_discount_factor_range(self):
        """Discount factor должен быть в (0, 1]."""
        from premium_engine import calculate_single_premium
        result = calculate_single_premium(
            probability=0.05,
            sum_insured=1_000_000.0,
            theta=0.15,
            discount_rate=0.10,
            policy_horizon_days=365.0,
        )
        assert 0.0 < result["discount_factor"] <= 1.0

    def test_no_discount_gives_factor_one(self):
        """Без дисконтирования factor = 1.0."""
        from premium_engine import calculate_single_premium
        result = calculate_single_premium(
            probability=0.05,
            sum_insured=1_000_000.0,
            theta=0.15,
            discount_rate=0.0,
        )
        assert result["discount_factor"] == 1.0

    def test_tariff_definition(self):
        """tariff = gross / sum_insured × 100."""
        from premium_engine import calculate_single_premium
        result = calculate_single_premium(
            probability=0.028,
            sum_insured=5_000_000.0,
            theta=0.15,
        )
        expected_tariff = result["gross_discounted"] / 5_000_000.0 * 100.0
        assert abs(result["tariff"] - expected_tariff) < 1e-10


# ===========================================================================
# 6. Согласованность severity_model и premium_engine
# ===========================================================================
class TestSeverityPremiumConsistency:
    """Согласованность между severity_model и premium_engine."""

    def test_severity_model_interface(self, synthetic_claims):
        """SeverityModel.expected_covered_loss() совместим с premium_engine."""
        from severity_model import build_severity_model
        from premium_engine import calculate_premium_with_severity

        events = synthetic_claims[synthetic_claims["event_flag"] == 1]
        model = build_severity_model(events)

        result = calculate_premium_with_severity(
            severity_model=model,
            probability=0.028,
            sum_insured=5_000_000.0,
            theta=0.15,
        )
        assert result["severity_based"] is True
        assert result["net_undiscounted"] >= 0
        assert result["gross_discounted"] >= 0

    def test_exact_covered_loss_non_negative(self, synthetic_claims):
        """Точный covered loss должен быть неотрицательным."""
        from severity_model import compute_exact_covered_loss

        events = synthetic_claims[synthetic_claims["event_flag"] == 1]
        result = compute_exact_covered_loss(events, deductible=10_000.0)
        assert result >= 0.0

    def test_exact_covered_loss_with_limit(self, synthetic_claims):
        """Covered loss с лимитом не должен превышать лимит."""
        from severity_model import compute_exact_covered_loss

        events = synthetic_claims[synthetic_claims["event_flag"] == 1]
        limit = 100_000.0
        result = compute_exact_covered_loss(
            events, deductible=0.0, coverage_limit=limit,
        )
        assert result <= limit


# ===========================================================================
# 7. Согласованность model_registry
# ===========================================================================
class TestModelRegistryConsistency:
    """Согласованность model_registry."""

    def test_filename_convention_matches_architecture(self):
        """Конвенция именования должна соответствовать architecture.md."""
        from model_registry import generate_model_filename, parse_model_filename

        # v0.x = simulation, v1.x = real claims
        sim_name = generate_model_filename("0.2", "light", "20250815")
        real_name = generate_model_filename("1.0", "heavy", "20250901")

        sim_info = parse_model_filename(sim_name)
        real_info = parse_model_filename(real_name)

        assert sim_info is not None and sim_info.major == 0
        assert real_info is not None and real_info.major == 1

    def test_all_segments_valid(self):
        """Все сегменты из конвенции должны парситься."""
        from model_registry import generate_model_filename, parse_model_filename

        for segment in ("light", "heavy"):
            name = generate_model_filename("1.0", segment, "20250101")
            info = parse_model_filename(name)
            assert info is not None
            assert info.segment == segment


# ===========================================================================
# 8. Согласованность claims_validator и constants
# ===========================================================================
class TestClaimsValidatorConsistency:
    """Согласованность claims_validator с constants.py."""

    def test_valid_failure_systems_match_freq_shares(self):
        """VALID_FAILURE_SYSTEMS должны совпадать с ключами FREQ_SHARES."""
        from claims_validator import VALID_FAILURE_SYSTEMS
        assert VALID_FAILURE_SYSTEMS == set(FREQ_SHARES.keys())

    def test_valid_brand_names_match_brand_map(self):
        """VALID_BRAND_NAMES должны совпадать с значениями BRAND_MAP."""
        from claims_validator import VALID_BRAND_NAMES
        assert VALID_BRAND_NAMES == set(BRAND_MAP.values())