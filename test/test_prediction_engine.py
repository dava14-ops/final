# -*- coding: utf-8 -*-
"""
tests/test_prediction_engine.py
Characterization-тесты prediction_engine.

Фиксируют поведение ДО рефакторинга constants.py (Фаза B).
Используют self-contained фикстуру minimal_model из conftest.py.
"""
from __future__ import annotations

import pytest

from prediction_engine import (
    transform_peak,
    baseline_cumulative_hazard,
    engine_hours_to_calendar_days,
    calendar_days_to_engine_hours,
    validate_model,
)


# ---------------------------------------------------------------------------
# transform_peak
# ---------------------------------------------------------------------------
class TestTransformPeak:
    def test_standardize_applies_center_and_scale(self, minimal_model):
        # center=0.7099, scale=0.2053 из фикстуры
        peak = 0.9152  # 0.7099 + 0.2053
        result = transform_peak(minimal_model, peak)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_standardize_center_maps_to_zero(self, minimal_model):
        result = transform_peak(minimal_model, 0.7099)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_none_transform_returns_peak_unchanged(self, minimal_model):
        minimal_model.transform_info = {"type": "none"}
        assert transform_peak(minimal_model, 0.5) == pytest.approx(0.5)

    def test_center_transform_subtracts_center(self, minimal_model):
        minimal_model.transform_info = {"type": "center", "center": 0.3}
        assert transform_peak(minimal_model, 0.7) == pytest.approx(0.4)

    def test_missing_transform_info_returns_peak(self, minimal_model):
        minimal_model.transform_info = {}
        assert transform_peak(minimal_model, 0.42) == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# baseline_cumulative_hazard
# ---------------------------------------------------------------------------
class TestBaselineCumulativeHazard:
    def test_h0_at_zero_is_zero(self, minimal_model):
        assert baseline_cumulative_hazard(minimal_model, 0.0) == pytest.approx(0.0)

    def test_h0_is_monotonic_non_decreasing(self, minimal_model):
        times = [0.0, 250.0, 500.0, 750.0, 1000.0, 1712.0]
        values = [baseline_cumulative_hazard(minimal_model, t) for t in times]
        for a, b in zip(values, values[1:]):
            assert b >= a

    def test_h0_matches_knot_value(self, minimal_model):
        # В точке 500.0 baseline = 0.035 (step-function, side=right)
        assert baseline_cumulative_hazard(minimal_model, 500.0) == pytest.approx(0.035)

    def test_h0_between_knots_uses_previous_value(self, minimal_model):
        # Между 500 и 1000 держится значение 0.035
        assert baseline_cumulative_hazard(minimal_model, 750.0) == pytest.approx(0.035)

    def test_h0_at_horizon(self, minimal_model):
        assert baseline_cumulative_hazard(minimal_model, 1712.0) == pytest.approx(0.14)

    def test_vector_input_returns_list(self, minimal_model):
        result = baseline_cumulative_hazard(minimal_model, [0.0, 500.0, 1712.0])
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.14)


# ---------------------------------------------------------------------------
# Конвертация единиц времени
# ---------------------------------------------------------------------------
class TestTimeConversion:
    def test_engine_hours_to_days(self):
        assert engine_hours_to_calendar_days(1712.0, 8.0) == pytest.approx(214.0)

    def test_days_to_engine_hours(self):
        assert calendar_days_to_engine_hours(214.0, 8.0) == pytest.approx(1712.0)

    def test_round_trip(self):
        hours = 1000.0
        days = engine_hours_to_calendar_days(hours, 8.0)
        back = calendar_days_to_engine_hours(days, 8.0)
        assert back == pytest.approx(hours)

    def test_nonpositive_hours_per_day_returns_input(self):
        # При hours_per_day <= 0 функция возвращает вход без конверсии
        assert engine_hours_to_calendar_days(100.0, 0.0) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# validate_model
# ---------------------------------------------------------------------------
class TestValidateModel:
    def test_valid_model_passes(self, minimal_model):
        assert validate_model(minimal_model) is True

    def test_missing_training_meta_fails(self, minimal_model):
        minimal_model.training_meta = None
        with pytest.raises(Exception):
            validate_model(minimal_model)

    def test_negative_residuals_std_fails(self, minimal_model):
        minimal_model.training_residuals_std = -1.0
        with pytest.raises(Exception):
            validate_model(minimal_model)

    def test_none_params_fails(self):
        with pytest.raises(Exception):
            validate_model(None)