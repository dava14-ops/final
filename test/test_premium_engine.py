# -*- coding: utf-8 -*-
"""
tests/test_premium_engine.py
Characterization-тесты контракта calculate_single_premium.

Фиксируют поведение ДО рефакторинга constants.py (Фаза B),
чтобы миграция импортов не внесла регрессию.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from premium_engine import calculate_single_premium, calculate_premium_with_severity
from exceptions import InvalidInputError


# ---------------------------------------------------------------------------
# Базовый расчёт (legacy: P * sum_insured)
# ---------------------------------------------------------------------------
class TestBasicPremium:
    def test_net_equals_probability_times_sum(self):
        result = calculate_single_premium(0.028, 5_000_000.0, theta=0.15)
        assert result["net_undiscounted"] == pytest.approx(0.028 * 5_000_000.0)
        assert result["net"] == pytest.approx(0.028 * 5_000_000.0)

    def test_gross_applies_theta_loading(self):
        result = calculate_single_premium(0.028, 5_000_000.0, theta=0.15)
        expected_gross = 0.028 * 5_000_000.0 * 1.15
        assert result["gross_undiscounted"] == pytest.approx(expected_gross)
        assert result["gross"] == pytest.approx(expected_gross)

    def test_tariff_is_gross_over_sum_insured_percent(self):
        result = calculate_single_premium(0.028, 5_000_000.0, theta=0.15)
        expected_tariff = 0.028 * 1.15 * 100.0
        assert result["tariff"] == pytest.approx(expected_tariff)

    def test_loading_amount_is_gross_minus_net(self):
        result = calculate_single_premium(0.028, 5_000_000.0, theta=0.15)
        assert result["loading_amount"] == pytest.approx(
            result["gross"] - result["net"]
        )

    def test_zero_theta_means_gross_equals_net(self):
        result = calculate_single_premium(0.028, 5_000_000.0, theta=0.0)
        assert result["gross"] == pytest.approx(result["net"])
        assert result["loading_amount"] == pytest.approx(0.0)

    def test_zero_probability_gives_zero_premium(self):
        result = calculate_single_premium(0.0, 5_000_000.0, theta=0.15)
        assert result["net"] == pytest.approx(0.0)
        assert result["gross"] == pytest.approx(0.0)
        assert result["tariff"] == pytest.approx(0.0)

    def test_full_probability_gives_full_sum(self):
        result = calculate_single_premium(1.0, 5_000_000.0, theta=0.0)
        assert result["net"] == pytest.approx(5_000_000.0)


# ---------------------------------------------------------------------------
# Валидация входов
# ---------------------------------------------------------------------------
class TestInputValidation:
    def test_probability_above_one_raises(self):
        with pytest.raises(InvalidInputError):
            calculate_single_premium(1.5, 5_000_000.0, theta=0.15)

    def test_negative_probability_raises(self):
        with pytest.raises(InvalidInputError):
            calculate_single_premium(-0.1, 5_000_000.0, theta=0.15)

    def test_nan_probability_raises(self):
        with pytest.raises(InvalidInputError):
            calculate_single_premium(float("nan"), 5_000_000.0, theta=0.15)

    def test_zero_sum_insured_raises(self):
        with pytest.raises(InvalidInputError):
            calculate_single_premium(0.028, 0.0, theta=0.15)

    def test_negative_sum_insured_raises(self):
        with pytest.raises(InvalidInputError):
            calculate_single_premium(0.028, -100.0, theta=0.15)

    def test_negative_theta_raises(self):
        with pytest.raises(InvalidInputError):
            calculate_single_premium(0.028, 5_000_000.0, theta=-0.1)

    def test_discount_rate_at_one_raises(self):
        with pytest.raises(InvalidInputError):
            calculate_single_premium(
                0.028, 5_000_000.0, theta=0.15, discount_rate=1.0
            )

    def test_negative_discount_rate_raises(self):
        with pytest.raises(InvalidInputError):
            calculate_single_premium(
                0.028, 5_000_000.0, theta=0.15, discount_rate=-0.05
            )


# ---------------------------------------------------------------------------
# Дисконтирование
# ---------------------------------------------------------------------------
class TestDiscounting:
    def test_zero_rate_no_discount(self):
        result = calculate_single_premium(
            0.028, 5_000_000.0, theta=0.0, discount_rate=0.0,
            policy_horizon_days=214.0,
        )
        assert result["discount_factor"] == pytest.approx(1.0)
        assert result["net_discounted"] == pytest.approx(result["net_undiscounted"])

    def test_positive_rate_reduces_net(self):
        no_disc = calculate_single_premium(
            0.028, 5_000_000.0, theta=0.0, discount_rate=0.0,
            policy_horizon_days=214.0,
        )
        with_disc = calculate_single_premium(
            0.028, 5_000_000.0, theta=0.0, discount_rate=0.10,
            policy_horizon_days=214.0,
        )
        assert with_disc["net_discounted"] < no_disc["net_discounted"]
        assert with_disc["discount_factor"] < 1.0

    def test_discount_factor_matches_continuous_formula(self):
        rate = 0.10
        days = 214.0
        result = calculate_single_premium(
            0.028, 5_000_000.0, theta=0.0, discount_rate=rate,
            policy_horizon_days=days,
        )
        expected_factor = math.exp(-rate * days / 365.0)
        assert result["discount_factor"] == pytest.approx(expected_factor)


# ---------------------------------------------------------------------------
# Severity-based режим (P-12)
# ---------------------------------------------------------------------------
class TestSeverityBased:
    def test_severity_flag_set(self):
        result = calculate_single_premium(
            0.028, 5_000_000.0, theta=0.15, expected_severity=262_110.0,
        )
        assert result["severity_based"] is True

    def test_net_uses_severity_not_sum_insured(self):
        severity = 262_110.0
        result = calculate_single_premium(
            0.028, 5_000_000.0, theta=0.0, expected_severity=severity,
        )
        assert result["net"] == pytest.approx(0.028 * severity)

    def test_deductible_reduces_covered_loss(self):
        severity = 262_110.0
        deductible = 10_000.0
        result = calculate_single_premium(
            0.028, 5_000_000.0, theta=0.0,
            expected_severity=severity, deductible=deductible,
        )
        expected_net = 0.028 * (severity - deductible)
        assert result["net"] == pytest.approx(expected_net)

    def test_deductible_exceeding_severity_gives_zero(self):
        result = calculate_single_premium(
            0.028, 5_000_000.0, theta=0.0,
            expected_severity=5_000.0, deductible=10_000.0,
        )
        assert result["net"] == pytest.approx(0.0)

    def test_coverage_limit_caps_loss(self):
        severity = 262_110.0
        limit = 100_000.0
        result = calculate_single_premium(
            0.028, 5_000_000.0, theta=0.0,
            expected_severity=severity, coverage_limit=limit,
        )
        assert result["net"] == pytest.approx(0.028 * limit)


# ---------------------------------------------------------------------------
# Severity integration tests (calculate_premium_with_severity)
# ---------------------------------------------------------------------------
class TestSeverityIntegration:
    def _make_severity_model(self, expected_loss: float):
        """Create a mock severity model with expected_covered_loss method."""
        model = MagicMock()
        model.expected_covered_loss.return_value = expected_loss
        return model

    def test_basic_integration(self):
        severity = self._make_severity_model(expected_loss=250_000.0)
        result = calculate_premium_with_severity(
            severity_model=severity,
            probability=0.028,
            sum_insured=5_000_000.0,
            theta=0.15,
            deductible=10_000.0,
            coverage_limit=500_000.0,
            policy_horizon_days=214.0,
        )
        # Verify expected_covered_loss was called with correct args
        severity.expected_covered_loss.assert_called_once_with(
            deductible=10_000.0,
            coverage_limit=500_000.0,
        )
        # Net should use severity, not sum_insured
        assert result["severity_based"] is True
        assert result["net"] == pytest.approx(
            0.028 * max(0.0, 250_000.0 - 10_000.0)
        )

    def test_calls_expected_covered_loss(self):
        severity = self._make_severity_model(expected_loss=100_000.0)
        calculate_premium_with_severity(
            severity_model=severity,
            probability=0.01,
            sum_insured=1_000_000.0,
        )
        severity.expected_covered_loss.assert_called_once()
        call_args = severity.expected_covered_loss.call_args
        assert call_args.kwargs["deductible"] == 0.0
        assert call_args.kwargs["coverage_limit"] is None

    def test_without_deductible_or_limit(self):
        severity = self._make_severity_model(expected_loss=150_000.0)
        result = calculate_premium_with_severity(
            severity_model=severity,
            probability=0.05,
            sum_insured=2_000_000.0,
        )
        assert result["severity_based"] is True
        assert result["net"] == pytest.approx(0.05 * 150_000.0)

    def test_non_finite_severity_raises(self):
        severity = MagicMock()
        severity.expected_covered_loss.return_value = float("inf")
        with pytest.raises(Exception):
            calculate_premium_with_severity(
                severity_model=severity,
                probability=0.01,
                sum_insured=1_000_000.0,
            )

    def test_missing_expected_covered_loss_raises(self):
        # Object without expected_covered_loss method
        bad_model = object()
        with pytest.raises(InvalidInputError):
            calculate_premium_with_severity(
                severity_model=bad_model,
                probability=0.01,
                sum_insured=1_000_000.0,
            )

    def test_discounting_applied(self):
        severity = self._make_severity_model(expected_loss=200_000.0)
        with_disc = calculate_premium_with_severity(
            severity_model=severity,
            probability=0.02,
            sum_insured=5_000_000.0,
            discount_rate=0.10,
            policy_horizon_days=214.0,
        )
        without_disc = calculate_premium_with_severity(
            severity_model=severity,
            probability=0.02,
            sum_insured=5_000_000.0,
            discount_rate=0.0,
            policy_horizon_days=214.0,
        )
        assert with_disc["net"] < without_disc["net"]
        assert with_disc["discount_factor"] == pytest.approx(
            math.exp(-0.10 * 214.0 / 365.0)
        )

    def test_theta_loading_applied(self):
        severity = self._make_severity_model(expected_loss=300_000.0)
        result = calculate_premium_with_severity(
            severity_model=severity,
            probability=0.01,
            sum_insured=1_000_000.0,
            theta=0.20,
        )
        expected_gross = result["net"] * 1.20
        assert result["gross"] == pytest.approx(expected_gross)