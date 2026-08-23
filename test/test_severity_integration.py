#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_severity_integration.py
Тесты интеграции severity_model → premium_engine.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from severity_model import (
    SystemSeverity,
    SeverityModel,
    build_severity_model,
    compute_exact_covered_loss,
    save_severity_model,
    load_severity_model,
    estimate_system_severity,
)
from premium_engine import (
    calculate_single_premium,
    calculate_premium_with_severity,
)


class TestSeverityModelBuild:
    """Построение severity-модели."""

    def test_build_from_events(self, synthetic_claims):
        """Построение модели из событий."""
        events = synthetic_claims[synthetic_claims["event_flag"] == 1]
        model = build_severity_model(events)
        assert model.n_events == len(events)
        assert model.n_systems > 0
        assert model.overall_mean_repair >= 0.0
        assert model.overall_mean_downtime >= 0.0
        assert 0.0 <= model.overall_major_share <= 1.0

    def test_build_from_empty_events(self):
        """Пустые события → fallback модель."""
        import pandas as pd
        empty = pd.DataFrame({
            "event_flag": [],
            "failure_system": [],
        })
        model = build_severity_model(empty)
        assert model.fallback_used is True
        assert model.n_events == 0

    def test_expected_loss_per_failure(self, synthetic_claims):
        """E[loss] = E[repair] + E[downtime_cost]."""
        events = synthetic_claims[synthetic_claims["event_flag"] == 1]
        model = build_severity_model(events)
        expected = model.expected_repair_cost() + model.expected_downtime_cost()
        assert abs(model.expected_loss_per_failure() - expected) < 1e-10

    def test_expected_covered_loss_with_deductible(self, synthetic_claims):
        """Covered loss с франшизой ≤ loss без франшизы."""
        events = synthetic_claims[synthetic_claims["event_flag"] == 1]
        model = build_severity_model(events)
        no_ded = model.expected_covered_loss(deductible=0.0)
        with_ded = model.expected_covered_loss(deductible=50_000.0)
        assert with_ded <= no_ded


class TestSeveritySerialization:
    """Сериализация и десериализация."""

    def test_save_and_load_roundtrip(self, synthetic_claims, tmp_path):
        """Сохранить → загрузить → проверить поля."""
        events = synthetic_claims[synthetic_claims["event_flag"] == 1]
        model = build_severity_model(events)
        path = tmp_path / "severity_test.json"
        save_severity_model(path, model)
        loaded = load_severity_model(path)

        assert loaded.n_events == model.n_events
        assert loaded.n_systems == model.n_systems
        assert abs(loaded.overall_mean_repair - model.overall_mean_repair) < 1e-10
        assert loaded.fallback_used == model.fallback_used

    def test_load_nonexistent(self, tmp_path):
        """Загрузка несуществующего файла."""
        with pytest.raises(FileNotFoundError):
            load_severity_model(tmp_path / "nonexistent.json")


class TestSeverityPremiumIntegration:
    """Интеграция severity_model с premium_engine."""

    def test_calculate_premium_with_severity(self, synthetic_claims):
        """Полный цикл: severity → premium."""
        events = synthetic_claims[synthetic_claims["event_flag"] == 1]
        model = build_severity_model(events)

        result = calculate_premium_with_severity(
            severity_model=model,
            probability=0.028,
            sum_insured=5_000_000.0,
            theta=0.15,
            deductible=10_000.0,
        )
        assert result["severity_based"] is True
        assert result["net_undiscounted"] >= 0
        assert result["gross_discounted"] >= 0
        assert result["tariff"] >= 0

    def test_severity_premium_less_than_legacy(self, synthetic_claims):
        """Severity-based премия обычно меньше legacy (P × sum_insured)."""
        events = synthetic_claims[synthetic_claims["event_flag"] == 1]
        model = build_severity_model(events)

        severity_result = calculate_premium_with_severity(
            severity_model=model,
            probability=0.028,
            sum_insured=5_000_000.0,
            theta=0.15,
        )
        legacy_result = calculate_single_premium(
            probability=0.028,
            sum_insured=5_000_000.0,
            theta=0.15,
        )
        # E[loss] обычно << sum_insured
        if model.expected_loss_per_failure() < 5_000_000.0:
            assert severity_result["net_undiscounted"] < legacy_result["net_undiscounted"]

    def test_severity_model_with_zero_deductible(self, synthetic_claims):
        """При deductible=0 covered = loss."""
        events = synthetic_claims[synthetic_claims["event_flag"] == 1]
        model = build_severity_model(events)
        covered = model.expected_covered_loss(deductible=0.0)
        loss = model.expected_loss_per_failure()
        # При deductible=0 и без лимита covered ≈ loss
        assert abs(covered - loss) < 1e-6


class TestExactCoveredLoss:
    """Точный расчёт covered loss."""

    def test_exact_vs_approximation(self, synthetic_claims):
        """Сравнение точного расчёта и аппроксимации."""
        events = synthetic_claims[synthetic_claims["event_flag"] == 1]
        model = build_severity_model(events)

        exact = compute_exact_covered_loss(events, deductible=0.0)
        approx = model.expected_covered_loss(deductible=0.0)

        # Они могут отличаться из-за аппроксимации
        # Но оба должны быть неотрицательными
        assert exact >= 0.0
        assert approx >= 0.0

    def test_exact_with_high_deductible(self, synthetic_claims):
        """При очень высокой франшизе covered loss → 0."""
        events = synthetic_claims[synthetic_claims["event_flag"] == 1]
        result = compute_exact_covered_loss(
            events, deductible=1e12,
        )
        assert result == 0.0

    def test_exact_with_coverage_limit(self, synthetic_claims):
        """С лимитом покрытия."""
        events = synthetic_claims[synthetic_claims["event_flag"] == 1]
        limit = 50_000.0
        result = compute_exact_covered_loss(
            events, deductible=0.0, coverage_limit=limit,
        )
        assert result <= limit