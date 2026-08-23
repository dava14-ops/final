#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_integration.py
Интеграционные тесты: end-to-end потоки данных через модули проекта.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from constants import (
    BRAND_MAP,
    BRAND_TO_CODE,
    CALIBRATION_HORIZON_ENGINE_HOURS,
    MODEL_TIME_UNIT,
)
from prediction_engine import (
    ModelParameters,
    load_model_params,
    save_model_params,
    validate_model,
    predict_probability,
    predict_many,
    baseline_cumulative_hazard,
    transform_peak,
    predict_first_stage,
    compute_pl_hat_exog,
    engine_hours_to_calendar_days,
    calendar_days_to_engine_hours,
    get_freq_shares,
    get_severity_weights,
    get_mtbf_baseline_hours,
    get_downtime_hours,
    classify_power_segment,
    kaplan_meier_check,
    coerce_brand_code,
)
from premium_engine import (
    calculate_single_premium,
    calculate_premium,
)
from model_registry import (
    parse_model_filename,
    generate_model_filename,
    list_model_versions,
)
from recalibration_triggers import (
    check_loss_ratio_deviation,
    check_calibration_error,
    check_major_share_change,
    check_f_statistic,
    check_instrument_drift,
    check_all_triggers,
    format_trigger_report,
    save_trigger_report,
)


# ===========================================================================
# 1. Модель: save → load → validate
# ===========================================================================
class TestModelSaveLoadValidate:
    """Цикл сохранения, загрузки и валидации модели."""

    def test_save_and_load_roundtrip(self, minimal_model, tmp_path):
        """Сохранить → загрузить → проверить поля."""
        path = tmp_path / "test_model.json"
        save_model_params(path, minimal_model)
        loaded = load_model_params(path, validate=True)

        assert loaded.model_version == minimal_model.model_version
        assert loaded.calibration_time_horizon == minimal_model.calibration_time_horizon
        assert loaded.event_definition == minimal_model.event_definition
        assert loaded.segment == minimal_model.segment

    def test_validate_model_passes(self, minimal_model):
        """Валидация минимальной модели должна пройти."""
        assert validate_model(minimal_model) is True

    def test_validate_model_rejects_empty_cox(self, minimal_model):
        """Модель с пустым cox должна быть отклонена."""
        minimal_model.cox = {}
        with pytest.raises(Exception):
            validate_model(minimal_model)

    def test_validate_model_rejects_negative_baseline(self, minimal_model):
        """Baseline с отрицательными значениями должна быть отклонена."""
        minimal_model.baseline_cumulative_hazard["values"][3] = -1.0
        with pytest.raises(Exception):
            validate_model(minimal_model)

    def test_load_nonexistent_file(self, tmp_path):
        """Загрузка несуществующего файла должна бросить ошибку."""
        with pytest.raises(Exception):
            load_model_params(tmp_path / "nonexistent.json")

    def test_load_malformed_json(self, tmp_path):
        """Загрузка некорректного JSON должна бросить ошибку."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json}", encoding="utf-8")
        with pytest.raises(Exception):
            load_model_params(bad_file)


# ===========================================================================
# 2. Предсказание: end-to-end
# ===========================================================================
class TestPredictionEndToEnd:
    """Полный цикл предсказания."""

    def test_predict_probability_basic(self, loaded_model):
        """Базовое предсказание вероятности."""
        prob = predict_probability(
            loaded_model,
            raw_peak=0.71,
            time_horizon=1712.0,
            residual_policy="plug-in",
            time_horizon_unit="engine_hours",
        )
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
        assert math.isfinite(prob)

    def test_predict_probability_monotonic_in_peak(self, loaded_model):
        """Вероятность должна расти с PeakLoad (при положительном γ)."""
        prob_low = predict_probability(
            loaded_model, raw_peak=0.3, time_horizon=1712.0,
            time_horizon_unit="engine_hours",
        )
        prob_high = predict_probability(
            loaded_model, raw_peak=1.5, time_horizon=1712.0,
            time_horizon_unit="engine_hours",
        )
        assert prob_high >= prob_low

    def test_predict_probability_monotonic_in_time(self, loaded_model):
        """Вероятность должна расти с горизонтом."""
        prob_short = predict_probability(
            loaded_model, raw_peak=0.71, time_horizon=500.0,
            time_horizon_unit="engine_hours",
        )
        prob_long = predict_probability(
            loaded_model, raw_peak=0.71, time_horizon=1712.0,
            time_horizon_unit="engine_hours",
        )
        assert prob_long >= prob_short

    def test_predict_many_returns_correct_count(self, loaded_model):
        """predict_many возвращает столько же вероятностей, сколько пиков."""
        peaks = [0.3, 0.71, 1.0, 1.5]
        result = predict_many(
            loaded_model, peaks, time_horizon=1712.0,
            time_horizon_unit="engine_hours",
        )
        assert len(result["probabilities"]) == len(peaks)
        assert len(result["peaks"]) == len(peaks)
        for p in result["probabilities"]:
            assert 0.0 <= p <= 1.0

    def test_predict_probability_rejects_out_of_range_peak(self, loaded_model):
        """PeakLoad вне диапазона обучения должен быть отклонён."""
        with pytest.raises(Exception):
            predict_probability(
                loaded_model, raw_peak=100.0, time_horizon=1712.0,
                time_horizon_unit="engine_hours",
            )

    def test_predict_probability_rejects_negative_horizon(self, loaded_model):
        """Отрицательный горизонт должен быть отклонён."""
        with pytest.raises(Exception):
            predict_probability(
                loaded_model, raw_peak=0.71, time_horizon=-1.0,
                time_horizon_unit="engine_hours",
            )

    def test_predict_probability_zero_horizon(self, loaded_model):
        """При горизонте 0 вероятность должна быть 0."""
        prob = predict_probability(
            loaded_model, raw_peak=0.71, time_horizon=0.0,
            time_horizon_unit="engine_hours",
        )
        assert prob == 0.0 or prob < 1e-10


# ===========================================================================
# 3. Baseline hazard
# ===========================================================================
class TestBaselineHazard:
    """Проверка базовой накопленной функции риска."""

    def test_baseline_at_zero(self, loaded_model):
        """H₀(0) = 0."""
        h0 = baseline_cumulative_hazard(loaded_model, 0.0)
        assert h0 == 0.0

    def test_baseline_monotonic(self, loaded_model):
        """H₀(t) должна быть неубывающей."""
        times = [100.0, 500.0, 1000.0, 1712.0]
        values = baseline_cumulative_hazard(loaded_model, times)
        for i in range(len(values) - 1):
            assert values[i + 1] >= values[i]

    def test_baseline_scalar_and_list_consistency(self, loaded_model):
        """Скалярный и списочный вызовы должны давать одинаковый результат."""
        scalar = baseline_cumulative_hazard(loaded_model, 1000.0)
        list_result = baseline_cumulative_hazard(loaded_model, [1000.0])
        assert abs(scalar - list_result[0]) < 1e-12


# ===========================================================================
# 4. Конвертация времени
# ===========================================================================
class TestTimeConversion:
    """Проверка конвертации единиц времени."""

    def test_engine_hours_to_days(self):
        """1712 мч = 214 дней при 8 мч/день."""
        days = engine_hours_to_calendar_days(1712.0, 8.0)
        assert abs(days - 214.0) < 1e-10

    def test_days_to_engine_hours(self):
        """214 дней = 1712 мч при 8 мч/день."""
        hours = calendar_days_to_engine_hours(214.0, 8.0)
        assert abs(hours - 1712.0) < 1e-10

    def test_roundtrip(self):
        """Конвертация туда-обратно должна возвращать исходное значение."""
        original = 1234.5
        days = engine_hours_to_calendar_days(original, 8.0)
        back = calendar_days_to_engine_hours(days, 8.0)
        assert abs(back - original) < 1e-10


# ===========================================================================
# 5. Premium: end-to-end
# ===========================================================================
class TestPremiumEndToEnd:
    """Полный цикл расчёта премии."""

    def test_single_premium_basic(self, premium_params):
        """Базовый расчёт премии."""
        result = calculate_single_premium(**premium_params)
        assert result["net_undiscounted"] > 0
        assert result["gross_discounted"] > 0
        assert result["tariff"] > 0
        assert result["severity_based"] is False

    def test_premium_formula_consistency(self, premium_params):
        """Проверка формулы: gross = net * (1 + theta)."""
        result = calculate_single_premium(**premium_params)
        expected_gross = result["net_discounted"] * (1 + premium_params["theta"])
        assert abs(result["gross_discounted"] - expected_gross) < 1e-6

    def test_premium_batch(self, premium_params):
        """Пакетный расчёт премий."""
        probs = [0.01, 0.028, 0.05]
        results = calculate_premium(
            probs,
            premium_params["sum_insured"],
            premium_params["theta"],
            premium_params["discount_rate"],
            policy_horizon_days=premium_params["policy_horizon_days"],
        )
        assert len(results) == len(probs)
        for r in results:
            assert r["net_undiscounted"] >= 0

    def test_premium_zero_probability(self, premium_params):
        """При P=0 премия должна быть 0."""
        result = calculate_single_premium(
            probability=0.0,
            sum_insured=premium_params["sum_insured"],
            theta=premium_params["theta"],
        )
        assert result["net_undiscounted"] == 0.0
        assert result["gross_discounted"] == 0.0

    def test_premium_probability_one(self, premium_params):
        """При P=1 нетто = sum_insured (legacy)."""
        result = calculate_single_premium(
            probability=1.0,
            sum_insured=premium_params["sum_insured"],
            theta=0.0,
        )
        assert abs(result["net_undiscounted"] - premium_params["sum_insured"]) < 1e-6

    def test_premium_severity_based(self, premium_params):
        """Severity-based расчёт отличается от legacy."""
        legacy = calculate_single_premium(**premium_params)
        severity = calculate_single_premium(
            probability=premium_params["probability"],
            sum_insured=premium_params["sum_insured"],
            theta=premium_params["theta"],
            expected_severity=200_000.0,
            deductible=10_000.0,
        )
        assert severity["severity_based"] is True
        assert severity["net_undiscounted"] != legacy["net_undiscounted"]

    def test_premium_rejects_invalid_probability(self, premium_params):
        """Вероятность вне [0,1] должна быть отклонена."""
        with pytest.raises(Exception):
            calculate_single_premium(
                probability=1.5,
                sum_insured=premium_params["sum_insured"],
            )

    def test_premium_rejects_negative_sum_insured(self, premium_params):
        """Отрицательная страховая сумма должна быть отклонена."""
        with pytest.raises(Exception):
            calculate_single_premium(
                probability=0.028,
                sum_insured=-100.0,
            )


# ===========================================================================
# 6. Prediction → Premium: полный pipeline
# ===========================================================================
class TestPredictionToPremium:
    """Полный pipeline: предсказание → премия."""

    def test_full_pipeline(self, loaded_model, premium_params):
        """Предсказать вероятность → рассчитать премию."""
        prob = predict_probability(
            loaded_model,
            raw_peak=0.71,
            time_horizon=1712.0,
            time_horizon_unit="engine_hours",
        )
        result = calculate_single_premium(
            probability=prob,
            sum_insured=premium_params["sum_insured"],
            theta=premium_params["theta"],
            discount_rate=premium_params["discount_rate"],
            policy_horizon_days=premium_params["policy_horizon_days"],
        )
        assert result["net_undiscounted"] >= 0
        assert result["gross_discounted"] >= 0
        assert 0.0 <= result["tariff"] <= 100.0

    def test_pipeline_multiple_peaks(self, loaded_model, premium_params):
        """Пакетный pipeline для нескольких пиков."""
        peaks = [0.3, 0.71, 1.5]
        result = predict_many(
            loaded_model, peaks, time_horizon=1712.0,
            time_horizon_unit="engine_hours",
        )
        premiums = calculate_premium(
            result["probabilities"],
            premium_params["sum_insured"],
            premium_params["theta"],
            premium_params["discount_rate"],
            policy_horizon_days=premium_params["policy_horizon_days"],
        )
        assert len(premiums) == len(peaks)
        # Премия должна расти с PeakLoad (при положительном γ)
        assert premiums[-1]["gross_discounted"] >= premiums[0]["gross_discounted"]


# ===========================================================================
# 7. Model registry
# ===========================================================================
class TestModelRegistry:
    """Проверка управления версиями моделей."""

    def test_generate_filename(self):
        """Генерация имени файла по конвенции."""
        name = generate_model_filename("1.0", "light", "20250815")
        assert name == "model_params_v1.0_light_20250815.json"

    def test_parse_filename(self):
        """Парсинг имени файла по конвенции."""
        info = parse_model_filename("model_params_v1.0_light_20250815.json")
        assert info is not None
        assert info.major == 1
        assert info.minor == 0
        assert info.segment == "light"
        assert info.date_str == "20250815"

    def test_parse_filename_invalid(self):
        """Некорректное имя файла должно вернуть None."""
        info = parse_model_filename("model_params.json")
        assert info is None

    def test_generate_and_parse_roundtrip(self):
        """Генерация → парсинг должны быть согласованы."""
        name = generate_model_filename("0.2", "heavy", "20260101")
        info = parse_model_filename(name)
        assert info is not None
        assert info.major == 0
        assert info.minor == 2
        assert info.segment == "heavy"
        assert info.date_str == "20260101"


# ===========================================================================
# 8. Recalibration triggers
# ===========================================================================
class TestRecalibrationTriggers:
    """Проверка триггеров перекалибровки."""

    def test_loss_ratio_triggers(self):
        """Loss ratio отклонение > 20% должно сработать."""
        result = check_loss_ratio_deviation(0.60, 0.80)
        assert result.triggered is True
        assert result.action == "recalibrate"

    def test_loss_ratio_ok(self):
        """Loss ratio отклонение < 20% не должно сработать."""
        result = check_loss_ratio_deviation(0.60, 0.65)
        assert result.triggered is False

    def test_f_statistic_triggers(self):
        """F < 14.18 должно сработать."""
        result = check_f_statistic(10.0)
        assert result.triggered is True
        assert result.action == "switch_to_predictive"

    def test_f_statistic_ok(self):
        """F > 14.18 не должно сработать."""
        result = check_f_statistic(50.0)
        assert result.triggered is False

    def test_major_share_triggers(self):
        """Изменение major share > 30% должно сработать."""
        result = check_major_share_change(0.30, 0.45)
        assert result.triggered is True
        assert result.action == "update_severity"

    def test_instrument_drift_triggers(self):
        """Падение partial R² > 30% должно сработать."""
        result = check_instrument_drift(0.05, 0.03)
        assert result.triggered is True

    def test_instrument_drift_ok(self):
        """Падение partial R² < 30% не должно сработать."""
        result = check_instrument_drift(0.05, 0.045)
        assert result.triggered is False

    def test_format_report(self):
        """Форматирование отчёта не должно падать."""
        report = {
            "timestamp": "2025-01-01T00:00:00",
            "model_path": "model_params.json",
            "claims_path": None,
            "n_events_checked": 0,
            "triggers": [],
            "summary": {
                "total_triggers": 0,
                "triggered_count": 0,
                "critical_count": 0,
                "warning_count": 0,
                "recommended_actions": [],
                "needs_recalibration": False,
                "needs_severity_update": False,
                "needs_mode_switch": False,
            },
        }
        text = format_trigger_report(report)
        assert isinstance(text, str)
        assert "ТРИГГЕРЫ" in text

    def test_save_report(self, tmp_path):
        """Сохранение отчёта в JSON."""
        report = {"timestamp": "2025-01-01", "triggers": []}
        path = str(tmp_path / "trigger_report.json")
        save_trigger_report(report, path)
        assert Path(path).exists()
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["timestamp"] == "2025-01-01"


# ===========================================================================
# 9. Kaplan-Meier validation
# ===========================================================================
class TestKaplanMeierValidation:
    """Проверка K-M валидации."""

    def test_km_check_basic(self, loaded_model):
        """Базовый вызов K-M check."""
        times = np.array([100.0, 500.0, 1000.0, 1500.0, 1712.0])
        events = np.array([1, 0, 1, 0, 1])
        result = kaplan_meier_check(
            loaded_model, times, events, eval_horizon=1712.0,
        )
        assert "km_survival" in result
        assert "model_survival" in result
        assert "abs_diff" in result
        assert 0.0 <= result["km_survival"] <= 1.0
        assert 0.0 <= result["model_survival"] <= 1.0

    def test_km_check_rejects_empty(self, loaded_model):
        """Пустые данные должны быть отклонены."""
        with pytest.raises(Exception):
            kaplan_meier_check(loaded_model, [], [], eval_horizon=1712.0)


# ===========================================================================
# 10. Brand helpers
# ===========================================================================
class TestBrandHelpers:
    """Проверка работы с брендами."""

    def test_coerce_brand_code_by_name(self, loaded_model):
        """Преобразование имени бренда в код."""
        assert coerce_brand_code(loaded_model, "MTZ82") == 0
        assert coerce_brand_code(loaded_model, "Versatile280") == 1

    def test_coerce_brand_code_by_number(self, loaded_model):
        """Преобразование числового кода."""
        assert coerce_brand_code(loaded_model, 0) == 0
        assert coerce_brand_code(loaded_model, 3) == 3

    def test_coerce_brand_code_rejects_invalid(self, loaded_model):
        """Некорректный бренд должен быть отклонён."""
        with pytest.raises(Exception):
            coerce_brand_code(loaded_model, "UnknownBrand")


# ===========================================================================
# 11. Power segments
# ===========================================================================
class TestPowerSegments:
    """Проверка классификации мощности."""

    def test_light_segment(self):
        assert classify_power_segment(100.0) == "light"
        assert classify_power_segment(199.9) == "light"

    def test_medium_segment(self):
        assert classify_power_segment(200.0) == "medium"
        assert classify_power_segment(319.9) == "medium"

    def test_heavy_segment(self):
        assert classify_power_segment(320.0) == "heavy"
        assert classify_power_segment(500.0) == "heavy"

    def test_boundary_values(self):
        """Граничные значения."""
        assert classify_power_segment(0.0) == "light"
        assert classify_power_segment(199.99) == "light"
        assert classify_power_segment(200.0) == "medium"
        assert classify_power_segment(320.0) == "heavy"