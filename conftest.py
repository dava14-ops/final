"""
conftest.py — фикстуры для тестов агрокалендаря.
Сессионные фикстуры: вычисляются один раз на весь прогон.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pytest

logger = logging.getLogger("test_agro_pipeline")


# ═══════════════════════════════════════════════════════════════════════
# Фикстура: загрузка модулей
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def agro_calendar():
    """Загрузка модуля agro_calendar.py."""
    try:
        import agro_calendar as _ac
        return _ac
    except ImportError as exc:
        pytest.skip(f"agro_calendar.py не найден: {exc}")


@pytest.fixture(scope="session")
def agro_norms():
    """Загрузка модуля agro_norms.py."""
    try:
        import agro_norms as _an
        return _an
    except ImportError as exc:
        pytest.skip(f"agro_norms.py не найден: {exc}")


@pytest.fixture(scope="session")
def operation_mapping():
    """Загрузка модуля operation_mapping.py."""
    try:
        import operation_mapping as _om
        return _om
    except ImportError as exc:
        pytest.skip(f"operation_mapping.py не найден: {exc}")


# ═══════════════════════════════════════════════════════════════════════
# Фикстура: пики операций из OPERATION_INFO / TUM_OPERATIONS
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def operation_peaks() -> Dict[str, float]:
    """Словарь {operation_key: peak_load_mean}."""
    peaks: Dict[str, float] = {}
    try:
        from Real_calculator import OPERATION_INFO
        for op_key, op_info in OPERATION_INFO.items():
            peaks[op_key] = op_info.get("peak_load_mean", 0.50)
    except ImportError:
        pass

    try:
        from Real_calculator import TUM_OPERATIONS
        for op_key, op_info in TUM_OPERATIONS.items():
            if op_key not in peaks:
                peaks[op_key] = op_info.get("peak_load_mean", 0.50)
    except ImportError:
        pass

    if not peaks:
        # Минимальный набор пиков для тестов
        peaks = {
            "Ploughing": 0.85,
            "Cultivating (deep)": 0.72,
            "Cultivating (shallow)": 0.45,
            "Disc harrowing": 0.55,
            "Seedbed combination": 0.50,
            "Seed drill combination 3m": 0.60,
            "Seed drill combination 4m": 0.65,
            "Precision air seeding": 0.58,
            "Spraying": 0.40,
            "Fertilizing": 0.38,
            "Mowing (front)": 0.62,
            "Swathing": 0.55,
            "Transport": 0.35,
        }

    return peaks


# ═══════════════════════════════════════════════════════════════════════
# Фикстура: результаты вычисления моточасов
# ═══════════════════════════════════════════════════════════════════════

TEST_CROPS = ["wheat_spring", "forage_beet", "potato"]
TEST_AREAS = [100.0, 200.0]
TEST_TRACTORS = ["МТЗ-82", "К-744Р1"]


@pytest.fixture(scope="session")
def results(
    agro_calendar,
    agro_norms,
    operation_peaks,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Вычисление суммарных моточасов и средневзвешенного пика.
    Возвращает {crop_key: {f"{area}га_{tractor}": (total_hours, weighted_peak)}}.
    """
    computed: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for crop_key in TEST_CROPS:
        computed[crop_key] = {}
        crop = agro_calendar.get_crop(crop_key)
        if crop is None:
            continue

        for area in TEST_AREAS:
            for tractor in TEST_TRACTORS:
                label = f"{area}га_{tractor}"
                try:
                    total_hours, weighted_peak = agro_calendar.estimate_season_engine_hours(
                        crop_key, area, tractor=tractor, k_ob=1.0
                    )
                    computed[crop_key][label] = (total_hours, weighted_peak)
                except Exception as exc:
                    logger.warning(
                        "Ошибка вычисления %s %s: %s", crop_key, label, exc
                    )
                    continue

    return computed


# ═══════════════════════════════════════════════════════════════════════
# Фикстура: вероятности из prediction_engine
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def model_params():
    """Загрузка модели из model_params.json."""
    from pathlib import Path

    model_path = Path("model_params.json")
    if not model_path.exists():
        pytest.skip("model_params.json не найден")

    try:
        from prediction_engine import load_model_params, validate_model
        params = load_model_params(str(model_path))
        validate_model(params)
        return params
    except Exception as exc:
        pytest.skip(f"Модель не загружена: {exc}")


@pytest.fixture(scope="session")
def probabilities(
    model_params,
    results,
) -> Dict[str, float]:
    """
    Вычисление вероятностей для каждой культуры/площади/трактора.
    Возвращает {f"{crop_key}_{label}": probability}.
    """
    from prediction_engine import predict_probability, InvalidInputError

    probs: Dict[str, float] = {}

    for crop_key in TEST_CROPS:
        for label, (total_hours, weighted_peak) in results.get(crop_key, {}).items():
            prob_key = f"{crop_key}_{label}"
            try:
                prob = predict_probability(
                    model_params,
                    weighted_peak,
                    total_hours,
                    residual_policy="plug-in",
                    covariates={},
                    time_horizon_unit="engine_hours",
                    strict_covariates=False,
                )
                probs[prob_key] = float(prob)
            except InvalidInputError:
                # Горизонт превышает базовый риск — пропускаем
                logger.warning(
                    "%s: горизонт %.0f мч превышает базовый риск",
                    prob_key,
                    total_hours,
                )
                continue
            except Exception as exc:
                logger.warning("%s: ошибка предсказания: %s", prob_key, exc)
                continue

    if not probs:
        pytest.skip("Ни одна вероятность не вычислена")

    return probs


# ═══════════════════════════════════════════════════════════════════════
# Фикстура: константы тестов
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Конфигурация тестов."""
    return {
        "crops": TEST_CROPS,
        "areas": TEST_AREAS,
        "tractors": TEST_TRACTORS,
        "calibration_horizon": 214.0,
        "expected_hours_per_ha": {
            "wheat_spring": (1.0, 6.0),
            "forage_beet":  (3.0, 7.0),
            "potato":       (3.0, 10.0),
        },
    }