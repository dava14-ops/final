#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_provenance.py
Валидация происхождения модели: кампания, определение события.
Предотвращает использование модели, обученной на одной кампании,
для предсказаний на другой.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

VALID_CAMPAIGNS = frozenset({"sowing", "harvest"})
VALID_EVENT_DEFINITIONS = frozenset({"total_loss", "major_claim", "any_failure"})


def get_model_weather_campaign(model_json: dict) -> str:
    """Извлечь weather_campaign из training_meta модели."""
    if not isinstance(model_json, dict):
        return "sowing"
    tm = model_json.get("training_meta", {})
    if not isinstance(tm, dict):
        return "sowing"
    dgp = tm.get("dgp", {})
    if not isinstance(dgp, dict):
        return "sowing"
    campaign = str(dgp.get("weather_campaign", "sowing")).lower()
    if campaign not in VALID_CAMPAIGNS:
        return "sowing"
    return campaign


def assert_prediction_campaign(
    model_json: dict, required_campaign: str
) -> None:
    """
    Проверить, что модель обучена на нужной кампании.
    Бросает ValueError при несовпадении.
    """
    trained = get_model_weather_campaign(model_json)
    required = str(required_campaign).strip().lower()
    if required not in VALID_CAMPAIGNS:
        raise ValueError(
            f"Неизвестная кампания: '{required}'. "
            f"Допустимые: {sorted(VALID_CAMPAIGNS)}."
        )
    if trained != required:
        raise ValueError(
            f"Модель обучена на campaign='{trained}', "
            f"но предсказание требует '{required}'. "
            f"Переобучите модель на соответствующей кампании."
        )


def assert_prediction_event_compatible(
    training_meta: dict, insurance_event: str
) -> None:
    """
    Проверить совместимость определения события модели
    с выбранным страховым случаем.

    insurance_event:
        "1" = любой отказ
        "2" = только major

    Если модель обучена на event_definition='any_failure',
    а страховой случай = 'только major', результат некорректен.
    """
    if not isinstance(training_meta, dict):
        return
    model_ed = str(training_meta.get("event_definition", "major_claim")).lower()
    if model_ed not in VALID_EVENT_DEFINITIONS:
        return

    if insurance_event == "2" and model_ed == "any_failure":
        raise ValueError(
            f"Страховой случай='только major', но модель обучена на "
            f"event_definition='{model_ed}'. "
            f"Модель включает minor-отказы в вероятность. "
            f"Переобучите модель с event_definition='major_claim'."
        )


def normalize_model_campaign_metadata(model_json: dict) -> dict:
    """Нормализовать метаданные кампании в training_meta."""
    if not isinstance(model_json, dict):
        return model_json
    tm = model_json.get("training_meta")
    if not isinstance(tm, dict):
        return model_json
    dgp = tm.get("dgp", {})
    if isinstance(dgp, dict):
        campaign = dgp.get("weather_campaign", "sowing")
        tm["weather_campaign_normalized"] = str(campaign).lower()
    return model_json
