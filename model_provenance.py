from __future__ import annotations
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_model_weather_campaign(model_json: dict) -> str:
    """Извлечь кампанию погоды из метаданных модели."""
    tm = model_json.get("training_meta", {})
    dgp = tm.get("dgp", {})
    return str(dgp.get("weather_campaign", "sowing"))


def assert_prediction_campaign(model_json: dict, required_campaign: str) -> None:
    """
    Проверить, что кампания предсказания совпадает с кампанией обучения модели.
    
    Args:
        model_json: JSON-представление модели с training_meta
        required_campaign: Требуемая кампания ('sowing' или 'harvest')
    
    Raises:
        ValueError: Если кампании не совпадают
    """
    trained = get_model_weather_campaign(model_json)
    if trained != required_campaign:
        raise ValueError(
            f"Модель обучена на campaign='{trained}', "
            f"но предсказание требует '{required_campaign}'. "
            f"Переобучите модель на соответствующей кампании."
        )


def assert_prediction_event_compatible(
    training_meta: dict, insurance_event: str
) -> None:
    """
    Проверить совместимость определения страхового события с моделью.
    
    Args:
        training_meta: Метаданные обучения модели
        insurance_event: Тип страхового события ('1' или '2')
    
    Raises:
        ValueError: Если событие несовместимо с моделью
    """
    model_ed = str(training_meta.get("event_definition", "major_claim"))
    if insurance_event == "2" and model_ed == "any_failure":
        raise ValueError(
            f"Страховой случай='только major', но модель обучена на "
            f"event_definition='{model_ed}'. Несовместимо."
        )


def normalize_model_campaign_metadata(model_json: dict) -> dict:
    """
    Нормализовать метаданные кампании в JSON модели.
    
    Args:
        model_json: JSON-представление модели
    
    Returns:
        Модель с нормализованными метаданными
    """
    tm = model_json.get("training_meta", {})
    dgp = tm.get("dgp", {})
    campaign = dgp.get("weather_campaign", "sowing")
    tm["weather_campaign_normalized"] = campaign
    return model_json
