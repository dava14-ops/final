#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api_server.py
Фаза 9.3: REST API для предсказания вероятностей и расчёта премий.

Интеграция:
    service.py          → predict_from_model_file()
    premium_engine.py   → calculate_single_premium(), calculate_premium()
    severity_model.py   → load_severity_model() (Фаза 7.9)
    model_registry.py   → get_latest_model_path() (Фаза 8.3)
    exceptions.py       → иерархия исключений

Запуск:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
    или
    python api_server.py

Документация:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from exceptions import (
    InvalidInputError,
    ModelLoadError,
    ModelValidationError,
    PredictionError,
    PremiumCalculationError,
    ProjectError,
)
from service import predict_from_model_file, clear_model_cache
from premium_engine import calculate_single_premium, calculate_premium
from model_registry import get_latest_model_path, list_model_versions

logger = logging.getLogger(__name__)

__all__ = ["app"]

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "model_params.json"
DEFAULT_SEVERITY_PATH = "severity_model_v1.json"
API_VERSION = "1.0.0"
MAX_PEAKS = 10_000
MAX_PROBABILITIES = 10_000
MAX_COVARIATES = 1_000

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# FastAPI приложение
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CF Cox Insurance Pricing API",
    description=(
        "REST API для предсказания вероятностей отказов сельхозтехники "
        "и расчёта страховых премий на основе модели CF Cox / IV-Cox.\n\n"
        "Модель предсказывает P(T ≤ t | x) через Cox proportional hazards "
        "с корректировкой эндогенности PeakLoad (Control Function / 2SRI)."
    ),
    version=API_VERSION,
)


# ---------------------------------------------------------------------------
# Pydantic модели запросов
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Запрос на предсказание вероятностей."""

    peaks: List[float] = Field(
        ...,
        min_length=1,
        max_length=MAX_PEAKS,
        description="Список значений PeakLoad для предсказания.",
        examples=[[0.5, 0.71, 0.9]],
    )
    time_horizon: float = Field(
        ...,
        gt=0,
        description=(
            "Горизонт предсказания в единицах модели "
            "(обычно engine_hours)."
        ),
        examples=[1712.0],
    )
    time_horizon_unit: Optional[str] = Field(
        default=None,
        description=(
            "Единица времени горизонта: 'engine_hours', 'hours', 'days'. "
            "Если None, используется единица модели."
        ),
        examples=["engine_hours"],
    )
    residual_policy: str = Field(
        default="plug-in",
        description=(
            "Политика обработки остатков: 'plug-in' (production), "
            "'mean' или 'zero' (diagnostic only)."
        ),
        examples=["plug-in"],
    )
    covariates: Optional[Dict[str, float]] = Field(
        default=None,
        max_length=MAX_COVARIATES,
        description=(
            "Опциональные ковариаты для переопределения шаблона. "
            "Например: {'x_age': 10.0, 'x_hours': 1000.0, 'Brand': 0}"
        ),
    )
    model_path: Optional[str] = Field(
        default=None,
        description=(
            "Путь к файлу модели. Если None, используется последняя "
            "версия из model_registry или model_params.json."
        ),
    )

    @field_validator("peaks")
    @classmethod
    def validate_peaks(cls, v: List[float]) -> List[float]:
        for i, p in enumerate(v):
            if not math.isfinite(p):
                raise ValueError(f"peaks[{i}] must be finite")
            if p < 0:
                raise ValueError(f"peaks[{i}] must be non-negative")
        return v

    @field_validator("residual_policy")
    @classmethod
    def validate_policy(cls, v: str) -> str:
        allowed = {"plug-in", "mean", "zero"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"residual_policy must be one of {sorted(allowed)}, "
                f"got '{v}'"
            )
        return v_lower


class PremiumRequest(BaseModel):
    """Запрос на расчёт премии для одной вероятности."""

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Вероятность события в [0, 1].",
        examples=[0.028],
    )
    sum_insured: float = Field(
        ...,
        gt=0,
        description="Страховая сумма (руб.).",
        examples=[5_000_000.0],
    )
    theta: float = Field(
        default=0.15,
        ge=0.0,
        description="Коэффициент страховой нагрузки (доля).",
        examples=[0.15],
    )
    discount_rate: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Годовая ставка дисконтирования (доля).",
        examples=[0.08],
    )
    policy_horizon_days: Optional[float] = Field(
        default=None,
        gt=0,
        description="Горизонт полиса в календарных днях.",
        examples=[214.0],
    )
    calibration_horizon_days: Optional[float] = Field(
        default=None,
        gt=0,
        description="Горизонт калибровки модели в календарных днях.",
    )
    expected_severity: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Ожидаемая стоимость убытка (руб.). Если задано, "
            "премия считается как P × E[covered_loss] вместо "
            "P × sum_insured."
        ),
    )
    deductible: float = Field(
        default=0.0,
        ge=0.0,
        description="Франшиза (руб.).",
    )
    coverage_limit: Optional[float] = Field(
        default=None,
        gt=0,
        description="Лимит покрытия (руб.).",
    )
    use_severity_model: bool = Field(
        default=False,
        description=(
            "Если True, загрузить severity_model_v1.json и "
            "использовать expected_loss_per_failure()."
        ),
    )


class BatchPremiumRequest(BaseModel):
    """Запрос на расчёт премий для нескольких вероятностей."""

    probabilities: List[float] = Field(
        ...,
        min_length=1,
        max_length=MAX_PROBABILITIES,
        description="Список вероятностей в [0, 1].",
    )
    sum_insured: float = Field(..., gt=0)
    theta: float = Field(default=0.15, ge=0.0)
    discount_rate: float = Field(default=0.0, ge=0.0, lt=1.0)
    policy_horizon_days: Optional[float] = Field(default=None, gt=0)
    calibration_horizon_days: Optional[float] = Field(default=None, gt=0)
    expected_severity: Optional[float] = Field(default=None, gt=0)
    deductible: float = Field(default=0.0, ge=0.0)
    coverage_limit: Optional[float] = Field(default=None, gt=0)

    @field_validator("probabilities")
    @classmethod
    def validate_probs(cls, v: List[float]) -> List[float]:
        for i, p in enumerate(v):
            if not math.isfinite(p):
                raise ValueError(f"probabilities[{i}] must be finite")
            if p < 0.0 or p > 1.0:
                raise ValueError(
                    f"probabilities[{i}] must be in [0, 1]"
                )
        return v


class EndToEndRequest(BaseModel):
    """Запрос на полный расчёт: PeakLoad → вероятность → премия."""

    peaks: List[float] = Field(..., min_length=1, max_length=MAX_PEAKS)
    time_horizon: float = Field(..., gt=0)
    time_horizon_unit: Optional[str] = None
    sum_insured: float = Field(..., gt=0)
    theta: float = Field(default=0.15, ge=0.0)
    discount_rate: float = Field(default=0.0, ge=0.0, lt=1.0)
    policy_horizon_days: Optional[float] = Field(default=None, gt=0)
    residual_policy: str = Field(default="plug-in")
    covariates: Optional[Dict[str, float]] = None
    expected_severity: Optional[float] = Field(default=None, gt=0)
    deductible: float = Field(default=0.0, ge=0.0)
    coverage_limit: Optional[float] = Field(default=None, gt=0)
    use_severity_model: bool = Field(default=False)
    model_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Pydantic модели ответов
# ---------------------------------------------------------------------------
class PredictResponse(BaseModel):
    """Ответ на предсказание."""

    probabilities: List[float]
    peaks: List[float]
    time_horizon: float
    time_horizon_unit: Optional[str] = None
    residual_policy: str
    warnings: Optional[List[str]] = None


class PremiumResponse(BaseModel):
    """Ответ на расчёт премии."""

    net_undiscounted: float
    net_discounted: float
    gross_undiscounted: float
    gross_discounted: float
    net: float
    gross: float
    tariff: float
    discount_factor: float
    loading_amount: float
    severity_based: bool = False
    severity_source: Optional[str] = None


class BatchPremiumResponse(BaseModel):
    """Ответ на пакетный расчёт премий."""

    premiums: List[Dict[str, Any]]
    count: int


class EndToEndResponse(BaseModel):
    """Ответ на полный расчёт."""

    predictions: PredictResponse
    premiums: List[Dict[str, Any]]
    tariff_by_peak: Dict[str, float]


class ModelInfoResponse(BaseModel):
    """Информация о модели."""

    model_path: str
    model_version: Optional[str] = None
    model_semantic_version: Optional[str] = None
    engine_convention: Optional[str] = None
    time_unit: Optional[str] = None
    calibration_time_horizon: Optional[float] = None
    event_definition: Optional[str] = None
    segment: Optional[str] = None
    n_events: Optional[int] = None
    iv_mode: Optional[str] = None
    retrained_on_real_claims: bool = False


class HealthResponse(BaseModel):
    """Проверка здоровья API."""

    status: str
    api_version: str
    model_available: bool
    severity_model_available: bool
    timestamp: float


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def _resolve_model_path(model_path: Optional[str] = None) -> str:
    """Определить путь к модели."""
    if model_path:
        path = Path(model_path)
        if path.exists():
            return str(path)
        raise ModelLoadError(f"Model file not found: {model_path}")

    # Попытка 1: последняя версия из model_registry
    latest = get_latest_model_path()
    if latest is not None and latest.exists():
        return str(latest)

    # Попытка 2: стандартный путь
    default = Path(DEFAULT_MODEL_PATH)
    if default.exists():
        return str(default)

    raise ModelLoadError(
        f"No model found. Checked: {latest}, {default}. "
        "Run train_model.py first."
    )


def _load_severity_expected_loss() -> Optional[float]:
    """Загрузить expected_loss_per_failure из severity_model_v1.json."""
    path = Path(DEFAULT_SEVERITY_PATH)
    if not path.exists():
        return None
    try:
        from severity_model import load_severity_model
        model = load_severity_model(path)
        if model.fallback_used:
            logger.warning(
                "severity_model_v1.json is fallback. "
                "Using expert constants."
            )
            return None
        return model.expected_loss_per_failure()
    except Exception as exc:
        logger.warning("Failed to load severity model: %s", exc)
        return None


def _get_model_info(model_path: str) -> Dict[str, Any]:
    """Извлечь метаданные модели."""
    import json
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    meta = data.get("training_meta", {})
    if not isinstance(meta, dict):
        meta = {}

    return {
        "model_path": model_path,
        "model_version": data.get("model_version"),
        "model_semantic_version": data.get("model_semantic_version"),
        "engine_convention": data.get("engine_convention"),
        "time_unit": meta.get("time_unit"),
        "calibration_time_horizon": data.get("calibration_time_horizon"),
        "event_definition": meta.get("event_definition"),
        "segment": meta.get("segment"),
        "n_events": meta.get("n_events"),
        "iv_mode": meta.get("iv_mode"),
        "retrained_on_real_claims": meta.get(
            "retrained_on_real_claims", False
        ),
    }


# ---------------------------------------------------------------------------
# Обработчики исключений
# ---------------------------------------------------------------------------
@app.exception_handler(InvalidInputError)
async def invalid_input_handler(request, exc):
    return _error_response(422, str(exc))


@app.exception_handler(ModelLoadError)
async def model_load_handler(request, exc):
    return _error_response(500, str(exc))


@app.exception_handler(ModelValidationError)
async def model_validation_handler(request, exc):
    return _error_response(500, str(exc))


@app.exception_handler(PredictionError)
async def prediction_handler(request, exc):
    return _error_response(500, str(exc))


@app.exception_handler(PremiumCalculationError)
async def premium_handler(request, exc):
    return _error_response(500, str(exc))


@app.exception_handler(ProjectError)
async def project_error_handler(request, exc):
    return _error_response(500, str(exc))


def _error_response(status_code: int, detail: str):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={"error": detail},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Проверка здоровья API.

    Возвращает статус, версию API и доступность модели.
    """
    model_available = False
    try:
        _resolve_model_path()
        model_available = True
    except Exception:
        pass

    severity_available = Path(DEFAULT_SEVERITY_PATH).exists()

    return HealthResponse(
        status="ok" if model_available else "degraded",
        api_version=API_VERSION,
        model_available=model_available,
        severity_model_available=severity_available,
        timestamp=time.time(),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info(
    model_path: Optional[str] = Query(
        default=None,
        description="Путь к модели. Если не задан, используется последняя.",
    ),
):
    """
    Информация о загруженной модели.

    Возвращает версию, единицы времени, горизонт калибровки,
    определение события, сегмент и IV-режим.
    """
    try:
        path = _resolve_model_path(model_path)
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    info = _get_model_info(path)
    return ModelInfoResponse(**info)


@app.get("/model/versions")
async def model_versions():
    """
    Список всех версий моделей в директории.

    Возвращает список файлов, соответствующих конвенции
    model_params_v{major}.{minor}_{segment}_{date}.json.
    """
    versions = list_model_versions()
    return {
        "count": len(versions),
        "versions": [v.to_dict() for v in versions],
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Предсказать вероятности отказов для списка PeakLoad.

    Использует Cox PH модель с корректировкой эндогенности
    через Control Function (2SRI).

    **Пример запроса:**
    ```json
    {
        "peaks": [0.5, 0.71, 0.9],
        "time_horizon": 1712.0,
        "time_horizon_unit": "engine_hours",
        "residual_policy": "plug-in"
    }
    ```
    """
    try:
        model_path = _resolve_model_path(req.model_path)
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        result = predict_from_model_file(
            model_path=model_path,
            peaks_raw=req.peaks,
            time_horizon=req.time_horizon,
            residual_policy=req.residual_policy,
            x_values=req.covariates,
        )
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except (ModelLoadError, ModelValidationError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except PredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return PredictResponse(
        probabilities=result.get("probabilities", []),
        peaks=result.get("peaks", req.peaks),
        time_horizon=req.time_horizon,
        time_horizon_unit=result.get("time_unit"),
        residual_policy=result.get("residual_policy", req.residual_policy),
        warnings=result.get("warnings"),
    )


@app.post("/premium", response_model=PremiumResponse)
async def premium(req: PremiumRequest):
    """
    Рассчитать страховую премию для одной вероятности.

    **Формула (legacy):**
        net = P × sum_insured × discount_factor
        gross = net × (1 + theta)
        tariff = gross / sum_insured × 100%

    **Формула (severity-based):**
        covered = max(0, expected_severity - deductible)
        covered = min(covered, coverage_limit) если задан
        net = P × covered × discount_factor
        gross = net × (1 + theta)

    Если `use_severity_model=true`, expected_severity извлекается
    из severity_model_v1.json.
    """
    expected_severity = req.expected_severity
    severity_source = None

    if req.use_severity_model:
        loaded = _load_severity_expected_loss()
        if loaded is not None:
            expected_severity = loaded
            severity_source = "severity_model_v1.json"
        else:
            severity_source = "fallback (not loaded)"

    try:
        result = calculate_single_premium(
            probability=req.probability,
            sum_insured=req.sum_insured,
            theta=req.theta,
            discount_rate=req.discount_rate,
            calibration_horizon_days=req.calibration_horizon_days,
            policy_horizon_days=req.policy_horizon_days,
            expected_severity=expected_severity,
            deductible=req.deductible,
            coverage_limit=req.coverage_limit,
        )
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except PremiumCalculationError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Premium calculation failed")
        raise HTTPException(status_code=500, detail=str(exc))

    result["severity_source"] = severity_source
    return PremiumResponse(**result)


@app.post("/premium/batch", response_model=BatchPremiumResponse)
async def premium_batch(req: BatchPremiumRequest):
    """
    Рассчитать премии для нескольких вероятностей.
    """
    try:
        results = calculate_premium(
            probabilities=req.probabilities,
            sum_insured=req.sum_insured,
            theta=req.theta,
            discount_rate=req.discount_rate,
            calibration_horizon_days=req.calibration_horizon_days,
            policy_horizon_days=req.policy_horizon_days,
            expected_severity=req.expected_severity,
            deductible=req.deductible,
            coverage_limit=req.coverage_limit,
        )
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except PremiumCalculationError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Batch premium calculation failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return BatchPremiumResponse(
        premiums=results,
        count=len(results),
    )


@app.post("/calculate", response_model=EndToEndResponse)
async def calculate_end_to_end(req: EndToEndRequest):
    """
    Полный расчёт: PeakLoad → вероятность → премия.

    Объединяет /predict и /premium в один вызов.

    **Поток:**
    1. Предсказание P(T ≤ t | PeakLoad) через Cox модель
    2. Расчёт net/gross премии через premium_engine
    3. Опциональная загрузка severity_model для expected_severity
    """
    # Шаг 1: предсказание
    try:
        model_path = _resolve_model_path(req.model_path)
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        prediction_result = predict_from_model_file(
            model_path=model_path,
            peaks_raw=req.peaks,
            time_horizon=req.time_horizon,
            residual_policy=req.residual_policy,
            x_values=req.covariates,
        )
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except (ModelLoadError, ModelValidationError, PredictionError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed in end-to-end")
        raise HTTPException(status_code=500, detail=str(exc))

    probabilities = prediction_result.get("probabilities", [])
    peaks_returned = prediction_result.get("peaks", req.peaks)

    if not probabilities:
        raise HTTPException(
            status_code=500,
            detail="Prediction returned empty probabilities",
        )

    # Шаг 2: severity
    expected_severity = req.expected_severity
    if req.use_severity_model:
        loaded = _load_severity_expected_loss()
        if loaded is not None:
            expected_severity = loaded

    # Шаг 3: премии
    try:
        premiums = calculate_premium(
            probabilities=probabilities,
            sum_insured=req.sum_insured,
            theta=req.theta,
            discount_rate=req.discount_rate,
            policy_horizon_days=req.policy_horizon_days,
            expected_severity=expected_severity,
            deductible=req.deductible,
            coverage_limit=req.coverage_limit,
        )
    except (InvalidInputError, PremiumCalculationError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Premium calculation failed in end-to-end")
        raise HTTPException(status_code=500, detail=str(exc))

    # Шаг 4: тарифы по пикам
    tariff_by_peak = {}
    for i, peak in enumerate(peaks_returned):
        if i < len(premiums):
            tariff_by_peak[str(peak)] = premiums[i].get("tariff", 0.0)

    predict_response = PredictResponse(
        probabilities=probabilities,
        peaks=peaks_returned,
        time_horizon=req.time_horizon,
        time_horizon_unit=prediction_result.get("time_unit"),
        residual_policy=req.residual_policy,
        warnings=prediction_result.get("warnings"),
    )

    return EndToEndResponse(
        predictions=predict_response,
        premiums=premiums,
        tariff_by_peak=tariff_by_peak,
    )


@app.post("/cache/clear")
async def clear_cache():
    """
    Очистить кэш загруженных моделей.

    Используйте после замены model_params.json на диске.
    """
    clear_model_cache()
    return {"status": "ok", "message": "Model cache cleared"}


# ---------------------------------------------------------------------------
# CLI запуск
# ---------------------------------------------------------------------------
def main() -> None:
    """Запуск через uvicorn."""
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()