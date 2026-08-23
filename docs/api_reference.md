# API Reference

Справочник публичных функций проекта CF Cox Insurance Pricing System.

**Версия:** 1.1  
**Дата:** 2026-08-15

---

## Конвенции проекта

- **Python:** ≥ 3.10; рекомендуется 3.11+.
- **Модельное время:** `engine_hours` (мото-часы).
- **MTTR / downtime:** календарные часы простоя; они не конвертируются в `engine_hours`.
- **Event definition:** `major_claim`.
- **IV-режим:** `predictive`; каузальная интерпретация не допускается без подтверждённого production-инструмента.
- **Сегменты:** DGP-сегменты `light` / `heavy` и operational classification `light` / `medium` / `heavy` — разные сущности.
- **Версионирование моделей:** `v{major}.{minor}_{segment}_{YYYYMMDD}`.

---

## `prediction_engine.py`

### Загрузка и сохранение модели

#### `load_model_params(path, validate=True) -> ModelParameters`

Загружает модель из JSON-файла.

| Параметр | Тип | Описание |
|---|---|---|
| `path` | `str \| Path` | Путь к файлу модели |
| `validate` | `bool` | Выполнять ли валидацию; по умолчанию `True` |

**Возвращает:** `ModelParameters`

**Исключения:** `ModelLoadError`, `ModelValidationError`

---

#### `save_model_params(path, params) -> None`

Сохраняет модель в JSON-файл.

| Параметр | Тип | Описание |
|---|---|---|
| `path` | `str \| Path` | Путь для сохранения |
| `params` | `ModelParameters` | Параметры модели |

**Исключения:** `ModelLoadError`

---

#### `validate_model(params) -> bool`

Полная валидация модели: структура, коэффициенты, baseline и метаданные.

| Параметр | Тип | Описание |
|---|---|---|
| `params` | `ModelParameters` | Параметры модели |

**Возвращает:** `True`, если модель валидна.

**Исключения:** `ModelValidationError`

---

### Предсказание

#### `predict_probability(params, raw_peak, time_horizon, residual_policy="plug-in", covariates=None, time_horizon_unit=None, strict_covariates=True) -> float`

Предсказывает `P(T ≤ t | x)` для одного значения PeakLoad.

| Параметр | Тип | По умолчанию | Описание |
|---|---|---:|---|
| `params` | `ModelParameters` | — | Параметры модели |
| `raw_peak` | `float` | — | Значение PeakLoad |
| `time_horizon` | `float` | — | Горизонт предсказания |
| `residual_policy` | `str` | `"plug-in"` | Политика остатков |
| `covariates` | `Dict[str, Any] \| None` | `None` | Переопределение ковариат |
| `time_horizon_unit` | `str \| None` | `None` | Единица времени горизонта |
| `strict_covariates` | `bool` | `True` | Строгая проверка ковариат |

**Возвращает:** `float` — вероятность в `[0, 1]`.

**Исключения:** `InvalidInputError`, `PredictionError`, `ModelValidationError`

---

#### `predict_many(params, raw_peaks, time_horizon, residual_policy="plug-in", covariates=None, time_horizon_unit=None, strict_covariates=True) -> Dict[str, Any]`

Пакетное предсказание для нескольких PeakLoad.

| Параметр | Тип | По умолчанию | Описание |
|---|---|---:|---|
| `params` | `ModelParameters` | — | Параметры модели |
| `raw_peaks` | `Sequence[float] \| np.ndarray` | — | Список PeakLoad |
| `time_horizon` | `float` | — | Горизонт предсказания |
| `residual_policy` | `str` | `"plug-in"` | Политика остатков |
| `covariates` | `Dict[str, Any] \| None` | `None` | Переопределение ковариат |
| `time_horizon_unit` | `str \| None` | `None` | Единица времени |
| `strict_covariates` | `bool` | `True` | Строгая проверка |

**Возвращает:**

```python
{
    "probabilities": List[float],
    "peaks": List[float],
    "time_horizon": float,
    "time_horizon_unit": str,
    "residual_policy": str,
}
```

---

### Трансформация и проверка признаков

#### `transform_peak(params, peak_raw) -> float`

Применяет к PeakLoad трансформацию, заданную моделью: `none`, `center` или `standardize`.

#### `name_is_cf(name) -> bool`

Проверяет, является ли имя признака CF-колонкой.

#### `is_cf_column_name(name, cf_cols) -> bool`

Проверяет имя признака относительно списка CF-колонок.

#### `name_is_brand_related(name) -> bool`

Проверяет, относится ли имя признака к brand-признакам.

#### `validate_index_value(value, name, reference) -> float`

Проверяет индексное значение и возвращает нормализованный `float`; допустимый диапазон — `[0, 1]`.

---

### Baseline hazard

#### `baseline_cumulative_hazard(params, time_horizon) -> float | List[float]`

Возвращает `H₀(t)` — базовую накопленную функцию риска.

| Параметр | Тип | Описание |
|---|---|---|
| `params` | `ModelParameters` | Параметры модели |
| `time_horizon` | `float \| Sequence[float]` | Точка или точки времени |

**Возвращает:** `float` для скаляра, `List[float]` для последовательности.

---

### Первая стадия и partial-out

#### `predict_first_stage(params, covariates=None, strict_covariates=True) -> float`

Вычисляет прогноз первой стадии `η = X'β`.

#### `compute_pl_hat_exog(params, pl_hat, covariates=None, strict_covariates=True) -> float`

Вычисляет `PL_hat_exog` с partial-out корректировкой.

---

### CF basis

#### `build_cf_basis_at_prediction(params, residuals_arr) -> Dict[str, np.ndarray]`

Строит CF basis для предсказания. Поддерживает `linear`, `powers` и `spline`.

---

### Конвертация времени

#### `engine_hours_to_calendar_days(engine_hours, hours_per_day=8.0) -> float`

Конвертация мото-часов в календарные дни.

#### `calendar_days_to_engine_hours(calendar_days, hours_per_day=8.0) -> float`

Конвертация календарных дней в мото-часы.

---

### Вспомогательные функции

#### `get_freq_shares(params) -> Dict[str, float]`

Возвращает доли отказов по системам, нормализованные к сумме 1.

#### `get_severity_weights(params) -> Dict[str, float]`

Возвращает веса стоимости отказов по системам.

#### `get_criticality_weights(params) -> Dict[str, float]`

Возвращает веса критичности систем.

#### `get_mtbf_baseline_hours(params) -> float`

Возвращает baseline MTBF в `engine_hours`.

#### `get_downtime_hours(params, mttr_hours) -> float`

Вычисляет downtime как `MTTR × downtime_per_mttr_factor`. MTTR и downtime измеряются в календарных часах.

#### `classify_power_segment(power_hp, params=None) -> str`

Operational-классификация мощности: `"light"`, `"medium"` или `"heavy"`.

> Эта классификация не является DGP-сегментацией. DGP использует отдельные сегменты `light` / `heavy`.

#### `kaplan_meier_check(params, times, events, eval_horizon=None) -> Dict[str, float]`

Kaplan–Meier валидация baseline survival.

**Возвращает:**

```python
{
    "eval_horizon": float,
    "km_survival": float,
    "model_survival": float,
    "abs_diff": float,
    "n_obs": int,
    "n_events": int,
}
```

#### `coerce_brand_code(params, value) -> int`

Преобразует название бренда или код в канонический код `[0, 4]`.

---

## `premium_engine.py`

### `calculate_single_premium(probability, sum_insured, ...) -> Dict[str, float]`

Расчёт премии для одной вероятности.

| Параметр | Тип | По умолчанию | Описание |
|---|---|---:|---|
| `probability` | `float` | — | Вероятность в `[0, 1]` |
| `sum_insured` | `float` | — | Страховая сумма |
| `theta` | `float` | `0.15` | Коэффициент нагрузки |
| `discount_rate` | `float` | `0.0` | Годовая ставка дисконтирования |
| `calibration_horizon_days` | `float \| None` | `None` | Горизонт калибровки |
| `policy_horizon_days` | `float \| None` | `None` | Горизонт полиса |
| `probability_horizon_days` | `float \| None` | `None` | Горизонт, к которому относится переданная вероятность |
| `expected_severity` | `float \| None` | `None` | Ожидаемая стоимость убытка |
| `deductible` | `float` | `0.0` | Франшиза |
| `coverage_limit` | `float \| None` | `None` | Лимит покрытия |

**Возвращает:**

```python
{
    "net_undiscounted": float,
    "net_discounted": float,
    "gross_undiscounted": float,
    "gross_discounted": float,
    "net": float,
    "gross": float,
    "tariff": float,          # в процентах
    "discount_factor": float,
    "loading_amount": float,
    "severity_based": bool,
}
```

---

### `calculate_premium(probabilities, sum_insured, ...) -> List[Dict[str, float]]`

Пакетный расчёт премий.

---

### `calculate_premium_with_severity(severity_model, probability, sum_insured, ...) -> Dict[str, float]`

Расчёт премии через severity-модель.

| Параметр | Тип | Описание |
|---|---|---|
| `severity_model` | `SeverityModel` | Объект с методом `expected_covered_loss()` |
| `probability` | `float` | Вероятность события |
| `sum_insured` | `float` | Страховая сумма |
| `theta` | `float` | Коэффициент нагрузки |
| `deductible` | `float` | Франшиза |
| `coverage_limit` | `float \| None` | Лимит покрытия |
| `discount_rate` | `float` | Ставка дисконтирования |
| `policy_horizon_days` | `float \| None` | Горизонт полиса |

---

### `calculate_from_prediction_result(prediction_result, sum_insured, ...) -> List[Dict[str, float]]`

Адаптер из результата `service.py` в расчёт премий.

---

### `_scale_probability_to_horizon(...)`

**Deprecated.** Устаревший внутренний механизм масштабирования вероятности к горизонту. Для нового кода использовать актуальные параметры горизонта и публичные функции `premium_engine`.

---

## `service.py`

### `predict_from_model_file(model_path, peaks_raw, time_horizon, residual_policy, x_values) -> Dict[str, Any]`

Главная точка входа для предсказания. Кэширует модели и валидирует входы.

| Параметр | Тип | Описание |
|---|---|---|
| `model_path` | `str` | Путь к `model_params.json` |
| `peaks_raw` | `Iterable[float]` | Список PeakLoad |
| `time_horizon` | `float` | Горизонт предсказания |
| `residual_policy` | `str \| ResidualPolicy` | Политика остатков |
| `x_values` | `Dict[str, float] \| None` | Переопределение ковариат |

**Возвращает:** `Dict` с `probabilities`, `peaks`, `metadata` и `warnings`.

### `clear_model_cache() -> None`

Очищает кэш загруженных моделей.

---

## `predictor.py`

### `Predictor(params, *, allow_diagnostic_residual_policies=False, allow_horizon_extrapolation=False, strict_covariates=False, allow_unknown_covariates=False)`

Высокоуровневая обёртка над `prediction_engine`.

| Параметр | Тип | По умолчанию | Описание |
|---|---|---:|---|
| `params` | `ModelParamsProtocol` | — | Параметры модели |
| `allow_diagnostic_residual_policies` | `bool` | `False` | Разрешить диагностические политики остатков |
| `allow_horizon_extrapolation` | `bool` | `False` | Разрешить предсказание вне поддерживаемого горизонта |
| `strict_covariates` | `bool` | `False` | Включить строгую проверку ковариат |
| `allow_unknown_covariates` | `bool` | `False` | Разрешить неизвестные ковариаты |

### Методы

| Метод | Описание |
|---|---|
| `predict_probability(raw_peak, time_horizon, residual_policy, covariates)` | Одиночное предсказание |
| `predict_many(raw_peaks, time_horizon, residual_policy, covariates)` | Пакетное предсказание |
| `hazard_ratios()` | Hazard ratios = `exp(coef)` |
| `from_model_json(path)` | Factory: загрузка из JSON |

### Свойства

| Свойство | Тип | Описание |
|---|---|---|
| `params` | `ModelParameters` | Параметры модели, read-only |
| `calibration_time_horizon` | `float` | Горизонт калибровки |
| `time_unit` | `str` | Единица времени |
| `coefficients` | `Dict[str, float]` | Cox-коэффициенты |

---

## `severity_model.py`

### `load_severity_model(path) -> SeverityModel`

Загрузка severity-модели из JSON.

### `save_severity_model(path, model) -> None`

Сохранение severity-модели в JSON.

### `load_claims_events(path) -> pd.DataFrame`

Загрузка и фильтрация событий из claims.

### `build_severity_model(events, hourly_downtime_cost=2500.0) -> SeverityModel`

Построение severity-модели из claims-событий.

### `estimate_system_severity(events, system, total_events) -> SystemSeverity`

Оценка severity для конкретной системы отказов.

### `compute_exact_covered_loss(events, deductible, coverage_limit, hourly_downtime_cost) -> float`

Точный расчёт `E[covered_loss]` по событиям.

### `SeverityModel` — основные методы

| Метод | Возвращает | Описание |
|---|---|---|
| `expected_repair_cost()` | `float` | `E[repair_cost]` |
| `expected_downtime_cost()` | `float` | `E[downtime_hours] × hourly_cost` |
| `expected_loss_per_failure()` | `float` | `E[repair] + E[downtime_cost]` |
| `expected_covered_loss(deductible, coverage_limit)` | `float` | `E[max(0, loss - deductible)]` |

---

## `model_registry.py`

### `parse_model_filename(filename) -> Optional[ModelVersionInfo]`

Разбирает имя файла по конвенции `v{major}.{minor}_{segment}_{date}`.

### `list_model_versions(directory=Path(".")) -> List[ModelVersionInfo]`

Возвращает список версий моделей в директории.

### `get_latest_model_path(directory=Path("."), segment=None, major=None) -> Optional[Path]`

Возвращает путь к последней версии модели.

### `load_model_metadata(path) -> Dict[str, Any]`

Загружает `training_meta` из файла модели.

### `compare_model_versions(path_a, path_b) -> Dict[str, Any]`

Сравнивает метаданные двух версий.

### `generate_model_filename(version, segment, date_str=None) -> str`

Генерирует имя файла по конвенции версионирования.

---

## `recalibration_triggers.py`

### `check_all_triggers(model_path, claims_path=None, ...) -> Dict[str, Any]`

Проверяет все триггеры перекалибровки.

### `check_loss_ratio_deviation(expected, actual, threshold=0.20) -> TriggerResult`

Проверяет отклонение loss ratio.

### `check_calibration_error(calibration_table, threshold=0.05) -> TriggerResult`

Проверяет ошибку калибровки.

### `check_major_share_change(baseline, current, threshold=0.30) -> TriggerResult`

Проверяет изменение доли major-отказов.

### `check_f_statistic(current_f, threshold=14.18) -> TriggerResult`

Проверяет силу инструмента по F-statistic.

### `check_instrument_drift(baseline_r2, current_r2, threshold=0.30) -> TriggerResult`

Проверяет относительный дрейф instrument diagnostics.

### `format_trigger_report(report) -> str`

Форматирует отчёт о триггерах для консоли.

### `save_trigger_report(report, output_path) -> None`

Сохраняет отчёт о триггерах в JSON.

---

## `instrument_monitor.py`

### `load_monitoring_baseline(model_path) -> Dict[str, Optional[float]]`

Загружает baseline-диагностики из `training_meta`.

### `compute_instrument_diagnostics(data) -> InstrumentHealthReport`

Пересчитывает IV-диагностики на новых данных.

### `check_instrument_drift(baseline_f, current_f, ...) -> Tuple[bool, Optional[float]]`

Детектирует дрейф инструмента по F-statistic.

### `check_partial_r2_drift(baseline_r2, current_r2, ...) -> Tuple[bool, Optional[float]]`

Детектирует дрейф по partial R².

### `compare_with_baseline(report, baseline) -> InstrumentHealthReport`

Сравнивает текущий отчёт с baseline и определяет дрейф.

### `recommend_recalibration(report) -> InstrumentHealthReport`

Формирует рекомендации по перекалибровке.

### `print_report(report) -> None`

Выводит отчёт в консоль.

### `save_report(report, output_path) -> None`

Сохраняет отчёт в JSON.

---

## `exceptions.py`

Иерархия исключений:

```text
ProjectError
├── ModelError
│   ├── ModelLoadError
│   ├── ModelValidationError
│   ├── ModelArtifactError
│   ├── ModelSerializationError
│   └── ModelCoefficientError
├── PredictionError
│   ├── InvalidInputError
│   ├── TransformationError
│   ├── ProbabilityError
│   ├── PredictionOverflowError
│   └── PredictionUnderflowError
├── PremiumCalculationError
├── ConfigurationError
├── CLIError
├── DataValidationError
│   └── FeatureError
└── NumericalError
    └── ConvergenceError
```

Все исключения поддерживают `cause` chaining через `__cause__`.

---

## REST API Endpoints

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/health` | Проверка здоровья API |
| `GET` | `/model/info` | Информация о модели |
| `GET` | `/model/versions` | Список версий моделей |
| `POST` | `/predict` | Предсказание вероятностей |
| `POST` | `/premium` | Расчёт премии |
| `POST` | `/premium/batch` | Пакетный расчёт премий |
| `POST` | `/calculate` | End-to-end расчёт |
| `POST` | `/cache/clear` | Очистка кэша моделей |

Swagger UI: `http://localhost:8000/docs`  
ReDoc: `http://localhost:8000/redoc`

---

## Примечания по совместимости

Этот документ описывает публичный API и зафиксированные проектные соглашения. Если фактическая сигнатура функции в исходном коде изменяется, `api_reference.md` должен обновляться одновременно с изменением кода.

Deprecated API не следует использовать в новом коде; deprecated-элементы сохраняются в справочнике только для совместимости и миграции.

---

## Критерии готовности

- [x] `docs/api_reference.md` создан.
- [x] Таблицы Markdown имеют корректные разделители.
- [x] Публичные функции `prediction_engine` из согласованного перечня задокументированы.
- [x] Публичные функции `premium_engine` из согласованного перечня задокументированы.
- [x] `service.py`, `predictor.py`, `severity_model.py` задокументированы.
- [x] `model_registry.py`, `recalibration_triggers.py`, `instrument_monitor.py` задокументированы.
- [x] Иерархия исключений описана.
- [x] REST API endpoints перечислены.
- [x] Добавлены проектные конвенции: Python ≥ 3.10, `major_claim`, `iv_mode=predictive`, единицы времени и разделение сегментов.
- [x] Deprecated `_scale_probability_to_horizon` отмечен.
