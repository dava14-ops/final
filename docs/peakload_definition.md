# Справочник параметров (Parameter Reference)

Все параметры, которые можно вводить вручную, сгруппированы по модулям.

---

## CLI параметры (`cli.py`)

| Параметр | По умолчанию | Описание | Единица |
|----------|-------------|----------|---------|
| `DEFAULT_MODEL` | `model_params.json` | Путь к файлу модели (JSON) | — |
| `DEFAULT_SUM_INSURED` | `5,000,000` | Максимальная выплата по полису при наступлении страхового случая | руб. |
| `DEFAULT_HORIZON_ENGINE_HOURS` | `1712` | Горизонт прогнозирования: 214 дней × 8 мч/день | мото-часов |
| `DEFAULT_HORIZON_DAYS` | `214` | Горизонт прогнозирования в календарных днях | дней |
| `DEFAULT_THETA` | `0.15` | Надбавка к чистой премии на администрирование, риск и прибыль | доля (0–1) |
| `DEFAULT_RESIDUAL_POLICY` | `plug-in` | Метод обработки остаточной эндогенности: plug-in, mean, zero | — |
| `DEFAULT_DISCOUNT_RATE` | `0.08` | Годовая ставка приведения будущей выплаты к текущей стоимости | доля (0–1) |
| `DEFAULT_TIME_UNIT` | `engine_hours` | Временная шкала модели | — |
| `DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY` | `8.0` | Сколько мото-часов работы припадает на один календарный день | мч/день |
| `CALIBRATION_HORIZON_DAYS` | `214` | Период калибровки модели для целевой вероятности | дней |
| `CALIBRATION_HORIZON_ENGINE_HOURS` | `1712` | Горизонт калибровки в мото-часах (= 214 × 8) | мото-часов |
| `MAJOR_FAILURE_SHARE` | `0.30` | Доля отказов, считаемых крупными (покрываются страховкой). Экспертная оценка | доля (0–1) |
| `COMMON_COVARIATES` | `['Z', 'X', 'Age', ...]` | Доступные переменные: возраст, наработка, климат, грунт, бренд, мощность | — |

---

## Premium Engine (`premium_engine.py`)

| Параметр | По умолчанию | Описание | Единица |
|----------|-------------|----------|---------|
| `probability` | *(требуется)* | Вероятность отказа, вычисленная Cox-моделью | доля (0–1) |
| `sum_insured` | *(требуется)* | Максимальная выплата по полису | руб. |
| `theta` | `0.15` | Надбавка к чистой премии: `брутто = нетто × (1 + theta)` | доля (0–1) |
| `discount_rate` | `0.0` | Ставка дисконтирования. Формула: `PV = FV × exp(-r × t/365)` | доля (0–1) |
| `calibration_horizon_days` | *(опционально)* | Период калибровки модели. Запасной для дисконтирования | дней |
| `policy_horizon_days` | *(опционально)* | Фактическая длительность страхового полиса. Если не указана — берётся горизонт прогнозирования | дней |

---

## Prediction Engine (`prediction_engine.py`)

| Параметр | По умолчанию | Описание | Единица |
|----------|-------------|----------|---------|
| `MAX_BATCH_SIZE` | `10,000` | Максимальное число прогнозов за одну итерацию | шт. |
| `MAX_LP` | `700.0` | Верхний предел для `exp(lp)`: предотвращает переполнение | — |
| `MIN_LP` | `-700.0` | Нижний предел для `exp(lp)`: предотвращает денормализацию | — |
| `PEAK_RANGE_TOLERANCE` | `5.0` | Допустимое отклонение от диапазона обучающей выборки при проверке пиковой нагрузки | — |
| `ENFORCE_PEAK_RANGE` | `True` | Валидация: пиковая нагрузка должна быть в пределах обучающей выборки | — |
| `PROBABILITY_EPSILON` | `1e-12` | Границы вероятности: `P ∈ [epsilon, 1-epsilon]` | — |
| `MODEL_TIME_UNIT` | `engine_hours` | Временная шкала предсказательного движка | — |
| `MTBF_INPUT_UNIT` | `engine_hours` | Единица измерения MTBF (средняя наработка на отказ) | — |
| `MTBF_TO_MODEL_TIME_FACTOR` | `1.0` | Множитель перевода часов MTBF в часы модели | — |
| `DEFAULT_WEIBULL_SHAPE` | `1.88` | Параметр формы распределения Вейбулла. >1 означает стареющую надёжность | — |

---

## Взаимосвязь параметров

```
cli.py → задаёт входные данные:
  ├── sum_insured → premium_engine
  ├── theta       → premium_engine
  ├── discount_rate → premium_engine
  └── horizon     → prediction_engine

prediction_engine.py:
  ├── horizon (мото-часы) → вычисляет probability (Cox)
  └── probability         → premium_engine

premium_engine.py:
  ├── probability         → net = probability × sum_insured
  ├── theta               → gross = net × (1 + theta)
  └── discount_rate       → discounted = gross × exp(-r×t/365)
```

---

## Формулы

### Чистая премия (Net Premium)
```
net_undiscounted   = probability × sum_insured
net_discounted     = net_undiscounted × exp(-discount_rate × horizon_days / 365)
```

### Брутто премия (Gross Premium)
```
gross_discounted   = net_discounted × (1 + theta)
tariff             = gross_discounted / sum_insured × 100
```

### Вероятность отказа (Cox PH model)
```
P(T ≤ t | x)      = 1 - exp(-H₀(t) × exp(β·x))
```
где `H₀(t)` — базовая накопленная функция риска, `β` — коэффициенты модели, `x` — вектор ковариат.

---

## Источники параметров

| Источник | Файл |
|----------|------|
| Единый реестр | `_param_registry.py` |
| Интерактивный CLI | `cli.py` |
| Расчёт премии | `premium_engine.py` |
| Предсказательный движок | `prediction_engine.py` |
| Файл модели (автоматический) | `model_params.json` |
