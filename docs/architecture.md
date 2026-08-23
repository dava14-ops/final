# Architecture: CF Cox / IV-Cox

> **Версия**: 1.0  
> **Дата**: 2026-08-15  
> **Статус**: pre-production / simulation-based

## 1. Назначение

Документ фиксирует фактическую архитектуру проекта, границы между DGP, обучением, inference и premium calculation, а также решения по ранее найденным несоответствиям.

## 2. Runtime

- Минимум: **Python 3.10**
- Рекомендуется: **Python 3.11+**
- Основные библиотеки: NumPy, pandas, SciPy, statsmodels, lifelines, joblib, FastAPI, Uvicorn.

## 3. Поток

```text
DGP / claims
    ↓
feature engineering
    ↓
PeakLoad + X + Z + time + event
    ↓
First stage: PeakLoad ~ Z + X
    ↓
Pl_hat + v_hat
    ↓
CF Cox / 2SRI
    ↓
baseline cumulative hazard
    ↓
model_params.json
    ↓
Predictor / API / CLI
    ↓
P(T ≤ t | X, PeakLoad)
    ↓
premium_engine
```

## 4. Время

Основная единица survival-модели:

```text
MODEL_TIME_UNIT = engine_hours
```

Калибровочный горизонт:

```text
214 календарных дней × 8 engine hours/day = 1712 engine_hours
```

`failure_time`, `time` и аналогичные survival-поля измеряются в engine hours.

`MTTR` и `downtime_hours` — **календарные часы простоя**. Они не переводятся в engine hours и не складываются с survival time.

## 5. Power segments

В проекте существуют две разные классификации.

### DGP / training

```text
light
heavy
```

`SEGMENTS = ("light", "heavy")`.

### Operational classification

```text
light  [0, 200)
medium [200, 320)
heavy  [320, ∞)
```

`medium` относится к operational classification, но не к DGP segments.

Имя `segment` в model artifact относится к training/DGP segment.

## 6. Event model

Текущее целевое событие:

```text
event_definition = major_claim
```

```text
event = 1 → наблюдаемое событие
event = 0 → цензура
```

Для `any_failure` требуется отдельное переобучение.

## 7. First stage

Оценивается:

```text
PeakLoad ~ Z + X
```

Диагностика включает F-statistic, HC3 robust F, partial R², Cragg-Donald и VIF.

Формируется:

```text
Pl_hat = E[PeakLoad | Z, X]
v_hat  = PeakLoad - Pl_hat
```

## 8. CF Cox

Вторая стадия:

```text
h(t|X,PeakLoad,v_hat)
 = h0(t) * exp(
    gamma * PeakLoad
    + lambda * v_hat
    + beta * X
  )
```

Используется `lifelines.CoxPHFitter`.

Поддерживаемые CF basis:

```text
linear
powers
spline
```

Residual может стандартизоваться как `(v_hat - mean(v_hat)) / std(v_hat)`.

## 9. Covariates

| Claims / source | Model variable |
|---|---|
| `age_at_event` | `x_age` |
| `hours_at_event` | `x_hours` |
| `power_hp` | `x_power` |
| `brand` | `x_brand` / dummies |
| `climate_index` | `x_climate` |
| `soil_index` | `x_soil` |
| `peak_load_proxy` | `PeakLoad` |

Brand encoding: dummies, reference category `MTZ82`.

## 10. Brand normalization

Канонические коды:

```text
0 MTZ82
1 Versatile280
2 NewHollandT9
3 DT75
4 Other
```

Перед lookup вход должен нормализоваться:

```text
strip → case normalization → alias lookup → canonical brand
```

Проверка текущего `constants.py` не выявила trailing spaces в `BRAND_ALIASES` и `RF_HEAVY_BRAND_CATALOG`.

## 11. Failure systems

Текущий simulation/reference справочник:

```text
гидравлика   0.30
электроника  0.30
двигатель    0.12
трансмиссия  0.20
прочее       0.08
```

Это не следует трактовать как эмпирические claims-частоты до Фазы 7.

## 12. MTTR / downtime

`MTTR_HOURS` и `DOWNTIME_STATS` описывают разные модельные представления и не должны применяться одновременно к одной записи как два независимых расчёта downtime.

В Фазе 7 приоритетным источником становятся фактические `downtime_hours` из claims.

## 13. Baseline

Simulation baseline поддерживает Weibull, Gompertz и Exponential.

Текущая основная simulation-конфигурация:

```text
Weibull
shape = 1.88
```

Для real-claims модели baseline должен быть переоценён по фактическим survival times; планируемый estimator — Breslow.

## 14. Model artifact

Формат:

```text
model_params_v{major}.{minor}_{segment}_{YYYYMMDD}.json
```

Где:

```text
major 0 = simulation
major 1 = real claims
major 2+ = architecture change

minor 0 = base
minor 1 = covariates / baseline
minor 2 = CF / segment / brand encoding
```

`segment` = training/DGP segment (`light` / `heavy`).

## 15. Prediction layer

```text
model_params
    ↓
Predictor
    ↓
prediction_engine
    ↓
CF transformation
    ↓
baseline cumulative hazard
    ↓
P(T ≤ t | X, PeakLoad)
```

Production residual policy: `plug-in`.

## 16. Premium layer

Prediction engine отвечает за вероятность события. `premium_engine.py` выполняет экономическую трансформацию:

```text
P → expected severity → deductible / coverage limit → discount → gross loading → tariff
```

## 17. API

Архитектурный путь:

```text
FastAPI
   ↓
service
   ↓
Predictor
   ↓
prediction_engine
```

API не должен дублировать математическую реализацию Cox/CF.

## 18. Simulation → real claims

### Сейчас

```text
DGP → Monte Carlo → CF Cox → model_params v0.x
```

### Фаза 7

```text
raw claims
 → data_contract validation
 → claims_clean
 → feature engineering
 → IV diagnostics
 → CF Cox retraining
 → empirical baseline
 → severity model
 → temporal backtesting
 → model_params v1.x
```

## 19. Validation

### Data

- mandatory fields;
- ranges;
- missing values;
- duplicates;
- event/censoring semantics.

### IV

- F-statistic;
- partial R²;
- Cragg-Donald;
- weak-instrument diagnostics;
- VIF.

### Survival

- Kaplan-Meier;
- calibration;
- Brier score;
- O/E;
- temporal backtesting;
- PH diagnostics.

## 20. Known limitations

На 2026-08-15:

1. Текущая модель simulation-based.
2. TUM PeakLoad основан на Fendt-телеметрии и не репрезентативен для всего парка РФ.
3. Weather instrument не подтверждён production claims.
4. Climate/Soil требуют эмпирической валидации.
5. Downtime assumptions ещё заменяются реальными claims.
6. Weibull shape 1.88 — simulation assumption.
7. `iv_mode = predictive`; causal interpretation запрещена до подтверждения инструмента.
8. `major_claim` — текущая целевая event definition.

## 21. Решения по выявленным несоответствиям

| № | Решение |
|---|---|
| 1 | Python ≥3.10; рекомендуется 3.11+ |
| 2 | DGP segments и operational classification — разные сущности |
| 3 | `MTTR_HOURS` и `DOWNTIME_STATS` не смешиваются |
| 4 | Model time = engine hours; MTTR/downtime = calendar hours |
| 5 | Текущий `constants.py`: trailing spaces не обнаружены |
| 6 | `SEGMENTS=(light, heavy)` сохраняется; `medium` только operational |

## 22. Source of truth

При конфликте:

1. исполняемый код и тесты;
2. versioned data contract;
3. model/training metadata;
4. architecture documentation;
5. README.

Изменение поведения модели сначала фиксируется в коде и тестах, затем в metadata, architecture и changelog.

## 23. Следующий этап

```text
Phase 6
  ↓
real claims collection
  ↓
data contract validation
  ↓
Phase 7 retraining
  ↓
v1.0
  ↓
temporal backtesting
  ↓
shadow pricing
```

Simulation v0.x сохраняются как reference/baseline и не заменяются автоматически.
