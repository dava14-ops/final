# Assumption Log

Все модельные допущения с обоснованием, статусом и чувствительностью.

---

## A-01: MAJOR_FAILURE_SHARE — Bayesian prior

| Поле | Значение |
|---|---|
| **Статус** | 🟢 CLOSED (Bayesian calibration) |
| **Prior** | Beta(9, 21) |
| **Prior mean** | 0.30 |
| **Prior effective_n** | 30 (эквивалент 30 pseudo-observations) |
| **Источник prior** | Expert elicitation, зафиксирован **до** использования данных |
| **Posterior** | Beta(α₀ + k, β₀ + n − k) |
| **Production value** | Posterior mean после обновления на данных |
| **Текущие данные** | k = 871 major, n = 29732 total (simulated hybrid) |
| **Posterior mean** | 0.0296 |
| **95% CrI** | [0.0277, 0.0315] |
| **Brand prior** | Тот же global Beta(9,21); для редких брендов shrinkage сильный |

### Методологические оговорки

1. Prior effective_n = 30 является отдельным модельным допущением.
   При n > 200 влияние prior на posterior mean практически исчезает.
2. Prior mean 0.30 зафиксирован до использования текущих данных.
3. Brand-level shrinkage: для редких брендов (n_b < 10) prior доминирует.
   В P1 планируется hierarchical/empirical-Bayes prior.

### Sensitivity

| Prior effective_n | Posterior mean |
|---:|---:|
| 10 | 0.0297 |
| 30 | 0.0296 |
| 100 | 0.0295 |

---

## A-02: Weibull baseline shape = 1.88

| Поле | Значение |
|---|---|
| **Статус** | 🟢 VALIDATED (независимая эмпирическая проверка) |
| **Значение** | 1.88 (wear-out regime) |
| **Интерпретация** | Hazard rate растёт со временем (износ) |
| **Валидация** | Zetor Weibull shape = 1.91 (Durczak et al., 2018) |
| **Отклонение** | **1.8%** |
| **Best fit** | Weibull (AIC=895.20) лучше Exponential/LogNormal/LogLogistic |
| **Артефакт** | `reports/zetor_survival_audit.json` |

### Обоснование

Shape > 1 означает wear-out regime: интенсивность отказов растёт
со временем. Это физически правдоподобно для сельскохозяйственной техники:
- Первый сезон: приработка, ранние отказы
- Последующие сезоны: износ компонентов, рост интенсивности отказов

### Независимая валидация

```text
Zetor published (70 tractors, Durczak et al. 2018):
  MTTF = 271 moto-hours (published) / 276 mth (empirical)
  Weibull shape = 1.91 (fitted)

Our DGP baseline:
  Weibull shape = 1.88
  Deviation: 1.8%
```

---

## A-03: MTBF_BASELINE_HOURS = 1500

| Поле | Значение |
|---|---|
| **Статус** | 🟡 PARTIALLY VALIDATED |
| **Значение** | 1500 engine hours |
| **Интерпретация** | Mean Time Between Failures для light segment |
| **Валидация** | MTTF/MTBF ratio = 0.184 ∈ [0.1, 0.6] |
| **Ограничение** | Zetor MTTF = 271 mth (first failure), не MTBF |

### Обоснование

MTBF = 1500h означает, что в среднем между последовательными отказами
проходит 1500 моточасов. Это согласуется с:
- Zetor MTTF = 271 mth (first failure) → ratio 0.184
- Литературные данные по сельхозтехнике: MTBF 1000–3000h

---

## A-04: HADCO Service & Severity Benchmark (Level 1)

| Поле | Значение |
|---|---|
| **Статус** | 🟢 CLOSED (external benchmark) |
| **Источник** | Al-Suhaibani & Wahby (литературная реконструкция, Level 1) |
| **Work orders** | 1670 (совпадает с published) |
| **Repair share** | 54.2% |
| **Maintenance share** | 45.8% |
| **Severity GLM R²** | 97.5% (internal consistency, NOT real-world fit) |
| **Power effect** | 0.0062 (log-log elasticity) |
| **Артефакт** | `reports/hadco_service_audit.json` |

### Ограничения

1. R² = 97.5% отражает self-consistency реконструкции, не empirical fit.
2. Repair/maintenance классификация HADCO не совпадает с production major/minor.
3. НЕ используется для обучения production модели.

---

## A-05: Interaction Age × Hours

| Поле | Значение |
|---|---|
| **Статус** | 🟢 CLOSED (LR test + production run) |
| **LR statistic** | 29.19 (df=1) |
| **p-value** | < 1e-9 |
| **β_interaction** | +0.173 ± 0.027 |
| **AIC full** | 15236.5 |
| **AIC restricted** | 15263.7 |
| **Penalized** | False |

### Интерпретация

Interaction Age × Hours статистически значим. Коэффициент β=0.173
означает синергетический эффект: старые машины с высокой наработкой
имеют повышенный риск отказа сверх аддитивного эффекта возраста и наработки.

---

## A-06: Cluster-robust inference (32 clusters)

| Поле | Значение |
|---|---|
| **Статус** | 🟢 CLOSED (5/5 regression tests) |
| **n_clusters** | 32 (Region × Year × Campaign) |
| **F_stat_cluster_robust** | 59930 |
| **π_Z (cluster-robust SE)** | 0.3237 ± 0.0013 |
| **Point estimates** | Идентичны naive (кластеризация влияет только на SE) |
| **SE direction** | Не гарантировано > naive (зависит от внутрикластерной корреляции) |

### Ограничение

32 кластера < 50 (рекомендация Stock & Yogo для полностью надёжного
cluster-robust inference). Результаты интерпретируются консервативно.

---

## A-07: IV mode = predictive

| Поле | Значение |
|---|---|
| **Статус** | 🟡 ACTIVE |
| **Текущий режим** | `predictive` |
| **Интерпретация γ** | Предсказательный, не каузальный |
| **Условие для causal** | Валидный инструмент на реальных данных + MC recovery |

### Обоснование

Инструмент Z (rainfall anomaly) адекватен на симуляции
(F_cluster = 59930, exclusion restriction пройден),
но production-инструмент не подтверждён на реальных данных.
До получения реальных claims γ интерпретируется как predictive.

---

## A-08: PeakLoad calibration (TUM)

| Поле | Значение |
|---|---|
| **Статус** | 🟢 CLOSED (Level 2 data) |
| **Источник** | TUM CAN bus operational telemetry |
| **Mean** | 0.7099 |
| **Std** | 0.2053 |
| **Применение** | Стандартизация PeakLoad в DGP и prediction |
| **Уровень** | Level 2 (published individual records) |
