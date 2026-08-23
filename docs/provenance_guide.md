# Data Provenance Guide

Руководство по уровням происхождения данных в проекте.

---

## Определение уровней

### Level 0: Synthetic

```python
provenance = DataProvenance.SYNTHETIC
```

Данные, сгенерированные нашим DGP (Data Generating Process).
Используются для Monte Carlo валидации estimator.

**Примеры:** MC recovery replications, hybrid training data.

**Что можно заявлять:** "Estimator validated on simulated data with known ground truth."

---

### Level 1: Literature Reconstructed

```python
provenance = DataProvenance.LITERATURE_RECONSTRUCTED
```

Опубликованные агрегаты, воспроизведённые через параметрическую реконструкцию.
Индивидуальные наблюдения синтетические, но статистические свойства
соответствуют опубликованным.

**Примеры:** Zetor (Durczak et al. 2018), HADCO (Al-Suhaibani & Wahby).

**Что можно заявлять:** "External benchmark consistent with published literature."

**Что НЕЛЬЗЯ заявлять:** "Validated on real individual-level data."

---

### Level 2: Published Individual

```python
provenance = DataProvenance.PUBLISHED_INDIVIDUAL
```

Реальные индивидуальные записи из опубликованных источников.

**Примеры:** TUM PeakLoad (CAN bus telemetry).

**Что можно заявлять:** "Calibrated on real operational data."

---

### Level 3: Proprietary Claims

```python
provenance = DataProvenance.PROPRIETARY_CLAIMS
```

Реальные данные от OEM / dealer / insurer / fleet operator.

**Примеры:** (пока отсутствуют)

**Что можно заявлять:** "Empirically validated on real insurance claims."

---

## Правила использования

| Уровень | Обучение production модели | Внешний бенчмарк | Публикация как "real data" |
|---|---|---|---|
| Level 0 | ✅ (hybrid mode) | — | ❌ |
| Level 1 | ❌ | ✅ | ❌ |
| Level 2 | ✅ (калибровка) | ✅ | ⚠️ С оговорками |
| Level 3 | ✅ | ✅ | ✅ |

---

## Проверка провенанса в коде

```python
from benchmarks.canonical_schema import DataProvenance

ds = load_zetor_dataset()
report = ds.provenance_report()

assert report["provenance"] == "literature_reconstructed"
assert report["provenance_level"] == 1
assert report["is_real_claims"] == False
assert report["is_individual_level"] == False
```

---

## Защита от misuse

Для HADCO (Level 1, recurrent events):

```python
ds = load_hadco_dataset()

# Заблокировано:
ds.to_survival_dataframe()  # ValueError: not survival-compatible

# Разрешено:
to_recurrent_event_dataframe(ds)  # OK
to_cost_dataframe(ds)             # OK
```
