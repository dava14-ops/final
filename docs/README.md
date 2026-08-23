# External Benchmarks

Внешние эмпирические бенчмарки для валидации статистических механизмов
CF-Cox / IV-Cox модели.

---

## ⚠️ Критическое ограничение

**Ни один из этих датасетов НЕ используется для обучения production модели.**

Они используются **только** как external sanity checks:
- Zetor → "Похожа ли наша survival shape на реальную?"
- HADCO → "Похожа ли наша cost/severity структура на реальную?"

---

## Data Provenance Levels

| Level | Название | Описание | Пример |
|---|---|---|---|
| 0 | `synthetic` | DGP / Monte Carlo | Наш DGP |
| 1 | `literature_reconstructed` | Published aggregates, synthetic reconstruction | **Zetor, HADCO** |
| 2 | `published_individual` | Individual records из published tables | TUM PeakLoad |
| 3 | `proprietary_claims` | OEM / dealer / insurer real data | (будущее) |

**Только Level 3 может называться "real claims dataset" в manuscript.**

---

## Zetor Benchmark

**Source:** Durczak W., Ekielski A., Żelaziński R. (2018).
"Analysis of tractor failures depending on their age and time of use."

**Published cohort:** 70 тракторов Zetor

**Published outcome:** first failure

**Published MTTF:** 271 moto-hours

**Current implementation:** literature-reconstructed synthetic individual observations

**Censoring:** not available (все 70 — observed failures)

**Use:** external survival-shape benchmark

**NOT used for:**
- causal estimation
- IV estimation
- production parameter estimation
- PeakLoad γ recovery

**Key result:** Weibull shape deviation = **1.8%** (Zetor 1.91 vs DGP 1.88)

---

## HADCO Benchmark

**Source:** Al-Suhaibani A., Wahby A.
"Farm tractors breakdown classification."

**Published cohort:** 40 тракторов HADCO, 1670 work-job orders (1988–1993)

**Published outcome:** repair/maintenance work orders

**Current implementation:** literature-reconstructed synthetic work orders

**Event-level engine hours:** NOT available (только mean annual hours)

**Use:** recurrent-event / cost-severity benchmark

**NOT used for:**
- Cox time-to-failure analysis
- survival shape estimation
- production parameter estimation

**Key result:** Repair share 54.2%, severity GLM R²=97.5% (internal consistency)

---

## Структура модуля

```text
benchmarks/
├── __init__.py
├── canonical_schema.py       # Pydantic-схемы + DataProvenance enum
├── adapters/
│   ├── __init__.py
│   ├── zetor_adapter.py      # Survival benchmark adapter
│   └── hadco_adapter.py      # Recurrent/cost benchmark adapter
├── zetor_benchmark.py        # Survival shape audit
├── hadco_benchmark.py        # Recurrent/cost audit
└── README.md                 # Этот файл
```

---

## Запуск бенчмарков

```bash
# Zetor survival benchmark
python benchmarks/zetor_benchmark.py --seed 42

# HADCO recurrent/cost benchmark
python benchmarks/hadco_benchmark.py --seed 42

# Тесты
pytest tests/test_zetor_adapter.py tests/test_hadco_adapter.py -v
```

---

## Результаты (сохранённые артефакты)

- `reports/zetor_survival_audit.json` — Zetor survival benchmark
- `reports/hadco_service_audit.json` — HADCO service benchmark
