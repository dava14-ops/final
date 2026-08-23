# Data Contract: Claims Dataset v1.0

> **Версия**: 1.0
> **Дата**: 2026-08-15
> **Статус**: pre-production (пилот)
> **Связанные документы**: `docs/data_sources.md`, `docs/ethics_checklist.md`, `docs/model_card.md`

## 1. Назначение

Определяет структуру, типы и правила валидации claims-датасета,
используемого для переобучения модели CF Cox / IV-Cox в Фазе 7.

## 2. Единицы времени

| Параметр | Значение |
|---|---|
| Единица времени модели | `engine_hours` (мото-часы) |
| Горизонт калибровки | 1712 мото-часов (214 дней × 8 мч/день) |
| Конверсия | 8 мото-часов = 1 календарный день |

Все временные поля (`failure_time`, `hours_at_event`, `downtime_hours`)
указываются в **мото-часах**, если не оговорено иное.

## 3. Обязательные поля

| Поле | Тип | Описание | Валидация |
|---|---|---|---|
| `machine_id` | str | Уникальный обезличенный ID машины | non-empty, unique |
| `brand` | str | Марка трактора | из BRAND_MAP (см. §6) |
| `power_hp` | float | Мощность двигателя, л.с. | [50, 500] |
| `production_year` | int | Год выпуска | [1990, 2025] |
| `age_at_event` | float | Возраст на момент события, лет | [0, 30] |
| `hours_at_event` | float | Наработка на момент события, мч | [0, 50000] |
| `failure_time` | float | Время до события/цензуры, мч | > 0 |
| `event_flag` | int | 1 = событие, 0 = цензура | {0, 1} |
| `failure_system` | str | Система отказа | из FREQ_SHARES (см. §6) |
| `major_failure_flag` | int | 1 = major, 0 = minor | {0, 1} |

## 4. Опциональные поля

| Поле | Тип | Описание |
|---|---|---|
| `model` | str | Модель трактора |
| `region` | str | Регион (из regions_mis) |
| `segment` | str | Сегмент мощности: light / heavy |
| `season` | str | Сезон работ |
| `peak_load_proxy` | float | Прокси PeakLoad из телеметрии [0, 1] |
| `climate_index` | float | Индекс климата [0, 1] |
| `soil_index` | float | Индекс почвы [0, 1] |
| `repair_cost` | float | Стоимость ремонта, руб. |
| `downtime_hours` | float | Простой, мч |
| `claim_amount` | float | Сумма выплаты, руб. |
| `deductible` | float | Франшиза, руб. |
| `coverage_limit` | float | Лимит покрытия, руб. |
| `event_definition` | str | total_loss / major_claim / any_failure |
| `working_days_window` | float | Погодный инструмент (Фаза 5) |
| `maintenance_history` | str | JSON с историей ТО |

## 5. Правила валидации

1. Все обязательные поля должны быть заполнены.
2. `failure_time > 0` для всех строк.
3. Если `event_flag = 1`, то `failure_time` — время до события.
4. Если `event_flag = 0`, то `failure_time` — время цензуры.
5. `event_flag` и `major_failure_flag` бинарные.
6. `brand` нормализуется к каноническим именам из BRAND_MAP.
7. `failure_system` нормализуется к ключам FREQ_SHARES.
8. Дубликаты по (`machine_id`, `failure_time`, `event_flag`) удаляются.
9. Строки с NaN в обязательных полях удаляются.
10. Отрицательные значения времени/стоимости не допускаются.

## 6. Справочники (из constants.py)

### 6.1 Бренды (BRAND_MAP)

| Код | Каноническое имя |
|---|---|
| 0 | MTZ82 |
| 1 | Versatile280 |
| 2 | NewHollandT9 |
| 3 | DT75 |
| 4 | Other |

### 6.2 Системы отказов (FREQ_SHARES)

| Система | Доля частоты |
|---|---|
| гидравлика | 0.30 |
| электроника | 0.30 |
| двигатель | 0.12 |
| трансмиссия | 0.20 |
| прочее | 0.08 |

### 6.3 Определения события (VALID_EVENT_DEFINITIONS)

- `total_loss` — полная гибель
- `major_claim` — крупный страховой случай
- `any_failure` — любой отказ

### 6.4 Сегменты мощности (SEGMENTS)

- `light` — лёгкий парк
- `heavy` — мощный парк РФ

## 7. Формат файла

- CSV, разделитель запятая, кодировка UTF-8.
- Первая строка — заголовки.
- Десятичный разделитель — точка.
- Путь по умолчанию: `data/raw/claims/claims_pilot_v1.csv`.

## 8. Требования к покрытию (для Фазы 7)

| Метрика | Минимум |
|---|---|
| Уникальных машин | ≥ 100 |
| Событий (отказов) | ≥ 200 |
| Горизонт наблюдения | ≥ 1712 мч |
| Брендов с ≥ 30 событиями | ≥ 3 |

## 9. Пример строки

```csv
machine_id,brand,power_hp,production_year,age_at_event,hours_at_event,failure_time,event_flag,failure_system,major_failure_flag
m_0001,MTZ82,82,2010,5.0,4500.0,1650.0,1,гидравлика,0
m_0002,NewHollandT9,340,2015,1.0,1200.0,1712.0,0,прочее,0
```
