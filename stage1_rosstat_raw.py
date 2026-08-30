#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage1_rosstat_raw.py

Детерминированный парсер официального файла Росстата
`cena_sx_07-2026.xlsx` (609 405 байт).

Архитектура:
    Workbook
    ├── sheet 1.1   → annual national 2010–2025 (годовое среднее)
    ├── sheet 1.2   → annual national 2010–2025 (на конец года)
    ├── sheet 2     → monthly national 2021
    ├── sheet 3     → monthly national 2022
    ├── sheet 4     → monthly national 2023
    ├── sheet 5     → monthly national 2024
    ├── sheet 6.1   → monthly RF + ФО + субъекты 2025
    ├── sheet 6.2   → annual RF + ФО + субъекты 2025
    └── sheet 7.1   → monthly RF + ФО + субъекты 2026 (янв–июль)

Принципы:
    1. Год восстанавливается из имени листа (не из строки данных)
    2. `...` = конфиденциальность, `-` = отсутствие данных
    3. Полный lineage до координаты Excel
    4. Никакого heuristic matching — только детерминированный парсинг

Использование:
    python stage1_rosstat_raw.py \
        --input cena_sx_07-2026.xlsx \
        --output rosstat_prices_raw_long.csv \
        --inventory crop_inventory.csv
"""

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Метаданные листов
# ---------------------------------------------------------------------------

# Маппинг имени листа → метаданные
SHEET_METADATA: Dict[str, dict] = {
    "1.1": {
        "year": None,  # Годы в колонках
        "frequency": "annual",
        "price_timing": "annual_average",
        "geography": "national",
        "description": "Годовые средние цены, РФ, 2010–2025",
    },
    "1.2": {
        "year": None,  # Годы в колонках
        "frequency": "annual",
        "price_timing": "end_of_year",
        "geography": "national",
        "description": "Цены на конец года, РФ, 2010–2025",
    },
    "2": {
        "year": 2021,
        "frequency": "monthly",
        "price_timing": "end_of_period",
        "geography": "national",
        "description": "Месячные цены, РФ, 2021",
    },
    "3": {
        "year": 2022,
        "frequency": "monthly",
        "price_timing": "end_of_period",
        "geography": "national",
        "description": "Месячные цены, РФ, 2022",
    },
    "4": {
        "year": 2023,
        "frequency": "monthly",
        "price_timing": "end_of_period",
        "geography": "national",
        "description": "Месячные цены, РФ, 2023",
    },
    "5": {
        "year": 2024,
        "frequency": "monthly",
        "price_timing": "end_of_period",
        "geography": "national",
        "description": "Месячные цены, РФ, 2024",
    },
    "6.1": {
        "year": 2025,
        "frequency": "monthly",
        "price_timing": "end_of_period",
        "geography": "mixed",  # РФ + ФО + субъекты
        "description": "Месячные цены, РФ + ФО + субъекты, 2025",
    },
    "6.2": {
        "year": 2025,
        "frequency": "annual",
        "price_timing": "annual_average",
        "geography": "mixed",  # РФ + ФО + субъекты
        "description": "Годовые цены, РФ + ФО + субъекты, 2025",
    },
    "7.1": {
        "year": 2026,
        "frequency": "monthly",
        "price_timing": "end_of_period",
        "geography": "mixed",  # РФ + ФО + субъекты
        "description": "Месячные цены, РФ + ФО + субъекты, 2026 (янв–июль)",
    },
}

# Названия месяцев для маппинга колонок
MONTH_NAMES = {
    "январь": 1, "февраль": 2, "март": 3, "апрель": 4,
    "май": 5, "июнь": 6, "июль": 7, "август": 8,
    "сентябрь": 9, "октябрь": 10, "ноябрь": 11, "декабрь": 12,
}

# Паттерны для определения типа территории
PATTERNS_TERRITORY = {
    "national": re.compile(r"Российская Федерация", re.IGNORECASE),
    "federal_district": re.compile(
        r"(Центральный|Северо-Западный|Южный|Северо-Кавказский|"
        r"Приволжский|Уральский|Сибирский|Дальневосточный)\s+"
        r"федеральный округ",
        re.IGNORECASE,
    ),
    # Всё остальное — субъект РФ (области, края, республики)
}


# ---------------------------------------------------------------------------
# Структура записи
# ---------------------------------------------------------------------------

@dataclass
class RawPriceRecord:
    """Одна ячейка данных с полным lineage."""
    source_file: str
    source_sheet: str
    source_row_excel: int      # Номер строки в Excel (1-based)
    source_column_excel: str   # Буква колонки в Excel
    year: Optional[int]
    month: Optional[int]
    frequency: str             # annual / monthly
    geography_level: str       # national / federal_district / subject
    region_name: str
    region_source_code: Optional[str]
    product_name_raw: str
    okpd2_raw: Optional[str]
    raw_value: str             # Исходная строка из ячейки
    value_numeric: Optional[float]
    value_status: str          # observed / confidential / unavailable / structurally_missing
    price_timing: str          # annual_average / end_of_period / end_of_year
    unit_raw: str


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def classify_value(raw: str) -> Tuple[Optional[float], str]:
    """
    Классифицировать значение ячейки.

    Возвращает:
        (value_numeric, value_status)

    Статусы:
        - observed: числовое значение
        - confidential: `...` (конфиденциальность)
        - unavailable: `-` (данных не имеется)
        - structurally_missing: пустая ячейка или не парсится
    """
    raw = raw.strip() if raw else ""

    if raw == "":
        return None, "structurally_missing"

    if raw == "...":
        return None, "confidential"

    if raw == "-":
        return None, "unavailable"

    # Попытка распарсить число
    # Росстат использует пробел как разделитель тысяч и запятую как десятичный
    cleaned = raw.replace(" ", "").replace("\u00a0", "").replace(",", ".")

    try:
        value = float(cleaned)
        return value, "observed"
    except ValueError:
        return None, "structurally_missing"


def classify_territory(name: str) -> str:
    """
    Определить уровень территории по названию.

    Возвращает:
        - "national"
        - "federal_district"
        - "subject"
    """
    name = name.strip()

    if PATTERNS_TERRITORY["national"].search(name):
        return "national"

    if PATTERNS_TERRITORY["federal_district"].search(name):
        return "federal_district"

    return "subject"


def extract_okpd2(text: str) -> Optional[str]:
    """
    Извлечь ОКПД2 код из текста.

    Примеры:
        "Пшеница 01.11.1" → "01.11.1"
        "Кукуруза 01.11.20" → "01.11.20"
        "Пшеница мягкая 3 класса 01.11.12.006.АГ" → "01.11.12.006.АГ"
    """
    # Паттерн ОКПД2: две цифры, точка, две цифры, далее опционально
    # .цифры.цифры.АГ или .цифры
    pattern = re.compile(
        r"(\d{2}\.\d{2}(?:\.\d{1,3})?(?:\.\d{3})?(?:\.АГ)?)"
    )
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None


def extract_product_name(text: str) -> str:
    """
    Извлечь название продукции, удалив ОКПД2 код.

    "Пшеница 01.11.1" → "Пшеница"
    """
    okpd2 = extract_okpd2(text)
    if okpd2:
        name = text.replace(okpd2, "").strip()
        return name
    return text.strip()


def col_index_to_letter(col_idx: int) -> str:
    """
    Преобразовать индекс колонки (0-based) в букву Excel.

    0 → 'A', 1 → 'B', ..., 25 → 'Z', 26 → 'AA', ...
    """
    result = ""
    idx = col_idx
    while True:
        result = chr(65 + idx % 26) + result
        idx = idx // 26 - 1
        if idx < 0:
            break
    return result


# ---------------------------------------------------------------------------
# Парсеры для разных типов листов
# ---------------------------------------------------------------------------

def parse_annual_national_sheet(
    sheet_name: str,
    df_raw: pd.DataFrame,
    metadata: dict,
    source_file: str,
) -> List[RawPriceRecord]:
    """
    Парсер для листов 1.1 и 1.2 (годовые национальные цены).

    Структура:
        - Строки: культуры (название + ОКПД2)
        - Колонки: годы (2010, 2011, ..., 2025)
    """
    logger = logging.getLogger(__name__)
    records: List[RawPriceRecord] = []

    logger.info(f"Парсинг листа '{sheet_name}': annual national")

    # Ищем строку с заголовками (где есть числовые годы)
    header_row_idx = None
    for i, row in df_raw.iterrows():
        row_values = [str(v).strip() for v in row.values if pd.notna(v)]
        # Ищем строку, где есть хотя бы два четырёхзначных числа (годы)
        years_found = [v for v in row_values if re.match(r"^20\d{2}$", v)]
        if len(years_found) >= 2:
            header_row_idx = i
            break

    if header_row_idx is None:
        logger.warning(f"Лист '{sheet_name}': не найдена строка с годами")
        return records

    # Извлекаем годы из заголовка
    header_row = df_raw.iloc[header_row_idx]
    year_columns: Dict[int, int] = {}  # col_index → year

    for col_idx, cell_value in enumerate(header_row):
        if pd.notna(cell_value):
            cell_str = str(cell_value).strip()
            if re.match(r"^20\d{2}$", cell_str):
                year_columns[col_idx] = int(cell_str)

    logger.info(
        f"Лист '{sheet_name}': найдено {len(year_columns)} годовых колонок, "
        f"диапазон {min(year_columns.values())}–{max(year_columns.values())}"
    )

    # Парсим строки данных
    for row_idx in range(header_row_idx + 1, len(df_raw)):
        row = df_raw.iloc[row_idx]

        # Первая колонка — название продукции (возможно с ОКПД2)
        product_cell = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""

        if not product_cell or product_cell == "nan":
            continue

        # Пропускаем служебные строки
        if product_cell.startswith("Источник") or product_cell.startswith("Примечание"):
            continue

        okpd2 = extract_okpd2(product_cell)
        product_name = extract_product_name(product_cell)

        # Парсим значения по годам
        for col_idx, year in year_columns.items():
            if col_idx < len(row):
                raw_value = str(row.iloc[col_idx]).strip() if pd.notna(row.iloc[col_idx]) else ""
            else:
                raw_value = ""

            value_numeric, value_status = classify_value(raw_value)

            records.append(RawPriceRecord(
                source_file=source_file,
                source_sheet=sheet_name,
                source_row_excel=row_idx + 1,  # 1-based для Excel
                source_column_excel=col_index_to_letter(col_idx),
                year=year,
                month=None,
                frequency="annual",
                geography_level="national",
                region_name="Российская Федерация",
                region_source_code=None,
                product_name_raw=product_name,
                okpd2_raw=okpd2,
                raw_value=raw_value,
                value_numeric=value_numeric,
                value_status=value_status,
                price_timing=metadata["price_timing"],
                unit_raw="RUB per tonne",
            ))

    logger.info(f"Лист '{sheet_name}': извлечено {len(records)} записей")
    return records


def parse_monthly_national_sheet(
    sheet_name: str,
    df_raw: pd.DataFrame,
    metadata: dict,
    source_file: str,
) -> List[RawPriceRecord]:
    """
    Парсер для листов 2–5 (месячные национальные цены).

    Структура:
        - Строки: культуры (название + ОКПД2)
        - Колонки: месяцы (Январь, Февраль, ..., Декабрь)
    """
    logger = logging.getLogger(__name__)
    records: List[RawPriceRecord] = []
    year = metadata["year"]

    logger.info(f"Парсинг листа '{sheet_name}': monthly national, year={year}")

    # Ищем строку с заголовками месяцев
    header_row_idx = None
    for i, row in df_raw.iterrows():
        row_values = [str(v).strip().lower() for v in row.values if pd.notna(v)]
        months_found = [v for v in row_values if v in MONTH_NAMES]
        if len(months_found) >= 2:
            header_row_idx = i
            break

    if header_row_idx is None:
        logger.warning(f"Лист '{sheet_name}': не найдена строка с месяцами")
        return records

    # Извлекаем месяцы из заголовка
    header_row = df_raw.iloc[header_row_idx]
    month_columns: Dict[int, int] = {}  # col_index → month

    for col_idx, cell_value in enumerate(header_row):
        if pd.notna(cell_value):
            cell_str = str(cell_value).strip().lower()
            if cell_str in MONTH_NAMES:
                month_columns[col_idx] = MONTH_NAMES[cell_str]

    logger.info(
        f"Лист '{sheet_name}': найдено {len(month_columns)} месячных колонок"
    )

    # Парсим строки данных
    for row_idx in range(header_row_idx + 1, len(df_raw)):
        row = df_raw.iloc[row_idx]

        # Первая колонка — название продукции
        product_cell = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""

        if not product_cell or product_cell == "nan":
            continue

        if product_cell.startswith("Источник") or product_cell.startswith("Примечание"):
            continue

        okpd2 = extract_okpd2(product_cell)
        product_name = extract_product_name(product_cell)

        # Парсим значения по месяцам
        for col_idx, month in month_columns.items():
            if col_idx < len(row):
                raw_value = str(row.iloc[col_idx]).strip() if pd.notna(row.iloc[col_idx]) else ""
            else:
                raw_value = ""

            value_numeric, value_status = classify_value(raw_value)

            records.append(RawPriceRecord(
                source_file=source_file,
                source_sheet=sheet_name,
                source_row_excel=row_idx + 1,
                source_column_excel=col_index_to_letter(col_idx),
                year=year,
                month=month,
                frequency="monthly",
                geography_level="national",
                region_name="Российская Федерация",
                region_source_code=None,
                product_name_raw=product_name,
                okpd2_raw=okpd2,
                raw_value=raw_value,
                value_numeric=value_numeric,
                value_status=value_status,
                price_timing=metadata["price_timing"],
                unit_raw="RUB per tonne",
            ))

    logger.info(f"Лист '{sheet_name}': извлечено {len(records)} записей")
    return records


def parse_monthly_mixed_sheet(
    sheet_name: str,
    df_raw: pd.DataFrame,
    metadata: dict,
    source_file: str,
) -> List[RawPriceRecord]:
    """
    Парсер для листов 6.1 и 7.1 (месячные цены, РФ + ФО + субъекты).

    Структура:
        - Строки: территории × культуры
        - Колонки: месяцы
    """
    logger = logging.getLogger(__name__)
    records: List[RawPriceRecord] = []
    year = metadata["year"]

    logger.info(f"Парсинг листа '{sheet_name}': monthly mixed, year={year}")

    # Ищем строку с заголовками месяцев
    header_row_idx = None
    for i, row in df_raw.iterrows():
        row_values = [str(v).strip().lower() for v in row.values if pd.notna(v)]
        months_found = [v for v in row_values if v in MONTH_NAMES]
        if len(months_found) >= 2:
            header_row_idx = i
            break

    if header_row_idx is None:
        logger.warning(f"Лист '{sheet_name}': не найдена строка с месяцами")
        return records

    # Извлекаем месяцы из заголовка
    header_row = df_raw.iloc[header_row_idx]
    month_columns: Dict[int, int] = {}

    for col_idx, cell_value in enumerate(header_row):
        if pd.notna(cell_value):
            cell_str = str(cell_value).strip().lower()
            if cell_str in MONTH_NAMES:
                month_columns[col_idx] = MONTH_NAMES[cell_str]

    logger.info(
        f"Лист '{sheet_name}': найдено {len(month_columns)} месячных колонок"
    )

    # Парсим строки данных
    # В листах 6.1 и 7.1 структура может быть:
    # Колонка 0: территория
    # Колонка 1: культура (или территория + культура в одной колонке)
    # Далее: значения по месяцам

    current_territory = ""

    for row_idx in range(header_row_idx + 1, len(df_raw)):
        row = df_raw.iloc[row_idx]

        # Определяем территорию и культуру
        # Это зависит от конкретной структуры листа
        # Вариант 1: колонка 0 = территория, колонка 1 = культура
        # Вариант 2: колонка 0 = территория + культура

        col0 = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        col1 = str(row.iloc[1]).strip() if len(row) > 1 and pd.notna(row.iloc[1]) else ""

        if not col0 or col0 == "nan":
            continue

        if col0.startswith("Источник") or col0.startswith("Примечание"):
            continue

        # Определяем, является ли col0 территорией или культурой
        territory_level = classify_territory(col0)

        if territory_level in ("national", "federal_district"):
            # Это строка территории — запоминаем и переходим к следующей
            current_territory = col0
            continue

        # Если col0 выглядит как территория-субъект
        if territory_level == "subject" and not extract_okpd2(col0):
            # Проверяем, есть ли культура в col1
            if col1 and extract_okpd2(col1):
                current_territory = col0
                product_cell = col1
                data_start_col = 2
            else:
                # col0 может содержать и территорию, и культуру
                current_territory = col0
                product_cell = col0
                data_start_col = 1
        else:
            # col0 — это культура (территория была в предыдущей строке)
            product_cell = col0
            data_start_col = 1

        okpd2 = extract_okpd2(product_cell)
        product_name = extract_product_name(product_cell)

        if not current_territory:
            current_territory = "Российская Федерация"

        geo_level = classify_territory(current_territory)

        # Парсим значения по месяцам
        for col_idx, month in month_columns.items():
            if col_idx < len(row):
                raw_value = str(row.iloc[col_idx]).strip() if pd.notna(row.iloc[col_idx]) else ""
            else:
                raw_value = ""

            value_numeric, value_status = classify_value(raw_value)

            records.append(RawPriceRecord(
                source_file=source_file,
                source_sheet=sheet_name,
                source_row_excel=row_idx + 1,
                source_column_excel=col_index_to_letter(col_idx),
                year=year,
                month=month,
                frequency="monthly",
                geography_level=geo_level,
                region_name=current_territory,
                region_source_code=None,
                product_name_raw=product_name,
                okpd2_raw=okpd2,
                raw_value=raw_value,
                value_numeric=value_numeric,
                value_status=value_status,
                price_timing=metadata["price_timing"],
                unit_raw="RUB per tonne",
            ))

    logger.info(f"Лист '{sheet_name}': извлечено {len(records)} записей")
    return records


def parse_annual_mixed_sheet(
    sheet_name: str,
    df_raw: pd.DataFrame,
    metadata: dict,
    source_file: str,
) -> List[RawPriceRecord]:
    """
    Парсер для листа 6.2 (годовые цены, РФ + ФО + субъекты).

    Структура аналогична 6.1, но колонки — годы.
    """
    logger = logging.getLogger(__name__)
    records: List[RawPriceRecord] = []
    year = metadata["year"]

    logger.info(f"Парсинг листа '{sheet_name}': annual mixed, year={year}")

    # Для годового листа колонки могут содержать только один год
    # или несколько лет. Парсим аналогично annual_national,
    # но с учётом территорий.

    # Ищем строку с заголовками
    header_row_idx = None
    for i, row in df_raw.iterrows():
        row_values = [str(v).strip() for v in row.values if pd.notna(v)]
        # Ищем годы или слово "год"
        years_found = [v for v in row_values if re.match(r"^20\d{2}$", v)]
        if len(years_found) >= 1 or any("год" in v.lower() for v in row_values):
            header_row_idx = i
            break

    if header_row_idx is None:
        # Если заголовок не найден, используем фиксированный год из метаданных
        logger.warning(
            f"Лист '{sheet_name}': заголовок не найден, "
            f"используем фиксированный год {year}"
        )
        header_row_idx = 0
        year_columns = {1: year}  # Предполагаем, что данные в колонке 1
    else:
        header_row = df_raw.iloc[header_row_idx]
        year_columns: Dict[int, int] = {}

        for col_idx, cell_value in enumerate(header_row):
            if pd.notna(cell_value):
                cell_str = str(cell_value).strip()
                if re.match(r"^20\d{2}$", cell_str):
                    year_columns[col_idx] = int(cell_str)

        if not year_columns:
            year_columns = {1: year}

    # Парсим строки данных
    current_territory = ""

    for row_idx in range(header_row_idx + 1, len(df_raw)):
        row = df_raw.iloc[row_idx]

        col0 = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""

        if not col0 or col0 == "nan":
            continue

        if col0.startswith("Источник") or col0.startswith("Примечание"):
            continue

        territory_level = classify_territory(col0)

        if territory_level in ("national", "federal_district") and not extract_okpd2(col0):
            current_territory = col0
            continue

        product_cell = col0
        okpd2 = extract_okpd2(product_cell)
        product_name = extract_product_name(product_cell)

        if not current_territory:
            current_territory = "Российская Федерация"

        geo_level = classify_territory(current_territory)

        for col_idx, yr in year_columns.items():
            if col_idx < len(row):
                raw_value = str(row.iloc[col_idx]).strip() if pd.notna(row.iloc[col_idx]) else ""
            else:
                raw_value = ""

            value_numeric, value_status = classify_value(raw_value)

            records.append(RawPriceRecord(
                source_file=source_file,
                source_sheet=sheet_name,
                source_row_excel=row_idx + 1,
                source_column_excel=col_index_to_letter(col_idx),
                year=yr,
                month=None,
                frequency="annual",
                geography_level=geo_level,
                region_name=current_territory,
                region_source_code=None,
                product_name_raw=product_name,
                okpd2_raw=okpd2,
                raw_value=raw_value,
                value_numeric=value_numeric,
                value_status=value_status,
                price_timing=metadata["price_timing"],
                unit_raw="RUB per tonne",
            ))

    logger.info(f"Лист '{sheet_name}': извлечено {len(records)} записей")
    return records


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------

def parse_workbook(input_path: Path) -> List[RawPriceRecord]:
    """
    Парсить весь workbook.
    """
    logger = logging.getLogger(__name__)
    source_file = input_path.name

    logger.info(f"Загрузка файла: {input_path}")
    logger.info(f"Размер: {input_path.stat().st_size:,} байт")

    xls = pd.ExcelFile(input_path)
    logger.info(f"Листы в файле: {xls.sheet_names}")

    all_records: List[RawPriceRecord] = []

    for sheet_name in xls.sheet_names:
        # Пропускаем служебные листы
        if sheet_name.lower() in ("содержание", "оглавление", "contents"):
            logger.info(f"Пропуск служебного листа: '{sheet_name}'")
            continue

        # Получаем метаданные листа
        metadata = SHEET_METADATA.get(sheet_name)
        if metadata is None:
            logger.warning(f"Неизвестный лист: '{sheet_name}', пропускаем")
            continue

        logger.info(f"Обработка листа '{sheet_name}': {metadata['description']}")

        # Читаем лист без заголовка (header=None)
        # чтобы сохранить исходную структуру
        df_raw = pd.read_excel(
            xls,
            sheet_name=sheet_name,
            header=None,
            dtype=str,  # Всё как строки для сохранения `...` и `-`
        )

        # Выбираем парсер в зависимости от типа листа
        if sheet_name in ("1.1", "1.2"):
            records = parse_annual_national_sheet(
                sheet_name, df_raw, metadata, source_file
            )
        elif sheet_name in ("2", "3", "4", "5"):
            records = parse_monthly_national_sheet(
                sheet_name, df_raw, metadata, source_file
            )
        elif sheet_name in ("6.1", "7.1"):
            records = parse_monthly_mixed_sheet(
                sheet_name, df_raw, metadata, source_file
            )
        elif sheet_name == "6.2":
            records = parse_annual_mixed_sheet(
                sheet_name, df_raw, metadata, source_file
            )
        else:
            logger.warning(f"Неизвестный тип листа: '{sheet_name}'")
            continue

        all_records.extend(records)

    logger.info(f"Всего извлечено записей: {len(all_records):,}")
    return all_records


def records_to_dataframe(records: List[RawPriceRecord]) -> pd.DataFrame:
    """
    Преобразовать записи в DataFrame.
    """
    data = []
    for r in records:
        data.append({
            "source_file": r.source_file,
            "source_sheet": r.source_sheet,
            "source_row_excel": r.source_row_excel,
            "source_column_excel": r.source_column_excel,
            "year": r.year,
            "month": r.month,
            "frequency": r.frequency,
            "geography_level": r.geography_level,
            "region_name": r.region_name,
            "region_source_code": r.region_source_code,
            "product_name_raw": r.product_name_raw,
            "okpd2_raw": r.okpd2_raw,
            "raw_value": r.raw_value,
            "value_numeric": r.value_numeric,
            "value_status": r.value_status,
            "price_timing": r.price_timing,
            "unit_raw": r.unit_raw,
        })

    df = pd.DataFrame(data)
    return df


def generate_crop_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Сгенерировать инвентарь культур.

    Для каждой культуры (ОКПД2):
        - Первое и последнее наблюдение
        - Наличие месячных/годовых данных
        - Наличие национальных/региональных данных
    """
    logger = logging.getLogger(__name__)
    logger.info("Генерация инвентаря культур...")

    inventory = []

    # Группируем по ОКПД2 (или по названию, если ОКПД2 нет)
    df_valid = df[df["okpd2_raw"].notna() | df["product_name_raw"].notna()]

    for (okpd2, product_name), group in df_valid.groupby(
        ["okpd2_raw", "product_name_raw"], dropna=False
    ):
        # Временной диапазон
        years = group["year"].dropna()
        year_min = int(years.min()) if len(years) > 0 else None
        year_max = int(years.max()) if len(years) > 0 else None

        # Типы данных
        has_monthly = (group["frequency"] == "monthly").any()
        has_annual = (group["frequency"] == "annual").any()

        # Уровни географии
        has_national = (group["geography_level"] == "national").any()
        has_regional = (group["geography_level"].isin(["federal_district", "subject"])).any()

        # Статусы значений
        n_observed = (group["value_status"] == "observed").sum()
        n_confidential = (group["value_status"] == "confidential").sum()
        n_unavailable = (group["value_status"] == "unavailable").sum()
        n_missing = (group["value_status"] == "structurally_missing").sum()

        inventory.append({
            "okpd2_raw": okpd2,
            "product_name_raw": product_name,
            "first_year": year_min,
            "last_year": year_max,
            "monthly_available": has_monthly,
            "annual_available": has_annual,
            "national_available": has_national,
            "regional_available": has_regional,
            "n_observations": len(group),
            "n_observed_values": n_observed,
            "n_confidential": n_confidential,
            "n_unavailable": n_unavailable,
            "n_structurally_missing": n_missing,
        })

    df_inventory = pd.DataFrame(inventory)
    df_inventory = df_inventory.sort_values(["okpd2_raw", "product_name_raw"])

    logger.info(f"Инвентарь: {len(df_inventory)} уникальных культур")
    return df_inventory


def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Сгенерировать сводку по листам.
    """
    logger = logging.getLogger(__name__)
    logger.info("Генерация сводки...")

    summary = []

    for sheet_name, group in df.groupby("source_sheet"):
        years = group["year"].dropna()
        year_min = int(years.min()) if len(years) > 0 else None
        year_max = int(years.max()) if len(years) > 0 else None

        n_products = group["product_name_raw"].nunique()
        n_regions = group["region_name"].nunique()
        n_observed = (group["value_status"] == "observed").sum()

        summary.append({
            "source_sheet": sheet_name,
            "frequency": group["frequency"].iloc[0] if len(group) > 0 else None,
            "geography_level": group["geography_level"].iloc[0] if len(group) > 0 else None,
            "n_rows": len(group),
            "n_observed_values": n_observed,
            "n_unique_products": n_products,
            "n_unique_regions": n_regions,
            "year_min": year_min,
            "year_max": year_max,
        })

    df_summary = pd.DataFrame(summary)
    logger.info(f"Сводка: {len(df_summary)} листов")
    return df_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Детерминированный парсер файла Росстата cena_sx_07-2026.xlsx"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="cena_sx_07-2026.xlsx",
        help="Путь к входному XLSX файлу",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rosstat_prices_raw_long.csv",
        help="Путь к выходному CSV с полными данными",
    )
    parser.add_argument(
        "--inventory",
        type=str,
        default="crop_inventory.csv",
        help="Путь к выходному CSV с инвентарём культур",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="stage1_summary.csv",
        help="Путь к выходному CSV со сводкой",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод",
    )

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Файл не найден: {input_path}")
        return 1

    # Парсим workbook
    records = parse_workbook(input_path)

    if not records:
        logger.error("Не удалось извлечь данные")
        return 1

    # Преобразуем в DataFrame
    df = records_to_dataframe(records)

    # Сохраняем полные данные
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Полные данные сохранены: {output_path} ({len(df):,} строк)")

    # Генерируем инвентарь культур
    df_inventory = generate_crop_inventory(df)
    inventory_path = Path(args.inventory)
    df_inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    logger.info(f"Инвентарь культур сохранён: {inventory_path}")

    # Генерируем сводку
    df_summary = generate_summary(df)
    summary_path = Path(args.summary)
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info(f"Сводка сохранена: {summary_path}")

    # Финальная статистика
    logger.info("=" * 70)
    logger.info("ИТОГОВАЯ СТАТИСТИКА")
    logger.info("=" * 70)
    logger.info(f"Всего записей: {len(df):,}")
    logger.info(f"Уникальных культур: {df['product_name_raw'].nunique()}")
    logger.info(f"Уникальных ОКПД2: {df['okpd2_raw'].nunique()}")
    logger.info(f"Уникальных территорий: {df['region_name'].nunique()}")
    logger.info(f"Диапазон лет: {df['year'].min()} – {df['year'].max()}")
    logger.info(f"Наблюдаемых значений: {(df['value_status'] == 'observed').sum():,}")
    logger.info(f"Конфиденциальных: {(df['value_status'] == 'confidential').sum():,}")
    logger.info(f"Отсутствующих: {(df['value_status'] == 'unavailable').sum():,}")
    logger.info(f"Структурно пропущенных: {(df['value_status'] == 'structurally_missing').sum():,}")

    # Статистика по географии
    logger.info("-" * 70)
    logger.info("По уровням географии:")
    for geo, group in df.groupby("geography_level"):
        logger.info(f"  {geo}: {len(group):,} записей, {group['region_name'].nunique()} территорий")

    return 0


if __name__ == "__main__":
    sys.exit(main())