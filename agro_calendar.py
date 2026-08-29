"""
agro_calendar.py (v1.0)
Агрономический календарь технологических карт для тракторных операций.
Источник: Сроки.ПДФ (РГАУ-МСХА им. К.А. Тимирязева, 2022), Приложение В.

Каждая культура описывается:
  - технологической картой (последовательность операций)
  - количеством проходов каждого типа
  - агрокалендарным окном (месяцы)

ВАЖНО:
  - Нормативы моточасов НЕ хранятся здесь.
    Они вычисляются динамически через agro_norms.get_engine_hours_per_ha()
    в зависимости от выбранного трактора и полевых условий (K_об).
  - Комбайновые операции помечены как machine_type="combine"
    и ИСКЛЮЧАЮТСЯ из расчёта тракторного риска.

Зависимости:
  - operation_mapping.py (маппинг операций документ → TUM)
  - agro_norms.py (нормативы выработки из Приложения Б)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Импорт зависимостей ──────────────────────────────────────────────
try:
    from operation_mapping import lookup_operation, is_tractor_operation, get_tum_key

    HAS_OPERATION_MAPPING = True
except ImportError:
    HAS_OPERATION_MAPPING = False

try:
    from agro_norms import get_engine_hours_per_ha, calculate_k_ob, Tractor

    HAS_AGRO_NORMS = True
except ImportError:
    HAS_AGRO_NORMS = False


# ─── Структуры данных ─────────────────────────────────────────────────


@dataclass(frozen=True)
class OperationPass:
    """
    Один проход (или группа проходов) конкретной операции.

    ВАЖНО: engine_hours_per_ha НЕ хранится здесь.
    Он вычисляется динамически через agro_norms.get_engine_hours_per_ha().
    """

    operation_key: str  # Ключ в TUM_OPERATIONS / OPERATION_INFO
    doc_operation: str  # Нормализованное название из Сроки.ПДФ
    passes: int  # Количество проходов
    window_start_month: int  # Месяц начала окна (1=январь)
    window_end_month: int  # Месяц конца окна
    description: str = ""  # Описание операции
    machine_type: str = "tractor"  # tractor | combine | truck | loader

    def __post_init__(self) -> None:
        if self.passes < 1:
            raise ValueError(f"passes must be >= 1, got {self.passes}")
        if not (1 <= self.window_start_month <= 12):
            raise ValueError(
                f"window_start_month must be in [1, 12], got {self.window_start_month}"
            )
        if not (1 <= self.window_end_month <= 12):
            raise ValueError(f"window_end_month must be in [1, 12], got {self.window_end_month}")
        if self.machine_type not in {"tractor", "combine", "truck", "loader"}:
            raise ValueError(
                f"machine_type must be one of tractor/combine/truck/loader, got {self.machine_type}"
            )

    @property
    def is_tractor(self) -> bool:
        """Является ли операция тракторной (входит в тракторный риск)."""
        return self.machine_type == "tractor"


@dataclass(frozen=True)
class CropTechnology:
    """Полная технологическая карта одной культуры."""

    crop_key: str
    crop_name_ru: str
    crop_name_en: str
    region_preference: str
    tillage_system: str  # conventional / mini-till / no-till
    operations: Tuple[OperationPass, ...]
    notes: str = ""

    @property
    def tractor_operations(self) -> List[OperationPass]:
        """Только тракторные операции (исключая комбайн)."""
        return [op for op in self.operations if op.is_tractor]

    @property
    def combine_operations(self) -> List[OperationPass]:
        """Только комбайновые операции."""
        return [op for op in self.operations if op.machine_type == "combine"]

    @property
    def n_operations(self) -> int:
        return len(self.operations)

    @property
    def n_tractor_operations(self) -> int:
        return len(self.tractor_operations)


# ─── Справочник культур (Приложение В, Сроки.ПДФ) ─────────────────────

CROP_CATALOG: Dict[str, CropTechnology] = {
    # ═══ ГОРОХ ═══════════════════════════════════════════════════════
    "pea": CropTechnology(
        crop_key="pea",
        crop_name_ru="Горох",
        crop_name_en="Pea",
        region_preference="Центр, Поволжье, Сибирь",
        tillage_system="conventional",
        operations=(
            OperationPass("Disc harrowing", "лущение стерни", 1, 8, 8, "Лущение стерни"),
            OperationPass(
                "Fertilizing", "погрузка органических удобрений", 1, 9, 9, "Погрузка орг. удобрений"
            ),
            OperationPass(
                "Fertilizing",
                "внесение органических удобрений",
                1,
                9,
                9,
                "Внесение орг. удобрений 15 т/га",
            ),
            OperationPass("Ploughing", "вспашка", 1, 9, 9, "Вспашка"),
            OperationPass("Seedbed combination", "планировка почвы", 1, 4, 4, "Планировка почвы"),
            OperationPass("Cultivating (deep)", "культивация", 1, 5, 5, "Культивация"),
            OperationPass(
                "Transport",
                "транспортировка семян",
                1,
                5,
                5,
                "Транспортировка семян и загрузка сеялок",
            ),
            OperationPass(
                "Seed drill combination 4m", "посев зерновых", 1, 5, 5, "Посев 0,28 т/га"
            ),
            OperationPass("Seedbed combination", "прикатывание", 1, 5, 5, "Прикатывание"),
            OperationPass(
                "Cultivating (shallow)", "боронование до всходов", 1, 5, 5, "Боронование до всходов"
            ),
            OperationPass(
                "Cultivating (shallow)", "боронование по всходам", 1, 5, 6, "Боронование по всходам"
            ),
            OperationPass(
                "Spraying", "внесение инсектицида", 1, 6, 6, "Внесение инсектицидов 200 л/га"
            ),
            OperationPass("Fertilizing", "полив с удобрениями", 1, 7, 7, "Полив с удобрениями"),
            OperationPass("Mowing (front)", "кошение трав", 1, 8, 8, "Кошение в валки"),
            # КОМБАЙН — исключаем из тракторного риска
            OperationPass(
                None,
                "подбор и обмолот валков",
                1,
                8,
                8,
                "Подбор и обмолот валков 18 ц/га",
                machine_type="combine",
            ),
            OperationPass("Transport", "транспортировка", 1, 8, 8, "Транспортировка зерна"),
        ),
        notes="Уборка — комбайн. 16 операций, 15 тракторных.",
    ),
    # ═══ КАРТОФЕЛЬ ═══════════════════════════════════════════════════
    "potato": CropTechnology(
        crop_key="potato",
        crop_name_ru="Картофель",
        crop_name_en="Potato",
        region_preference="Поволжье, Центр, Нечерноземье",
        tillage_system="conventional",
        operations=(
            OperationPass("Disc harrowing", "лущение стерни", 1, 8, 8, "Лущение стерни"),
            OperationPass(
                "Fertilizing", "погрузка органических удобрений", 1, 8, 8, "Погрузка орг. удобрений"
            ),
            OperationPass(
                "Fertilizing",
                "внесение органических удобрений",
                1,
                8,
                9,
                "Внесение орг. удобрений 50 т/га",
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора минеральных удобрений",
                1,
                8,
                8,
                "Транспортировка раствора мин. удобрений 300 л/га",
            ),
            OperationPass(
                "Fertilizing",
                "внесение раствора минеральных удобрений",
                1,
                8,
                8,
                "Внесение раствора минеральных удобрений",
            ),
            OperationPass("Ploughing", "вспашка", 1, 8, 9, "Вспашка"),
            OperationPass("Cultivating (shallow)", "боронование", 1, 4, 4, "Боронование"),
            OperationPass("Ploughing", "перепашка", 1, 4, 4, "Перепашка"),
            OperationPass("Cultivating (deep)", "культивация", 1, 4, 4, "Культивация"),
            OperationPass(
                "Transport", "транспортировка клубней", 1, 5, 5, "Транспортировка клубней"
            ),
            OperationPass(
                "Transport", "транспортировка семян", 1, 5, 5, "Загрузка картофелесажалок на поле"
            ),
            OperationPass(
                "Precision air seeding", "посадка картофеля", 1, 5, 5, "Посадка картофеля 2,8 т/га"
            ),
            OperationPass(
                "Cultivating (shallow)",
                "междурядная обработка картофеля",
                1,
                5,
                5,
                "Междурядная обработка первая",
            ),
            OperationPass(
                "Cultivating (shallow)",
                "междурядная обработка картофеля",
                1,
                5,
                6,
                "Междурядная обработка вторая",
            ),
            OperationPass(
                "Cultivating (shallow)",
                "междурядная обработка картофеля",
                1,
                6,
                6,
                "Междурядная обработка третья",
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора ядохимикатов",
                1,
                6,
                6,
                "Транспортировка раствора ядохимикатов",
            ),
            OperationPass("Spraying", "опрыскивание", 1, 6, 6, "Опрыскивание 200 л/га"),
            OperationPass(
                "Cultivating (shallow)",
                "междурядная обработка картофеля",
                1,
                7,
                7,
                "Междурядная обработка четвёртая",
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора ядохимикатов",
                1,
                8,
                8,
                "Транспортировка раствора ядохимикатов",
            ),
            OperationPass("Spraying", "опрыскивание", 1, 8, 8, "Опрыскивание 300 л/га"),
            OperationPass("Mowing (front)", "скашивание ботвы", 1, 9, 9, "Скашивание ботвы"),
            # КОМБАЙН — исключаем из тракторного риска
            OperationPass(
                None,
                "уборка прямым комбайнированием",
                1,
                9,
                10,
                "Уборка прямым комбайнированием 18 т/га",
                machine_type="combine",
            ),
            OperationPass(
                "Transport", "транспортировка клубней", 1, 9, 10, "Транспортировка клубней"
            ),
        ),
        notes="Самая трудоёмкая культура: 23 операции, 22 тракторных. "
        "4 междурядных + 2 опрыскивания.",
    ),
    # ═══ КЛЕВЕР ══════════════════════════════════════════════════════
    "clover": CropTechnology(
        crop_key="clover",
        crop_name_ru="Клевер",
        crop_name_en="Clover",
        region_preference="Центр, Нечерноземье",
        tillage_system="conventional",
        operations=(
            OperationPass("Disc harrowing", "лущение стерни", 1, 9, 9, "Лущение стерни"),
            OperationPass("Ploughing", "вспашка", 1, 9, 9, "Вспашка"),
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                4,
                4,
                "Культивация с боронованием",
            ),
            OperationPass(
                "Transport",
                "транспортировка семян",
                1,
                5,
                5,
                "Транспортировка семян и загрузка сеялок",
            ),
            OperationPass("Seed drill combination 3m", "посев клевера", 1, 5, 5, "Посев 23 кг/га"),
            OperationPass("Seedbed combination", "прикатывание", 1, 5, 5, "Прикатывание"),
            OperationPass(
                "Cultivating (shallow)", "боронование до всходов", 1, 5, 5, "Боронование до всходов"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора гербицидов",
                1,
                5,
                5,
                "Транспортировка раствора гербицида",
            ),
            OperationPass("Spraying", "внесение гербицида", 1, 5, 5, "Внесение гербицида 100 л/га"),
            OperationPass(
                "Transport",
                "транспортировка раствора инсектицидов",
                1,
                6,
                6,
                "Транспортировка раствора инсектицида",
            ),
            OperationPass(
                "Spraying", "внесение инсектицида", 1, 6, 6, "Внесение инсектицида 300 л/га"
            ),
            OperationPass("Mowing (front)", "кошение трав", 1, 7, 7, "Кошение в расстил"),
            OperationPass("Swathing", "сгребание в валки", 1, 7, 7, "Сгребание в валки"),
            OperationPass(
                "Swathing",
                "подбор валков с измельчением",
                1,
                7,
                7,
                "Подбор валков с измельчением 10 т/га",
            ),
            OperationPass(
                "Transport",
                "транспортировка сенажной массы",
                1,
                7,
                7,
                "Транспортировка сенажной массы",
            ),
            OperationPass(
                "Transport", "трамбовка сенажной массы", 1, 7, 7, "Трамбовка сенажной массы"
            ),
        ),
        notes="16 операций, все тракторные.",
    ),
    # ═══ КОРМОВАЯ СВЁКЛА ═════════════════════════════════════════════
    "forage_beet": CropTechnology(
        crop_key="forage_beet",
        crop_name_ru="Кормовая свёкла",
        crop_name_en="Forage Beet",
        region_preference="Центр, Нечерноземье",
        tillage_system="conventional",
        operations=(
            OperationPass("Disc harrowing", "лущение стерни", 1, 9, 9, "Лущение стерни"),
            OperationPass(
                "Transport",
                "транспортировка удобрений",
                1,
                9,
                9,
                "Транспортировка минеральных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                9,
                9,
                "Внесение минеральных удобрений 500 кг/га",
            ),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                9,
                9,
                "Погрузка органических удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение органических удобрений",
                1,
                9,
                9,
                "Внесение орг. удобрений 10 т/га",
            ),
            OperationPass("Ploughing", "вспашка", 1, 9, 10, "Вспашка"),
            OperationPass("Seedbed combination", "планировка почвы", 1, 4, 4, "Планировка почвы"),
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                4,
                4,
                "Культивация с боронованием",
            ),
            OperationPass("Seedbed combination", "нарезка гребней", 1, 4, 4, "Нарезка гребней"),
            OperationPass("Precision air seeding", "посев свёклы", 1, 4, 4, "Посев 15 кг/га"),
            OperationPass(
                "Cultivating (shallow)",
                "междурядная обработка свёклы",
                1,
                6,
                6,
                "Междурядная обработка первая",
            ),
            OperationPass(
                "Cultivating (shallow)",
                "междурядная обработка свёклы",
                1,
                7,
                7,
                "Междурядная обработка вторая",
            ),
            OperationPass("Mowing (front)", "скашивание ботвы", 1, 9, 9, "Скашивание ботвы"),
            # КОМБАЙН — исключаем из тракторного риска
            OperationPass(
                None,
                "уборка корнеплодов",
                1,
                9,
                10,
                "Уборка корнеплодов 9 т/га",
                machine_type="combine",
            ),
            OperationPass(
                "Transport", "транспортировка корнеплодов", 1, 9, 10, "Транспортировка корнеплодов"
            ),
        ),
        notes="15 операций, 14 тракторных.",
    ),
    # ═══ КУКУРУЗА НА СИЛОС/ЗЕРНО ═════════════════════════════════════
    "corn": CropTechnology(
        crop_key="corn",
        crop_name_ru="Кукуруза (силос/зерно)",
        crop_name_en="Corn (Maize)",
        region_preference="Юг, Центр",
        tillage_system="conventional",
        operations=(
            OperationPass(
                "Fertilizing", "погрузка органических удобрений", 1, 3, 4, "Погрузка орг. удобрений"
            ),
            OperationPass("Transport", "вывозка навоза", 1, 3, 4, "Вывозка навоза в бурты"),
            OperationPass("Disc harrowing", "лущение стерни", 1, 4, 4, "Лущение стерни"),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                4,
                4,
                "Погрузка орг. удобрений на поле",
            ),
            OperationPass(
                "Fertilizing",
                "внесение органических удобрений",
                1,
                4,
                4,
                "Внесение органических удобрений 20 т/га",
            ),
            OperationPass("Ploughing", "вспашка", 1, 4, 5, "Вспашка"),
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                5,
                5,
                "Культивация с боронованием",
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора гербицидов",
                1,
                5,
                5,
                "Транспортировка раствора гербицидов",
            ),
            OperationPass(
                "Spraying", "внесение гербицида", 1, 5, 5, "Внесение гербицидов 300 л/га"
            ),
            OperationPass(
                "Transport",
                "транспортировка семян",
                1,
                5,
                5,
                "Транспортировка семян и загрузка сеялок",
            ),
            OperationPass(
                "Transport",
                "транспортировка удобрений",
                1,
                5,
                5,
                "Транспортировка удобрений и загрузка сеялок",
            ),
            OperationPass(
                "Precision air seeding",
                "посев кукурузы",
                1,
                5,
                5,
                "Посев 18 кг/га с внесением удобрений 70 кг/га",
            ),
            OperationPass(
                "Cultivating (shallow)", "боронование до всходов", 1, 5, 5, "Боронование до всходов"
            ),
            OperationPass(
                "Cultivating (shallow)", "боронование по всходам", 1, 5, 5, "Боронование по всходам"
            ),
            OperationPass(
                "Cultivating (shallow)",
                "междурядная обработка кукурузы",
                1,
                6,
                6,
                "Междурядная обработка первая",
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора фунгицидов",
                1,
                6,
                6,
                "Транспортировка раствора фунгицидов",
            ),
            OperationPass(
                "Spraying", "внесение фунгицида", 1, 6, 6, "Внесение фунгицидов 300 л/га"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора гербицидов",
                1,
                6,
                6,
                "Транспортировка раствора гербицидов",
            ),
            OperationPass(
                "Spraying", "внесение гербицида", 1, 6, 6, "Внесение гербицидов 300 л/га"
            ),
            OperationPass(
                "Cultivating (shallow)",
                "междурядная обработка кукурузы",
                1,
                6,
                7,
                "Междурядная обработка вторая",
            ),
            # На силос:
            OperationPass(
                None, "уборка на силос", 1, 9, 9, "Уборка на силос 50 т/га", machine_type="combine"
            ),
            OperationPass("Transport", "транспортировка силоса", 1, 9, 9, "Транспортировка силоса"),
            OperationPass("Transport", "трамбовка силоса", 1, 9, 9, "Трамбовка силоса"),
            # На зерно:
            OperationPass(
                None,
                "уборка прямым комбайнированием",
                1,
                8,
                8,
                "Уборка прямым комбайнированием 8 т/га",
                machine_type="combine",
            ),
            OperationPass("Transport", "транспортировка зерна", 1, 8, 8, "Транспортировка зерна"),
        ),
        notes="25 операций, 23 тракторных. Самая тяжёлая культура.",
    ),
    # ═══ ЛЮПИН НА ЗЕРНО ══════════════════════════════════════════════
    "lupin": CropTechnology(
        crop_key="lupin",
        crop_name_ru="Люпин на зерно",
        crop_name_en="Lupin (grain)",
        region_preference="Нечерноземье, Центр",
        tillage_system="conventional",
        operations=(
            OperationPass("Disc harrowing", "лущение стерни", 1, 8, 8, "Лущение стерни"),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                9,
                9,
                "Погрузка калийных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                9,
                9,
                "Внесение калийных удобрений 300 кг/га",
            ),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                9,
                9,
                "Погрузка фосфорных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                9,
                9,
                "Внесение фосфорных удобрений 300 кг/га",
            ),
            OperationPass("Ploughing", "вспашка", 1, 9, 9, "Вспашка"),
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                4,
                4,
                "Культивация с боронованием",
            ),
            OperationPass("Transport", "транспортировка семян", 1, 4, 4, "Подвоз семян"),
            OperationPass(
                "Seed drill combination 4m", "посев зерновых", 1, 4, 4, "Посев 200 кг/га"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора гербицидов",
                1,
                4,
                4,
                "Транспортировка раствора гербицидов",
            ),
            OperationPass(
                "Spraying", "внесение гербицида", 1, 4, 4, "Внесение гербицидов 100 л/га"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора фунгицидов",
                1,
                5,
                5,
                "Транспортировка раствора фунгицидов",
            ),
            OperationPass(
                "Spraying", "внесение фунгицида", 1, 5, 5, "Внесение фунгицидов 300 л/га"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора инсектицидов",
                1,
                6,
                6,
                "Транспортировка раствора инсектицидов",
            ),
            OperationPass(
                "Spraying", "внесение инсектицида", 1, 6, 6, "Внесение инсектицидов 200 л/га"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора для десикации",
                1,
                7,
                7,
                "Транспортировка раствора для десикации",
            ),
            OperationPass("Spraying", "десикация", 1, 7, 7, "Десикация 200 л/га"),
            # КОМБАЙН — исключаем из тракторного риска
            OperationPass(
                None,
                "уборка прямым комбайнированием",
                1,
                8,
                8,
                "Уборка прямым комбайнированием 20 ц/га",
                machine_type="combine",
            ),
            OperationPass("Transport", "транспортировка зерна", 1, 8, 8, "Транспортировка зерна"),
        ),
        notes="19 операций, 18 тракторных.",
    ),
    # ═══ ЛЮЦЕРНА МНОГОЛЕТНЯЯ НА СЕНО ═════════════════════════════════
    "alfalfa": CropTechnology(
        crop_key="alfalfa",
        crop_name_ru="Люцерна многолетняя на сено",
        crop_name_en="Alfalfa (hay)",
        region_preference="Юг, Центр",
        tillage_system="conventional",
        operations=(
            # Первичный посев
            OperationPass("Disc harrowing", "лущение стерни", 1, 9, 9, "Лущение стерни"),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                9,
                9,
                "Погрузка минеральных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                9,
                9,
                "Внесение минеральных удобрений 150 кг/га",
            ),
            OperationPass("Ploughing", "вспашка", 1, 10, 10, "Вспашка"),
            OperationPass("Seedbed combination", "планировка почвы", 1, 4, 4, "Планировка почвы"),
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                4,
                4,
                "Культивация с боронованием",
            ),
            OperationPass("Seed drill combination 3m", "посев клевера", 1, 4, 4, "Посев 12 кг/га"),
            OperationPass("Seedbed combination", "прикатывание", 1, 4, 4, "Прикатывание"),
            # Дальнейшие работы (в т.ч. в последующий год)
            OperationPass("Mowing (front)", "скашивание ботвы", 1, 5, 5, "Скашивание сорняков"),
            OperationPass("Fertilizing", "полив с удобрениями", 1, 5, 5, "Полив"),
            OperationPass(
                "Transport",
                "транспортировка раствора инсектицидов",
                1,
                5,
                5,
                "Транспортировка раствора инсектицидов",
            ),
            OperationPass(
                "Spraying", "внесение инсектицида", 1, 5, 5, "Внесение инсектицидов 300 л/га"
            ),
            # Первый укос
            OperationPass(None, "первый укос", 1, 6, 6, "Первый укос", machine_type="combine"),
            OperationPass("Swathing", "ворошение сена", 3, 6, 6, "Ворошение ×3"),
            OperationPass(
                "Swathing", "сгребание сена в валки", 1, 6, 6, "Сгребание сена в валки 6 т/га"
            ),
            OperationPass(
                "Swathing",
                "подбор валков с прессованием и транспортировка",
                1,
                6,
                6,
                "Подбор валков с прессованием и транспортировка",
            ),
            # Второй укос
            OperationPass("Fertilizing", "полив с удобрениями", 1, 6, 6, "Полив"),
            OperationPass("Fertilizing", "полив с удобрениями", 1, 7, 7, "Полив"),
            OperationPass(None, "второй укос", 1, 8, 8, "Второй укос", machine_type="combine"),
            OperationPass("Swathing", "ворошение сена", 3, 8, 8, "Ворошение ×3"),
            OperationPass(
                "Swathing", "сгребание сена в валки", 1, 8, 8, "Сгребание сена в валки 6 т/га"
            ),
            OperationPass(
                "Swathing",
                "подбор валков с прессованием и транспортировка",
                1,
                8,
                8,
                "Подбор валков с прессованием и транспортировка",
            ),
            OperationPass("Fertilizing", "полив с удобрениями", 1, 8, 8, "Полив"),
            OperationPass("Fertilizing", "полив с удобрениями", 1, 9, 9, "Полив"),
        ),
        notes="24 операции, 22 тракторных. 2 укоса за сезон.",
    ),
    # ═══ МНОГОЛЕТНИЕ ТРАВЫ НА СЕНАЖ ══════════════════════════════════
    "forage_grass": CropTechnology(
        crop_key="forage_grass",
        crop_name_ru="Многолетние травы на сенаж",
        crop_name_en="Forage Grass (silage)",
        region_preference="Все регионы",
        tillage_system="conventional",
        operations=(
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                4,
                4,
                "Культивация с боронованием",
            ),
            OperationPass("Seed drill combination 3m", "посев клевера", 1, 5, 5, "Посев 30 кг/га"),
            OperationPass("Seedbed combination", "прикатывание", 1, 5, 5, "Прикатывание"),
            OperationPass(
                "Cultivating (shallow)", "боронование по всходам", 1, 5, 5, "Боронование по всходам"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора минеральных удобрений",
                1,
                5,
                5,
                "Транспортировка раствора минеральных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение раствора минеральных удобрений",
                1,
                5,
                5,
                "Внесение раствора минеральных удобрений 140 л/га",
            ),
            # 1-е кошение
            OperationPass("Mowing (front)", "кошение трав", 1, 6, 6, "1-е кошение в расстил"),
            OperationPass("Swathing", "сгребание в валки", 1, 6, 6, "Сгребание в валки"),
            OperationPass(
                "Swathing",
                "подбор валков с измельчением",
                1,
                6,
                6,
                "Подбор валков с измельчением 3,5 т/га",
            ),
            OperationPass(
                "Transport", "транспортировка сенажной массы", 1, 6, 6, "Транспортировка сенажа"
            ),
            OperationPass("Transport", "трамбовка сенажной массы", 1, 6, 6, "Трамбование сенажа"),
            # 2-е кошение
            OperationPass("Mowing (front)", "кошение трав", 1, 8, 8, "2-е кошение в расстил"),
            OperationPass("Swathing", "сгребание в валки", 1, 8, 8, "Сгребание в валки"),
            OperationPass(
                "Swathing",
                "подбор валков с измельчением",
                1,
                8,
                8,
                "Подбор валков с измельчением 3 т/га",
            ),
            OperationPass(
                "Transport", "транспортировка сенажной массы", 1, 8, 8, "Транспортировка сенажа"
            ),
            OperationPass("Transport", "трамбовка сенажной массы", 1, 8, 8, "Трамбование сенажа"),
        ),
        notes="16 операций, все тракторные. 2 укоса за сезон.",
    ),
    # ═══ ОВЁС ════════════════════════════════════════════════════════
    "oats": CropTechnology(
        crop_key="oats",
        crop_name_ru="Овёс",
        crop_name_en="Oats",
        region_preference="Сибирь, Урал, Нечерноземье",
        tillage_system="conventional",
        operations=(
            OperationPass("Disc harrowing", "лущение стерни", 1, 8, 8, "Лущение стерни"),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                9,
                9,
                "Погрузка калийных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                9,
                9,
                "Внесение калийных удобрений 300 кг/га",
            ),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                9,
                9,
                "Погрузка фосфорных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                9,
                9,
                "Внесение фосфорных удобрений 300 кг/га",
            ),
            OperationPass("Ploughing", "вспашка", 1, 9, 9, "Вспашка"),
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                4,
                4,
                "Культивация с боронованием",
            ),
            OperationPass("Transport", "транспортировка семян", 1, 4, 4, "Подвоз семян"),
            OperationPass(
                "Seed drill combination 4m", "посев зерновых", 1, 4, 4, "Посев 200 кг/га"
            ),
            OperationPass("Seedbed combination", "прикатывание", 1, 4, 4, "Прикатывание"),
            OperationPass(
                "Cultivating (shallow)", "боронование до всходов", 1, 4, 5, "Боронование до всходов"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора гербицидов",
                1,
                5,
                5,
                "Транспортировка раствора гербицидов",
            ),
            OperationPass(
                "Spraying", "внесение гербицида", 1, 5, 5, "Внесение раствора гербицидов 200 л/га"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора азотных удобрений",
                1,
                5,
                5,
                "Транспортировка раствора азотных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение раствора азотных удобрений",
                1,
                5,
                5,
                "Внесение раствора азотных удобрений 200 л/га",
            ),
            # КОМБАЙН — исключаем из тракторного риска
            OperationPass(
                None,
                "уборка прямым комбайнированием",
                1,
                7,
                7,
                "Уборка 30 ц/га",
                machine_type="combine",
            ),
            OperationPass("Transport", "транспортировка зерна", 1, 7, 7, "Транспортировка зерна"),
            OperationPass(
                "Swathing",
                "подбор валков с прессованием и транспортировка",
                1,
                7,
                7,
                "Прессование соломы 40 ц/га",
            ),
            OperationPass("Transport", "транспортировка соломы", 1, 7, 7, "Транспортировка соломы"),
            OperationPass("Transport", "скирдование сена и соломы", 1, 7, 7, "Скирдование соломы"),
        ),
        notes="20 операций, 19 тракторных.",
    ),
    # ═══ ОЗИМАЯ ПШЕНИЦА ══════════════════════════════════════════════
    "wheat_winter": CropTechnology(
        crop_key="wheat_winter",
        crop_name_ru="Пшеница озимая",
        crop_name_en="Winter Wheat",
        region_preference="Юг, Центр, Поволжье",
        tillage_system="conventional",
        operations=(
            OperationPass("Disc harrowing", "лущение стерни", 1, 7, 8, "Лущение стерни"),
            OperationPass(
                "Fertilizing", "погрузка органических удобрений", 1, 8, 8, "Погрузка орг. удобрений"
            ),
            OperationPass(
                "Fertilizing",
                "внесение органических удобрений",
                1,
                8,
                8,
                "Внесение орг. удобрений 22 т/га",
            ),
            OperationPass(
                "Fertilizing", "погрузка органических удобрений", 1, 8, 8, "Погрузка мин. удобрений"
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                8,
                8,
                "Внесение мин. удобрений 240 кг/га",
            ),
            OperationPass("Ploughing", "вспашка", 1, 8, 8, "Вспашка"),
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                8,
                8,
                "Культивация с боронованием",
            ),
            OperationPass(
                "Transport",
                "транспортировка семян и удобрений",
                1,
                8,
                8,
                "Транспортировка семян и удобрений",
            ),
            OperationPass(
                "Seed drill combination 4m",
                "посев зерновых",
                1,
                8,
                9,
                "Посев 150 кг/га с внесением удобрений 50 кг/га",
            ),
            OperationPass(
                "Cultivating (shallow)", "боронование посевов", 1, 4, 4, "Боронование посевов"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора гербицидов",
                1,
                6,
                6,
                "Транспортировка раствора гербицида",
            ),
            OperationPass(
                "Spraying", "внесение гербицида", 1, 6, 6, "Внесение раствора гербицида 100 л/га"
            ),
            # КОМБАЙН — исключаем из тракторного риска
            OperationPass(
                None,
                "прямое комбайнирование",
                1,
                8,
                9,
                "Прямое комбайнирование 35 ц/га",
                machine_type="combine",
            ),
            OperationPass("Transport", "транспортировка зерна", 1, 8, 9, "Транспортировка зерна"),
            OperationPass(
                "Swathing",
                "подбор валков с прессованием и транспортировка",
                1,
                8,
                9,
                "Прессование соломы 40 ц/га",
            ),
            OperationPass("Transport", "транспортировка соломы", 1, 8, 9, "Транспортировка соломы"),
            OperationPass("Transport", "скирдование сена и соломы", 1, 8, 9, "Скирдование соломы"),
        ),
        notes="17 операций, 16 тракторных. Уборка — комбайн.",
    ),
    # ═══ ОЗИМАЯ РОЖЬ ═════════════════════════════════════════════════
    "rye_winter": CropTechnology(
        crop_key="rye_winter",
        crop_name_ru="Рожь озимая",
        crop_name_en="Winter Rye",
        region_preference="Центр, Нечерноземье",
        tillage_system="conventional",
        operations=(
            OperationPass("Disc harrowing", "лущение стерни", 1, 8, 8, "Лущение стерни"),
            OperationPass(
                "Fertilizing", "погрузка органических удобрений", 1, 8, 8, "Погрузка орг. удобрений"
            ),
            OperationPass(
                "Fertilizing",
                "внесение органических удобрений",
                1,
                8,
                8,
                "Внесение орг. удобрений 15 т/га",
            ),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                8,
                8,
                "Погрузка минеральных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                8,
                8,
                "Внесение минеральных удобрений 100 кг/га",
            ),
            OperationPass("Ploughing", "вспашка", 1, 8, 8, "Вспашка"),
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                8,
                8,
                "Культивация с боронованием",
            ),
            OperationPass("Transport", "транспортировка семян", 1, 8, 8, "Транспортировка семян"),
            OperationPass(
                "Seed drill combination 4m", "посев зерновых", 1, 8, 9, "Посев 220 кг/га"
            ),
            OperationPass("Seedbed combination", "прикатывание", 1, 8, 9, "Прикатывание"),
            OperationPass(
                "Cultivating (shallow)", "боронование до всходов", 1, 9, 9, "Боронование до всходов"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора фунгицидов",
                1,
                9,
                9,
                "Транспортировка раствора фунгицида",
            ),
            OperationPass(
                "Spraying", "внесение фунгицида", 1, 9, 9, "Внесение раствора фунгицида 300 л/га"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора гербицидов",
                1,
                9,
                9,
                "Транспортировка раствора гербицида",
            ),
            OperationPass(
                "Spraying", "внесение гербицида", 1, 9, 9, "Внесение раствора гербицида 100 л/га"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора азотных удобрений",
                1,
                4,
                4,
                "Транспортировка раствора азотных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение раствора азотных удобрений",
                1,
                4,
                4,
                "Внесение раствора азотных удобрений 200 л/га",
            ),
            OperationPass(
                "Cultivating (shallow)", "боронование по всходам", 1, 5, 5, "Боронование по всходам"
            ),
            # КОМБАЙН — исключаем из тракторного риска
            OperationPass(
                None,
                "прямое комбайнирование",
                1,
                7,
                8,
                "Прямое комбайнирование 35 ц/га",
                machine_type="combine",
            ),
            OperationPass("Transport", "транспортировка зерна", 1, 7, 8, "Транспортировка зерна"),
            OperationPass(
                "Swathing",
                "подбор валков с прессованием и транспортировка",
                1,
                7,
                8,
                "Прессование соломы 30 ц/га",
            ),
            OperationPass("Transport", "транспортировка соломы", 1, 7, 8, "Транспортировка соломы"),
            OperationPass("Transport", "скирдование сена и соломы", 1, 7, 8, "Скирдование соломы"),
        ),
        notes="23 операции, 22 тракторных.",
    ),
    # ═══ ПОДСОЛНЕЧНИК ════════════════════════════════════════════════
    "sunflower": CropTechnology(
        crop_key="sunflower",
        crop_name_ru="Подсолнечник",
        crop_name_en="Sunflower",
        region_preference="Юг, Поволжье",
        tillage_system="conventional",
        operations=(
            OperationPass("Disc harrowing", "лущение стерни", 1, 9, 9, "Лущение стерни"),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                9,
                9,
                "Погрузка минеральных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                9,
                9,
                "Внесение минеральных удобрений 230 кг/га",
            ),
            OperationPass(
                "Fertilizing", "погрузка органических удобрений", 1, 9, 9, "Погрузка орг. удобрений"
            ),
            OperationPass(
                "Fertilizing",
                "внесение органических удобрений",
                1,
                9,
                10,
                "Внесение орг. удобрений 20 т/га",
            ),
            OperationPass("Ploughing", "вспашка", 1, 9, 10, "Вспашка"),
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                4,
                4,
                "Культивация с боронованием",
            ),
            OperationPass(
                "Transport", "транспортировка семян", 1, 4, 4, "Подвоз и загрузка семян в сеялки"
            ),
            OperationPass("Precision air seeding", "посев кукурузы", 1, 4, 4, "Посев 30 кг/га"),
            OperationPass(
                "Cultivating (shallow)", "боронование до всходов", 1, 4, 4, "Боронование до всходов"
            ),
            OperationPass(
                "Cultivating (shallow)", "боронование по всходам", 1, 4, 5, "Боронование по всходам"
            ),
            OperationPass(
                "Cultivating (shallow)",
                "междурядная обработка кукурузы",
                1,
                5,
                5,
                "Первая междурядная обработка",
            ),
            OperationPass(
                "Transport",
                "транспортировка удобрений",
                1,
                5,
                5,
                "Транспортировка и загрузка минеральных удобрений",
            ),
            OperationPass(
                "Cultivating (shallow)",
                "междурядная обработка кукурузы",
                1,
                5,
                5,
                "Вторая междурядная обработка с подкормкой 100 кг/га",
            ),
            OperationPass(
                "Cultivating (shallow)",
                "междурядная обработка кукурузы",
                1,
                5,
                6,
                "Третья междурядная обработка",
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора для десикации",
                1,
                9,
                9,
                "Транспортировка раствора для десикации",
            ),
            OperationPass("Spraying", "десикация", 1, 9, 9, "Десикация 200 л/га"),
            # КОМБАЙН — исключаем из тракторного риска
            OperationPass(
                None,
                "уборка с измельчением корзинок",
                1,
                9,
                9,
                "Уборка с измельчением корзинок 25 ц/га",
                machine_type="combine",
            ),
            OperationPass("Transport", "транспортировка семян", 1, 9, 9, "Транспортировка семян"),
            OperationPass(
                "Transport",
                "транспортировка измельчённых корзинок",
                1,
                9,
                9,
                "Транспортировка измельчённых корзинок 15 ц/га",
            ),
            OperationPass(
                "Transport", "скирдование измельч. корзинок", 1, 9, 9, "Скирдование корзинок"
            ),
        ),
        notes="21 операция, 20 тракторных.",
    ),
    # ═══ ТИМОФЕЕВКА + КЛЕВЕР ═════════════════════════════════════════
    "timothy_clover": CropTechnology(
        crop_key="timothy_clover",
        crop_name_ru="Тимофеевка + клевер",
        crop_name_en="Timothy + Clover",
        region_preference="Нечерноземье, Центр",
        tillage_system="conventional",
        operations=(
            OperationPass("Disc harrowing", "лущение стерни", 1, 9, 9, "Лущение стерни"),
            OperationPass("Ploughing", "вспашка", 1, 9, 9, "Вспашка"),
            OperationPass("Cultivating (shallow)", "боронование", 1, 3, 4, "Боронование"),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                4,
                4,
                "Погрузка минеральных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                4,
                4,
                "Внесение минеральных удобрений 100 кг/га",
            ),
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                4,
                4,
                "Культивация с боронованием",
            ),
            OperationPass("Seed drill combination 3m", "посев клевера", 1, 4, 4, "Посев 30 кг/га"),
            # Первый укос на силос
            OperationPass(
                None,
                "уборка на силос",
                1,
                7,
                7,
                "Первый укос на силос 13 т/га",
                machine_type="combine",
            ),
            OperationPass("Transport", "транспортировка силоса", 1, 7, 7, "Транспортировка силоса"),
            OperationPass("Transport", "трамбовка силоса", 1, 7, 7, "Трамбовка силоса"),
            # Второй укос на силос
            OperationPass(
                None,
                "уборка на силос",
                1,
                8,
                8,
                "Второй укос на силос 10 т/га",
                machine_type="combine",
            ),
            OperationPass("Transport", "транспортировка силоса", 1, 8, 8, "Транспортировка силоса"),
            OperationPass("Transport", "трамбовка силоса", 1, 8, 8, "Трамбовка силоса"),
        ),
        notes="13 операций, 11 тракторных. 2 укоса на силос.",
    ),
    # ═══ ЯРОВАЯ ПШЕНИЦА ══════════════════════════════════════════════
    "wheat_spring": CropTechnology(
        crop_key="wheat_spring",
        crop_name_ru="Пшеница яровая",
        crop_name_en="Spring Wheat",
        region_preference="Сибирь, Урал",
        tillage_system="conventional",
        operations=(
            OperationPass("Disc harrowing", "лущение стерни", 1, 9, 9, "Лущение стерни"),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                9,
                9,
                "Погрузка калийных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                9,
                9,
                "Внесение калийных удобрений 300 кг/га",
            ),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                9,
                9,
                "Погрузка фосфорных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                9,
                9,
                "Внесение фосфорных удобрений 300 кг/га",
            ),
            OperationPass("Ploughing", "вспашка", 1, 9, 10, "Вспашка"),
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                4,
                4,
                "Культивация с боронованием",
            ),
            OperationPass(
                "Transport", "транспортировка семян", 1, 4, 4, "Подвоз семян и загрузка сеялок"
            ),
            OperationPass(
                "Seed drill combination 4m", "посев зерновых", 1, 4, 4, "Посев 220 кг/га"
            ),
            OperationPass("Seedbed combination", "прикатывание", 1, 4, 4, "Прикатывание посевов"),
            OperationPass(
                "Transport",
                "транспортировка раствора гербицидов",
                1,
                5,
                5,
                "Транспортировка раствора гербицидов",
            ),
            OperationPass(
                "Spraying", "внесение гербицида", 1, 5, 5, "Внесение раствора гербицидов 400 л/га"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора инсектицидов",
                1,
                6,
                6,
                "Транспортировка раствора инсектицидов",
            ),
            OperationPass(
                "Spraying",
                "внесение инсектицида",
                1,
                6,
                6,
                "Внесение раствора инсектицидов 300 л/га",
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора для дефолиации",
                1,
                7,
                7,
                "Транспортировка раствора для дефолиации",
            ),
            OperationPass("Spraying", "дефолиация", 1, 7, 7, "Дефолиация 200 л/га"),
            # КОМБАЙН — исключаем из тракторного риска
            OperationPass(
                None,
                "уборка прямым комбайнированием",
                1,
                7,
                8,
                "Уборка прямым комбайнированием 35 ц/га",
                machine_type="combine",
            ),
            OperationPass("Transport", "транспортировка зерна", 1, 7, 8, "Транспортировка зерна"),
            OperationPass(
                "Swathing",
                "подбор валков с прессованием и транспортировка",
                1,
                7,
                8,
                "Прессование соломы 30 ц/га",
            ),
            OperationPass("Transport", "транспортировка соломы", 1, 7, 8, "Транспортировка соломы"),
            OperationPass("Transport", "скирдование сена и соломы", 1, 7, 8, "Скирдование соломы"),
        ),
        notes="21 операция, 20 тракторных.",
    ),
    # ═══ ЯРОВОЙ РАПС ═════════════════════════════════════════════════
    "rapeseed_spring": CropTechnology(
        crop_key="rapeseed_spring",
        crop_name_ru="Рапс яровой",
        crop_name_en="Spring Rapeseed (Canola)",
        region_preference="Центр, Сибирь",
        tillage_system="mini-till",
        operations=(
            OperationPass("Disc harrowing", "лущение стерни", 1, 9, 9, "Лущение стерни"),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                9,
                9,
                "Погрузка минеральных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                9,
                9,
                "Внесение минеральных удобрений 300 кг/га",
            ),
            OperationPass("Ploughing", "вспашка", 1, 9, 10, "Вспашка"),
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                4,
                4,
                "Культивация с боронованием",
            ),
            OperationPass(
                "Seed drill combination 3m",
                "посев клевера",
                1,
                5,
                5,
                "Посев 10 кг/га с внесением удобрений 12 кг/га",
            ),
            OperationPass("Seedbed combination", "прикатывание", 1, 5, 5, "Прикатывание"),
            OperationPass(
                "Transport",
                "транспортировка раствора пестицидов",
                1,
                6,
                6,
                "Транспортировка раствора пестицидов",
            ),
            OperationPass(
                "Spraying", "внесение пестицидов", 1, 6, 6, "Внесение раствора пестицидов 300 л/га"
            ),
            # КОМБАЙН — исключаем из тракторного риска
            OperationPass(
                None,
                "уборка прямым комбайнированием",
                1,
                8,
                8,
                "Уборка прямым комбайнированием 15 ц/га",
                machine_type="combine",
            ),
            OperationPass("Transport", "транспортировка урожая", 1, 8, 8, "Транспортировка урожая"),
        ),
        notes="11 операций, 10 тракторных. Mini-till.",
    ),
    # ═══ ЯРОВОЙ ЯЧМЕНЬ ═══════════════════════════════════════════════
    "barley_spring": CropTechnology(
        crop_key="barley_spring",
        crop_name_ru="Ячмень яровой",
        crop_name_en="Spring Barley",
        region_preference="Поволжье, Сибирь, Центр",
        tillage_system="conventional",
        operations=(
            OperationPass("Disc harrowing", "лущение стерни", 1, 9, 9, "Лущение стерни"),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                9,
                9,
                "Погрузка калийных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                9,
                9,
                "Внесение калийных удобрений 300 кг/га",
            ),
            OperationPass(
                "Fertilizing",
                "погрузка органических удобрений",
                1,
                9,
                9,
                "Погрузка фосфорных удобрений",
            ),
            OperationPass(
                "Fertilizing",
                "внесение минеральных удобрений",
                1,
                9,
                9,
                "Внесение фосфорных удобрений 300 кг/га",
            ),
            OperationPass(
                "Fertilizing", "погрузка органических удобрений", 1, 9, 9, "Погрузка КАС-удобрений"
            ),
            OperationPass(
                "Fertilizing", "внесение кас-удобрений", 1, 9, 9, "Внесение КАС-удобрений 200 кг/га"
            ),
            OperationPass("Ploughing", "вспашка", 1, 9, 10, "Вспашка"),
            OperationPass(
                "Cultivating (deep)",
                "культивация с боронованием",
                1,
                4,
                4,
                "Культивация с боронованием",
            ),
            OperationPass(
                "Transport", "транспортировка семян", 1, 4, 4, "Подвоз семян и загрузка сеялок"
            ),
            OperationPass(
                "Seed drill combination 4m", "посев зерновых", 1, 4, 4, "Посев 220 кг/га"
            ),
            OperationPass("Seedbed combination", "прикатывание", 1, 4, 4, "Прикатывание"),
            OperationPass(
                "Transport",
                "транспортировка раствора гербицидов",
                1,
                5,
                5,
                "Транспортировка раствора гербицидов",
            ),
            OperationPass(
                "Spraying", "внесение гербицида", 1, 5, 5, "Внесение раствора гербицидов 400 л/га"
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора инсектицидов",
                1,
                6,
                6,
                "Транспортировка раствора инсектицидов",
            ),
            OperationPass(
                "Spraying",
                "внесение инсектицида",
                1,
                6,
                6,
                "Внесение раствора инсектицидов 300 л/га",
            ),
            OperationPass(
                "Transport",
                "транспортировка раствора для дефолиации",
                1,
                7,
                7,
                "Транспортировка раствора для дефолиации",
            ),
            OperationPass("Spraying", "дефолиация", 1, 7, 7, "Дефолиация 200 л/га"),
            # КОМБАЙН — исключаем из тракторного риска
            OperationPass(
                None,
                "уборка прямым комбайнированием",
                1,
                7,
                8,
                "Уборка прямым комбайнированием 35 ц/га",
                machine_type="combine",
            ),
            OperationPass("Transport", "транспортировка зерна", 1, 7, 8, "Транспортировка зерна"),
            OperationPass(
                "Swathing",
                "подбор валков с прессованием и транспортировка",
                1,
                7,
                8,
                "Прессование соломы 30 ц/га",
            ),
            OperationPass("Transport", "транспортировка соломы", 1, 7, 8, "Транспортировка соломы"),
            OperationPass("Transport", "скирдование сена и соломы", 1, 7, 8, "Скирдование соломы"),
        ),
        notes="23 операции, 22 тракторных.",
    ),
}


# ─── Вспомогательные функции ──────────────────────────────────────────


def list_crops() -> List[str]:
    """Список ключей всех доступных культур (отсортированный)."""
    return sorted(CROP_CATALOG.keys())


def get_crop(crop_key: str) -> Optional[CropTechnology]:
    """Получить технологическую карту по ключу."""
    return CROP_CATALOG.get(crop_key.strip().lower())


def get_operations_for_crop(
    crop_key: str,
    season_months: Optional[Tuple[int, int]] = None,
    tractor_only: bool = True,
) -> List[OperationPass]:
    """
    Вернуть список операций для культуры.

    Parameters
    ----------
    crop_key : str
        Ключ культуры.
    season_months : Tuple[int, int], optional
        (месяц начала, месяц конца) — фильтр по агрокному.
    tractor_only : bool
        Если True — только тракторные операции (по умолчанию).

    Returns
    -------
    List[OperationPass]
    """
    crop = get_crop(crop_key)
    if crop is None:
        return []

    if tractor_only:
        ops = crop.tractor_operations
    else:
        ops = list(crop.operations)

    if season_months is not None:
        m_start, m_end = season_months
        ops = [
            op for op in ops if op.window_end_month >= m_start and op.window_start_month <= m_end
        ]

    return ops


def estimate_season_engine_hours(
    crop_key: str,
    area_ha: float,
    tractor: str = "МТЗ-82",
    k_ob: float = 1.0,
) -> Tuple[float, float]:
    """
    Оценить суммарные моточасы и средневзвешенный PeakLoad на сезон.

    Формула:
        total_hours = Σ(мч/га × проходы × площадь)
        weighted_peak = Σ(hours_i × PL_i) / Σ(hours_i)

    Parameters
    ----------
    crop_key : str
        Ключ культуры.
    area_ha : float
        Площадь в гектарах.
    tractor : str
        Марка трактора (для нормативов из Приложения Б).
    k_ob : float
        Коэффициент полевых условий (Таблица 1.1).

    Returns
    -------
    Tuple[float, float]
        (total_engine_hours, weighted_peak_load)

    Raises
    ------
    ValueError
        Если культура не найдена или площадь <= 0.
    """
    crop = get_crop(crop_key)
    if crop is None:
        raise ValueError(f"Unknown crop: {crop_key}")
    if area_ha <= 0.0:
        raise ValueError(f"area_ha must be > 0, got {area_ha}")

    # Импорт пиков из TUM_OPERATIONS / OPERATION_INFO
    try:
        from Real_calculator import TUM_OPERATIONS, OPERATION_INFO

        operation_peaks = {}
        for op_key, op_info in TUM_OPERATIONS.items():
            operation_peaks[op_key] = op_info.get("peak_load_mean", 0.50)
        for op_key, op_info in OPERATION_INFO.items():
            if op_key not in operation_peaks:
                operation_peaks[op_key] = op_info.get("peak_load_mean", 0.50)
    except ImportError:
        operation_peaks = {}

    weighted_sum = 0.0
    total_hours = 0.0

    for op in crop.tractor_operations:
        # Норматив мч/га из Приложения Б
        if HAS_AGRO_NORMS:
            hours_per_ha, _ = get_engine_hours_per_ha(op.doc_operation, tractor, k_ob)
        else:
            hours_per_ha = 0.20  # fallback

        hours = hours_per_ha * op.passes * area_ha
        peak = operation_peaks.get(op.operation_key, 0.50)
        peak = max(0.0, min(1.0, float(peak)))

        weighted_sum += hours * peak
        total_hours += hours

    if total_hours <= 0.0:
        return 0.0, 0.50

    weighted_peak = weighted_sum / total_hours
    weighted_peak = max(0.0, min(1.0, weighted_peak))

    return total_hours, weighted_peak


def format_crop_summary(
    crop_key: str,
    area_ha: float,
    peaks: list[float],
    operation_names_ru: Optional[Dict[str, str]] = None,
    k_ob: float = 1.0,
) -> str:
    """
    Сформировать текстовую сводку технологической карты.
    """
    crop = get_crop(crop_key)
    if crop is None:
        return f"Культура '{crop_key}' не найдена."

    if operation_names_ru is None:
        operation_names_ru = {}

    # Определить трактор из peaks (по умолчанию МТЗ-82)
    tractor = "МТЗ-82"

    lines = []
    lines.append("=" * 70)
    lines.append(f"ТЕХНОЛОГИЧЕСКАЯ КАРТА: {crop.crop_name_ru}")
    lines.append(f"Регион: {crop.region_preference} | Технология: {crop.tillage_system}")
    lines.append(f"Трактор: {tractor} | K_об: {k_ob:.4f}")
    lines.append("=" * 70)
    lines.append(
        f"{'Операция':<35s} {'Проходы':>7s} {'Мч/га':>7s} "
        f"{'Мч всего':>9s} {'Окно':>8s} {'Техника':>8s}"
    )
    lines.append("-" * 70)

    for op in crop.operations:
        name_ru = operation_names_ru.get(op.operation_key, op.doc_operation)
        window = f"{op.window_start_month:02d}-{op.window_end_month:02d}"

        if HAS_AGRO_NORMS and op.is_tractor:
            hours_per_ha, _ = get_engine_hours_per_ha(op.doc_operation, tractor, k_ob)
        else:
            hours_per_ha = 0.0

        total_hours = hours_per_ha * op.passes * area_ha

        machine = "трактор" if op.is_tractor else "КОМБАЙН"

        lines.append(
            f"  {name_ru:<33s} {op.passes:>7d} "
            f"{hours_per_ha:>7.2f} {total_hours:>9.1f} "
            f"{window:>8s} {machine:>8s}"
        )

    total_season_hours, weighted_peak = estimate_season_engine_hours(
        crop_key, area_ha, tractor, k_ob
    )

    lines.append("-" * 70)
    lines.append(
        f"ИТОГО: {crop.n_tractor_operations} тракторных операций (из {crop.n_operations} всего)"
    )
    lines.append(f"Суммарные моточасы: ≈ {total_season_hours:.0f} на {area_ha:.0f} га")
    lines.append(f"Средневзвешенный PeakLoad: {weighted_peak:.3f}")
    if crop.notes:
        lines.append(f"Примечание: {crop.notes}")

    return "\n".join(lines)
