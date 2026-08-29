"""
operation_mapping.py (v1.0)
Маппинг операций из технологических карт Сроки.ПДФ (Приложение В)
на ключи TUM_OPERATIONS / OPERATION_INFO из Real_calculator.py.

Каждая запись содержит:
  - tum_key: ключ в TUM_OPERATIONS / OPERATION_INFO (None = не тракторная)
  - machine_type: "tractor" | "combine" | "truck" | "loader"
  - description_ru: описание операции
  - doc_operation: оригинальное название из документа

ВАЖНО: операции с machine_type="combine" ИСКЛЮЧАЮТСЯ из расчёта
тракторного риска. Они выполняются комбайном, не трактором.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class OperationMapping:
    """Одна операция из технологической карты документа."""

    doc_operation: str  # Оригинальное название из Сроки.ПДФ
    tum_key: Optional[str]  # Ключ в TUM_OPERATIONS (None = не трактор)
    machine_type: str  # tractor | combine | truck | loader
    description_ru: str = ""
    is_soil_preparation: bool = False  # Обработка почвы
    is_sowing: bool = False  # Посев / посадка
    is_crop_care: bool = False  # Уход за посевами
    is_harvest: bool = False  # Уборка
    is_transport: bool = False  # Транспорт
    is_fertilizing: bool = False  # Внесение удобрений
    is_spraying: bool = False  # Опрыскивание / химзащита
    is_forage: bool = False  # Кормозаготовка
    is_post_harvest: bool = False  # Послеуборочные


# ─── Полный маппинг операций ──────────────────────────────────────────
# Ключ: нормализованное название операции из документа (нижний регистр)
# Значение: OperationMapping

OPERATION_MAPPING: Dict[str, OperationMapping] = {
    # ═══ ОБРАБОТКА ПОЧВЫ ═══════════════════════════════════════════
    "лущение стерни": OperationMapping(
        doc_operation="Лущение стерни",
        tum_key="Disc harrowing",
        machine_type="tractor",
        description_ru="Лущение стерни дисковыми орудиями",
        is_soil_preparation=True,
    ),
    "вспашка": OperationMapping(
        doc_operation="Вспашка",
        tum_key="Ploughing",
        machine_type="tractor",
        description_ru="Глубокая вспашка плугом",
        is_soil_preparation=True,
    ),
    "перепашка": OperationMapping(
        doc_operation="Перепашка",
        tum_key="Ploughing",
        machine_type="tractor",
        description_ru="Повторная вспашка (например, после уборки корнеплодов)",
        is_soil_preparation=True,
    ),
    "планировка почвы": OperationMapping(
        doc_operation="Планировка почвы",
        tum_key="Seedbed combination",
        machine_type="tractor",
        description_ru="Планировка поверхности почвы",
        is_soil_preparation=True,
    ),
    "культивация": OperationMapping(
        doc_operation="Культивация",
        tum_key="Cultivating (deep)",
        machine_type="tractor",
        description_ru="Культивация почвы",
        is_soil_preparation=True,
    ),
    "культивация с боронованием": OperationMapping(
        doc_operation="Культивация с боронованием",
        tum_key="Cultivating (deep)",
        machine_type="tractor",
        description_ru="Культивация с одновременным боронованием",
        is_soil_preparation=True,
    ),
    "боронование": OperationMapping(
        doc_operation="Боронование",
        tum_key="Cultivating (shallow)",
        machine_type="tractor",
        description_ru="Боронование почвы",
        is_soil_preparation=True,
    ),
    "боронование до всходов": OperationMapping(
        doc_operation="Боронование до всходов",
        tum_key="Cultivating (shallow)",
        machine_type="tractor",
        description_ru="Боронование до появления всходов",
        is_crop_care=True,
    ),
    "боронование по всходам": OperationMapping(
        doc_operation="Боронование по всходам",
        tum_key="Cultivating (shallow)",
        machine_type="tractor",
        description_ru="Боронование по всходам",
        is_crop_care=True,
    ),
    "боронование посевов": OperationMapping(
        doc_operation="Боронование посевов",
        tum_key="Cultivating (shallow)",
        machine_type="tractor",
        description_ru="Боронование посевов (озимые)",
        is_crop_care=True,
    ),
    "нарезка гребней": OperationMapping(
        doc_operation="Нарезка гребней",
        tum_key="Seedbed combination",
        machine_type="tractor",
        description_ru="Нарезка гребней для посадки",
        is_soil_preparation=True,
    ),
    # ═══ ПОСЕВ / ПОСАДКА ════════════════════════════════════════════
    "посев": OperationMapping(
        doc_operation="Посев",
        tum_key="Seed drill combination 4m",
        machine_type="tractor",
        description_ru="Посев зерновых / трав",
        is_sowing=True,
    ),
    "посев зерновых": OperationMapping(
        doc_operation="Посев зерновых",
        tum_key="Seed drill combination 4m",
        machine_type="tractor",
        description_ru="Посев зерновых культур",
        is_sowing=True,
    ),
    "посев гороха": OperationMapping(
        doc_operation="Посев гороха",
        tum_key="Seed drill combination 4m",
        machine_type="tractor",
        description_ru="Посев гороха",
        is_sowing=True,
    ),
    "посев клевера": OperationMapping(
        doc_operation="Посев клевера",
        tum_key="Seed drill combination 3m",
        machine_type="tractor",
        description_ru="Посев клевера, люцерны, трав, рапса",
        is_sowing=True,
    ),
    "посев кукурузы": OperationMapping(
        doc_operation="Посев кукурузы",
        tum_key="Precision air seeding",
        machine_type="tractor",
        description_ru="Точный посев кукурузы пневматической сеялкой",
        is_sowing=True,
    ),
    "посев подсолнечника": OperationMapping(
        doc_operation="Посев подсолнечника",
        tum_key="Precision air seeding",
        machine_type="tractor",
        description_ru="Точный посев подсолнечника",
        is_sowing=True,
    ),
    "посев свёклы": OperationMapping(
        doc_operation="Посев свёклы",
        tum_key="Precision air seeding",
        machine_type="tractor",
        description_ru="Точный посев свёклы (12+ рядков)",
        is_sowing=True,
    ),
    "посадка картофеля": OperationMapping(
        doc_operation="Посадка картофеля",
        tum_key="Precision air seeding",
        machine_type="tractor",
        description_ru="Посадка картофеля картофелесажалкой",
        is_sowing=True,
    ),
    "прикатывание": OperationMapping(
        doc_operation="Прикатывание",
        tum_key="Seedbed combination",
        machine_type="tractor",
        description_ru="Прикатывание посевов",
        is_sowing=True,
    ),
    # ═══ УХОД ЗА ПОСЕВАМИ ════════════════════════════════════════════
    "междурядная обработка": OperationMapping(
        doc_operation="Междурядная обработка",
        tum_key="Cultivating (shallow)",
        machine_type="tractor",
        description_ru="Междурядная обработка (культивация)",
        is_crop_care=True,
    ),
    "междурядная обработка картофеля": OperationMapping(
        doc_operation="Междурядная обработка картофеля",
        tum_key="Cultivating (shallow)",
        machine_type="tractor",
        description_ru="Междурядная обработка / окучивание картофеля",
        is_crop_care=True,
    ),
    "междурядная обработка кукурузы": OperationMapping(
        doc_operation="Междурядная обработка кукурузы",
        tum_key="Cultivating (shallow)",
        machine_type="tractor",
        description_ru="Междурядная обработка кукурузы",
        is_crop_care=True,
    ),
    "междурядная обработка свёклы": OperationMapping(
        doc_operation="Междурядная обработка свёклы",
        tum_key="Cultivating (shallow)",
        machine_type="tractor",
        description_ru="Междурядная обработка свёклы",
        is_crop_care=True,
    ),
    # ═══ ХИМЗАЩИТА / ОПРЫСКИВАНИЕ ════════════════════════════════════
    "опрыскивание": OperationMapping(
        doc_operation="Опрыскивание",
        tum_key="Spraying",
        machine_type="tractor",
        description_ru="Опрыскивание посевов",
        is_spraying=True,
    ),
    "внесение гербицида": OperationMapping(
        doc_operation="Внесение гербицида",
        tum_key="Spraying",
        machine_type="tractor",
        description_ru="Внесение раствора гербицида",
        is_spraying=True,
    ),
    "внесение фунгицида": OperationMapping(
        doc_operation="Внесение фунгицида",
        tum_key="Spraying",
        machine_type="tractor",
        description_ru="Внесение раствора фунгицида",
        is_spraying=True,
    ),
    "внесение инсектицида": OperationMapping(
        doc_operation="Внесение инсектицида",
        tum_key="Spraying",
        machine_type="tractor",
        description_ru="Внесение раствора инсектицида",
        is_spraying=True,
    ),
    "внесение пестицидов": OperationMapping(
        doc_operation="Внесение пестицидов",
        tum_key="Spraying",
        machine_type="tractor",
        description_ru="Внесение раствора пестицидов",
        is_spraying=True,
    ),
    "десикация": OperationMapping(
        doc_operation="Десикация",
        tum_key="Spraying",
        machine_type="tractor",
        description_ru="Десикация (предуборочное высушивание)",
        is_spraying=True,
    ),
    "дефолиация": OperationMapping(
        doc_operation="Дефолиация",
        tum_key="Spraying",
        machine_type="tractor",
        description_ru="Дефолиация (удаление листьев)",
        is_spraying=True,
    ),
    # ═══ ВНЕСЕНИЕ УДОБРЕНИЙ ════════════════════════════════════════════
    "внесение минеральных удобрений": OperationMapping(
        doc_operation="Внесение минеральных удобрений",
        tum_key="Fertilizing",
        machine_type="tractor",
        description_ru="Внесение минеральных удобрений",
        is_fertilizing=True,
    ),
    "внесение органических удобрений": OperationMapping(
        doc_operation="Внесение органических удобрений",
        tum_key="Fertilizing",
        machine_type="tractor",
        description_ru="Транспортировка и внесение органических удобрений",
        is_fertilizing=True,
    ),
    "внесение раствора минеральных удобрений": OperationMapping(
        doc_operation="Внесение раствора минеральных удобрений",
        tum_key="Fertilizing",
        machine_type="tractor",
        description_ru="Внесение жидких минеральных удобрений",
        is_fertilizing=True,
    ),
    "внесение раствора азотных удобрений": OperationMapping(
        doc_operation="Внесение раствора азотных удобрений",
        tum_key="Fertilizing",
        machine_type="tractor",
        description_ru="Внесение жидких азотных удобрений",
        is_fertilizing=True,
    ),
    "внесение кас-удобрений": OperationMapping(
        doc_operation="Внесение КАС-удобрений",
        tum_key="Fertilizing",
        machine_type="tractor",
        description_ru="Внесение КАС (карбамидно-аммиачная смесь)",
        is_fertilizing=True,
    ),
    "погрузка удобрений": OperationMapping(
        doc_operation="Погрузка удобрений",
        tum_key="Fertilizing",
        machine_type="tractor",
        description_ru="Погрузка удобрений (погрузчик на тракторе)",
        is_fertilizing=True,
    ),
    "погрузка органических удобрений": OperationMapping(
        doc_operation="Погрузка органических удобрений",
        tum_key="Fertilizing",
        machine_type="tractor",
        description_ru="Погрузка органических удобрений",
        is_fertilizing=True,
    ),
    "полив с удобрениями": OperationMapping(
        doc_operation="Полив с удобрениями",
        tum_key="Fertilizing",
        machine_type="tractor",
        description_ru="Полив с одновременным внесением удобрений",
        is_fertilizing=True,
    ),
    "подкормка": OperationMapping(
        doc_operation="Подкормка",
        tum_key="Fertilizing",
        machine_type="tractor",
        description_ru="Подкормка посевов",
        is_fertilizing=True,
    ),
    # ═══ УБОРКА — КОМБАЙН (ИСКЛЮЧИТЬ ИЗ ТРАКТОРНОГО РИСКА) ═══════════
    "прямое комбайнирование": OperationMapping(
        doc_operation="Прямое комбайнирование",
        tum_key=None,
        machine_type="combine",
        description_ru="Уборка зерноуборочным комбайном — НЕ ТРАКТОР",
        is_harvest=True,
    ),
    "уборка комбайном": OperationMapping(
        doc_operation="Уборка комбайном",
        tum_key=None,
        machine_type="combine",
        description_ru="Уборка комбайном — НЕ ТРАКТОР",
        is_harvest=True,
    ),
    "уборка прямым комбайнированием": OperationMapping(
        doc_operation="Уборка прямым комбайнированием",
        tum_key=None,
        machine_type="combine",
        description_ru="Уборка прямым комбайнированием — НЕ ТРАКТОР",
        is_harvest=True,
    ),
    "подбор и обмолот валков": OperationMapping(
        doc_operation="Подбор и обмолот валков",
        tum_key=None,
        machine_type="combine",
        description_ru="Подбор и обмолот валков комбайном — НЕ ТРАКТОР",
        is_harvest=True,
    ),
    "уборка на силос": OperationMapping(
        doc_operation="Уборка на силос",
        tum_key=None,
        machine_type="combine",
        description_ru="Уборка силосоуборочным комбайном — НЕ ТРАКТОР",
        is_harvest=True,
    ),
    "уборка корнеплодов": OperationMapping(
        doc_operation="Уборка корнеплодов",
        tum_key=None,
        machine_type="combine",
        description_ru="Уборка корнеплодов комбайном — НЕ ТРАКТОР",
        is_harvest=True,
    ),
    "уборка с измельчением корзинок": OperationMapping(
        doc_operation="Уборка с измельчением корзинок",
        tum_key=None,
        machine_type="combine",
        description_ru="Уборка подсолнечника комбайном — НЕ ТРАКТОР",
        is_harvest=True,
    ),
    "первый укос": OperationMapping(
        doc_operation="Первый укос",
        tum_key=None,
        machine_type="combine",
        description_ru="Укос кормовых трав комбайном — НЕ ТРАКТОР",
        is_harvest=True,
    ),
    "второй укос": OperationMapping(
        doc_operation="Второй укос",
        tum_key=None,
        machine_type="combine",
        description_ru="Второй укос кормовых трав комбайном — НЕ ТРАКТОР",
        is_harvest=True,
    ),
    "скашивание сорняков": OperationMapping(
        doc_operation="Скашивание сорняков",
        tum_key="Mowing (front)",
        machine_type="tractor",
        description_ru="Скашивание сорняков косилкой",
        is_crop_care=True,
    ),
    # ═══ КОРМОЗАГОТОВКА (тракторные операции) ═══════════════════════════
    "кошение в расстил": OperationMapping(
        doc_operation="Кошение в расстил",
        tum_key="Mowing (front)",
        machine_type="tractor",
        description_ru="Кошение трав в расстил",
        is_forage=True,
    ),
    "кошение в валки": OperationMapping(
        doc_operation="Кошение в валки",
        tum_key="Mowing (front)",
        machine_type="tractor",
        description_ru="Кошение трав в валки",
        is_forage=True,
    ),
    "сгребание в валки": OperationMapping(
        doc_operation="Сгребание в валки",
        tum_key="Swathing",
        machine_type="tractor",
        description_ru="Сгребание скошенной массы в валки",
        is_forage=True,
    ),
    "сгребание сена в валки": OperationMapping(
        doc_operation="Сгребание сена в валки",
        tum_key="Swathing",
        machine_type="tractor",
        description_ru="Сгребание сена в валки",
        is_forage=True,
    ),
    "ворошение сена": OperationMapping(
        doc_operation="Ворошение сена",
        tum_key="Swathing",
        machine_type="tractor",
        description_ru="Ворошение сена для просушки",
        is_forage=True,
    ),
    "подбор валков с измельчением": OperationMapping(
        doc_operation="Подбор валков с измельчением",
        tum_key="Swathing",
        machine_type="tractor",
        description_ru="Подбор валков с измельчением (подборщик)",
        is_forage=True,
    ),
    "подбор валков с прессованием и транспортировка": OperationMapping(
        doc_operation="Подбор валков с прессованием и транспортировка",
        tum_key="Swathing",
        machine_type="tractor",
        description_ru="Подбор валков с прессованием",
        is_forage=True,
    ),
    "скашивание ботвы": OperationMapping(
        doc_operation="Скашивание ботвы",
        tum_key="Mowing (front)",
        machine_type="tractor",
        description_ru="Скашивание ботвы картофеля / свёклы",
        is_harvest=True,
    ),
    # ═══ ТРАНСПОРТ ════════════════════════════════════════════════════
    "транспортировка": OperationMapping(
        doc_operation="Транспортировка",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Транспортные перевозки",
        is_transport=True,
    ),
    "транспортировка зерна": OperationMapping(
        doc_operation="Транспортировка зерна",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Вывоз зерна с поля",
        is_transport=True,
    ),
    "транспортировка сенажной массы": OperationMapping(
        doc_operation="Транспортировка сенажной массы",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Вывоз сенажной массы",
        is_transport=True,
    ),
    "транспортировка силоса": OperationMapping(
        doc_operation="Транспортировка силоса",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Вывоз силоса",
        is_transport=True,
    ),
    "транспортировка соломы": OperationMapping(
        doc_operation="Транспортировка соломы",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Вывоз соломы",
        is_transport=True,
    ),
    "транспортировка клубней": OperationMapping(
        doc_operation="Транспортировка клубней",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Вывоз клубней картофеля",
        is_transport=True,
    ),
    "транспортировка корнеплодов": OperationMapping(
        doc_operation="Транспортировка корнеплодов",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Вывоз корнеплодов",
        is_transport=True,
    ),
    "транспортировка семян": OperationMapping(
        doc_operation="Транспортировка семян",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Подвоз семян к сеялкам",
        is_transport=True,
    ),
    "транспортировка семян и загрузка сеялок": OperationMapping(
        doc_operation="Транспортировка семян и загрузка сеялок",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Подвоз семян и загрузка сеялок",
        is_transport=True,
    ),
    "транспортировка семян и удобрений": OperationMapping(
        doc_operation="Транспортировка семян и удобрений",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Подвоз семян и удобрений",
        is_transport=True,
    ),
    "транспортировка удобрений": OperationMapping(
        doc_operation="Транспортировка удобрений",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Подвоз удобрений",
        is_transport=True,
    ),
    "транспортировка раствора ядохимикатов": OperationMapping(
        doc_operation="Транспортировка раствора ядохимикатов",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Подвоз рабочего раствора к опрыскивателю",
        is_transport=True,
    ),
    "транспортировка раствора гербицидов": OperationMapping(
        doc_operation="Транспортировка раствора гербицидов",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Подвоз раствора гербицидов",
        is_transport=True,
    ),
    "транспортировка раствора фунгицидов": OperationMapping(
        doc_operation="Транспортировка раствора фунгицидов",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Подвоз раствора фунгицидов",
        is_transport=True,
    ),
    "транспортировка раствора инсектицидов": OperationMapping(
        doc_operation="Транспортировка раствора инсектицидов",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Подвоз раствора инсектицидов",
        is_transport=True,
    ),
    "транспортировка раствора для десикации": OperationMapping(
        doc_operation="Транспортировка раствора для десикации",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Подвоз раствора для десикации",
        is_transport=True,
    ),
    "транспортировка раствора для дефолиации": OperationMapping(
        doc_operation="Транспортировка раствора для дефолиации",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Подвоз раствора для дефолиации",
        is_transport=True,
    ),
    "транспортировка раствора минеральных удобрений": OperationMapping(
        doc_operation="Транспортировка раствора минеральных удобрений",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Подвоз жидких удобрений",
        is_transport=True,
    ),
    "транспортировка раствора азотных удобрений": OperationMapping(
        doc_operation="Транспортировка раствора азотных удобрений",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Подвоз жидких азотных удобрений",
        is_transport=True,
    ),
    "транспортировка измельчённых корзинок": OperationMapping(
        doc_operation="Транспортировка измельчённых корзинок",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Вывоз измельчённых корзинок подсолнечника",
        is_transport=True,
    ),
    "транспортировка урожая": OperationMapping(
        doc_operation="Транспортировка урожая",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Вывоз урожая",
        is_transport=True,
    ),
    "вывозка навоза": OperationMapping(
        doc_operation="Вывозка навоза",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Вывозка навоза в бурты",
        is_transport=True,
    ),
    # ═══ ПОСЛЕУБОРОЧНЫЕ ════════════════════════════════════════════════
    "прессование соломы": OperationMapping(
        doc_operation="Прессование соломы",
        tum_key="Swathing",
        machine_type="tractor",
        description_ru="Прессование соломы в рулоны",
        is_post_harvest=True,
    ),
    "скирдование соломы": OperationMapping(
        doc_operation="Скирдование соломы",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Скирдование соломы / корзинок",
        is_post_harvest=True,
    ),
    "скирдование сена": OperationMapping(
        doc_operation="Скирдование сена",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Скирдование сена",
        is_post_harvest=True,
    ),
    "скирдование корзинок": OperationMapping(
        doc_operation="Скирдование корзинок",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Скирдование измельчённых корзинок",
        is_post_harvest=True,
    ),
    "трамбовка силоса": OperationMapping(
        doc_operation="Трамбовка силоса",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Трамбовка силоса гусеничным трактором",
        is_post_harvest=True,
    ),
    "трамбовка сенажной массы": OperationMapping(
        doc_operation="Трамбовка сенажной массы",
        tum_key="Transport",
        machine_type="tractor",
        description_ru="Трамбовка сенажной массы",
        is_post_harvest=True,
    ),
    "полив": OperationMapping(
        doc_operation="Полив",
        tum_key="Fertilizing",
        machine_type="tractor",
        description_ru="Полив дождевальной машиной",
        is_crop_care=True,
    ),
}


# ─── Функции доступа ──────────────────────────────────────────────────


def _normalize_operation_name(name: str) -> str:
    """Нормализация названия операции для поиска в маппинге."""
    import re

    name = name.strip().lower()
    # Удалить номера проходов: "первая", "вторая", "третья", "четвёртая"
    name = re.sub(r"\s+(первая|вторая|третья|четвёртая|1-я|2-я|3-я|4-я)$", "", name)
    # Удалить суффиксы норм внесения: "20 т/га", "300 кг/га" и т.д.
    name = re.sub(r"\s+\d+[\.,]?\d*\s*(т|кг|ц|л)/га$", "", name)
    # Удалить суффиксы объёма: "15 кг/га", "220 кг/га" и т.д.
    name = re.sub(r"\s+\d+[\.,]?\d*\s*(т|кг|ц|л)/га$", "", name)
    # Удалить "с внесением удобрений ..."
    name = re.sub(r"\s+с внесением.*$", "", name)
    # Удалить "с внесением мин. удобрений"
    name = re.sub(r"\s+с внесением.*$", "", name)
    # Удалить "и загрузка сеялок"
    name = name.replace(" и загрузка сеялок", "")
    # Удалить "на поле"
    name = name.replace(" на поле", "")
    # Удалить "в бурты на краю поля"
    name = re.sub(r"\s+в бурты.*$", "", name)
    # Заменить "транспортировка и внесение" → "внесение"
    name = name.replace("транспортировка и внесение", "внесение")
    # Заменить "подвоз и загрузка семян в сеялки" → "посев"
    if "подвоз" in name and "семян" in name:
        name = "транспортировка семян"
    # Заменить "загрузка картофелесажалок" → "посадка картофеля"
    if "загрузка картофелесажалок" in name:
        name = "транспортировка семян"
    # Заменить "внесение раствора" → "опрыскивание" для ядохимикатов
    if "внесение раствора" in name and any(
        w in name for w in ["гербицид", "фунгицид", "инсектицид", "пестицид"]
    ):
        name = "внесение " + name.split("раствора ")[-1].split()[0]
    # Удалить объёмы: "100 л/га", "300 л/га", "200 л/га", "400 л/га"
    name = re.sub(r"\s+\d+\s*л/га$", "", name)
    name = re.sub(r"\s+\d+\s*кг/га$", "", name)
    name = re.sub(r"\s+\d+\s*т/га$", "", name)
    name = re.sub(r"\s+\d+\s*ц/га$", "", name)
    # Удалить "×3" и подобные
    name = re.sub(r"\s*×\d+$", "", name)
    return name.strip()


def lookup_operation(doc_name: str) -> Optional[OperationMapping]:
    """
    Найти маппинг операции по названию из документа.

    Parameters
    ----------
    doc_name : str
        Название операции из технологической карты (Приложение В).

    Returns
    -------
    OperationMapping или None
        Маппинг операции. None если операция не найдена.
    """
    normalized = _normalize_operation_name(doc_name)

    # Точное совпадение
    if normalized in OPERATION_MAPPING:
        return OPERATION_MAPPING[normalized]

    # Поиск по частичному совпадению (сначала более длинные ключи)
    for key in sorted(OPERATION_MAPPING.keys(), key=len, reverse=True):
        if key in normalized or normalized in key:
            return OPERATION_MAPPING[key]

    return None


def is_tractor_operation(doc_name: str) -> bool:
    """
    Проверить, является ли операция тракторной.
    Комбайновые операции возвращают False.
    """
    mapping = lookup_operation(doc_name)
    if mapping is None:
        return False
    return mapping.machine_type == "tractor"


def get_tum_key(doc_name: str) -> Optional[str]:
    """Получить ключ в TUM_OPERATIONS для операции из документа."""
    mapping = lookup_operation(doc_name)
    if mapping is None:
        return None
    return mapping.tum_key


def list_all_operations() -> list[str]:
    """Список всех операций в маппинге."""
    return sorted(OPERATION_MAPPING.keys())


def list_combine_operations() -> list[str]:
    """Список комбайновых операций (исключаются из тракторного риска)."""
    return sorted(key for key, m in OPERATION_MAPPING.items() if m.machine_type == "combine")


def list_tractor_operations() -> list[str]:
    """Список тракторных операций."""
    return sorted(key for key, m in OPERATION_MAPPING.items() if m.machine_type == "tractor")
