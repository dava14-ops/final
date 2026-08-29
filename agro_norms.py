"""
agro_norms.py (v1.0)
Нормативы выработки (Приложение Б) и поправочные коэффициенты (Таблица 1.1)
из методического пособия РГАУ-МСХА им. К.А. Тимирязева.

Формула: engine_hours_per_ha = T_shift / (W_shift * K_ob)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ─── Константы смен ───────────────────────────────────────────────────
STANDARD_SHIFT_HOURS = 7.0
CHEMICAL_SHIFT_HOURS = 6.0  # Для работы с ядохимикатами (пестициды, гербициды)


# ─── Справочник тракторов (из Приложений А и Б) ───────────────────────
class Tractor:
    K744R1 = "К-744Р1"
    T150K = "Т-150К"
    DT75M = "ДТ-75М"
    MTZ82 = "МТЗ-82"
    T40AM = "Т-40АМ"
    T70S = "Т-70С"


@dataclass(frozen=True)
class NormEntry:
    """Норматив для связки (Операция, Трактор)."""

    w_shift: float  # Сменная выработка, га/смену (из Прил. Б)
    t_shift: float  # Длительность смены, часов (7 или 6)
    fuel_per_ha: float  # Расход топлива, л/га (для справки)

    @property
    def base_hours_per_ha(self) -> float:
        """Базовые мч/га без учета коэффициентов поля (Kоб = 1.0)."""
        if self.w_shift <= 0:
            return 0.0
        return self.t_shift / self.w_shift


# ─── База нормативов (Приложение Б, стр. 31-34) ───────────────────────
# Ключ: нормализованное имя операции (нижний регистр)
# Значение: Dict[Трактор, NormEntry]
OPERATION_NORMS: Dict[str, Dict[str, NormEntry]] = {
    "вспашка": {
        Tractor.K744R1: NormEntry(17.7, 7.0, 17.2),
        Tractor.T150K: NormEntry(13.2, 7.0, 15.2),
        Tractor.DT75M: NormEntry(8.0, 7.0, 14.2),
        Tractor.MTZ82: NormEntry(4.8, 7.0, 19.0),
    },
    "перепашка": {
        Tractor.K744R1: NormEntry(18.1, 7.0, 15.9),
        Tractor.T150K: NormEntry(14.2, 7.0, 13.1),
        Tractor.DT75M: NormEntry(9.1, 7.0, 12.4),
        Tractor.MTZ82: NormEntry(5.6, 7.0, 18.2),
    },
    "лущение стерни": {
        Tractor.K744R1: NormEntry(92.0, 7.0, 3.0),
        Tractor.T150K: NormEntry(63.0, 7.0, 2.8),
        Tractor.DT75M: NormEntry(37.0, 7.0, 2.8),
        Tractor.MTZ82: NormEntry(27.0, 7.0, 3.1),
    },
    "культивация": {
        Tractor.K744R1: NormEntry(76.0, 7.0, 3.0),
        Tractor.T150K: NormEntry(65.0, 7.0, 3.0),
        Tractor.DT75M: NormEntry(45.0, 7.0, 2.9),
        Tractor.MTZ82: NormEntry(23.0, 7.0, 3.5),
    },
    "культивация с боронованием": {
        Tractor.K744R1: NormEntry(70.0, 7.0, 3.9),
        Tractor.T150K: NormEntry(52.0, 7.0, 3.6),
        Tractor.DT75M: NormEntry(31.0, 7.0, 3.6),
        Tractor.MTZ82: NormEntry(18.0, 7.0, 4.0),
    },
    "боронование": {  # Боронование до/по всходам
        Tractor.DT75M: NormEntry(82.0, 7.0, 1.1),  # До всходов (21*БЗСС-1.0)
        Tractor.MTZ82: NormEntry(63.0, 7.0, 1.4),
        Tractor.T40AM: NormEntry(39.0, 7.0, 1.3),
    },
    "посев зерновых": {
        Tractor.K744R1: NormEntry(65.0, 7.0, 3.1),
        Tractor.T150K: NormEntry(52.0, 7.0, 2.8),
        Tractor.DT75M: NormEntry(42.0, 7.0, 2.0),
        Tractor.MTZ82: NormEntry(33.0, 7.0, 2.4),
    },
    "посев кукурузы": {  # И подсолнечника
        Tractor.T150K: NormEntry(30.0, 7.0, 2.6),  # С удобрениями
        Tractor.DT75M: NormEntry(26.0, 7.0, 2.2),
        Tractor.MTZ82: NormEntry(12.0, 7.0, 3.1),
    },
    "посев свёклы": {
        Tractor.MTZ82: NormEntry(24.0, 7.0, 2.3),
        Tractor.T70S: NormEntry(24.0, 7.0, 2.3),
    },
    "посадка картофеля": {
        Tractor.MTZ82: NormEntry(9.9, 7.0, 6.4),
    },
    "междурядная обработка картофеля": {
        Tractor.MTZ82: NormEntry(15.0, 7.0, 5.2),  # Первая
        Tractor.T70S: NormEntry(15.0, 7.0, 5.5),
    },
    "междурядная обработка кукурузы": {
        Tractor.MTZ82: NormEntry(23.2, 7.0, 3.4),  # Первая
        Tractor.T70S: NormEntry(18.7, 7.0, 2.8),
    },
    "междурядная обработка свёклы": {
        Tractor.MTZ82: NormEntry(14.0, 7.0, 3.8),  # Первая
        Tractor.T70S: NormEntry(14.0, 7.0, 3.8),
    },
    "прикатывание": {
        Tractor.T150K: NormEntry(122.3, 7.0, 1.1),
        Tractor.DT75M: NormEntry(85.3, 7.0, 1.2),
        Tractor.MTZ82: NormEntry(76.0, 7.0, 1.2),  # МТЗ-80/82
    },
    "кошение трав": {
        Tractor.MTZ82: NormEntry(16.0, 7.0, 2.7),  # В расстил
    },
    # ─── Химзащита (T_см = 6 часов!) ──────────────────────────────────
    # В Прил. Б нет прямой строки, берем типовую для МТЗ-82 + ОПМ-2.0/Реал-15
    # W_см ~ 40 га (при ширине 15м и скорости 7 км/ч с учетом tau=0.7)
    "опрыскивание": {
        Tractor.MTZ82: NormEntry(40.0, CHEMICAL_SHIFT_HOURS, 1.5),
        Tractor.T150K: NormEntry(60.0, CHEMICAL_SHIFT_HOURS, 2.0),
    },
    "внесение гербицида": {
        Tractor.MTZ82: NormEntry(40.0, CHEMICAL_SHIFT_HOURS, 1.5),
        Tractor.T150K: NormEntry(60.0, CHEMICAL_SHIFT_HOURS, 2.0),
    },
    "внесение фунгицида": {
        Tractor.MTZ82: NormEntry(40.0, CHEMICAL_SHIFT_HOURS, 1.5),
    },
    "внесение инсектицида": {
        Tractor.MTZ82: NormEntry(40.0, CHEMICAL_SHIFT_HOURS, 1.5),
    },
}

# ─── Поправочные коэффициенты (Таблица 1.1, стр. 8-9) ────────────────
# K_K: Каменистость
K_STONINESS = {
    "пахотные": {"отсутствует": 1.00, "слабая": 0.98, "средняя": 0.92, "сильная": 0.85},
    "непахотные": {"отсутствует": 1.00, "слабая": 0.99, "средняя": 0.93, "сильная": 0.82},
}

# K_P: Рельеф (угол склона)
K_RELIEF = {
    "пахотные": {"<=1": 1.00, "1-3": 0.97},
    "непахотные": {"<=1": 1.00, "1-3": 0.95},
}


def calculate_k_ob(
    operation_type: str = "пахотные",
    stoniness: str = "отсутствует",
    relief_slope: str = "<=1",
    # Остальные коэффициенты (K_h, K_C, K_П) пока принимаем за 1.0,
    # но архитектура готова к их добавлению
) -> float:
    """
    Вычисляет обобщенный поправочный коэффициент времени смены K_об.
    """
    op_type = "пахотные" if "пахот" in operation_type.lower() else "непахотные"

    k_k = K_STONINESS.get(op_type, {}).get(stoniness, 1.0)
    k_p = K_RELIEF.get(op_type, {}).get(relief_slope, 1.0)

    # K_h (высота над уровнем моря) для РФ обычно = 1.0 (кроме Кавказа/Алтая)
    k_h = 1.0
    # K_C (конфигурация) и K_П (препятствия) по умолчанию 1.0
    k_c, k_p_obstacles = 1.0, 1.0

    return k_k * k_h * k_c * k_p_obstacles * k_p


def get_engine_hours_per_ha(
    doc_operation: str,
    tractor: str,
    k_ob: float = 1.0,
) -> Tuple[float, float]:
    """
    Возвращает (мч/га, T_см) для указанной операции и трактора.

    Parameters
    ----------
    doc_operation : str
        Нормализованное название из Сроки.ПДФ (напр. "вспашка").
    tractor : str
        Марка трактора (напр. "МТЗ-82").
    k_ob : float
        Коэффициент полевых условий (Таблица 1.1).

    Returns
    -------
    Tuple[float, float]
        (engine_hours_per_ha, shift_hours)
    """
    op_key = doc_operation.strip().lower()
    norms_for_op = OPERATION_NORMS.get(op_key)

    if not norms_for_op:
        # Fallback для операций, которых нет в таблице (напр. Транспорт)
        return 0.15, STANDARD_SHIFT_HOURS

        # Ищем трактор (сначала точное совпадение, потом подстрока)
    entry = norms_for_op.get(tractor)
    if entry is None:
        # Fallback на самый популярный трактор для этой операции
        # (обычно МТЗ-82 или ДТ-75М)
        fallback_tractors = [Tractor.MTZ82, Tractor.DT75M, Tractor.T150K, Tractor.K744R1]
        for fb in fallback_tractors:
            if fb in norms_for_op:
                entry = norms_for_op[fb]
                break

    if entry is None:
        return 0.20, STANDARD_SHIFT_HOURS

    # Формула из пособия: W_факт = W_см * K_об  =>  t_га = T_см / (W_см * K_об)
    effective_w_shift = entry.w_shift * max(0.1, k_ob)
    hours_per_ha = entry.t_shift / effective_w_shift

    return hours_per_ha, entry.t_shift
