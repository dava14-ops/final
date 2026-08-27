#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
severity_model.py (v1.1)
Фаза 7, Итерация 2: оценка severity-модели из реальных claims.

Задачи:
    7.5: Оценка E[covered_loss] из реальных claims-данных
    7.6: Расчёт expected_severity для передачи в premium_engine.py

Использование:
    python severity_model.py

Вход:
    data/processed/claims/claims_clean.csv

Выход:
    severity_model_v1.json
    data/processed/claims/severity_report.json

Интеграция:
    - premium_engine.calculate_premium_with_severity(severity_model, ...)
    - Real_calculator.py (замена экспертных констант)
    - cli.py (передача expected_severity)
"""
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from constants import (
    FREQ_SHARES,
    SEVERITY_WEIGHTS,
    CRITICALITY_WEIGHTS,
    BRAND_MAP,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SystemSeverity",
    "SeverityModel",
    "load_claims_events",
    "estimate_system_severity",
    "compute_exact_covered_loss",
    "build_severity_model",
    "save_severity_model",
    "load_severity_model",
    "build_report",
]

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
CLAIMS_PATH = Path("data/processed/claims/claims_clean.csv")
OUTPUT_DIR = Path("data/processed/claims")
OUTPUT_MODEL_PATH = Path("severity_model_v1.json")
OUTPUT_REPORT_PATH = OUTPUT_DIR / "severity_report.json"

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
MIN_EVENTS_PER_SYSTEM = 10
MIN_EVENTS_PER_BRAND = 5
DEFAULT_HOURLY_DOWNTIME_COST = 2500.0  # руб/час простоя (типовая для РФ)
FALLBACK_FREQ_SHARES = dict(FREQ_SHARES)
FALLBACK_SEVERITY_WEIGHTS = dict(SEVERITY_WEIGHTS)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SystemSeverity:
    """Severity-статистика для одной системы отказов."""

    system: str
    n_events: int
    freq_share: float
    mean_repair_cost: float
    median_repair_cost: float
    p95_repair_cost: float
    mean_downtime_hours: float
    major_share: float
    std_repair_cost: float = 0.0

    def expected_loss(self, hourly_downtime_cost: float) -> float:
        """E[loss] для данной системы = E[repair] + E[downtime] * cost."""
        return self.mean_repair_cost + self.mean_downtime_hours * hourly_downtime_cost


@dataclass
class BrandSeverity:
    """Severity-статистика для одного бренда."""

    brand: str
    n_events: int
    mean_repair_cost: float
    median_repair_cost: float
    mean_downtime_hours: float
    major_share: float


@dataclass
class SeverityModel:
    """
    Полная severity-модель, обученная на claims.

    Совместима с premium_engine.calculate_premium_with_severity(),
    который вызывает метод expected_covered_loss(deductible, coverage_limit).
    """

    version: str
    source: str
    created_at: str
    n_events: int
    n_systems: int
    systems: Dict[str, SystemSeverity]
    overall_mean_repair: float
    overall_median_repair: float
    overall_p95_repair: float
    overall_mean_downtime: float
    overall_major_share: float
    overall_std_repair: float = 0.0
    hourly_downtime_cost: float = DEFAULT_HOURLY_DOWNTIME_COST
    fallback_used: bool = False
    notes: List[str] = field(default_factory=list)
    # Расширенные данные для точного расчёта
    _micro_losses: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )
    # Companion artifact containing event-level losses. JSON stays small while
    # exact deductible/limit calculations remain reproducible after reload.
    micro_losses_file: Optional[str] = None
    brands: Dict[str, BrandSeverity] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Публичные методы расчёта
    # ------------------------------------------------------------------
    def expected_repair_cost(self) -> float:
        """E[repair_cost] по всем системам."""
        return self.overall_mean_repair

    def expected_downtime_cost(self) -> float:
        """E[downtime_cost] = E[downtime_hours] * hourly_cost."""
        return self.overall_mean_downtime * self.hourly_downtime_cost

    def expected_loss_per_failure(self) -> float:
        """E[loss] = E[repair] + E[downtime_cost]."""
        return self.expected_repair_cost() + self.expected_downtime_cost()

    def expected_covered_loss(
        self,
        deductible: float = 0.0,
        coverage_limit: Optional[float] = None,
    ) -> float:
        """
        E[covered_loss] = E[max(0, min(loss, limit) - deductible)].

        Если доступны микро-данные (массив потерь по каждому событию),
        используется точный расчёт. Иначе — аппроксимация через среднее.

        Этот метод вызывается premium_engine.calculate_premium_with_severity().
        """
        deductible = max(0.0, float(deductible))

        # Точный расчёт по микро-данным
        if self._micro_losses is not None and self._micro_losses.size > 0:
            losses = self._micro_losses
            covered = losses - deductible
            covered = np.maximum(covered, 0.0)
            if coverage_limit is not None and coverage_limit > 0:
                covered = np.minimum(covered, float(coverage_limit))
            result = float(np.mean(covered))
            return max(0.0, result)

        # Аппроксимация: E[max(0, E[loss] - deductible)]
        # Это точная формула только для вырожденного распределения.
        # Для скошенных распределений даёт систематическую ошибку.
        expected = self.expected_loss_per_failure()
        covered = max(0.0, expected - deductible)
        if coverage_limit is not None and coverage_limit > 0:
            covered = min(covered, float(coverage_limit))
        return covered

    def expected_covered_loss_by_system(
        self,
        deductible: float = 0.0,
        coverage_limit: Optional[float] = None,
    ) -> Dict[str, float]:
        """E[covered_loss] раздельно по каждой системе отказов."""
        result: Dict[str, float] = {}
        for name, sys_sev in self.systems.items():
            loss = sys_sev.expected_loss(self.hourly_downtime_cost)
            covered = max(0.0, loss - max(0.0, deductible))
            if coverage_limit is not None and coverage_limit > 0:
                covered = min(covered, float(coverage_limit))
            result[name] = covered
        return result

    def expected_loss_by_brand(self, brand: str) -> Optional[float]:
        """E[loss] для конкретного бренда. None если данных нет."""
        brand_sev = self.brands.get(brand)
        if brand_sev is None or brand_sev.n_events == 0:
            return None
        return (
            brand_sev.mean_repair_cost
            + brand_sev.mean_downtime_hours * self.hourly_downtime_cost
        )

    # ------------------------------------------------------------------
    # Сериализация
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация для JSON (без микро-данных)."""
        return {
            "version": self.version,
            "source": self.source,
            "created_at": self.created_at,
            "n_events": self.n_events,
            "n_systems": self.n_systems,
            "systems": {
                k: asdict(v) for k, v in self.systems.items()
            },
            "brands": {
                k: asdict(v) for k, v in self.brands.items()
            },
            "overall_mean_repair": self.overall_mean_repair,
            "overall_median_repair": self.overall_median_repair,
            "overall_p95_repair": self.overall_p95_repair,
            "overall_mean_downtime": self.overall_mean_downtime,
            "overall_major_share": self.overall_major_share,
            "overall_std_repair": self.overall_std_repair,
            "hourly_downtime_cost": self.hourly_downtime_cost,
            "fallback_used": self.fallback_used,
            "notes": self.notes,
            "micro_losses_file": self.micro_losses_file,
            "micro_losses_count": (
                int(self._micro_losses.size)
                if self._micro_losses is not None
                else 0
            ),
        }


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------
def load_claims_events(path: Path) -> pd.DataFrame:
    """Загрузить claims и отфильтровать только события (event_flag=1)."""
    if not path.exists():
        raise FileNotFoundError(f"Файл claims не найден: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    logger.info("Загружено строк: %d", len(df))

    if "event_flag" not in df.columns:
        raise ValueError("Отсутствует колонка event_flag")
    if "failure_system" not in df.columns:
        raise ValueError("Отсутствует колонка failure_system")

    events = df[df["event_flag"] == 1].copy()
    logger.info("Событий (event_flag=1): %d", len(events))

    # Приведение типов
    for col in ("repair_cost", "downtime_hours", "claim_amount"):
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce")

    if "major_failure_flag" in events.columns:
        events["major_failure_flag"] = (
            pd.to_numeric(events["major_failure_flag"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        events["major_failure_flag"] = 0

    return events


# ---------------------------------------------------------------------------
# Оценка severity по системам
# ---------------------------------------------------------------------------
def estimate_system_severity(
    events: pd.DataFrame,
    system: str,
    total_events: int,
) -> SystemSeverity:
    """Оценить severity для одной системы отказов."""
    mask = events["failure_system"] == system
    sys_events = events[mask]
    n = len(sys_events)

    if n == 0:
        return SystemSeverity(
            system=system,
            n_events=0,
            freq_share=0.0,
            mean_repair_cost=0.0,
            median_repair_cost=0.0,
            p95_repair_cost=0.0,
            mean_downtime_hours=0.0,
            major_share=0.0,
            std_repair_cost=0.0,
        )

    # Repair cost
    if "repair_cost" in sys_events.columns:
        repair = sys_events["repair_cost"].dropna()
        if len(repair) > 0:
            mean_repair = float(repair.mean())
            median_repair = float(repair.median())
            p95_repair = float(repair.quantile(0.95))
            std_repair = float(repair.std()) if len(repair) > 1 else 0.0
        else:
            mean_repair = median_repair = p95_repair = std_repair = 0.0
    else:
        mean_repair = median_repair = p95_repair = std_repair = 0.0

    # Downtime
    if "downtime_hours" in sys_events.columns:
        downtime = sys_events["downtime_hours"].dropna()
        mean_downtime = float(downtime.mean()) if len(downtime) > 0 else 0.0
    else:
        mean_downtime = 0.0

    # Major share
    if "major_failure_flag" in sys_events.columns:
        major_share = float(sys_events["major_failure_flag"].mean())
    else:
        major_share = 0.0

    freq_share = n / total_events if total_events > 0 else 0.0

    return SystemSeverity(
        system=system,
        n_events=n,
        freq_share=freq_share,
        mean_repair_cost=mean_repair,
        median_repair_cost=median_repair,
        p95_repair_cost=p95_repair,
        mean_downtime_hours=mean_downtime,
        major_share=major_share,
        std_repair_cost=std_repair,
    )


# ---------------------------------------------------------------------------
# Оценка severity по брендам
# ---------------------------------------------------------------------------
def estimate_brand_severity(events: pd.DataFrame) -> Dict[str, BrandSeverity]:
    """Оценить severity для каждого бренда."""
    brands: Dict[str, BrandSeverity] = {}

    if "brand" not in events.columns:
        return brands

    for brand in events["brand"].dropna().unique():
        brand_events = events[events["brand"] == brand]
        n = len(brand_events)
        if n < MIN_EVENTS_PER_BRAND:
            continue

        repair = brand_events["repair_cost"].dropna() if "repair_cost" in brand_events.columns else pd.Series(dtype=float)
        downtime = brand_events["downtime_hours"].dropna() if "downtime_hours" in brand_events.columns else pd.Series(dtype=float)

        brands[str(brand)] = BrandSeverity(
            brand=str(brand),
            n_events=n,
            mean_repair_cost=float(repair.mean()) if len(repair) > 0 else 0.0,
            median_repair_cost=float(repair.median()) if len(repair) > 0 else 0.0,
            mean_downtime_hours=float(downtime.mean()) if len(downtime) > 0 else 0.0,
            major_share=(
                float(brand_events["major_failure_flag"].mean())
                if "major_failure_flag" in brand_events.columns
                else 0.0
            ),
        )

    return brands


# ---------------------------------------------------------------------------
# Точный расчёт E[covered_loss] из микро-данных
# ---------------------------------------------------------------------------
def compute_exact_covered_loss(
    events: pd.DataFrame,
    deductible: float = 0.0,
    coverage_limit: Optional[float] = None,
    hourly_downtime_cost: float = DEFAULT_HOURLY_DOWNTIME_COST,
) -> float:
    """
    Точный расчёт E[covered_loss] по каждому событию.

    covered_loss_i = max(0, min(repair_i + downtime_cost_i, limit) - deductible)
    E[covered_loss] = mean(covered_loss_i)
    """
    if len(events) == 0:
        return 0.0

    repair = events["repair_cost"].fillna(0.0).to_numpy(dtype=float)

    if "downtime_hours" in events.columns:
        downtime = events["downtime_hours"].fillna(0.0).to_numpy(dtype=float)
        downtime_cost = downtime * hourly_downtime_cost
    else:
        downtime_cost = np.zeros_like(repair)

    total_loss = repair + downtime_cost
    covered = total_loss - max(0.0, deductible)
    covered = np.maximum(covered, 0.0)

    if coverage_limit is not None and coverage_limit > 0:
        covered = np.minimum(covered, float(coverage_limit))

    return float(np.mean(covered))


def _extract_micro_losses(
    events: pd.DataFrame,
    hourly_downtime_cost: float,
) -> Optional[np.ndarray]:
    """
    Извлечь массив суммарных потерь по каждому событию.
    Используется для точного расчёта expected_covered_loss().
    """
    if len(events) == 0:
        return None

    if "repair_cost" not in events.columns:
        return None

    repair = events["repair_cost"].fillna(0.0).to_numpy(dtype=float)

    if "downtime_hours" in events.columns:
        downtime = events["downtime_hours"].fillna(0.0).to_numpy(dtype=float)
        downtime_cost = downtime * hourly_downtime_cost
    else:
        downtime_cost = np.zeros_like(repair)

    total_loss = repair + downtime_cost

    # Отбрасываем отрицательные и неконечные значения
    valid = np.isfinite(total_loss) & (total_loss >= 0.0)
    if not valid.all():
        n_invalid = int((~valid).sum())
        logger.warning(
            "Исключено %d событий с некорректными потерями", n_invalid
        )
        total_loss = total_loss[valid]

    if total_loss.size == 0:
        return None

    return total_loss


# ---------------------------------------------------------------------------
# Построение полной severity-модели
# ---------------------------------------------------------------------------
def build_severity_model(
    events: pd.DataFrame,
    hourly_downtime_cost: float = DEFAULT_HOURLY_DOWNTIME_COST,
) -> SeverityModel:
    """Построить severity-модель из событий."""
    total_events = len(events)
    notes: List[str] = []
    fallback_used = False

    if total_events == 0:
        notes.append("Нет событий для оценки — используется fallback")
        fallback_used = True
        return _build_fallback_model(notes=notes)

    # Определяем уникальные системы
    observed_systems = events["failure_system"].dropna().unique().tolist()
    expected_systems = list(FREQ_SHARES.keys())
    all_systems = sorted(set(observed_systems) | set(expected_systems))

    # Оценка по каждой системе
    systems: Dict[str, SystemSeverity] = {}
    for system in all_systems:
        sys_sev = estimate_system_severity(events, system, total_events)
        if 0 < sys_sev.n_events < MIN_EVENTS_PER_SYSTEM:
            notes.append(
                f"{system}: только {sys_sev.n_events} событий "
                f"(< {MIN_EVENTS_PER_SYSTEM}), оценка нестабильна"
            )
            fallback_used = True
        systems[system] = sys_sev

    # Оценка по брендам
    brands = estimate_brand_severity(events)
    if not brands:
        notes.append("Недостаточно данных для оценки severity по брендам")

    # Общие статистики
    repair_all = (
        events["repair_cost"].dropna()
        if "repair_cost" in events.columns
        else pd.Series(dtype=float)
    )
    downtime_all = (
        events["downtime_hours"].dropna()
        if "downtime_hours" in events.columns
        else pd.Series(dtype=float)
    )

    overall_mean_repair = float(repair_all.mean()) if len(repair_all) > 0 else 0.0
    overall_median_repair = float(repair_all.median()) if len(repair_all) > 0 else 0.0
    overall_p95_repair = float(repair_all.quantile(0.95)) if len(repair_all) > 0 else 0.0
    overall_std_repair = float(repair_all.std()) if len(repair_all) > 1 else 0.0
    overall_mean_downtime = float(downtime_all.mean()) if len(downtime_all) > 0 else 0.0

    if "major_failure_flag" in events.columns:
        overall_major_share = float(events["major_failure_flag"].mean())
    else:
        overall_major_share = 0.0

    # Микро-данные для точного расчёта
    micro_losses = _extract_micro_losses(events, hourly_downtime_cost)
    if micro_losses is not None:
        notes.append(
            f"Микро-данные сохранены ({micro_losses.size} событий) "
            "для точного расчёта expected_covered_loss"
        )

    return SeverityModel(
        version="1.1",
        source="claims_clean.csv",
        created_at=pd.Timestamp.now().isoformat(),
        n_events=total_events,
        n_systems=len(systems),
        systems=systems,
        overall_mean_repair=overall_mean_repair,
        overall_median_repair=overall_median_repair,
        overall_p95_repair=overall_p95_repair,
        overall_mean_downtime=overall_mean_downtime,
        overall_major_share=overall_major_share,
        overall_std_repair=overall_std_repair,
        hourly_downtime_cost=hourly_downtime_cost,
        fallback_used=fallback_used,
        notes=notes,
        _micro_losses=micro_losses,
        brands=brands,
        micro_losses_file=None,
    )


def _build_fallback_model(notes: List[str]) -> SeverityModel:
    """Fallback: модель на основе экспертных констант."""
    systems: Dict[str, SystemSeverity] = {}
    for system, freq in FALLBACK_FREQ_SHARES.items():
        systems[system] = SystemSeverity(
            system=system,
            n_events=0,
            freq_share=freq,
            mean_repair_cost=0.0,
            median_repair_cost=0.0,
            p95_repair_cost=0.0,
            mean_downtime_hours=0.0,
            major_share=0.30,
            std_repair_cost=0.0,
        )

    return SeverityModel(
        version="1.1",
        source="fallback (constants.py)",
        created_at=pd.Timestamp.now().isoformat(),
        n_events=0,
        n_systems=len(systems),
        systems=systems,
        overall_mean_repair=0.0,
        overall_median_repair=0.0,
        overall_p95_repair=0.0,
        overall_mean_downtime=0.0,
        overall_major_share=0.30,
        overall_std_repair=0.0,
        hourly_downtime_cost=DEFAULT_HOURLY_DOWNTIME_COST,
        fallback_used=True,
        notes=notes,
        _micro_losses=None,
        micro_losses_file=None,
        brands={},
    )


# ---------------------------------------------------------------------------
# Сериализация / десериализация
# ---------------------------------------------------------------------------
def save_severity_model(path: Path, model: SeverityModel) -> None:
    """Сохранить severity-модель и точные event-level losses.

    The loss vector is stored in a companion NPZ rather than embedded in JSON,
    so the production artifact remains manageable while
    ``E[(L-d)_+]`` stays exact after reload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    micro_path: Optional[Path] = None
    if model._micro_losses is not None and model._micro_losses.size > 0:
        micro_path = path.with_name(f"{path.stem}.micro_losses.npz")
        np.savez_compressed(micro_path, losses=np.asarray(model._micro_losses, dtype=float))
        model.micro_losses_file = micro_path.name
    else:
        model.micro_losses_file = None

    with open(path, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info("Severity-модель сохранена: %s", path)


def load_severity_model(path: Path) -> SeverityModel:
    """Загрузить severity-модель из JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Severity-модель не найдена: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    systems: Dict[str, SystemSeverity] = {}
    for name, sys_data in data.get("systems", {}).items():
        systems[name] = SystemSeverity(**sys_data)

    brands: Dict[str, BrandSeverity] = {}
    for name, brand_data in data.get("brands", {}).items():
        brands[name] = BrandSeverity(**brand_data)

    micro_losses: Optional[np.ndarray] = None
    micro_name = data.get("micro_losses_file")
    if micro_name:
        micro_path = path.parent / str(micro_name)
        if not micro_path.exists():
            raise FileNotFoundError(
                f"Severity micro-loss artifact not found: {micro_path}"
            )
        with np.load(micro_path, allow_pickle=False) as npz:
            if "losses" not in npz.files:
                raise ValueError(f"Invalid severity micro-loss artifact: {micro_path}")
            micro_losses = np.asarray(npz["losses"], dtype=float)
        if not np.all(np.isfinite(micro_losses)):
            raise ValueError("Severity micro-loss artifact contains non-finite values")

    return SeverityModel(
        version=data.get("version", "1.0"),
        source=data.get("source", "unknown"),
        created_at=data.get("created_at", ""),
        n_events=data.get("n_events", 0),
        n_systems=data.get("n_systems", 0),
        systems=systems,
        overall_mean_repair=data.get("overall_mean_repair", 0.0),
        overall_median_repair=data.get("overall_median_repair", 0.0),
        overall_p95_repair=data.get("overall_p95_repair", 0.0),
        overall_mean_downtime=data.get("overall_mean_downtime", 0.0),
        overall_major_share=data.get("overall_major_share", 0.0),
        overall_std_repair=data.get("overall_std_repair", 0.0),
        hourly_downtime_cost=data.get(
            "hourly_downtime_cost", DEFAULT_HOURLY_DOWNTIME_COST
        ),
        fallback_used=data.get("fallback_used", False),
        notes=data.get("notes", []),
        _micro_losses=micro_losses,
        micro_losses_file=str(micro_name) if micro_name else None,
        brands=brands,
    )


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------
def build_report(
    model: SeverityModel,
    events: pd.DataFrame,
    exact_covered_losses: Dict[str, float],
) -> Dict[str, Any]:
    """Построить JSON-отчёт по severity."""
    return {
        "model_version": model.version,
        "source": model.source,
        "created_at": model.created_at,
        "n_events": model.n_events,
        "n_systems": model.n_systems,
        "fallback_used": model.fallback_used,
        "notes": model.notes,
        "overall": {
            "mean_repair_cost": model.overall_mean_repair,
            "median_repair_cost": model.overall_median_repair,
            "p95_repair_cost": model.overall_p95_repair,
            "std_repair_cost": model.overall_std_repair,
            "mean_downtime_hours": model.overall_mean_downtime,
            "major_share": model.overall_major_share,
            "hourly_downtime_cost": model.hourly_downtime_cost,
            "expected_loss_per_failure": model.expected_loss_per_failure(),
            "expected_downtime_cost": model.expected_downtime_cost(),
            "expected_covered_loss_no_ded": model.expected_covered_loss(0.0, None),
        },
        "exact_covered_losses": exact_covered_losses,
        "systems": {
            name: {
                "n_events": s.n_events,
                "freq_share": s.freq_share,
                "mean_repair_cost": s.mean_repair_cost,
                "median_repair_cost": s.median_repair_cost,
                "p95_repair_cost": s.p95_repair_cost,
                "std_repair_cost": s.std_repair_cost,
                "mean_downtime_hours": s.mean_downtime_hours,
                "major_share": s.major_share,
                "expected_loss": s.expected_loss(model.hourly_downtime_cost),
                "reliable": s.n_events >= MIN_EVENTS_PER_SYSTEM,
            }
            for name, s in model.systems.items()
        },
        "brands": {
            name: {
                "n_events": b.n_events,
                "mean_repair_cost": b.mean_repair_cost,
                "median_repair_cost": b.median_repair_cost,
                "mean_downtime_hours": b.mean_downtime_hours,
                "major_share": b.major_share,
                "expected_loss": (
                    b.mean_repair_cost
                    + b.mean_downtime_hours * model.hourly_downtime_cost
                ),
            }
            for name, b in model.brands.items()
        },
        "comparison_with_expert": {
            "expert_freq_shares": FALLBACK_FREQ_SHARES,
            "expert_severity_weights": FALLBACK_SEVERITY_WEIGHTS,
            "empirical_freq_shares": {
                name: s.freq_share for name, s in model.systems.items()
            },
        },
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("=" * 70)
    print("Фаза 7, Итерация 2: Оценка severity-модели из claims")
    print("=" * 70)

    # Загрузка events
    try:
        events = load_claims_events(CLAIMS_PATH)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Ошибка загрузки claims: %s", exc)
        return 1

    if len(events) == 0:
        logger.warning(
            "Нет событий. Будет использована fallback-модель "
            "из экспертных констант."
        )

    # Построение модели
    model = build_severity_model(events, DEFAULT_HOURLY_DOWNTIME_COST)
    logger.info(
        "Severity-модель построена: %d событий, %d систем, %d брендов",
        model.n_events,
        model.n_systems,
        len(model.brands),
    )

    if model.notes:
        for note in model.notes:
            logger.info("  → %s", note)

    # Точные расчёты covered loss для разных сценариев
    scenarios = {
        "no_deductible_no_limit": (0.0, None),
        "deductible_10k_no_limit": (10_000.0, None),
        "deductible_50k_no_limit": (50_000.0, None),
        "deductible_10k_limit_500k": (10_000.0, 500_000.0),
        "deductible_10k_limit_1M": (10_000.0, 1_000_000.0),
    }

    exact_covered_losses: Dict[str, float] = {}
    for scenario_name, (ded, lim) in scenarios.items():
        exact = compute_exact_covered_loss(
            events, ded, lim, DEFAULT_HOURLY_DOWNTIME_COST
        )
        exact_covered_losses[scenario_name] = exact

    # Вывод результатов
    print()
    print("-" * 70)
    print("ОБЩАЯ СТАТИСТИКА")
    print("-" * 70)
    print(f"  Событий:                {model.n_events}")
    print(f"  Систем отказов:         {model.n_systems}")
    print(f"  Брендов с данными:      {len(model.brands)}")
    print(f"  Средняя стоимость:      {model.overall_mean_repair:,.0f} руб.")
    print(f"  Медиана стоимости:      {model.overall_median_repair:,.0f} руб.")
    print(f"  P95 стоимости:          {model.overall_p95_repair:,.0f} руб.")
    print(f"  Стд. откл. стоимости:   {model.overall_std_repair:,.0f} руб.")
    print(f"  Средний простой:        {model.overall_mean_downtime:.1f} ч")
    print(f"  Доля major-отказов:     {model.overall_major_share:.4f}")
    print(f"  E[loss per failure]:    {model.expected_loss_per_failure():,.0f} руб.")
    print(f"  E[covered, ded=0]:      {model.expected_covered_loss(0.0, None):,.0f} руб.")
    print(f"  Fallback использован:   {'да' if model.fallback_used else 'нет'}")

    print()
    print("-" * 70)
    print("ПО СИСТЕМАМ ОТКАЗОВ")
    print("-" * 70)
    for name, s in sorted(
        model.systems.items(), key=lambda x: x[1].freq_share, reverse=True
    ):
        reliable = "✅" if s.n_events >= MIN_EVENTS_PER_SYSTEM else "⚠️"
        exp_loss = s.expected_loss(model.hourly_downtime_cost)
        print(
            f"  {reliable} {name:15s}: "
            f"n={s.n_events:4d}, "
            f"freq={s.freq_share:.3f}, "
            f"E[repair]={s.mean_repair_cost:>12,.0f}, "
            f"E[loss]={exp_loss:>12,.0f}, "
            f"major={s.major_share:.2f}"
        )

    if model.brands:
        print()
        print("-" * 70)
        print("ПО БРЕНДАМ")
        print("-" * 70)
        for name, b in sorted(
            model.brands.items(), key=lambda x: x[1].n_events, reverse=True
        ):
            exp_loss = (
                b.mean_repair_cost
                + b.mean_downtime_hours * model.hourly_downtime_cost
            )
            print(
                f"  {name:20s}: "
                f"n={b.n_events:4d}, "
                f"E[repair]={b.mean_repair_cost:>12,.0f}, "
                f"E[loss]={exp_loss:>12,.0f}, "
                f"major={b.major_share:.2f}"
            )

    print()
    print("-" * 70)
    print("E[COVERED LOSS] ПО СЦЕНАРИЯМ (точный расчёт)")
    print("-" * 70)
    for scenario_name, value in exact_covered_losses.items():
        print(f"  {scenario_name:35s}: {value:>12,.0f} руб.")

    # Проверка согласованности: точный vs аппроксимация
    approx = model.expected_covered_loss(0.0, None)
    exact_no_ded = exact_covered_losses.get("no_deductible_no_limit", 0.0)
    if exact_no_ded > 0:
        approx_error = abs(approx - exact_no_ded) / exact_no_ded
        print()
        print(f"  Аппроксимация vs точный: ошибка {approx_error * 100:.1f}%")
        if approx_error > 0.10:
            print(
                "  ⚠️  Аппроксимация неточна. "
                "Используйте expected_covered_loss() с микро-данными."
            )

    # Сохранение
    save_severity_model(OUTPUT_MODEL_PATH, model)
    print()
    print(f"Модель сохранена: {OUTPUT_MODEL_PATH}")

    report = build_report(model, events, exact_covered_losses)
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Отчёт сохранён:   {OUTPUT_REPORT_PATH}")

    print()
    print("=" * 70)
    print("Итерация 2 Фазы 7 завершена")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())