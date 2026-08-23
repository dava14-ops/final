#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_retrain.py
===============
Фаза 9.2: автоматическое переобучение модели при срабатывании триггеров.

Цикл работы:
  1. Чтение триггеров перекалибровки:
       - из instrument_monitor.py (exit code 1 + JSON-отчёт)
       - из production_monitor.py (loss ratio drift)
       - из календаря (раз в N дней, настраивается)
  2. Загрузка данных:
       - claims_clean.csv (Фаза 6) — если есть
       - синтетика (fallback) — если claims нет
  3. Программный вызов train_model.run_training(config)
  4. Пост-валидация (K-M, F-statistic, post-fit prob)
  5. Ротация артефактов: model_params.json → archive/v{N}.json
  6. Генерация retrain_report.json

Не содержит интерактивности. Все параметры — через CLI или config.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("auto_retrain")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "model_params.json"
DEFAULT_ARCHIVE_DIR = "archive"
DEFAULT_REPORT_PATH = "retrain_report.json"
DEFAULT_CLAIMS_PATH = "data/processed/claims/claims_clean.csv"
MIN_CLAIMS_FOR_REAL_TRAIN = 100   # минимум машин для реального обучения
MIN_EVENTS_FOR_REAL_TRAIN = 200   # минимум событий

# Триггеры (согласованы с instrument_monitor.py Фаза 8.5)
TRIGGER_INSTRUMENT_DRIFT = "instrument_drift"
TRIGGER_F_BELOW_THRESHOLD = "f_below_threshold"
TRIGGER_PARTIAL_R2_TOO_LOW = "partial_r2_too_low"
TRIGGER_LOSS_RATIO_DRIFT = "loss_ratio_drift"
TRIGGER_SCHEDULED = "scheduled_retrain"
TRIGGER_MANUAL = "manual"
VALID_TRIGGERS = frozenset({
    TRIGGER_INSTRUMENT_DRIFT,
    TRIGGER_F_BELOW_THRESHOLD,
    TRIGGER_PARTIAL_R2_TOO_LOW,
    TRIGGER_LOSS_RATIO_DRIFT,
    TRIGGER_SCHEDULED,
    TRIGGER_MANUAL,
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RetrainRequest:
    """Запрос на переобучение."""
    trigger: str
    reason: str
    severity: str = "medium"          # low / medium / high / critical
    requested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrainReport:
    """Отчёт о переобучении."""
    retrain_id: str
    request: RetrainRequest
    started_at: str
    completed_at: Optional[str] = None
    status: str = "pending"           # pending / success / failed / skipped
    old_model_version: Optional[str] = None
    new_model_version: Optional[str] = None
    data_source: str = "unknown"      # real_claims / synthetic / hybrid
    n_observations: int = 0
    n_events: int = 0
    km_abs_diff: Optional[float] = None
    f_statistic: Optional[float] = None
    post_fit_target_error: Optional[float] = None
    archive_path: Optional[str] = None
    new_model_path: Optional[str] = None
    error: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrain_id": self.retrain_id,
            "request": self.request.to_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "old_model_version": self.old_model_version,
            "new_model_version": self.new_model_version,
            "data_source": self.data_source,
            "n_observations": self.n_observations,
            "n_events": self.n_events,
            "km_abs_diff": self.km_abs_diff,
            "f_statistic": self.f_statistic,
            "post_fit_target_error": self.post_fit_target_error,
            "archive_path": self.archive_path,
            "new_model_path": self.new_model_path,
            "error": self.error,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------
def read_instrument_monitor_report(
    path: Path,
) -> Optional[RetrainRequest]:
    """
    Прочитать отчёт instrument_monitor.py (Фаза 8.5).
    Если instrument_adequate=False или есть триггеры → RetrainRequest.
    """
    if not path.exists():
        logger.debug("instrument_monitor report not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None

    instrument_adequate = bool(data.get("instrument_adequate", True))
    triggers = list(data.get("triggers", []) or [])
    recommended_iv_mode = str(data.get("recommended_iv_mode", "causal"))

    if not triggers and instrument_adequate:
        return None

    # Выбираем главный триггер по приоритету
    priority = [
        TRIGGER_F_BELOW_THRESHOLD,
        TRIGGER_PARTIAL_R2_TOO_LOW,
        TRIGGER_INSTRUMENT_DRIFT,
    ]
    main_trigger = TRIGGER_INSTRUMENT_DRIFT
    for p in priority:
        if p in triggers:
            main_trigger = p
            break

    reason = (
        f"Instrument monitor: triggers={triggers}, "
        f"recommended_iv_mode={recommended_iv_mode}"
    )
    severity = "high" if main_trigger == TRIGGER_F_BELOW_THRESHOLD else "medium"
    return RetrainRequest(
        trigger=main_trigger,
        reason=reason,
        severity=severity,
        metadata={
            "instrument_report_path": str(path),
            "triggers": triggers,
            "recommended_iv_mode": recommended_iv_mode,
        },
    )


def check_scheduled_retrain(
    model_path: Path,
    interval_days: int = 90,
) -> Optional[RetrainRequest]:
    """
    Проверить, прошло ли больше interval_days с последнего обучения.
    """
    if interval_days <= 0:
        return None
    if not model_path.exists():
        return RetrainRequest(
            trigger=TRIGGER_SCHEDULED,
            reason="No model file exists, initial training required",
            severity="critical",
        )
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            model = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    created_str = (
        model.get("training_meta", {}).get("created")
        if isinstance(model.get("training_meta"), dict)
        else None
    )
    if not created_str:
        return RetrainRequest(
            trigger=TRIGGER_SCHEDULED,
            reason="Model has no 'created' timestamp; retrain recommended",
            severity="medium",
        )
    try:
        created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
    except ValueError:
        return None
    age_days = (datetime.now(timezone.utc) - created).days
    if age_days >= interval_days:
        return RetrainRequest(
            trigger=TRIGGER_SCHEDULED,
            reason=f"Model age is {age_days} days (threshold: {interval_days})",
            severity="low",
            metadata={"age_days": age_days},
        )
    return None


def detect_retrain_request(
    args: argparse.Namespace,
) -> Optional[RetrainRequest]:
    """
    Определить, нужно ли переобучение, по всем источникам.
    """
    if args.force:
        return RetrainRequest(
            trigger=TRIGGER_MANUAL,
            reason="Manual --force flag",
            severity="medium",
        )

    # Приоритет 1: instrument_monitor
    req = read_instrument_monitor_report(
        Path(args.instrument_report)
    )
    if req is not None:
        return req

    # Приоритет 2: scheduled
    req = check_scheduled_retrain(
        Path(args.model),
        interval_days=args.schedule_days,
    )
    if req is not None:
        return req

    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_real_claims(path: Path) -> Optional[pd.DataFrame]:
    """
    Загрузить реальные claims из Фаза 6.
    Возвращает None если файла нет или недостаточно данных.
    """
    if not path.exists():
        logger.info("Real claims file not found: %s", path)
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read claims: %s", exc)
        return None

    n_obs = len(df)
    n_events = int(df["event"].sum()) if "event" in df.columns else 0

    if n_obs < MIN_CLAIMS_FOR_REAL_TRAIN:
        logger.info(
            "Claims insufficient: %d < %d machines",
            n_obs, MIN_CLAIMS_FOR_REAL_TRAIN,
        )
        return None
    if n_events < MIN_EVENTS_FOR_REAL_TRAIN:
        logger.info(
            "Events insufficient: %d < %d events",
            n_events, MIN_EVENTS_FOR_REAL_TRAIN,
        )
        return None

    logger.info(
        "Real claims loaded: %d machines, %d events", n_obs, n_events,
    )
    return df


def choose_data_source(
    claims_path: Path,
    allow_synthetic: bool,
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Определить источник данных для переобучения.
    Возвращает (source, dataframe_or_none).
    """
    real = load_real_claims(claims_path)
    if real is not None:
        return "real_claims", real
    if allow_synthetic:
        logger.info("Falling back to synthetic DGP for retraining")
        return "synthetic", None
    return "insufficient", None


# ---------------------------------------------------------------------------
# Training pipeline (calls train_model programmatically)
# ---------------------------------------------------------------------------
def run_training_programmatically(
    config_overrides: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Программный вызов обучения через декомпозированный train_model.py.

    Returns (model_params, training_meta).
    """
    # Импорты здесь, чтобы не тянуть все зависимости на старте
    try:
        from train_model import (
            TrainingConfig,
            build_dgp_from_config,
            calibrate_model,
            generate_training_data,
            fit_first_stage_and_cf,
            build_model_artifact,
            validate_save_and_smoke_test,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import from train_model.py (Phase 1b decomposition "
            f"required): {exc}"
        ) from exc

    cfg = TrainingConfig()
    for key, value in config_overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    logger.info("Building DGP from config...")
    dgp = build_dgp_from_config(cfg)

    logger.info("Calibrating model...")
    baseline_h, censoring_scale, baseline_diag = calibrate_model(cfg, dgp)

    logger.info("Generating training data...")
    data, data_mod, transform_info, achieved_event_rate, peak_stats, \
        training_meta_flat = generate_training_data(
            cfg, dgp, baseline_h, censoring_scale,
        )

    logger.info("Fitting first stage + CF Cox...")
    fit = fit_first_stage_and_cf(
        cfg, dgp, data, data_mod,
        baseline_h, censoring_scale, baseline_diag,
        transform_info, achieved_event_rate, peak_stats, training_meta_flat,
    )

    # Обновить ссылки (могли измениться при автокоррекции)
    data = fit["data"]
    data_mod = fit["data_mod"]
    peak_stats = peak_stats  # unchanged

    logger.info("Building model artifact...")
    model_params = build_model_artifact(
        cfg, dgp, fit, transform_info,
        achieved_event_rate, peak_stats, training_meta_flat,
    )

    return model_params, {
        "data": data,
        "template_dict": fit["template_dict"],
        "peak_stats": peak_stats,
        "iv_diagnostics": fit["iv_diagnostics"],
        "fitted_fs": fit["fitted_fs"],
        "n_events": int(fit["data_mod"]["event"].sum()),
    }


# ---------------------------------------------------------------------------
# Post-validation
# ---------------------------------------------------------------------------
def validate_new_model(
    model_params: Any,
    data: pd.DataFrame,
    target_time: float,
) -> Dict[str, Any]:
    """
    Пост-валидация новой модели.
    Возвращает словарь метрик.
    """
    metrics: Dict[str, Any] = {}
    try:
        from prediction_engine import kaplan_meier_check
        result = kaplan_meier_check(
            params=model_params,
            times=data["time"].astype(float).to_numpy(),
            events=data["event"].astype(int).to_numpy(),
            eval_horizon=float(target_time),
        )
        metrics["km_abs_diff"] = float(result.get("abs_diff", float("nan")))
    except Exception as exc:
        logger.warning("K-M validation failed: %s", exc)

    meta = getattr(model_params, "training_meta", {}) or {}
    if isinstance(meta, dict):
        iv = meta.get("iv_diagnostics", {}) or {}
        if isinstance(iv, dict):
            metrics["f_statistic"] = iv.get("f_statistic")
        metrics["post_fit_target_error"] = meta.get("post_fit_target_error")

    return metrics


# ---------------------------------------------------------------------------
# Archive rotation
# ---------------------------------------------------------------------------
def rotate_model_file(
    model_path: Path,
    archive_dir: Path,
) -> Optional[str]:
    """
    Переместить текущую модель в archive/v{N}.json.
    Возвращает путь к архиву или None.
    """
    if not model_path.exists():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Найти следующий свободный номер версии
    existing = list(archive_dir.glob("model_v*.json"))
    versions = []
    for p in existing:
        try:
            v = int(p.stem.replace("model_v", ""))
            versions.append(v)
        except ValueError:
            pass
    next_v = max(versions, default=0) + 1

    dest = archive_dir / f"model_v{next_v}.json"
    try:
        shutil.copy2(model_path, dest)
        logger.info("Archived old model: %s", dest)
        return str(dest)
    except OSError as exc:
        logger.warning("Failed to archive old model: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main retraining flow
# ---------------------------------------------------------------------------
def perform_retrain(
    request: RetrainRequest,
    args: argparse.Namespace,
) -> RetrainReport:
    """
    Выполнить полный цикл переобучения.
    """
    report = RetrainReport(
        retrain_id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        request=request,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info("=" * 70)
    logger.info("RETRAIN STARTED")
    logger.info("  trigger : %s", request.trigger)
    logger.info("  reason  : %s", request.reason)
    logger.info("  severity: %s", request.severity)
    logger.info("=" * 70)

    model_path = Path(args.model)
    claims_path = Path(args.claims)

    # 1. Чтение текущей версии модели
    try:
        from prediction_engine import load_model_params
        if model_path.exists():
            old_params = load_model_params(model_path)
            report.old_model_version = getattr(
                old_params, "model_version", None
            )
    except Exception as exc:
        logger.warning("Could not read old model version: %s", exc)

    # 2. Определение источника данных
    source, real_df = choose_data_source(
        claims_path, allow_synthetic=not args.real_only,
    )
    report.data_source = source
    if source == "insufficient":
        report.status = "skipped"
        report.error = (
            f"Real claims not available (<{MIN_CLAIMS_FOR_REAL_TRAIN} machines) "
            f"and --real-only flag set. Retrain skipped."
        )
        report.completed_at = datetime.now(timezone.utc).isoformat()
        return report

    # 3. Config overrides
    config_overrides: Dict[str, Any] = {
        "out_path": str(model_path),
    }
    if source == "real_claims" and real_df is not None:
        report.n_observations = len(real_df)
        report.n_events = int(real_df["event"].sum())
        # TODO (Фаза 7): интеграция реальных claims в DGP.
        # Пока используем симуляцию с эмпирическими параметрами.
        report.notes.append(
            "Real claims available but DGP integration is not yet "
            "implemented (Phase 7). Using synthetic DGP with "
            "empirically informed parameters."
        )

    # 4. Обучение
    try:
        model_params, train_meta = run_training_programmatically(
            config_overrides,
        )
        report.n_observations = report.n_observations or train_meta.get(
            "data", pd.DataFrame()
        ).shape[0] or 0
        report.n_events = train_meta.get("n_events", 0)

        # 5. Пост-валидация
        target_time = float(
            getattr(model_params, "calibration_time_horizon", 1712.0)
        )
        metrics = validate_new_model(
            model_params, train_meta["data"], target_time,
        )
        report.km_abs_diff = metrics.get("km_abs_diff")
        report.f_statistic = metrics.get("f_statistic")
        report.post_fit_target_error = metrics.get("post_fit_target_error")
        report.new_model_version = getattr(
            model_params, "model_version", None
        )

        # 6. Сохранение через prediction_engine.save_model_params
        # (уже сделано в train_model.validate_save_and_smoke_test,
        #  но явно пересохраняем для гарантии атомарности)
        from prediction_engine import save_model_params
        tmp_path = model_path.with_suffix(".tmp.json")
        save_model_params(tmp_path, model_params)

        # 7. Ротация
        archive_path = rotate_model_file(model_path, Path(args.archive_dir))
        report.archive_path = archive_path

        # 8. Атомарная замена
        tmp_path.replace(model_path)
        report.new_model_path = str(model_path)
        report.status = "success"

    except Exception as exc:
        logger.exception("Retrain failed")
        report.status = "failed"
        report.error = f"{type(exc).__name__}: {exc}"

    report.completed_at = datetime.now(timezone.utc).isoformat()
    return report


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------
def save_report(report: RetrainReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Report saved: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Фаза 9.2: автоматическое переобучение модели "
            "при срабатывании триггеров."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to model_params.json",
    )
    p.add_argument(
        "--claims",
        type=str,
        default=DEFAULT_CLAIMS_PATH,
        help="Path to claims_clean.csv (Фаза 6)",
    )
    p.add_argument(
        "--instrument-report",
        type=str,
        default="instrument_health.json",
        help="Output of instrument_monitor.py",
    )
    p.add_argument(
        "--archive-dir",
        type=str,
        default=DEFAULT_ARCHIVE_DIR,
        help="Directory to archive old model versions",
    )
    p.add_argument(
        "--report",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help="Where to save retrain report",
    )
    p.add_argument(
        "--schedule-days",
        type=int,
        default=90,
        help="Scheduled retrain interval (days). 0 disables.",
    )
    p.add_argument(
        "--real-only",
        action="store_true",
        help="Refuse to train on synthetic data; skip if claims missing",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force retrain regardless of triggers",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only detect triggers, don't actually retrain",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("AUTO-RETRAIN v1.0 (Фаза 9.2)")
    print("=" * 70)
    print()

    # 1. Детекция триггеров
    request = detect_retrain_request(args)
    if request is None:
        print("[OK] No retrain triggers detected. Model is healthy.")
        return 0

    print(f"[WARN] Retrain trigger detected:")
    print(f"    trigger : {request.trigger}")
    print(f"    reason  : {request.reason}")
    print(f"    severity: {request.severity}")
    print()

    if args.dry_run:
        print("[dry-run] Would retrain. Exiting.")
        return 0

    # 2. Выполнение переобучения
    report = perform_retrain(request, args)

    # 3. Сохранение отчёта
    save_report(report, Path(args.report))

    # 4. Финальный вывод
    print()
    print("=" * 70)
    if report.status == "success":
        print(f"[OK] RETRAIN SUCCESSFUL")
        print(f"    Old version: {report.old_model_version}")
        print(f"    New version: {report.new_model_version}")
        print(f"    Data source: {report.data_source}")
        print(f"    N obs:       {report.n_observations}")
        print(f"    N events:    {report.n_events}")
        if report.km_abs_diff is not None:
            mark = "[OK]" if report.km_abs_diff < 0.15 else "[WARN]"
            print(f"    {mark} K-M |diff|:  {report.km_abs_diff:.4f}")
        if report.f_statistic is not None:
            mark = "[OK]" if report.f_statistic > 14.18 else "[WARN]"
            print(f"    {mark} F-stat:    {report.f_statistic:.2f}")
        if report.archive_path:
            print(f"    Archived:    {report.archive_path}")
        exit_code = 0
    elif report.status == "skipped":
        print(f"[SKIP] RETRAIN SKIPPED")
        print(f"    Reason: {report.error}")
        exit_code = 0
    else:
        print(f"[FAIL] RETRAIN FAILED")
        print(f"    Error: {report.error}")
        exit_code = 1
    print("=" * 70)
    print(f"Report: {args.report}")
    print()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())