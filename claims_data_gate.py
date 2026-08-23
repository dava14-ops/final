#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
claims_data_gate.py

Phase 7.0 — strict Real Claims data gate for production retraining.

Principle:
    Real-claims retraining MUST fail closed.
    Missing real covariates are never replaced with synthetic values.

The gate validates the Claims Dataset v1.0 contract and the Phase 7
minimum coverage requirements before Cox/CF-Cox training is allowed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

try:
    from constants import BRAND_MAP, VALID_EVENT_DEFINITIONS
except Exception:  # pragma: no cover - standalone diagnostic fallback
    BRAND_MAP = {0: "MTZ82", 1: "Versatile280", 2: "NewHollandT9", 3: "DT75", 4: "Other"}
    VALID_EVENT_DEFINITIONS = {"total_loss", "major_claim", "any_failure"}


TARGET_EVENT_DEFINITION = "major_claim"
MIN_MACHINES = 100
MIN_EVENTS = 200
MIN_HORIZON_HOURS = 1712.0
MIN_EVENTS_PER_BRAND = 30
MIN_BRANDS_WITH_EVENTS = 3

REQUIRED_COLUMNS = {
    "machine_id",
    "brand",
    "power_hp",
    "production_year",
    "age_at_event",
    "hours_at_event",
    "failure_time",
    "event_flag",
    "failure_system",
    "major_failure_flag",
}

# These are optional in the published Data Contract, but are REQUIRED for
# the actual production CF-Cox retraining because they are model regressors.
PRODUCTION_MODEL_COLUMNS = {
    "peak_load_proxy",
    "climate_index",
    "soil_index",
}


@dataclass
class GateResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClaimsDataGateError(ValueError):
    """Raised when real claims data fail the Phase 7 gate."""


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _check_required_columns(df: pd.DataFrame, result: GateResult) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    result.checks["required_columns"] = not missing
    if missing:
        result.errors.append(
            "Missing required Claims Dataset v1.0 columns: " + ", ".join(missing)
        )

    missing_model = sorted(PRODUCTION_MODEL_COLUMNS - set(df.columns))
    result.checks["production_model_columns"] = not missing_model
    if missing_model:
        result.errors.append(
            "Missing real production model columns; synthetic fallback is forbidden: "
            + ", ".join(missing_model)
        )


def _check_types_and_ranges(df: pd.DataFrame, result: GateResult) -> None:
    if not REQUIRED_COLUMNS.issubset(df.columns):
        return

    # IDs
    machine = df["machine_id"].astype("string")
    empty_machine = machine.isna() | machine.str.strip().eq("")
    result.checks["machine_id_nonempty"] = not bool(empty_machine.any())
    if empty_machine.any():
        result.errors.append(f"machine_id empty: {int(empty_machine.sum())} rows")

    # Numeric fields
    numeric_columns = [
        "power_hp",
        "production_year",
        "age_at_event",
        "hours_at_event",
        "failure_time",
        "event_flag",
        "major_failure_flag",
    ]
    numeric_ok = True
    for col in numeric_columns:
        values = _numeric(df, col)
        bad = values.isna()
        if bad.any():
            numeric_ok = False
            result.errors.append(f"{col}: {int(bad.sum())} non-numeric/NaN values")
    result.checks["numeric_fields"] = numeric_ok

    if not numeric_ok:
        return

    power = _numeric(df, "power_hp")
    year = _numeric(df, "production_year")
    age = _numeric(df, "age_at_event")
    hours = _numeric(df, "hours_at_event")
    time = _numeric(df, "failure_time")
    event = _numeric(df, "event_flag")
    major = _numeric(df, "major_failure_flag")

    range_checks = {
        "power_hp_range": power.between(50, 500, inclusive="both").all(),
        "production_year_range": year.between(1990, 2025, inclusive="both").all(),
        "age_range": age.between(0, 30, inclusive="both").all(),
        "hours_range": hours.between(0, 50000, inclusive="both").all(),
        "failure_time_positive": (time > 0).all(),
        "event_binary": event.isin([0, 1]).all(),
        "major_binary": major.isin([0, 1]).all(),
    }
    result.checks.update({k: bool(v) for k, v in range_checks.items()})

    if not range_checks["power_hp_range"]:
        result.errors.append("power_hp outside [50, 500] hp")
    if not range_checks["production_year_range"]:
        result.errors.append("production_year outside [1990, 2025]")
    if not range_checks["age_range"]:
        result.errors.append("age_at_event outside [0, 30] years")
    if not range_checks["hours_range"]:
        result.errors.append("hours_at_event outside [0, 50000] mч")
    if not range_checks["failure_time_positive"]:
        result.errors.append("failure_time <= 0 detected")
    if not range_checks["event_binary"]:
        result.errors.append("event_flag must be binary {0,1}")
    if not range_checks["major_binary"]:
        result.errors.append("major_failure_flag must be binary {0,1}")

    bad_event_time = (event == 1) & (time <= 0)
    result.checks["event_time_consistency"] = not bool(bad_event_time.any())
    if bad_event_time.any():
        result.errors.append(
            f"event_flag=1 with non-positive failure_time: {int(bad_event_time.sum())} rows"
        )


def _check_model_covariates(df: pd.DataFrame, result: GateResult) -> None:
    if not PRODUCTION_MODEL_COLUMNS.issubset(df.columns):
        return

    finite_ok = True
    ranges_ok = True
    for col in sorted(PRODUCTION_MODEL_COLUMNS):
        values = _numeric(df, col)
        bad = ~np.isfinite(values.to_numpy(dtype=float))
        if bad.any():
            finite_ok = False
            result.errors.append(f"{col}: {int(bad.sum())} non-finite values")

        if col == "peak_load_proxy":
            invalid = values.notna() & ~values.between(0, 1, inclusive="both")
            if invalid.any():
                ranges_ok = False
                result.errors.append(
                    f"peak_load_proxy outside [0,1]: {int(invalid.sum())} rows"
                )

        if col in {"climate_index", "soil_index"}:
            invalid = values.notna() & ~values.between(0, 1, inclusive="both")
            if invalid.any():
                ranges_ok = False
                result.errors.append(
                    f"{col} outside [0,1]: {int(invalid.sum())} rows"
                )

        if values.notna().any() and float(values.dropna().std()) < 1e-12:
            result.errors.append(f"{col} has zero/negligible variance")
            ranges_ok = False

    result.checks["model_covariates_finite"] = finite_ok
    result.checks["model_covariate_ranges"] = ranges_ok

    for col in sorted(PRODUCTION_MODEL_COLUMNS):
        missing_count = int(_numeric(df, col).isna().sum())
        if missing_count:
            result.errors.append(
                f"{col}: {missing_count} missing values; median imputation is not allowed in Phase 7.0"
            )


def _check_event_definition(df: pd.DataFrame, result: GateResult) -> None:
    if "event_definition" not in df.columns:
        result.warnings.append(
            "event_definition column is absent. Gate assumes event_flag is already the production target "
            f"({TARGET_EVENT_DEFINITION}), but this provenance must be documented before final model release."
        )
        result.checks["event_definition"] = True
        return

    definitions = (
        df["event_definition"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    invalid = ~definitions.isin(set(VALID_EVENT_DEFINITIONS))
    if invalid.any():
        result.errors.append(
            f"event_definition has {int(invalid.sum())} unknown values"
        )

    non_target = definitions.notna() & definitions.ne(TARGET_EVENT_DEFINITION)
    if non_target.any():
        counts = definitions[non_target].value_counts().to_dict()
        result.errors.append(
            f"event_definition is not consistently '{TARGET_EVENT_DEFINITION}': {counts}"
        )

    result.checks["event_definition"] = not invalid.any() and not non_target.any()


def _check_duplicates(df: pd.DataFrame, result: GateResult) -> None:
    key = ["machine_id", "failure_time", "event_flag"]
    if not set(key).issubset(df.columns):
        result.checks["duplicate_rows"] = False
        return

    duplicated = df.duplicated(subset=key, keep=False)
    result.checks["duplicate_rows"] = not bool(duplicated.any())
    if duplicated.any():
        result.errors.append(
            f"Duplicate (machine_id, failure_time, event_flag) rows: {int(duplicated.sum())}"
        )


def _check_machine_structure(df: pd.DataFrame, result: GateResult) -> None:
    if "machine_id" not in df.columns:
        return

    per_machine = df.groupby("machine_id", dropna=False).size()
    multirow = int((per_machine > 1).sum())
    result.metrics["machines_with_multiple_rows"] = multirow

    if multirow:
        result.warnings.append(
            f"{multirow} machines have multiple rows. This gate does not silently collapse them; "
            "survival-risk-set construction must explicitly resolve repeated observations."
        )


def _check_coverage(df: pd.DataFrame, result: GateResult) -> None:
    if "machine_id" not in df.columns or "event_flag" not in df.columns:
        return

    event_mask = _numeric(df, "event_flag").eq(1)
    n_machines = int(df["machine_id"].nunique())
    n_events = int(event_mask.sum())
    max_time = float(_numeric(df, "failure_time").max())

    result.metrics.update(
        {
            "n_rows": int(len(df)),
            "n_machines": n_machines,
            "n_events": n_events,
            "n_censored": int((~event_mask).sum()),
            "event_rate": float(event_mask.mean()) if len(df) else 0.0,
            "max_observation_time_hours": max_time,
        }
    )

    result.checks["minimum_machines"] = n_machines >= MIN_MACHINES
    result.checks["minimum_events"] = n_events >= MIN_EVENTS
    result.checks["minimum_horizon"] = max_time >= MIN_HORIZON_HOURS

    if n_machines < MIN_MACHINES:
        result.errors.append(f"Only {n_machines} unique machines; minimum is {MIN_MACHINES}")
    if n_events < MIN_EVENTS:
        result.errors.append(f"Only {n_events} events; minimum is {MIN_EVENTS}")
    if max_time < MIN_HORIZON_HOURS:
        result.errors.append(
            f"Maximum observation time {max_time:.1f} mч < required {MIN_HORIZON_HOURS:.1f} mч"
        )

    if "brand" in df.columns:
        event_brands = df.loc[event_mask, "brand"].astype("string").value_counts()
        brands_ge_min = int((event_brands >= MIN_EVENTS_PER_BRAND).sum())
        result.metrics["event_count_by_brand"] = {
            str(k): int(v) for k, v in event_brands.to_dict().items()
        }
        result.metrics["brands_with_min_events"] = brands_ge_min
        result.checks["brand_coverage"] = brands_ge_min >= MIN_BRANDS_WITH_EVENTS
        if brands_ge_min < MIN_BRANDS_WITH_EVENTS:
            result.errors.append(
                f"Only {brands_ge_min} brands have >= {MIN_EVENTS_PER_BRAND} events; "
                f"minimum is {MIN_BRANDS_WITH_EVENTS}"
            )


def _check_brand_values(df: pd.DataFrame, result: GateResult) -> None:
    if "brand" not in df.columns:
        return
    valid_names = {str(v) for v in BRAND_MAP.values()}
    values = df["brand"].astype("string").str.strip()
    invalid = ~values.isin(valid_names)
    result.checks["brand_catalog"] = not bool(invalid.any())
    if invalid.any():
        examples = sorted(values[invalid].dropna().unique().tolist())[:10]
        result.errors.append(
            f"Unknown brand values ({int(invalid.sum())} rows), examples={examples}"
        )


def validate_claims_dataframe(df: pd.DataFrame) -> GateResult:
    result = GateResult(passed=False)

    _check_required_columns(df, result)
    if result.errors and not REQUIRED_COLUMNS.issubset(df.columns):
        result.metrics["columns"] = sorted(map(str, df.columns))
        return result

    _check_types_and_ranges(df, result)
    _check_model_covariates(df, result)
    _check_event_definition(df, result)
    _check_duplicates(df, result)
    _check_machine_structure(df, result)
    _check_brand_values(df, result)
    _check_coverage(df, result)

    # Hard production rule: no missing/invalid target/model inputs.
    result.passed = len(result.errors) == 0
    return result


def load_and_validate_claims(path: Path) -> tuple[pd.DataFrame, GateResult]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Claims file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    result = validate_claims_dataframe(df)
    return df, result


def write_gate_report(result: GateResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def assert_gate(result: GateResult) -> None:
    if result.passed:
        return
    lines = ["Real Claims DATA GATE: FAILED"]
    lines.extend(f"  - {msg}" for msg in result.errors)
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {msg}" for msg in result.warnings)
    raise ClaimsDataGateError("\n".join(lines))


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Phase 7.0 Real Claims data gate")
    parser.add_argument("claims", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/claims/retrain_data_audit.json"),
    )
    args = parser.parse_args()

    frame, gate = load_and_validate_claims(args.claims)
    write_gate_report(gate, args.report)

    print("=" * 70)
    print("PHASE 7.0 — REAL CLAIMS DATA GATE")
    print("=" * 70)
    print(f"File:        {args.claims}")
    print(f"Status:      {'PASS' if gate.passed else 'FAIL'}")
    print(f"Rows:        {len(frame):,}")
    for key, value in gate.metrics.items():
        print(f"{key}: {value}")

    if gate.errors:
        print("\nERRORS:")
        for error in gate.errors:
            print(f"  - {error}")
    if gate.warnings:
        print("\nWARNINGS:")
        for warning in gate.warnings:
            print(f"  - {warning}")

    print(f"\nAudit report: {args.report}")
    raise SystemExit(0 if gate.passed else 1)