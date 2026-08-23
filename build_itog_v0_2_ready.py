#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
from pathlib import Path

SRC = Path("Итог.py")
DST = Path("Итог_v0.2_ready.py")

s = SRC.read_text(encoding="utf-8").replace("\r\n", "\n")


def rep(old: str, new: str, msg: str) -> None:
    global s
    if old not in s:
        raise SystemExit(f"Marker not found: {msg}")
    s = s.replace(old, new, 1)


def repsub(pattern: str, new: str, msg: str, flags: int = 0) -> None:
    global s
    t = re.sub(pattern, lambda m: new, s, count=1, flags=flags)
    if t == s:
        raise SystemExit(f"Regex not applied: {msg}")
    s = t


# ---------------------------------------------------------------------------
# 1. Удаляем дублирующий нижний блок P-05 / P-07
# ---------------------------------------------------------------------------
s = re.sub(
    r"# -{10,}\n# P-05: FREQ_SHARES / SEVERITY_WEIGHTS\n# -{10,}\n.*?"
    r"(?=# -{10,}\n# Censoring calibration\n# -{10,})",
    "",
    s,
    count=1,
    flags=re.S,
)

# ---------------------------------------------------------------------------
# 2. Запрещаем использовать новые outcome-колонки как X
# ---------------------------------------------------------------------------
rep(
    "_ALL_PENALIZERS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]",
    'FORBIDDEN_EXTRA_X_COLS.update({"time_minor", "event_minor", "failure_type", '
    '"event_definition", "T_minor", "downtime_hours"})\n\n'
    "_ALL_PENALIZERS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]",
    "FORBIDDEN_EXTRA_X_COLS update",
)

# ---------------------------------------------------------------------------
# 3. Добавляем P-08, P-09, downtime helper
# ---------------------------------------------------------------------------
rep(
    'EVENT_DEFINITIONS = ("total_loss", "major_claim", "any_failure")',
    '''EVENT_DEFINITIONS = ("total_loss", "major_claim", "any_failure")

# P-08: MTBF baseline
MTBF_BASELINE_HOURS = 1500.0
DEFAULT_BASELINE_HAZARD = 1.0 / MTBF_BASELINE_HOURS

# P-09: DOWNTIME по MTTR
MTTR_HOURS = {"minor": 8.0, "major": 48.0}

def downtime_hours_from_failure_type(failure_type):
    ft = np.asarray(failure_type, dtype=str)
    out = np.zeros(ft.shape, dtype=float)
    out[ft == "minor"] = MTTR_HOURS["minor"]
    out[ft == "major"] = MTTR_HOURS["major"]
    return out
''',
    "P-08/P-09 constants",
)

# ---------------------------------------------------------------------------
# 4. Добавляем поля DGP
# ---------------------------------------------------------------------------
repsub(
    r'(    baseline_family: str = "weibull"\n)',
    '''    baseline_family: str = "weibull"
    competing_risks: bool = False
    minor_failure_rate: float = 0.002
    event_definition: str = "major_claim"
    segment: str = "light"
''',
    "DGP fields",
)

# ---------------------------------------------------------------------------
# 5. Добавляем валидацию новых DGP-полей
# ---------------------------------------------------------------------------
rep(
    "    if dgp.structural_intercept is not None:",
    '''    if not isinstance(getattr(dgp, "competing_risks", False), (bool, np.bool_)):
        raise ValueError("dgp.competing_risks must be boolean")
    dgp.competing_risks = bool(dgp.competing_risks)

    dgp.minor_failure_rate = _as_finite_float(
        getattr(dgp, "minor_failure_rate", 0.002),
        "dgp.minor_failure_rate",
    )
    if dgp.minor_failure_rate <= 0.0:
        raise ValueError("dgp.minor_failure_rate must be > 0")

    event_definition = str(getattr(dgp, "event_definition", "major_claim")).lower()
    if event_definition not in EVENT_DEFINITIONS:
        raise ValueError(f"Unknown event_definition: '{event_definition}'")
    dgp.event_definition = event_definition

    segment = str(getattr(dgp, "segment", "light")).lower()
    if segment not in SEGMENTS:
        raise ValueError(f"Unknown segment: '{segment}'")
    dgp.segment = segment

    if dgp.structural_intercept is not None:''',
    "DGP validation",
)

# ---------------------------------------------------------------------------
# 6. P-10: Power по сегментам
# ---------------------------------------------------------------------------
rep(
    "def generate_data(",
    '''def _generate_power_by_segment(segment: str, n: int, rng: np.random.Generator) -> np.ndarray:
    segment = str(segment).lower()
    if segment == "heavy" and RF_HEAVY_BRAND_CATALOG:
        names = list(RF_HEAVY_BRAND_CATALOG.keys())
        shares = np.array([RF_HEAVY_BRAND_CATALOG[x]["share"] for x in names], dtype=float)
        shares = np.clip(shares, 0.0, None)
        total = float(shares.sum())
        if total <= 0.0:
            power = rng.normal(300.0, 60.0, size=n)
        else:
            probs = shares / total
            chosen = rng.choice(names, size=n, p=probs)
            power = np.array([RF_HEAVY_BRAND_CATALOG[x]["power_hp"] for x in chosen], dtype=float)
            power = power + rng.normal(0.0, 15.0, size=n)
        return np.clip(power, 200.0, 500.0)

    power = rng.normal(140.0, 50.0, size=n)
    return np.clip(power, 50.0, 350.0)


def generate_data(''',
    "power segment helper",
)

# ---------------------------------------------------------------------------
# 7. P-02: Hours -> LogNormal
# ---------------------------------------------------------------------------
rep(
    "    hours = rng.exponential(1350.0, size=n)",
    '''    _hp = HOURS_PRIOR.get(getattr(dgp, "segment", "light"), HOURS_PRIOR["light"])
    hours = rng.lognormal(
        mean=float(np.log(_hp["median"])),
        sigma=float(_hp["sigma"]),
        size=n,
    )
    hours = np.clip(hours, _hp["clip_min"], _hp["clip_max"])''',
    "Hours LogNormal",
)

# ---------------------------------------------------------------------------
# 8. Заменяем генерацию Power
# ---------------------------------------------------------------------------
rep(
    '''    power = rng.normal(180.0, 80.0, size=n)
    power = np.clip(power, 50.0, 350.0)''',
    '''    power = _generate_power_by_segment(getattr(dgp, "segment", "light"), n, rng)''',
    "Power generation",
)

# ---------------------------------------------------------------------------
# 9. P-01/P-03: competing risks + event_definition реально меняет event/time
# ---------------------------------------------------------------------------
new_survival = '''    # Simulate times
    u_event = rng.uniform(low=np.nextafter(0.0, 1.0), high=1.0, size=n)
    u_censor = rng.uniform(low=np.nextafter(0.0, 1.0), high=1.0, size=n)
    true_time, individual_hazard = _simulate_event_times(
        safe_lp=safe_lp,
        baseline_hazard=baseline_hazard,
        baseline_family=dgp.baseline_family,
        baseline_shape=dgp.baseline_shape,
        u_event=u_event,
    )

    censoring_time = -np.log(u_censor) * censoring_scale
    censoring_time = np.nan_to_num(censoring_time, nan=1e12, posinf=1e12, neginf=1e-12)
    censoring_time = np.maximum(censoring_time, 1e-12)

    if getattr(dgp, "competing_risks", False):
        u_minor = rng.uniform(low=np.nextafter(0.0, 1.0), high=1.0, size=n)
        minor_hazard = dgp.minor_failure_rate * np.exp(safe_lp)
        minor_hazard = np.clip(minor_hazard, 1e-300, 1e300)
        true_minor_time = -np.log(u_minor) / minor_hazard
        true_minor_time = np.nan_to_num(true_minor_time, nan=1e12, posinf=1e12, neginf=1e-12)
        true_minor_time = np.maximum(true_minor_time, 1e-12)
        event_minor = true_minor_time <= censoring_time
    else:
        true_minor_time = np.full(n, 1e12, dtype=float)
        event_minor = np.zeros(n, dtype=bool)

    event_def = getattr(dgp, "event_definition", "major_claim")

    if getattr(dgp, "competing_risks", False) and event_def == "any_failure":
        observed_time = np.minimum(np.minimum(true_time, true_minor_time), censoring_time)

        major_first = true_time <= np.minimum(true_minor_time, censoring_time)
        minor_first = (true_minor_time < true_time) & (true_minor_time <= censoring_time)

        event = major_first | minor_first
        failure_type = np.where(
            major_first,
            "major",
            np.where(minor_first, "minor", "censored"),
        )
    else:
        observed_time = np.minimum(true_time, censoring_time)
        event = true_time <= censoring_time

        if getattr(dgp, "competing_risks", False):
            minor_preempts = event_minor & (true_minor_time < observed_time)
            observed_time = np.where(minor_preempts, true_minor_time, observed_time)
            event = event & ~minor_preempts

            failure_type = np.where(
                event,
                "major",
                np.where(event_minor & (true_minor_time <= observed_time), "minor", "censored"),
            )
        else:
            failure_type = np.where(event, "major", "censored")

    observed_time = np.nan_to_num(observed_time, nan=1e-12, posinf=1e12, neginf=1e-12)
    observed_time = np.maximum(observed_time, 1e-12)

    true_minor_time = np.nan_to_num(true_minor_time, nan=1e12, posinf=1e12, neginf=1e12)
    time_minor = np.where(event_minor, true_minor_time, observed_time)
    downtime_hours = downtime_hours_from_failure_type(failure_type)
'''

start = s.find("    # Simulate times\n")
if start == -1:
    start = s.find(
        "    u_event = rng.uniform(low=np.nextafter(0.0, 1.0), high=1.0, size=n)\n"
    )

if start == -1:
    raise SystemExit("Marker not found: beginning of survival block")

end = "    event = true_time <= censoring_time\n"
stop = s.find(end, start)

if stop == -1:
    raise SystemExit("Marker not found: end of survival block")

stop += len(end)
s = s[:start] + new_survival + s[stop:]
# ---------------------------------------------------------------------------
# 10. Добавляем новые колонки в DataFrame
# ---------------------------------------------------------------------------
rep(
    '        "x_brand": x_brand_legacy,\n',
    '''        "x_brand": x_brand_legacy,
        "T_minor": true_minor_time,
        "time_minor": time_minor,
        "event_minor": event_minor,
        "failure_type": failure_type,
        "event_definition": getattr(dgp, "event_definition", "major_claim"),
        "downtime_hours": downtime_hours,
''',
    "DataFrame columns",
)

# ---------------------------------------------------------------------------
# 11. P-12: Kaplan-Meier validator
# ---------------------------------------------------------------------------
rep(
    "# ---------------------------------------------------------------------------\n# Censoring calibration",
    '''def kaplan_meier_validator(
    data: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
) -> Dict[str, Any]:
    """P-12: Kaplan-Meier validator."""
    from lifelines import KaplanMeierFitter

    if time_col not in data.columns or event_col not in data.columns:
        raise KeyError("KM validator missing required columns")

    df = data.dropna(subset=[time_col, event_col])
    if len(df) == 0:
        raise ValueError("KM validator: empty data")

    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])

    return {
        "n": int(len(df)),
        "events": int(df[event_col].astype(int).sum()),
        "event_rate": float(df[event_col].astype(int).mean()),
        "median_survival_time": float(kmf.median_survival_time_),
    }


# ---------------------------------------------------------------------------
# Censoring calibration''',
    "Kaplan-Meier validator",
)

# ---------------------------------------------------------------------------
# 12. Калибровка: baseline_hazard по умолчанию = 1/1500
# ---------------------------------------------------------------------------
rep(
    "baseline_hazard: float = 0.05,",
    "baseline_hazard: Optional[float] = None,",
    "calibrate baseline signature",
)

rep(
    '''    if post_check_n <= 0:
        raise ValueError("post_check_n must be > 0")
    if baseline_hazard <= 0.0:
        raise ValueError("baseline_hazard must be > 0")''',
    '''    if post_check_n <= 0:
        raise ValueError("post_check_n must be > 0")
    if baseline_hazard is None:
        baseline_hazard = DEFAULT_BASELINE_HAZARD
    if baseline_hazard <= 0.0:
        raise ValueError("baseline_hazard must be > 0")
    dgp = _validate_dgp(dgp)''',
    "calibrate baseline default",
)

# ---------------------------------------------------------------------------
# 13. Калибровка: event_rate должен учитывать event_definition и competing risks
# ---------------------------------------------------------------------------
rep(
    '''    def event_rate_for_scale(scale: float) -> float:
        censoring_time = -np.log(u_censor) * scale
        event = true_time <= censoring_time
        return float(event.mean())''',
    '''    T_minor = np.asarray(base["T_minor"], dtype=float)
    event_def = str(getattr(dgp, "event_definition", "major_claim")).lower()
    competing = bool(getattr(dgp, "competing_risks", False))

    def event_rate_for_scale(scale: float) -> float:
        censoring_time = -np.log(u_censor) * scale

        if competing and event_def == "any_failure":
            event = (true_time <= censoring_time) | (T_minor <= censoring_time)
        elif competing:
            event = (true_time <= censoring_time) & (true_time <= T_minor)
        else:
            event = true_time <= censoring_time

        return float(event.mean())''',
    "calibrate event rate",
)

# ---------------------------------------------------------------------------
# 14. Сохраняем итоговый файл
# ---------------------------------------------------------------------------
DST.write_text(s, encoding="utf-8")
print(f"Ready: {DST}")