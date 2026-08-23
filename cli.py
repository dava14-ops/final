"""
cli.py (v3.1)

Interactive CLI for insurance premium calculation.

Aligned with:
DGP v3.0 (engine_hours);
prediction_engine.py v3.1.0;
service.py;
premium_engine.py.

Ключевые конвенции:
• Пользовательский ввод времени по умолчанию — мото-часы.
• Дисконтирование выполняется в календарных днях.
• Вероятность передаётся в premium_engine для того же горизонта,
  что и период страхования/покрытия.
• Единицы модели читаются из training_meta.time_unit и корректно
  конвертируются в мото-часы и календарные дни.
• Если вероятность модели относится к более широкому событию,
  используется доля застрахованного события.
• Если вероятность уже относится к застрахованному событию,
  следует задать долю 1.0 или указать это в метаданных модели.
• При наличии major_failure_share_prior.mean CLI использует prior.mean
  как fallback вместо константы MAJOR_FAILURE_SHARE.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Set

from exceptions import (
    ModelLoadError,
    PredictionError,
    InvalidInputError,
)
from service import predict_from_model_file
from premium_engine import calculate_premium

# --- Фаза 7.9: severity_model интеграция ---
try:
    from severity_model import load_severity_model
    HAS_SEVERITY_MODEL = True
except ImportError:
    HAS_SEVERITY_MODEL = False

from constants import (
    DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY,
    CALIBRATION_HORIZON_DAYS,
    CALIBRATION_HORIZON_ENGINE_HOURS,
    MAJOR_FAILURE_SHARE,
)


# ---------------------------------------------------------------------------
# Constants (removed: now imported from constants.py)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "model_params.json"
DEFAULT_SUM_INSURED = 5_000_000.0
# DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY, CALIBRATION_HORIZON_DAYS,
# CALIBRATION_HORIZON_ENGINE_HOURS, MAJOR_FAILURE_SHARE: from constants

DEFAULT_HORIZON_ENGINE_HOURS = CALIBRATION_HORIZON_ENGINE_HOURS
DEFAULT_HORIZON_DAYS = CALIBRATION_HORIZON_DAYS

DEFAULT_THETA = 0.15
DEFAULT_DISCOUNT_RATE = 0.08
DEFAULT_RESIDUAL_POLICY = "plug-in"
DEFAULT_TIME_UNIT = "engine_hours"

# Экспертная доля застрахованного события используется только как fallback,
# если в метаданных нет явной доли, event_type и major_failure_share_prior.
# MAJOR_FAILURE_SHARE: from constants

ALLOWED_RESIDUAL_POLICIES = {"plug-in", "mean", "zero"}
YES_ANSWERS = {"y", "yes", "да"}

# Технические/сгенерированные поля, которые нельзя переопределять вручную.
TECHNICAL_COVARIATES = {
    "const",
    "intercept",
    "_const",
    "_intercept",
    "peakload",
    "peak_load",
    "v_hat",
    "vhat",
}

ENGINE_HOUR_UNITS = {
    "engine_hours",
    "engine_hour",
    "engine hours",
    "engine-hours",
    "enginehours",
    "motor_hours",
    "motor-hours",
    "motor hours",
    "moto_hours",
    "moto-hours",
    "moto hours",
    "hours",
    "hour",
    "h",
}

DAY_UNITS = {
    "days",
    "day",
    "calendar_days",
    "calendar_day",
    "calendar days",
    "calendar-days",
    "d",
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging and universal helpers
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """
    Configure logging for CLI usage.

    Warnings/errors are written to stderr, while interactive prompts
    and regular result output remain on stdout.
    """
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
            stream=sys.stderr,
        )


def _get_dict(value: Any) -> Dict[str, Any]:
    """
    Safely return a dict from JSON value.

    If value is None or not a dict, return empty dict.
    """
    return value if isinstance(value, dict) else {}


def _normalize_name(value: Any) -> str:
    """
    Normalize covariate/field name.
    """
    if value is None:
        return ""
    return str(value).strip()


def _optional_float(value: Any) -> Optional[float]:
    """
    Convert value to finite float or return None.

    Supports comma as decimal separator for string values.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, str):
        value = value.strip().replace(",", ".")

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def _to_float(value: Any, field: str) -> float:
    """
    Convert value to float or raise InvalidInputError.
    """
    result = _optional_float(value)
    if result is None:
        raise InvalidInputError(
            f"Поле '{field}' должно быть конечным числом, получено: {value!r}"
        )
    return result


def _format_number(value: Any) -> str:
    """
    Format numeric value for prompts and reports.
    """
    number = _optional_float(value)
    if number is None:
        return str(value)

    if math.isclose(number, round(number), abs_tol=1e-12):
        return str(int(round(number)))

    text = f"{number:.10f}".rstrip("0").rstrip(".")
    return text or "0"


def _format_optional_float(
    value: Any,
    *,
    fmt: str = "{:.3f}",
    default: str = "n/a",
) -> str:
    """
    Safely format optional float value.

    Returns default if value is absent or not a finite number.
    """
    number = _optional_float(value)
    if number is None:
        return default

    return fmt.format(number)


def _premium_float(premium: Any, key: str) -> float:
    """
    Extract required numeric field from premium dict.
    """
    if not isinstance(premium, dict):
        raise PredictionError("Premium calculation result must be a dictionary")

    value = _optional_float(premium.get(key))
    if value is None:
        raise PredictionError(
            f"В результате расчёта премии отсутствует корректное поле: {key}"
        )

    return value


def _discount_factor(rate: Optional[float], days: Optional[float]) -> float:
    """
    Calculate continuous discount factor:

        exp(-r * t / 365)

    If rate or days are non-positive/non-finite, return 1.0.
    """
    rate_value = _optional_float(rate) or 0.0
    days_value = _optional_float(days) or 0.0

    if rate_value <= 0.0 or days_value <= 0.0:
        return 1.0

    return math.exp(-rate_value * days_value / 365.0)


def _as_sequence(value: Any, field: str) -> List[Any]:
    """
    Convert prediction/premium result to list.

    Raises PredictionError for strings, bytes and dictionaries because
    these types are almost certainly invalid result containers here.
    """
    if value is None:
        raise PredictionError(f"Result must contain non-empty '{field}'")

    if isinstance(value, (str, bytes, dict)):
        raise PredictionError(
            f"Field '{field}' must be a sequence, not {type(value).__name__}"
        )

    if isinstance(value, (list, tuple)):
        return list(value)

    if isinstance(value, Iterable):
        try:
            return list(value)
        except TypeError as exc:
            raise PredictionError(
                f"Field '{field}' looks iterable but cannot be converted to list"
            ) from exc

    return [value]


def _validate_probability(value: Any, index: int) -> float:
    """
    Validate and normalize one probability value.

    Small numerical overshoots outside [0, 1] are clamped.
    Material overshoots raise PredictionError.
    """
    p = _optional_float(value)
    if p is None:
        raise PredictionError(
            f"Вероятность с индексом {index} должна быть конечным числом"
        )

    tolerance = 1e-12
    if p < -tolerance or p > 1.0 + tolerance:
        raise PredictionError(
            f"Вероятность с индексом {index} вне диапазона [0, 1]: {p!r}"
        )

    return min(max(p, 0.0), 1.0)


def _validate_probabilities(probabilities: List[Any]) -> List[float]:
    """
    Validate list of probabilities.
    """
    if not probabilities:
        raise PredictionError("Probability list is empty")

    return [_validate_probability(p, i) for i, p in enumerate(probabilities)]


def _validate_premium(
    premium: Any,
    index: int,
    *,
    discount_rate: float,
    theta: float,
) -> None:
    """
    Basic sanity checks for premium engine output.
    """
    net_undiscounted = _premium_float(premium, "net_undiscounted")
    gross_discounted = _premium_float(premium, "gross_discounted")
    loading_amount = _premium_float(premium, "loading_amount")

    # Tariff is required by the existing contract, even though the CLI
    # later computes its own tariff from gross premium and sum insured.
    _premium_float(premium, "tariff")

    tolerance = 1e-6 + 1e-9 * max(
        abs(net_undiscounted),
        abs(gross_discounted),
        abs(loading_amount),
        1.0,
    )

    if net_undiscounted < -tolerance:
        raise PredictionError(
            f"Premium result #{index}: net_undiscounted must be non-negative"
        )

    if gross_discounted < -tolerance:
        raise PredictionError(
            f"Premium result #{index}: gross_discounted must be non-negative"
        )

    if theta >= 0.0 and loading_amount < -tolerance:
        raise PredictionError(
            f"Premium result #{index}: loading_amount must be non-negative for theta >= 0"
        )

    base_net = net_undiscounted

    if discount_rate > 0.0:
        net_discounted = _premium_float(premium, "net_discounted")
        base_net = net_discounted

        if net_discounted < -tolerance:
            raise PredictionError(
                f"Premium result #{index}: net_discounted must be non-negative"
            )

        if net_discounted > net_undiscounted + tolerance:
            raise PredictionError(
                f"Premium result #{index}: net_discounted cannot exceed net_undiscounted"
            )

        discount_factor = _optional_float(_get_dict(premium).get("discount_factor"))
        if discount_factor is not None:
            if discount_factor < -1e-12 or discount_factor > 1.0 + 1e-12:
                raise PredictionError(
                    f"Premium result #{index}: discount_factor must be in [0, 1]"
                )

    if theta >= 0.0 and gross_discounted + tolerance < base_net:
        raise PredictionError(
            f"Premium result #{index}: gross_discounted cannot be less than net premium"
        )


# ---------------------------------------------------------------------------
# Time conversion helpers
# ---------------------------------------------------------------------------


def engine_hours_to_calendar_days(
    engine_hours: float,
    hours_per_day: float = DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY,
) -> float:
    """
    Convert engine hours to calendar days.
    """
    if hours_per_day <= 0.0:
        return engine_hours

    return engine_hours / hours_per_day


def calendar_days_to_engine_hours(
    calendar_days: float,
    hours_per_day: float = DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY,
) -> float:
    """
    Convert calendar days to engine hours.
    """
    if hours_per_day <= 0.0:
        return calendar_days

    return calendar_days * hours_per_day


def _unit_category(unit: Optional[str]) -> str:
    """
    Normalize time unit to one of:

    - engine_hours
    - days

    Raises ModelLoadError for unsupported units.
    """
    if unit is None:
        return "engine_hours"

    normalized = str(unit).strip().lower()
    if not normalized:
        return "engine_hours"

    if normalized in ENGINE_HOUR_UNITS:
        return "engine_hours"

    if normalized in DAY_UNITS:
        return "days"

    raise ModelLoadError(
        "Неподдерживаемая единица времени модели: "
        f"{unit!r}. Ожидаются engine_hours/days или совместимые алиасы."
    )


def _engine_hours_to_model_units(
    engine_hours: float,
    model_unit: Optional[str],
    hours_per_day: float,
) -> float:
    """
    Convert user/engine-hours horizon to model time units.
    """
    category = _unit_category(model_unit)

    if category == "engine_hours":
        return engine_hours

    if hours_per_day <= 0.0:
        raise ModelLoadError(
            "default_engine_hours_per_calendar_day должен быть положительным"
        )

    return engine_hours / hours_per_day


def _model_units_to_engine_hours(
    value: float,
    model_unit: Optional[str],
    hours_per_day: float,
) -> float:
    """
    Convert model time units to engine hours.
    """
    category = _unit_category(model_unit)

    if category == "engine_hours":
        return value

    if hours_per_day <= 0.0:
        raise ModelLoadError(
            "default_engine_hours_per_calendar_day должен быть положительным"
        )

    return value * hours_per_day


def _model_units_to_calendar_days(
    value: float,
    model_unit: Optional[str],
    hours_per_day: float,
) -> float:
    """
    Convert model time units to calendar days.
    """
    category = _unit_category(model_unit)

    if category == "days":
        return value

    if hours_per_day <= 0.0:
        raise ModelLoadError(
            "default_engine_hours_per_calendar_day должен быть положительным"
        )

    return value / hours_per_day


# ---------------------------------------------------------------------------
# Model JSON helpers
# ---------------------------------------------------------------------------


def load_model_json(path: str) -> Dict[str, Any]:
    """
    Load model JSON file and validate that root is a dictionary.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise ModelLoadError(f"Cannot read model file: {path}") from exc

    if not isinstance(data, dict):
        raise ModelLoadError(f"Model file must contain a JSON object: {path}")

    return data


def _read_time_unit(model_json: Dict[str, Any]) -> Optional[str]:
    """
    Read time unit from model metadata.
    """
    meta = _get_dict(model_json.get("training_meta"))
    unit = meta.get("time_unit")

    if unit is None:
        return None

    unit = str(unit).strip().lower()
    return unit or None


def _read_hours_per_day(model_json: Dict[str, Any]) -> float:
    """
    Read engine hours per calendar day from model metadata.
    """
    meta = _get_dict(model_json.get("training_meta"))
    raw = meta.get(
        "default_engine_hours_per_calendar_day",
        DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY,
    )

    if isinstance(raw, str):
        raw = raw.strip().replace(",", ".")

    try:
        hpd = float(raw)
        if math.isfinite(hpd) and hpd > 0.0:
            return hpd
    except (TypeError, ValueError):
        pass

    return DEFAULT_ENGINE_HOURS_PER_CALENDAR_DAY


def _read_calibration_horizon(model_json: Dict[str, Any]) -> Optional[float]:
    """
    Read calibration horizon from model metadata.

    The value is returned in model time units.
    """
    meta = _get_dict(model_json.get("training_meta"))
    return _optional_float(meta.get("calibration_time_horizon"))


def _read_covered_event_share(model_json: Dict[str, Any]) -> float:
    """
    Read default covered-event share from model metadata.

    This is the share of the modelled event that is actually insured.

    For example, if the model predicts any failure, but the policy covers
    only major failure, this share can be 0.30.

    If the model probability already corresponds to the insured event,
    metadata should set this share to 1.0.
    """
    meta = _get_dict(model_json.get("training_meta"))

    if meta.get("premium_engine_applies_covered_event_share") in (
        True,
        1,
        "1",
        "true",
        "True",
    ):
        return 1.0

    keys = (
        "covered_event_share",
        "insurance_event_share",
        "insured_event_share",
        "major_failure_share",
    )

    for key in keys:
        value = _optional_float(meta.get(key))
        if value is None:
            continue

        if 0.0 <= value <= 1.0:
            return value

        if 1.0 < value <= 100.0:
            logger.warning(
                "Значение '%s' похоже на проценты (%s). Конвертирую в долю.",
                key,
                _format_number(value),
            )
            return value / 100.0

        logger.warning(
            "Некорректное значение '%s' в метаданных: %r. Ожидается доля от 0 до 1.",
            key,
            value,
        )

    event_type = str(
        meta.get("probability_event_type")
        or meta.get("event_type")
        or ""
    ).strip().lower()

    if event_type in {
        "major_failure",
        "major failure",
        "covered_event",
        "insured_event",
        "insured event",
    }:
        return 1.0

    # P-04: если задан Beta-prior для доли major-отказов, используем его mean
    # как fallback вместо жёсткой константы MAJOR_FAILURE_SHARE.
    prior = _get_dict(meta.get("major_failure_share_prior"))
    prior_mean = _optional_float(prior.get("mean"))

    if prior_mean is not None:
        if 0.0 <= prior_mean <= 1.0:
            return prior_mean

        if 1.0 < prior_mean <= 100.0:
            logger.warning(
                "major_failure_share_prior.mean похоже на проценты (%s). "
                "Конвертирую в долю.",
                _format_number(prior_mean),
            )
            return prior_mean / 100.0

        logger.warning(
            "Некорректное major_failure_share_prior.mean: %r. "
            "Ожидается доля от 0 до 1.",
            prior_mean,
        )

    return MAJOR_FAILURE_SHARE


# ---------------------------------------------------------------------------
# PeakLoad selection
# ---------------------------------------------------------------------------


def _parse_peak_values(raw: str) -> List[float]:
    """
    Parse user-provided PeakLoad values.

    Expected separator for multiple values is semicolon:

        1.5; 2.3; 4,7

    Decimal comma is supported.

    If no semicolon is present, the whole input is treated as one value.
    """
    raw = raw.strip()
    if not raw:
        raise InvalidInputError("Введите хотя бы одно значение PeakLoad")

    if ";" in raw:
        parts = [part.strip() for part in raw.split(";") if part.strip()]
    else:
        parts = [raw]

    values: List[float] = []
    for part in parts:
        values.append(_to_float(part.replace(",", "."), "PeakLoad"))

    return values


def choose_peaks(model_json: Dict[str, Any]) -> List[float]:
    """
    Ask user to select PeakLoad value(s).
    """
    meta = _get_dict(model_json.get("training_meta"))

    p25 = _optional_float(meta.get("peakload_p25"))
    median = _optional_float(meta.get("peakload_median"))
    p75 = _optional_float(meta.get("peakload_p75"))

    if p25 is None and median is None and p75 is None:
        legacy_peakload = _get_dict(meta.get("peakload"))
        p25 = _optional_float(legacy_peakload.get("p25"))
        median = _optional_float(legacy_peakload.get("median"))
        p75 = _optional_float(legacy_peakload.get("p75"))

    available: Dict[str, float] = {}

    for key, value in (
        ("p25", p25),
        ("median", median),
        ("p75", p75),
    ):
        if value is not None:
            available[key] = value

    if not available:
        raise InvalidInputError("No PeakLoad statistics available in training_meta")

    display_keys = [key for key in ("p25", "median", "p75") if key in available]

    labels = {
        "p25": "НИЖНЯЯ КВАРТИЛЬ (25%)",
        "median": "МЕДИАНА (50%)",
        "p75": "ВЕРХНЯЯ КВАРТИЛЬ (75%)",
    }

    if "median" in display_keys:
        default_choice = str(display_keys.index("median") + 1)
    else:
        default_choice = "1"

    while True:
        print("\nВыберите значение пиковой нагрузки (PeakLoad):")

        for i, key in enumerate(display_keys, 1):
            print(
                f"{i}) {labels.get(key, key.upper())} = "
                f"{_format_number(available[key])}"
            )

        print("4) Все значения")
        print("5) Своё значение")

        choice = input(f"Выбор [{default_choice}]: ").strip() or default_choice

        if choice == "5":
            raw = input(
                "Введите значения через точку с запятой "
                "(десятичная запятая поддерживается): "
            ).strip()

            try:
                values = _parse_peak_values(raw)
            except InvalidInputError as exc:
                logger.warning("%s. Повторите ввод.", exc)
                continue

            if not values:
                logger.warning("Введите хотя бы одно числовое значение.")
                continue

            return values

        key_map = {str(i): key for i, key in enumerate(display_keys, 1)}
        key_map["4"] = "all"

        if choice not in key_map:
            logger.warning(
                "Неверный выбор. Выберите номер из списка, "
                "4 для всех значений или 5 для своего значения."
            )
            continue

        target = key_map[choice]

        if target == "all":
            return [available[key] for key in display_keys]

        return [available[target]]


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def ask_float(
    text: str,
    default: Any,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """
    Ask user for float value with default and optional range validation.

    Supports comma as decimal separator.

    The default value is validated/clamped before being offered.
    """
    default_value = _optional_float(default)

    if default_value is None:
        default_value = 0.0 if min_value is None else min_value

    if min_value is not None and default_value < min_value:
        logger.warning(
            "Значение по умолчанию %s меньше допустимого минимума %s. "
            "Используется минимум.",
            _format_number(default_value),
            _format_number(min_value),
        )
        default_value = min_value

    if max_value is not None and default_value > max_value:
        logger.warning(
            "Значение по умолчанию %s больше допустимого максимума %s. "
            "Используется максимум.",
            _format_number(default_value),
            _format_number(max_value),
        )
        default_value = max_value

    while True:
        prompt = f"{text} [{_format_number(default_value)}]: "
        raw = input(prompt).strip().replace(",", ".")

        if not raw:
            return default_value

        value = _optional_float(raw)
        if value is None:
            logger.warning("Нужно ввести конечное число, например 123.45.")
            continue

        if min_value is not None and value < min_value:
            logger.warning(
                "Значение должно быть не меньше %s.",
                _format_number(min_value),
            )
            continue

        if max_value is not None and value > max_value:
            logger.warning(
                "Значение должно быть не больше %s.",
                _format_number(max_value),
            )
            continue

        return value


# ---------------------------------------------------------------------------
# Covariates input
# ---------------------------------------------------------------------------


def _get_available_covariates(model_json: Dict[str, Any]) -> List[str]:
    """
    Collect covariate names from:

    - template_covariates
    - first_stage.exog_names
    - cox.exog_names

    Technical/generated fields are excluded globally.
    """
    available: List[str] = []
    seen: Set[str] = set()

    def add_names(names: Any, excluded: Set[str]) -> None:
        if names is None or isinstance(names, (str, bytes)):
            return

        try:
            names_list = list(names)
        except TypeError:
            return

        for name in names_list:
            normalized = _normalize_name(name)
            if not normalized:
                continue

            if normalized.lower() in excluded:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            available.append(normalized)

    template_covariates = _get_dict(model_json.get("template_covariates"))
    add_names(template_covariates.keys(), TECHNICAL_COVARIATES)

    first_stage = _get_dict(model_json.get("first_stage"))
    add_names(first_stage.get("exog_names", []), TECHNICAL_COVARIATES)

    cox = _get_dict(model_json.get("cox"))
    add_names(cox.get("exog_names", []), TECHNICAL_COVARIATES)

    return available


def _parse_covariate_value(
    name: str,
    raw_value: Any,
    categorical_encoding: Dict[str, Any],
) -> Optional[float]:
    """
    Parse covariate value.

    Supports:
    - numeric values
    - comma as decimal separator
    - categorical labels if categorical_encoding is present
    - numeric categorical codes if they match encoding values
    """
    text = "" if raw_value is None else str(raw_value).strip()
    if not text:
        return None

    cat_mapping = _get_dict(categorical_encoding.get(name))

    if not cat_mapping:
        return _optional_float(text)

    # 1) Direct categorical label: Brand=BrandA
    if text in cat_mapping:
        return _optional_float(cat_mapping[text])

    # 2) Numeric encoded value: Brand=1
    numeric_value = _optional_float(text)
    if numeric_value is not None:
        for encoded_value in cat_mapping.values():
            encoded_numeric = _optional_float(encoded_value)
            if encoded_numeric is not None and math.isclose(
                encoded_numeric,
                numeric_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return numeric_value

    return None


def ask_covariates(model_json: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Ask user for optional covariate overrides.

    Returns user-provided covariate values, or None if skipped/empty.
    """
    print()
    print("[ОПЦИОНАЛЬНО] Индивидуальные ковариаты (X, Z, Age, Climate, Soil...)")
    print("Оставьте пустым для использования профиля по умолчанию (template).")

    use_input = input("Ввести ковариаты? [y/N]: ").strip().lower()
    if use_input not in YES_ANSWERS:
        print("Используется профиль по умолчанию (template_covariates).")
        return None

    available = _get_available_covariates(model_json)
    if not available:
        logger.warning(
            "Не найдено ковариат в модели. Используется профиль по умолчанию."
        )
        return None

    available_set = set(available)
    categorical_encoding = _get_dict(model_json.get("categorical_encoding"))

    json_path = input(
        "Путь к JSON-файлу с ковариатами (или Enter для ручного ввода): "
    ).strip()

    if json_path:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if not isinstance(loaded, dict):
                raise ValueError(
                    "JSON-файл с ковариатами должен содержать объект {name: value}"
                )

            covariates: Dict[str, float] = {}

            for raw_key, raw_value in loaded.items():
                name = _normalize_name(raw_key)

                if name not in available_set:
                    logger.warning(
                        "Ковариата '%s' из JSON не найдена в модели, пропущена",
                        name,
                    )
                    continue

                value = _parse_covariate_value(name, raw_value, categorical_encoding)
                if value is None:
                    logger.warning(
                        "Пропущено некорректное значение ковариаты из JSON: %s=%r",
                        name,
                        raw_value,
                    )
                    continue

                covariates[name] = value

            if not covariates:
                logger.warning(
                    "В JSON не найдено корректных ковариат. "
                    "Используется профиль по умолчанию."
                )
                return None

            print(f"Ковариаты загружены из {json_path}")
            return covariates

        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Не удалось загрузить файл: %s", exc)
            logger.warning("Переход к ручному вводу.")

    print(f"\nДоступные ковариаты: {', '.join(available)}")

    if categorical_encoding:
        print("\nКатегориальные переменные (можно вводить значения или их числовые коды):")
        for feat, enc in categorical_encoding.items():
            if feat not in available_set:
                continue

            enc_dict = _get_dict(enc)
            if enc_dict:
                print(f"  {feat}: {list(enc_dict.keys())}")

    print("\nВведите значения через точку с запятой: name1=value1; name2=value2")
    print("Десятичная запятая поддерживается, например: Z=0,75; Age=3")

    raw = input("Ковариаты: ").strip()

    if not raw:
        print("Используется профиль по умолчанию (template_covariates).")
        return None

    if ";" not in raw and raw.count("=") > 1:
        logger.warning(
            "Несколько пар ковариат нужно разделять точкой с запятой (;). "
            "Попытка разобрать ввод как есть."
        )

    if ";" in raw:
        pairs = [pair.strip() for pair in raw.split(";") if pair.strip()]
    else:
        pairs = [raw]

    covariates = {}

    for pair in pairs:
        if "=" not in pair:
            logger.warning(
                "Пропущен некорректный формат: %s (ожидается name=value)",
                pair,
            )
            continue

        name, value = pair.split("=", 1)
        name = _normalize_name(name)

        if name not in available_set:
            logger.warning("Ковариата '%s' не найдена в модели, пропущена", name)
            continue

        parsed_value = _parse_covariate_value(name, value, categorical_encoding)
        if parsed_value is None:
            logger.warning(
                "Пропущено некорректное значение ковариаты: %s=%r",
                name,
                value,
            )
            continue

        covariates[name] = parsed_value

    if not covariates:
        print("Используется профиль по умолчанию (template_covariates).")
        return None

    print(f"Введены ковариаты: {covariates}")
    return covariates


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def run_cli() -> int:
    """
    Run interactive CLI.

    Returns
    -------
    int
        Exit code:

        0 - success
        2 - expected user/model/prediction/input error
        3 - unexpected error
    """
    try:
        model_path = (
            input(f"Путь к файлу модели [{DEFAULT_MODEL}]: ")
            .strip()
            or DEFAULT_MODEL
        )

        model_json = load_model_json(model_path)
        meta = _get_dict(model_json.get("training_meta"))

        model_unit = _read_time_unit(model_json)
        model_unit_category = _unit_category(model_unit)
        hours_per_day = _read_hours_per_day(model_json)

        if model_unit is None:
            logger.info(
                "В модели не указан time_unit. Предполагается '%s'.",
                DEFAULT_TIME_UNIT,
            )
        elif model_unit_category != "engine_hours":
            logger.warning(
                "Модель использует единицы времени '%s'. "
                "Конверсия в мото-часы будет выполнена автоматически.",
                model_unit,
            )

        # -------------------------------------------------------------------
        # Calibration horizon
        # -------------------------------------------------------------------
        calib_model_units = _read_calibration_horizon(model_json)
        calib_engine_hours: Optional[float] = None
        calib_calendar_days: Optional[float] = None

        if calib_model_units is not None:
            calib_engine_hours_value = _model_units_to_engine_hours(
                calib_model_units,
                model_unit,
                hours_per_day,
            )
            calib_calendar_days_value = _model_units_to_calendar_days(
                calib_model_units,
                model_unit,
                hours_per_day,
            )

            calib_engine_hours = calib_engine_hours_value
            calib_calendar_days = calib_calendar_days_value

            calib_prob = _optional_float(meta.get("calibration_target_probability"))
            prob_str = ""

            if calib_prob is not None:
                calib_prob_value = calib_prob

                if 0.0 <= calib_prob_value <= 1.0:
                    prob_str = f"{calib_prob_value * 100:.1f}%"
                elif 1.0 < calib_prob_value <= 100.0:
                    prob_str = f"{calib_prob_value:.1f}%"
                    logger.warning(
                        "calibration_target_probability похожа на проценты: %s",
                        _format_number(calib_prob_value),
                    )
                else:
                    prob_str = _format_number(calib_prob_value)
                    logger.warning(
                        "Некорректная calibration_target_probability: %r",
                        calib_prob_value,
                    )

            calib_method = str(meta.get("calibration_method", "") or "")

            print()
            print("[INFO] Горизонт калибровки модели:")
            print(
                f"  Обучение: {_format_number(calib_engine_hours_value)} мото-часов "
                f"(≈ {_format_number(calib_calendar_days_value)} календарных дней "
                f"при {_format_number(hours_per_day)} мч/день)"
            )
            print(
                f"  В единицах модели: {_format_number(calib_model_units)} "
                f"{model_unit or DEFAULT_TIME_UNIT}"
            )

            if prob_str:
                print(f"  Целевая вероятность: {prob_str}")

            if calib_method:
                print(f"  Метод калибровки: {calib_method}")

            # P-03 / P-04: определение события и prior
            event_definition = meta.get("event_definition")
            if event_definition:
                print(f"  Определение события: {event_definition}")

            major_failure_prior = _get_dict(meta.get("major_failure_share_prior"))
            prior_mean = _optional_float(major_failure_prior.get("mean"))

            if prior_mean is not None:
                prior_ci_low_text = _format_optional_float(
                    major_failure_prior.get("ci_low")
                )
                prior_ci_high_text = _format_optional_float(
                    major_failure_prior.get("ci_high")
                )

                print(
                    f"  Доля major-отказов: mean={prior_mean:.3f} "
                    f"95%CI=[{prior_ci_low_text}, {prior_ci_high_text}]"
                )

            print()

        # -------------------------------------------------------------------
        # PeakLoad selection
        # -------------------------------------------------------------------
        peaks: List[float] = choose_peaks(model_json)

        # -------------------------------------------------------------------
        # Financial inputs
        # -------------------------------------------------------------------
        sum_insured = ask_float(
            "Сумма страхового возмещения (максимальная выплата по полису, руб.)",
            DEFAULT_SUM_INSURED,
            min_value=1e-9,
        )

        default_covered_event_share = _read_covered_event_share(model_json)
        covered_event_share = ask_float(
            "Доля застрахованного события "
            "(например, major failure share; 1.0, если вероятность уже застрахованного события)",
            default_covered_event_share,
            min_value=0.0,
            max_value=1.0,
        )

        # -------------------------------------------------------------------
        # Horizon input.
        #
        # The same horizon is used for:
        #   1) probability prediction;
        #   2) insurance coverage period;
        #   3) discounting.
        #
        # This removes horizon mismatch risk.
        # -------------------------------------------------------------------
        print()
        print("Как задать горизонт страхования/прогнозирования?")
        print("  1) Мото-часы (рекомендуется)")
        print("  2) Календарные дни (конвертация в мото-часы)")

        while True:
            time_input_choice = input("Выбор [1]: ").strip() or "1"
            if time_input_choice in {"1", "2"}:
                break
            logger.warning("Выберите 1 или 2.")

        if time_input_choice == "2":
            horizon_days_default = (
                calib_calendar_days
                if calib_calendar_days is not None
                else DEFAULT_HORIZON_DAYS
            )

            horizon_calendar_days = ask_float(
                "Горизонт страхования/прогнозирования (календарные дни)",
                horizon_days_default,
                min_value=1e-9,
            )

            horizon_engine_hours = calendar_days_to_engine_hours(
                horizon_calendar_days,
                hours_per_day,
            )

            print(
                f"  → {_format_number(horizon_calendar_days)} дн. × "
                f"{_format_number(hours_per_day)} мч/дн. = "
                f"{_format_number(horizon_engine_hours)} мото-часов"
            )

        else:
            horizon_engine_default = (
                calib_engine_hours
                if calib_engine_hours is not None
                else DEFAULT_HORIZON_ENGINE_HOURS
            )

            horizon_engine_hours = ask_float(
                "Горизонт страхования/прогнозирования (мото-часы)",
                horizon_engine_default,
                min_value=1e-9,
            )

            horizon_calendar_days = engine_hours_to_calendar_days(
                horizon_engine_hours,
                hours_per_day,
            )

        horizon_model_units = _engine_hours_to_model_units(
            horizon_engine_hours,
            model_unit,
            hours_per_day,
        )

        if model_unit_category != "engine_hours":
            print(
                f"  [INFO] Конверсия горизонта: "
                f"{_format_number(horizon_engine_hours)} мото-часов = "
                f"{_format_number(horizon_model_units)} "
                f"{model_unit or 'единиц модели'}"
            )

        if (
            calib_engine_hours is not None
            and calib_engine_hours > 0.0
            and abs(horizon_engine_hours - calib_engine_hours) > 1e-6
        ):
            print()
            print(
                f"[INFO] Вы ввели {_format_number(horizon_engine_hours)} мото-часов, "
                f"но модель калибрована на {_format_number(calib_engine_hours)} мото-часов."
            )
            print(
                "Вероятность вычисляется Cox-моделью напрямую для заданного горизонта "
                "через базовую накопленную функцию риска H0(t)."
            )

            if horizon_engine_hours > 1.25 * calib_engine_hours:
                logger.warning(
                    "Горизонт существенно больше калибровочного. "
                    "Результат может быть экстраполяцией."
                )
            elif horizon_engine_hours < 0.75 * calib_engine_hours:
                logger.warning(
                    "Горизонт существенно меньше калибровочного. "
                    "Проверьте корректность базового риска для коротких горизонтов."
                )

            print()

        # -------------------------------------------------------------------
        # Loading and discount
        # -------------------------------------------------------------------
        theta = ask_float(
            "Нагрузка на надбавку (theta, коэффициент страховой нагрузки, доля)",
            DEFAULT_THETA,
            min_value=0.0,
        )

        if theta > 1.0:
            logger.warning(
                "theta = %s выглядит больше 100%%. "
                "Проверьте, что вы ввели долю, а не проценты.",
                _format_number(theta),
            )

        discount_rate = ask_float(
            "Дисконтная ставка (годовая, доля)",
            DEFAULT_DISCOUNT_RATE,
            min_value=0.0,
        )

        if discount_rate > 1.0:
            logger.warning(
                "Дисконтная ставка = %s выглядит больше 100%%. "
                "Проверьте, что вы ввели долю, а не проценты.",
                _format_number(discount_rate),
            )

        policy_horizon_days = horizon_calendar_days

        if discount_rate > 0.0:
            print("\n[INFO] Премия будет дисконтирована:")
            print("  PV = FV * exp(-r * t / 365)")
            print(
                f"  r = {_format_number(discount_rate)} "
                f"({discount_rate * 100:.2f}% годовых)"
            )
            print(
                f"  t = {_format_number(policy_horizon_days)} календарных дней"
            )

        # -------------------------------------------------------------------
        # Residual endogeneity policy
        # -------------------------------------------------------------------
        residual_prompt = (
            f"Метод учёта остаточной эндогенности [{DEFAULT_RESIDUAL_POLICY}]:\n"
            "  plug-in — стандартный метод (по умолчанию)\n"
            "  bootstrap — НЕ ПОДДЕРЖИВАЕТСЯ (используйте plug-in)\n"
            "  mean — по среднему значению\n"
            "  zero — обнулить\n"
            f"Ввод [{DEFAULT_RESIDUAL_POLICY}]: "
        )

        while True:
            residual_policy = input(residual_prompt).strip().lower()

            if not residual_policy:
                residual_policy = DEFAULT_RESIDUAL_POLICY

            if residual_policy == "bootstrap":
                logger.warning(
                    "bootstrap не поддерживается. Используйте plug-in, mean или zero."
                )
                continue

            if residual_policy in ALLOWED_RESIDUAL_POLICIES:
                break

            logger.warning("Допустимые значения: plug-in, mean, zero.")

        # -------------------------------------------------------------------
        # Optional covariates
        # -------------------------------------------------------------------
        covariates = ask_covariates(model_json)

        # -------------------------------------------------------------------
        # Prediction
        # -------------------------------------------------------------------
        prediction = predict_from_model_file(
            model_path,
            peaks,
            horizon_model_units,
            residual_policy,
            x_values=covariates,
        )

        if not isinstance(prediction, dict):
            raise PredictionError("Prediction service returned invalid result")

        if covariates:
            print("\n[INFO] Расчёт выполнен с индивидуальными ковариатами")
        else:
            print("\n[INFO] Расчёт выполнен с профилем по умолчанию (template)")

        probabilities_raw = prediction.get("probabilities")
        probabilities_list = _as_sequence(probabilities_raw, "probabilities")
        probabilities = _validate_probabilities(probabilities_list)

        raw_peaks_source = prediction.get("raw_peaks", peaks)
        raw_peaks: List[Any]

        if isinstance(raw_peaks_source, (list, tuple)):
            raw_peaks = list(raw_peaks_source)
        elif isinstance(raw_peaks_source, (str, bytes, dict)):
            raw_peaks = list(peaks)
        elif isinstance(raw_peaks_source, Iterable):
            try:
                raw_peaks = list(raw_peaks_source)
            except TypeError:
                raw_peaks = list(peaks)
        else:
            raw_peaks = list(peaks)

        if len(raw_peaks) != len(probabilities):
            if len(peaks) == len(probabilities):
                raw_peaks = list(peaks)
            else:
                raw_peaks = [f"Peak #{i + 1}" for i in range(len(probabilities))]

        # -------------------------------------------------------------------
        # Covered-event probability.
        #
        # If the model probability is for a broader event than the insured
        # event, multiply by the covered-event share. If the probability is
        # already for the insured event, covered_event_share should be 1.0.
        # -------------------------------------------------------------------
        covered_probabilities = [p * covered_event_share for p in probabilities]
        covered_probabilities = _validate_probabilities(covered_probabilities)

        # -------------------------------------------------------------------
        # Premium calculation
        # -------------------------------------------------------------------
        calibration_horizon_days = (
            calib_calendar_days
            if calib_calendar_days is not None
            else policy_horizon_days
        )

        premiums_raw = calculate_premium(
            covered_probabilities,
            sum_insured,
            theta,
            discount_rate,
            calibration_horizon_days,
            policy_horizon_days,
        )

        if premiums_raw is None:
            raise PredictionError("Premium engine returned empty result")

        premium_items: List[Any]

        if isinstance(premiums_raw, dict):
            premium_items = [premiums_raw]
        elif isinstance(premiums_raw, (list, tuple)):
            premium_items = list(premiums_raw)
        elif isinstance(premiums_raw, Iterable):
            try:
                premium_items = list(premiums_raw)
            except TypeError as exc:
                raise PredictionError(
                    "Premium engine returned an iterable that cannot be converted to list"
                ) from exc
        else:
            premium_items = [premiums_raw]

        if len(premium_items) != len(probabilities):
            raise PredictionError(
                "Premium engine returned wrong number of premium results"
            )

        premiums: List[Dict[str, Any]] = []

        for i, premium in enumerate(premium_items):
            if not isinstance(premium, dict):
                raise PredictionError(
                    f"Premium result #{i} must be a dictionary, "
                    f"got {type(premium).__name__}"
                )

            _validate_premium(
                premium,
                i,
                discount_rate=discount_rate,
                theta=theta,
            )
            premiums.append(premium)

        # -------------------------------------------------------------------
        # Additional info
        # -------------------------------------------------------------------
        print()
        print(
            f"[INFO] Вероятность рассчитана для горизонта: "
            f"{_format_number(horizon_engine_hours)} мото-часов"
        )

        if model_unit_category != "engine_hours":
            print(
                f"  В единицах модели: {_format_number(horizon_model_units)} "
                f"{model_unit or 'единиц модели'}"
            )

        print(
            f"[INFO] Период покрытия/дисконтирования: "
            f"{_format_number(policy_horizon_days)} календарных дней"
        )

        if abs(covered_event_share - 1.0) > 1e-12:
            print(
                f"[INFO] Доля застрахованного события: "
                f"{_format_number(covered_event_share)}"
            )

        if discount_rate > 0.0:
            discount_factor_info = _discount_factor(discount_rate, policy_horizon_days)
            print(
                f"[INFO] Дисконтная ставка: {discount_rate * 100:.2f}%, "
                f"фактор дисконтирования: {discount_factor_info:.4f}"
            )

        print()

        # -------------------------------------------------------------------
        # Results
        # -------------------------------------------------------------------
        print("\nResults:\n")

        for peak, model_probability, covered_probability, premium in zip(
            raw_peaks,
            probabilities,
            covered_probabilities,
            premiums,
        ):
            peak_number = _optional_float(peak)

            if peak_number is not None:
                peak_text = _format_number(peak_number)
            elif peak is None:
                peak_text = "n/a"
            else:
                peak_text = str(peak)

            print("Пиковая нагрузка (PeakLoad):")
            print(peak_text)

            print(
                f"Вероятность события по модели (Cox, "
                f"{_format_number(horizon_engine_hours)} мото-часов):"
            )
            print(f"{model_probability:.8f}")

            if abs(covered_event_share - 1.0) > 1e-12:
                print()
                print("Вероятность застрахованного события:")
                print(f"{covered_probability:.8f}")
                print(
                    f"(модельная вероятность × доля застрахованного события "
                    f"{_format_number(covered_event_share)})"
                )

            net_undiscounted = _premium_float(premium, "net_undiscounted")
            gross_discounted = _premium_float(premium, "gross_discounted")
            loading_amount = _premium_float(premium, "loading_amount")

            print()
            print("Чистая премия (нетто, без дисконта):")
            print(f"{net_undiscounted:.2f} руб.")

            if discount_rate > 0.0:
                net_discounted = _premium_float(premium, "net_discounted")
                discount_factor = _optional_float(
                    _get_dict(premium).get("discount_factor")
                )

                if discount_factor is None:
                    discount_factor = _discount_factor(
                        discount_rate,
                        policy_horizon_days,
                    )

                print()
                print("Дисконтированная нетто-премия:")
                print(
                    f"r = {discount_rate * 100:.1f}% годовых, "
                    f"t = {_format_number(policy_horizon_days)} дн."
                )
                print(f"{net_discounted:.2f} руб.")
                print(f"Фактор дисконтирования: {discount_factor:.6f}")

            tariff_ratio = gross_discounted / sum_insured

            print()
            print(
                f"Брутто премия (нетто + нагрузка {theta * 100:.0f}%, "
                "дисконтированная):"
            )
            print(f"{gross_discounted:.2f} руб.")

            print("В т.ч. нагрузка:")
            print(f"{loading_amount:.2f} руб.")

            print("Тариф (премия на единицу страховой суммы):")
            print(f"{tariff_ratio * 100:.6f}%")

            print()

        return 0

    except EOFError:
        logger.error("Error: unexpected end of input")
        return 2

    except (
        ModelLoadError,
        PredictionError,
        InvalidInputError,
    ) as exc:
        logger.error("Error: %s", exc)
        return 2

    # noinspection PyBroadException
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error")
        return 3


def main() -> None:
    _configure_logging()

    try:
        sys.exit(run_cli())
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()