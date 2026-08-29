"""
premium_engine.py
Premium calculation engine.

Input:
    probability
    sum_insured
    theta loading
    discount_rate (optional)
    calibration_horizon_days (optional)
    policy_horizon_days (optional)
    probability_horizon_days (optional)
    expected_severity (optional)
    deductible (optional)
    coverage_limit (optional)

Output:
    net premium (discounted)
    gross premium (discounted)
    tariff %
    loading amount
    severity-based diagnostics

No CLI logic.
No model logic.

Assumptions:
- `probability` is expected to correspond to the target policy horizon.
  If `probability_horizon_days` is provided and differs from
  `policy_horizon_days`, a warning is emitted and no probability
  rescaling is performed.

- Discounting applies continuous compounding to the whole expected loss
  using the selected discount horizon.

- If `expected_severity` is provided, net premium is calculated as:
      probability * covered_expected_severity
  where covered_expected_severity is:
      max(0, expected_severity - deductible)
      capped by coverage_limit if provided.

- If `expected_severity` is not provided, legacy behaviour is used:
      probability * sum_insured

- This engine intentionally does not perform survival-model calculations,
  probability scaling, LGD modelling, multiple-event modelling, or
  competing-risk modelling.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


try:
    from .exceptions import InvalidInputError, PremiumCalculationError
except ImportError:
    try:
        from exceptions import InvalidInputError, PremiumCalculationError
    except ImportError:
        class InvalidInputError(ValueError):
            pass

        class PremiumCalculationError(RuntimeError):
            pass


_DAYS_IN_YEAR = 365.0


# ==================================================
# Helpers
# ==================================================


def _as_finite_float(value: Any, name: str) -> float:
    """
    Convert value to float and ensure it is finite.
    """
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidInputError(
            f"{name} must be a number, got {value!r}"
        ) from exc

    if not math.isfinite(value):
        raise InvalidInputError(
            f"{name} must be finite, got {value!r}"
        )

    return value


def _validate_positive_finite(value: Any, name: str) -> float:
    """
    Ensure value is finite and strictly positive.
    """
    value = _as_finite_float(value, name)

    if value <= 0.0:
        raise InvalidInputError(
            f"{name} must be positive, got {value}"
        )

    return value


def _validate_probability(value: Any) -> float:
    """
    Validate probability and return it as float.
    """
    value = _as_finite_float(value, "probability")

    if not (0.0 <= value <= 1.0):
        raise InvalidInputError(
            f"Probability must be in [0, 1], got {value}"
        )

    # Normalize signed zero and exact boundary values.
    if value == 0.0:
        return 0.0

    if value == 1.0:
        return 1.0

    return value


def _validate_sum_insured(value: Any) -> float:
    """
    Validate sum insured and return it as float.
    """
    return _validate_positive_finite(value, "sum_insured")


def _validate_theta(value: Any) -> float:
    """
    Validate theta loading and return it as float.
    """
    value = _as_finite_float(value, "theta")

    if value < 0.0:
        raise InvalidInputError(
            f"Theta cannot be negative, got {value}"
        )

    return value


def _validate_discount_rate(value: Any) -> float:
    """
    Validate annual discount rate and return it as float.

    Supported range: [0, 1), i.e. 0% inclusive, 100% exclusive.
    """
    value = _as_finite_float(value, "discount_rate")

    if value < 0.0:
        raise InvalidInputError(
            f"Discount rate cannot be negative, got {value}"
        )

    if value >= 1.0:
        raise InvalidInputError(
            f"Discount rate {value:.4f} must be less than 100%"
        )

    return value


def _validate_optional_horizon(value: Any, name: str) -> Optional[float]:
    """
    Validate optional horizon in days.

    If None, returns None.
    Otherwise value must be finite and positive.
    """
    if value is None:
        return None

    return _validate_positive_finite(value, name)


def _scale_probability_to_horizon(
    probability: float,
    calib_horizon_days: float,
    target_horizon_days: float,
) -> float:
    """
    REMOVED: Эта функция предполагала постоянный hazard (экспоненциальное),
    что противоречит Cox-модели с переменной базовой опасностью.
    """
    raise NotImplementedError(
        "_scale_probability_to_horizon удалена. "
        "Для Cox-модели вероятность на произвольном горизонте "
        "вычисляется через predict_probability(time_horizon=t). "
        "Не используйте экспоненциальное масштабирование."
    )


def _covered_loss_lognormal(
    mu: float,
    sigma: float,
    deductible: float = 0.0,
    limit: Optional[float] = None,
) -> float:
    """
    PATCH-02: E[max(0, min(X, limit) - d)] for X ~ Lognormal(mu, sigma).

    Uses analytical formula for LogNormal distribution:
    E[(X - d)+] = E[X·1{X>d}] - d·P(X>d)
                = exp(μ + σ²/2) · Φ(σ - z_d) - d · Φ(-z_d)
    where z_d = (ln(d) - μ) / σ

    For limited loss with limit L:
    E[min(X, L) - d]+ = E[(X - d)+] - E[(X - L)+]
    """
    if sigma <= 0:
        # Degenerate case: treat as point mass at exp(mu)
        expected = math.exp(mu)
        covered = max(0.0, expected - deductible)
        if limit is not None and limit > 0:
            covered = min(covered, max(0.0, limit - deductible))
        return covered

    try:
        from scipy.stats import norm
    except ImportError:
        # Fallback without scipy: use simple approximation
        expected = math.exp(mu + 0.5 * sigma**2)
        covered = max(0.0, expected - deductible)
        if limit is not None and limit > 0:
            covered = min(covered, max(0.0, limit - deductible))
        return covered

    mean_X = math.exp(mu + 0.5 * sigma**2)

    # E[(X - d)+] for deductible d > 0
    if deductible > 0:
        z_d = (math.log(deductible) - mu) / sigma
        # E[X·1{X>d}] = exp(μ + σ²/2) · Φ(σ - z_d)
        partial = mean_X * norm.sf(z_d - sigma)
        # d·P(X>d) = d · Φ(-z_d) = d · sf(z_d)
        covered = partial - deductible * norm.sf(z_d)
    else:
        covered = mean_X

    # Apply policy limit: subtract excess above limit
    if limit is not None and limit > 0 and limit > deductible:
        z_l = (math.log(limit) - mu) / sigma
        partial_l = mean_X * norm.sf(z_l - sigma)
        excess_l = partial_l - limit * norm.sf(z_l)
        covered -= max(0.0, excess_l)

    return max(0.0, covered)


def _apply_discount(
    amount: float,
    annual_rate: float,
    horizon_days: float,
) -> float:
    """
    Apply continuous compounding discount to a future cash flow.

    PV = FV * exp(-r * t / 365)

    Parameters
    ----------
    amount : float
        Future value.
    annual_rate : float
        Annual discount rate, e.g. 0.10 for 10%.
    horizon_days : float
        Time horizon in days.

    Returns
    -------
    float
        Present value.
    """
    if not math.isfinite(amount):
        raise PremiumCalculationError(
            "Amount to discount must be finite"
        )

    if not math.isfinite(annual_rate):
        raise InvalidInputError(
            f"Annual discount rate must be finite, got {annual_rate}"
        )

    if not math.isfinite(horizon_days):
        raise InvalidInputError(
            f"Horizon days must be finite, got {horizon_days}"
        )

    if annual_rate < 0.0:
        raise InvalidInputError(
            f"Annual discount rate cannot be negative, got {annual_rate}"
        )

    if annual_rate >= 1.0:
        raise InvalidInputError(
            f"Annual discount rate {annual_rate:.4f} must be less than 100%"
        )

    if horizon_days < 0.0:
        raise InvalidInputError(
            f"Horizon days cannot be negative, got {horizon_days}"
        )

    if annual_rate == 0.0 or horizon_days == 0.0:
        return float(amount)

    t = horizon_days / _DAYS_IN_YEAR
    discounted = amount * math.exp(-annual_rate * t)

    if not math.isfinite(discounted):
        raise PremiumCalculationError(
            "Discounting produced a non-finite value"
        )

    return float(discounted)


def _prepare_common_inputs(
    sum_insured: Any,
    theta: Any,
    discount_rate: Any,
    calibration_horizon_days: Any,
    policy_horizon_days: Any,
    probability_horizon_days: Any = None,
):
    """
    Validate and resolve common premium calculation inputs.

    Returns:
        sum_insured
        theta
        discount_rate
        discount_horizon
    """
    sum_insured = _validate_sum_insured(sum_insured)
    theta = _validate_theta(theta)
    discount_rate = _validate_discount_rate(discount_rate)

    calibration_horizon_days = _validate_optional_horizon(
        calibration_horizon_days,
        "calibration_horizon_days",
    )
    policy_horizon_days = _validate_optional_horizon(
        policy_horizon_days,
        "policy_horizon_days",
    )
    probability_horizon_days = _validate_optional_horizon(
        probability_horizon_days,
        "probability_horizon_days",
    )

    # Explicit probability horizon mismatch warning.
    if (
        probability_horizon_days is not None
        and policy_horizon_days is not None
        and not math.isclose(
            probability_horizon_days,
            policy_horizon_days,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    ):
        warnings.warn(
            "probability_horizon_days differs from policy_horizon_days. "
            "The probability will be used as provided and will NOT be rescaled.",
            UserWarning,
            stacklevel=3,
        )

    # Legacy heuristic mismatch warning.
    if (
        probability_horizon_days is None
        and calibration_horizon_days is not None
        and policy_horizon_days is not None
        and not math.isclose(
            calibration_horizon_days,
            policy_horizon_days,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    ):
        warnings.warn(
            "calibration_horizon_days differs from policy_horizon_days. "
            "The probability will be used as provided and will NOT be rescaled.",
            UserWarning,
            stacklevel=3,
        )

    # Resolve discount horizon.
    #
    # Priority:
    #   1. policy_horizon_days
    #   2. probability_horizon_days
    #   3. calibration_horizon_days
    #
    # The last fallback preserves legacy behaviour.
    if policy_horizon_days is not None:
        discount_horizon = policy_horizon_days
    elif probability_horizon_days is not None:
        discount_horizon = probability_horizon_days
    else:
        discount_horizon = calibration_horizon_days

    if discount_rate > 0.0 and discount_horizon is None:
        warnings.warn(
            "discount_rate > 0 but no horizon was provided; "
            "discounting was not applied.",
            UserWarning,
            stacklevel=3,
        )

    return (
        sum_insured,
        theta,
        discount_rate,
        discount_horizon,
    )


def _calculate_single_premium_validated(
    probability: float,
    sum_insured: float,
    theta: float,
    discount_rate: float,
    discount_horizon: Optional[float],
    expected_severity: Optional[float] = None,
    deductible: float = 0.0,
    coverage_limit: Optional[float] = None,
    severity_already_covered: bool = False,
    severity_lognormal_mu: float = 11.5,
    severity_lognormal_sigma: float = 0.6,
) -> Dict[str, float]:
    """
    Internal calculation function.

    All arguments are expected to be already validated.

    Parameters for Jensen's inequality correction (PATCH-02):
        severity_lognormal_mu: μ parameter of LogNormal severity distribution
        severity_lognormal_sigma: σ parameter of LogNormal severity distribution
        severity_already_covered: если True, expected_severity уже содержит
            E[covered loss] (франшиза и лимит учтены). Повторное применение
            НЕ выполняется. Если False, применяется формула через
            _covered_loss_lognormal.
    """
    # ================================================================
    # Step 1: expected covered loss.
    #
    # Если expected_severity задан — считаем P * E[covered loss],
    # иначе legacy: P * sum_insured.
    # ================================================================
    if expected_severity is not None:
        if severity_already_covered:
            # PATCH-02: франшиза и лимит уже учтены в expected_severity.
            # НЕ применяем повторно.
            covered = max(0.0, expected_severity)
        else:
            # Legacy: используем точный расчёт через логнормальную формулу
            covered = _covered_loss_lognormal(
                mu=severity_lognormal_mu,
                sigma=severity_lognormal_sigma,
                deductible=deductible,
                limit=coverage_limit,
            )
        net_undiscounted = probability * covered
    else:
        net_undiscounted = probability * sum_insured

    if not math.isfinite(net_undiscounted):
        raise PremiumCalculationError(
            "Net undiscounted premium is not finite"
        )

    # ================================================================
    # Step 2: Discounting
    # ================================================================
    discount_factor = 1.0
    net_discounted = net_undiscounted

    if discount_rate > 0.0 and discount_horizon is not None:
        # Compute discount factor independently so that it remains
        # meaningful even when net_undiscounted == 0.
        discount_factor = _apply_discount(
            1.0,
            discount_rate,
            discount_horizon,
        )

        if not math.isfinite(discount_factor):
            raise PremiumCalculationError(
                "Discount factor is not finite"
            )

        net_discounted = net_undiscounted * discount_factor

        if not math.isfinite(net_discounted):
            raise PremiumCalculationError(
                "Net discounted premium is not finite"
            )

    # ================================================================
    # Step 3: Gross premium (net + loading)
    #
    # Loading is applied to the discounted net premium:
    #     gross_discounted = net_discounted * (1 + theta)
    # ================================================================
    gross_undiscounted = net_undiscounted * (1.0 + theta)
    gross_discounted = net_discounted * (1.0 + theta)

    if not math.isfinite(gross_undiscounted):
        raise PremiumCalculationError(
            "Gross undiscounted premium is not finite"
        )

    if not math.isfinite(gross_discounted):
        raise PremiumCalculationError(
            "Gross discounted premium is not finite"
        )

    # ================================================================
    # Step 4: Tariff and diagnostics
    # ================================================================
    tariff = gross_discounted / sum_insured * 100.0
    loading_amount = gross_discounted - net_discounted

    result = {
        "net_undiscounted": float(net_undiscounted),
        "net_discounted": float(net_discounted),
        "gross_undiscounted": float(gross_undiscounted),
        "gross_discounted": float(gross_discounted),
        "net": float(net_discounted),
        "gross": float(gross_discounted),
        "tariff": float(tariff),
        "discount_factor": float(discount_factor),
        "loading_amount": float(loading_amount),

        # P-12: severity-based net premium diagnostics
        "severity_based": bool(expected_severity is not None),
        "deductible": float(deductible),
        "coverage_limit": (
            float(coverage_limit)
            if coverage_limit is not None
            else None
        ),
    }

    for field_name, field_value in result.items():
        # Optional diagnostic field may be None.
        if field_name == "coverage_limit" and field_value is None:
            continue

        # Boolean diagnostic field is not a monetary amount.
        if field_name == "severity_based":
            continue

        if not math.isfinite(field_value):
            raise PremiumCalculationError(
                "Premium calculation produced non-finite value "
                f"for '{field_name}'"
            )

    return result


# ==================================================
# Single premium
# ==================================================


def calculate_single_premium(
    probability: float,
    sum_insured: float,
    theta: float = 0.15,
    discount_rate: float = 0.0,
    calibration_horizon_days: Optional[float] = None,
    policy_horizon_days: Optional[float] = None,
    probability_horizon_days: Optional[float] = None,
    expected_severity: Optional[float] = None,
    deductible: float = 0.0,
    coverage_limit: Optional[float] = None,
    severity_already_covered: bool = False,
) -> Dict[str, float]:
    """
    Calculate premium for one probability.

    Designed for use with Cox survival model:
        - The Cox model already computes P(T <= t | x) for the target horizon.
        - No additional exponential probability scaling is applied.
        - Discounting uses the policy coverage horizon when provided.
        - Loading theta is applied to the discounted net premium.

    Formula chain legacy:
        net_undiscounted   = probability * sum_insured
        net_discounted     = net_undiscounted * exp(-r * t / 365)
        gross_undiscounted = net_undiscounted * (1 + theta)
        gross_discounted   = net_discounted * (1 + theta)
        tariff             = gross_discounted / sum_insured * 100

    Formula chain severity-based:
        covered_loss       = max(0, expected_severity - deductible)
        covered_loss       = min(covered_loss, coverage_limit) if coverage_limit
        net_undiscounted   = probability * covered_loss
        net_discounted     = net_undiscounted * exp(-r * t / 365)
        gross_undiscounted = net_undiscounted * (1 + theta)
        gross_discounted   = net_discounted * (1 + theta)
        tariff             = gross_discounted / sum_insured * 100

    Parameters
    ----------
    severity_already_covered : bool
        Если True, expected_severity уже содержит E[covered loss]
        (франшиза и лимит учтены). Повторное применение НЕ выполняется.
    """
    try:
        (
            sum_insured,
            theta,
            discount_rate,
            discount_horizon,
        ) = _prepare_common_inputs(
            sum_insured=sum_insured,
            theta=theta,
            discount_rate=discount_rate,
            calibration_horizon_days=calibration_horizon_days,
            policy_horizon_days=policy_horizon_days,
            probability_horizon_days=probability_horizon_days,
        )

        probability = _validate_probability(probability)

        # P-12: severity-based net premium
        if expected_severity is not None:
            expected_severity = _validate_positive_finite(
                expected_severity,
                "expected_severity",
            )

        deductible = max(0.0, _as_finite_float(deductible, "deductible"))

        if coverage_limit is not None:
            coverage_limit = _validate_positive_finite(
                coverage_limit,
                "coverage_limit",
            )

        return _calculate_single_premium_validated(
            probability=probability,
            sum_insured=sum_insured,
            theta=theta,
            discount_rate=discount_rate,
            discount_horizon=discount_horizon,
            expected_severity=expected_severity,
            deductible=deductible,
            coverage_limit=coverage_limit,
            severity_already_covered=severity_already_covered,
            severity_lognormal_mu=11.5,
            severity_lognormal_sigma=0.6,
        )

    except (InvalidInputError, PremiumCalculationError):
        raise
    except Exception as exc:
        raise PremiumCalculationError("Premium calculation failed") from exc


# ==================================================
# Batch premium
# ==================================================


def calculate_premium(
    probabilities: Sequence[float],
    sum_insured: float,
    theta: float = 0.15,
    discount_rate: float = 0.0,
    calibration_horizon_days: Optional[float] = None,
    policy_horizon_days: Optional[float] = None,
    probability_horizon_days: Optional[float] = None,
    expected_severity: Optional[float] = None,
    deductible: float = 0.0,
    coverage_limit: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    Calculate premiums for multiple probabilities.
    """
    if probabilities is None:
        raise InvalidInputError("probabilities must not be None")

    if isinstance(probabilities, (str, bytes, bytearray, Mapping)):
        raise InvalidInputError(
            "probabilities must be a sequence or iterable of numbers"
        )

    try:
        (
            sum_insured,
            theta,
            discount_rate,
            discount_horizon,
        ) = _prepare_common_inputs(
            sum_insured=sum_insured,
            theta=theta,
            discount_rate=discount_rate,
            calibration_horizon_days=calibration_horizon_days,
            policy_horizon_days=policy_horizon_days,
            probability_horizon_days=probability_horizon_days,
        )

        # P-12: severity-based net premium
        if expected_severity is not None:
            expected_severity = _validate_positive_finite(
                expected_severity,
                "expected_severity",
            )

        deductible = max(0.0, _as_finite_float(deductible, "deductible"))

        if coverage_limit is not None:
            coverage_limit = _validate_positive_finite(
                coverage_limit,
                "coverage_limit",
            )

    except (InvalidInputError, PremiumCalculationError):
        raise
    except Exception as exc:
        raise PremiumCalculationError("Premium calculation failed") from exc

    try:
        probabilities = list(probabilities)
    except TypeError as exc:
        raise InvalidInputError(
            "probabilities must be an iterable collection"
        ) from exc
    except Exception as exc:
        raise InvalidInputError(
            "probabilities could not be converted to a list"
        ) from exc

    if len(probabilities) == 0:
        raise InvalidInputError("Probability list is empty")

    result: List[Dict[str, float]] = []

    for idx, probability in enumerate(probabilities):
        try:
            validated_probability = _validate_probability(probability)

            result.append(
                _calculate_single_premium_validated(
                    probability=validated_probability,
                    sum_insured=sum_insured,
                    theta=theta,
                    discount_rate=discount_rate,
                    discount_horizon=discount_horizon,
                    expected_severity=expected_severity,
                    deductible=deductible,
                    coverage_limit=coverage_limit,
                    severity_lognormal_mu=11.5,
                    severity_lognormal_sigma=0.6,
                )
            )

        except InvalidInputError as exc:
            raise InvalidInputError(
                f"probabilities[{idx}]: {exc}"
            ) from exc
        except PremiumCalculationError as exc:
            raise PremiumCalculationError(
                f"probabilities[{idx}]: {exc}"
            ) from exc

    return result


# ==================================================
# Severity integration
# ==================================================


def calculate_premium_with_severity(
    severity_model: Any,
    probability: float,
    sum_insured: float,
    theta: float = 0.15,
    deductible: float = 0.0,
    coverage_limit: Optional[float] = None,
    discount_rate: float = 0.0,
    policy_horizon_days: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate premium using a severity model instance.

    This is a convenience wrapper that:
      1. Calls severity_model.expected_covered_loss(deductible, coverage_limit)
      2. Passes the result to calculate_single_premium()

    Parameters
    ----------
    severity_model : object
        Must have method ``expected_covered_loss(deductible, coverage_limit)``
        that returns a finite float.
    probability : float
        Event probability in [0, 1].
    sum_insured : float
        Total sum insured.
    theta : float
        Loading factor (default 0.15).
    deductible : float
        Deductible amount (default 0).
    coverage_limit : float, optional
        Coverage limit.
    discount_rate : float
        Annual discount rate (default 0).
    policy_horizon_days : float, optional
        Policy horizon in days.

    Returns
    -------
    dict
        Premium calculation result from ``calculate_single_premium``.

    Example
    -------

    """
    if not hasattr(severity_model, "expected_covered_loss"):
        raise InvalidInputError(
            "severity_model must have method expected_covered_loss"
        )

    try:
        expected_severity = severity_model.expected_covered_loss(
            deductible=deductible,
            coverage_limit=coverage_limit,
        )
    except Exception as exc:
        raise PremiumCalculationError(
            "Failed to compute expected_covered_loss from severity_model"
        ) from exc

    if not math.isfinite(expected_severity):
        raise PremiumCalculationError(
            f"expected_covered_loss is not finite: {expected_severity}"
        )

    # PATCH-02: expected_severity уже содержит E[covered loss] из severity_model.
    # Передаём severity_already_covered=True, чтобы избежать двойного
    # применения франшизы в calculate_single_premium.
    return calculate_single_premium(
        probability=probability,
        sum_insured=sum_insured,
        theta=theta,
        expected_severity=float(expected_severity),
        deductible=0.0,                      # ← НЕ применяем повторно
        coverage_limit=None,                 # ← НЕ применяем повторно
        discount_rate=discount_rate,
        policy_horizon_days=policy_horizon_days,
        severity_already_covered=True,       # ← НОВЫЙ ФЛАГ
    )


# ==================================================
# FIX 7: Variance-based risk margin premium
# ==================================================

def calculate_premium_with_severity_variance(
    probability: float,
    sum_insured: float,
    expected_severity: float,
    severity_variance: float,
    confidence_level: float = 0.95,
    discount_rate: float = 0.0,
    policy_horizon_days: Optional[float] = None,
) -> Dict[str, float]:
    """Calculate premium using variance-based risk margin.

    This replaces the flat theta loading with a principled risk margin
    that accounts for severity dispersion.

    Formula:
        gross_premium = E[Loss] + RiskMargin
        E[Loss] = p * E[Severity]
        RiskMargin = z_alpha * sqrt(p * Var[Severity] + p(1-p) * E[Severity]^2)

    Parameters
    ----------
    probability : float
        Event probability in [0, 1].
    sum_insured : float
        Total sum insured.
    expected_severity : float
        E[Severity] from Gamma distribution or empirical model.
    severity_variance : float
        Var[Severity] from Gamma distribution or empirical model.
    confidence_level : float
        Confidence level for risk margin (default 0.95).
    discount_rate : float
        Annual discount rate (default 0).
    policy_horizon_days : float, optional
        Policy horizon in days.

    Returns
    -------
    dict
        Premium calculation result with fields:
        - net_premium: E[Loss] discounted
        - risk_margin: z_alpha * sqrt(Var[Loss])
        - gross_premium: net_premium + risk_margin
        - tariff: gross_premium / sum_insured * 100
    """
    try:
        from scipy import stats as _scipy_stats
        z = _scipy_stats.norm.ppf(confidence_level)
    except ImportError:
        z = 1.645  # z_0.95

    # Expected loss
    expected_loss = probability * expected_severity

    # Variance of loss: Var[L] = p * Var[S] + p(1-p) * E[S]^2
    loss_variance = (
        probability * severity_variance
        + probability * (1.0 - probability) * expected_severity ** 2
    )
    loss_variance = max(0.0, loss_variance)

    # Risk margin
    risk_margin = z * np.sqrt(loss_variance)

    # Discounting
    if policy_horizon_days is None:
        from constants import CALIBRATION_HORIZON_DAYS
        policy_horizon_days = CALIBRATION_HORIZON_DAYS

    t = policy_horizon_days / 365.0
    discount_factor = math.exp(-discount_rate * t)

    net_premium = expected_loss * discount_factor
    risk_margin_discounted = risk_margin * discount_factor
    gross_premium = net_premium + risk_margin_discounted
    tariff = gross_premium / sum_insured * 100.0 if sum_insured > 0 else 0.0

    return {
        "net_premium": float(net_premium),
        "risk_margin": float(risk_margin_discounted),
        "gross_premium": float(gross_premium),
        "tariff": float(tariff),
        "expected_loss": float(expected_loss),
        "loss_variance": float(loss_variance),
        "confidence_level": confidence_level,
    }


# ==================================================
# Extended helper
# ==================================================


def calculate_from_prediction_result(
    prediction_result: Dict[str, Any],
    sum_insured: float,
    theta: float = 0.15,
    discount_rate: float = 0.0,
    calibration_horizon_days: Optional[float] = None,
    policy_horizon_days: Optional[float] = None,
    probability_horizon_days: Optional[float] = None,
    expected_severity: Optional[float] = None,
    deductible: float = 0.0,
    coverage_limit: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    Direct adapter from service prediction output.
    """
    if not isinstance(prediction_result, Mapping):
        raise InvalidInputError(
            "prediction_result must be a dictionary-like object"
        )

    if "probabilities" not in prediction_result:
        raise InvalidInputError("Prediction result has no probabilities")

    if (
        probability_horizon_days is None
        and "probability_horizon_days" in prediction_result
    ):
        probability_horizon_days = prediction_result["probability_horizon_days"]

    return calculate_premium(
        probabilities=prediction_result["probabilities"],
        sum_insured=sum_insured,
        theta=theta,
        discount_rate=discount_rate,
        calibration_horizon_days=calibration_horizon_days,
        policy_horizon_days=policy_horizon_days,
        probability_horizon_days=probability_horizon_days,
        expected_severity=expected_severity,
        deductible=deductible,
        coverage_limit=coverage_limit,
    )