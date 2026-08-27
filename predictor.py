"""
predictor.py

High-level prediction interface for CF Cox / IV-Cox models.

Responsibilities:
- validate inference inputs;
- call prediction engine;
- validate model outputs;
- expose model statistics.

Aligned with:
- prediction_engine.py v3.1+;
- exceptions.py;
- Real_calculator.py;
- train_model.py.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from os import PathLike
from types import MappingProxyType
from typing import Any, Optional, Protocol, Union

import numpy as np

from exceptions import (
    InvalidInputError,
    ModelValidationError,
    PredictionError,
    ProbabilityError,
)
from prediction_engine import (
    load_model_params,
    validate_model,
    predict_many as engine_predict_many,
    predict_probability as engine_predict_probability,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------

PEAK_RANGE_TOLERANCE = 5.0
PEAK_RELATIVE_TOLERANCE = 1e-6
PROBABILITY_EPSILON = 1e-12
TIME_EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Batch limits
# ---------------------------------------------------------------------------

MAX_BATCH_SIZE = 10_000


# ---------------------------------------------------------------------------
# Cox model safety limits
# ---------------------------------------------------------------------------

MAX_COX_COEFFICIENT = 100.0
MAX_EXP_INPUT = float(np.log(np.finfo(float).max))


# ---------------------------------------------------------------------------
# Type guards
# ---------------------------------------------------------------------------

_BINARY_TYPES = (bytes, bytearray, memoryview)
_TEXT_OR_BINARY_TYPES = (str, bytes, bytearray, memoryview)


# ---------------------------------------------------------------------------
# Residual policy
# ---------------------------------------------------------------------------


class ResidualPolicy(Enum):
    """
    Enum for runtime validation consistency with prediction.py.

    Production default: PLUG_IN.

    BOOTSTRAP is not allowed in prediction_engine.
    MEAN and ZERO are diagnostic-only and require explicit opt-in.

    NOTE:
    In some engine versions policy "zero" internally behaved as "mean".
    This predictor explicitly treats MEAN/ZERO as diagnostic-only.
    """

    PLUG_IN = "plug-in"
    BOOTSTRAP = "bootstrap"
    MEAN = "mean"
    ZERO = "zero"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class CoxProtocol(Protocol):
    """Expected Cox model interface."""

    coefs: dict[str, float]


class ModelParamsProtocol(Protocol):
    """Expected model parameters interface."""

    training_meta: dict[str, Any]
    cox: CoxProtocol
    calibration_time_horizon: float


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class Predictor:
    """
    High-level prediction interface.

    Responsibilities:
    - validate inference inputs;
    - call prediction engine;
    - validate model outputs;
    - expose model statistics.
    """

    VALID_RESIDUAL_POLICIES = frozenset(ResidualPolicy)
    VALID_RESIDUAL_POLICY_STRINGS = frozenset(
        policy.value for policy in ResidualPolicy
    )

    _PRODUCTION_RESIDUAL_POLICY_STRINGS = frozenset(
        {ResidualPolicy.PLUG_IN.value}
    )
    _DIAGNOSTIC_RESIDUAL_POLICY_STRINGS = frozenset(
        {
            ResidualPolicy.MEAN.value,
            ResidualPolicy.ZERO.value,
        }
    )

    def __init__(
        self,
        params: ModelParamsProtocol,
        *,
        allow_diagnostic_residual_policies: bool = False,
        allow_horizon_extrapolation: bool = False,
        strict_covariates: bool = False,
        allow_unknown_covariates: bool = False,
    ) -> None:
        """
        Initialize Predictor.

        Args:
            params:
                Model parameters.

            allow_diagnostic_residual_policies:
                If False, MEAN/ZERO residual policies are rejected.

            allow_horizon_extrapolation:
                If False, time_horizon greater than calibration horizon
                raises InvalidInputError. If True, a warning is emitted.

            strict_covariates:
                If True, missing required covariates raise InvalidInputError.
                If False, missing required covariates emit a warning.

            allow_unknown_covariates:
                If False and required covariate metadata exists,
                unknown covariates raise InvalidInputError.
        """

        if params is None:
            raise ModelValidationError("Model parameters required")

        try:
            validation_result = validate_model(params)
        except ModelValidationError:
            raise
        except Exception as exc:
            raise ModelValidationError("Invalid model parameters") from exc

        # Strict contract: validate_model() must return a real truthy bool.
        # This protects against None, 0, np.bool_(False), custom falsey objects.
        if (
            not isinstance(validation_result, (bool, np.bool_))
            or not bool(validation_result)
        ):
            raise ModelValidationError("Invalid model parameters")

        try:
            (
                training_meta,
                _cox,
                coefs,
                calibration_time_horizon,
            ) = self._validate_model_structure(params)

            coefficients = self._validate_coefficients(coefs)
            peak_range = self._extract_peak_range(training_meta)
            time_unit = self._extract_time_unit(training_meta)
            required_covariates = self._extract_required_covariates(
                training_meta
            )
        except ModelValidationError:
            raise
        except Exception as exc:
            raise ModelValidationError("Invalid model parameters") from exc

        self._allow_diagnostic_residual_policies = bool(
            allow_diagnostic_residual_policies
        )
        self._allow_horizon_extrapolation = bool(
            allow_horizon_extrapolation
        )
        self._strict_covariates = bool(strict_covariates)
        self._allow_unknown_covariates = bool(allow_unknown_covariates)

        self._params = params
        self._calibration_time_horizon = calibration_time_horizon
        self._time_unit = time_unit
        self._peak_range = peak_range
        self._coefficients = MappingProxyType(coefficients)
        self._canonical_coefficient_names = frozenset(
            name.strip().lower() for name in coefficients
        )
        self._required_covariates = required_covariates

        self._warned_residual_policies: set[str] = set()
        self._warned_horizon_extrapolation = False
        self._cached_ratios: Optional[MappingProxyType] = None

        if self._peak_range is None:
            warnings.warn(
                "PeakLoad training range is not available; "
                "PeakLoad OOD validation is disabled.",
                UserWarning,
                stacklevel=2,
            )

        if self._time_unit == "unknown":
            warnings.warn(
                "Model time unit is unknown; "
                "time_horizon units cannot be verified.",
                UserWarning,
                stacklevel=2,
            )

        if self._required_covariates is None and coefficients:
            warnings.warn(
                "Required covariate metadata is missing; "
                "covariate completeness cannot be fully enforced.",
                UserWarning,
                stacklevel=2,
            )

        if self._calibration_time_horizon == 0.0:
            warnings.warn(
                "Calibration time horizon is zero.",
                UserWarning,
                stacklevel=2,
            )

    # -----------------------------------------------------------------
    # Read-only compatibility properties
    # -----------------------------------------------------------------

    @property
    def params(self) -> ModelParamsProtocol:
        """
        Read-only access to model parameters.

        The predictor intentionally does not allow reassigning params
        after initialization.
        """

        return self._params

    @property
    def calibration_time_horizon(self) -> float:
        return self._calibration_time_horizon

    @property
    def time_unit(self) -> str:
        return self._time_unit

    @property
    def coefficients(self) -> dict[str, float]:
        """
        Return a copy of validated Cox coefficients.
        """

        return dict(self._coefficients)

    # -----------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _get_field(obj: Any, name: str) -> Any:
        """
        Get field from either a Mapping or an object.
        """

        if isinstance(obj, Mapping):
            return obj.get(name)
        return getattr(obj, name, None)

    @staticmethod
    def _coerce_finite_float(
        value: Any,
        name: str,
        error_cls: type[Exception],
        *,
        allow_strings: bool,
    ) -> float:
        """
        Convert value to a finite scalar float.

        Raises:
            error_cls:
                InvalidInputError, ModelValidationError, PredictionError,
                or ProbabilityError depending on context.
        """

        if isinstance(value, np.ndarray):
            if value.ndim != 0:
                raise error_cls(f"{name} must be scalar")
            value = value.item()

        if isinstance(value, (bool, np.bool_)):
            raise error_cls(f"{name} cannot be boolean")

        if isinstance(value, _BINARY_TYPES):
            raise error_cls(f"{name} cannot be binary")

        if isinstance(value, str):
            if not allow_strings:
                raise error_cls(f"{name} must be numeric")
            value = value.strip()

        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise error_cls(f"Invalid {name}") from exc

        if not np.isfinite(value):
            raise error_cls(f"{name} must be finite")

        return value

    @staticmethod
    def _validate_calibration_time_horizon(value: Any) -> float:
        value = Predictor._coerce_finite_float(
            value,
            "Calibration time horizon",
            ModelValidationError,
            allow_strings=False,
        )

        if value < 0:
            raise ModelValidationError(
                "Calibration time horizon cannot be negative"
            )

        return value

    @staticmethod
    def _validate_model_structure(
        params: ModelParamsProtocol,
    ) -> tuple[Mapping[str, Any], Any, Mapping[str, Any], float]:
        """
        Validate required model fields.

        Returns:
            training_meta, cox, coefs, calibration_time_horizon.

        Raises:
            ModelValidationError: if required fields are missing or invalid.
        """

        training_meta = Predictor._get_field(params, "training_meta")
        if training_meta is None:
            raise ModelValidationError("Missing training metadata")

        if not isinstance(training_meta, Mapping):
            raise ModelValidationError(
                "Training metadata must be a mapping"
            )

        cox = Predictor._get_field(params, "cox")
        if cox is None:
            raise ModelValidationError("Missing Cox model")

        if isinstance(cox, Mapping):
            coefs = cox.get("coefs")
        else:
            coefs = getattr(cox, "coefs", None)

        if coefs is None:
            raise ModelValidationError("Missing Cox coefficients")

        if not isinstance(coefs, Mapping):
            raise ModelValidationError(
                "Cox coefficients must be a mapping"
            )

        calibration_time_horizon = Predictor._get_field(
            params,
            "calibration_time_horizon",
        )
        if calibration_time_horizon is None:
            raise ModelValidationError(
                "Missing calibration time horizon"
            )

        calibration_time_horizon = (
            Predictor._validate_calibration_time_horizon(
                calibration_time_horizon
            )
        )

        return (
            training_meta,
            cox,
            coefs,
            calibration_time_horizon,
        )

    @staticmethod
    def _validate_coefficients(
        coefs: Mapping[str, Any],
    ) -> dict[str, float]:
        """
        Validate Cox coefficients eagerly during initialization.
        """

        validated: dict[str, float] = {}
        seen_canonical: set[str] = set()

        for name, coefficient in coefs.items():
            if not isinstance(name, str):
                raise ModelValidationError(
                    "Invalid Cox coefficient name"
                )

            cleaned_name = name.strip()
            if not cleaned_name:
                raise ModelValidationError(
                    "Cox coefficient name cannot be empty"
                )

            if cleaned_name != name:
                raise ModelValidationError(
                    "Cox coefficient name has leading/trailing "
                    f"whitespace: {name!r}"
                )

            canonical_name = cleaned_name.lower()
            if canonical_name in seen_canonical:
                raise ModelValidationError(
                    "Duplicate Cox coefficient after normalization: "
                    f"{canonical_name!r}"
                )

            seen_canonical.add(canonical_name)

            coefficient_value = Predictor._coerce_finite_float(
                coefficient,
                f"Cox coefficient '{cleaned_name}'",
                ModelValidationError,
                allow_strings=False,
            )

            if coefficient_value > MAX_EXP_INPUT:
                raise ModelValidationError(
                    f"Hazard ratio overflow: {cleaned_name}"
                )

            if abs(coefficient_value) > MAX_COX_COEFFICIENT:
                raise ModelValidationError(
                    f"Cox coefficient too large: {cleaned_name}"
                )

            validated[cleaned_name] = coefficient_value

        return validated

    @staticmethod
    def _extract_peak_range(
        training_meta: Mapping[str, Any],
    ) -> Optional[tuple[float, float]]:
        """
        Extract and validate PeakLoad training range.
        """

        pmin = training_meta.get("peakload_min")
        pmax = training_meta.get("peakload_max")

        if pmin is None and pmax is None:
            return None

        if pmin is None or pmax is None:
            raise ModelValidationError(
                "Incomplete peakload training range"
            )

        pmin = Predictor._coerce_finite_float(
            pmin,
            "peakload_min",
            ModelValidationError,
            allow_strings=False,
        )
        pmax = Predictor._coerce_finite_float(
            pmax,
            "peakload_max",
            ModelValidationError,
            allow_strings=False,
        )

        if pmin > pmax:
            raise ModelValidationError(
                "Peakload minimum exceeds maximum"
            )

        return (pmin, pmax)

    @staticmethod
    def _extract_time_unit(training_meta: Mapping[str, Any]) -> str:
        """
        Extract time unit from metadata.

        Supported metadata keys:
        - time_unit
        - time_units
        - calibration_time_unit
        """

        for key in (
            "time_unit",
            "time_units",
            "calibration_time_unit",
        ):
            raw = training_meta.get(key)
            if raw is None:
                continue

            if not isinstance(raw, str) or not raw.strip():
                raise ModelValidationError(
                    f"Invalid time unit in training metadata: {key}"
                )

            return raw.strip().lower()

        return "unknown"

    @staticmethod
    def _extract_required_covariates(
        training_meta: Mapping[str, Any],
    ) -> Optional[frozenset[str]]:
        """
        Extract required covariate names from metadata.

        Supported metadata keys:
        - required_covariates
        - covariate_names
        - feature_names

        Returns:
            frozenset of canonical lower-cased required covariate names,
            empty frozenset if metadata explicitly says no covariates
            are required, or None if metadata is absent.
        """

        for key in (
            "required_covariates",
            "covariate_names",
            "feature_names",
        ):
            raw = training_meta.get(key)
            if raw is None:
                continue

            if isinstance(raw, _TEXT_OR_BINARY_TYPES):
                raise ModelValidationError(
                    f"Invalid required covariate metadata: {key}"
                )

            if isinstance(raw, np.ndarray):
                if raw.ndim != 1:
                    raise ModelValidationError(
                        f"Invalid required covariate metadata: {key}"
                    )
                items = raw.tolist()
            elif isinstance(raw, Sequence):
                items = raw
            else:
                raise ModelValidationError(
                    f"Invalid required covariate metadata: {key}"
                )

            canonical_names: set[str] = set()

            for item in items:
                if not isinstance(item, str):
                    raise ModelValidationError(
                        f"Invalid required covariate metadata: {key}"
                    )

                cleaned = item.strip()
                if not cleaned:
                    raise ModelValidationError(
                        f"Invalid required covariate metadata: {key}"
                    )

                canonical = cleaned.lower()
                if canonical in canonical_names:
                    raise ModelValidationError(
                        "Duplicate required covariate in metadata: "
                        f"{canonical!r}"
                    )

                canonical_names.add(canonical)

            return frozenset(canonical_names)

        return None

    # -----------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------

    @classmethod
    def from_model_json(
        cls,
        path: Union[str, PathLike[str]],
        *,
        allow_diagnostic_residual_policies: bool = False,
        allow_horizon_extrapolation: bool = False,
        strict_covariates: bool = False,
        allow_unknown_covariates: bool = False,
    ) -> "Predictor":
        """
        Load and validate model parameters.
        """

        try:
            params = load_model_params(path)
        except OSError as exc:
            raise ModelValidationError(
                "Unable to read model file"
            ) from exc
        except (
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            raise ModelValidationError(
                "Malformed model file"
            ) from exc

        return cls(
            params,
            allow_diagnostic_residual_policies=(
                allow_diagnostic_residual_policies
            ),
            allow_horizon_extrapolation=allow_horizon_extrapolation,
            strict_covariates=strict_covariates,
            allow_unknown_covariates=allow_unknown_covariates,
        )

    # -----------------------------------------------------------------
    # Input normalization
    # -----------------------------------------------------------------

    def _normalize_number(
        self,
        value: Any,
        name: str,
    ) -> float:
        """
        Convert user input to finite float.

        Numeric strings are allowed for user-facing inputs.
        """

        return self._coerce_finite_float(
            value,
            name,
            InvalidInputError,
            allow_strings=True,
        )

    def _normalize_with_validator(
        self,
        value: Any,
        name: str,
        validator: Callable[[float], None],
    ) -> float:
        """
        Normalize value and apply domain validation.
        """

        value = self._normalize_number(value, name)
        validator(value)
        return value

    def _normalize_peak(self, peak: Any) -> float:
        return self._normalize_with_validator(
            peak,
            "PeakLoad",
            self._validate_peak,
        )

    def _normalize_time(self, time_horizon: Any) -> float:
        return self._normalize_with_validator(
            time_horizon,
            "time horizon",
            self._validate_time,
        )

    def _normalize_covariates(
        self,
        covariates: Optional[Mapping[str, Any]],
    ) -> Optional[dict[str, float]]:
        """
        Normalize optional covariates mapping.
        """

        if covariates is None:
            self._validate_covariates_for_prediction(None)
            return None

        if isinstance(covariates, _BINARY_TYPES) or not isinstance(
            covariates,
            Mapping,
        ):
            raise InvalidInputError(
                "covariates must be a mapping or None"
            )

        normalized: dict[str, float] = {}
        canonical_seen: dict[str, str] = {}

        for raw_name, value in covariates.items():
            if not isinstance(raw_name, str):
                raise InvalidInputError(
                    "Covariate names must be strings"
                )

            name = raw_name.strip()
            if not name:
                raise InvalidInputError(
                    "Covariate names must be non-empty"
                )

            canonical = name.lower()
            if canonical in canonical_seen:
                raise InvalidInputError(
                    "Duplicate covariate after normalization: "
                    f"{canonical!r}"
                )

            canonical_seen[canonical] = name
            normalized[name] = self._normalize_number(
                value,
                f"covariate '{name}'",
            )

        self._validate_covariates_for_prediction(normalized)
        return normalized

    # -----------------------------------------------------------------
    # Covariate schema validation
    # -----------------------------------------------------------------

    def _validate_covariates_for_prediction(
        self,
        covariates: Optional[dict[str, float]],
    ) -> None:
        """
        Validate covariates against required metadata if available.
        """

        provided_canonical = (
            frozenset(
                name.strip().lower() for name in covariates.keys()
            )
            if covariates
            else frozenset()
        )

        if self._required_covariates is not None:
            missing = self._required_covariates - provided_canonical
            unknown = provided_canonical - self._required_covariates

            if missing:
                message = (
                    "Missing required covariates: "
                    f"{sorted(missing)}"
                )

                if self._strict_covariates:
                    raise InvalidInputError(message)

                warnings.warn(
                    message,
                    UserWarning,
                    stacklevel=4,
                )

            if unknown and not self._allow_unknown_covariates:
                raise InvalidInputError(
                    f"Unknown covariates: {sorted(unknown)}"
                )

            return

        # No explicit required-covariate metadata.
        # We avoid hard failures because coefficient names may include
        # generated features: spline bases, brand dummies, interactions.
        if covariates:
            unknown_against_coefs = (
                provided_canonical
                - self._canonical_coefficient_names
            )
            if unknown_against_coefs:
                logger.debug(
                    "Covariates not found among Cox coefficient names: %s",
                    sorted(unknown_against_coefs),
                )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _normalize_residual_policy(
        self,
        policy: Union[ResidualPolicy, str],
    ) -> str:
        """
        Validate and normalize residual policy to string.

        Accepts both Enum and string values.

        Raises:
            InvalidInputError:
                If policy is unknown, wrong type, bootstrap,
                or diagnostic without opt-in.
        """

        if isinstance(policy, ResidualPolicy):
            policy_str = policy.value
        elif isinstance(policy, str):
            policy_str = policy.strip().lower()
        else:
            raise InvalidInputError(
                "residual_policy must be ResidualPolicy enum or str, "
                f"got {type(policy).__name__}"
            )

        if policy_str == ResidualPolicy.BOOTSTRAP.value:
            raise InvalidInputError(
                "Bootstrap residual policy is not supported"
            )

        if policy_str in self._DIAGNOSTIC_RESIDUAL_POLICY_STRINGS:
            if not self._allow_diagnostic_residual_policies:
                raise InvalidInputError(
                    "Diagnostic residual policy requires explicit opt-in"
                )

            if policy_str not in self._warned_residual_policies:
                warnings.warn(
                    f"Residual policy '{policy_str}' is diagnostic-only. "
                    "Results are not production estimates.",
                    UserWarning,
                    stacklevel=3,
                )
                self._warned_residual_policies.add(policy_str)

            return policy_str

        if policy_str in self._PRODUCTION_RESIDUAL_POLICY_STRINGS:
            return policy_str

        raise InvalidInputError(
            f"Unknown residual policy: {policy}"
        )

    def _validate_peak(self, peak: float) -> None:
        """
        Validate PeakLoad against training range with tolerance.

        Raises:
            InvalidInputError:
                If peak is outside training range plus tolerance.
        """

        if self._peak_range is None:
            return

        pmin, pmax = self._peak_range

        scale = max(1.0, abs(pmin), abs(pmax))
        tolerance = max(
            PEAK_RANGE_TOLERANCE,
            PEAK_RELATIVE_TOLERANCE * scale,
        )

        lower_bound = pmin - tolerance
        upper_bound = pmax + tolerance

        if peak < lower_bound or peak > upper_bound:
            raise InvalidInputError(
                f"PeakLoad {peak} outside training range "
                f"[{pmin}, {pmax}] with tolerance {tolerance}"
            )

    def _warn_peaks_outside_training_range(
        self,
        peaks: Sequence[float],
    ) -> None:
        """
        Emit one warning if any peak lies outside the training range
        but still within tolerance.
        """

        if self._peak_range is None:
            return

        pmin, pmax = self._peak_range

        outside_count = sum(
            1 for peak in peaks if peak < pmin or peak > pmax
        )

        if outside_count:
            warnings.warn(
                f"{outside_count} PeakLoad value(s) are outside "
                f"training range [{pmin}, {pmax}] but within tolerance.",
                UserWarning,
                stacklevel=4,
            )

    def _validate_time(self, time_horizon: float) -> None:
        """
        Validate prediction time horizon.

        Raises:
            InvalidInputError:
                If time horizon is negative or exceeds calibration horizon
                while extrapolation is disabled.
        """

        if time_horizon < 0:
            raise InvalidInputError(
                "Time horizon cannot be negative"
            )

        if (
            self._calibration_time_horizon > 0
            and time_horizon
            > self._calibration_time_horizon + TIME_EPSILON
        ):
            message = (
                f"time_horizon={time_horizon} exceeds calibration "
                f"horizon={self._calibration_time_horizon}. "
                "Baseline hazard may be constant-extrapolated."
            )

            if not self._allow_horizon_extrapolation:
                raise InvalidInputError(message)

            if not self._warned_horizon_extrapolation:
                warnings.warn(
                    message,
                    UserWarning,
                    stacklevel=4,
                )
                self._warned_horizon_extrapolation = True

    def _validate_probability(self, value: Any) -> float:
        """
        Validate probability returned by prediction engine.

        This is intentionally strict:
        - booleans are rejected;
        - strings are rejected;
        - binary values are rejected;
        - non-scalar arrays are rejected.
        """

        value = self._coerce_finite_float(
            value,
            "Probability",
            ProbabilityError,
            allow_strings=False,
        )

        if (
            value < -PROBABILITY_EPSILON
            or value > 1 + PROBABILITY_EPSILON
        ):
            raise ProbabilityError(
                f"Probability outside [0,1]: {value}"
            )

        if value < 0.0 or value > 1.0:
            logger.debug(
                "Probability %r is outside [0,1] within epsilon "
                "and will be clamped.",
                value,
            )

        return min(max(value, 0.0), 1.0)

    # -----------------------------------------------------------------
    # Prediction output validation
    # -----------------------------------------------------------------

    def _validate_probabilities(
        self,
        probabilities: Any,
        expected_size: int,
    ) -> list[float]:
        if isinstance(probabilities, _TEXT_OR_BINARY_TYPES):
            raise PredictionError(
                "Invalid probability output format"
            )

        if isinstance(probabilities, np.ndarray):
            if probabilities.ndim != 1:
                raise PredictionError(
                    "Invalid probability output format"
                )
            size = probabilities.size
        elif isinstance(probabilities, Sequence):
            size = len(probabilities)
        else:
            raise PredictionError(
                "Invalid probability output format"
            )

        if size != expected_size:
            raise PredictionError(
                "Prediction size mismatch: "
                f"expected {expected_size}, got {size}"
            )

        validated: list[float] = []

        for index, value in enumerate(probabilities):
            try:
                validated.append(self._validate_probability(value))
            except ProbabilityError as exc:
                raise ProbabilityError(
                    f"Invalid probability at index {index}"
                ) from exc

        return validated

    def _normalize_returned_peak(self, peak: Any) -> float:
        """
        Normalize peak returned by prediction engine.

        This intentionally does not apply input-range validation,
        because returned peaks are model outputs, not user inputs.

        Engine outputs are validated strictly: strings are rejected.
        """

        return self._coerce_finite_float(
            peak,
            "returned peak",
            PredictionError,
            allow_strings=False,
        )

    def _validate_returned_peaks(
        self,
        peaks: Any,
        expected_size: int,
    ) -> list[float]:
        if isinstance(peaks, _TEXT_OR_BINARY_TYPES):
            raise PredictionError(
                "Invalid peaks output format"
            )

        if isinstance(peaks, np.ndarray):
            if peaks.ndim != 1:
                raise PredictionError(
                    "Invalid peaks output format"
                )
            size = peaks.size
        elif isinstance(peaks, Sequence):
            size = len(peaks)
        else:
            raise PredictionError(
                "Invalid peaks output format"
            )

        if size != expected_size:
            raise PredictionError(
                "Returned peaks size mismatch: "
                f"expected {expected_size}, got {size}"
            )

        validated: list[float] = []

        for index, peak in enumerate(peaks):
            try:
                validated.append(self._normalize_returned_peak(peak))
            except PredictionError as exc:
                raise PredictionError(
                    f"Invalid returned peak at index {index}"
                ) from exc

        return validated

    def _validate_batch_size(
        self,
        raw_peaks: Union[Sequence[Any], np.ndarray],
    ) -> int:
        if isinstance(raw_peaks, _TEXT_OR_BINARY_TYPES):
            raise InvalidInputError(
                "PeakLoad list must not be string or binary"
            )

        if isinstance(raw_peaks, np.ndarray):
            if raw_peaks.ndim != 1:
                raise InvalidInputError(
                    "PeakLoad array must be one-dimensional"
                )
            size = raw_peaks.size
        elif isinstance(raw_peaks, Sequence):
            size = len(raw_peaks)
        else:
            raise InvalidInputError(
                "PeakLoad sequence required"
            )

        if size == 0:
            raise InvalidInputError("Empty PeakLoad list")

        if size > MAX_BATCH_SIZE:
            raise InvalidInputError("Batch size exceeds limit")

        return size

    def _warn_returned_peaks_mismatch(
        self,
        normalized_peaks: list[float],
        returned_peaks: list[float],
        engine_provided_peaks: bool,
    ) -> None:
        """
        Warn if engine-provided returned peaks differ from normalized inputs.
        """

        if not engine_provided_peaks:
            return

        if len(normalized_peaks) != len(returned_peaks):
            return

        try:
            close = np.allclose(
                np.asarray(normalized_peaks, dtype=float),
                np.asarray(returned_peaks, dtype=float),
                rtol=1e-9,
                atol=1e-9,
            )
        except Exception:
            logger.debug(
                "Unable to compare returned peaks with input peaks.",
                exc_info=True,
            )
            return

        if not close:
            warnings.warn(
                "Returned peaks differ from normalized input peaks.",
                UserWarning,
                stacklevel=4,
            )

    # -----------------------------------------------------------------
    # Batch prediction
    # -----------------------------------------------------------------

    def predict_many(
        self,
        raw_peaks: Union[Sequence[Any], np.ndarray],
        time_horizon: Any,
        residual_policy: Union[
            ResidualPolicy,
            str,
        ] = ResidualPolicy.PLUG_IN,
        covariates: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Predict probabilities for multiple PeakLoad values.
        """

        batch_size = self._validate_batch_size(raw_peaks)
        residual_policy_str = self._normalize_residual_policy(
            residual_policy
        )

        try:
            raw_peaks_list = list(raw_peaks)
        except TypeError as exc:
            raise InvalidInputError(
                "PeakLoad sequence is not iterable"
            ) from exc

        if len(raw_peaks_list) != batch_size:
            raise InvalidInputError(
                "PeakLoad sequence size mismatch"
            )

        peaks = [
            self._normalize_peak(peak)
            for peak in raw_peaks_list
        ]

        self._warn_peaks_outside_training_range(peaks)

        time_horizon = self._normalize_time(time_horizon)
        covariates = self._normalize_covariates(covariates)

        peaks_snapshot = list(peaks)
        peaks_for_engine = list(peaks_snapshot)
        covariates_for_engine = (
            dict(covariates)
            if covariates is not None
            else None
        )

        # Resolve time_horizon_unit for the engine.
        # self._time_unit is extracted from training_meta during __init__.
        # If unknown, fall back to the project default (engine_hours).
        _th_unit = self._time_unit if self._time_unit != "unknown" else "engine_hours"

        try:
            result = engine_predict_many(
                self._params,
                peaks_for_engine,
                time_horizon,
                residual_policy=residual_policy_str,
                covariates=covariates_for_engine,
                time_horizon_unit=_th_unit,
            )
        except (
            InvalidInputError,
            ModelValidationError,
            PredictionError,
            ProbabilityError,
        ):
            raise
        except Exception as exc:
            raise PredictionError(
                f"Batch prediction failed for {len(peaks_snapshot)} samples"
            ) from exc

        if not isinstance(result, Mapping):
            raise PredictionError(
                "Prediction engine returned invalid result"
            )

        probabilities = self._validate_probabilities(
            result.get("probabilities"),
            len(peaks_snapshot),
        )

        engine_provided_peaks = "peaks" in result
        returned_peaks_raw = result.get("peaks", peaks_snapshot)
        returned_peaks = self._validate_returned_peaks(
            returned_peaks_raw,
            len(peaks_snapshot),
        )

        self._warn_returned_peaks_mismatch(
            peaks_snapshot,
            returned_peaks,
            engine_provided_peaks,
        )

        response: dict[str, Any] = {
            "raw_peaks": raw_peaks_list,
            "normalized_peaks": peaks_snapshot,
            "peaks": returned_peaks,
            "probabilities": probabilities,
            "time_horizon": time_horizon,
            "residual_policy": residual_policy_str,
            "time_unit": self._time_unit,
            "model_calibration_horizon": self._calibration_time_horizon,
            "model_calibration_time_unit": self._time_unit,
        }

        # Backward-compatible field.
        # It is only meaningful when model time unit is explicitly days.
        response["model_calibration_horizon_days"] = (
            self._calibration_time_horizon
            if self._time_unit == "days"
            else None
        )

        return response

    # -----------------------------------------------------------------
    # Single prediction
    # -----------------------------------------------------------------

    def predict_probability(
        self,
        raw_peak: Any,
        time_horizon: Any,
        residual_policy: Union[
            ResidualPolicy,
            str,
        ] = ResidualPolicy.PLUG_IN,
        covariates: Optional[Mapping[str, Any]] = None,
    ) -> float:
        """
        Predict probability for a single PeakLoad value.
        """

        residual_policy_str = self._normalize_residual_policy(
            residual_policy
        )
        peak = self._normalize_peak(raw_peak)

        self._warn_peaks_outside_training_range([peak])

        time_horizon = self._normalize_time(time_horizon)
        covariates = self._normalize_covariates(covariates)

        covariates_for_engine = (
            dict(covariates)
            if covariates is not None
            else None
        )

        # Resolve time_horizon_unit for the engine.
        _th_unit = self._time_unit if self._time_unit != "unknown" else "engine_hours"

        try:
            value = engine_predict_probability(
                self._params,
                peak,
                time_horizon,
                residual_policy=residual_policy_str,
                covariates=covariates_for_engine,
                time_horizon_unit=_th_unit,
            )
        except (
            InvalidInputError,
            ModelValidationError,
            PredictionError,
            ProbabilityError,
        ):
            raise
        except Exception as exc:
            raise PredictionError("Single prediction failed") from exc

        return self._validate_probability(value)

    # -----------------------------------------------------------------
    # Cox hazard ratios
    # -----------------------------------------------------------------

    def hazard_ratios(self) -> dict[str, float]:
        """
        Return Cox model hazard ratios.

        HR = exp(coefficient)

        Results are cached after the first computation.
        The returned dictionary is a copy, so mutating it does not
        affect predictor state.

        Raises:
            ModelValidationError: if coefficients are invalid.
        """

        if self._cached_ratios is not None:
            return dict(self._cached_ratios)

        ratios: dict[str, float] = {}

        for name, coefficient in self._coefficients.items():
            # Coefficients were validated during initialization.
            # These checks remain as defensive guards.
            if coefficient > MAX_EXP_INPUT:
                raise ModelValidationError(
                    f"Hazard ratio overflow: {name}"
                )

            if abs(coefficient) > MAX_COX_COEFFICIENT:
                raise ModelValidationError(
                    f"Cox coefficient too large: {name}"
                )

            try:
                with np.errstate(over="raise"):
                    ratio = float(np.exp(coefficient))
            except FloatingPointError as exc:
                raise ModelValidationError(
                    f"Hazard ratio overflow: {name}"
                ) from exc

            if not np.isfinite(ratio):
                raise ModelValidationError(
                    f"Hazard ratio overflow: {name}"
                )

            if ratio <= 0.0:
                raise ModelValidationError(
                    f"Hazard ratio underflow: {name}"
                )

            ratios[name] = ratio

        self._cached_ratios = MappingProxyType(ratios)
        return dict(self._cached_ratios)