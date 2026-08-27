# parametric_baseline.py (v1.0)
"""
Параметрический базовый риск для CF Cox / IV-Cox.
Заменяет непараметрическую ступеньку Бреслоу на гладкие семейства
Weibull / Gompertz / Exponential, поддерживающие экстраполяцию.

Гарантии совместимости:
  - Чистые функции, без побочных эффектов и импорта проекта.
  - Параметрический H0(t) подгоняется ПОД КРИВУЮ Бреслоу, поэтому
    в наблюдаемом диапазоне предсказания воспроизводят текущие результаты.
  - Экстраполяция за последнее событие — новая возможность.
  - При family == "breslow" вызывающий код использует старый путь.

Параметризация СТРОГО согласована с Итог.py::_simulate_event_times:
  Weibull:      H0(t) = λ · t^k
  Gompertz:     H0(t) = (λ / b) · (exp(b·t) - 1)
  Exponential:  H0(t) = λ · t
Индивидуальный кумулятивный риск: H(t|X) = H0(t) · exp(lp).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

VALID_PARAMETRIC_FAMILIES = frozenset({"weibull", "gompertz", "exponential"})
ALL_BASELINE_FAMILIES = VALID_PARAMETRIC_FAMILIES | {"breslow"}
MAX_EXP_ARG = 700.0


# ────────────────────────────────────────────────────────────────────────────
# Контейнер спецификации
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class BaselineSpec:
    """Спецификация базового риска, сохраняемая в артефакт модели."""
    family: str = "breslow"
    # weibull:     params = {"lambda": λ, "shape": k}
    # gompertz:    params = {"lambda": λ, "rate":  b}
    # exponential: params = {"lambda": λ}
    params: Dict[str, float] = field(default_factory=dict)
    fit_r2: Optional[float] = None
    fit_rmse_log: Optional[float] = None
    max_fitted_time: Optional[float] = None
    n_fit_points: Optional[int] = None

    def is_parametric(self) -> bool:
        return self.family.lower() in VALID_PARAMETRIC_FAMILIES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "params": dict(self.params),
            "fit_r2": self.fit_r2,
            "fit_rmse_log": self.fit_rmse_log,
            "max_fitted_time": self.max_fitted_time,
            "n_fit_points": self.n_fit_points,
        }

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "BaselineSpec":
        if not isinstance(d, dict):
            return BaselineSpec()
        return BaselineSpec(
            family=str(d.get("family", "breslow")).lower(),
            params={str(k): float(v) for k, v in (d.get("params") or {}).items()},
            fit_r2=d.get("fit_r2"),
            fit_rmse_log=d.get("fit_rmse_log"),
            max_fitted_time=d.get("max_fitted_time"),
            n_fit_points=d.get("n_fit_points"),
        )


# ────────────────────────────────────────────────────────────────────────────
# Аналитические формы H0(t) и h0(t)
# ────────────────────────────────────────────────────────────────────────────
def _positive_t(t: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(t, dtype=float), 0.0)


def h0_cumulative(spec: Dict[str, Any], t: np.ndarray) -> np.ndarray:
    """Базовый КУМУЛЯТИВНЫЙ риск H0(t)."""
    family = str(spec.get("family", "")).lower()
    p = spec.get("params", {})
    t = _positive_t(t)
    if family == "weibull":
        lam, k = float(p["lambda"]), float(p["shape"])
        return lam * np.power(t, k)
    if family == "gompertz":
        lam, b = float(p["lambda"]), float(p["rate"])
        if abs(b) < 1e-12:
            return lam * t
        arg = np.clip(b * t, -MAX_EXP_ARG, MAX_EXP_ARG)
        return (lam / b) * (np.exp(arg) - 1.0)
    if family == "exponential":
        return float(p["lambda"]) * t
    raise ValueError(f"h0_cumulative: не параметрическое семейство '{family}'")


def h0_instantaneous(spec: Dict[str, Any], t: np.ndarray) -> np.ndarray:
    """Базовый мгновенный риск h0(t) = dH0/dt (нужен для плотности и CIF)."""
    family = str(spec.get("family", "")).lower()
    p = spec.get("params", {})
    t = _positive_t(t)
    if family == "weibull":
        lam, k = float(p["lambda"]), float(p["shape"])
        return lam * k * np.power(np.maximum(t, 1e-300), k - 1.0)
    if family == "gompertz":
        lam, b = float(p["lambda"]), float(p["rate"])
        arg = np.clip(b * t, -MAX_EXP_ARG, MAX_EXP_ARG)
        return lam * np.exp(arg)
    if family == "exponential":
        return np.full_like(t, float(p["lambda"]))
    raise ValueError(f"h0_instantaneous: не параметрическое семейство '{family}'")


def individual_cumulative_hazard(spec: Dict[str, Any], lp: float, t: np.ndarray) -> np.ndarray:
    """H(t|X) = H0(t) · exp(lp)."""
    return h0_cumulative(spec, t) * math.exp(float(lp))


def individual_survival(spec: Dict[str, Any], lp: float, t: np.ndarray) -> np.ndarray:
    """S(t|X) = exp(-H(t|X))."""
    H = individual_cumulative_hazard(spec, lp, t)
    return np.exp(-np.clip(H, 0.0, MAX_EXP_ARG))


def individual_density(spec: Dict[str, Any], lp: float, t: np.ndarray) -> np.ndarray:
    """Плотность отказа f(t|X) = h(t|X)·S(t|X). Основа для CIF."""
    t = _positive_t(t)
    h = h0_instantaneous(spec, t) * math.exp(float(lp))
    S = individual_survival(spec, lp, t)
    return h * S


# ────────────────────────────────────────────────────────────────────────────
# Подгонка параметрической кривой ПОД Бреслоу
# ────────────────────────────────────────────────────────────────────────────
def _prepare_breslow_points(
    times: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = (
        np.isfinite(times) & np.isfinite(values) & (times > 0.0) & (values > 0.0)
    )
    t, H = times[mask], values[mask]
    order = np.argsort(t)
    return t[order], H[order]


def _gof_log(t: np.ndarray, H: np.ndarray, H_fit: np.ndarray) -> Tuple[float, float]:
    """R² и RMSE в log-шкале (правильная метрика для кумулятивного риска)."""
    log_H, log_fit = np.log(H), np.log(np.maximum(H_fit, 1e-300))
    resid = log_H - log_fit
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((log_H - log_H.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(resid**2)))
    return r2, rmse


def fit_weibull_to_breslow(times: np.ndarray, values: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Weibull H0(t)=λ·t^k. В log-шкале: log H0 = log λ + k·log t → линейная OLS.
    Возвращает (lambda, shape, r2, rmse_log). Неитеративно и робастно.
    """
    t, H = _prepare_breslow_points(times, values)
    if len(t) < 3:
        raise ValueError("fit_weibull_to_breslow: < 3 положительных точек Бреслоу")
    log_t, log_H = np.log(t), np.log(H)
    slope, intercept = np.polyfit(log_t, log_H, 1)
    shape = float(slope)
    lam = float(math.exp(intercept))
    if shape <= 0.0 or not math.isfinite(shape):
        raise ValueError(f"Weibull shape должен быть > 0, получено {shape}")
    if lam <= 0.0 or not math.isfinite(lam):
        raise ValueError(f"Weibull lambda должен быть > 0, получено {lam}")
    r2, rmse = _gof_log(t, H, lam * np.power(t, shape))
    return lam, shape, r2, rmse


def fit_gompertz_to_breslow(
    times: np.ndarray,
    values: np.ndarray,
    rate_init: float = 0.01,
) -> Tuple[float, float, float, float]:
    """Gompertz H0(t)=(λ/b)(e^{bt}-1). Нелинейный МНК через scipy."""
    from scipy.optimize import curve_fit

    t, H = _prepare_breslow_points(times, values)
    if len(t) < 4:
        raise ValueError("fit_gompertz_to_breslow: < 4 точек Бреслоу")

    def model(tt: np.ndarray, lam: float, b: float) -> np.ndarray:
        arg = np.clip(b * tt, -MAX_EXP_ARG, MAX_EXP_ARG)
        return (lam / b) * (np.exp(arg) - 1.0)

    lam_init = float(H[-1] / t[-1]) if t[-1] > 0 else 1e-4
    try:
        popt, _ = curve_fit(
            model, t, H, p0=[lam_init, rate_init],
            bounds=([0.0, 1e-6], [np.inf, 1.0]), maxfev=20000,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Gompertz fit не сошёлся: {exc}") from exc
    lam, b = float(popt[0]), float(popt[1])
    r2, rmse = _gof_log(t, H, model(t, lam, b))
    return lam, b, r2, rmse


def fit_exponential_to_breslow(times: np.ndarray, values: np.ndarray) -> Tuple[float, float, float]:
    """Exponential H0(t)=λ·t. λ = наклон через начало (взвешенный)."""
    t, H = _prepare_breslow_points(times, values)
    if len(t) < 2:
        raise ValueError("fit_exponential_to_breslow: < 2 точек")
    lam = float(np.sum(t * H) / np.sum(t * t))  # LSQ через 0
    if lam <= 0.0:
        raise ValueError("Exponential lambda должен быть > 0")
    r2, rmse = _gof_log(t, H, lam * t)
    return lam, r2, rmse


# ────────────────────────────────────────────────────────────────────────────
# Вершина: собрать спецификацию из кривой Бреслоу
# ────────────────────────────────────────────────────────────────────────────
def fit_parametric_baseline(
    breslow_times: np.ndarray,
    breslow_values: np.ndarray,
    family: str,
) -> BaselineSpec:
    """Подогнать параметрический базовый риск под кривую Бреслоу."""
    fam = str(family).lower()
    if fam not in VALID_PARAMETRIC_FAMILIES:
        raise ValueError(
            f"fit_parametric_baseline: семейство '{fam}' не поддерживается. "
            f"Допустимы: {sorted(VALID_PARAMETRIC_FAMILIES)}"
        )
    t, _ = _prepare_breslow_points(breslow_times, breslow_values)
    max_t = float(t[-1]) if len(t) else None
    n_pts = int(len(t))

    if fam == "weibull":
        lam, shape, r2, rmse = fit_weibull_to_breslow(breslow_times, breslow_values)
        params = {"lambda": lam, "shape": shape}
    elif fam == "gompertz":
        lam, rate, r2, rmse = fit_gompertz_to_breslow(breslow_times, breslow_values)
        params = {"lambda": lam, "rate": rate}
    else:  # exponential
        lam, r2, rmse = fit_exponential_to_breslow(breslow_times, breslow_values)
        params = {"lambda": lam}

    spec = BaselineSpec(
        family=fam, params=params, fit_r2=r2, fit_rmse_log=rmse,
        max_fitted_time=max_t, n_fit_points=n_pts,
    )
    logger.info(
        "Параметрический базовый риск [%s]: params=%s, R²(log)=%.4f, точек=%d",
        fam, params, r2, n_pts,
    )
    return spec


def validate_baseline_spec(spec: Dict[str, Any]) -> bool:
    """Валидация спецификации перед сохранением/загрузкой."""
    fam = str(spec.get("family", "breslow")).lower()
    if fam == "breslow":
        return True
    if fam not in VALID_PARAMETRIC_FAMILIES:
        raise ValueError(f"Неизвестное семейство базового риска: '{fam}'")
    p = spec.get("params", {})
    if fam == "weibull":
        lam, k = float(p["lambda"]), float(p["shape"])
        if lam <= 0 or k <= 0:
            raise ValueError("Weibull: lambda>0 и shape>0 обязательны")
    elif fam == "gompertz":
        lam, b = float(p["lambda"]), float(p["rate"])
        if lam <= 0 or b <= 0:
            raise ValueError("Gompertz: lambda>0 и rate>0 обязательны")
    elif fam == "exponential":
        if float(p["lambda"]) <= 0:
            raise ValueError("Exponential: lambda>0 обязателен")
    return True


def compute_cif(
    spec: Dict[str, Any],
    lp: float,
    horizon: float,
    omega_fn,
    n_quad: int = 96,
) -> float:
    """
    Cumulative Incidence Function для причины k:
        CIF_k(t) = ∫₀ᵗ ω_k(u|X) · f(u|X) du,
    где f(u|X) = h0(u)·exp(lp)·exp(-H0(u)·exp(lp)) — плотность отказа,
    ω_k(u|X) — доля причины (может зависеть от времени).

    Использует квадратуру Гаусса–Лежандра на [0, horizon].
    Требует ПАРАМЕТРИЧЕСКИЙ базовый риск (гладкую плотность).

    Параметры
    ---------
    spec     : параметрическая спецификация базового риска
    lp       : линейный предиктор Кокса β'X
    horizon  : горизонт интегрирования
    omega_fn : callable(t)->float, доля причины в момент времени
    n_quad   : число узлов квадратуры (64–128 достаточно)
    """
    if str(spec.get("family", "breslow")).lower() not in VALID_PARAMETRIC_FAMILIES:
        raise ValueError("compute_cif требует параметрический базовый риск")
    horizon = float(horizon)
    if horizon <= 0.0:
        return 0.0

    # Узлы Гаусса–Лежандра на [-1, 1] → отображаем на [0, horizon]
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    t_nodes = 0.5 * horizon * (nodes + 1.0)
    w_nodes = 0.5 * horizon * weights

    f_vals = individual_density(spec, lp, t_nodes)
    omega_vals = np.array([float(omega_fn(float(t))) for t in t_nodes], dtype=float)
    omega_vals = np.clip(omega_vals, 0.0, 1.0)

    cif = float(np.sum(w_nodes * omega_vals * f_vals))
    return float(min(max(cif, 0.0), 1.0))


# Маркер доступности модуля
HAS_PARAMETRIC_BASELINE = True
