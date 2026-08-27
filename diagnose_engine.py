# diagnose_engine.py
"""Диагностика: почему engine даёт P=0, а ручной расчёт P=0.0085."""
from prediction_engine import (
    load_model_params,
    predict_first_stage,
    transform_peak,
    _cox_linear_predictor_details,
    _resolve_covariate_value,
)

model = load_model_params("model_params.json")

# Те же ковариаты, что ввёл калькулятор (МТЗ-82, 10 лет, 1000 мч, 80 л.с.)
covariates = {
    "x_age": 10.0,
    "x_hours": 1000.0,
    "x_power": 80.0,
    "x_climate": 0.6056,
    "x_soil": 0.8863,
    "x_age_hours": 0.0,
    "brand_Versatile280": 0.0,
    "brand_NewHollandT9": 0.0,
    "brand_DT75": 0.0,
    "brand_Other": 0.0,
}
peak_raw = 0.6663
horizon = 1712.0

print("=" * 72)
print("1) ПЕРВАЯ СТАДИЯ")
print("=" * 72)
pl_hat = predict_first_stage(model, covariates, strict_covariates=True)
peak_t = transform_peak(model, peak_raw)
print(f"pl_hat             = {pl_hat:+.6f}   (в ручном было +0.9826)")
print(f"peak_transformed   = {peak_t:+.6f}")
print(f"raw_residual       = {peak_t - pl_hat:+.6f}")

print("\n" + "=" * 72)
print("2) РАЗРЕШЕНИЕ КАЖДОЙ КОВАРИАТЫ COX")
print("=" * 72)
cox = model.cox
cox_names = cox.get("exog_names", [])
cox_coefs = cox.get("coefs", {})
print(f"{'ковариата':26s}{'значение':>14s}{'коэфф':>12s}{'вклад':>12s}")
print("-" * 64)
total = 0.0
for name in cox_names:
    try:
        val = _resolve_covariate_value(model, name, covariates, strict=True)
        coef = cox_coefs.get(name, 0.0)
        contrib = coef * val
        total += contrib
        flag = "  <<< ОГРОМНЫЙ ВКЛАД" if abs(contrib) > 3 else ""
        print(f"{name:26s}{val:+14.6f}{coef:+12.6f}{contrib:+12.6f}{flag}")
    except Exception as e:
        print(f"{name:26s}  ОШИБКА: {e}")
print("-" * 64)
print(f"{'ИТОГО lp':26s}{'':>14s}{'':>12s}{total:+12.6f}")

print("\n" + "=" * 72)
print("3) ПОЛНЫЙ РАСЧЁТ ЧЕРЕЗ ENGINE")
print("=" * 72)
details = _cox_linear_predictor_details(
    params=model,
    peak_raw=peak_raw,
    time_horizon=horizon,
    residual_policy="plug-in",
    covariates=covariates,
    time_horizon_unit="engine_hours",
    strict_covariates=True,
)
print(f"v_hat       = {details['v_hat']:+.6f}")
print(f"lp          = {details['lp']:+.6f}")
print(f"h0          = {details['h0_t']:.6f}")
print(f"probability = {details['probability']:.6f}")