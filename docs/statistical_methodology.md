# Statistical Methodology: CF-Cox / IV-Cox

---

## 1. Problem Statement

Оценка структурного эффекта пиковой нагрузки (PeakLoad) на интенсивность
отказов сельскохозяйственной техники в условиях эндогенности.

**Эндогенность:** PeakLoad коррелирует с ненаблюдаемыми факторами
(качество почвы, квалификация оператора, интенсивность использования),
которые также влияют на риск отказа.

---

## 2. Identification Strategy

### 2.1 Instrumental Variable: Rainfall Anomaly

Инструмент Z = rainfall anomaly (NASA POWER, campaign=sowing).

**Relevance:** corr(Z, PeakLoad) = +0.642, F_cluster = 59,930  
**Exclusion:** corr(Z, x_climate) = +0.093, corr(Z, x_soil) = +0.081  
**Exogeneity:** Предполагается (не тестируется напрямую)

### 2.2 Control Function Approach

Вместо 2SLS (неприменим для нелинейных моделей) используется
Control Function:

1. First stage: `PeakLoad = π₀ + π_Z·Z + π_X·X + u`
2. Residual: `v_hat = PeakLoad − fitted(PeakLoad)`
3. CF-Cox: `h(t|X) = h₀(t)·exp(β'X + γ·PeakLoad + λ·v_hat)`

Если λ ≠ 0 → эндогенность присутствует.
Если λ = 0 → CF-Cox редуцируется к стандартному Cox.

### 2.3 Interpretation

**В нелинейной модели Cox γ + λ НЕ является каузальным эффектом.**

Для каузальной интерпретации требуется:
- Monte Carlo recovery (P0-1)
- Валидный инструмент на реальных данных
- Реальные claims для estimation

---

## 3. Cluster-Robust Inference

Данные имеют кластерную структуру: 32 кластера (Region × Year × Campaign).

**First stage:** Cluster-robust F-statistic (sandwich estimator).  
**Cox model:** Cluster-robust SE через `lifelines.CoxPHFitter(cluster_col=...)`.

**Важно:**
- Point estimates НЕ меняются от кластеризации
- SE могут быть как больше, так и меньше naive SE
- 32 кластера < 50 (Stock-Yogo threshold) → интерпретировать консервативно

---

## 4. Baseline Hazard

**Family:** Weibull  
**Shape:** k = 1.88 (wear-out regime, hazard растёт со временем)  
**Scale:** MTBF = 1500 engine hours  
**Валидация:** Zetor shape = 1.91, deviation = 1.8%

---

## 5. Bayesian Calibration

Major failure share калибруется через Beta-Binomial conjugate update:

```text
Prior:     Beta(9, 21)     → mean = 0.30, effective_n = 30
Data:      k major из n total
Posterior: Beta(9 + k, 21 + n − k)
Production: posterior mean
```

Для брендов: empirical Bayes shrinkage к global prior.

---

## 6. Interaction Term

Age × Hours interaction (центрированный и стандартизованный):

```text
x_age_hours = (age − mean_age) × (hours − mean_hours) / std_combined
```

**Валидация:** LR test = 29.19, p < 1e-9, β = +0.173.

---

## 7. Prediction

```text
P(T ≤ t_horizon | X) = 1 − exp(−H₀(t_horizon) · exp(β'X + γ·PeakLoad + λ·v_hat))
```

где H₀(t) — cumulative baseline hazard (Weibull).

**Kalman-Meier validation:** |S₀(1712) − S_KM(1712)| = 0.0023 < 0.15 ✅

---

## 8. Diagnostics

| Diagnostic | Method | Threshold |
|---|---|---|
| Instrument relevance | Cluster F-statistic | > 10 (Stock-Yogo) |
| Exclusion restriction | corr(Z, X) | < 0.15 |
| Endogeneity | λ ≠ 0 (t-test) | p < 0.05 |
| Proportional hazards | Schoenfeld residuals | p > 0.05 per variable |
| Interaction significance | LR test | p < 0.05 |
| Baseline calibration | KM vs model survival | |diff| < 0.15 |
| Marginal probability | Mean P vs target | |error| < 0.03 |
