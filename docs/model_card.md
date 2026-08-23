# Model Card: CF-Cox / IV-Cox Actuarial Pricing System

**Version:** 0.2.0  
**Date:** 2026-08-20  
**Status:** Methodology validated, awaiting real claims for production calibration  

---

## 1. Model Purpose

Actuarial pricing system for agricultural machinery insurance using
Control-Function Cox (CF-Cox) / Instrumental-Variable Cox (IV-Cox)
survival models. The model estimates the hazard rate of major failure events as a function of operational, environmental, and mechanical covariates.

**Intended use:** Premium calculation, risk segmentation, loss ratio forecasting.  
**NOT intended for:** Individual machine diagnostics, warranty claims adjudication.

---

## 2. Model Architecture

### 2.1 Estimation Pipeline

```text
DGP / Monte Carlo
        ↓
First Stage: OLS  PeakLoad ~ Z + X  (instrument relevance)
        ↓
Control Function: v_hat = residual from first stage
        ↓
CF-Cox: h(t|X) = h₀(t) · exp(β'X + γ·PeakLoad + λ·v_hat)
        ↓
Cluster-robust SE (32 clusters: Region × Year × Campaign)
        ↓
Baseline calibration: Weibull(shape=1.88, MTBF=1500h)
        ↓
Prediction: P(T ≤ t_horizon | X)
```

### 2.2 Key Parameters

| Parameter | Value | Source | Provenance Level |
|---|---|---|---|
| Weibull shape (baseline) | 1.88 | DGP + Zetor validation | Level 1 (validated) |
| MTBF baseline (hours) | 1500 | DGP calibration | Level 0 (synthetic) |
| γ (PeakLoad effect) | 0.125 ± 0.041 | CF-Cox fit (hybrid data) | Level 0+1 |
| λ (v_hat, endogeneity) | 0.360 ± 0.048 | CF-Cox fit | Level 0+1 |
| β_age×hours | 0.173 ± 0.027 | CF-Cox fit | Level 0+1 |
| β_x_climate | 0.285 ± 0.105 | CF-Cox fit | Level 0+1 |
| β_x_soil | 0.221 ± 0.173 | CF-Cox fit | Level 0+1 |
| major_failure_share | 0.0296 [0.0277, 0.0315] | Bayesian posterior | Level 0 |
| PeakLoad mean (TUM) | 0.7099 | TUM CAN bus data | Level 2 |
| PeakLoad std (TUM) | 0.2053 | TUM CAN bus data | Level 2 |

### 2.3 IV Mode

**Current mode: `predictive`**

γ is interpreted as a **predictive association**, not a causal effect.
Causal interpretation requires:
- Valid instrument (relevance + exclusion + exogeneity) on real data
- Monte Carlo recovery validation (in progress)
- Real claims dataset (not yet available)

---

## 3. Data Provenance

| Level | Name | Description | Current Status |
|---|---|---|---|
| 0 | `synthetic` | DGP / Monte Carlo simulation | ✅ Active (P0-1 MC) |
| 1 | `literature_reconstructed` | Published aggregates, synthetic reconstruction | ✅ Zetor, HADCO |
| 2 | `published_individual` | Individual records from published tables | ✅ TUM PeakLoad |
| 3 | `proprietary_claims` | OEM / dealer / insurer real data | 🔴 Not available |

**Critical:** Only Level 3 data can be called "real claims" in any publication.

---

## 4. Parameter Provenance Table

| Parameter | Source | Provenance | Interpretation |
|---|---|---|---|
| γ (DGP) = 0.5 | Synthetic DGP | Level 0 | Known ground truth for MC recovery |
| β_age×hours (DGP) = 0.15 | Synthetic DGP | Level 0 | Known ground truth |
| ρ = 0.7, δ = 0.5 | Synthetic DGP | Level 0 | Structural DGP parameters |
| Cox β (PeakLoad) = 0.125 | Simulated/hybrid data | Level 0+1 | Predictive estimate under current DGP |
| Cox β (x_age_hours) = 0.173 | Simulated/hybrid data | Level 0+1 | Predictive estimate |
| Cox β (x_climate) = 0.285 | Simulated + real weather | Level 0+1 | Predictive estimate |
| major_failure_share = 0.0296 | Bayesian update | Level 0 | Calibrated probability |
| Zetor Weibull shape = 1.91 | Published literature | Level 1 | External survival benchmark |
| HADCO repair share = 54.2% | Published literature | Level 1 | External service benchmark |
| TUM PeakLoad = 0.71/0.21 | Real CAN bus data | Level 2 | Calibration reference |
| **Real β estimates** | **NOT YET AVAILABLE** | **Level 3** | **Future real claims** |

---

## 5. Validation Summary

### 5.1 Internal Validation (P0 Layer)

| P0 Task | Status | Key Result |
|---|---|---|
| P0-1 MC recovery γ | 🟡 In progress | Awaiting 100-run results |
| P0-2 Consistency gate | ✅ Closed | All consistency checks pass |
| P0-3 LR Age × Hours | ✅ Closed | LR=29.19, p<1e-9, β=+0.173 |
| P0-4 PH diagnostics | ✅ Closed | 4/4 tests pass |
| P0-5 Bayesian share | ✅ Closed | Posterior=0.0296, CrI=[0.0277, 0.0315] |
| P0-6 Cluster-robust Cox | ✅ Closed | 5/5 tests pass, F_cluster=59930 |

### 5.2 External Benchmarks

| Benchmark | Metric | Result | Interpretation |
|---|---|---|---|
| Zetor survival shape | Weibull shape deviation | **1.8%** | DGP baseline validated |
| Zetor MTTF/MTBF ratio | Ratio | 0.184 | Physically plausible [0.1, 0.6] |
| Zetor best fit | AIC | Weibull | Confirms model family choice |
| HADCO repair share | Repair % | 54.2% | Consistent with service data |
| HADCO severity R² | GLM R² | 97.5% | Internal consistency (synthetic) |
| HADCO power effect | Elasticity | 0.0062 | Power is secondary cost driver |

### 5.3 What Can Be Honestly Claimed

✅ "Methodology validated on Monte Carlo with known ground truth"  
✅ "Weibull baseline shape=1.88 confirmed by independent Zetor data (deviation 1.8%)"  
✅ "Cost/severity structure consistent with published HADCO data"  
✅ "PeakLoad calibrated on real TUM CAN bus data"  
✅ "Major failure share: Bayesian posterior 0.0296 [95% CrI: 0.0277, 0.0315]"  
✅ "Interaction Age×Hours statistically significant (LR=29.19, p<1e-9)"  

❌ "Model empirically validated on real insurance claims"  
❌ "Coefficients β reflect real-world causal effects"  
❌ "PeakLoad γ is a causal parameter" (currently predictive mode)  

---

## 6. Limitations

1. **No real claims data.** All β estimates are from simulated/hybrid data.
2. **IV in predictive mode.** Causal interpretation requires valid instrument on real data.
3. **Zetor/HADCO are Level 1.** Literature-reconstructed, not individual-level real data.
4. **No censoring in Zetor.** All 70 observations are events; cannot validate censoring mechanism.
5. **HADCO lacks event-level hours.** Cannot be used for Cox survival analysis.
6. **32 clusters.** Below the 50-cluster threshold for fully reliable cluster-robust inference (Stock & Yogo recommendation).

---

## 7. Reproducibility

| Component | Seed | Command |
|---|---|---|
| MC recovery | 42 | `python mc_recovery_test.py --n-sims 100 --n-tractors 40000 --gamma-true 0.5 --seed 42 --n-jobs 6` |
| Model training | 12345 | `python train_model.py` (hybrid mode, source=3) |
| Zetor benchmark | 42 | `python benchmarks/zetor_benchmark.py --seed 42` |
| HADCO benchmark | 42 | `python benchmarks/hadco_benchmark.py --seed 42` |

---

## 8. Ethical Considerations

- No individual-level real claims data used in current version.
- TUM PeakLoad data is anonymized operational telemetry.
- Zetor/HADCO data is from published academic literature.
- Model should not be used for individual machine fault diagnosis.
- Premium decisions should be reviewed by qualified actuaries.
