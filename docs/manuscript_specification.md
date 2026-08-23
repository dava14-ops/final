# Manuscript Specification: CF-Cox Framework

**Status:** FROZEN v1.0 — binding contract between code and manuscript
**Created:** 2026-08-21
**Last updated:** 2026-08-21
**Freeze point:** All hypotheses and success criteria frozen BEFORE final MC results reviewed
**Next action:** Populate Table 3 AFTER 100 MC runs complete

---

## 1. Title and Positioning

### Chosen title (Variant A — methodology-first)

> **A Control-Function Cox Framework for Endogenous Operational Load in Agricultural Machinery Reliability Modeling**

### Backup titles (if reviewers push for applied framing)

- Variant B: "Endogeneity-Aware Survival Modeling of Agricultural Machinery Failure Risk for Insurance Pricing"
- Variant C: "An Endogeneity-Aware Survival and Severity Framework for Tractor Failure Risk"

### Positioning statement (one sentence)

> This paper presents a **methodological framework** for endogeneity-aware survival modeling of operational load in agricultural machinery, validated through Monte Carlo recovery and external literature benchmarks; empirical calibration to real insurance claims is an explicit next stage, not a claim of this study.

### What this paper is NOT

- ❌ Not a "ready insurance pricing model"
- ❌ Not an empirical validation on real claims
- ❌ Not a causal claim about PeakLoad → failure
- ❌ Not a comparative study against alternative survival models

---

## 2. Target Journals (ranked)

| Priority | Journal | Rationale |
|---|---|---|
| 1 | *Reliability Engineering & System Safety* | Survival + endogeneity + reliability |
| 2 | *ASTIN Bulletin* | Actuarial + methodological |
| 3 | *Risk Analysis* | Endogeneity + insurance application |
| 4 | *Journal of Risk and Insurance* | Insurance methodology |
| 5 | *Biosystems Engineering* | Agricultural machinery focus |

---

## 3. Research Question

**Primary:**

> Can a control-function Cox proportional hazards framework recover a known structural effect of an endogenous operational load variable in clustered survival data for agricultural machinery, while remaining consistent with published reliability benchmarks?

**Secondary:**

1. Is the estimator finite-sample unbiased under the specified DGP?
2. Is the selected Weibull baseline shape empirically plausible?
3. Does the framework produce internally consistent actuarial outputs?
4. Are external literature benchmarks (Zetor, HADCO) consistent with the model's structural assumptions?

---

## 4. Hypotheses (H1–H5)

Each hypothesis has **pre-specified success criteria** to prevent post-hoc rationalization. These criteria were frozen **before** reviewing final MC results.

### H1: Estimator Recovery (Monte Carlo)

> **H1:** The CF-Cox estimator recovers the true structural coefficient γ_true = 0.5 with relative bias < 5% and RMSE < 0.05.

**Pre-specified success criteria (all must hold):**
- |relative bias| < 5%
- RMSE < 0.05
- MCSE < 0.01
- Failure rate < 2%
- Penalized fit rate = 0%

**H1-cov (secondary diagnostic, not pass/fail gate):**
> When bootstrap-based per-replication 95% CIs are computed, the Monte Carlo coverage rate should be consistent with nominal 95% within sampling uncertainty.

**Assessment method for H1-cov:**
- Coverage is a **secondary diagnostic**, not a pass/fail gate for H1
- For n_sims = 100 and nominal coverage p = 0.95, the Monte Carlo standard error is SE = √(p·(1−p)/n) ≈ 0.022
- The pre-specified **MC uncertainty interval** for observed coverage is [p − 2·SE, p + 2·SE] ≈ [0.906, 0.994]
- Observed coverage within this interval is consistent with nominal 95%
- Observed coverage below 0.86 or above 1.00 triggers diagnostic review

**Current status (n_bootstrap = 0):** coverage not computed in the primary MC run. A separate bootstrap coverage run (`--n-bootstrap 200` on a subset of replications) is required for this diagnostic and will be reported in Supplementary S3 if performed.

---

### H2: Instrument Relevance (Weak-Instrument Screening)

> **H2:** The rainfall anomaly instrument is strongly relevant for PeakLoad under the specified DGP, passing weak-instrument screening criteria.

**Pre-specified success criteria (all must hold):**
- Mean F_cluster > 10 (Staiger–Stock weak-instrument screening threshold)
- Min F_cluster > 10 across all replications
- |corr(Z, PeakLoad)| > 0.3 (relevance)
- |corr(Z, x_climate)| < 0.15 (partial exclusion probe)
- |corr(Z, x_soil)| < 0.15 (partial exclusion probe)

**Pre-specified disclaimer (will appear in Methods §8.6):**
> The F > 10 threshold is used as a pre-specified weak-instrument screening criterion following Staiger & Stock (1997). It is **not** claimed to be a universal Stock–Yogo critical value, which depends on the number of instruments, number of endogenous regressors, and chosen maximal relative bias / size distortion tolerance.

**Observed high F values (≈ 50,000) are reported as results, not criteria.**

**Explicitly NOT tested (stated in Limitations §11.2):**
- Instrument exogeneity (assumed, not tested)
- Exclusion restriction (only partially probed via observed X)

---

### H3: Weibull Baseline Plausibility (External Benchmark)

> **H3:** The selected Weibull shape k = 1.88 (wear-out regime) is consistent with independent published tractor reliability statistics.

**Pre-specified success criteria (all must hold):**
- |shape_Zetor − shape_DGP| / shape_DGP < 20%
- Weibull is best fit by AIC among {Weibull, Exponential, LogNormal, LogLogistic}
- Shape interpretation qualitatively consistent with wear-out regime (shape > 1)

**Removed from success criteria:**
- ~~MTTF/MTBF ratio~~: this ratio compares a published Zetor first-failure metric (MTTF ≈ 271 mth) to a DGP assumption (MTBF = 1500 h) and is therefore **not an independent validation statistic**. The ratio is reported as a plausibility diagnostic in Section 10, not as an H3 success criterion.

**Provenance wording (mandatory):**
> The DGP Weibull shape is **benchmarked against published Zetor reliability statistics** (Durczak et al., 2018), not validated against individual-level raw Zetor data. The Zetor benchmark uses literature-reconstructed synthetic observations (Level 1 provenance).

---

### H4: Cluster-Robust Inference Correctness

> **H4:** The cluster-robust sandwich covariance specification produces valid inference under the DGP's cluster structure.

**Pre-specified success criteria:**
- Point estimates (γ̂, β̂) are identical between naive and cluster-robust fits to within machine precision (≈ 1e-12)
- First-stage cluster F-statistic is computable for all replications
- All 5 regression tests in `tests/test_cluster_robust_cox.py` pass
- No silent fallback to naive SE when cluster_col is specified

**NOT a hypothesis (regression test, not statistical claim):**
- ~~SE_cluster ≠ SE_naive~~: cluster-robust SE may equal naive SE in low within-cluster correlation regimes. This is a regression test, not a statistical hypothesis. The manuscript will not state "correct cluster SE must differ from naive SE."

**Pre-specified limitation (will appear in Discussion §15.3):**
> The DGP generates 32 clusters (Region × Year × Campaign). This is below the asymptotic regime (G ≳ 50) where cluster-robust standard errors are fully reliable (small-G problem, distinct from Stock–Yogo weak-IV thresholds). Results should be interpreted conservatively.

---

### H5: Internal Calibration and Model-Consistency Diagnostics

#### H5a: Marginal calibration (calibration)

> The model-predicted marginal event probability at the calibration horizon matches the DGP target within 3 percentage points.

**Criterion:** |mean P̂(T ≤ 1712) − P_target| < 0.03

#### H5b: Baseline survival agreement (calibration)

> The fitted baseline survival S₀(t) agrees with the Kaplan–Meier estimate at the calibration horizon within 0.15 absolute difference.

**Criterion:** |S₀(1712) − S_KM(1712)| < 0.15

#### H5c: Interaction specification (specification)

> The Age × Hours interaction term is statistically significant, supporting the full specification over the restricted model.

**Criterion:** LR test p-value < 0.05, AIC_full < AIC_restricted

#### H5d: Bayesian calibration (implementation)

> The Beta-Binomial posterior for major-failure share is finite, non-degenerate, and shows appropriate shrinkage behavior.

**Implementation integrity criterion (not a statistical hypothesis):**
- Posterior mean, CrI bounds all finite
- Prior sensitivity reported in Table 7 (effective_n ∈ {10, 30, 100})

**NOT claimed:**
- ~~"Bayesian posterior well-defined proves model validity"~~: any proper conjugate update yields a finite posterior; this is a unit test, not evidence.

---

## 5. Primary and Secondary Estimands

| Estimand | Type | Definition |
|---|---|---|
| γ (structural PeakLoad effect) | Structural | Coefficient of PeakLoad in CF-Cox hazard |
| λ (endogeneity coefficient) | Diagnostic | Coefficient of v̂ in CF-Cox |
| β_age×hours (interaction) | Structural | Synergy effect |
| β_X (covariates) | Predictive | Climate, soil, power, brand coefficients |
| S₀(t) (baseline survival) | Function | Weibull baseline with shape=1.88 |
| P(T ≤ 1712 \| X) (marginal probability) | Predictive | Event probability at calibration horizon |
| p_major (major share posterior) | Calibrated | Beta-Binomial posterior mean |

---

## 6. Data Provenance Matrix

| Source | Level | Provenance | Use in paper | Allowed claims |
|---|---|---|---|---|
| Synthetic DGP | 0 | Our code | MC recovery, Cox fit | "known ground truth" |
| Hybrid DGP + weather/soil | 0+1 | DGP + NASA/GLDAS | Training Cox | "hybrid training data" |
| Zetor (Durczak et al. 2018) | 1 | Literature-reconstructed | External survival benchmark | "consistent with published aggregates" |
| HADCO (Al-Suhaibani) | 1 | Literature-reconstructed | External service/severity benchmark | "consistent with published aggregates" |
| TUM CAN bus (raw telemetry) | 2a | Published individual-level records | Source of calibration statistics | "calibrated on real operational telemetry" |
| `tum_peakload_stats.json` | 2b | **Aggregated calibration statistics derived from Level 2a data** | PeakLoad standardization in DGP and prediction | "uses summary statistics from TUM study" |
| NASA POWER | 2 | Public satellite data | Instrument Z, x_climate | "public satellite data" |
| GLDAS-2.1 | 2 | Public satellite data | x_soil | "public satellite data" |
| **Real insurance claims** | **3** | **Not available** | **Not used** | **Explicitly stated as future work** |

**Critical distinction for TUM:**
The production pipeline consumes only the **aggregated statistics** (Level 2b: mean=0.7099, std=0.2053), not individual-level raw telemetry. The manuscript will state:

> "PeakLoad standardization uses mean and standard deviation derived from TUM CAN-bus operational telemetry (source study). Individual machine-level records are not included in the training or prediction pipeline."

### Forbidden claims (must NOT appear anywhere)

- ❌ "validated on real insurance claims"
- ❌ "empirical β estimates for real tractors"
- ❌ "causal effect of PeakLoad established"
- ❌ "exclusion restriction confirmed"
- ❌ "production-ready insurance model"
- ❌ "publication-quality validation" (replace with "consistent with published data")
- ❌ "real Zetor data" (replace with "published Zetor reliability statistics")

### Required disclaimers

Every claim about DGP-derived β must include:
> "These are estimates under the specified DGP and hybrid training data; empirical calibration on real longitudinal claims is a required next stage."

Every claim about IV validity must include:
> "Instrument exogeneity is assumed; empirical validation on real claims is pending."

---

## 6.5. Identification Status

The framework operates in two mutually exclusive modes. The current study is entirely in **predictive mode**.

### Predictive mode (CURRENT)

- γ is interpreted as a **predictive association** between PeakLoad and failure hazard, conditional on observed X and estimated v̂
- No causal claim is made
- Instrument Z is used for endogeneity correction, not for identification of causal γ

### Causal mode (NOT CLAIMED)

For γ to be interpretable as a causal effect, the following must hold:

1. ✅ Monte Carlo recovery (H1, this study)
2. ❌ Real-data first-stage relevance (pending real claims)
3. ❌ Exclusion restriction assessment (partially probed; full assessment requires real data)
4. ❌ Exogeneity argument with unobserved confounders (not possible without real data)
5. ❌ Sensitivity to alternative instruments (requires real data)

**Pre-specified statement (will appear in Discussion §15.3):**
> This study establishes methodology in predictive mode. Transition to causal mode requires real longitudinal insurance claims with observed and unobserved confounders, and is deferred to future work.

---

## 7. Validation Criteria (pre-specified)

### 7.1 Estimator validation (H1)

| Metric | Target | Pass threshold |
|---|---|---|
| Relative bias | 0% | < 5% |
| RMSE | < 0.05 | < 0.05 |
| MCSE | < 0.01 | < 0.01 |
| Failure rate | 0% | < 2% |
| Penalized rate | 0% | = 0% |

### 7.2 First-stage validation (H2)

| Metric | Target | Pass threshold |
|---|---|---|
| Mean F_cluster | > 10 | > 10 (Staiger–Stock) |
| Min F_cluster | > 10 | > 10 |
| corr(Z, PeakLoad) | High | > 0.3 |
| corr(Z, X_obs) | Low | < 0.15 |

### 7.3 Baseline validation (H3)

| Metric | Target | Pass threshold |
|---|---|---|
| Weibull shape deviation | < 20% | < 20% |
| Best fit by AIC | Weibull | Weibull |

### 7.4 Diagnostics (H4, H5)

| Diagnostic | Target |
|---|---|
| LR Age×Hours p-value | < 0.05 |
| PH per-variable violations | Documented, not auto-reject |
| Marginal probability error | < 0.03 |
| Kaplan–Meier baseline deviation | < 0.15 |
| Cluster-robust regression tests | 5/5 pass |

---

## 8. Figures Plan (7 mandatory, 2 optional)

### Figure 1: Conceptual framework (causal diagram)
**Purpose:** Show the endogeneity problem and IV solution
**Content:** DAG with U (unobserved) → PeakLoad, Z → PeakLoad, PeakLoad → Failure, X → Failure, X → PeakLoad
**Note:** Must explicitly state "exogeneity of Z is assumed, not tested"

### Figure 2: Monte Carlo distribution of γ̂
**Purpose:** Visualize estimator recovery
**Content:** Histogram of 100 γ̂ values, vertical line at γ_true = 0.5, vertical lines at mean(γ̂), 95% empirical interval
**Data:** mc_recovery_results.csv

### Figure 3: Empirical distribution of γ̂ (mandatory, always available)

**Purpose:** Visualize the Monte Carlo sampling distribution of the estimator.

**Content:**
- Histogram of 100 γ̂ values
- Vertical line at γ_true = 0.5
- Vertical lines at mean(γ̂), 2.5th and 97.5th percentiles
- Rug plot of individual estimates

**Data:** `mc_recovery_results.csv` (no bootstrap required)

### Figure 3b (optional): MC 95% CI coverage (only if bootstrap run performed)

**Condition:** Shown only if a separate bootstrap coverage run (`--n-bootstrap 200` on subset of replications) is completed before submission.

**Content:** Forest plot of per-replication bootstrap CIs, colored by coverage of γ_true.

**If bootstrap run NOT performed:** Figure 3b omitted; coverage assessment reported as future work in Limitations.

### Figure 4: Zetor survival benchmark
**Purpose:** External validation of baseline shape
**Content:** Kaplan–Meier curve + fitted Weibull/Exponential/LogNormal, AIC comparison
**Data:** reports/zetor_survival_audit.json

### Figure 5: Age × Hours interaction surface
**Purpose:** Visualize significant interaction
**Content:** Hazard ratio surface over (age_std, hours_std) grid
**Data:** Cox model coefficients

### Figure 6: PH diagnostics summary
**Purpose:** Show proportional hazards assessment
**Content:** Forest plot of per-variable Schoenfeld p-values, global test indicator
**Data:** PH report from training

### Figure 7: Actuarial pipeline
**Purpose:** Show end-to-end pricing framework
**Content:** Flowchart: X → hazard → P(major) × E[severity] → E[loss] → premium
**Note:** Clearly mark which components are empirical vs. synthetic vs. expert

### Figure 8 (optional): Real claims calibration
**Only if real claims become available before submission**

### Figure 9 (optional): External temporal validation
**Only if real claims become available before submission**

---

## 9. Tables Plan (7 mandatory, 2 optional)

### Table 1: Data provenance
**Columns:** Source, Level, n, Variables, Use, Provenance status
**Content:** All data sources from provenance matrix

### Table 2: DGP parameters
**Columns:** Parameter, Value, Rationale
**Content:** γ, ρ, δ, β_age×hours, Weibull shape, baseline hazard, intercept, etc.

### Table 3: Monte Carlo Recovery Results (PRIMARY TABLE — populated after 100 MC runs)

| Metric | Pre-specified criterion | Final (100 runs) | H1 verdict |
|---|---|---|---|
| Mean γ̂ | — | TBD | — |
| Bias | \|bias\| < 0.025 | TBD | PASS/FAIL |
| Relative bias | < 5% | TBD | PASS/FAIL |
| SD(γ̂) | — | TBD | — |
| RMSE | < 0.05 | TBD | PASS/FAIL |
| MCSE | < 0.01 | TBD | PASS/FAIL |
| Failure rate | < 2% | TBD | PASS/FAIL |
| Penalized rate | = 0% | TBD | PASS/FAIL |
| Mean F_cluster | > 10 | TBD | PASS/FAIL |
| Min F_cluster | > 10 | TBD | PASS/FAIL |

**Coverage row:** reported only if separate bootstrap run completed; otherwise noted as "not assessed (deferred to future work)."

### Table 4: Cox coefficients
**Columns:** Variable, β̂, SE_cluster, 95% CI, p-value
**Content:** PeakLoad, v̂, Age, Hours, Age×Hours, climate, soil, power, brands

### Table 5: P0 diagnostic summary
**Columns:** Diagnostic, Result, Status
**Content:** LR test, PH, cluster robustness, marginal calibration, Kaplan–Meier baseline

### Table 6: External benchmark comparison
**Columns:** Benchmark, Metric, Published, Our DGP, Deviation
**Content:** Zetor shape/MTTF, HADCO repair share/severity

### Table 7: Sensitivity analysis
**Columns:** Scenario, γ̂, Bias, RMSE, F_cluster
**Content:** Varying ρ, π_Z, event rate, baseline shape, prior effective_n

### Table 8 (optional): Real claims cohort
**Only if real claims available**

### Table 9 (optional): Real model performance
**Only if real claims available**

---

## 10. Code-to-Manuscript Mapping

Every manuscript claim must trace to a specific code artifact.

| Manuscript claim | Code artifact | Verification |
|---|---|---|
| MC recovery | mc_recovery_results.csv + summary | Table 3 |
| Cox coefficients | model_params.json | Table 4 |
| PH diagnostics | training output PH report | Table 5, Figure 6 |
| Weibull validation | reports/zetor_survival_audit.json | Table 6, Figure 4 |
| HADCO benchmark | reports/hadco_service_audit.json | Table 6 |
| Cluster robustness | 5 regression tests (tests/test_cluster_robust_cox.py) | Table 5 |
| Bayesian calibration | model_params.json training_meta | Table 4, Appendix S7 |
| PeakLoad calibration | tum_peakload_stats.json | Table 1 |
| Instrument strength | mc_recovery_results.csv F_cluster column | Table 3 |
| LR interaction | training output LR test block | Table 5 |

### Reproducibility checklist (Supplementary S11)

- [ ] All 100 MC results in `mc_recovery_results.csv`
- [ ] Frozen seed (42) documented
- [ ] Environment: `requirements.txt` frozen
- [ ] All P0 regression tests pass
- [ ] All benchmark tests pass
- [ ] Final model_params.json archived
- [ ] reports/ directory archived

---

## 11. Limitations (pre-determined, will NOT be hidden)

### 11.1 Methodological limitations

1. **No real insurance claims.** All β estimates are from simulated/hybrid data.
2. **IV exogeneity assumed, not tested.** Rainfall anomaly validity on real data pending.
3. **Causal interpretation deferred.** Current mode is "predictive," not causal.
4. **Simulation-to-reality gap.** DGP may not fully capture real-world complexity.

### 11.2 Statistical limitations

5. **32 clusters < 50.** Below the asymptotic threshold for fully reliable cluster-robust inference (small-G issue distinct from Stock–Yogo weak-IV thresholds). Results should be interpreted conservatively.
6. **PH assumption.** Some covariates may show marginal PH violations; documented but not auto-rejected.
7. **Finite-sample bias in CF-Cox.** Known −2.69% relative bias at n=40,000.
8. **Recurrent minor failures not modeled.** The current implementation does not model recurrent minor failures as time-varying predictors of subsequent major failure in the production Cox specification. Minor failures are generated in the DGP with their own intensity (λ_minor = 0.002 mch⁻¹) and stored as separate events, but the major-claim Cox model treats only major failures as events and does not condition on the minor-failure history. A full recurrent-event specification is deferred to future work.

### 11.3 Data limitations

9. **Zetor/HADCO are Level 1.** Literature-reconstructed, not individual-level raw data.
10. **No censoring in Zetor.** Cannot validate censoring mechanism against published data.
11. **HADCO lacks event-level engine hours.** Cannot validate survival shape directly.
12. **No dynamic PeakLoad.** Static operational load per tractor.
13. **No renewal/selection behavior.** Assumes independent failures.
14. **Limited brand coverage.** Specific to brands in DGP/training data.

### 11.4 What this study does NOT establish (explicit)

> The following are **not established** by this study:
> - Real-world causal effect of PeakLoad on failure
> - Empirical β coefficients for insurance pricing
> - Production-ready premium calculation
> - External validity across regions/brands not in training
> - Insurance market equilibrium behavior

---

## 12. Pre-Anticipated Reviewer Questions

### Reviewer 1: "Where do the β coefficients come from?"

**Prepared answer:**
> β coefficients are estimated on hybrid data (synthetic structural variables + real weather/soil covariates) under the specified DGP. They represent predictive associations under the current simulation framework, not empirical insurance market estimates. Real claims calibration is explicitly deferred to future work (Section 15.3, Limitations §1).

### Reviewer 2: "Why should rainfall anomaly be a valid instrument?"

**Prepared answer:**
> Relevance is demonstrated empirically (F_cluster ≈ 50,000 ≫ 10, Staiger–Stock weak-instrument threshold). Exclusion restriction is partially probed: |corr(Z, x_climate)| = 0.093 and |corr(Z, x_soil)| = 0.081 are below the 0.15 threshold. However, exogeneity of Z with respect to unobserved determinants of failure remains an assumption. Empirical validation requires real longitudinal claims data with observed and unobserved confounders, which is not available in this study (Limitations §2).

### Reviewer 3: "How do you know Weibull shape 1.88 is realistic?"

**Prepared answer:**
> The selected shape is externally benchmarked against independent published data. Fitting Weibull/Exponential/LogNormal/LogLogistic to first-failure data from 70 Zetor tractors (Durczak et al., 2018), the best fit by AIC is Weibull with shape 1.91. The deviation from our DGP value is 1.8%, well within the 20% tolerance (Table 6, Figure 4, Appendix S8).

### Reviewer 4: "Why should a simulation-trained model be useful for insurance?"

**Prepared answer:**
> It should not yet be used for production pricing. This study establishes the methodological framework and demonstrates internal validity (Monte Carlo recovery, consistency diagnostics, external benchmarks). The framework is a necessary precursor to real-claims calibration: without validated methodology, empirical estimation on real data would lack structural identification. The explicit next stage is calibration on proprietary insurance claims (Section 15.3, Limitations §1).

### Reviewer 5: "Is 32 clusters enough for cluster-robust inference?"

**Prepared answer:**
> The 32 clusters (Region × Year × Campaign) is below the asymptotic threshold of ≈50 clusters recommended for reliable cluster-robust standard errors. We note this as a limitation (Limitations §5). However, our regression tests demonstrate the technical correctness of the cluster-robust implementation, and first-stage F-statistics are orders of magnitude above weak-instrument thresholds, suggesting robustness despite the small-G issue.

---

## 13. Freeze Points and Timeline

### Phase 1: Code freeze (DONE)
- [x] All P0 validation closed
- [x] Benchmark layer audit-ready
- [x] Documentation structure complete

### Phase 2: MC completion (IN PROGRESS)
- [ ] Wait for 100 MC runs to complete
- [ ] Populate Table 3 with final numbers
- [ ] Archive mc_recovery_results.csv

### Phase 3: Manuscript specification freeze (DONE — THIS DOCUMENT)
- [x] Title, hypotheses, figures, tables, claims frozen
- [x] No new hypotheses after this point

### Phase 4: Methods section
- [ ] Write Statistical Methods (Section 8)
- [ ] Write Data and Provenance (Section 7)
- [ ] Write Conceptual Framework (Section 6)

### Phase 5: Results section
- [ ] Estimator recovery (Table 3, Figures 2–3)
- [ ] Cox specification (Table 4, Figures 5–6)
- [ ] External benchmarks (Table 6, Figure 4)
- [ ] Actuarial calibration (Figure 7)

### Phase 6: Discussion and Limitations
- [ ] Main findings
- [ ] Comparison with literature
- [ ] "What is NOT proven" subsection
- [ ] Limitations (Section 16)

### Phase 7: Supplementary material
- [ ] S1–S12 appendices
- [ ] TRIPOD checklist (S12)
- [ ] Reproducibility package (S11)

### Phase 8: Final audit
- [ ] Code ↔ manuscript claim mapping verified
- [ ] All "forbidden claims" checked absent
- [ ] All "required disclaimers" present
- [ ] Reviewer package assembled

---

## 14. TRIPOD Compliance Checklist

| TRIPOD Item | Where addressed | Status |
|---|---|---|
| 1. Title | Section 1 | ✅ |
| 2. Abstract | TBD | ⏳ |
| 3. Background | Introduction | ⏳ |
| 4. Objectives | Section 3 | ✅ |
| 5. Study design | Section 7 | ⏳ |
| 6. Setting | Section 7 | ⏳ |
| 7. Participants | Section 7 (provenance matrix) | ✅ |
| 8. Outcome | Section 5 (estimands) | ✅ |
| 9. Predictors | Section 5 | ✅ |
| 10. Sample size | Section 9 (MC design) | ⏳ |
| 11. Model specification | Section 8 (methods) | ⏳ |
| 12. Model performance | Table 3, 5 | ⏳ |
| 13. Validation | Section 7 (criteria) | ✅ |
| 14. Interpretation | Discussion | ⏳ |
| 15. Limitations | Section 16 | ✅ |
| 16. Funding | TBD | ⏳ |
| 17. Competing interests | TBD | ⏳ |
| 18. Data availability | Reproducibility (S11) | ⏳ |
| 19. Code availability | Reproducibility (S11) | ⏳ |
| 20. Supplementary | S1–S12 | ⏳ |

---

## 15. Next Immediate Actions

### This week
1. **Wait for 100 MC runs** (do not write Results yet)
2. **Freeze manuscript_specification.md** (this document)
3. **Freeze codebase** — no changes to Итог.py, train_model.py, prediction_engine.py

### After MC completes
4. Populate Table 3 with final MC numbers
5. Write Methods section (Section 8 of paper)
6. Write Data and Provenance (Section 7)
7. Draft Conceptual Framework and Introduction

### After Methods drafted
8. Write Results (using frozen Table 3)
9. Write Discussion with "What is NOT proven" subsection
10. Assemble Supplementary material
11. Run TRIPOD checklist audit
12. Assemble reviewer package (S11)

---

## 16. Sign-off

This specification is the binding contract between the codebase and the manuscript. Any deviation requires explicit amendment with date and rationale.

**Signed:** [Author]
**Date:** 2026-08-21
**Version:** 1.0 (FROZEN)
**Amendments:** None