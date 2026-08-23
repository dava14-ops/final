# Changelog

## [0.2.0] - 2026-08-20

### Added
- P0-3: LR test for Age × Hours interaction (LR=29.19, p<1e-9)
- P0-4: Structured PH diagnostics report (Schoenfeld residuals)
- P0-5: Bayesian major-failure share calibration (Beta-Binomial)
- P0-6: Cluster-robust Cox SE (lifelines cluster_col)
- External benchmark layer: Zetor survival + HADCO service
- DataProvenance enum (4 levels: synthetic → proprietary_claims)
- Survival guard: HADCO blocked from Cox survival export
- benchmarks/canonical_schema.py with Pydantic validation
- benchmarks/adapters/zetor_adapter.py (13 tests)
- benchmarks/adapters/hadco_adapter.py (8 tests)
- benchmarks/zetor_benchmark.py (Weibull shape validation)
- benchmarks/hadco_benchmark.py (recurrent/cost analysis)
- docs/model_card.md with full provenance table
- docs/assumption_log.md with 8 documented assumptions
- docs/validation_report.md with P0 + benchmark results
- docs/provenance_guide.md with 4-level data classification
- docs/statistical_methodology.md with CF-Cox methodology

### Changed
- major_failure_share: expert constant 0.30 → Bayesian posterior 0.0296
- Cox SE: naive → cluster-robust (32 clusters)
- IV mode: explicitly set to "predictive" (not causal)
- PH diagnostics: from Optional[str] to structured dict report

### Validated
- Weibull shape=1.88 confirmed by Zetor data (deviation 1.8%)
- MTTF/MTBF ratio = 0.184 (plausible range [0.1, 0.6])
- Interaction Age×Hours: LR=29.19, β=+0.173, p<1e-9
- Cluster-robust Cox: 5/5 regression tests passed
- MC recovery (preliminary): bias=−3.2%, RMSE=0.029

### Not Changed
- DGP parameters (MC recovery in progress)
- Core estimation pipeline (Итог.py)
- Production prediction engine
- Real_calculator.py
