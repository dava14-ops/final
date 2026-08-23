# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI/CD pipeline with GitHub Actions
- Automated testing on Python 3.10, 3.11, 3.12
- Code coverage reporting with Codecov
- Security scanning with Bandit and Safety
- Pre-commit hooks for code quality
- Makefile for common development tasks
- Nightly test suite for comprehensive testing

### Changed
- Standardized code formatting with Ruff and Black
- Added type hints and mypy checking

## [0.2.0] - 2026-08-15

### Added
- Severity model integration with premium engine
- Model versioning system (v{major}.{minor}_{segment}_{date})
- Recalibration triggers and monitoring
- REST API with FastAPI
- Comprehensive documentation (README, architecture, API reference)
- Integration and consistency test suites
- Kaplan-Meier validation for baseline survival

### Changed
- Refactored prediction engine for better maintainability
- Improved error handling and validation
- Updated documentation to reflect current architecture

## [0.1.0] - 2026-08-01

### Added
- Initial CF Cox / IV-Cox model implementation
- Data generation process (DGP) for simulation
- First stage and control function estimation
- Basic premium calculation engine
- CLI interface for model training and prediction
- Initial test coverage

[Unreleased]: https://github.com/your-org/cf-cox-insurance/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/your-org/cf-cox-insurance/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/your-org/cf-cox-insurance/releases/tag/v0.1.0
