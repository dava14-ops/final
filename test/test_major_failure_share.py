"""
P0-5 regression tests: Bayesian major failure calibration.

Проверяет:
1. Корректность conjugate Beta update.
2. Наличие raw + posterior brand estimates.
3. Shrinkage редкого бренда к prior mean.
4. Корректный prior-only fallback при отсутствии events.
"""
import numpy as np
import pandas as pd
import pytest

try:
    import train_model
    HAS_TRAIN_MODEL = True
except ImportError:
    HAS_TRAIN_MODEL = False


def _make_test_data(n_major=76, n_total=250, n_brands=4, seed=42):
    """Создаёт синтетические данные для тестирования calibration."""
    rng = np.random.default_rng(seed)

    # Распределяем events по брендам
    brand_codes = rng.integers(0, n_brands, size=n_total)
    failure_types = np.array(
        ["major"] * n_major + ["minor"] * (n_total - n_major)
    )
    rng.shuffle(failure_types)

    # Все observations (events + censored)
    n_obs = n_total * 2  # вдвое больше наблюдений, чем events
    all_brands = np.concatenate([
        brand_codes,
        rng.integers(0, n_brands, size=n_obs - n_total),
    ])
    all_events = np.concatenate([
        np.ones(n_total, dtype=int),
        np.zeros(n_obs - n_total, dtype=int),
    ])
    all_failure_types = np.concatenate([
        failure_types,
        np.array(["censored"] * (n_obs - n_total)),
    ])

    data = pd.DataFrame({
        "event": all_events,
        "failure_type": all_failure_types,
        "Brand": all_brands,
        "brand_code": all_brands,
    })

    return data


@pytest.mark.skipif(not HAS_TRAIN_MODEL, reason="train_model not importable")
class TestMajorFailureCalibration:
    """P0-5: Bayesian major failure calibration."""

    def test_conjugate_beta_update(self):
        """Conjugate Beta update даёт корректный posterior."""
        data = _make_test_data(n_major=76, n_total=250)

        result = train_model._compute_major_failure_calibration(
            data=data,
            prior_mean=0.30,
            prior_effective_n=30.0,
        )

        overall = result["overall"]

        # Prior: Beta(9, 21)
        assert overall["prior_alpha"] == pytest.approx(9.0)
        assert overall["prior_beta"] == pytest.approx(21.0)
        assert overall["prior_mean"] == pytest.approx(0.30)
        assert overall["prior_effective_n"] == pytest.approx(30.0)

        # Posterior: Beta(9 + 76, 21 + 174) = Beta(85, 195)
        assert overall["posterior_alpha"] == pytest.approx(85.0)
        assert overall["posterior_beta"] == pytest.approx(195.0)

        # Posterior mean = 85 / 280 ≈ 0.30357
        expected_mean = 85.0 / 280.0
        assert overall["posterior_mean"] == pytest.approx(expected_mean, abs=1e-6)

        # Observed share = 76 / 250 = 0.304
        assert overall["observed_share"] == pytest.approx(0.304, abs=1e-6)

        # CrI должен содержать posterior mean
        assert overall["ci_low"] < overall["posterior_mean"] < overall["ci_high"]

    def test_brand_estimates_present(self):
        """Brand-level estimates присутствуют и корректны."""
        data = _make_test_data(n_major=76, n_total=250, n_brands=4)

        result = train_model._compute_major_failure_calibration(
            data=data,
            prior_mean=0.30,
            prior_effective_n=30.0,
        )

        # Все три brand-структуры должны присутствовать
        assert "by_brand" in result
        assert "by_brand_observed" in result
        assert "by_brand_posterior" in result

        # Должны быть оценки для всех брендов
        assert len(result["by_brand"]) > 0
        assert len(result["by_brand_observed"]) > 0
        assert len(result["by_brand_posterior"]) > 0

        # Ключи должны совпадать
        assert set(result["by_brand"].keys()) == set(result["by_brand_observed"].keys())
        assert set(result["by_brand"].keys()) == set(result["by_brand_posterior"].keys())

    def test_rare_brand_shrinkage(self):
        """Редкий бренд shrinked к prior mean сильнее, чем частый."""
        # Создаём данные: бренд 0 — частый (100 events), бренд 3 — редкий (5 events)
        rng = np.random.default_rng(123)

        # Бренд 0: 100 events, 30 major (share = 0.30)
        brand_0_events = 100
        brand_0_major = 30

        # Бренд 3: 5 events, 4 major (raw share = 0.80, но shrinked к 0.30)
        brand_3_events = 5
        brand_3_major = 4

        n_total = brand_0_events + brand_3_events
        n_major = brand_0_major + brand_3_major

        brands = np.concatenate([
            np.zeros(brand_0_events, dtype=int),
            np.full(brand_3_events, 3, dtype=int),
        ])
        failure_types = np.array(
            ["major"] * brand_0_major + ["censored"] * (brand_0_events - brand_0_major) +
            ["major"] * brand_3_major + ["censored"] * (brand_3_events - brand_3_major)
        )

        data = pd.DataFrame({
            "event": np.ones(n_total, dtype=int),
            "failure_type": failure_types,
            "Brand": brands,
            "brand_code": brands,
        })

        result = train_model._compute_major_failure_calibration(
            data=data,
            prior_mean=0.30,
            prior_effective_n=30.0,
        )

        # Редкий бренд (3) должен быть shrinked к prior mean
        # Raw share = 4/5 = 0.80
        # Posterior mean = (9 + 4) / (30 + 5) = 13/35 ≈ 0.371
        # Это ближе к 0.30, чем 0.80
        brand_3_posterior = result["by_brand"]["3"]
        brand_3_observed = result["by_brand_observed"]["3"]

        assert brand_3_observed == pytest.approx(0.80, abs=0.01)
        assert brand_3_posterior < brand_3_observed  # shrinkage к prior
        assert brand_3_posterior > 0.30  # но не ниже prior (данные тянут вверх)

        # Частый бренд (0) должен быть ближе к observed share
        brand_0_posterior = result["by_brand"]["0"]
        brand_0_observed = result["by_brand_observed"]["0"]

        # Для частого бренда posterior ближе к observed, чем для редкого
        brand_0_shrinkage = abs(brand_0_posterior - brand_0_observed)
        brand_3_shrinkage = abs(brand_3_posterior - brand_3_observed)
        assert brand_0_shrinkage < brand_3_shrinkage

    def test_prior_only_fallback(self):
        """При отсутствии events posterior равен prior."""
        # Пустые данные (нет events)
        data = pd.DataFrame({
            "event": np.zeros(10, dtype=int),
            "failure_type": np.array(["censored"] * 10),
            "Brand": np.zeros(10, dtype=int),
            "brand_code": np.zeros(10, dtype=int),
        })

        result = train_model._compute_major_failure_calibration(
            data=data,
            prior_mean=0.30,
            prior_effective_n=30.0,
        )

        overall = result["overall"]

        # Posterior должен быть равен prior (k=0, n=0)
        assert overall["posterior_alpha"] == pytest.approx(9.0)
        assert overall["posterior_beta"] == pytest.approx(21.0)
        assert overall["posterior_mean"] == pytest.approx(0.30)
        assert overall["n_events"] == 0
        assert overall["n_major"] == 0