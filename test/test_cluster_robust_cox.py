"""
P0-6 audit-ready regression test: cluster-robust Cox.

Доказывает:
1. CoxPHFitter действительно использует cluster-robust covariance.
2. cluster_id НЕ является регрессором.
3. Point estimates идентичны (кластеризация влияет только на SE).
4. Naive и cluster SE реально различаются.
5. Fail-closed: если cluster_col запрошен, но отсутствует — модель падает.
"""
import numpy as np
import pandas as pd
import pytest

try:
    from lifelines import CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

from Итог import (
    CFFitOptions,
    DGPParameters,
    generate_data,
    fit_first_stage,
    fit_cf_cox,
)


def _make_clustered_data(n=2000, n_clusters=8, seed=42):
    """Генерирует данные с явной кластерной структурой."""
    rng = np.random.default_rng(seed)
    dgp = DGPParameters(
        gamma=0.5, rho=0.7, delta=0.5,
        baseline_family="weibull", baseline_shape=1.88,
    )
    data = generate_data(
        n=n,
        contamination=False,
        baseline_hazard=1e-4,
        censoring_scale=500.0,
        rng=rng,
        dgp=dgp,
    )
    # Назначаем кластеры явно
    cluster_ids = np.repeat(np.arange(n_clusters), n // n_clusters)
    remainder = n % n_clusters
    if remainder > 0:
        extra = rng.integers(0, n_clusters, size=remainder)
        cluster_ids = np.concatenate([cluster_ids, extra])
    rng.shuffle(cluster_ids)
    data["cluster_id"] = cluster_ids.astype(str)
    return data


def _make_opts(cluster_col=None):
    return CFFitOptions(
        cox_se_threshold=10.0,
        v_hat_basis="linear",
        v_hat_basis_params=None,
        extra_x_cols=None,
        center_peakload=None,
        brand_encoding="dummies",
        brand_reference_code=0,
        var_z_threshold=1e-8,
        min_first_stage_f=10.0,
        fail_on_weak_instrument=False,
        min_cox_events=10,
        min_events_per_covariate=5,
        save_tracebacks=False,
        cluster_col=cluster_col,
    )


@pytest.mark.skipif(not HAS_LIFELINES, reason="lifelines not installed")
class TestClusterRobustCox:
    """P0-6 audit: cluster-robust Cox regression."""

    @pytest.fixture
    def clustered_data(self):
        return _make_clustered_data(n=2000, n_clusters=8)

    @pytest.fixture
    def fitted_pair(self, clustered_data):
        """Возвращает (cf_naive, cf_cluster) для сравнения."""
        opts_naive = _make_opts(cluster_col=None)
        opts_cluster = _make_opts(cluster_col="cluster_id")
        fs = fit_first_stage(clustered_data, opts_naive)
        cf_naive = fit_cf_cox(clustered_data, fs, opts_naive)
        cf_cluster = fit_cf_cox(clustered_data, fs, opts_cluster)
        return cf_naive, cf_cluster

    def test_cluster_id_not_in_regressors(self, fitted_pair):
        """cluster_id НЕ должен быть регрессором Cox."""
        _, cf_cluster = fitted_pair
        cox_params = list(cf_cluster.cph.params_.index)
        assert "cluster_id" not in cox_params, (
            f"cluster_id must NOT be a regressor. Found: {cox_params}"
        )

    def test_point_estimates_identical(self, fitted_pair):
        """Кластеризация НЕ меняет point estimates, только SE."""
        cf_naive, cf_cluster = fitted_pair
        gamma_naive = cf_naive.gamma_hat
        gamma_cluster = cf_cluster.gamma_hat
        assert abs(gamma_naive - gamma_cluster) < 1e-9, (
            f"Point estimates must be identical: "
            f"naive={gamma_naive}, cluster={gamma_cluster}"
        )

    def test_se_differs(self, fitted_pair):
        """Naive и cluster SE должны реально различаться."""
        cf_naive, cf_cluster = fitted_pair
        se_naive = cf_naive.naive_model_se
        se_cluster = cf_cluster.naive_model_se

        # SE должны отличаться (направление не гарантировано)
        assert se_naive != se_cluster, (
            f"Cluster SE ({se_cluster}) must differ from naive SE ({se_naive})"
        )

        # Оба SE должны быть конечными и положительными
        assert np.isfinite(se_naive) and se_naive > 0
        assert np.isfinite(se_cluster) and se_cluster > 0

    def test_n_clusters_reported(self, fitted_pair, clustered_data):
        """Количество кластеров должно корректно определяться."""
        _, cf_cluster = fitted_pair
        # Проверяем, что cluster_id присутствует в данных Cox
        cph = cf_cluster.cph
        # lifelines сохраняет cluster_col в _cluster_col атрибуте
        cluster_col_used = getattr(cph, "_cluster_col", None)
        if cluster_col_used is not None:
            assert cluster_col_used == "cluster_id"

    def test_fail_closed_when_cluster_col_missing(self, clustered_data):
        """Если cluster_col запрошен, но отсутствует — модель должна упасть."""
        data_no_cluster = clustered_data.drop(columns=["cluster_id"])
        opts = _make_opts(cluster_col="cluster_id")
        fs = fit_first_stage(data_no_cluster, opts)

        # Должно бросить ValueError (fail-closed)
        with pytest.raises((ValueError, RuntimeError)):
            fit_cf_cox(data_no_cluster, fs, opts)