# tests/test_zetor_adapter.py
"""
Regression tests для Zetor adapter.

Проверяет:
1. Синтетическая генерация создаёт валидный CanonicalFleetDataset.
2. MTTF синтетики близок к опубликованному (271 mth).
3. Survival DataFrame имеет правильную структуру.
4. Все 70 observations — events (нет censoring, как в статье).
5. Assembly groups и модели в ожидаемых диапазонах.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.canonical_schema import CanonicalFleetDataset
from benchmarks.adapters.zetor_adapter import (
    generate_synthetic_zetor,
    load_zetor_dataset,
    ZETOR_PUBLISHED_MTTF,
    ZETOR_WEIBULL_SHAPE,
    ZETOR_MODELS,
    ZETOR_ASSEMBLY_GROUPS,
)


class TestZetorAdapter:
    """Zetor adapter regression tests."""

    def test_synthetic_generation_default_size(self):
        """Синтетика создаёт 70 тракторов (как в статье)."""
        ds = generate_synthetic_zetor(seed=42)

        assert isinstance(ds, CanonicalFleetDataset)
        assert ds.source == "zetor"
        assert ds.n_tractors == 70
        assert ds.n_events == 70
        assert ds.n_windows == 70

    def test_all_observations_are_events(self):
        """
        Zetor: все observations — observed first failures (нет censoring).
        Это соответствует оригинальной статье.
        """
        ds = generate_synthetic_zetor(seed=42)
        df = ds.to_survival_dataframe()

        assert (df["event"] == 1).all(), (
            "Zetor should have no censored observations"
        )
        assert df["event"].sum() == 70

    def test_mttf_close_to_published(self):
        """
        MTTF синтетики близок к опубликованному 271 mth.
        Допускаем 10% отклонения из-за стохастичности.
        """
        # Используем больше тракторов для уменьшения variance
        ds = generate_synthetic_zetor(n_tractors=500, seed=42)
        df = ds.to_survival_dataframe()

        mttf = df["time"].mean()
        relative_error = abs(mttf - ZETOR_PUBLISHED_MTTF) / ZETOR_PUBLISHED_MTTF

        assert relative_error < 0.10, (
            f"MTTF {mttf:.1f} differs from published {ZETOR_PUBLISHED_MTTF} "
            f"by {relative_error*100:.1f}% (expected < 10%)"
        )

    def test_weibull_shape_preserved(self):
        """Проверяем, что synthetic данные следуют Weibull с shape ≈ 1.88."""
        pytest.importorskip("lifelines")
        from lifelines import WeibullFitter

        ds = generate_synthetic_zetor(n_tractors=500, seed=42)
        df = ds.to_survival_dataframe()

        wf = WeibullFitter()
        wf.fit(df["time"], event_observed=df["event"])

        fitted_shape = wf.rho_  # lifelines notation
        relative_error = abs(fitted_shape - ZETOR_WEIBULL_SHAPE) / ZETOR_WEIBULL_SHAPE

        assert relative_error < 0.15, (
            f"Fitted Weibull shape {fitted_shape:.2f} differs from "
            f"target {ZETOR_WEIBULL_SHAPE} by {relative_error*100:.1f}%"
        )

    def test_models_within_power_ranges(self):
        """Мощность каждой модели в опубликованном диапазоне."""
        ds = generate_synthetic_zetor(n_tractors=200, seed=42)

        for tractor in ds.tractors:
            spec = ZETOR_MODELS[tractor.model]
            assert spec["power_min_kw"] <= tractor.power_kw <= spec["power_max_kw"], (
                f"{tractor.model} power {tractor.power_kw} outside "
                f"[{spec['power_min_kw']}, {spec['power_max_kw']}]"
            )

    def test_assembly_groups_valid(self):
        """Все компоненты из списка ZETOR_ASSEMBLY_GROUPS."""
        ds = generate_synthetic_zetor(seed=42)
        df = ds.to_survival_dataframe()

        components = set(df["component"].dropna().unique())
        assert components.issubset(set(ZETOR_ASSEMBLY_GROUPS)), (
            f"Unexpected components: {components - set(ZETOR_ASSEMBLY_GROUPS)}"
        )

    def test_survival_dataframe_structure(self):
        """Survival DataFrame имеет обязательные колонки."""
        ds = generate_synthetic_zetor(seed=42)
        df = ds.to_survival_dataframe()

        required = {"tractor_id", "time", "event", "brand", "model", "power_kw"}
        assert required.issubset(df.columns)

        # Типы данных
        assert df["time"].dtype in (np.float64, np.float32)
        assert df["event"].dtype in (np.int64, np.int32, int)
        assert (df["time"] > 0).all(), "Survival times must be positive"

    def test_reproducibility_with_seed(self):
        """Одинаковый seed даёт одинаковый датасет."""
        ds1 = generate_synthetic_zetor(seed=123)
        ds2 = generate_synthetic_zetor(seed=123)

        df1 = ds1.to_survival_dataframe().sort_values("tractor_id").reset_index(drop=True)
        df2 = ds2.to_survival_dataframe().sort_values("tractor_id").reset_index(drop=True)

        assert df1["time"].equals(df2["time"]), "Seed should ensure reproducibility"
        assert df1["model"].equals(df2["model"])
        assert df1["component"].equals(df2["component"])

    def test_different_seeds_differ(self):
        """Разные seeds дают разные датасеты."""
        ds1 = generate_synthetic_zetor(seed=1)
        ds2 = generate_synthetic_zetor(seed=2)

        df1 = ds1.to_survival_dataframe()
        df2 = ds2.to_survival_dataframe()

        # Времена должны отличаться (не идентичны)
        assert not df1["time"].equals(df2["time"]), (
            "Different seeds should produce different datasets"
        )

    def test_load_without_csv_falls_back_to_synthetic(self):
        """load_zetor_dataset без CSV возвращает синтетику."""
        ds = load_zetor_dataset(csv_path=None, seed=42)
        assert ds.source == "zetor"
        assert ds.source_metadata.get("synthetic") is True

    def test_load_with_nonexistent_csv_falls_back(self):
        """load_zetor_dataset с несуществующим CSV fallback на синтетику."""
        ds = load_zetor_dataset(
            csv_path=Path("/nonexistent/zetor.csv"),
            seed=42,
        )
        assert ds.source == "zetor"
        assert ds.source_metadata.get("synthetic") is True

    def test_summary_contains_expected_keys(self):
        """summary() возвращает полную сводку."""
        ds = generate_synthetic_zetor(seed=42)
        summary = ds.summary()

        expected_keys = {
            "source", "n_tractors", "n_events", "n_censored",
            "event_rate", "mttf_empirical", "n_components",
        }
        assert expected_keys.issubset(summary.keys())
        assert summary["source"] == "zetor"
        assert summary["event_rate"] == 1.0  # Zetor: все observed
        assert summary["n_censored"] == 0

        def test_published_aggregate_recovery(self):
            """
            Published aggregate recovery test.

            Проверяет, что synthetic reconstruction воспроизводит
            ВСЕ опубликованные агрегаты, а не только MTTF.
            """
            ds = generate_synthetic_zetor(n_tractors=500, seed=42)
            df = ds.to_survival_dataframe()

            # Published aggregates из Durczak et al. (2018)
            published = {
                "n_tractors": 70,
                "mttf_mth": 271.0,
                "lambda_mth": 1.0 / 271.0,  # ≈ 0.00369
                "n_models": 4,  # Proxima, Proxima Power, Proxima Plus, Forterra
                "power_min_kw": 45.0,
                "power_max_kw": 90.0,
                "n_assembly_groups": 29,
            }

            # Проверка MTTF
            mttf = df["time"].mean()
            assert abs(mttf - published["mttf_mth"]) / published["mttf_mth"] < 0.15, (
                f"MTTF {mttf:.1f} differs from published {published['mttf_mth']} "
                f"by {abs(mttf - published['mttf_mth']) / published['mttf_mth'] * 100:.1f}%"
            )

            # Проверка количества моделей
            n_models = df["model"].nunique()
            assert n_models == published["n_models"], (
                f"Expected {published['n_models']} models, got {n_models}"
            )

            # Проверка диапазона мощности
            power_min = df["power_kw"].min()
            power_max = df["power_kw"].max()
            assert power_min >= published["power_min_kw"] * 0.9, (
                f"Power min {power_min} below published {published['power_min_kw']}"
            )
            assert power_max <= published["power_max_kw"] * 1.1, (
                f"Power max {power_max} above published {published['power_max_kw']}"
            )

            # Проверка количества assembly groups
            n_components = df["component"].nunique()
            assert n_components >= 10, (  # Минимум 10 из 29
                f"Too few components: {n_components}"
            )

            # Проверка lambda (hazard rate)
            lambda_emp = 1.0 / mttf
            lambda_deviation = abs(lambda_emp - published["lambda_mth"]) / published["lambda_mth"]
            assert lambda_deviation < 0.15, (
                f"Lambda {lambda_emp:.5f} differs from published {published['lambda_mth']:.5f}"
            )

    def test_citation_present(self):
        """Citation сохраняется в dataset."""
        ds = generate_synthetic_zetor(seed=42)
        assert ds.citation is not None
        assert "Durczak" in ds.citation