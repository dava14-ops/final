"""
Regression tests для HADCO adapter.

Проверяет:
1. HADCO создаёт 40 тракторов и 1670 work orders.
2. Provenance = literature_reconstructed Level 1.
3. HADCO не является real claims.
4. HADCO не является individual-level dataset.
5. Survival export запрещён.
6. Recurrent-event dataframe корректен.
7. Cost dataframe корректен.
8. Есть repair и maintenance события.
"""
import pytest
import numpy as np
import pandas as pd

from benchmarks.canonical_schema import CanonicalFleetDataset, DataProvenance
from benchmarks.adapters.hadco_adapter import (
    load_hadco_dataset,
    generate_synthetic_hadco,
    to_recurrent_event_dataframe,
    to_cost_dataframe,
)


class TestHadcoAdapter:
    """HADCO adapter regression tests."""

    def test_default_generation_size(self):
        """HADCO reconstruction создаёт 40 тракторов и 1670 WJO."""
        ds = generate_synthetic_hadco(seed=42)

        assert isinstance(ds, CanonicalFleetDataset)
        assert ds.source == "hadco"
        assert ds.n_tractors == 40
        assert ds.n_events == 1670
        assert ds.n_windows == 40

    def test_provenance_level(self):
        """HADCO — Level 1 literature reconstructed, не real claims."""
        ds = load_hadco_dataset(seed=42)
        report = ds.provenance_report()

        assert ds.provenance == DataProvenance.LITERATURE_RECONSTRUCTED
        assert report["provenance"] == "literature_reconstructed"
        assert report["provenance_level"] == 1
        assert report["is_real_claims"] is False
        assert report["is_individual_level"] is False

    def test_survival_export_blocked(self):
        """HADCO нельзя экспортировать как survival dataframe."""
        ds = load_hadco_dataset(seed=42)

        with pytest.raises(ValueError, match="not survival-compatible"):
            ds.to_survival_dataframe()

    def test_recurrent_event_dataframe_structure(self):
        """Recurrent-event dataframe имеет одну строку на work order."""
        ds = load_hadco_dataset(seed=42)
        df = to_recurrent_event_dataframe(ds)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1670

        required = {
            "tractor_id",
            "event_id",
            "event_time",
            "event_type",
            "category",
            "power_kw",
            "parts_cost",
            "labor_cost",
            "total_cost",
        }
        assert required.issubset(df.columns)

        assert (df["event_time"] >= 0).all()
        assert (df["power_kw"] > 0).all()

    def test_cost_dataframe_structure(self):
        """Cost dataframe содержит non-negative cost breakdown."""
        ds = load_hadco_dataset(seed=42)
        df = to_cost_dataframe(ds)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1670

        required = {
            "event_id",
            "tractor_id",
            "category",
            "total_cost",
            "parts_cost",
            "labor_cost",
            "power_kw",
        }
        assert required.issubset(df.columns)

        assert (df["total_cost"] >= 0).all()
        assert (df["parts_cost"] >= 0).all()
        assert (df["labor_cost"] >= 0).all()

        # total_cost должен примерно равняться parts + labor
        diff = np.abs(df["total_cost"] - (df["parts_cost"] + df["labor_cost"]))
        assert diff.max() < 1e-6

    def test_repair_and_maintenance_present(self):
        """В HADCO должны быть repair и maintenance work orders."""
        ds = load_hadco_dataset(seed=42)
        df = to_recurrent_event_dataframe(ds)

        event_types = set(df["event_type"].unique())

        assert "repair" in event_types
        assert "maintenance" in event_types

        repair_share = (df["event_type"] == "repair").mean()
        maintenance_share = (df["event_type"] == "maintenance").mean()

        assert 0.30 <= repair_share <= 0.70
        assert 0.30 <= maintenance_share <= 0.70

    def test_costs_positive_and_skewed(self):
        """Repair costs должны быть положительными и правоскошенными."""
        ds = load_hadco_dataset(seed=42)
        df = to_cost_dataframe(ds)

        assert df["total_cost"].mean() > 0
        assert df["total_cost"].median() > 0

        # Для repair/cost данных среднее обычно выше медианы
        assert df["total_cost"].mean() >= df["total_cost"].median()

    def test_load_without_csv_falls_back_to_literature_reconstruction(self):
        """load_hadco_dataset без CSV возвращает literature reconstruction."""
        ds = load_hadco_dataset(csv_path=None, seed=42)

        assert ds.source == "hadco"
        assert ds.provenance == DataProvenance.LITERATURE_RECONSTRUCTED
        assert ds.source_metadata.get("synthetic") is True
        assert ds.source_metadata.get("survival_compatible") is False
        assert ds.source_metadata.get("recurrent_event_compatible") is True
        assert ds.source_metadata.get("cost_compatible") is True