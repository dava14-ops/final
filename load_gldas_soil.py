#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_soil_index.py

Фаза Soil IV
============

Преобразование ежедневной влажности почвы GLDAS-2.1
в регионально-временные показатели для сельскохозяйственных кампаний.

Input:
    data/raw/soil/gldas_soil_daily.csv

Output:
    data/processed/soil/soil_windows.csv

Кампании согласованы с weather_iv.py:

    sowing:
        15 апреля — 15 июня

    harvest:
        1 августа — 15 октября

Основной показатель:

    soil_moisture_mean_0_10cm

Дополнительно:

    soil_moisture_median_0_10cm
    soil_moisture_std_0_10cm
    soil_moisture_min_0_10cm
    soil_moisture_max_0_10cm
    soil_index

ВАЖНО:

soil_index здесь является эмпирическим индексом состояния почвы,
а не ещё одним синтетическим фактором.

Он строится внутри каждого региона относительно распределения
этого региона за доступный период.

Формула:

    soil_index =
        (campaign_mean - regional_mean)
        / regional_std

То есть:

    soil_index > 0
        более влажно обычного

    soil_index < 0
        суше обычного

    soil_index ≈ 0
        около региональной нормы
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# PATHS
# ============================================================================

INPUT_PATH = Path(
    "data/raw/soil/gldas_soil_daily.csv"
)

OUTPUT_DIR = Path(
    "data/processed/soil"
)

OUTPUT_PATH = OUTPUT_DIR / "soil_windows.csv"


# ============================================================================
# CAMPAIGNS
# ============================================================================

CAMPAIGN_WINDOWS: Dict[str, Dict[str, str]] = {
    "sowing": {
        "start": "04-15",
        "end": "06-15",
    },
    "harvest": {
        "start": "08-01",
        "end": "10-15",
    },
}


EXPECTED_REGIONS = [
    "volga",
    "north_caucasus",
    "northwest",
    "central_chernozem",
    "kuban",
    "altai",
    "amur",
    "vladimir",
]


EXPECTED_YEARS = [
    2022,
    2023,
    2024,
    2025,
]


SOIL_COLUMN = "soil_moisture_0_10cm"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_input() -> pd.DataFrame:
    """Load and validate raw GLDAS data."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "date",
        "region_code",
        "year",
        SOIL_COLUMN,
        "source",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    ).astype("Int64")

    df[SOIL_COLUMN] = pd.to_numeric(
        df[SOIL_COLUMN],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "date",
            "region_code",
            "year",
            SOIL_COLUMN,
        ]
    ).copy()

    df["year"] = df["year"].astype(int)

    # Soil moisture cannot be negative.
    df.loc[
        df[SOIL_COLUMN] < 0,
        SOIL_COLUMN,
    ] = np.nan

    df = df.dropna(
        subset=[SOIL_COLUMN]
    )

    return df


# ============================================================================
# CAMPAIGN EXTRACTION
# ============================================================================

def get_campaign_data(
    df: pd.DataFrame,
    region_code: str,
    year: int,
    campaign: str,
) -> pd.DataFrame:
    """Return rows belonging to one region/year/campaign."""

    window = CAMPAIGN_WINDOWS[campaign]

    start_date = pd.Timestamp(
        f"{year}-{window['start']}"
    )

    end_date = pd.Timestamp(
        f"{year}-{window['end']}"
    )

    mask = (
        (df["region_code"] == region_code)
        & (df["year"] == year)
        & (df["date"] >= start_date)
        & (df["date"] <= end_date)
    )

    return df.loc[mask].copy()


# ============================================================================
# REGIONAL BASELINE
# ============================================================================

def compute_regional_baseline(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute regional baseline.

    Baseline is calculated using all available daily observations
    for the region and available years.

    This avoids comparing wet regions such as Kuban directly with
    naturally drier regions such as Volga.
    """

    baseline = (
        df.groupby("region_code")[SOIL_COLUMN]
        .agg(
            regional_mean="mean",
            regional_std="std",
            regional_median="median",
            regional_min="min",
            regional_max="max",
            regional_n="count",
        )
        .reset_index()
    )

    # Constant-value regions are theoretically possible.
    baseline["regional_std"] = baseline[
        "regional_std"
    ].replace(0.0, np.nan)

    return baseline


# ============================================================================
# SOIL INDEX
# ============================================================================

def calculate_soil_index(
    campaign_mean: float,
    regional_mean: float,
    regional_std: float,
) -> float:
    """
    Standardized soil-moisture index.

    No synthetic fallback is used.

    If regional_std is unavailable, return NaN.
    """
    if not np.isfinite(campaign_mean):
        return float("nan")

    if not np.isfinite(regional_mean):
        return float("nan")

    if not np.isfinite(regional_std):
        return float("nan")

    if regional_std <= 0:
        return float("nan")

    return float(
        (campaign_mean - regional_mean)
        / regional_std
    )


# ============================================================================
# ONE CAMPAIGN
# ============================================================================

def compute_campaign_window(
    df: pd.DataFrame,
    baseline: pd.DataFrame,
    region_code: str,
    year: int,
    campaign: str,
) -> dict:
    """Compute one region/year/campaign record."""

    campaign_df = get_campaign_data(
        df,
        region_code,
        year,
        campaign,
    )

    total_window_days = int(
        (
            pd.Timestamp(
                f"{year}-{CAMPAIGN_WINDOWS[campaign]['end']}"
            )
            - pd.Timestamp(
                f"{year}-{CAMPAIGN_WINDOWS[campaign]['start']}"
            )
        ).days
        + 1
    )

    baseline_row = baseline[
        baseline["region_code"] == region_code
    ]

    if baseline_row.empty:
        return {
            "region_code": region_code,
            "year": year,
            "campaign": campaign,
            "soil_moisture_mean_0_10cm": np.nan,
            "soil_moisture_median_0_10cm": np.nan,
            "soil_moisture_std_0_10cm": np.nan,
            "soil_moisture_min_0_10cm": np.nan,
            "soil_moisture_max_0_10cm": np.nan,
            "regional_mean_0_10cm": np.nan,
            "regional_std_0_10cm": np.nan,
            "soil_index": np.nan,
            "observations": 0,
            "total_window_days": total_window_days,
            "source": "gldas_2.1",
        }

    base = baseline_row.iloc[0]

    if campaign_df.empty:
        return {
            "region_code": region_code,
            "year": year,
            "campaign": campaign,
            "soil_moisture_mean_0_10cm": np.nan,
            "soil_moisture_median_0_10cm": np.nan,
            "soil_moisture_std_0_10cm": np.nan,
            "soil_moisture_min_0_10cm": np.nan,
            "soil_moisture_max_0_10cm": np.nan,
            "regional_mean_0_10cm": float(
                base["regional_mean"]
            ),
            "regional_std_0_10cm": float(
                base["regional_std"]
            ),
            "soil_index": np.nan,
            "observations": 0,
            "total_window_days": total_window_days,
            "source": "gldas_2.1",
        }

    values = campaign_df[
        SOIL_COLUMN
    ].dropna()

    if values.empty:
        return {
            "region_code": region_code,
            "year": year,
            "campaign": campaign,
            "soil_moisture_mean_0_10cm": np.nan,
            "soil_moisture_median_0_10cm": np.nan,
            "soil_moisture_std_0_10cm": np.nan,
            "soil_moisture_min_0_10cm": np.nan,
            "soil_moisture_max_0_10cm": np.nan,
            "regional_mean_0_10cm": float(
                base["regional_mean"]
            ),
            "regional_std_0_10cm": float(
                base["regional_std"]
            ),
            "soil_index": np.nan,
            "observations": 0,
            "total_window_days": total_window_days,
            "source": "gldas_2.1",
        }

    campaign_mean = float(values.mean())

    soil_index = calculate_soil_index(
        campaign_mean,
        float(base["regional_mean"]),
        float(base["regional_std"]),
    )

    return {
        "region_code": region_code,
        "year": year,
        "campaign": campaign,
        "soil_moisture_mean_0_10cm": campaign_mean,
        "soil_moisture_median_0_10cm": float(
            values.median()
        ),
        "soil_moisture_std_0_10cm": float(
            values.std()
        ),
        "soil_moisture_min_0_10cm": float(
            values.min()
        ),
        "soil_moisture_max_0_10cm": float(
            values.max()
        ),
        "regional_mean_0_10cm": float(
            base["regional_mean"]
        ),
        "regional_std_0_10cm": float(
            base["regional_std"]
        ),
        "soil_index": soil_index,
        "observations": int(values.count()),
        "total_window_days": total_window_days,
        "source": "gldas_2.1",
    }


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

def compute_soil_windows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all 8 × 4 × 2 = 64 windows."""

    baseline = compute_regional_baseline(df)

    results: List[dict] = []

    available_regions = sorted(
        set(df["region_code"])
    )

    available_years = sorted(
        set(df["year"])
    )

    for region_code in EXPECTED_REGIONS:

        if region_code not in available_regions:
            logger.warning(
                "Region absent from GLDAS data: %s",
                region_code,
            )

        for year in EXPECTED_YEARS:

            if year not in available_years:
                logger.warning(
                    "Year absent from GLDAS data: %d",
                    year,
                )

            for campaign in CAMPAIGN_WINDOWS:

                result = compute_campaign_window(
                    df,
                    baseline,
                    region_code,
                    year,
                    campaign,
                )

                results.append(result)

                observations = result[
                    "observations"
                ]

                soil_index = result[
                    "soil_index"
                ]

                logger.info(
                    "%s / %d / %-7s : "
                    "%3d observations, soil_index=% .4f",
                    region_code,
                    year,
                    campaign,
                    observations,
                    soil_index
                    if np.isfinite(soil_index)
                    else float("nan"),
                )

    return pd.DataFrame(results)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_output(
    output: pd.DataFrame,
) -> None:
    """Strict validation."""

    expected_records = (
        len(EXPECTED_REGIONS)
        * len(EXPECTED_YEARS)
        * len(CAMPAIGN_WINDOWS)
    )

    if len(output) != expected_records:
        raise ValueError(
            f"Expected {expected_records} records, "
            f"got {len(output)}"
        )

    if output[
        ["region_code", "year", "campaign"]
    ].duplicated().any():
        raise ValueError(
            "Duplicate region/year/campaign records detected."
        )

    sources = set(
        output["source"].dropna()
    )

    if sources != {"gldas_2.1"}:
        raise ValueError(
            f"Unexpected sources: {sorted(sources)}"
        )

    missing_index = int(
        output["soil_index"].isna().sum()
    )

    if missing_index:
        raise ValueError(
            f"{missing_index} soil_index values are missing."
        )

    if (
        output["soil_moisture_mean_0_10cm"]
        <= 0
    ).any():
        raise ValueError(
            "Non-positive soil moisture detected."
        )

    if (
        output["soil_index"].abs() > 10
    ).any():
        raise ValueError(
            "Extreme soil_index detected (|index| > 10)."
        )


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(
    output: pd.DataFrame,
) -> None:
    """Print output summary."""

    print()
    print("=" * 70)
    print("SOIL WINDOWS")
    print("=" * 70)

    print(
        f"Records:        {len(output)}"
    )
    print(
        f"Regions:        {output['region_code'].nunique()}"
    )
    print(
        f"Years:          {output['year'].nunique()}"
    )
    print(
        f"Campaigns:      "
        f"{sorted(output['campaign'].unique())}"
    )
    print(
        f"Sources:        "
        f"{sorted(output['source'].unique())}"
    )

    print()
    print(
        output[
            [
                "region_code",
                "year",
                "campaign",
                "soil_moisture_mean_0_10cm",
                "soil_index",
                "observations",
            ]
        ].to_string(index=False)
    )

    print()
    print("Soil index statistics:")

    print(
        output["soil_index"]
        .describe()
        .to_string()
    )


# ============================================================================
# CLI
# ============================================================================

def main() -> int:

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    print("=" * 70)
    print("ФАЗА SOIL IV — COMPUTE SOIL INDEX")
    print("=" * 70)
    print()
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    try:
        df = load_input()

        logger.info(
            "Loaded GLDAS records: %d",
            len(df),
        )

        output = compute_soil_windows(df)

        validate_output(output)

    except (
        FileNotFoundError,
        ValueError,
    ) as exc:

        logger.error(str(exc))
        return 1

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print_summary(output)

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())