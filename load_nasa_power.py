#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
load_nasa_power.py

Фаза 6.6
Загрузка реальных погодных данных NASA POWER для регионов РФ.

Источник:
    NASA POWER Daily Point API

Выход:
    data/raw/weather/nasa_power_daily.csv

Параметры:
    PRECTOTCORR - скорректированные осадки, mm/day
    T2M         - средняя температура на высоте 2 м, °C
    T2M_MAX     - максимальная температура, °C
    T2M_MIN     - минимальная температура, °C
    WS2M        - скорость ветра на высоте 2 м, m/s

Важно:
    Эти данные используются для построения working_days_window.
    Они НЕ являются телеметрией тракторов и НЕ содержат отказов.

    source="nasa_power" означает реальный внешний источник данных,
    а не synthetic.

NASA POWER documentation:
https://power.larc.nasa.gov/docs/services/api/temporal/daily/
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Final

import pandas as pd
import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("load_nasa_power")


# ============================================================================
# Paths
# ============================================================================

OUTPUT_DIR: Final[Path] = Path("data/raw/weather")
OUTPUT_PATH: Final[Path] = OUTPUT_DIR / "nasa_power_daily.csv"


# ============================================================================
# NASA POWER
# ============================================================================

NASA_POWER_URL: Final[str] = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
)

COMMUNITY: Final[str] = "AG"

PARAMETERS: Final[tuple[str, ...]] = (
    "PRECTOTCORR",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "WS2M",
)

TIME_STANDARD: Final[str] = "LST"


# ============================================================================
# Target regions
# ============================================================================
#
# Coordinates are representative points for the regional strata used by
# the existing model.
#
# They are NOT station coordinates and must not be interpreted as such.
# ============================================================================

REGION_COORDINATES: Final[dict[str, tuple[float, float]]] = {
    "volga": (52.0, 47.0),
    "north_caucasus": (44.5, 40.0),
    "northwest": (59.0, 30.0),
    "central_chernozem": (51.5, 39.0),
    "kuban": (45.0, 39.0),
    "altai": (52.5, 84.0),
    "amur": (50.0, 127.0),
    "vladimir": (56.0, 40.0),
}


# ============================================================================
# Years
# ============================================================================

YEARS = list(range(2015, 2026))


# ============================================================================
# HTTP configuration
# ============================================================================

REQUEST_TIMEOUT_SECONDS: Final[int] = 120
REQUEST_RETRIES: Final[int] = 4
REQUEST_BACKOFF_FACTOR: Final[float] = 1.5
REQUEST_PAUSE_SECONDS: Final[float] = 1.0


# ============================================================================
# Helpers
# ============================================================================


def build_session() -> requests.Session:
    """
    Create an HTTP session with conservative retry behaviour.
    """

    session = requests.Session()

    retry = Retry(
        total=REQUEST_RETRIES,
        connect=REQUEST_RETRIES,
        read=REQUEST_RETRIES,
        status=REQUEST_RETRIES,
        backoff_factor=REQUEST_BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "tractor-risk-model/6.6 "
                "(NASA POWER weather ETL)"
            )
        }
    )

    return session


def fetch_nasa_power(
    session: requests.Session,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Fetch one regional point from NASA POWER.
    """

    params = {
        "parameters": ",".join(PARAMETERS),
        "community": COMMUNITY,
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
        "time-standard": TIME_STANDARD,
    }

    logger.info(
        "NASA POWER request: lat=%.4f lon=%.4f %s-%s",
        latitude,
        longitude,
        start_date,
        end_date,
    )

    response: Response = session.get(
        NASA_POWER_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError("NASA POWER response is not a JSON object")

    return payload


def parse_nasa_power_response(
    region_code: str,
    year: int,
    latitude: float,
    longitude: float,
    payload: dict,
) -> pd.DataFrame:
    """
    Convert NASA POWER JSON response into daily wide-format DataFrame.

    Output columns include:

        date
        region_code
        year
        latitude
        longitude
        PRECTOTCORR
        T2M
        T2M_MAX
        T2M_MIN
        WS2M
    """

    properties = payload.get("properties")

    if not isinstance(properties, dict):
        raise ValueError("Missing 'properties' in NASA POWER response")

    parameter_data = properties.get("parameter")

    if not isinstance(parameter_data, dict):
        raise ValueError(
            "Missing 'properties.parameter' in NASA POWER response"
        )

    if not parameter_data:
        raise ValueError("NASA POWER returned no parameter data")

    frame_parts: list[pd.Series] = []

    for parameter in PARAMETERS:
        values = parameter_data.get(parameter)

        if not isinstance(values, dict):
            logger.warning(
                "%s: parameter %s is missing",
                region_code,
                parameter,
            )
            continue

        series = pd.Series(values, name=parameter)
        frame_parts.append(series)

    if not frame_parts:
        raise ValueError(
            f"No supported parameters returned for {region_code}"
        )

    df = pd.concat(frame_parts, axis=1)

    df.index.name = "date"
    df = df.reset_index()

    df["region_code"] = region_code
    df["year"] = year
    df["latitude"] = latitude
    df["longitude"] = longitude

    # ------------------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------------------

    for parameter in PARAMETERS:
        if parameter not in df.columns:
            df[parameter] = pd.NA

        df[parameter] = pd.to_numeric(
            df[parameter],
            errors="coerce",
        )

        # NASA POWER missing value.
        df.loc[df[parameter] <= -998.0, parameter] = pd.NA

    # ------------------------------------------------------------------
    # Date normalization
    # ------------------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y%m%d",
        errors="coerce",
    )

    if df["date"].isna().any():
        raise ValueError(
            f"{region_code}/{year}: invalid date values"
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    df["source"] = "nasa_power"
    df["time_standard"] = TIME_STANDARD

    return df[
        [
            "date",
            "region_code",
            "year",
            "latitude",
            "longitude",
            *PARAMETERS,
            "source",
            "time_standard",
        ]
    ].sort_values("date").reset_index(drop=True)


def validate_daily_frame(df: pd.DataFrame) -> None:
    """
    Validate the raw daily weather table before writing it.
    """

    required = {
        "date",
        "region_code",
        "year",
        "latitude",
        "longitude",
        *PARAMETERS,
        "source",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Weather dataset is empty")

    if df["date"].isna().any():
        raise ValueError("Weather dataset contains invalid dates")

    if df["region_code"].isna().any():
        raise ValueError("region_code contains missing values")

    invalid_source = ~df["source"].eq("nasa_power")

    if invalid_source.any():
        raise ValueError(
            "Raw NASA POWER dataset contains non-nasa_power sources"
        )

    for parameter in PARAMETERS:
        numeric = pd.to_numeric(df[parameter], errors="coerce")

        if numeric.notna().sum() == 0:
            raise ValueError(
                f"Parameter {parameter} contains no usable observations"
            )


def load_all_regions() -> pd.DataFrame:
    """
    Download all configured regions and years.
    """

    session = build_session()

    frames: list[pd.DataFrame] = []

    for region_code, (latitude, longitude) in REGION_COORDINATES.items():

        for year in YEARS:

            start_date = f"{year}0101"
            end_date = f"{year}1231"

            try:
                payload = fetch_nasa_power(
                    session=session,
                    latitude=latitude,
                    longitude=longitude,
                    start_date=start_date,
                    end_date=end_date,
                )

                frame = parse_nasa_power_response(
                    region_code=region_code,
                    year=year,
                    latitude=latitude,
                    longitude=longitude,
                    payload=payload,
                )

                if frame.empty:
                    logger.warning(
                        "%s/%d: empty response",
                        region_code,
                        year,
                    )
                    continue

                frames.append(frame)

                logger.info(
                    "%s / %d: %d daily records",
                    region_code,
                    year,
                    len(frame),
                )

            except (
                requests.RequestException,
                ValueError,
                KeyError,
            ) as exc:
                logger.error(
                    "%s / %d: %s",
                    region_code,
                    year,
                    exc,
                )

            # Be conservative with API traffic.
            time.sleep(REQUEST_PAUSE_SECONDS)

    if not frames:
        raise RuntimeError(
            "NASA POWER returned no usable datasets"
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Remove exact duplicates.
    # ------------------------------------------------------------------

    combined = combined.drop_duplicates(
        subset=["region_code", "date"],
        keep="first",
    )

    combined = combined.sort_values(
        ["region_code", "date"],
    ).reset_index(drop=True)

    validate_daily_frame(combined)

    return combined


def print_summary(df: pd.DataFrame) -> None:
    """
    Print coverage summary.
    """

    print()
    print("=" * 70)
    print("NASA POWER DAILY WEATHER DATA")
    print("=" * 70)

    print(f"Records:       {len(df):,}")
    print(f"Regions:       {df['region_code'].nunique()}")
    print(f"Years:         {df['year'].nunique()}")
    print(
        f"Date range:    "
        f"{df['date'].min().date()} — {df['date'].max().date()}"
    )
    print(f"Source:        {sorted(df['source'].unique())}")

    print()
    print("Coverage:")

    coverage = (
        df.groupby("region_code")["year"]
        .agg(
            years="nunique",
            first_year="min",
            last_year="max",
            records="count",
        )
        .sort_index()
    )

    print(coverage.to_string())


# ============================================================================
# Main
# ============================================================================


def main() -> int:

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    print("=" * 70)
    print("ФАЗА 6.6 — NASA POWER WEATHER ETL")
    print("=" * 70)
    print(f"Regions:   {len(REGION_COORDINATES)}")
    print(f"Years:     {list(YEARS)}")
    print(f"Parameters: {', '.join(PARAMETERS)}")
    print()

    try:
        df = load_all_regions()

    except Exception as exc:
        logger.exception(
            "Weather download failed: %s",
            exc,
        )
        return 1

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Atomic-ish write: temporary file first.
    tmp_path = OUTPUT_PATH.with_suffix(".tmp")

    df.to_csv(
        tmp_path,
        index=False,
        encoding="utf-8",
    )

    tmp_path.replace(OUTPUT_PATH)

    print_summary(df)

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())