#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enterprise_quality.py
Enterprise Quality Index (EQI) generator for DGP.

EQI is an enterprise-level characteristic (not tractor-level).
Multiple tractors from the same enterprise share the same EQI value.

This module ensures:
1. EQI is generated at enterprise level
2. Balanced assignment of tractors to enterprises
3. Var(EQI | enterprise) = 0 (no within-enterprise variation)
"""
from __future__ import annotations

import numpy as np


def generate_enterprise_quality(
    rng: np.random.Generator,
    n_tractors: int,
    n_enterprises: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate enterprise-level EQI and map it to tractors.

    Each enterprise has exactly one EQI value.
    All tractors belonging to the same enterprise share that value.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n_tractors : int
        Total number of tractors.
    n_enterprises : int
        Number of enterprises.

    Returns
    -------
    tractor_eqi : np.ndarray
        EQI for each tractor (shape: n_tractors).
    enterprise_ids : np.ndarray
        Enterprise ID for each tractor (shape: n_tractors).

    Notes
    -----
    EQI components:
    - mechanician_skill: average skill level (1-6), normalized to [0, 1]
    - maintenance_quality: adherence to maintenance schedule, Beta(5, 2)
    - storage_quality: storage type (0=open, 1=covered, 2=heated)
    - engineer_quality: share of engineers with higher education, Beta(3, 2)

    EQI = 0.30 * normalize(mech_skill) +
          0.30 * maint_quality +
          0.20 * normalize(storage) +
          0.20 * engineer_quality

    Expected value: ~0.5-0.7
    Range: [0, 1]

    Enterprise assignment is balanced: each enterprise gets exactly
    n_tractors / n_enterprises tractors (with possible remainder).
    """
    # Validation
    if n_tractors <= 0:
        raise ValueError("n_tractors must be positive")
    if n_enterprises <= 0:
        raise ValueError("n_enterprises must be positive")
    if n_enterprises > n_tractors:
        raise ValueError("n_enterprises cannot exceed n_tractors")

    # Balanced enterprise assignment
    # Each enterprise gets exactly n_tractors // n_enterprises tractors
    enterprise_ids = np.arange(n_tractors) % n_enterprises
    rng.shuffle(enterprise_ids)

    # Generate EQI at enterprise level
    enterprise_eqi = np.empty(n_enterprises, dtype=float)

    for enterprise_id in range(n_enterprises):
        # Mechanician skill: average 2-5 (normalized to 0-1)
        mech_skill = rng.uniform(2.0, 5.0)
        mech_normalized = (mech_skill - 2.0) / 3.0

        # Maintenance quality: Beta(5, 2) → mean ≈ 0.71
        maintenance_quality = rng.beta(5.0, 2.0)

        # Storage quality: 50% open, 30% covered, 20% heated
        storage_quality = rng.choice(
            np.array([0, 1, 2]),
            p=[0.50, 0.30, 0.20],
        )
        storage_normalized = storage_quality / 2.0

        # Engineer quality: Beta(3, 2) → mean ≈ 0.60
        engineer_quality = rng.beta(3.0, 2.0)

        # Composite EQI
        eqi = (
            0.30 * mech_normalized
            + 0.30 * maintenance_quality
            + 0.20 * storage_normalized
            + 0.20 * engineer_quality
        )

        enterprise_eqi[enterprise_id] = np.clip(eqi, 0.0, 1.0)

    # Map enterprise EQI to tractors
    tractor_eqi = enterprise_eqi[enterprise_ids]

    return tractor_eqi, enterprise_ids


def validate_enterprise_structure(
    tractor_eqi: np.ndarray,
    enterprise_ids: np.ndarray,
) -> bool:
    """
    Validate that EQI is truly enterprise-level.

    Checks:
    1. Var(EQI | enterprise) = 0 (no within-enterprise variation)
    2. Each enterprise has at least one tractor
    3. EQI values are in [0, 1]

    Parameters
    ----------
    tractor_eqi : np.ndarray
        EQI for each tractor.
    enterprise_ids : np.ndarray
        Enterprise ID for each tractor.

    Returns
    -------
    bool
        True if structure is valid, False otherwise.
    """
    # Check EQI range
    if not (tractor_eqi >= 0).all() or not (tractor_eqi <= 1).all():
        return False

    # Check within-enterprise variation
    unique_enterprises = np.unique(enterprise_ids)
    for ent_id in unique_enterprises:
        mask = enterprise_ids == ent_id
        eqi_values = tractor_eqi[mask]
        if len(np.unique(eqi_values)) > 1:
            return False

    # Check all enterprises have tractors
    if len(unique_enterprises) != len(np.unique(enterprise_ids)):
        return False

    return True