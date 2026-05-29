"""
Synthetic Industrial Equipment Data Generator.

Generates a realistic dataset for predicting equipment "Success Score" (0-100).
Features include operational metrics, maintenance history, and environmental factors.
"""

import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import get_config
from src.utils import logger
from src.utils.helpers import set_seed


def generate_equipment_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic industrial equipment dataset.

    The Success Score is computed as a weighted combination of features
    with realistic noise, simulating a real industrial environment.
    """
    set_seed(seed)

    # ── Categorical Features ──
    equipment_types = ["Pump", "Compressor", "Turbine", "Motor", "Generator", "Heat_Exchanger"]
    manufacturers = ["Siemens", "ABB", "GE", "Schneider", "Mitsubishi", "Honeywell"]
    facilities = ["Plant_A", "Plant_B", "Plant_C", "Plant_D"]

    equipment_type = np.random.choice(equipment_types, size=n_samples)
    manufacturer = np.random.choice(manufacturers, size=n_samples)
    facility_location = np.random.choice(facilities, size=n_samples)

    # ── Numeric Features (with realistic ranges) ──
    # Operating temperature in Celsius
    operating_temperature = np.random.normal(65, 15, n_samples)
    operating_temperature = np.clip(operating_temperature, 20, 120)

    # Vibration level (mm/s RMS)
    vibration_level = np.random.exponential(2.5, n_samples)
    vibration_level = np.clip(vibration_level, 0.1, 15.0)

    # Pressure reading (bar)
    pressure_reading = np.random.normal(8.5, 2.0, n_samples)
    pressure_reading = np.clip(pressure_reading, 2.0, 18.0)

    # Power consumption (kW)
    power_consumption = np.random.gamma(5, 15, n_samples)
    power_consumption = np.clip(power_consumption, 10, 200)

    # Runtime hours since last major overhaul
    runtime_hours = np.random.gamma(3, 800, n_samples)
    runtime_hours = np.clip(runtime_hours, 50, 10000)

    # Days since last maintenance
    days_since_maintenance = np.random.poisson(45, n_samples)
    days_since_maintenance = np.clip(days_since_maintenance, 1, 180)

    # Error count in last 24h
    error_count_24h = np.random.poisson(1.5, n_samples)
    error_count_24h = np.clip(error_count_24h, 0, 15)

    # Oil quality index (0-100, higher is better)
    oil_quality_index = np.random.beta(7, 2, n_samples) * 100

    # Load factor (%)
    load_factor = np.random.normal(72, 12, n_samples)
    load_factor = np.clip(load_factor, 30, 100)

    # Ambient temperature (Celsius)
    ambient_temperature = np.random.normal(28, 8, n_samples)
    ambient_temperature = np.clip(ambient_temperature, 5, 45)

    # ── Compute Success Score (Target) ──
    # Base score starts at 70
    score = 70.0

    # Temperature penalty (optimal around 55-75)
    score -= 0.8 * np.abs(operating_temperature - 65)

    # Vibration penalty (higher vibration = lower score)
    score -= 3.5 * vibration_level

    # Pressure bonus/penalty (optimal around 8-10)
    score -= 1.2 * np.abs(pressure_reading - 9)

    # Power efficiency (lower consumption per load = better)
    power_efficiency = power_consumption / (load_factor + 1)
    score -= 0.05 * power_efficiency

    # Runtime penalty (more hours = more wear)
    score -= 0.003 * runtime_hours

    # Maintenance freshness bonus
    score += 0.3 * np.maximum(0, 60 - days_since_maintenance)

    # Error penalty
    score -= 4.0 * error_count_24h

    # Oil quality bonus
    score += 0.25 * oil_quality_index

    # Load factor bonus (near 80% is optimal)
    score -= 0.4 * np.abs(load_factor - 80)

    # Ambient penalty (extreme temps hurt performance)
    score -= 0.3 * np.abs(ambient_temperature - 25)

    # Equipment type modifiers
    type_modifiers = {
        "Pump": -2, "Compressor": -3, "Turbine": 1,
        "Motor": 2, "Generator": 0, "Heat_Exchanger": -1
    }
    for i, eq_type in enumerate(equipment_type):
        score[i] += type_modifiers[eq_type]

    # Manufacturer modifiers (reputation factor)
    mfr_modifiers = {
        "Siemens": 2, "ABB": 1, "GE": 0,
        "Schneider": -1, "Mitsubishi": 1, "Honeywell": 0
    }
    for i, mfr in enumerate(manufacturer):
        score[i] += mfr_modifiers[mfr]

    # Add realistic noise
    score += np.random.normal(0, 3, n_samples)

    # Clip to valid range
    score = np.clip(score, 0, 100)

    # ── Assemble DataFrame ──
    df = pd.DataFrame({
        "equipment_id": [f"EQ_{i:05d}" for i in range(n_samples)],
        "equipment_type": equipment_type,
        "manufacturer": manufacturer,
        "facility_location": facility_location,
        "operating_temperature": np.round(operating_temperature, 2),
        "vibration_level": np.round(vibration_level, 3),
        "pressure_reading": np.round(pressure_reading, 2),
        "power_consumption": np.round(power_consumption, 2),
        "runtime_hours": np.round(runtime_hours, 0).astype(int),
        "days_since_maintenance": days_since_maintenance.astype(int),
        "error_count_24h": error_count_24h.astype(int),
        "oil_quality_index": np.round(oil_quality_index, 2),
        "load_factor": np.round(load_factor, 2),
        "ambient_temperature": np.round(ambient_temperature, 2),
        "success_score": np.round(score, 2),
    })

    return df


def save_raw_data(df: pd.DataFrame, path: str) -> None:
    """Save generated data to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Raw data saved: {} rows → {}", len(df), path)


def main() -> None:
    """CLI entry point for data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic equipment data")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    config = get_config()
    n_samples = args.samples or config.data.n_samples
    output_path = args.output or os.path.join(config.paths.data_raw, "equipment_data.csv")

    logger.info("Generating {} synthetic equipment records...", n_samples)
    df = generate_equipment_data(n_samples=n_samples, seed=config.project.random_seed)
    save_raw_data(df, output_path)

    logger.info("Data generation complete!")
    logger.info("Score distribution: mean={:.2f}, std={:.2f}", df["success_score"].mean(), df["success_score"].std())


if __name__ == "__main__":
    main()
