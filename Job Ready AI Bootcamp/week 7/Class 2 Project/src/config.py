"""
Configuration loader using dataclasses and YAML.
Centralizes all hyperparameters, paths, and constants.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class ProjectConfig:
    name: str
    version: str
    random_seed: int


@dataclass
class PathsConfig:
    data_raw: str
    data_processed: str
    models_dir: str
    logs_dir: str
    artifacts_dir: str


@dataclass
class DataConfig:
    n_samples: int
    test_size: float
    val_size: float
    target_column: str
    feature_columns: List[str]
    categorical_columns: List[str]


@dataclass
class ModelConfig:
    input_dim: int
    hidden_units: List[int]
    dropout_rate: float
    learning_rate: float
    batch_size: int
    epochs: int
    early_stopping_patience: int
    checkpoint_monitor: str


@dataclass
class TrainingConfig:
    optimizer: str
    loss_function: str
    metrics: List[str]
    validation_split: float


@dataclass
class PreprocessingConfig:
    numeric_scaler: str
    categorical_encoder: str
    handle_missing: str


@dataclass
class Config:
    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    preprocessing: PreprocessingConfig

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file."""
        if not os.path.exists(path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_dir, "..", "..", path)

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            project=ProjectConfig(**raw["project"]),
            paths=PathsConfig(**raw["paths"]),
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
            preprocessing=PreprocessingConfig(**raw["preprocessing"]),
        )

    def ensure_dirs(self) -> None:
        """Create all necessary directories if they don't exist."""
        dirs = [
            self.paths.data_raw,
            self.paths.data_processed,
            self.paths.models_dir,
            self.paths.logs_dir,
            self.paths.artifacts_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


_config_instance = None


def get_config() -> Config:
    """Get or create the singleton Config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config.from_yaml()
        _config_instance.ensure_dirs()
    return _config_instance
