#!/usr/bin/env python3
"""
Unified Pipeline Orchestrator.

Runs the complete ML pipeline end-to-end:
    1. Data Generation
    2. Preprocessing
    3. Model Training
    4. Evaluation
    5. Report Generation

Usage:
    python run_pipeline.py [--skip-generation] [--skip-training] [--epochs 50]
"""

import os
import sys
import argparse
import time
from datetime import datetime

from src.config import get_config
from src.utils import logger, setup_logger
from src.utils.helpers import set_seed, save_json
from src.data.generator import generate_equipment_data, save_raw_data
from src.data.preprocessor import preprocess_pipeline
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator


def run_full_pipeline(
    skip_generation: bool = False,
    skip_training: bool = False,
    epochs: int = None,
    batch_size: int = None,
) -> dict:
    """
    Execute the complete ML pipeline.

    Returns:
        Dictionary with paths to all artifacts and final metrics.
    """
    config = get_config()
    setup_logger()
    set_seed(config.project.random_seed)

    start_time = time.time()
    pipeline_start = datetime.now().isoformat()

    logger.info("=" * 60)
    logger.info("PIPELINE STARTED | Project: {} | Seed: {}", config.project.name, config.project.random_seed)
    logger.info("=" * 60)

    artifacts = {
        "pipeline_start": pipeline_start,
        "config": {
            "n_samples": config.data.n_samples,
            "hidden_units": config.model.hidden_units,
            "dropout": config.model.dropout_rate,
            "learning_rate": config.model.learning_rate,
        },
        "stages": {},
    }

    # ── STAGE 1: Data Generation ──
    if not skip_generation:
        logger.info("\n[STAGE 1/4] Data Generation")
        raw_path = os.path.join(config.paths.data_raw, "equipment_data.csv")

        if os.path.exists(raw_path):
            logger.info("Raw data already exists at {}. Regenerating...", raw_path)

        df = generate_equipment_data(n_samples=config.data.n_samples, seed=config.project.random_seed)
        save_raw_data(df, raw_path)

        artifacts["stages"]["generation"] = {
            "status": "success",
            "rows": len(df),
            "columns": len(df.columns),
            "path": raw_path,
        }
        logger.info("Stage 1 complete: {} rows generated", len(df))
    else:
        raw_path = os.path.join(config.paths.data_raw, "equipment_data.csv")
        artifacts["stages"]["generation"] = {"status": "skipped", "path": raw_path}
        logger.info("Stage 1 skipped (using existing raw data)")

    # ── STAGE 2: Preprocessing ──
    logger.info("\n[STAGE 2/4] Preprocessing")
    preprocess_result = preprocess_pipeline(raw_path, config.paths.data_processed)

    artifacts["stages"]["preprocessing"] = {
        "status": "success",
        "train_shape": preprocess_result["X_train_shape"],
        "val_shape": preprocess_result["X_val_shape"],
        "test_shape": preprocess_result["X_test_shape"],
        "features": preprocess_result["feature_names"],
        "output_dir": preprocess_result["output_dir"],
    }
    logger.info("Stage 2 complete: {} features after encoding", len(preprocess_result["feature_names"]))

    # ── STAGE 3: Training ──
    if not skip_training:
        logger.info("\n[STAGE 3/4] Model Training")
        trainer = ModelTrainer()
        history = trainer.train(
            processed_dir=config.paths.data_processed,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Load metadata
        import json
        metadata_path = os.path.join(config.paths.models_dir, "training_metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        artifacts["stages"]["training"] = {
            "status": "success",
            "epochs_trained": metadata["epochs_trained"],
            "final_train_loss": metadata["final_train_loss"],
            "final_val_loss": metadata["final_val_loss"],
            "test_metrics": metadata["test_metrics"],
            "model_params": metadata["model_params"],
            "model_path": metadata["model_path"],
        }
        logger.info("Stage 3 complete: {} epochs, test MAE={:.4f}", 
                    metadata["epochs_trained"], metadata["test_metrics"].get("mae", 0))
    else:
        artifacts["stages"]["training"] = {"status": "skipped"}
        logger.info("Stage 3 skipped")

    # ── STAGE 4: Evaluation ──
    logger.info("\n[STAGE 4/4] Evaluation & Reporting")
    evaluator = ModelEvaluator()
    report_dir = evaluator.generate_report()

    # Load evaluation report
    import json
    report_path = os.path.join(report_dir, "evaluation_report.json")
    with open(report_path, "r") as f:
        eval_report = json.load(f)

    artifacts["stages"]["evaluation"] = {
        "status": "success",
        "test_metrics": eval_report["test_metrics"],
        "val_metrics": eval_report["val_metrics"],
        "report_dir": report_dir,
        "plots": [
            os.path.join(report_dir, "predictions_scatter.png"),
            os.path.join(report_dir, "residuals_distribution.png"),
            os.path.join(report_dir, "feature_importance.png"),
        ],
    }
    logger.info("Stage 4 complete: report saved to {}", report_dir)

    # ── Final Summary ──
    elapsed = time.time() - start_time
    artifacts["pipeline_end"] = datetime.now().isoformat()
    artifacts["elapsed_seconds"] = round(elapsed, 2)

    # Save pipeline manifest
    manifest_path = os.path.join(config.paths.artifacts_dir, "pipeline_manifest.json")
    save_json(artifacts, manifest_path)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE | Elapsed: {:.1f}s", elapsed)
    logger.info("=" * 60)
    logger.info("Artifacts location: {}", config.paths.artifacts_dir)
    logger.info("Models location: {}", config.paths.models_dir)
    logger.info("Manifest: {}", manifest_path)

    return artifacts


def main():
    parser = argparse.ArgumentParser(
        description="Run the full ML pipeline for Equipment Success Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Full pipeline
  python run_pipeline.py --skip-generation  # Use existing raw data
  python run_pipeline.py --epochs 50        # Train for 50 epochs
  python run_pipeline.py --skip-training    # Only generate + preprocess + evaluate existing model
        """,
    )
    parser.add_argument("--skip-generation", action="store_true", help="Skip data generation")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    result = run_full_pipeline(
        skip_generation=args.skip_generation,
        skip_training=args.skip_training,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Print summary to console
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for stage, info in result["stages"].items():
        status = info.get("status", "unknown")
        icon = "✅" if status == "success" else "⏭️" if status == "skipped" else "❌"
        print(f"{icon} {stage.upper()}: {status}")
        if status == "success" and "test_metrics" in info:
            for metric, value in info["test_metrics"].items():
                print(f"   └─ {metric}: {value:.4f}")
    print(f"\n⏱️  Total time: {result['elapsed_seconds']:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
