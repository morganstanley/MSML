"""Experiment validation helpers to catch common errors before execution.

This module provides validation functions that experiments can call before training
to detect and prevent common failure modes:
- Empty training data
- Mismatched tensor dimensions
- Excessive runtime predictions
- Missing features
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("alpha_lab.experiment_validation")


def validate_training_data(
    data_path: str,
    date_col: str,
    id_col: str,
    feature_cols: list[str],
    target_col: str,
    train_start: str | None = None,
    train_end: str | None = None,
) -> dict[str, Any]:
    """Validate training data before starting experiment.

    Returns validation results dict with:
    - valid: bool (True if passed all checks)
    - warnings: list[str] (non-fatal issues)
    - errors: list[str] (fatal issues that will cause failure)
    - stats: dict (data statistics)

    Common failure modes detected:
    - Data file doesn't exist
    - Required columns missing
    - No training samples in date range
    - All features are NaN
    - Target column has no variance
    """
    result = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "stats": {},
    }

    # Check file exists
    if not Path(data_path).exists():
        result["valid"] = False
        result["errors"].append(f"Data file not found: {data_path}")
        return result

    try:
        import pandas as pd

        df = pd.read_parquet(data_path)

        # Check required columns
        required_cols = [date_col, id_col, target_col] + feature_cols
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            result["valid"] = False
            result["errors"].append(f"Missing columns: {missing}")
            return result

        # Filter to training period if specified. Convert both sides explicitly
        # so mixed-dtype comparisons (e.g. object-dtype date_col + string bound)
        # don't silently drop rows via lexicographic ordering.
        if train_start or train_end:
            df[date_col] = pd.to_datetime(df[date_col])
            if train_start:
                df = df[df[date_col] >= pd.to_datetime(train_start)]
            if train_end:
                df = df[df[date_col] <= pd.to_datetime(train_end)]

        # Check we have training samples
        n_samples = len(df)
        if n_samples == 0:
            result["valid"] = False
            result["errors"].append(
                f"No training samples found in date range "
                f"[{train_start}, {train_end}]"
            )
            return result

        result["stats"]["n_samples"] = n_samples
        result["stats"]["n_entities"] = df[id_col].nunique()
        result["stats"]["n_dates"] = df[date_col].nunique()

        # Check for all-NaN features
        feature_data = df[feature_cols]
        nan_cols = [col for col in feature_cols if feature_data[col].isna().all()]
        if nan_cols:
            result["valid"] = False
            result["errors"].append(f"Features are all NaN: {nan_cols}")

        # Check for zero-variance target
        target_data = df[target_col].dropna()
        if len(target_data) == 0:
            result["valid"] = False
            result["errors"].append(f"Target column '{target_col}' is all NaN")
        elif target_data.std() == 0:
            result["warnings"].append(
                f"Target column '{target_col}' has zero variance "
                f"(all values = {target_data.iloc[0]})"
            )

        # Check sparsity. Guard against empty feature_cols (divide-by-zero):
        # required_cols already covers missing columns, but feature_cols=[] is
        # a legitimate config (metadata-only runs).
        total_values = len(df) * len(feature_cols)
        if total_values > 0:
            nan_values = feature_data.isna().sum().sum()
            sparsity = nan_values / total_values
            result["stats"]["feature_sparsity"] = sparsity

            if sparsity > 0.9:
                result["warnings"].append(
                    f"Features are very sparse ({sparsity:.1%} missing). "
                    f"Consider imputation or different features."
                )
        else:
            sparsity = 0.0
            result["stats"]["feature_sparsity"] = sparsity

        # Check sample balance. n_entities > 0 is implied by n_samples > 0
        # (we returned early above otherwise), but keep the guard explicit so
        # a future refactor of the early-return doesn't reintroduce div-by-zero.
        n_entities = df[id_col].nunique()
        samples_per_entity = len(df) / n_entities if n_entities > 0 else 0.0
        result["stats"]["avg_samples_per_entity"] = samples_per_entity

        if samples_per_entity < 5:
            result["warnings"].append(
                f"Very few samples per entity (avg {samples_per_entity:.1f}). "
                f"Model may struggle to learn patterns."
            )

        logger.info(
            f"Data validation: {n_samples} samples, "
            f"{df[id_col].nunique()} entities, "
            f"{df[date_col].nunique()} dates, "
            f"{sparsity:.1%} sparsity"
        )

    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Validation failed: {e}")

    return result


def estimate_training_time(
    model_type: str,
    library: str,
    n_samples: int,
    n_features: int,
    n_epochs: int = 10,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Estimate training time for experiment.

    Returns:
    - estimated_seconds: int (predicted runtime)
    - within_budget: bool (whether it fits time limit)
    - recommendation: str (what to do if over budget)

    Used to prevent timeouts by catching slow experiments before submission.
    """
    # Heuristic runtime estimates (seconds per sample per epoch)
    time_per_sample_per_epoch = {
        # Neural networks (GPU)
        "tft": 0.002,
        "lstm": 0.001,
        "gru": 0.001,
        "tcn": 0.0015,
        "nhits": 0.0012,
        "patchtst": 0.0018,
        "timesnet": 0.002,
        "transformer": 0.0025,
        "deepar": 0.0015,
        # Tree ensembles (CPU)
        "lightgbm": 0.00005,
        "catboost": 0.0001,
        "xgboost": 0.00008,
        "randomforest": 0.0002,
        # Linear models (CPU)
        "ridge": 0.00001,
        "lasso": 0.00002,
        "elasticnet": 0.00002,
    }

    # Get base time
    model_lower = model_type.lower()
    base_time = time_per_sample_per_epoch.get(model_lower, 0.001)

    # Adjust for features (more features = more computation)
    feature_scale = max(1.0, (n_features / 100) ** 0.5)

    # Estimate total time
    if "lightgbm" in model_lower or "xgboost" in model_lower or "catboost" in model_lower:
        # Tree ensembles: n_samples * feature_scale * base_time
        estimated = n_samples * feature_scale * base_time
    else:
        # Neural networks: n_samples * n_epochs * feature_scale * base_time / batch_size
        batches = max(1, n_samples / batch_size)
        estimated = batches * n_epochs * feature_scale * base_time

    # Add overhead (data loading, validation, etc.)
    estimated = estimated * 1.5

    # Determine time limits by model type (case-insensitive — configs
    # may spell libraries as "PyTorch" / "TensorFlow" etc.)
    library_lower = (library or "").lower()
    if library_lower in {"pytorch", "tensorflow", "neuralforecast", "pytorch-forecasting", "darts"}:
        time_limit = 21600  # 6 hours for GPU
    else:
        time_limit = 3600  # 1 hour for CPU

    within_budget = estimated < time_limit

    recommendation = ""
    if not within_budget:
        # Suggest fixes
        reduction_factor = estimated / time_limit
        if reduction_factor < 2:
            recommendation = f"Reduce epochs from {n_epochs} to {int(n_epochs / reduction_factor)}"
        elif reduction_factor < 3:
            recommendation = (
                f"Reduce training set size by {int((1 - 1/reduction_factor) * 100)}% "
                f"(e.g., sample {int(100/reduction_factor)}% of data)"
            )
        else:
            recommendation = (
                f"This experiment is {reduction_factor:.1f}x over budget. "
                f"Consider: (1) reducing epochs to {int(n_epochs / reduction_factor)}, "
                f"(2) sampling {int(100/reduction_factor)}% of data, "
                f"or (3) using a faster model architecture."
            )

    return {
        "estimated_seconds": int(estimated),
        "time_limit_seconds": time_limit,
        "within_budget": within_budget,
        "recommendation": recommendation,
    }


def validate_tensor_dimensions(
    input_size: int,
    batch_size: int,
    n_features: int,
) -> dict[str, Any]:
    """Validate tensor dimensions for neural network models.

    Common failure modes:
    - input_size > available sequence length
    - batch_size too large for memory
    - n_features too large or zero

    Returns validation result dict.
    """
    result = {
        "valid": True,
        "warnings": [],
        "errors": [],
    }

    # Check input_size is reasonable
    if input_size > 365:
        result["warnings"].append(
            f"input_size={input_size} is very large (>1 year). "
            f"This may cause memory issues or slow training."
        )

    if input_size < 5:
        result["warnings"].append(
            f"input_size={input_size} is very small. "
            f"Model may not have enough context to learn patterns."
        )

    # Check batch_size
    if batch_size > 2048:
        result["warnings"].append(
            f"batch_size={batch_size} is large. "
            f"May cause OOM on GPU. Consider reducing to 512-1024."
        )

    if batch_size < 16:
        result["warnings"].append(
            f"batch_size={batch_size} is small. "
            f"Training may be slow due to underutilized GPU."
        )

    # Check features
    if n_features > 500:
        result["warnings"].append(
            f"n_features={n_features} is very large. "
            f"Consider feature selection or dimensionality reduction."
        )

    if n_features == 0:
        result["valid"] = False
        result["errors"].append("No features specified (n_features=0)")

    return result
