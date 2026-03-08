from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from mt4599.models.transformer import TransformerConfig, build_transformer_models


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Transformer encoder for next-step prediction on EuRoC windowed datasets.\n"
            "Expects a dataset .npz produced by build_window_dataset.py."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to windowed dataset .npz (from build_window_dataset.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store model, encoder, history, and config.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=128,
        help="Window length W (must match dataset; default: 128).",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Transformer model dimension (default: 128).",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of encoder layers (default: 3).",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=256,
        help="Feed-forward hidden dimension (default: 256).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of epochs (default: 50).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam (default: 1e-3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def _load_dataset(path: Path) -> Dict[str, Any]:
    npz = np.load(path, allow_pickle=False)
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "mu", "sigma"]
    for key in required:
        if key not in npz:
            raise ValueError(f"Dataset {path} is missing required array '{key}'.")
    data: Dict[str, Any] = {k: npz[k] for k in npz.files}

    meta_raw = data.get("meta_json", None)
    meta: Dict[str, Any] = json.loads(str(meta_raw)) if meta_raw is not None else {}
    data["meta"] = meta
    return data


def _build_datasets(
    arrays: Dict[str, Any], batch_size: int
) -> Dict[str, tf.data.Dataset]:
    X_train = arrays["X_train"]
    y_train = arrays["y_train"]
    X_val = arrays["X_val"]
    y_val = arrays["y_val"]
    X_test = arrays["X_test"]
    y_test = arrays["y_test"]

    def make_ds(X: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if training:
            ds = ds.shuffle(buffer_size=min(len(X), 10_000))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    return {
        "train": make_ds(X_train, y_train, training=True),
        "val": make_ds(X_val, y_val, training=False),
        "test": make_ds(X_test, y_test, training=False),
    }


def _save_history(history: tf.keras.callbacks.History, path: Path) -> None:
    hist_dict = history.history
    with path.open("w") as f:
        json.dump(hist_dict, f, indent=2)


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Basic seeding for reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    arrays = _load_dataset(dataset_path)
    meta = arrays["meta"]

    X_train = arrays["X_train"]
    window_length = X_train.shape[1]
    feature_dim = X_train.shape[2]

    if window_length != args.window:
        raise ValueError(
            f"Window length mismatch: dataset has {window_length}, but --window={args.window}."
        )

    config = TransformerConfig(
        window_length=window_length,
        feature_dim=feature_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )

    model, encoder = build_transformer_models(config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=["mse", "mae"],
    )

    datasets = _build_datasets(arrays, batch_size=args.batch_size)

    ckpt_path = output_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Evaluate on test set
    test_metrics = model.evaluate(datasets["test"], return_dict=True)

    # Save models
    model.save(output_dir / "model")
    encoder.save(output_dir / "encoder")

    # Save history and metrics
    _save_history(history, output_dir / "history.json")
    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save config and dataset meta
    full_config: Dict[str, Any] = {
        "transformer_config": config.to_dict(),
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "dataset": {
            "path": str(dataset_path),
            "meta": meta,
        },
    }
    with (output_dir / "config.json").open("w") as f:
        json.dump(full_config, f, indent=2)

    print("Training complete.")
    print(f"  Model directory: {output_dir}")
    print(f"  Best checkpoint: {ckpt_path if ckpt_path.exists() else 'n/a'}")
    print(f"  Test metrics   : {test_metrics}")


if __name__ == "__main__":
    # Disable excessive TF logs by default
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()

