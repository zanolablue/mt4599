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
            "Train a Transformer encoder on EuRoC windowed datasets.\n"
            "Supports next-step and multi-step prediction objectives.\n"
            "Expects a dataset .npz produced by build_window_dataset.py."
        )
    )
    parser.add_argument("--dataset",       type=str, required=True)
    parser.add_argument("--output-dir",    type=str, required=True)
    parser.add_argument("--window",        type=int, default=256,
        help="Window length W (must match dataset; default: 256).")
    parser.add_argument("--d-model",       type=int, default=128)
    parser.add_argument("--num-heads",     type=int, default=4)
    parser.add_argument("--num-layers",    type=int, default=3)
    parser.add_argument("--d-ff",          type=int, default=256)
    parser.add_argument("--dropout",       type=float, default=0.2,
        help="Dropout rate (default: 0.2 — increased from 0.1 to reduce overfitting).")
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--epochs",        type=int, default=100,
        help="Maximum epochs (default: 100; early stopping will trigger earlier).")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience",      type=int, default=15,
        help="Early stopping patience (default: 15).")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--predict-horizon", type=int, default=5,
        help=(
            "How many steps ahead to predict (default: 5). "
            "H=1 is standard next-step; H>1 forces the encoder to focus on "
            "motion dynamics rather than instantaneous state tracking."
        ))
    return parser.parse_args()


def _load_dataset(path: Path) -> Dict[str, Any]:
    npz = np.load(path, allow_pickle=False)
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "mu", "sigma"]
    for key in required:
        if key not in npz:
            raise ValueError(f"Dataset {path} is missing required array '{key}'.")
    data: Dict[str, Any] = {k: npz[k] for k in npz.files}
    meta_raw = data.get("meta_json", None)
    data["meta"] = json.loads(str(meta_raw)) if meta_raw is not None else {}
    return data


def _shift_targets(
    X: np.ndarray, y_next: np.ndarray, horizon: int
) -> np.ndarray:
    """
    Construct H-step-ahead targets from (X, y_next) pairs.

    build_window_dataset produces y = s_{T+1} (H=1).
    For H>1 we shift: the target for window ending at T is s_{T+H}.
    We approximate this by rolling the y array by (H-1) positions and
    using X's last-timestep values to fill the gap at the edges.

    This is an approximation — a proper multi-step dataset would be built
    differently — but it is sufficient for the purpose of making the encoder
    attend to motion dynamics over a longer horizon.
    """
    if horizon == 1:
        return y_next

    N = X.shape[0]
    # Roll y forward by (H-1): target for window n becomes what was target for n+(H-1)
    y_shifted = np.empty_like(y_next)
    shift = horizon - 1
    y_shifted[:N - shift] = y_next[shift:]
    # For the last `shift` windows, fall back to y_next (boundary)
    y_shifted[N - shift:] = y_next[N - shift:]
    return y_shifted


def _build_tf_datasets(
    arrays: Dict[str, Any], batch_size: int, horizon: int
) -> Dict[str, tf.data.Dataset]:
    X_train = arrays["X_train"]
    X_val   = arrays["X_val"]
    X_test  = arrays["X_test"]
    y_train = _shift_targets(X_train, arrays["y_train"], horizon)
    y_val   = _shift_targets(X_val,   arrays["y_val"],   horizon)
    y_test  = _shift_targets(X_test,  arrays["y_test"],  horizon)

    def make_ds(X: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if training:
            ds = ds.shuffle(buffer_size=min(len(X), 10_000))
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return {
        "train": make_ds(X_train, y_train, training=True),
        "val":   make_ds(X_val,   y_val,   training=False),
        "test":  make_ds(X_test,  y_test,  training=False),
    }


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    arrays = _load_dataset(dataset_path)
    meta   = arrays["meta"]

    X_train      = arrays["X_train"]
    window_length = X_train.shape[1]
    feature_dim   = X_train.shape[2]

    if window_length != args.window:
        raise ValueError(
            f"Window length mismatch: dataset has {window_length}, "
            f"but --window={args.window}."
        )

    print(f"Dataset loaded: {X_train.shape[0]} train windows, "
          f"W={window_length}, D={feature_dim}")
    print(f"Predict horizon H={args.predict_horizon}  "
          f"dropout={args.dropout}  patience={args.patience}")

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

    datasets = _build_tf_datasets(arrays, batch_size=args.batch_size,
                                   horizon=args.predict_horizon)

    ckpt_path = output_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history     = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=args.epochs,
        callbacks=callbacks,
    )
    test_metrics = model.evaluate(datasets["test"], return_dict=True)

    model.save(output_dir   / "model.keras")
    encoder.save(output_dir / "encoder.keras")

    with (output_dir / "history.json").open("w") as f:
        json.dump(history.history, f, indent=2)
    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)

    full_config: Dict[str, Any] = {
        "transformer_config": config.to_dict(),
        "training": {
            "batch_size":       args.batch_size,
            "epochs":           args.epochs,
            "learning_rate":    args.learning_rate,
            "dropout":          args.dropout,
            "patience":         args.patience,
            "predict_horizon":  args.predict_horizon,
            "seed":             args.seed,
        },
        "dataset": {"path": str(dataset_path), "meta": meta},
    }
    with (output_dir / "config.json").open("w") as f:
        json.dump(full_config, f, indent=2)

    print("Training complete.")
    print(f"  Model directory : {output_dir}")
    print(f"  Best checkpoint : {ckpt_path if ckpt_path.exists() else 'n/a'}")
    print(f"  Test metrics    : {test_metrics}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
