from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

# Register the custom layer so Keras can deserialise encoder.keras
from mt4599.models.transformer import LastTimestep  # noqa: F401


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract encoder embeddings from a trained Transformer for EuRoC windows.\n"
            "Saves emb_seq (N, W, d_model) and/or emb_last (N, d_model) plus\n"
            "per-window sequence origin metadata for downstream PCA/clustering/HMM."
        )
    )
    parser.add_argument("--dataset",    type=str, required=True,
        help="Path to windowed dataset .npz (from build_window_dataset.py).")
    parser.add_argument("--model-dir",  type=str, required=True,
        help="Directory containing encoder.keras (train_transformer.py output).")
    parser.add_argument("--split",      type=str, default="all",
        choices=["train", "val", "test", "all"],
        help="Which split(s) to extract embeddings for (default: all).")
    parser.add_argument("--representation", type=str, default="both",
        choices=["last", "sequence", "both"],
        help="Embedding representation to save (default: both).")
    parser.add_argument("--output",     type=str, required=True,
        help="Output .npz file for embeddings.")
    parser.add_argument("--batch-size", type=int, default=64,
        help="Batch size for inference (default: 64).")
    return parser.parse_args()


def _load_dataset(
    dataset_path: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    npz = np.load(dataset_path, allow_pickle=False)
    arrays = {
        "X_train": npz["X_train"],
        "X_val":   npz["X_val"],
        "X_test":  npz["X_test"],
    }
    meta_raw = npz.get("meta_json", None)
    meta: Dict[str, Any] = json.loads(str(meta_raw)) if meta_raw is not None else {}
    return arrays, meta


def _select_split(
    arrays: Dict[str, np.ndarray], split: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Return the data array and a list of split-labels per window.
    Labels are used downstream to map embeddings back to their origin split,
    which in turn maps to specific EuRoC sequences (via dataset metadata).
    """
    if split == "train":
        X = arrays["X_train"]
        labels = ["train"] * len(X)
    elif split == "val":
        X = arrays["X_val"]
        labels = ["val"] * len(X)
    elif split == "test":
        X = arrays["X_test"]
        labels = ["test"] * len(X)
    else:  # all
        X = np.concatenate([arrays["X_train"], arrays["X_val"], arrays["X_test"]], axis=0)
        labels = (
            ["train"] * len(arrays["X_train"]) +
            ["val"]   * len(arrays["X_val"])   +
            ["test"]  * len(arrays["X_test"])
        )
    return X, labels


def _extract_embeddings(
    encoder: tf.keras.Model,
    X: np.ndarray,
    batch_size: int,
    representation: str,
) -> Dict[str, np.ndarray]:
    """
    Run the encoder over X in batches.

    encoder output shape: (batch, W, d_model)  — full sequence of hidden states.

    Saved arrays:
        emb_seq  (N, W, d_model)  — per-timestep encoder states (sequence repr)
        emb_last (N, d_model)     — final-token state (used for PCA / clustering)
    """
    ds = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
    parts: List[np.ndarray] = []
    for batch in ds:
        parts.append(encoder(batch, training=False).numpy())

    if not parts:
        return {}

    emb_seq = np.concatenate(parts, axis=0)   # (N, W, d_model)
    outputs: Dict[str, np.ndarray] = {}

    if representation in ("sequence", "both"):
        outputs["emb_seq"]  = emb_seq
    if representation in ("last", "both"):
        outputs["emb_last"] = emb_seq[:, -1, :]   # (N, d_model)

    return outputs


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    model_dir    = Path(args.model_dir)
    output_path  = Path(args.output)

    arrays, dataset_meta = _load_dataset(dataset_path)
    X, split_labels      = _select_split(arrays, args.split)

    if X.shape[0] == 0:
        raise ValueError(f"No samples found in split '{args.split}'.")

    encoder = tf.keras.models.load_model(
        model_dir / "encoder.keras",
        custom_objects={"LastTimestep": LastTimestep},
    )

    emb_dict = _extract_embeddings(
        encoder=encoder,
        X=X,
        batch_size=args.batch_size,
        representation=args.representation,
    )
    if not emb_dict:
        raise ValueError("No embeddings produced.")

    # Per-window split labels as a UTF-8 byte array (numpy can't store str in npz)
    split_labels_arr = np.array(split_labels, dtype="S8")  # 8-char byte strings

    emb_meta: Dict[str, Any] = {
        "split":           args.split,
        "representation":  args.representation,
        "n_windows":       int(X.shape[0]),
        "window_length":   int(X.shape[1]),
        "feature_dim":     int(X.shape[2]),
        "d_model":         int(next(iter(emb_dict.values())).shape[-1]),
        "dataset_path":    str(dataset_path),
        "model_dir":       str(model_dir),
        # Sequence-level splits from the dataset — needed to map embeddings back
        # to specific EuRoC trajectories for Section 7.7 (sequence-level profiles)
        "train_sequences": dataset_meta.get("train_sequences", []),
        "val_sequences":   dataset_meta.get("val_sequences",   []),
        "test_sequences":  dataset_meta.get("test_sequences",  []),
        "window_length_ds": dataset_meta.get("window_length",  None),
        "stride":          dataset_meta.get("stride",          None),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        meta_json=json.dumps(emb_meta),
        split_labels=split_labels_arr,
        **emb_dict,
    )

    print(f"Saved embeddings to {output_path}")
    for key, value in emb_dict.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    print(f"  split_labels: {split_labels_arr.shape} "
          f"(train={split_labels.count('train')}, "
          f"val={split_labels.count('val')}, "
          f"test={split_labels.count('test')})")


if __name__ == "__main__":
    main()
