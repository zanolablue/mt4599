from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract encoder embeddings from a trained Transformer model for EuRoC windows.\n"
            "Loads the encoder SavedModel and a windowed dataset .npz."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to windowed dataset .npz (from build_window_dataset.py).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the trained encoder SavedModel (train_transformer.py output).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Which split to extract embeddings for (default: train).",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="both",
        choices=["last", "sequence", "both"],
        help="Which embedding representation to save (default: both).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npz file for embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding extraction (default: 64).",
    )
    return parser.parse_args()


def _load_dataset(dataset_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    npz = np.load(dataset_path, allow_pickle=False)
    arrays = {
        "X_train": npz["X_train"],
        "X_val": npz["X_val"],
        "X_test": npz["X_test"],
    }
    meta_raw = npz.get("meta_json", None)
    meta: Dict[str, Any] = json.loads(str(meta_raw)) if meta_raw is not None else {}
    return arrays, meta


def _select_split(
    arrays: Dict[str, np.ndarray], split: str
) -> Tuple[np.ndarray, str]:
    if split == "train":
        return arrays["X_train"], "train"
    if split == "val":
        return arrays["X_val"], "val"
    if split == "test":
        return arrays["X_test"], "test"
    # 'all': concatenate in train, val, test order and tag split='all'
    X_all = np.concatenate(
        [arrays["X_train"], arrays["X_val"], arrays["X_test"]], axis=0
    )
    return X_all, "all"


def _extract_embeddings(
    encoder: tf.keras.Model,
    X: np.ndarray,
    batch_size: int,
    representation: str,
) -> Dict[str, np.ndarray]:
    ds = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
    seq_emb_list = []
    for batch in ds:
        seq_emb = encoder(batch, training=False)
        seq_emb_list.append(seq_emb.numpy())
    if not seq_emb_list:
        return {}

    emb_seq = np.concatenate(seq_emb_list, axis=0)
    outputs: Dict[str, np.ndarray] = {}

    if representation in ("sequence", "both"):
        outputs["emb_seq"] = emb_seq
    if representation in ("last", "both"):
        outputs["emb_last"] = emb_seq[:, -1, :]
    return outputs


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)

    arrays, meta = _load_dataset(dataset_path)
    X, split_used = _select_split(arrays, args.split)

    if X.shape[0] == 0:
        raise ValueError(f"No samples found in split '{split_used}'.")

    encoder = tf.keras.models.load_model(model_dir / "encoder")

    emb_dict = _extract_embeddings(
        encoder=encoder,
        X=X,
        batch_size=args.batch_size,
        representation=args.representation,
    )
    if not emb_dict:
        raise ValueError("No embeddings produced.")

    # Build metadata for embeddings
    emb_meta: Dict[str, Any] = {
        "split": split_used,
        "representation": args.representation,
        "window_length": int(X.shape[1]),
        "feature_dim": int(X.shape[2]),
        "d_model": int(next(iter(emb_dict.values())).shape[-1]),
        "dataset_path": str(dataset_path),
        "model_dir": str(model_dir),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, meta_json=json.dumps(emb_meta), **emb_dict)

    print(f"Saved embeddings to {output_path}")
    for key, value in emb_dict.items():
        print(f"  {key}: shape={value.shape}")


if __name__ == "__main__":
    main()

