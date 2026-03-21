from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from mt4599.models.transformer import LastTimestep  # noqa: F401


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       type=str, required=True)
    parser.add_argument("--model-dir",     type=str, required=True)
    parser.add_argument("--split",         type=str, default="all",
                        choices=["train","val","test","all"])
    parser.add_argument("--representation",type=str, default="both",
                        choices=["last","sequence","both"])
    parser.add_argument("--output",        type=str, required=True)
    parser.add_argument("--batch-size",    type=int, default=64)
    return parser.parse_args()


def _load_dataset(path: Path):
    npz  = np.load(path, allow_pickle=False)
    arrays = {k: npz[k] for k in ["X_train","X_val","X_test"]}
    meta   = json.loads(str(npz["meta_json"])) if "meta_json" in npz else {}

    # Load sequence origin labels if present
    seq_labels = {}
    for split in ["train","val","test"]:
        key = f"seq_labels_{split}"
        if key in npz:
            seq_labels[split] = npz[key].astype(str)
        else:
            # Fallback: label everything with split name
            seq_labels[split] = np.array(
                [split] * arrays[f"X_{split}"].shape[0], dtype=str)

    return arrays, seq_labels, meta


def _select_split(arrays, seq_labels, split):
    if split == "all":
        X = np.concatenate([arrays["X_train"], arrays["X_val"], arrays["X_test"]], axis=0)
        labels = np.concatenate([
            seq_labels["train"], seq_labels["val"], seq_labels["test"]
        ])
        split_tags = np.array(
            ["train"] * len(arrays["X_train"]) +
            ["val"]   * len(arrays["X_val"])   +
            ["test"]  * len(arrays["X_test"])
        )
    else:
        X          = arrays[f"X_{split}"]
        labels     = seq_labels[split]
        split_tags = np.array([split] * len(X))
    return X, labels, split_tags


def _extract(encoder, X, batch_size, representation):
    ds    = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
    parts = [encoder(b, training=False).numpy() for b in ds]
    if not parts:
        return {}
    emb_seq = np.concatenate(parts, axis=0)   # (N, W, d_model)
    out = {}
    if representation in ("sequence","both"):
        out["emb_seq"]  = emb_seq
    if representation in ("last","both"):
        out["emb_last"] = emb_seq[:, -1, :]
    # Always save mean-pooled — needed for sequence-level analysis
    out["emb_mean"] = emb_seq.mean(axis=1)
    return out


def main() -> None:
    args         = _parse_args()
    dataset_path = Path(args.dataset)
    model_dir    = Path(args.model_dir)
    output_path  = Path(args.output)

    arrays, seq_labels, meta = _load_dataset(dataset_path)
    X, seq_origin, split_tags = _select_split(arrays, seq_labels, args.split)

    if X.shape[0] == 0:
        raise ValueError(f"No samples in split '{args.split}'.")

    encoder = tf.keras.models.load_model(
        model_dir / "encoder.keras",
        custom_objects={"LastTimestep": LastTimestep},
    )

    emb_dict = _extract(encoder, X, args.batch_size, args.representation)
    if not emb_dict:
        raise ValueError("No embeddings produced.")

    emb_meta: Dict[str, Any] = {
        "split":           args.split,
        "representation":  args.representation,
        "n_windows":       int(X.shape[0]),
        "window_length":   int(X.shape[1]),
        "feature_dim":     int(X.shape[2]),
        "d_model":         int(next(iter(emb_dict.values())).shape[-1]),
        "dataset_path":    str(dataset_path),
        "model_dir":       str(model_dir),
        "train_sequences": meta.get("train_sequences", []),
        "val_sequences":   meta.get("val_sequences",   []),
        "test_sequences":  meta.get("test_sequences",  []),
        "window_length_ds": meta.get("window_length",  None),
        "stride":          meta.get("stride",          None),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        meta_json=json.dumps(emb_meta),
        seq_origin=seq_origin.astype("S64"),     # e.g. "MH_01_easy"
        split_labels=split_tags.astype("S8"),    # "train"/"val"/"test"
        **emb_dict,
    )

    print(f"Saved embeddings to {output_path}")
    for k, v in emb_dict.items():
        print(f"  {k}: {v.shape}")
    seqs, counts = np.unique(seq_origin, return_counts=True)
    print(f"  seq_origin: {len(seqs)} unique sequences")
    for s, c in zip(seqs, counts):
        print(f"    {s}: {c} windows")


if __name__ == "__main__":
    main()
