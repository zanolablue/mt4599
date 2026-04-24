from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from mt4599.datasets import (
    SequenceData,
    build_windowed_datasets,
    load_sequences_from_manifest,
    _split_sequences,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build windowed train/val/test datasets for next-step prediction.\n"
            "Saves sequence-origin labels per window alongside X/y arrays.\n"
            "Uses a manifest produced by preprocess_multiple_sequences.py."
        )
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output",   type=str, required=True)
    parser.add_argument("--window",   type=int, default=128)
    parser.add_argument("--stride",   type=int, default=16)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac",   type=float, default=0.2)
    parser.add_argument("--test-frac",  type=float, default=0.2)
    parser.add_argument("--train-seqs", type=str, default="")
    parser.add_argument("--val-seqs",   type=str, default="")
    parser.add_argument("--test-seqs",  type=str, default="")
    return parser.parse_args()


def _parse_seq_list(arg: str) -> List[str]:
    return [s.strip() for s in arg.split(",") if s.strip()]


def _compute_splits(names, args):
    explicit_train = _parse_seq_list(args.train_seqs)
    explicit_val   = _parse_seq_list(args.val_seqs)
    explicit_test  = _parse_seq_list(args.test_seqs)
    if any([explicit_train, explicit_val, explicit_test]):
        return explicit_train, explicit_val, explicit_test
    sorted_names = sorted(names)
    return _split_sequences(sorted_names, args.train_frac, args.val_frac, args.test_frac)


def _build_seq_labels(
    sequences_map: Dict[str, Any],
    split_names: List[str],
    window: int,
    stride: int,
) -> np.ndarray:
    """
    Reconstruct per-window sequence origin labels for a given split.
    """
    labels = []
    for name in split_names:
        if name not in sequences_map:
            continue
        seq   = sequences_map[name]
        # SequenceData has a .state attribute of shape (T, D)
        T     = seq.state.shape[0]
        # Number of windows: floor((T - window) / stride) + 1
        n_win = max(0, (T - window) // stride + 1)
        labels.extend([name] * n_win)
    return np.array(labels, dtype="S64")   


def main() -> None:
    args          = _parse_args()
    manifest_path = Path(args.manifest)
    output_path   = Path(args.output)

    sequences_map = load_sequences_from_manifest(manifest_path)
    names         = sorted(sequences_map.keys())

    train_names, val_names, test_names = _compute_splits(names, args)
    if not train_names:
        raise ValueError("Training split is empty.")

    arrays, split_meta, mu, sigma = build_windowed_datasets(
        sequences=sequences_map,
        train_names=train_names,
        val_names=val_names,
        test_names=test_names,
        window=args.window,
        stride=args.stride,
    )

    seq_labels_train = _build_seq_labels(sequences_map, train_names, args.window, args.stride)
    seq_labels_val   = _build_seq_labels(sequences_map, val_names,   args.window, args.stride)
    seq_labels_test  = _build_seq_labels(sequences_map, test_names,  args.window, args.stride)

    for split, labels, key in [
        ("train", seq_labels_train, "X_train"),
        ("val",   seq_labels_val,   "X_val"),
        ("test",  seq_labels_test,  "X_test"),
    ]:
        n_windows = arrays[key].shape[0]
        if len(labels) != n_windows:
            print(f"  Warning: {split} label count {len(labels)} != window count {n_windows}. "
                  f"Using truncation/padding to match.")
            if len(labels) < n_windows:
                labels = np.concatenate([
                    labels,
                    np.array([labels[-1]] * (n_windows - len(labels)), dtype="S64")
                ])
            else:
                labels = labels[:n_windows]
        if split == "train":
            seq_labels_train = labels
        elif split == "val":
            seq_labels_val = labels
        else:
            seq_labels_test = labels

    meta: Dict[str, Any] = {
        "train_sequences": split_meta["train_sequences"],
        "val_sequences":   split_meta["val_sequences"],
        "test_sequences":  split_meta["test_sequences"],
        "window_length":   split_meta["window_length"],
        "stride":          split_meta["stride"],
        "feature_dim":     int(arrays["X_train"].shape[-1]),
        "rate_hz":         200.0,
        "task":            "next_step",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        meta_json=json.dumps(meta),
        mu=mu, sigma=sigma,
        seq_labels_train=seq_labels_train,
        seq_labels_val=seq_labels_val,
        seq_labels_test=seq_labels_test,
        **arrays,
    )

    print(f"Saved windowed dataset to {output_path}")
    print(f"  train: {arrays['X_train'].shape[0]} windows  "
          f"seqs={train_names}")
    print(f"  val:   {arrays['X_val'].shape[0]} windows  "
          f"seqs={val_names}")
    print(f"  test:  {arrays['X_test'].shape[0]} windows  "
          f"seqs={test_names}")
    print(f"  seq_labels saved: train={len(seq_labels_train)} "
          f"val={len(seq_labels_val)} test={len(seq_labels_test)}")


if __name__ == "__main__":
    main()
