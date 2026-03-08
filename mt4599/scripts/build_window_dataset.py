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
            "Build windowed train/val/test datasets for next-step prediction from processed EuRoC sequences.\n"
            "Uses a manifest produced by preprocess_multiple_sequences.py."
        )
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest JSON produced by preprocess_multiple_sequences.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output dataset .npz file.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=128,
        help="Input window length W (default: 128).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Stride S between windows (default: 16).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.6,
        help="Fraction of sequences assigned to train (default: 0.6).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of sequences assigned to validation (default: 0.2).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of sequences assigned to test (default: 0.2).",
    )
    parser.add_argument(
        "--train-seqs",
        type=str,
        default="",
        help="Optional comma-separated list of sequence names to force into the training split.",
    )
    parser.add_argument(
        "--val-seqs",
        type=str,
        default="",
        help="Optional comma-separated list of sequence names to force into the validation split.",
    )
    parser.add_argument(
        "--test-seqs",
        type=str,
        default="",
        help="Optional comma-separated list of sequence names to force into the test split.",
    )
    return parser.parse_args()


def _parse_seq_list(arg: str) -> List[str]:
    return [s.strip() for s in arg.split(",") if s.strip()]


def _compute_splits(
    names: List[str],
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], List[str]]:
    explicit_train = _parse_seq_list(args.train_seqs)
    explicit_val = _parse_seq_list(args.val_seqs)
    explicit_test = _parse_seq_list(args.test_seqs)

    if any([explicit_train, explicit_val, explicit_test]):
        # Use explicit lists; any sequences not mentioned will be ignored.
        train_names = explicit_train
        val_names = explicit_val
        test_names = explicit_test
    else:
        sorted_names = sorted(names)
        train_names, val_names, test_names = _split_sequences(
            sorted_names, args.train_frac, args.val_frac, args.test_frac
        )
    return train_names, val_names, test_names


def _save_dataset(
    output_path: Path,
    arrays: Dict[str, np.ndarray],
    mu: np.ndarray,
    sigma: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta)
    np.savez_compressed(output_path, meta_json=meta_json, mu=mu, sigma=sigma, **arrays)


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    sequences_map = load_sequences_from_manifest(manifest_path)
    names = sorted(sequences_map.keys())

    train_names, val_names, test_names = _compute_splits(names, args)
    if not train_names:
        raise ValueError("Training split is empty after applying split configuration.")

    arrays, split_meta, mu, sigma = build_windowed_datasets(
        sequences=sequences_map,
        train_names=train_names,
        val_names=val_names,
        test_names=test_names,
        window=args.window,
        stride=args.stride,
    )

    # Add some global metadata fields
    meta: Dict[str, Any] = {
        "train_sequences": split_meta["train_sequences"],
        "val_sequences": split_meta["val_sequences"],
        "test_sequences": split_meta["test_sequences"],
        "window_length": split_meta["window_length"],
        "stride": split_meta["stride"],
        "feature_dim": int(arrays["X_train"].shape[-1]),
        "rate_hz": 200.0,  # current preprocessing convention
        "task": "next_step",
    }

    _save_dataset(output_path, arrays, mu, sigma, meta)

    print(f"Saved windowed dataset to {output_path}")
    print(f"  train windows: {arrays['X_train'].shape[0]}")
    print(f"  val windows  : {arrays['X_val'].shape[0]}")
    print(f"  test windows : {arrays['X_test'].shape[0]}")


if __name__ == "__main__":
    main()

