from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import json

import numpy as np


@dataclass
class SequenceData:
    name: str
    state: np.ndarray  # (T, D)
    meta: Dict


def load_processed_sequence(npz_path: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load a single processed EuRoC sequence .npz produced by the preprocessing step.

    Expects keys:
      - 'state'  (T, D)
      - 't'      (T,)
      - 'meta_json' (JSON-encoded metadata dict)
    """
    npz = np.load(npz_path, allow_pickle=False)
    if "state" not in npz or "t" not in npz:
        raise ValueError(f"{npz_path} is missing 'state' or 't' arrays.")
    state = npz["state"]
    t = npz["t"]
    meta_raw = npz.get("meta_json", None)
    meta = json.loads(str(meta_raw)) if meta_raw is not None else {}

    if state.ndim != 2:
        raise ValueError(f"Expected 2D state array in {npz_path}, got shape {state.shape}.")
    if t.ndim != 1 or t.shape[0] != state.shape[0]:
        raise ValueError(f"t and state length mismatch in {npz_path}: {t.shape}, {state.shape}.")
    if state.shape[1] != 16:
        raise ValueError(f"Expected feature_dim 16 in {npz_path}, got {state.shape[1]}.")
    if not (np.all(np.isfinite(state)) and np.all(np.isfinite(t))):
        raise ValueError(f"Non-finite values in state or timestamps for {npz_path}.")

    return state.astype(np.float32), meta


def load_sequences_from_manifest(manifest_path: Path) -> Mapping[str, SequenceData]:
    """
    Load all successful sequences from a multi-sequence manifest.

    Manifest format is produced by preprocess_multiple_sequences.py.
    """
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    if "sequences" not in manifest:
        raise ValueError(f"Manifest {manifest_path} does not contain 'sequences' key.")

    sequences: Dict[str, SequenceData] = {}
    for entry in manifest["sequences"]:
        if entry.get("status") != "ok":
            continue
        name = entry["name"]
        processed_path = entry.get("processed_path")
        if not processed_path:
            continue

        npz_path = Path(processed_path)
        state, meta = load_processed_sequence(npz_path)
        sequences[name] = SequenceData(name=name, state=state, meta=meta)
    if not sequences:
        raise ValueError(f"No successful sequences found in manifest {manifest_path}.")
    return sequences


def _split_sequences(
    names: Sequence[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Tuple[List[str], List[str], List[str]]:
    if not names:
        raise ValueError("No sequence names provided for splitting.")
    if train_frac <= 0.0 or val_frac < 0.0 or test_frac < 0.0:
        raise ValueError("Fractions must be non-negative, with train_frac > 0.")
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"Fractions must sum to 1.0, got {total}.")

    n = len(names)
    n_train = max(1, int(round(train_frac * n)))
    n_val = int(round(val_frac * n))
    # Ensure we don't exceed n and leave at least one for test if test_frac > 0
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val
    if test_frac > 0.0 and n_test == 0:
        # Steal one from train if needed
        if n_train > 1:
            n_train -= 1
            n_test = 1

    train_names = list(names[:n_train])
    val_names = list(names[n_train : n_train + n_val])
    test_names = list(names[n_train + n_val :])
    return train_names, val_names, test_names


def compute_normalisation(
    sequences: Iterable[SequenceData],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute feature-wise mean and std from a collection of training sequences.
    """
    states = [seq.state for seq in sequences]
    if not states:
        raise ValueError("No training sequences provided for normalisation.")

    concat = np.concatenate(states, axis=0)
    mu = concat.mean(axis=0)
    sigma = concat.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1e-6, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


def window_sequence(
    state: np.ndarray,
    window: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create next-step supervised windows from a single normalised state sequence.

    Args:
        state: (T, D) normalised state sequence.
        window: input window length W.
        stride: stride between windows.

    Returns:
        X: (N, W, D)
        y: (N, D)
    """
    T, D = state.shape
    if T <= window:
        return np.zeros((0, window, D), dtype=state.dtype), np.zeros((0, D), dtype=state.dtype)

    indices: List[int] = []
    i = 0
    while i + window < T:
        indices.append(i)
        i += stride

    N = len(indices)
    X = np.empty((N, window, D), dtype=state.dtype)
    y = np.empty((N, D), dtype=state.dtype)

    for k, start in enumerate(indices):
        end = start + window
        X[k] = state[start:end]
        y[k] = state[end]

    return X, y


def build_windowed_datasets(
    sequences: Mapping[str, SequenceData],
    train_names: Sequence[str],
    val_names: Sequence[str],
    test_names: Sequence[str],
    window: int,
    stride: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Build windowed train/val/test datasets and normalisation stats.

    Args:
        sequences: mapping from name to SequenceData.
        train_names, val_names, test_names: sequence names per split.
        window: window length W.
        stride: stride S.

    Returns:
        arrays: dict with X_*/y_* arrays.
        meta: dict with split sequence names and window/stride.
        mu: (D,) mean.
        sigma: (D,) std.
    """
    all_names = set(sequences.keys())
    for split_names in (train_names, val_names, test_names):
        for name in split_names:
            if name not in all_names:
                raise KeyError(f"Sequence '{name}' not found in loaded sequences.")

    train_seqs = [sequences[name] for name in train_names]
    val_seqs = [sequences[name] for name in val_names]
    test_seqs = [sequences[name] for name in test_names]

    if not train_seqs:
        raise ValueError("Training split is empty.")

    mu, sigma = compute_normalisation(train_seqs)

    def normalise(state: np.ndarray) -> np.ndarray:
        return (state - mu[None, :]) / sigma[None, :]

    def build_split(split_sequences: Iterable[SequenceData]) -> Tuple[np.ndarray, np.ndarray]:
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        for seq in split_sequences:
            state_norm = normalise(seq.state)
            X_seq, y_seq = window_sequence(state_norm, window=window, stride=stride)
            if X_seq.shape[0] == 0:
                continue
            X_list.append(X_seq)
            y_list.append(y_seq)
        if not X_list:
            return (
                np.zeros((0, window, mu.shape[0]), dtype=np.float32),
                np.zeros((0, mu.shape[0]), dtype=np.float32),
            )
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X, y

    X_train, y_train = build_split(train_seqs)
    X_val, y_val = build_split(val_seqs)
    X_test, y_test = build_split(test_seqs)

    for name, arr in [
        ("X_train", X_train),
        ("y_train", y_train),
        ("X_val", X_val),
        ("y_val", y_val),
        ("X_test", X_test),
        ("y_test", y_test),
    ]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values found in {name}.")

    if X_train.shape[0] == 0:
        raise ValueError("No training windows created. Check window/stride and sequence lengths.")
    if X_val.shape[0] == 0:
        raise ValueError("No validation windows created.")
    if X_test.shape[0] == 0:
        raise ValueError("No test windows created.")

    arrays: Dict[str, np.ndarray] = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    meta: Dict[str, object] = {
        "train_sequences": list(train_names),
        "val_sequences": list(val_names),
        "test_sequences": list(test_names),
        "window_length": int(window),
        "stride": int(stride),
    }
    return arrays, meta, mu, sigma


