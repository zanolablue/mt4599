from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from mt4599.preprocessing import load_resampled_state_sequence


def _save_npz(output_path: Path, state: np.ndarray, t: np.ndarray, meta: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save metadata as JSON string to keep the npz simple
    meta_json = json.dumps(meta)
    np.savez_compressed(output_path, state=state, t=t, meta_json=meta_json)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess a single EuRoC sequence into state vectors at 200 Hz. "
            "State vectors have shape (T, 16) with [p, v, q, omega, accel]."
        )
    )
    parser.add_argument(
        "--seq-root",
        type=str,
        required=True,
        help="Path to a EuRoC sequence folder (e.g. data/raw/V1_01_easy).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .npz file (e.g. data/processed/V1_01_easy_state_200hz.npz).",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=200.0,
        help="Target sampling rate in Hz (default: 200.0).",
    )

    args = parser.parse_args()
    seq_root = Path(args.seq_root)
    output_path = Path(args.output)

    state, t, meta = load_resampled_state_sequence(seq_root, target_rate=args.rate_hz)

    # Basic assertions / sanity
    assert state.ndim == 2 and state.shape[1] == 16, f"Unexpected state shape: {state.shape}"
    assert t.ndim == 1 and t.shape[0] == state.shape[0], "t and state length mismatch."

    _save_npz(output_path, state, t, meta)

    print(f"Saved preprocessed sequence:")
    print(f"  seq_root : {meta.get('seq_root')}")
    print(f"  pose_src : {meta.get('pose_source')}")
    print(f"  rate_hz  : {meta.get('rate_hz')}")
    print(f"  steps    : {state.shape[0]}")
    print(f"  out      : {output_path}")


if __name__ == "__main__":
    main()

