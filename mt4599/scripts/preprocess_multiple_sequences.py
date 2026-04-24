from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mt4599.preprocessing import load_resampled_state_sequence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess multiple EuRoC sequences into state vectors at 200 Hz.\n"
            "For each valid sequence, saves a .npz with state (T, 16), timestamps, and metadata,\n"
            "and writes a manifest JSON summarising successes and failures."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-root",
        type=str,
        help=(
            "Directory containing multiple extracted EuRoC sequences, each as a subdirectory "
            "(e.g. data/raw/V1_01_easy, data/raw/MH_01_easy, ...)."
        ),
    )
    group.add_argument(
        "--seq-roots",
        type=str,
        help=(
            "Comma-separated list of explicit EuRoC sequence root paths. "
            "Example: --seq-roots data/raw/V1_01_easy,data/raw/MH_01_easy"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store processed .npz files (will be created if needed).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest JSON file summarising all processed sequences.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=200.0,
        help="Target sampling rate in Hz (default: 200.0).",
    )
    return parser.parse_args()


def _discover_sequence_roots(input_root: Optional[Path], seq_roots_arg: Optional[str]) -> List[Path]:
    if input_root is not None:
        if not input_root.exists() or not input_root.is_dir():
            raise NotADirectoryError(f"--input-root is not a directory: {input_root}")
        # Any immediate child directory is treated as a candidate sequence root.
        candidates = [p for p in input_root.iterdir() if p.is_dir()]
        return sorted(candidates)

    assert seq_roots_arg is not None
    roots: List[Path] = []
    for part in seq_roots_arg.split(","):
        s = part.strip()
        if not s:
            continue
        p = Path(s)
        if not p.exists() or not p.is_dir():
            print(f"Warning: sequence root does not exist or is not a directory: {p}", file=sys.stderr)
            continue
        roots.append(p)
    return roots


def _save_npz(output_path: Path, state: np.ndarray, t: np.ndarray, meta: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta)
    np.savez_compressed(output_path, state=state, t=t, meta_json=meta_json)


def main() -> None:
    args = _parse_args()
    input_root = Path(args.input_root) if args.input_root is not None else None
    seq_roots_arg = args.seq_roots
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    seq_roots = _discover_sequence_roots(input_root, seq_roots_arg)
    if not seq_roots:
        print("No candidate sequence roots found. Nothing to do.", file=sys.stderr)
        sys.exit(1)

    manifest: Dict[str, Any] = {"sequences": []}

    for seq_root in seq_roots:
        name = seq_root.name
        record: Dict[str, Any] = {
            "name": name,
            "seq_root": str(seq_root),
            "processed_path": None,
            "pose_source": None,
            "num_timesteps": 0,
            "feature_dim": None,
            "rate_hz": None,
            "status": "failed",
            "error": None,
        }

        try:
            print(f"Processing sequence: {seq_root}")
            state, t, meta = load_resampled_state_sequence(seq_root, target_rate=args.rate_hz)

            if state.ndim != 2 or state.shape[1] != 16:
                raise ValueError(f"Unexpected state shape {state.shape}, expected (T, 16).")
            if t.ndim != 1 or t.shape[0] != state.shape[0]:
                raise ValueError("Timestamp array length does not match state array length.")
            if not (np.all(np.isfinite(state)) and np.all(np.isfinite(t))):
                raise ValueError("Non-finite values found in state or timestamps.")

            out_path = output_dir / f"{name}_state_{int(args.rate_hz)}hz.npz"
            _save_npz(out_path, state, t, meta)

            record.update(
                {
                    "processed_path": str(out_path),
                    "pose_source": meta.get("pose_source"),
                    "num_timesteps": int(state.shape[0]),
                    "feature_dim": int(state.shape[1]),
                    "rate_hz": float(meta.get("rate_hz", args.rate_hz)),
                    "status": "ok",
                }
            )

            print(
                f"  OK: {name} | pose_source={record['pose_source']} "
                f"| T={record['num_timesteps']} | out={out_path}"
            )
        except Exception as exc:  # pragma: no cover - robustness path
            record["error"] = str(exc)
            print(f"  FAILED: {name} | {exc}", file=sys.stderr)

        manifest["sequences"].append(record)

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote manifest with {len(manifest['sequences'])} entries to {manifest_path}")


if __name__ == "__main__":
    main()

