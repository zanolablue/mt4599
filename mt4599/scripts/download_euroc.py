from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict
from urllib.request import urlretrieve


# Minimal curated subset of EuRoC MAV sequences with direct URLs.
# These are the "zip" archives containing the EuRoC folder structure
# (mav0, cam0, cam1, etc.) once extracted.
DATASETS: Dict[str, str] = {
    # Vicon room 1
    "V1_01_easy": (
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/"
        "vicon_room1/V1_01_easy/V1_01_easy.zip"
    ),
    # Machine Hall 01 (example machine hall sequence)
    "MH_01_easy": (
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/"
        "machine_hall/MH_01_easy/MH_01_easy.zip"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected EuRoC MAV sequences (zip archives) into a target directory. "
            "By default, it supports a small curated subset (e.g. V1_01_easy, MH_01_easy)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory in which to store downloaded zip files (will be created if needed).",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default="V1_01_easy",
        help=(
            "Comma-separated list of EuRoC sequence names to download. "
            f"Supported: {', '.join(sorted(DATASETS.keys()))}. "
            "Example: --sequences V1_01_easy,MH_01_easy"
        ),
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help=(
            "If set, do not download; only print the URLs and suggested wget commands. "
            "Useful when you want to copy commands to another machine."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing files with the same name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = [s.strip() for s in args.sequences.split(",") if s.strip()]
    unsupported = [s for s in requested if s not in DATASETS]
    if unsupported:
        print(
            f"Error: unsupported sequences requested: {', '.join(unsupported)}\n"
            f"Supported sequences: {', '.join(sorted(DATASETS.keys()))}",
            file=sys.stderr,
        )
        sys.exit(1)

    for name in requested:
        url = DATASETS[name]
        filename = f"{name}.zip"
        dest = out_dir / filename

        if args.print_only:
            print(f"# {name}")
            print(f"# URL: {url}")
            print(f"wget -O {filename} {url}")
            print()
            continue

        if dest.exists() and not args.overwrite:
            print(f"Skipping {name}: {dest} already exists (use --overwrite to replace).")
            continue

        print(f"Downloading {name} from {url}")
        print(f"  -> {dest}")
        try:
            urlretrieve(url, dest)
        except Exception as exc:  # pragma: no cover - simple CLI helper
            print(f"Failed to download {name} from {url}: {exc}", file=sys.stderr)
            continue

        print(f"Finished {name}")

    if args.print_only:
        print(
            "# Copy the printed wget commands to another machine if needed, "
            "then extract the zip files and point preprocessing at the extracted folders."
        )


if __name__ == "__main__":
    main()

