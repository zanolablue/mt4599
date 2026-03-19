from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


SeqRootLike = Union[str, Path]


@dataclass
class EurocFilePaths:
    imu: Path
    pose: Path
    pose_source: str


def _ensure_path(path: SeqRootLike) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sequence root does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Sequence root is not a directory: {p}")
    return p


def find_euroc_files(seq_root: SeqRootLike) -> EurocFilePaths:
    """
    Detect IMU and pose / ground-truth CSV files under a EuRoC sequence root.

    Priority:
    - IMU:   mav0/imu0/data.csv (required for MVP)
    - Pose:  mav0/state_groundtruth_estimate0/data.csv (primary)
             vicon0/data.csv (fallback)
    """
    root = _ensure_path(seq_root)

    # IMU (required)
    imu_candidates = [
        root / "mav0" / "imu0" / "data.csv",
        root / "imu0" / "data.csv",
    ]
    imu_path = next((p for p in imu_candidates if p.exists()), None)
    if imu_path is None:
        # Try a slightly more flexible fallback: any */imu0/data.csv directly under root
        for child in root.iterdir():
            candidate = child / "imu0" / "data.csv"
            if candidate.exists():
                imu_path = candidate
                break
    if imu_path is None:
        raise FileNotFoundError(
            f"Could not find IMU CSV under {root}. "
            "Tried mav0/imu0/data.csv, imu0/data.csv, and */imu0/data.csv."
        )

    # Pose / ground-truth
    pose_source = "unknown"
    pose_path = root / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    if pose_path.exists():
        pose_source = "state_groundtruth"
    else:
        # Fallback to Vicon if available
        pose_path = root / "vicon0" / "data.csv"
        if pose_path.exists():
            pose_source = "vicon"
        else:
            raise FileNotFoundError(
                f"Could not find pose / ground-truth CSV under {root}. "
                "Expected mav0/state_groundtruth_estimate0/data.csv or vicon0/data.csv."
            )

    return EurocFilePaths(imu=imu_path, pose=pose_path, pose_source=pose_source)


def _load_euroc_csv(path: Path) -> pd.DataFrame:
    """
    Load a EuRoC-style CSV (IMU or pose) into a DataFrame.

    EuRoC headers look like:  #timestamp [ns],w_RS_S_x [rad s^-1],...
    The leading '#' is part of the column name, NOT a comment line.
    We read normally and strip the '#' from the first column name.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # Strip leading '#' from first column name (e.g. "#timestamp [ns]" -> "timestamp [ns]")
    df.columns = df.columns.str.strip()
    if df.columns[0].startswith("#"):
        df = df.rename(columns={df.columns[0]: df.columns[0].lstrip("#").strip()})

    return df


def _extract_timestamp_seconds(df: pd.DataFrame) -> np.ndarray:
    """
    Extract timestamp column and convert from nanoseconds to seconds.

    EuRoC typically uses a column named 'timestamp' or '#timestamp'.
    """
    ts_col = None
    for candidate in ("timestamp [ns]", "timestamp", "#timestamp [ns]", "#timestamp"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        # Fall back to the first column if nothing matches, but fail loudly
        ts_col = df.columns[0]

    ts_ns = df[ts_col].to_numpy(dtype=np.int64)
    # nanoseconds -> seconds
    t = ts_ns.astype(np.float64) * 1e-9
    if not np.all(np.isfinite(t)):
        raise ValueError("Timestamps contain non-finite values.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("Timestamps are not strictly increasing.")
    return t


def _select_columns_by_prefix(
    df: pd.DataFrame, prefix: str, expected_dim: int
) -> Tuple[np.ndarray, List[str]]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(cols) < expected_dim:
        raise ValueError(
            f"Expected at least {expected_dim} columns with prefix '{prefix}', "
            f"found {len(cols)} in columns {list(df.columns)}"
        )
    # Preserve original ordering; use the first expected_dim columns
    cols = cols[:expected_dim]
    arr = df[cols].to_numpy(dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(
            f"Columns {cols} did not produce expected shape (*, {expected_dim}). "
            f"Got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values found in columns with prefix '{prefix}'.")
    return arr, cols


def load_euroc_sequence(
    seq_root: SeqRootLike,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Load IMU and pose / ground-truth tables for a EuRoC sequence.

    Returns:
        imu_df: DataFrame with at least timestamp + gyro + accel columns.
        pose_df: DataFrame with at least timestamp + position + quaternion columns.
        meta: small dict with pose_source and sequence root.
    """
    paths = find_euroc_files(seq_root)

    imu_df = _load_euroc_csv(paths.imu)
    pose_df = _load_euroc_csv(paths.pose)

    # Attach timestamp in seconds as a standard column for later functions
    imu_df = imu_df.copy()
    pose_df = pose_df.copy()
    imu_df["timestamp_sec"] = _extract_timestamp_seconds(imu_df)
    pose_df["timestamp_sec"] = _extract_timestamp_seconds(pose_df)

    meta = {
        "pose_source": paths.pose_source,
        "seq_root": str(_ensure_path(seq_root)),
    }
    return imu_df, pose_df, meta


def _interpolate_1d(
    t_src: np.ndarray, values: np.ndarray, t_target: np.ndarray
) -> np.ndarray:
    """
    Interpolate 1D or multi-channel data along time using linear interpolation.

    Args:
        t_src: (N,) strictly increasing.
        values: (N, D) array.
        t_target: (M,) target times.
    """
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != t_src.shape[0]:
        raise ValueError("t_src and values must have the same length.")

    out = np.empty((t_target.shape[0], values.shape[1]), dtype=np.float64)
    for d in range(values.shape[1]):
        out[:, d] = np.interp(t_target, t_src, values[:, d])
    return out


def _interpolate_quaternions(
    t_src: np.ndarray, quat: np.ndarray, t_target: np.ndarray
) -> np.ndarray:
    """
    Interpolate unit quaternions over time using SLERP.

    Args:
        t_src: (N,) strictly increasing.
        quat: (N, 4) array of quaternions. Assumes scalar-last (x, y, z, w) or
              scalar-first (w, x, y, z); we detect and normalise them.
        t_target: (M,) target times.
    """
    if quat.shape[0] != t_src.shape[0] or quat.shape[1] != 4:
        raise ValueError(f"Expected quat shape (N, 4), got {quat.shape}")

    # Normalise input quaternions
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat_unit = quat / norm

    # Determine component order by column names if possible is better, but here
    # we assume scalar-last (x, y, z, w) which is typical in EuRoC CSVs.
    # If needed, this can be made more flexible by passing in column order.
    rotations = Rotation.from_quat(quat_unit)  # expects (x, y, z, w)
    slerp = Slerp(t_src, rotations)
    interp_rot = slerp(t_target)
    quat_interp = interp_rot.as_quat()
    # Ensure unit length
    norm_interp = np.linalg.norm(quat_interp, axis=1, keepdims=True)
    quat_interp /= norm_interp
    return quat_interp


def resample_streams(
    imu_df: pd.DataFrame, pose_df: pd.DataFrame, target_rate: float = 200.0
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Resample IMU and pose streams to a common uniform time grid.

    - IMU gyro and accel: linear interpolation to target_rate (e.g., 200 Hz).
    - Position: linear interpolation.
    - Orientation: SLERP quaternion interpolation.
    """
    t_imu = imu_df["timestamp_sec"].to_numpy()
    t_pose = pose_df["timestamp_sec"].to_numpy()

    if not (np.all(np.diff(t_imu) > 0) and np.all(np.diff(t_pose) > 0)):
        raise ValueError("Input timestamps must be strictly increasing.")

    # Determine overlap interval
    t_start = max(t_imu[0], t_pose[0])
    t_end = min(t_imu[-1], t_pose[-1])
    if t_end <= t_start:
        raise ValueError(
            f"No temporal overlap between IMU [{t_imu[0]}, {t_imu[-1]}] "
            f"and pose [{t_pose[0]}, {t_pose[-1]}]."
        )

    dt = 1.0 / float(target_rate)
    t_grid = np.arange(t_start, t_end, dt, dtype=np.float64)
    if t_grid.size < 2:
        raise ValueError("Resampled time grid is too small.")

    # IMU: gyro and accel (body frame)
    # EuRoC column names use full prefixes: w_RS_S_ (gyro) and a_RS_S_ (accel)
    gyro, gyro_cols = _select_columns_by_prefix(imu_df, "w_RS_S_", expected_dim=3)
    accel, accel_cols = _select_columns_by_prefix(imu_df, "a_RS_S_", expected_dim=3)
    gyro_grid = _interpolate_1d(t_imu, gyro, t_grid)
    accel_grid = _interpolate_1d(t_imu, accel, t_grid)

    # Pose: position and quaternion
    pos, pos_cols = _select_columns_by_prefix(pose_df, "p_", expected_dim=3)
    quat, quat_cols = _select_columns_by_prefix(pose_df, "q_", expected_dim=4)
    pos_grid = _interpolate_1d(t_pose, pos, t_grid)
    quat_grid = _interpolate_quaternions(t_pose, quat, t_grid)

    # Sanity checks
    for name, arr in [
        ("t_grid", t_grid),
        ("pos", pos_grid),
        ("quat", quat_grid),
        ("gyro", gyro_grid),
        ("accel", accel_grid),
    ]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values found in resampled {name}.")

    data = {
        "t": t_grid,
        "pos": pos_grid,
        "quat": quat_grid,
        "gyro": gyro_grid,
        "accel": accel_grid,
    }
    meta = {
        "dt": dt,
        "rate_hz": float(target_rate),
        "gyro_cols": gyro_cols,
        "accel_cols": accel_cols,
        "pos_cols": pos_cols,
        "quat_cols": quat_cols,
    }
    return data, meta


def build_state_vectors(
    resampled: Dict[str, np.ndarray],
    rate_hz: float,
) -> Tuple[np.ndarray, Dict[str, Union[float, List[str]]]]:
    """
    Build state vectors s_t = [p, v, q, ω, a] from resampled streams.

    Args:
        resampled: dict with keys 't', 'pos', 'quat', 'gyro', 'accel'.
        rate_hz: sampling rate used for resampling (e.g. 200.0).
    """
    t = resampled["t"]
    pos = resampled["pos"]
    quat = resampled["quat"]
    gyro = resampled["gyro"]
    accel = resampled["accel"]

    dt = 1.0 / float(rate_hz)

    if pos.shape[0] != t.shape[0]:
        raise ValueError("pos and t must have the same length.")

    # Velocity from finite differences of position (world frame)
    # Use np.gradient for simplicity; it handles edges with one-sided differences.
    vel = np.gradient(pos, dt, axis=0)

    # Stack into [p (3), v (3), q (4), ω (3), a (3)] -> 16 features
    state = np.concatenate([pos, vel, quat, gyro, accel], axis=1)
    if state.shape[1] != 16:
        raise AssertionError(f"Expected state dim 16, got {state.shape[1]}")

    if not np.all(np.isfinite(state)):
        raise ValueError("Non-finite values found in state vectors.")

    feature_names: List[str] = [
        "p_x",
        "p_y",
        "p_z",
        "v_x",
        "v_y",
        "v_z",
        "q_x",
        "q_y",
        "q_z",
        "q_w",
        "w_x",
        "w_y",
        "w_z",
        "a_x",
        "a_y",
        "a_z",
    ]

    meta = {
        "dt": dt,
        "rate_hz": float(rate_hz),
        "feature_names": feature_names,
        "frame_imu": "body",
        "accel_includes_gravity": True,
    }
    return state, meta


def load_resampled_state_sequence(
    seq_root: SeqRootLike, target_rate: float = 200.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[str, float, List[str]]]]:
    """
    Convenience function: load, resample, and build state vectors for one EuRoC sequence.

    Args:
        seq_root: path to a EuRoC sequence folder (e.g. V1_01_easy).
        target_rate: sampling rate in Hz (default 200.0).

    Returns:
        state: (T, 16) array of state vectors.
        t: (T,) timestamps in seconds.
        meta: metadata dict with sampling rate, feature names, pose source, etc.
    """
    imu_df, pose_df, load_meta = load_euroc_sequence(seq_root)
    resampled, resample_meta = resample_streams(imu_df, pose_df, target_rate=target_rate)
    state, state_meta = build_state_vectors(resampled, rate_hz=target_rate)

    meta: Dict[str, Union[str, float, List[str]]] = {}
    meta.update(
        {
            "seq_root": load_meta.get("seq_root", ""),
            "pose_source": load_meta.get("pose_source", ""),
        }
    )
    meta.update(
        {
            "dt": resample_meta["dt"],
            "rate_hz": resample_meta["rate_hz"],
            "feature_names": state_meta["feature_names"],
            "frame_imu": state_meta["frame_imu"],
            "accel_includes_gravity": state_meta["accel_includes_gravity"],
        }
    )
    return state, resampled["t"], meta