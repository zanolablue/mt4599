from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


SeqRootLike = Union[str, Path]

# Small epsilon to contract the interpolation grid just inside source boundaries,
# preventing floating-point overshoot from crashing SLERP (fixes MH_02/05, V2_03).
_GRID_EPS = 1e-9

# Gravitational acceleration magnitude (m/s²)
_G = 9.81


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
    - IMU:   mav0/imu0/data.csv (required)
    - Pose:  mav0/state_groundtruth_estimate0/data.csv (primary)
             vicon0/data.csv (fallback)
    """
    root = _ensure_path(seq_root)

    imu_candidates = [
        root / "mav0" / "imu0" / "data.csv",
        root / "imu0" / "data.csv",
    ]
    imu_path = next((p for p in imu_candidates if p.exists()), None)
    if imu_path is None:
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

    pose_source = "unknown"
    pose_path = root / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    if pose_path.exists():
        pose_source = "state_groundtruth"
    else:
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
    Load a EuRoC-style CSV into a DataFrame.

    EuRoC headers look like:  #timestamp [ns],w_RS_S_x [rad s^-1],...
    The leading '#' is part of the column name, NOT a comment marker.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if df.columns[0].startswith("#"):
        df = df.rename(columns={df.columns[0]: df.columns[0].lstrip("#").strip()})
    return df


def _extract_timestamp_seconds(df: pd.DataFrame) -> np.ndarray:
    """Extract timestamp column and convert nanoseconds → seconds."""
    ts_col = None
    for candidate in ("timestamp [ns]", "timestamp", "#timestamp [ns]", "#timestamp"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        ts_col = df.columns[0]

    ts_ns = df[ts_col].to_numpy(dtype=np.int64)
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
    paths = find_euroc_files(seq_root)
    imu_df  = _load_euroc_csv(paths.imu).copy()
    pose_df = _load_euroc_csv(paths.pose).copy()
    imu_df["timestamp_sec"]  = _extract_timestamp_seconds(imu_df)
    pose_df["timestamp_sec"] = _extract_timestamp_seconds(pose_df)
    meta = {
        "pose_source": paths.pose_source,
        "seq_root": str(_ensure_path(seq_root)),
    }
    return imu_df, pose_df, meta


def _interpolate_1d(
    t_src: np.ndarray, values: np.ndarray, t_target: np.ndarray
) -> np.ndarray:
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
    SLERP interpolation of unit quaternions (scalar-last: x, y, z, w).
    t_target must lie strictly within [t_src[0], t_src[-1]] — guaranteed by
    the _GRID_EPS contraction in resample_streams.
    """
    if quat.shape[0] != t_src.shape[0] or quat.shape[1] != 4:
        raise ValueError(f"Expected quat shape (N, 4), got {quat.shape}")
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat_unit = quat / np.where(norm > 0, norm, 1.0)
    rotations = Rotation.from_quat(quat_unit)
    slerp = Slerp(t_src, rotations)
    quat_interp = slerp(t_target).as_quat()
    norm_interp = np.linalg.norm(quat_interp, axis=1, keepdims=True)
    quat_interp /= np.where(norm_interp > 0, norm_interp, 1.0)
    return quat_interp


def _smooth_velocity(pos: np.ndarray, dt: float, window_sec: float = 0.05) -> np.ndarray:
    """
    Estimate velocity via Savitzky-Golay derivative filter (2nd-order, ~50 ms window).
    Cleaner than raw np.gradient at 200 Hz.
    """
    wl = max(int(window_sec / dt) | 1, 5)
    if wl % 2 == 0:
        wl += 1
    return savgol_filter(pos, window_length=wl, polyorder=2, deriv=1, delta=dt, axis=0)


def _compensate_gravity(
    accel_body: np.ndarray, quat: np.ndarray
) -> np.ndarray:
    """
    Rotate body-frame IMU acceleration into the world frame and subtract gravity.

    The EuRoC IMU measures specific force in the body frame:
        a_body = R^T (a_world - g_world)
    so the gravity-compensated world-frame acceleration is:
        a_world_comp = R * a_body - g_world

    Args:
        accel_body: (T, 3)  raw IMU acceleration in body frame (gravity included).
        quat:       (T, 4)  orientation quaternion, scalar-last (x, y, z, w),
                            rotating body → world.
    Returns:
        accel_comp: (T, 3)  linear acceleration in world frame, gravity removed.
    """
    g_world = np.array([0.0, 0.0, _G])
    rotations   = Rotation.from_quat(quat)          # body → world
    accel_world = rotations.apply(accel_body)        # rotate to world frame
    accel_comp  = accel_world - g_world              # subtract gravity
    return accel_comp


def resample_streams(
    imu_df: pd.DataFrame, pose_df: pd.DataFrame, target_rate: float = 200.0
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Resample IMU and pose streams to a common uniform time grid.

    Grid is contracted by _GRID_EPS on both ends so SLERP never receives a
    query point outside the source range (fixes MH_02, MH_05, V2_03 crashes).
    """
    t_imu  = imu_df["timestamp_sec"].to_numpy()
    t_pose = pose_df["timestamp_sec"].to_numpy()

    if not (np.all(np.diff(t_imu) > 0) and np.all(np.diff(t_pose) > 0)):
        raise ValueError("Input timestamps must be strictly increasing.")

    t_start = max(t_imu[0],  t_pose[0])  + _GRID_EPS
    t_end   = min(t_imu[-1], t_pose[-1]) - _GRID_EPS

    if t_end <= t_start:
        raise ValueError(
            f"No temporal overlap between IMU [{t_imu[0]:.6f}, {t_imu[-1]:.6f}] "
            f"and pose [{t_pose[0]:.6f}, {t_pose[-1]:.6f}]."
        )

    dt     = 1.0 / float(target_rate)
    t_grid = np.arange(t_start, t_end, dt, dtype=np.float64)
    if t_grid.size < 2:
        raise ValueError("Resampled time grid is too small.")

    gyro,  gyro_cols  = _select_columns_by_prefix(imu_df, "w_RS_S_", expected_dim=3)
    accel, accel_cols = _select_columns_by_prefix(imu_df, "a_RS_S_", expected_dim=3)
    gyro_grid  = _interpolate_1d(t_imu, gyro,  t_grid)
    accel_grid = _interpolate_1d(t_imu, accel, t_grid)

    pos,  pos_cols  = _select_columns_by_prefix(pose_df, "p_", expected_dim=3)
    quat, quat_cols = _select_columns_by_prefix(pose_df, "q_", expected_dim=4)
    pos_grid  = _interpolate_1d(t_pose, pos,  t_grid)
    quat_grid = _interpolate_quaternions(t_pose, quat, t_grid)

    for name, arr in [("t_grid", t_grid), ("pos", pos_grid), ("quat", quat_grid),
                      ("gyro", gyro_grid), ("accel", accel_grid)]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values found in resampled {name}.")

    data = {"t": t_grid, "pos": pos_grid, "quat": quat_grid,
            "gyro": gyro_grid, "accel": accel_grid}
    meta = {"dt": dt, "rate_hz": float(target_rate),
            "gyro_cols": gyro_cols, "accel_cols": accel_cols,
            "pos_cols": pos_cols, "quat_cols": quat_cols}
    return data, meta


def build_state_vectors(
    resampled: Dict[str, np.ndarray],
    rate_hz: float,
) -> Tuple[np.ndarray, Dict[str, Union[float, List[str]]]]:
    """
    Build state vectors s_t = [Δp, v, q, ω, a_comp] from resampled streams.

    Changes vs. baseline run:
    ─────────────────────────
    1. Position → position increments Δp_t = p_t − p_{t−1}
       Removes room-location bias from clustering; keeps kinematically relevant
       displacement signal. Δp at t=0 is set to zero.

    2. Gravity compensation on acceleration
       Raw IMU accel is dominated by ~9.81 m/s² gravity in all clusters.
       We rotate to world frame and subtract g, giving a_comp ∈ R³ that is
       near-zero during hover and large during aggressive manoeuvres.

    3. Savitzky-Golay velocity derivative (as before)

    State layout (16 features):
        dp_x, dp_y, dp_z     — position increment (world frame, m)
        v_x,  v_y,  v_z      — velocity (world frame, SG-smoothed, m/s)
        q_x,  q_y,  q_z, q_w — orientation quaternion (scalar-last)
        w_x,  w_y,  w_z      — angular velocity (body frame, rad/s)
        ac_x, ac_y, ac_z     — gravity-compensated accel (world frame, m/s²)
    """
    t     = resampled["t"]
    pos   = resampled["pos"]
    quat  = resampled["quat"]
    gyro  = resampled["gyro"]
    accel = resampled["accel"]

    dt = 1.0 / float(rate_hz)

    if pos.shape[0] != t.shape[0]:
        raise ValueError("pos and t must have the same length.")

    # 1. Position increments
    dp      = np.diff(pos, axis=0, prepend=pos[:1])   # Δp_t = p_t - p_{t-1}
    dp[0]   = 0.0                                       # boundary: zero at t=0

    # 2. Velocity (SG derivative)
    vel = _smooth_velocity(pos, dt)

    # 3. Gravity-compensated acceleration (world frame)
    accel_comp = _compensate_gravity(accel, quat)

    state = np.concatenate([dp, vel, quat, gyro, accel_comp], axis=1)

    if state.shape[1] != 16:
        raise AssertionError(f"Expected state dim 16, got {state.shape[1]}")
    if not np.all(np.isfinite(state)):
        raise ValueError("Non-finite values found in state vectors.")

    feature_names: List[str] = [
        "dp_x", "dp_y", "dp_z",
        "v_x",  "v_y",  "v_z",
        "q_x",  "q_y",  "q_z",  "q_w",
        "w_x",  "w_y",  "w_z",
        "ac_x", "ac_y", "ac_z",
    ]

    meta = {
        "dt":                    dt,
        "rate_hz":               float(rate_hz),
        "feature_names":         feature_names,
        "frame_imu":             "body_for_gyro_world_for_accel",
        "accel_includes_gravity": False,
        "velocity_method":       "savgol_deriv_50ms",
        "position_representation": "increments",
    }
    return state, meta


def load_resampled_state_sequence(
    seq_root: SeqRootLike, target_rate: float = 200.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[str, float, List[str]]]]:
    """
    Convenience wrapper: load → resample → build state vectors for one EuRoC sequence.
    """
    imu_df, pose_df, load_meta   = load_euroc_sequence(seq_root)
    resampled, resample_meta     = resample_streams(imu_df, pose_df, target_rate=target_rate)
    state, state_meta            = build_state_vectors(resampled, rate_hz=target_rate)

    meta: Dict[str, Union[str, float, List[str]]] = {}
    meta.update({
        "seq_root":    load_meta["seq_root"],
        "pose_source": load_meta["pose_source"],
    })
    meta.update({
        "dt":                     resample_meta["dt"],
        "rate_hz":                resample_meta["rate_hz"],
        "feature_names":          state_meta["feature_names"],
        "frame_imu":              state_meta["frame_imu"],
        "accel_includes_gravity": state_meta["accel_includes_gravity"],
        "velocity_method":        state_meta["velocity_method"],
        "position_representation": state_meta["position_representation"],
    })
    return state, resampled["t"], meta
