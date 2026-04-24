"""
mt4599: EuRoC MAV preprocessing utilities for thesis experiments.

This package currently focuses on a single-sequence preprocessing pipeline:

- Detect IMU and pose / ground-truth files under a EuRoC sequence root.
- Resample both streams to a uniform 200 Hz grid.
- Build state vectors s_t = [p_t, v_t, q_t, ω_t, a_t].
"""

