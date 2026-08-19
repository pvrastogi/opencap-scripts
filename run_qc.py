"""
OpenCap Patient QC Runner
=========================

Single entry point for a patient's OpenCapData_<id> export. Run this one
script and it generates the full QC bundle into ./Outputs/:

  1. Marker frame-to-frame jump-size check
                                 -> Outputs/MarkerJumpCheck/<trial>.png
                                    Outputs/MarkerJumpCheck/marker_jump_summary.csv
  2. Anatomical angle plots     -> Outputs/AnglePlots/{balance,gait,ue}/*.png
                                    (delegates to plot_opencap_balance.py,
                                    plot_opencap_gait.py, plot_opencap_ue.py
                                    in OpenSimData/Kinematics/)
  3. Coordinate waveform QC     -> Outputs/CoordsQC/trials/*.png
     (delegates to opencap_qc_standalone.py, run for this session only)
                                    Outputs/CoordsQC/qc_features.csv
                                    Outputs/CoordsQC/qc_trials.csv
                                    Outputs/CoordsQC/coords_<trial>.png
                                    Outputs/CoordsQC/data_dictionary.md

This script uses only paths relative to its own location, so it can be
dropped unchanged into any OpenCapData_<id> folder with the same layout
(MarkerData/, OpenSimData/, Videos/) and just re-run.

USAGE:
    python3 run_qc.py

Requires: numpy, pandas, matplotlib, scipy
"""

import os
import sys
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. PATHS
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MARKER_DIR = os.path.join(ROOT_DIR, "MarkerData")
KINEMATICS_DIR = os.path.join(ROOT_DIR, "OpenSimData", "Kinematics")
QC_STANDALONE_SCRIPT = os.path.join(ROOT_DIR, "opencap_qc_standalone.py")

OUTPUT_DIR = os.path.join(ROOT_DIR, "Outputs")
MARKER_JUMP_DIR = os.path.join(OUTPUT_DIR, "MarkerJumpCheck")
ANGLE_PLOTS_DIR = os.path.join(OUTPUT_DIR, "AnglePlots")
COORDS_QC_DIR = os.path.join(OUTPUT_DIR, "CoordsQC")


# ===========================================================================
# 1. MARKER FRAME-TO-FRAME JUMP-SIZE CHECK
# ===========================================================================
#
# For each task-relevant marker: delta_x = x[n+1]-x[n] (and delta_y, delta_z),
# jump_size = sqrt(delta_x^2 + delta_y^2 + delta_z^2) -- the marker's raw
# frame-to-frame displacement. A real triangulation/tracking glitch shows up
# as a sharp spike (the marker teleporting, usually snapping back next
# frame); OpenCap markers riding on a rigid bone shouldn't jump more than a
# few cm between frames at 60 Hz.
#
# NOTE: this is a per-marker displacement check, not a between-marker rigid
# segment length check (that would be sqrt of squared differences between
# two DIFFERENT markers' positions, which is a different computation and is
# not what's plotted here).

JUMP_THRESHOLD_M = 0.05  # 5 cm/frame reference line

# Task-relevant markers per trial (matched by .trc filename), grouped by
# movement type rather than using the same markers everywhere: lower-
# extremity tasks use ankle/knee/pelvis, upper-extremity tasks use
# shoulder/elbow/wrist, static/postural tasks use pelvis only.
TASK_MARKERS = {
    "arms.trc":            ["r_shoulder_study", "r_lelbow_study", "r_lwrist_study",
                             "L_shoulder_study", "L_lelbow_study", "L_lwrist_study"],
    "gait_1.trc":          ["r_ankle_study", "r_knee_study", "r.ASIS_study", "L_ankle_study", "L_knee_study", "L.ASIS_study"],
    "tandem.trc":          ["r_ankle_study", "r_knee_study", "r.ASIS_study", "L_ankle_study", "L_knee_study", "L.ASIS_study"],
    "gait-start.trc":      ["r_ankle_study", "r_knee_study", "r.ASIS_study", "L_ankle_study", "L_knee_study", "L.ASIS_study"],
    "left_leg_raise.trc":  ["r_ankle_study", "r_knee_study", "r.ASIS_study", "L_ankle_study", "L_knee_study", "L.ASIS_study"],
    "sts.trc":             ["r_ankle_study", "r_knee_study", "r.ASIS_study", "L_ankle_study", "L_knee_study", "L.ASIS_study"],
    "semi-tandem.trc":     ["r_ankle_study", "r_knee_study", "r.ASIS_study", "L_ankle_study", "L_knee_study", "L.ASIS_study"],
    "gait-tandem.trc":     ["r_ankle_study", "r_knee_study", "r.ASIS_study", "L_ankle_study", "L_knee_study", "L.ASIS_study"],
    "rtt.trc":             ["r_ankle_study", "r_knee_study", "r_toe_study", "L_ankle_study", "L_knee_study", "L_toe_study"],
    "dexterityright.trc":  ["r_lwrist_study", "r_lelbow_study",
                             "L_lwrist_study", "L_lelbow_study"],
    "grapevine.trc":       ["r_ankle_study", "r_knee_study", "r.ASIS_study", "L_ankle_study", "L_knee_study", "L.ASIS_study"],
    "gait-headturn.trc":   ["r_ankle_study", "r_knee_study", "r.ASIS_study", "L_ankle_study", "L_knee_study", "L.ASIS_study"],
    "march.trc":           ["r_ankle_study", "r_knee_study", "r.ASIS_study", "L_ankle_study", "L_knee_study", "L.ASIS_study"],
    "gait-pivot.trc":      ["r_ankle_study", "r_knee_study", "r.ASIS_study", "L_ankle_study", "L_knee_study", "L.ASIS_study"],
    "neutral.trc":         ["r.ASIS_study"],
    "dexterity1.trc":      ["r_lwrist_study", "r_lelbow_study",
                             "L_lwrist_study", "L_lelbow_study"],
    "stand.trc":           ["r.ASIS_study"],
    "reach.trc":           ["r_lwrist_study", "r_shoulder_study",
                             "L_lwrist_study", "L_shoulder_study"],
    "y-balance.trc":       ["r_ankle_study", "r_knee_study", "r.ASIS_study", "L_ankle_study", "L_knee_study", "L.ASIS_study"],
}


def _read_trc(filepath):
    """Returns (time, marker_names, data) where data maps marker -> Nx3 (x,y,z)."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_fields = lines[1].strip().split("\t")
    header_values = lines[2].strip().split("\t")
    header = dict(zip(header_fields, header_values))

    marker_names_raw = lines[3].strip().split("\t")[2:]
    marker_names = [m for m in marker_names_raw if m]

    data_lines = lines[6:]
    n_frames = len(data_lines)
    data = {name: np.full((n_frames, 3), np.nan) for name in marker_names}
    time = np.full(n_frames, np.nan)

    for i, line in enumerate(data_lines):
        if not line.strip():
            continue
        cells = line.strip().split("\t")
        if len(cells) < 2:
            continue
        time[i] = float(cells[1])
        for m_idx, name in enumerate(marker_names):
            col_start = 2 + m_idx * 3
            try:
                x, y, z = (float(cells[col_start]), float(cells[col_start + 1]),
                           float(cells[col_start + 2]))
                data[name][i] = [x, y, z]
            except (ValueError, IndexError):
                continue

    frame_rate = float(header.get("DataRate", 60.0))
    return time, frame_rate, marker_names, data


def plot_marker_jumps(trial_label, time, frame_rate, markers_data, marker_names, out_path):
    fig, axes = plt.subplots(len(marker_names), 1, figsize=(9, 2.8 * len(marker_names)), squeeze=False)

    summary_rows = []
    for row, marker in enumerate(marker_names):
        ax = axes[row][0]
        pos = markers_data[marker]
        diffs = np.diff(pos, axis=0)
        jump_size = np.sqrt(np.sum(diffs ** 2, axis=1))

        ax.plot(time[1:], jump_size, color="darkred", linewidth=1)
        ax.axhline(y=JUMP_THRESHOLD_M, color="gray", linestyle="--", linewidth=1,
                   label=f"{JUMP_THRESHOLD_M * 100:.0f}cm/frame reference line")
        ax.set_title(f"{marker} -- frame-to-frame jump size")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (m)")
        ax.legend(fontsize=7)

        big_jumps = jump_size[~np.isnan(jump_size)] > JUMP_THRESHOLD_M
        summary_rows.append({
            "trial": trial_label,
            "marker": marker,
            "frame_rate_hz": frame_rate,
            "max_jump_cm": round(float(np.nanmax(jump_size)) * 100, 2) if len(jump_size) else np.nan,
            "n_jumps_over_5cm": int(np.sum(big_jumps)),
        })

    fig.suptitle(f"{trial_label} — marker frame-to-frame jump size", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return summary_rows


def run_marker_jump_check():
    print("=" * 70)
    print("1. MARKER FRAME-TO-FRAME JUMP-SIZE CHECK")
    print("=" * 70)

    all_summary_rows = []
    for trc_filename, marker_names in TASK_MARKERS.items():
        trc_path = os.path.join(MARKER_DIR, trc_filename)
        if not os.path.exists(trc_path):
            print(f"[SKIP] {trc_filename}: not found")
            continue

        trial_label = os.path.splitext(trc_filename)[0]
        time, frame_rate, available_markers, data = _read_trc(trc_path)

        markers_to_plot = [m for m in marker_names if m in available_markers]
        missing = [m for m in marker_names if m not in available_markers]
        if missing:
            print(f"[WARNING] {trial_label}: markers not found, skipping: {missing}")
        if not markers_to_plot:
            print(f"[SKIP] {trial_label}: none of the requested markers found")
            continue

        out_path = os.path.join(MARKER_JUMP_DIR, f"{trial_label}.png")
        summary_rows = plot_marker_jumps(trial_label, time, frame_rate, data, markers_to_plot, out_path)
        all_summary_rows.extend(summary_rows)

        flagged = [r for r in summary_rows if r["n_jumps_over_5cm"] > 0]
        if flagged:
            for r in flagged:
                print(f"[FLAG] {trial_label} / {r['marker']}: {r['n_jumps_over_5cm']} "
                      f"frame(s) > 5cm jump (max {r['max_jump_cm']} cm)")
        else:
            print(f"[OK] {trial_label}: no marker exceeds 5cm/frame")

    summary_df = pd.DataFrame(all_summary_rows)
    os.makedirs(MARKER_JUMP_DIR, exist_ok=True)
    csv_path = os.path.join(MARKER_JUMP_DIR, "marker_jump_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print()


# ===========================================================================
# 2. ANATOMICAL ANGLE PLOTS
# ===========================================================================
#
# Delegates to the three category-specific plotters, which already know
# which angles matter for balance/gait/upper-extremity tasks. They write
# into Outputs/AnglePlots/{balance,gait,ue}/ (configured in each script).

def run_angle_plots():
    print("=" * 70)
    print("2. ANATOMICAL ANGLE PLOTS")
    print("=" * 70)

    if KINEMATICS_DIR not in sys.path:
        sys.path.insert(0, KINEMATICS_DIR)

    for module_name in ("plot_opencap_balance", "plot_opencap_gait", "plot_opencap_ue"):
        script_path = os.path.join(KINEMATICS_DIR, f"{module_name}.py")
        if not os.path.exists(script_path):
            print(f"[SKIP] {module_name}.py not found in {KINEMATICS_DIR}")
            continue
        module = __import__(module_name)
        module.main()
    print()


# ===========================================================================
# 3. COORDINATE WAVEFORM QC (raw coordinates, low-pass residual, f99 Hz, ...)
# ===========================================================================
#
# Delegates to opencap_qc_standalone.py, a standalone multi-session QC tool
# (also usable on its own for batch-auditing several patients at once). Here
# it's invoked as a subprocess scoped to just this session's folder, so its
# proven feature computations/plots stay exactly as authored.

def run_coords_qc():
    print("=" * 70)
    print("3. COORDINATE WAVEFORM QC (opencap_qc_standalone.py)")
    print("=" * 70)

    if not os.path.exists(QC_STANDALONE_SCRIPT):
        print(f"[SKIP] {QC_STANDALONE_SCRIPT} not found")
        return

    os.makedirs(COORDS_QC_DIR, exist_ok=True)
    result = subprocess.run(
        [sys.executable, QC_STANDALONE_SCRIPT, "--data", ROOT_DIR, "--out", COORDS_QC_DIR],
        cwd=ROOT_DIR,
    )
    if result.returncode != 0:
        print(f"[ERROR] opencap_qc_standalone.py exited with code {result.returncode}")
    print()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_marker_jump_check()
    run_angle_plots()
    run_coords_qc()

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"All outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
