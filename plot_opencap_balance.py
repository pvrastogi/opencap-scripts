"""
OpenCap Balance Task Plotter
============================

Parses OpenCap .mot files (OpenSim Storage format) and plots, for each
selected joint angle: raw angle, raw first derivative (velocity), and
raw second derivative (acceleration), as a 3-panel row per angle.

Mirrors the structure of plot_opencap_ue.py, but for the balance tasks:
stand, semi-tandem, tandem, left_leg_raise, y-balance, sts, rtt, march,
grapevine.

Notes on angle choices:
- pelvis_tilt / pelvis_list / pelvis_rotation are flagged by the OpenCap
  team as potentially unreliable, so pelvis posture is NOT plotted here.
  Instead, "pelvis_position" plots pelvis_tx/ty/tz (the pelvis's x/y/z
  coordinates in meters) together on one row, used here as the postural
  sway / center-of-mass-excursion proxy.
- left_leg_raise: subject raises the right leg, then the left leg, one at
  a time. Plotting hip_flexion/knee_angle bilaterally (R and L overlaid)
  shows this as two sequential peaks (right first, then left).
- rtt ("raise to toes"): subject stands on both feet, rises onto their
  toes for 5s, then returns to flat — a bilateral, synchronized movement.
- march and grapevine are also plotted in plot_opencap_gait.py with a
  different angle set (they're relevant to both categories).

USAGE:
    python3 plot_opencap_balance.py

Output:
    PNG files saved to ./output_plots/balance/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. CONFIGURATION — edit this section to change tasks, files, or angles
# ---------------------------------------------------------------------------

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(DATA_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "Outputs", "AnglePlots", "balance")

# Task definitions:
#   file        -> .mot filename
#   angles      -> list of base angle names to plot. Each is one of:
#                    - a bilateral joint angle (e.g. "hip_flexion") -> plots
#                      "<base>_r" and "<base>_l" on the same axes
#                    - a midline angle with no _r/_l (e.g. "lumbar_extension")
#                    - the special name "pelvis_position", which plots
#                      pelvis_tx, pelvis_ty, pelvis_tz together (units: m)
TASKS = {
    "stand": {
        "file": "stand.mot",
        "angles": ["pelvis_position", "ankle_angle", "hip_flexion"],
    },
    "semi_tandem": {
        "file": "semi-tandem.mot",
        "angles": ["pelvis_position", "hip_adduction", "ankle_angle", "subtalar_angle"],
    },
    "tandem": {
        "file": "tandem.mot",
        "angles": ["pelvis_position", "hip_adduction", "ankle_angle", "subtalar_angle"],
    },
    "left_leg_raise": {
        "file": "left_leg_raise.mot",
        "angles": ["hip_flexion", "knee_angle", "hip_adduction", "pelvis_position"],
    },
    "y_balance": {
        "file": "y-balance.mot",
        "angles": ["hip_flexion", "knee_angle", "ankle_angle", "pelvis_position"],
    },
    "sts": {
        "file": "sts.mot",
        "angles": ["hip_flexion", "knee_angle", "lumbar_extension", "pelvis_position"],
    },
    "rtt": {
        "file": "rtt.mot",
        "angles": ["ankle_angle", "subtalar_angle", "pelvis_position"],
    },
    "march": {
        "file": "march.mot",
        "angles": ["hip_flexion", "hip_adduction", "pelvis_position"],
    },
    "grapevine": {
        "file": "grapevine.mot",
        "angles": ["hip_adduction", "ankle_angle", "pelvis_position"],
    },
}

# Human-readable labels for titles/axis labels
ANGLE_LABELS = {
    "hip_flexion": "Hip Flexion",
    "hip_adduction": "Hip Adduction/Abduction",
    "hip_rotation": "Hip Rotation",
    "knee_angle": "Knee Flexion",
    "ankle_angle": "Ankle Dorsi/Plantarflexion",
    "subtalar_angle": "Subtalar Inversion/Eversion",
    "lumbar_extension": "Lumbar Flexion/Extension",
    "lumbar_bending": "Lumbar Lateral Bending",
    "lumbar_rotation": "Lumbar Rotation",
    "pelvis_position": "Pelvis Position",
}

# Units per angle base: "deg" for joint angles, "m" for pelvis position
UNITS = {
    "pelvis_position": "m",
}
DEFAULT_UNIT = "deg"


# ---------------------------------------------------------------------------
# 2. FILE PARSING
# ---------------------------------------------------------------------------

def load_mot_file(filepath):
    """
    Parses an OpenCap/OpenSim .mot file.
    Header ends at a line containing 'endheader'; the next line is the
    tab-separated column names, followed by tab-separated numeric rows.
    Returns a pandas DataFrame.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_end_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end_idx = i
            break

    if header_end_idx is None:
        raise ValueError(f"Could not find 'endheader' in {filepath}")

    col_line = lines[header_end_idx + 1]
    columns = col_line.strip().split("\t")

    data_lines = lines[header_end_idx + 2:]
    data = [line.strip().split("\t") for line in data_lines if line.strip()]

    df = pd.DataFrame(data, columns=columns)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")

    return df


# ---------------------------------------------------------------------------
# 3. DERIVATIVES (raw, unfiltered)
# ---------------------------------------------------------------------------

def compute_derivatives(time, values):
    """
    Computes raw first derivative (velocity) and second derivative
    (acceleration) using numpy's central-difference gradient.
    No smoothing/filtering is applied.
    """
    velocity = np.gradient(values, time)
    acceleration = np.gradient(velocity, time)
    return velocity, acceleration


# ---------------------------------------------------------------------------
# 4. SERIES RESOLUTION — decide what columns/labels/colors go into a row
# ---------------------------------------------------------------------------

PELVIS_AXIS_SERIES = [
    ("pelvis_tx", "X (fore-aft)", "tab:blue"),
    ("pelvis_ty", "Y (vertical)", "tab:green"),
    ("pelvis_tz", "Z (lateral)", "tab:red"),
]


def resolve_series(df, angle_base):
    """
    Returns (label, unit, series_list) where series_list is a list of
    (name, values, color) tuples to overlay on the same axes.
    """
    if angle_base == "pelvis_position":
        series_list = [
            (name, df[col].values, color)
            for col, name, color in PELVIS_AXIS_SERIES
            if col in df.columns
        ]
        return ANGLE_LABELS.get(angle_base, angle_base), "m", series_list

    col_r = f"{angle_base}_r"
    col_l = f"{angle_base}_l"
    label = ANGLE_LABELS.get(angle_base, angle_base)

    if col_r in df.columns or col_l in df.columns:
        series_list = []
        if col_r in df.columns:
            series_list.append(("Right", df[col_r].values, "tab:blue"))
        if col_l in df.columns:
            series_list.append(("Left", df[col_l].values, "tab:red"))
        return label, DEFAULT_UNIT, series_list

    if angle_base in df.columns:
        return label, DEFAULT_UNIT, [("Trunk", df[angle_base].values, "tab:purple")]

    return label, DEFAULT_UNIT, []


# ---------------------------------------------------------------------------
# 5. PLOTTING
# ---------------------------------------------------------------------------

def plot_task(task_name, task_config):
    filepath = os.path.join(DATA_DIR, task_config["file"])
    if not os.path.exists(filepath):
        print(f"[SKIP] {task_name}: file not found at {filepath}")
        return

    df = load_mot_file(filepath)
    time = df["time"].values

    angles = task_config["angles"]

    n_rows = len(angles)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.2 * n_rows), squeeze=False)

    for row_idx, angle_base in enumerate(angles):
        label, unit, series_list = resolve_series(df, angle_base)
        vel_unit = f"{unit}/s"
        acc_unit = f"{unit}/s²"

        for series_name, values, color in series_list:
            velocity, acceleration = compute_derivatives(time, values)

            axes[row_idx, 0].plot(time, values, color=color, label=series_name)
            axes[row_idx, 1].plot(time, velocity, color=color, label=series_name)
            axes[row_idx, 2].plot(time, acceleration, color=color, label=series_name)

        axes[row_idx, 0].set_ylabel(f"{label}\n({unit})")
        axes[row_idx, 1].set_ylabel(f"Velocity\n({vel_unit})")
        axes[row_idx, 2].set_ylabel(f"Acceleration\n({acc_unit})")

        for col in range(3):
            axes[row_idx, col].axhline(0, color="gray", linewidth=0.5, linestyle="--")
            axes[row_idx, col].grid(alpha=0.3)
            if len(series_list) > 1:
                axes[row_idx, col].legend(fontsize=8, loc="upper right")

        if row_idx == 0:
            axes[row_idx, 0].set_title("Angle/Position (raw)")
            axes[row_idx, 1].set_title("Velocity (raw derivative)")
            axes[row_idx, 2].set_title("Acceleration (raw 2nd derivative)")

        if row_idx == n_rows - 1:
            for col in range(3):
                axes[row_idx, col].set_xlabel("Time (s)")

    fig.suptitle(f"OpenCap Balance Task: {task_name}  ({task_config['file']})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{task_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] {task_name}: saved plot to {out_path}")


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    for task_name, task_config in TASKS.items():
        plot_task(task_name, task_config)


if __name__ == "__main__":
    main()
