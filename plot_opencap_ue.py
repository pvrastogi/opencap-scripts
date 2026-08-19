"""
OpenCap Upper Extremity Task Plotter
=====================================

Parses OpenCap .mot files (OpenSim Storage format) and plots, for each
selected joint angle: raw angle, raw first derivative (velocity), and
raw second derivative (acceleration), as a 3-panel row per angle.

USAGE (see bottom of file / README instructions):
    python3 plot_opencap_ue.py

Expects .mot files in the same folder as this script, named:
    reach.mot, dexterity1.mot, dexterityright.mot, arms.mot

Output:
    PNG files saved to ./output_plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. CONFIGURATION — edit this section to change tasks, files, or angles
# ---------------------------------------------------------------------------

# Folder where your .mot files live (default: same folder as this script)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder where plots will be saved
ROOT_DIR = os.path.abspath(os.path.join(DATA_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "Outputs", "AnglePlots", "ue")

# Task definitions:
#   file        -> .mot filename
#   angles      -> list of base angle names to plot (without _r/_l suffix
#                  for bilateral joints; exact name for unilateral use)
#   bilateral   -> True if angle has _r and _l versions plotted on same axes
TASKS = {
    "reach": {
        "file": "reach.mot",
        "angles": ["arm_flex", "arm_add", "arm_rot", "elbow_flex"],
        "bilateral": True,
    },
    "dexterity_left": {
        "file": "dexterity1.mot",
        "angles": ["arm_flex_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l"],
        "bilateral": False,
    },
    "dexterity_right": {
        "file": "dexterityright.mot",
        "angles": ["arm_flex_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r"],
        "bilateral": False,
    },
    "arms": {
        "file": "arms.mot",
        "angles": ["arm_flex", "arm_add", "arm_rot", "elbow_flex"],
        "bilateral": True,
    },
}

# Human-readable labels for titles/axis labels
ANGLE_LABELS = {
    "arm_flex": "Shoulder Flexion",
    "arm_add": "Shoulder Adduction/Abduction",
    "arm_rot": "Shoulder Rotation",
    "elbow_flex": "Elbow Flexion",
    "pro_sup": "Forearm Pronation/Supination",
}


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
# 4. PLOTTING
# ---------------------------------------------------------------------------

def plot_task(task_name, task_config):
    filepath = os.path.join(DATA_DIR, task_config["file"])
    if not os.path.exists(filepath):
        print(f"[SKIP] {task_name}: file not found at {filepath}")
        return

    df = load_mot_file(filepath)
    time = df["time"].values

    angles = task_config["angles"]
    bilateral = task_config["bilateral"]

    n_rows = len(angles)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.2 * n_rows), squeeze=False)

    for row_idx, angle_base in enumerate(angles):
        if bilateral:
            col_r = f"{angle_base}_r"
            col_l = f"{angle_base}_l"
            label = ANGLE_LABELS.get(angle_base, angle_base)
            series_list = []
            if col_r in df.columns:
                series_list.append(("Right", df[col_r].values, "tab:blue"))
            if col_l in df.columns:
                series_list.append(("Left", df[col_l].values, "tab:red"))
        else:
            # angle_base already includes _r or _l suffix
            base_clean = angle_base.rsplit("_", 1)[0]
            label = ANGLE_LABELS.get(base_clean, angle_base)
            series_list = []
            if angle_base in df.columns:
                side = "Right" if angle_base.endswith("_r") else "Left"
                series_list.append((side, df[angle_base].values, "tab:blue"))

        for side_name, values, color in series_list:
            velocity, acceleration = compute_derivatives(time, values)

            axes[row_idx, 0].plot(time, values, color=color, label=side_name)
            axes[row_idx, 1].plot(time, velocity, color=color, label=side_name)
            axes[row_idx, 2].plot(time, acceleration, color=color, label=side_name)

        axes[row_idx, 0].set_ylabel(f"{label}\n(deg)")
        axes[row_idx, 1].set_ylabel("Velocity\n(deg/s)")
        axes[row_idx, 2].set_ylabel("Acceleration\n(deg/s\u00b2)")

        for col in range(3):
            axes[row_idx, col].axhline(0, color="gray", linewidth=0.5, linestyle="--")
            axes[row_idx, col].grid(alpha=0.3)
            if len(series_list) > 1:
                axes[row_idx, col].legend(fontsize=8, loc="upper right")

        if row_idx == 0:
            axes[row_idx, 0].set_title("Angle (raw)")
            axes[row_idx, 1].set_title("Velocity (raw derivative)")
            axes[row_idx, 2].set_title("Acceleration (raw 2nd derivative)")

        if row_idx == n_rows - 1:
            for col in range(3):
                axes[row_idx, col].set_xlabel("Time (s)")

    fig.suptitle(f"OpenCap Task: {task_name}  ({task_config['file']})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{task_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] {task_name}: saved plot to {out_path}")


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main():
    for task_name, task_config in TASKS.items():
        plot_task(task_name, task_config)


if __name__ == "__main__":
    main()