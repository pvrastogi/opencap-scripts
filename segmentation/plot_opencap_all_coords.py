import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no interactive window; write straight to file
import matplotlib.pyplot as plt


# Folder where the .mot files live (default: same folder as this script)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder where plots will be saved
ROOT_DIR = os.path.abspath(os.path.join(DATA_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "Outputs", "AnglePlots", "all_coords")

# True -> _r and _l on one axes (24 panels); False -> 39 panels
BILATERAL_OVERLAY = True

# ROM in the panel corner; catches locked and implausible coordinates
SHOW_ROM = True

# Panels per row.
N_COLS = 4

# base names to skip; "deleted" anywhere in a name is automatic
SKIP_TRIALS = set()

# OLD_GX used non-standard trial names; user-confirmed mapping
TRIAL_ALIASES = {
    "stand": "standing-ec",
    "reach": "frt",
    "sts": "5tsts",
    "left_leg_raise": "sls-ec",
    "gait_1": "gait",
    "gait-headturn": "gait-head-turn",
    "dexterity1": "dext-left",
    "dexterityright": "dext-right",
}


COORD_GROUPS = [
    ("Pelvis translation", "m",   False, ["pelvis_tx", "pelvis_ty", "pelvis_tz"]),
    ("Pelvis orientation", "deg", False, ["pelvis_tilt", "pelvis_list", "pelvis_rotation"]),
    ("Lower limb",         "deg", True,  ["hip_flexion", "hip_adduction", "hip_rotation",
                                          "knee_angle", "ankle_angle", "subtalar_angle",
                                          "mtp_angle"]),
    ("Lumbar",             "deg", False, ["lumbar_extension", "lumbar_bending",
                                          "lumbar_rotation"]),
    ("Shoulder translation", "m", True,  ["sh_tx", "sh_ty", "sh_tz"]),
    ("Upper limb",         "deg", True,  ["sh_plane_elev", "sh_elev", "sh_axial_rot",
                                          "elbow_flex", "pro_sup"]),
]

COORD_LABELS = {
    "pelvis_tx": "Pelvis X (fore-aft)",
    "pelvis_ty": "Pelvis Y (vertical)",
    "pelvis_tz": "Pelvis Z (lateral)",
    "pelvis_tilt": "Pelvis Tilt",
    "pelvis_list": "Pelvis List",
    "pelvis_rotation": "Pelvis Rotation",
    "hip_flexion": "Hip Flexion",
    "hip_adduction": "Hip Add/Abduction",
    "hip_rotation": "Hip Rotation",
    "knee_angle": "Knee Flexion",
    "ankle_angle": "Ankle Dorsi/Plantarflexion",
    "subtalar_angle": "Subtalar Inv/Eversion",
    "mtp_angle": "MTP (toe)",
    "lumbar_extension": "Lumbar Flexion/Extension",
    "lumbar_bending": "Lumbar Lateral Bending",
    "lumbar_rotation": "Lumbar Rotation",
    "sh_tx": "Shoulder Translation X",
    "sh_ty": "Shoulder Translation Y",
    "sh_tz": "Shoulder Translation Z",
    "sh_plane_elev": "Plane of Elevation",
    "sh_elev": "Shoulder Elevation",
    "sh_axial_rot": "Shoulder Axial Rotation",
    "elbow_flex": "Elbow Flexion",
    "pro_sup": "Forearm Pronation/Supination",
}

# standing reliability caveat; marked in the title, still plotted
CAUTION_COORDS = {
    "pelvis_tilt": "flagged unreliable",
    "pelvis_list": "flagged unreliable",
    "pelvis_rotation": "flagged unreliable",
    "pelvis_ty": "lab-frame offset",
    "lumbar_extension": "reliability pending",
    "lumbar_bending": "reliability pending",
    "lumbar_rotation": "reliability pending",
    "pro_sup": "not reliably estimated",
}

# locked in this model; annotated so a flat line isn't read as a fault
KNOWN_LOCKED = {"mtp_angle"}

# Column names that identify a file as the OLDER, non-shoulder model.
OLD_MODEL_MARKERS = ("arm_flex_r", "arm_add_r", "arm_rot_r")

COLOR_R = "tab:blue"
COLOR_L = "tab:red"
COLOR_MID = "tab:purple"


def load_mot_file(filepath):
    """Parses an OpenCap/OpenSim .mot file."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_end_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end_idx = i
            break

    if header_end_idx is None:
        raise ValueError(f"Could not find 'endheader' in {filepath}")

    columns = lines[header_end_idx + 1].strip().split("\t")
    data_lines = lines[header_end_idx + 2:]
    data = [line.strip().split("\t") for line in data_lines if line.strip()]

    df = pd.DataFrame(data, columns=columns)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")

    return df


def discover_trials(data_dir):
    """Finds the .mot file to use for each trial in this session."""
    candidates = {}
    for fname in os.listdir(data_dir):
        if not fname.endswith(".mot"):
            continue
        if "deleted" in fname.lower():
            print(f"[SKIP] {fname}: 'deleted' in filename")
            continue

        base = fname[:-4]
        if base.endswith("_shoulder"):
            trial, preferred = base[:-len("_shoulder")], True
        else:
            trial, preferred = base, False

        if trial in SKIP_TRIALS:
            continue

        # A preferred (_shoulder) file always wins over a plain one.
        if trial not in candidates or (preferred and not candidates[trial][1]):
            candidates[trial] = (fname, preferred)

    resolved = []
    for trial, (fname, _) in candidates.items():
        resolved.append((TRIAL_ALIASES.get(trial, trial), fname))
    return sorted(resolved)


def build_panels(df):
    """Expands COORD_GROUPS into a flat list of panel specs against the columns"""
    panels = []
    missing = []

    for _, unit, bilateral, bases in COORD_GROUPS:
        for base in bases:
            label = COORD_LABELS.get(base, base)
            caveat = CAUTION_COORDS.get(base)

            if bilateral and BILATERAL_OVERLAY:
                wanted = [("Right", f"{base}_r", COLOR_R),
                          ("Left", f"{base}_l", COLOR_L)]
                present = [(s, c, col) for s, c, col in wanted if c in df.columns]
                missing += [c for _, c, _ in wanted if c not in df.columns]
                if present:
                    panels.append((label, unit, present, caveat))

            elif bilateral:
                for side, suffix, color in (("Right", "_r", COLOR_R),
                                            ("Left", "_l", COLOR_L)):
                    col = f"{base}{suffix}"
                    if col in df.columns:
                        panels.append((f"{label} ({side})", unit,
                                       [(side, col, color)], caveat))
                    else:
                        missing.append(col)

            else:
                if base in df.columns:
                    panels.append((label, unit, [("", base, COLOR_MID)], caveat))
                else:
                    missing.append(base)

    return panels, missing


def plot_trial(task_name, filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[SKIP] {task_name}: file not found at {filepath}")
        return

    df = load_mot_file(filepath)

    if "time" not in df.columns:
        print(f"[SKIP] {task_name}: no 'time' column in {filename}")
        return

    if any(m in df.columns for m in OLD_MODEL_MARKERS):
        print(f"[SKIP] {task_name}: {filename} is the OLDER LaiUhlrich2022 layout "
              f"(found arm_flex/arm_add/arm_rot). Expected the shoulder model. "
              f"Is there a matching '_shoulder.mot' for this trial?")
        return

    time = df["time"].values
    panels, missing = build_panels(df)

    if not panels:
        print(f"[SKIP] {task_name}: no recognised coordinates in {filename}")
        return

    n_rows = math.ceil(len(panels) / N_COLS)
    fig, axes = plt.subplots(n_rows, N_COLS,
                             figsize=(4.6 * N_COLS, 2.5 * n_rows),
                             squeeze=False)

    locked_found = []

    for idx, (label, unit, series, caveat) in enumerate(panels):
        ax = axes[idx // N_COLS][idx % N_COLS]

        roms = []
        for series_label, col, color in series:
            values = df[col].values
            ax.plot(time, values, color=color, linewidth=0.9,
                    label=series_label or None)
            finite = values[np.isfinite(values)]
            if finite.size:
                roms.append((series_label or col, float(finite.max() - finite.min())))

        # A locked coordinate is flat by design, not by fault -- say so.
        base_guess = series[0][1].rsplit("_", 1)[0] if series[0][1].endswith(("_r", "_l")) else series[0][1]
        if base_guess in KNOWN_LOCKED and all(r < 1e-6 for _, r in roms):
            locked_found.append(label)
            title = f"{label}  [locked DOF]"
        elif caveat:
            title = f"{label}  [{caveat}]"
        else:
            title = label

        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel(unit, fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

        if len(series) > 1:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.6)

        if SHOW_ROM and roms:
            txt = "  ".join(f"{n}: {v:.3g}" if n else f"ROM {v:.3g}"
                            for n, v in roms)
            ax.text(0.02, 0.03, f"ROM {txt}" if roms[0][0] else txt,
                    transform=ax.transAxes, fontsize=6.5, color="dimgray",
                    va="bottom", ha="left")

        # x label only on the bottom-most populated panel of each column
        if idx >= len(panels) - N_COLS:
            ax.set_xlabel("Time (s)", fontsize=8)

    # blank out unused grid slots
    for idx in range(len(panels), n_rows * N_COLS):
        axes[idx // N_COLS][idx % N_COLS].axis("off")

    duration = time[-1] - time[0] if len(time) > 1 else 0.0
    fs = (len(time) - 1) / duration if duration > 0 else float("nan")
    fig.suptitle(
        f"{task_name}  --  all coordinates (position)\n"
        f"{filename}   |   {len(time)} frames   |   {duration:.2f} s   |   {fs:.1f} Hz",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97 if n_rows > 3 else 0.94])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{task_name}.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    note = ""
    if missing:
        note += f"  [missing: {', '.join(sorted(set(missing)))}]"
    if locked_found:
        note += f"  [locked: {', '.join(locked_found)}]"
    print(f"[OK] {task_name}: {len(panels)} panels -> {out_path}{note}")


def main():
    global DATA_DIR, OUTPUT_DIR
    if len(sys.argv) > 1:
        _here = os.path.dirname(os.path.abspath(__file__))
        for _p in (os.path.join(_here, "segmentation"), _here):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        import seg_common as sc
        sd = sc.session_dir(sys.argv[1])
        DATA_DIR = os.path.join(sd, "OpenSimData", "Kinematics")
        OUTPUT_DIR = os.path.join(sd, "Outputs", "AnglePlots", "all_coords")

    print("=" * 70)
    print("ALL-COORDINATE POSITION PLOTS (shoulder model)")
    print("=" * 70)
    print(f"Reading from : {DATA_DIR}")
    print(f"Writing to   : {OUTPUT_DIR}")
    print(f"Mode         : {'bilateral overlay' if BILATERAL_OVERLAY else 'one panel per coordinate'}")
    print()

    trials = discover_trials(DATA_DIR)
    if not trials:
        print("[SKIP] no .mot files found in this folder.")
        return

    for task_name, filename in trials:
        plot_trial(task_name, filename)

    print()
    print(f"Done. {len(trials)} trial(s) processed.")


if __name__ == "__main__":
    main()
