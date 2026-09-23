import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))

# Panels per row. 63 markers -> 8 x 8 grid with one blank slot.
N_COLS = 8

# inches. matches all_coords density so both sets read at one zoom
PANEL_W, PANEL_H = 4.6, 2.5
DPI = 140   # readable on screen without huge files

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

COLOR_X = "#1f77b4"
COLOR_Y = "#d62728"
COLOR_Z = "#2ca02c"

SHOW_RANGE = True

CENTER_COMPONENTS = False


def load_trc_file(filepath):
    """Parse an OpenCap post-augmentation .trc file."""
    with open(filepath, "r") as fh:
        lines = fh.read().split("\n")

    meta_keys = lines[1].split("\t")
    meta_vals = lines[2].split("\t")
    meta = {k.strip(): v.strip() for k, v in zip(meta_keys, meta_vals) if k.strip()}

    # marker names on line 4, two empty leading and trailing fields
    name_fields = lines[3].split("\t")
    names = [n.strip() for n in name_fields[2:] if n.strip()]

    rows = []
    for raw in lines[6:]:
        if not raw.strip():
            continue
        parts = raw.split("\t")
        rows.append([float(p) if p.strip() else np.nan for p in parts])

    if not rows:
        raise ValueError(f"no data rows in {filepath}")

    width = 2 + 3 * len(names)
    data = np.full((len(rows), width), np.nan)
    for i, row in enumerate(rows):
        n = min(len(row), width)
        data[i, :n] = row[:n]

    frames = data[:, 0].astype(int)
    time = data[:, 1]
    markers = {}
    for j, name in enumerate(names):
        markers[name] = data[:, 2 + 3 * j: 5 + 3 * j]

    return frames, time, markers, meta


def plot_trial(task_name, filepath, output_dir, event_lines=None):
    """Write one PNG for one trial."""
    frames, time, markers, meta = load_trc_file(filepath)

    if not markers:
        print(f"[SKIP] {task_name}: no markers parsed")
        return None

    names = list(markers.keys())
    n_rows = math.ceil(len(names) / N_COLS)
    fig, axes = plt.subplots(n_rows, N_COLS,
                             figsize=(PANEL_W * N_COLS, PANEL_H * n_rows),
                             squeeze=False)

    dropouts = []

    for idx, name in enumerate(names):
        ax = axes[idx // N_COLS][idx % N_COLS]
        xyz = markers[name]

        ranges = []
        for comp, color, label in ((0, COLOR_X, "X"),
                                   (1, COLOR_Y, "Y (vert)"),
                                   (2, COLOR_Z, "Z")):
            values = xyz[:, comp]
            finite = values[np.isfinite(values)]
            ranges.append(float(finite.max() - finite.min()) if finite.size else np.nan)
            if CENTER_COMPONENTS and finite.size:
                values = values - float(np.nanmean(values))
            ax.plot(time, values, color=color, linewidth=0.9, label=label)

        if not np.all(np.isfinite(xyz)):
            dropouts.append(name)

        ax.set_title(name, fontsize=8.5)
        ax.set_ylabel("m (centred)" if CENTER_COMPONENTS else "m", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.6, ncol=3)

        if SHOW_RANGE:
            ax.text(0.02, 0.03,
                    "range " + " ".join(f"{v:.3g}" for v in ranges),
                    transform=ax.transAxes, fontsize=6.5, color="dimgray",
                    va="bottom", ha="left")

        if event_lines:
            for t, label, color in event_lines:
                ax.axvline(t, color=color, linestyle="--", linewidth=0.8,
                           alpha=0.75)

        if idx >= len(names) - N_COLS:
            ax.set_xlabel("Time (s)", fontsize=8)

    for idx in range(len(names), n_rows * N_COLS):
        axes[idx // N_COLS][idx % N_COLS].axis("off")

    if event_lines:
        _draw_event_legend(fig, axes, names, event_lines)

    duration = time[-1] - time[0] if len(time) > 1 else 0.0
    fs = (len(time) - 1) / duration if duration > 0 else float("nan")
    title = (f"{task_name}  --  all .trc marker positions\n"
             f"{os.path.basename(filepath)}   |   {len(names)} markers   |   "
             f"{len(time)} frames   |   {duration:.2f} s   |   {fs:.1f} Hz")
    if event_lines:
        title += f"   |   {len(event_lines)} segmentation events"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{task_name}.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)

    note = f"  [dropouts: {', '.join(dropouts)}]" if dropouts else ""
    print(f"[OK]   {task_name}: {len(names)} markers, {len(time)} frames -> "
          f"{os.path.basename(out_path)}{note}")
    return out_path


def _draw_event_legend(fig, axes, names, event_lines):
    """Put the event labels in the unused grid slot, or below the figure."""
    free_idx = len(names)
    n_rows = axes.shape[0]
    if free_idx < n_rows * N_COLS:
        ax = axes[free_idx // N_COLS][free_idx % N_COLS]
        ax.axis("off")
        handles = [plt.Line2D([0], [0], color=c, linestyle="--", linewidth=1.2)
                   for _, _, c in event_lines]
        labels = [f"{lab}  {t:.3f}s" for t, lab, _ in event_lines]
        ax.legend(handles, labels, fontsize=6.5, loc="center",
                  title="segmentation events", title_fontsize=7.5, ncol=1)


def is_session(path):
    """A session folder is any folder holding a MarkerData subfolder."""
    return os.path.isdir(os.path.join(path, "MarkerData"))


def session_label(path):
    """Short name for a session folder, e.g. PR_Trial_OpenCapData_9f3 -> PR."""
    base = os.path.basename(os.path.abspath(path).rstrip(os.sep))
    return base.split("_OpenCapData")[0].replace("_Trial", "") or base


def sessions_under(parent):
    """Every session folder directly inside a parent folder."""
    return [os.path.join(parent, name) for name in sorted(os.listdir(parent))
            if is_session(os.path.join(parent, name))]


def find_session_from(start):
    """Walk up from a folder to the first session folder at or above it.

    Lets the script be dropped inside a patient folder -- next to MarkerData,
    or in MarkerData/ or OpenSimData/Kinematics/ alongside the other plotters
    -- and run with no arguments, the same way plot_opencap_all_coords.py
    finds its own session.
    """
    path = os.path.abspath(start)
    for _ in range(4):
        if is_session(path):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None


def resolve_sessions(target=None):
    """[(label, session folder)] for a path, a parent of sessions, or no arg."""
    if target is None:
        for start in (os.getcwd(), HERE):
            found = find_session_from(start)
            if found:
                return [(session_label(found), found)]
        print("[WARN] no session folder found at or above this script or the "
              "current folder. Pass the patient folder as an argument:\n"
              "         python plot_opencap_trc_markers.py \"/path/to/patient folder\"")
        return []

    if is_session(target):
        path = os.path.abspath(target)
        return [(session_label(path), path)]

    if os.path.isdir(target):
        found = sessions_under(target)
        if not found:
            print(f"[WARN] {target} holds no session folders "
                  f"(no subfolder with a MarkerData folder in it)")
        return [(session_label(p), p) for p in found]

    # Not a path: hand it to seg_common, which resolves study labels like "PR"
    for _p in (os.path.join(HERE, "segmentation"), HERE):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import seg_common as sc
    return [(sc.short_label(target), sc.session_dir(target))]


def resolve_trials(marker_dir):
    """Return [(task_name, filename)] for a session's MarkerData folder."""
    resolved = []
    for fname in sorted(os.listdir(marker_dir)):
        if not fname.endswith(".trc"):
            continue
        if "deleted" in fname.lower():
            print(f"[SKIP] {fname}: 'deleted' in filename")
            continue
        trial = fname[:-4]
        if trial == "neutral":
            continue
        resolved.append((TRIAL_ALIASES.get(trial, trial), fname))
    return resolved


def main():
    args = sys.argv[1:]
    sessions = []
    for target in (args or [None]):
        try:
            sessions += resolve_sessions(target)
        except Exception as exc:
            print(f"[FAIL] {target}: {type(exc).__name__}: {exc}")
    if not sessions:
        return 1

    total = 0
    for label, session_dir in sessions:
        marker_dir = os.path.join(session_dir, "MarkerData")
        output_dir = os.path.join(session_dir, "Outputs", "AnglePlots",
                                  "marker_plots")
        print(f"\n=== {label}")
        print(f"Reading from : {marker_dir}")
        print(f"Writing to   : {output_dir}")
        for task_name, fname in resolve_trials(marker_dir):
            try:
                if plot_trial(task_name, os.path.join(marker_dir, fname),
                              output_dir):
                    total += 1
            except Exception as exc:
                print(f"[FAIL] {task_name}: {type(exc).__name__}: {exc}")

    print(f"\nWrote {total} marker-plot PNGs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
