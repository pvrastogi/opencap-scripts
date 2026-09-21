"""
OpenCap TRC Marker Position Plotter
===================================

Plots the X / Y / Z position trace of every marker in a trial's .trc file,
one multi-panel figure per trial, one panel per marker with the three
components overlaid.

WHY .trc AND NOT .mot
    The segmentation markers document defines nearly every event on marker
    positions from the post-augmentation .trc files (r_calc_study,
    L_toe_study, r.PSIS_study, wrist markers, ...), not on the joint angles
    in the .mot files. plot_opencap_all_coords.py covers the .mot side; this
    is the marker-position counterpart.

MARKER SET
    There is exactly one .trc per trial in every session -- marker
    augmentation is model-independent, so there is no _shoulder.trc variant
    the way there is a _shoulder.mot. All five sessions carry the same
    63-marker set at 60 Hz in metres, so no file selection is needed here.
    Note the asymmetric capitalisation in the OpenCap marker set: right side
    is lowercase (r_calc_study), left side is capital (L_calc_study).

AXES
    OpenCap's lab frame is Y-up. X and Z are the two horizontal axes; which
    one is "forward" depends on how the subject was oriented relative to the
    cameras, which is why the segmentation definitions project onto a
    per-frame pelvis heading rather than trusting a lab axis.

USAGE
    python3 plot_opencap_trc_markers.py            # all sessions
    python3 plot_opencap_trc_markers.py PR         # one session by prefix

Output
    PNG files saved to <session>/Outputs/AnglePlots/marker_plots/
"""

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Session folder prefixes, in the canonical order used across the analysis.
SESSION_PREFIXES = ["MG_Trial", "OLD_MG_Trial", "NEW_GX_Trial",
                    "OLD_GX_Trial", "PR_Trial"]

# Panels per row. 63 markers -> 8 x 8 grid with one blank slot.
N_COLS = 8

# Per-panel size in inches. Matches the density of the all_coords plots so
# the two figure sets are readable at the same zoom.
PANEL_W, PANEL_H = 4.6, 2.5
DPI = 140

# OLD_GX used non-standard trial names. Output files are named by the
# canonical task so they line up across sessions. User-confirmed mapping.
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

# Print each marker's per-axis range in the panel corner, same idea as the
# ROM readout in the all_coords plots: a flat trace is either a marker that
# genuinely does not move or a dropout, and the number tells you which.
SHOW_RANGE = True

# With all three components on one shared axis, a walking trial's fore-aft
# translation (several metres) flattens the vertical and lateral traces
# (tens of centimetres) into near-straight lines. Set this True to plot each
# component minus its own mean, which puts all three on a comparable scale
# and makes heel-rise and lateral sway legible, at the cost of losing
# absolute position. The range readout stays in absolute metres either way.
CENTER_COMPONENTS = False


# ---------------------------------------------------------------------------
# 2. TRC PARSING
# ---------------------------------------------------------------------------

def load_trc_file(filepath):
    """Parse an OpenCap post-augmentation .trc file.

    Layout:
        line 1  PathFileType 4 (X/Y/Z) <path>
        line 2  header field names
        line 3  header field values
        line 4  Frame#  Time  <marker1>  ''  ''  <marker2> ...
        line 5  ''  ''  X1  Y1  Z1  X2  Y2  Z2 ...
        line 6  blank
        line 7+ data rows

    Returns (frames, time, markers, meta) where markers is an ordered dict of
    name -> (n_frames, 3) array in metres and frames is the file's own 1-based
    Frame# column.
    """
    with open(filepath, "r") as fh:
        lines = fh.read().split("\n")

    meta_keys = lines[1].split("\t")
    meta_vals = lines[2].split("\t")
    meta = {k.strip(): v.strip() for k, v in zip(meta_keys, meta_vals) if k.strip()}

    # Marker names sit on line 4, padded with two empty leading fields
    # (Frame#, Time) and two empty trailing fields per marker.
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


# ---------------------------------------------------------------------------
# 3. FIGURE
# ---------------------------------------------------------------------------

def plot_trial(task_name, filepath, output_dir, event_lines=None):
    """Write one PNG for one trial.

    event_lines, when given, is a list of (time_s, label, color) drawn as
    dashed vertical lines across every panel.
    """
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


# ---------------------------------------------------------------------------
# 4. DRIVER
# ---------------------------------------------------------------------------

def resolve_sessions(filter_prefix=None):
    # An argument that is not one of the five controls -- a DCM_### label or
    # a path to a session folder -- goes through seg_common.session_dir,
    # which knows both folder-naming conventions and DCM_SESSION_ROOTS.
    # Resolving it here instead would mean two copies of that logic.
    if filter_prefix and not any(p.startswith(filter_prefix)
                                 for p in SESSION_PREFIXES):
        # seg_common sits either in a segmentation/ subfolder (the
        # Data/Analysis layout) or right next to this script (the
        # single-folder layout the SOP describes). Both are tried.
        _here = os.path.dirname(os.path.abspath(__file__))
        for _p in (os.path.join(_here, "segmentation"), _here):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        import seg_common as sc
        return [(sc.short_label(filter_prefix),
                 sc.session_dir(filter_prefix))]

    sessions = []
    for prefix in SESSION_PREFIXES:
        if filter_prefix and not prefix.startswith(filter_prefix):
            continue
        matches = [d for d in os.listdir(BASE)
                   if d.startswith(prefix + "_OpenCapData")]
        if not matches:
            print(f"[WARN] no session folder found for {prefix}")
            continue
        sessions.append((prefix.replace("_Trial", ""),
                         os.path.join(BASE, matches[0])))
    return sessions


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
    filter_prefix = sys.argv[1] if len(sys.argv) > 1 else None
    sessions = resolve_sessions(filter_prefix)

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


if __name__ == "__main__":
    main()
