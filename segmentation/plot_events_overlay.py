"""
Draw the segmentation events onto both plot sets as dashed vertical lines.
=========================================================================

Two figure sets get the overlay:

    <session>/Outputs/AnglePlots/marker_plots_events/<task>.png
        the .trc marker-position grid, where nearly every event is defined

    <session>/Outputs/AnglePlots/all_coords_events/<task>.png
        the .mot joint-angle grid, for the events defined on coordinates
        (pelvis_rotation turns, the sit-to-stand pelvis_ty and torso events)
        and for reading marker-defined events against joint kinematics

The two originals in marker_plots/ and all_coords/ are left untouched, so a
clean and an annotated copy of each figure exist side by side.

The coordinate grouping, labels, colours and panel layout for the .mot grid
are imported from Analysis/plot_opencap_all_coords.py rather than restated,
so the annotated figures stay in step with the plain ones.

Events whose marker column says to read them off the video are not produced
by these scripts and so never appear here.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seg_common as sc

_ANALYSIS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

import plot_opencap_all_coords as ac
import plot_opencap_trc_markers as tm

# One colour per S-label, cycling. Distinct hues so S1 and S4 are not
# confusable in a dense stride sequence.
PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
           "#393b79", "#637939"]


def _label_colors(events):
    labels = sorted({e["event"] for e in events},
                    key=lambda s: int(s[1:]) if s[1:].isdigit() else 999)
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}


def _event_lines(events):
    """[(time, label, color)] with one legend entry per S-label."""
    colors = _label_colors(events)
    return [(e["time_s"], e["event"], colors[e["event"]]) for e in events]


def _draw_lines(ax, lines):
    for t, _, color in lines:
        ax.axvline(t, color=color, linestyle="--", linewidth=0.9, alpha=0.8)


def _legend_entries(events):
    colors = _label_colors(events)
    seen = {}
    for e in events:
        seen.setdefault(e["event"], e["name"])

    handles = [plt.Line2D([0], [0], color=colors[lab], linestyle="--",
                          linewidth=1.4) for lab in seen]
    labels = [f"{lab}  {name}" for lab, name in seen.items()]
    return handles, labels


def overlay_markers(label, task, events, verbose=True):
    """Annotated copy of the .trc marker grid."""
    sd = sc.session_dir(label)
    trc = os.path.join(sd, "MarkerData", sc.trial_name(label, task) + ".trc")
    out = os.path.join(sd, "Outputs", "AnglePlots", "marker_plots_events")
    path = tm.plot_trial(task, trc, out, event_lines=_event_lines(events))
    return path


def overlay_coords(label, task, events, verbose=True):
    """Annotated copy of the .mot joint-angle grid."""
    sd = sc.session_dir(label)
    time, cols, mot_file = sc.load_mot(label, task)
    df = pd.DataFrame({"time": time, **cols})

    panels, missing = ac.build_panels(df)
    if not panels:
        print(f"[SKIP] {task}: no recognised coordinates in {mot_file}")
        return None

    lines = _event_lines(events)
    n_cols = ac.N_COLS
    n_rows = math.ceil(len(panels) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.6 * n_cols, 2.5 * n_rows),
                             squeeze=False)

    for idx, (plabel, unit, series, caveat) in enumerate(panels):
        ax = axes[idx // n_cols][idx % n_cols]
        roms = []
        for series_label, col, color in series:
            values = df[col].values
            ax.plot(time, values, color=color, linewidth=0.9,
                    label=series_label or None)
            finite = values[np.isfinite(values)]
            if finite.size:
                roms.append((series_label or col,
                             float(finite.max() - finite.min())))

        base = (series[0][1].rsplit("_", 1)[0]
                if series[0][1].endswith(("_r", "_l")) else series[0][1])
        if base in ac.KNOWN_LOCKED and all(r < 1e-6 for _, r in roms):
            title = f"{plabel}  [locked DOF]"
        elif caveat:
            title = f"{plabel}  [{caveat}]"
        else:
            title = plabel

        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel(unit, fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        if len(series) > 1:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.6)

        _draw_lines(ax, lines)

        if idx >= len(panels) - n_cols:
            ax.set_xlabel("Time (s)", fontsize=8)

    for idx in range(len(panels), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    handles, leg_labels = _legend_entries(events)
    free = len(panels)
    bottom = 0.0
    if free < n_rows * n_cols:
        ax = axes[free // n_cols][free % n_cols]
        ax.axis("off")
        ax.legend(handles, leg_labels, fontsize=7, loc="center",
                  title="segmentation events", title_fontsize=8)
    else:
        # The coordinate grid fills every slot (24 panels in 6x4), so the
        # legend needs its own strip or it lands on the bottom row's axis
        # labels.
        ncol = min(6, len(handles))
        legend_rows = math.ceil(len(handles) / ncol)
        bottom = min(0.02 + 0.014 * legend_rows, 0.12)
        fig.legend(handles, leg_labels, fontsize=7, loc="lower center",
                   ncol=ncol)

    duration = time[-1] - time[0] if len(time) > 1 else 0.0
    fig.suptitle(
        f"{task}  --  all coordinates (position) with segmentation events\n"
        f"{mot_file}   |   {len(time)} frames   |   {duration:.2f} s   |   "
        f"{len(events)} events",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, bottom, 1, 0.97 if n_rows > 3 else 0.94])

    out_dir = os.path.join(sd, "Outputs", "AnglePlots", "all_coords_events")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    if verbose:
        print(f"[OK]   {task}: {len(events)} events -> "
              f"all_coords_events/{os.path.basename(out_path)}")
    return out_path


def overlay_all(label, events_by_task, verbose=True):
    written = []
    for task, events in sorted(events_by_task.items()):
        if not events:
            continue
        events = sorted(events, key=lambda e: e["time_s"])
        try:
            p = overlay_markers(label, task, events, verbose=verbose)
            if p:
                written.append(p)
        except Exception as exc:
            print(f"[FAIL] {task} marker overlay: {type(exc).__name__}: {exc}")
        try:
            p = overlay_coords(label, task, events, verbose=verbose)
            if p:
                written.append(p)
        except Exception as exc:
            print(f"[FAIL] {task} coords overlay: {type(exc).__name__}: {exc}")
    return written
