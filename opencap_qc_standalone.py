#!/usr/bin/env python3
"""
opencap_qc_standalone.py
========================

Waveform quality-control for OpenCap kinematics, in a single self-contained file.

Point it at a folder of OpenCap sessions (or one session) and it produces, per run:

  trials/<session>_<trial>.png    one diagnostic page per trial, right/left overlaid
  qc_features.csv                 one row per trial x coordinate (observable features)
  qc_trials.csv                   one row per trial (worst-offender summary)
  coords_<trial>.png              per-coordinate detail, one plot per distinct trial name
  data_dictionary.md              what every column means and how it is computed

No participant list is needed -- trials are discovered by walking the folder.
No pass/fail thresholds are applied; these are observable features, and criteria
should come later from real distributions.

USAGE
-----
    python opencap_qc_standalone.py --data "PATH/TO/OpenCap/sessions" --out qc_out

`--data` may be either:
  * a parent folder containing OpenCapData_<uuid>/ session folders, or
  * a single OpenCapData_<uuid>/ session folder.

It looks for kinematics at  <session>/OpenSimData/Kinematics/*.mot .

Filter to specific trials with, e.g.:
    python opencap_qc_standalone.py --data ... --trials squat walk

REQUIREMENTS
------------
    numpy, pandas, scipy, matplotlib     (pip install numpy pandas scipy matplotlib)

NOTES
-----
* "Trial name" is just the .mot filename stem, lower-cased. No task vocabulary is
  imposed, so this works whatever your protocol's trials are called.
* Two derived shoulder quantities are added when the inputs exist:
    arm_elev = arccos(cos(arm_flex)*cos(arm_add))   humeral elevation, well-conditioned
    euler_cond = |1/cos(arm_add)|                   error amplification for arm_flex/arm_rot
  See the data dictionary for interpretation.
* Sampling rate is always inferred per trial from the time column; OpenCap data
  can contain both 59 and 60 Hz trials.

Shared as a courtesy QC starter. Adapt freely.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Coordinate bookkeeping
# ---------------------------------------------------------------------------

PAIRED_BASES = [
    "hip_flexion", "hip_adduction", "hip_rotation",
    "knee_angle", "ankle_angle", "subtalar_angle",
    "arm_flex", "arm_add", "arm_rot", "elbow_flex", "pro_sup",
    "arm_elev", "euler_cond",           # derived; see add_derived_columns
]
UNPAIRED = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
]
TRANSLATION_DOFS = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}
EXCLUDED_PREFIXES = ("mtp_",)          # fixed joint, constant 0
EXCLUDED_SUFFIXES = ("_beta",)         # dependent coordinate, old processing only
EXCLUDED_FEATURE_PREFIXES = ("euler_cond",)   # diagnostic covariate, not a signal

REGIONS = {
    "pelvis": ["pelvis_tilt", "pelvis_list", "pelvis_rotation",
               "pelvis_tx", "pelvis_ty", "pelvis_tz"],
    "lumbar": ["lumbar_extension", "lumbar_bending", "lumbar_rotation"],
    "hip":    ["hip_flexion_r", "hip_flexion_l", "hip_adduction_r", "hip_adduction_l",
               "hip_rotation_r", "hip_rotation_l"],
    "knee":   ["knee_angle_r", "knee_angle_l"],
    "foot":   ["ankle_angle_r", "ankle_angle_l", "subtalar_angle_r", "subtalar_angle_l"],
    "arm":    ["arm_elev_r", "arm_elev_l", "arm_flex_r", "arm_flex_l",
               "arm_add_r", "arm_add_l", "arm_rot_r", "arm_rot_l",
               "elbow_flex_r", "elbow_flex_l", "pro_sup_r", "pro_sup_l"],
}
REGION_ORDER = {r: i for i, r in enumerate(["pelvis", "lumbar", "hip", "knee", "foot", "arm", "other"])}
PLAUSIBLE_DEG = (-180.0, 180.0)

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_mot(mot_path: Path) -> pd.DataFrame:
    """Read an OpenSim .mot into a DataFrame with a numeric 'time' column."""
    mot_path = Path(mot_path)
    with mot_path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()
    end_idx = next((i for i, ln in enumerate(lines) if "endheader" in ln.strip().lower()), None)
    if end_idx is None:
        raise ValueError(f"no 'endheader' in {mot_path}")
    df = pd.read_csv(mot_path, engine="python", skiprows=end_idx + 1, sep=r"\t")
    df.columns = [c.strip() for c in df.columns]
    if "time" not in df.columns:
        lower = {c.lower(): c for c in df.columns}
        if "time" in lower:
            df = df.rename(columns={lower["time"]: "time"})
        else:
            raise ValueError(f"no 'time' column in {mot_path}")
    df = df.dropna(how="all").reset_index(drop=True)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).reset_index(drop=True)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append humeral elevation and Euler conditioning where inputs exist."""
    for side in ("r", "l"):
        flex, add = f"arm_flex_{side}", f"arm_add_{side}"
        if flex in df.columns and add in df.columns:
            f = np.radians(df[flex].to_numpy(float))
            a = np.radians(df[add].to_numpy(float))
            df[f"arm_elev_{side}"] = np.degrees(np.arccos(np.clip(np.cos(f) * np.cos(a), -1, 1)))
            with np.errstate(divide="ignore", invalid="ignore"):
                df[f"euler_cond_{side}"] = np.abs(1.0 / np.cos(a))
    return df


def panel_layout(columns):
    present = set(columns)
    panels = [(b, [b]) for b in UNPAIRED if b in present]
    for base in PAIRED_BASES:
        cols = [c for c in (f"{base}_r", f"{base}_l") if c in present]
        if cols:
            panels.append((base, cols))
    return panels


def analysis_columns(columns):
    out = []
    for c in columns:
        if c == "time" or c.startswith(EXCLUDED_PREFIXES) or c.endswith(EXCLUDED_SUFFIXES):
            continue
        if c.startswith(EXCLUDED_FEATURE_PREFIXES):
            continue
        out.append(c)
    return out


def region_of(coord: str) -> str:
    for region, members in REGIONS.items():
        if coord in members:
            return region
    if coord in UNPAIRED:
        return "pelvis" if coord.startswith("pelvis") else "lumbar"
    return "other"

# ---------------------------------------------------------------------------
# Signal features
# ---------------------------------------------------------------------------

def infer_fs(time, fallback=60.0):
    time = np.asarray(time, float)
    if time.size < 2:
        return float(fallback)
    dt = float(np.median(np.diff(time)))
    return float(1.0 / dt) if np.isfinite(dt) and dt > 0 else float(fallback)


def lowpass(x, fs, cutoff_hz, order=4):
    x = np.asarray(x, float)
    if x.size < order * 3 + 1 or not np.isfinite(x).any():
        return x
    nans = ~np.isfinite(x)
    if nans.all():
        return x
    filled = x.copy()
    if nans.any():
        idx = np.arange(x.size)
        filled[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    if cutoff_hz / (0.5 * fs) >= 1.0:
        return x
    b, a = sps.butter(order, cutoff_hz / (0.5 * fs), btype="low")
    y = sps.filtfilt(b, a, filled)
    y[nans] = np.nan
    return y


def psd(x, fs, nperseg_sec=4.0):
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    if finite.sum() < 16:
        return np.array([]), np.array([])
    xi = x.copy()
    if not finite.all():
        idx = np.arange(x.size)
        xi[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    xi = xi - np.mean(xi)
    nperseg = int(min(len(xi), max(64, round(nperseg_sec * fs))))
    return sps.welch(xi, fs=fs, nperseg=nperseg)


def power_frequency(freqs, power, fraction):
    if freqs.size == 0:
        return np.nan
    cumulative = np.cumsum(power)
    total = cumulative[-1]
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return float(np.interp(fraction * total, cumulative, freqs))


def longest_flat_run(x, eps):
    x = np.asarray(x, float)
    if x.size < 2:
        return 0
    flat = np.abs(np.diff(x)) < eps
    best = run = 0
    for f in flat:
        run = run + 1 if f else 0
        best = max(best, run)
    return int(best)


def settling_transient(x, max_frames=30, factor=3.0):
    x = np.asarray(x, float)
    d = np.abs(np.diff(x))
    if d.size < 5:
        return np.nan, 0
    med = float(np.median(d))
    if not np.isfinite(med) or med <= 0:
        return np.nan, 0
    n = 0
    for v in d[:max_frames]:
        if np.isfinite(v) and v > factor * med:
            n += 1
        else:
            break
    return float(d[0] / med), int(n)


def hf_residual(x, fs, cutoff_hz=6.0):
    x = np.asarray(x, float)
    if x.size < 20 or not np.isfinite(x).any():
        return np.nan
    return float(np.nanpercentile(np.abs(x - lowpass(x, fs, cutoff_hz)), 99))


def describe(x, fs, is_translation):
    x = np.asarray(x, float)
    n = x.size
    finite = np.isfinite(x)
    out = {"units": "m" if is_translation else "deg", "n_frames": n,
           "n_nan": int((~finite).sum()),
           "pct_nan": float((~finite).sum() / n * 100) if n else np.nan}
    blank = {k: np.nan for k in
             ("value_min", "p05", "p50", "p95", "value_max", "rom", "spike_ratio",
              "max_abs_vel", "flat_run_frames", "hf_resid_p99", "hf_resid_pct_rom",
              "first_step_ratio", "n_settle_frames", "f95_hz", "f99_hz")}
    if finite.sum() < 4:
        return {**out, **blank}
    xf = x[finite]
    lo, hi = float(np.min(xf)), float(np.max(xf))
    p05, p50, p95 = (float(v) for v in np.percentile(xf, [5, 50, 95]))
    rom, robust = hi - lo, p95 - p05
    out.update({
        "value_min": lo, "p05": p05, "p50": p50, "p95": p95, "value_max": hi, "rom": rom,
        "spike_ratio": float(rom / robust) if robust > 0 else np.nan,
        "max_abs_vel": float(np.nanmax(np.abs(np.diff(x))) * fs),
        "flat_run_frames": longest_flat_run(x, 1e-6 if is_translation else 1e-4),
    })
    resid = hf_residual(x, fs)
    out["hf_resid_p99"] = resid
    out["hf_resid_pct_rom"] = float(100 * resid / rom) if rom > 0 and np.isfinite(resid) else np.nan
    ratio, n_settle = settling_transient(x)
    out["first_step_ratio"], out["n_settle_frames"] = ratio, n_settle
    freqs, power = psd(x, fs)
    out["f95_hz"] = power_frequency(freqs, power, 0.95)
    out["f99_hz"] = power_frequency(freqs, power, 0.99)
    return out

# ---------------------------------------------------------------------------
# Trial-level summary
# ---------------------------------------------------------------------------

def trial_summary(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    keys = ["session", "trial", "fs_hz"]
    ang = features[features.units == "deg"]

    def worst(group, column):
        sub = group[group[column].notna()]
        if sub.empty:
            return np.nan, None
        row = sub.loc[sub[column].idxmax()]
        return float(row[column]), str(row["coord"])

    rows = []
    for key, g in features.groupby(keys, dropna=False):
        noise, noise_c = worst(g, "hf_resid_pct_rom")
        vel, vel_c = worst(g, "max_abs_vel")
        flat, flat_c = worst(g, "flat_run_frames")
        ga = ang[ang.index.isin(g.index)]
        imp = ga[(ga.value_min < PLAUSIBLE_DEG[0]) | (ga.value_max > PLAUSIBLE_DEG[1])]
        rows.append({
            **dict(zip(keys, key)),
            "n_frames": int(g.n_frames.max()),
            "duration_s": float(g.n_frames.max() / g.fs_hz.max()),
            "n_coords": int(g.coord.nunique()),
            "pct_nan_max": float(g.pct_nan.max()),
            "worst_hf_resid_pct_rom": noise, "worst_hf_resid_coord": noise_c,
            "max_abs_vel": vel, "max_abs_vel_coord": vel_c,
            "settle_frames_max": int(g.n_settle_frames.max()) if g.n_settle_frames.notna().any() else 0,
            "flat_run_max": int(flat) if np.isfinite(flat) else 0, "flat_run_coord": flat_c,
            "flat_run_pct_of_trial": float(100 * flat / g.n_frames.max()) if np.isfinite(flat) else np.nan,
            "n_coords_implausible": int(len(imp)),
            "implausible_coords": ";".join(sorted(imp.coord)) or None,
            "f99_hz_max": float(g.f99_hz.max()), "f99_hz_median": float(g.f99_hz.median()),
        })
    return pd.DataFrame(rows).sort_values(["trial", "session"])

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

C_RIGHT, C_LEFT, C_SINGLE = "#0072B2", "#D55E00", "#0072B2"
GRID = dict(linewidth=0.5, alpha=0.35)


def _side_style(col):
    if col.endswith("_r"):
        return C_RIGHT, "right"
    if col.endswith("_l"):
        return C_LEFT, "left"
    return C_SINGLE, col


def trial_sheet(df, title, out_path, ncols=4, dpi=110):
    time = df["time"].to_numpy(float)
    panels = panel_layout(list(df.columns))
    if not panels:
        return None
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.1 * nrows), dpi=dpi, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, cols) in zip(axes, panels):
        for col in cols:
            colour, label = _side_style(col)
            ax.plot(time, df[col].to_numpy(float), lw=1.0, color=colour, label=label)
        ax.set_title(name, fontsize=8.5)
        ax.grid(True, **GRID)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if len(cols) > 1:
            ax.legend(fontsize=6.5, frameon=False, loc="best")
        ax.set_ylabel("m" if name in TRANSLATION_DOFS else "deg", fontsize=7)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11, y=0.998)
    fig.supxlabel("time (s)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def coordinate_report(features, trial, out_path, dpi=115):
    """Per-coordinate detail for one trial name, region-grouped. Angles only."""
    sub = features[(features.trial == trial) & (features.units == "deg")].copy()
    if sub.empty:
        return None
    sub["region"] = [region_of(c) for c in sub.coord]
    order = sub.groupby("coord").region.first().reset_index()
    order["rrank"] = order.region.map(lambda r: REGION_ORDER.get(r, 99))
    order = order.sort_values(["rrank", "region", "coord"])
    coords = list(order.coord)
    ypos = {c: i for i, c in enumerate(reversed(coords))}
    panels = [("rom", "range of motion (deg)"),
              ("hf_resid_pct_rom", "high-frequency residual (% of ROM)"),
              ("f99_hz", "f99 (Hz)")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 0.34 * len(coords) + 3.0), dpi=dpi, sharey=True)
    rng = np.random.default_rng(0)
    for ax, (col, xlabel) in zip(axes, panels):
        part = sub[sub[col].notna()]
        y = np.array([ypos[c] for c in part.coord]) + rng.uniform(-0.17, 0.17, len(part))
        ax.scatter(part[col], y, s=22, color=C_SINGLE, alpha=0.5, linewidths=0)
        for c in coords:
            vals = sub.loc[sub.coord == c, col].dropna()
            if len(vals):
                ax.plot([vals.median()] * 2, [ypos[c] - 0.32, ypos[c] + 0.32],
                        color="#111111", lw=1.6, solid_capstyle="round", zorder=5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(True, axis="x", **GRID)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(range(len(coords)))
    axes[0].set_yticklabels(list(reversed(coords)), fontsize=8.5)
    axes[0].set_ylim(-0.8, len(coords) - 0.2)
    reg = {c: region_of(c) for c in coords}
    rev = list(reversed(coords))
    for i in range(1, len(coords)):
        if reg[rev[i - 1]] != reg[rev[i]]:
            for ax in axes:
                ax.axhline(i - 0.5, color="0.8", lw=0.8)
    n = sub.groupby("coord").size().max()
    fig.suptitle(f"{trial} — per-coordinate detail   (one dot per trial instance, n={n})", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ---------------------------------------------------------------------------
# Data dictionary
# ---------------------------------------------------------------------------

DICTIONARY_MD = """# QC data dictionary

Observable signal-quality features. No pass/fail thresholds are applied.

## qc_features.csv  (one row per trial x coordinate)

| column | units | how it is computed | how to read it |
|---|---|---|---|
| session | - | OpenCapData_<uuid> folder name | trial provenance |
| trial | - | .mot filename stem, lower-cased | your protocol's trial name |
| fs_hz | Hz | 1 / median(diff(time)) | inferred per trial; OpenCap mixes 59 and 60 Hz |
| coord | - | OpenSim coordinate, or a derived one | one row per coordinate |
| units | deg / m | m for pelvis translations, else deg | never pool units in one statistic |
| n_frames | count | rows after dropping blank trailing lines | duration = n_frames / fs_hz |
| n_nan, pct_nan | count, % | non-finite values | missingness |
| value_min, value_max | deg / m | min and max of finite values | noise-sensitive: right for plausibility, wrong for movement. An angle outside +/-180 deg is impossible |
| p05, p50, p95 | deg / m | 5th / 50th / 95th percentile | robust description of movement |
| rom | deg / m | value_max - value_min | full excursion incl. spikes |
| spike_ratio | ratio | rom / (p95 - p05) | > 2 means the range is driven by outliers |
| max_abs_vel | deg/s or m/s | max abs frame-to-frame change x fs, UNFILTERED | a spike detector, not physiological velocity |
| flat_run_frames | count | longest run of |diff| < 1e-4 deg (1e-6 m) | dropout, held values, or a coordinate clamped at a model limit |
| hf_resid_p99 | deg / m | 99th pct of |x - lowpass(x, 6 Hz)| | departure from a smooth version, in the coordinate's units |
| hf_resid_pct_rom | % | 100 x hf_resid_p99 / rom | noise relative to range, comparable across coordinates |
| first_step_ratio | ratio | |x[1]-x[0]| / median|diff| | inverse-kinematics start-up transient |
| n_settle_frames | count | leading frames with step > 3x median | how many frames to discard from the start |
| f95_hz, f99_hz | Hz | freq below which 95% / 99% of Welch PSD power lies | filter-cutoff guide; read together with rom |

## qc_trials.csv  (one row per trial)

Worst-offender roll-up of the above: worst_hf_resid_pct_rom (+coord), max_abs_vel
(+coord), settle_frames_max, flat_run_max (+coord, +pct_of_trial),
n_coords_implausible (+names), f99_hz_max, f99_hz_median.

## Derived coordinates (added when arm_flex and arm_add are present)

| coordinate | units | how it is computed | how to read it |
|---|---|---|---|
| arm_elev_r/l | deg | arccos(cos(arm_flex) x cos(arm_add)) | humeral elevation: how far the arm is raised, any direction. 0 down, 90 horizontal, 180 overhead. Free of the arm_add = +/-90 deg singularity that corrupts arm_flex/arm_rot |
| euler_cond_r/l | ratio | \\|1 / cos(arm_add)\\| | per-frame error amplification for arm_flex and arm_rot ONLY. 1 at 0 deg, 11.5 at 85 deg, unbounded at 90 deg. Plotted per trial; kept out of the feature table (its ROM/spectrum are meaningless) |

## Excluded

mtp_angle_r/l (fixed joint), knee_angle_*_beta (dependent coordinate, old
processing), euler_cond (diagnostic covariate, not a signal).
"""

# ---------------------------------------------------------------------------
# Discovery + driver
# ---------------------------------------------------------------------------

def find_sessions(data_root: Path):
    """Return [(session_name, kinematics_dir), ...]."""
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.exists():
        sys.exit(f"ERROR: --data path does not exist: {data_root}")
    # a single session folder?
    if (data_root / "OpenSimData" / "Kinematics").is_dir():
        return [(data_root.name, data_root / "OpenSimData" / "Kinematics")]
    # otherwise every OpenCapData_* (or any folder with the right subtree)
    sessions = []
    for child in sorted(data_root.iterdir()):
        kin = child / "OpenSimData" / "Kinematics"
        if kin.is_dir():
            sessions.append((child.name, kin))
    if not sessions:
        # last resort: any *.mot anywhere under data_root
        mots = list(data_root.rglob("*.mot"))
        if mots:
            return [(data_root.name, data_root)]
        sys.exit(f"ERROR: no OpenSim Kinematics folders or .mot files under {data_root}")
    return sessions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="OpenCap sessions parent folder, or one session folder")
    ap.add_argument("--out", default="qc_out", help="output folder (created if absent)")
    ap.add_argument("--trials", nargs="*", default=None,
                    help="only these trial names (filename stems), e.g. --trials squat walk")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    (out_dir / "trials").mkdir(parents=True, exist_ok=True)

    sessions = find_sessions(args.data)
    print(f"found {len(sessions)} session(s) under {args.data}\n")

    keep = {t.lower() for t in args.trials} if args.trials else None
    feature_rows, loaded = [], {}
    n_ok = n_fail = 0

    for session_name, kin_dir in sessions:
        session_id = session_name.replace("OpenCapData_", "")
        for mot in sorted(kin_dir.glob("*.mot")):
            stem = mot.stem
            trial = re.sub(r"_\d+$", "", stem).lower()      # squat_1 -> squat
            if keep is not None and trial not in keep:
                continue
            tag = f"{session_id[:8]}_{stem}"
            try:
                df = add_derived_columns(read_mot(mot))
            except Exception as exc:
                n_fail += 1
                print(f"  FAILED  {tag}: {type(exc).__name__}: {exc}")
                continue
            fs = infer_fs(df["time"].to_numpy(float))
            for c in analysis_columns(list(df.columns)):
                feature_rows.append({
                    "session": session_id, "trial": trial, "raw_trial": stem,
                    "fs_hz": round(fs, 3), "coord": c,
                    **describe(df[c].to_numpy(float), fs, c in TRANSLATION_DOFS),
                })
            trial_sheet(df, f"{session_id[:8]} · {stem} · {fs:.1f} Hz",
                        out_dir / "trials" / f"{tag}.png")
            loaded.setdefault(trial, []).append(df)
            n_ok += 1
            print(f"  ok      {tag}  {len(df)} frames  {fs:.1f} Hz")

    if not feature_rows:
        sys.exit("\nno trials processed -- check --data and --trials")

    features = pd.DataFrame(feature_rows)
    trials = trial_summary(features)
    features.to_csv(out_dir / "qc_features.csv", index=False)
    trials.to_csv(out_dir / "qc_trials.csv", index=False)
    (out_dir / "data_dictionary.md").write_text(DICTIONARY_MD, encoding="utf-8")

    for trial in sorted(loaded):
        coordinate_report(features, trial, out_dir / f"coords_{trial}.png")

    print("\n" + "=" * 64)
    print(f"{n_ok} trials processed, {n_fail} failed")
    print(f"{len(features)} trial x coordinate rows, {len(trials)} trials, "
          f"{features.trial.nunique()} distinct trial names")
    print("=" * 64)
    print(f"\nwrote:\n  {out_dir/'trials'}/*.png      per-trial diagnostic pages")
    print(f"  {out_dir/'qc_features.csv'}")
    print(f"  {out_dir/'qc_trials.csv'}")
    print(f"  {out_dir}/coords_<trial>.png   per-coordinate detail")
    print(f"  {out_dir/'data_dictionary.md'}")

    if not trials.empty:
        print("\ntrials with anatomically implausible angles (|angle| > 180 deg):")
        imp = trials[trials.n_coords_implausible > 0]
        if len(imp):
            print(imp[["session", "trial", "n_coords_implausible", "implausible_coords"]].to_string(index=False))
        else:
            print("  none")


if __name__ == "__main__":
    main()
