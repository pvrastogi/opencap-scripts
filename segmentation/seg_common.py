import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))


def _find_opencap():
    """Locate the opencap-processing checkout."""
    override = os.environ.get("OPENCAP_PROCESSING_DIR", "").strip()
    candidates = ([override] if override else []) + [
        os.path.join(BASE, "..", "opencap-processing"),   # Data/Analysis/segmentation
        os.path.join(HERE, "..", "opencap-processing"),   # one folder up
        os.path.join(HERE, "opencap-processing"),         # alongside the scripts
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(os.path.join(c, "ActivityAnalyses")):
            return c
    return os.path.abspath(candidates[-1])


OPENCAP_DIR = _find_opencap()

for _p in (OPENCAP_DIR, os.path.join(OPENCAP_DIR, "ActivityAnalyses")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("API_TOKEN", "offline-local-data-only")

SESSION_PREFIXES = ["MG_Trial", "OLD_MG_Trial", "NEW_GX_Trial",
                    "OLD_GX_Trial", "PR_Trial"]

SESSION_ROOTS = [BASE] + [p for p in os.environ.get(
    "DCM_SESSION_ROOTS", "").split(os.pathsep) if p.strip()]

TASKS_IN_SCOPE = [
    "standing-ec", "rtt", "semi-tandem", "tandem", "sls-ec",
    "5tsts", "march",
    "gait", "gait-start", "gait-head-turn", "gait-tandem",
    "gait-bckw", "gait-bckw-3s", "gait-pivot",
    "frt", "y-balance", "arms", "tug",
]

TASKS_OUT_OF_SCOPE = ["grapevine", "dext-left", "dext-right", "sls-rtt-ec"]

# OLD_GX used non-standard trial names; user-confirmed mapping.
TRIAL_ALIASES = {
    "standing-ec": "stand",
    "frt": "reach",
    "5tsts": "sts",
    "sls-ec": "left_leg_raise",
    "gait": "gait_1",
    "gait-head-turn": "gait-headturn",
    "dext-left": "dexterity1",
    "dext-right": "dexterityright",
}

SESSION_TRIAL_ALIASES = {
    "OLD_GX": TRIAL_ALIASES,
}

SAMPLE_RATE = 60.0   # Hz, OpenCap's fixed capture rate


def session_dir(label):
    """Resolve a session label or folder path to its session folder."""
    if os.path.isdir(os.path.join(label, "MarkerData")):
        return os.path.abspath(label)

    prefixes = (label + "_OpenCapData", label + "_Trial_OpenCapData")
    searched = []
    for root in SESSION_ROOTS:
        if not os.path.isdir(root):
            continue
        searched.append(root)
        matches = sorted(d for d in os.listdir(root)
                         if d.startswith(prefixes))
        if matches:
            return os.path.join(root, matches[0])
    raise FileNotFoundError(
        f"no session folder for {label} in {searched}. Pass the folder path, "
        f"or add its parent to DCM_SESSION_ROOTS.")


def short_label(label):
    """Filename-safe short name, for a label or a session folder path."""
    if os.sep not in label:
        return label
    base = os.path.basename(os.path.abspath(label))
    return base.split("_OpenCapData")[0] or base


def trial_name(label, task):
    """Map a canonical task name to the on-disk trial name for a session."""
    # short_label, because run_all passes a PATH: a raw prefix test never fires
    for prefix, aliases in SESSION_TRIAL_ALIASES.items():
        if short_label(label).startswith(prefix):
            return aliases.get(task, task)
    return task


def trial_exists(label, task):
    sd = session_dir(label)
    trc = os.path.join(sd, "MarkerData", trial_name(label, task) + ".trc")
    return os.path.exists(trc)


def load_mot(label, task):
    """Read a trial's joint-angle .mot as (time, {column: array})."""
    kin = os.path.join(session_dir(label), "OpenSimData", "Kinematics")
    tn = trial_name(label, task)
    for cand in (tn + "_shoulder.mot", tn + ".mot"):
        path = os.path.join(kin, cand)
        if os.path.exists(path):
            break
    else:
        raise FileNotFoundError(f"no .mot for {label}/{task} in {kin}")

    with open(path) as fh:
        lines = fh.read().split("\n")
    hdr = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader")
    cols = lines[hdr + 1].split()
    rows = [[float(v) for v in l.split()] for l in lines[hdr + 2:] if l.strip()]
    data = np.asarray(rows)
    return data[:, 0], {c: data[:, i] for i, c in enumerate(cols)}, os.path.basename(path)


PEAK_MIN_DEG = 90.0      # below this, the elevation peak is the task, not a raise
DROP_FRACTION = 0.25     # "drop-after-peak" = fall to this fraction of the peak
PEAK_PROMINENCE_DEG = 30.0   # to count as a distinct elevation peak

# max share of a trial the trim may eat; controls top out at 54.9%
MAX_TRIM_FRACTION = 0.75

NO_CALIBRATION = {
    "OLD_GX": {"*"},    # performed no calibration raise on any trial
}

NO_CALIBRATION_FILE = "no_calibration.txt"


def no_calibration(label, task):
    """True if this trial is declared to have no usable calibration raise."""
    short = short_label(label)
    for prefix, trials in NO_CALIBRATION.items():
        if short.startswith(prefix) and ("*" in trials or task in trials):
            return True

    try:
        path = os.path.join(session_dir(label), NO_CALIBRATION_FILE)
    except FileNotFoundError:
        return False
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        declared = {ln.strip() for ln in fh
                    if ln.strip() and not ln.lstrip().startswith("#")}
    return "*" in declared or task in declared


def calibration_trim(label, task):
    """Return (trimming_start_seconds, applied, detail)."""
    if no_calibration(label, task):
        return 0.0, False, "declared: no usable calibration raise on this trial"

    time, cols, _ = load_mot(label, task)
    if "sh_elev_r" not in cols or "sh_elev_l" not in cols:
        return 0.0, False, "no sh_elev columns"

    elev = np.maximum(cols["sh_elev_r"], cols["sh_elev_l"])

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(elev, prominence=PEAK_PROMINENCE_DEG)
    qualifying = [int(p) for p in peaks if elev[p] >= PEAK_MIN_DEG]

    if qualifying:
        k = qualifying[0]
    else:
        # no peak clears the bar; fall back to the global max
        k = int(np.argmax(elev))

    peak = float(elev[k])
    if peak < PEAK_MIN_DEG:
        return 0.0, False, f"peak elevation {peak:.1f} deg < {PEAK_MIN_DEG:.0f}, no raise"

    thr = DROP_FRACTION * peak
    j = k
    while j < len(elev) - 1 and elev[j] > thr:
        j += 1

    if time[j] > MAX_TRIM_FRACTION * time[-1]:
        return (0.0, False,
                f"peak {peak:.1f} deg at {time[k]:.3f}s but the drop-after-"
                f"peak is at {time[j]:.3f}s of {time[-1]:.3f}s "
                f"({time[j] / time[-1]:.0%} of the trial) -- implausible as "
                f"a calibration raise, no trim applied")

    return (float(time[j]), True,
            f"peak {peak:.1f} deg at {time[k]:.3f}s, dropped below "
            f"{thr:.1f} deg at {time[j]:.3f}s")


def frame_of(t, rate=SAMPLE_RATE):
    """Absolute 1-based .trc Frame# for an absolute time in seconds."""
    return int(round(float(t) * rate)) + 1


def exc_detail(exc):
    """"TypeError: msg  [at file.py:123 in func]" for a caught exception."""
    import traceback
    tb = traceback.extract_tb(exc.__traceback__)
    if not tb:
        return f"{type(exc).__name__}: {exc}"
    last = tb[-1]
    return (f"{type(exc).__name__}: {exc}  [at "
            f"{os.path.basename(last.filename)}:{last.lineno} in {last.name}]")


def event(task, label, name, t, source, definition, cycle=None, side=None,
          anchor_leg=None):
    """One row of the output table."""
    return {
        "task": task,
        "event": label,
        "name": name,
        "cycle": cycle,
        "anchor_leg": anchor_leg,
        "side": side,
        "frame": frame_of(t),
        "time_s": round(float(t), 4),
        "source": source,
        "definition": definition,
    }


EVENT_FIELDS = ["task", "event", "name", "cycle", "anchor_leg", "side",
                "frame", "time_s", "source", "definition"]


def check_order(events):
    """Verify no S(N+1) lands before its S(N)."""
    def snum(e):
        s = e["event"].lstrip("S")
        return int(s) if s.isdigit() else -1

    violations = []
    tasks = sorted({e["task"] for e in events})

    for task in tasks:
        rows = [e for e in events if e["task"] == task]

        # anchor legs overlap in time by design; check them separately
        by_cycle = {}
        for e in rows:
            key = (e.get("anchor_leg"), e["cycle"])
            by_cycle.setdefault(key, []).append(e)

        # within-cycle
        for (anchor, cycle), group in by_cycle.items():
            if cycle is None:
                continue
            ordered = sorted(group, key=snum)
            for a, b in zip(ordered, ordered[1:]):
                if b["time_s"] < a["time_s"]:
                    violations.append({
                        "task": task, "kind": "within-cycle",
                        "cycle": cycle, "anchor_leg": anchor,
                        "detail": f"{b['event']} at {b['time_s']}s precedes "
                                  f"{a['event']} at {a['time_s']}s"})

        # across-cycle
        anchors = {a for a, c in by_cycle if c is not None}
        for anchor in sorted(anchors, key=lambda x: (x is None, x)):
            numbered = sorted(c for a, c in by_cycle
                              if c is not None and a == anchor)
            for c1, c2 in zip(numbered, numbered[1:]):
                end1 = max(e["time_s"] for e in by_cycle[(anchor, c1)])
                start2 = min(e["time_s"] for e in by_cycle[(anchor, c2)])
                if start2 < end1:
                    violations.append({
                        "task": task, "kind": "across-cycle", "cycle": c2,
                        "anchor_leg": anchor,
                        "detail": f"cycle {c2} starts at {start2}s, before "
                                  f"cycle {c1} ends at {end1}s"})

        # unnumbered
        loose = sorted(by_cycle.get((None, None), []), key=snum)
        for a, b in zip(loose, loose[1:]):
            if b["time_s"] < a["time_s"]:
                violations.append({
                    "task": task, "kind": "unnumbered", "cycle": None,
                    "detail": f"{b['event']} at {b['time_s']}s precedes "
                              f"{a['event']} at {a['time_s']}s"})

    return violations


INTERVAL_TASKS = {"gait", "gait-start", "gait-head-turn", "gait-tandem",
                  "gait-bckw", "gait-bckw-3s", "gait-pivot", "tug"}
INTERVAL_RATIO = 2.5        # controls top out at 1.91x, the DCM misstep was 3.20x
INTERVAL_FLOOR_S = 0.15     # s. keeps a big ratio on a tiny interval quiet
INTERVAL_MIN_SAMPLES = 3    # need 3+ of an event pair before its median means anything


def flag_interval_outliers(events):
    """Stride intervals far from the median of their own event pair."""
    import collections
    series = collections.defaultdict(list)
    for e in events:
        if e["task"] in INTERVAL_TASKS and e.get("cycle"):
            series[(e["task"], e.get("anchor_leg") or "")].append(
                (e["time_s"], e["event"], e.get("cycle")))

    out = []
    for (task, anchor), evs in sorted(series.items()):
        evs.sort()
        gaps = collections.defaultdict(list)
        for (t0, e0, c0), (t1, e1, _) in zip(evs, evs[1:]):
            gaps[(e0, e1)].append((t1 - t0, t0, e0, e1, c0))
        for pair, vals in gaps.items():
            if len(vals) < INTERVAL_MIN_SAMPLES:
                continue
            median = float(np.median([v[0] for v in vals]))
            if median <= 0:
                continue
            for gap, t0, e0, e1, cycle in vals:
                if abs(gap - median) < INTERVAL_FLOOR_S:
                    continue
                if gap / median < INTERVAL_RATIO:
                    continue
                out.append({
                    "task": task, "anchor_leg": anchor, "cycle": cycle,
                    "from_event": e0, "to_event": e1,
                    "at_s": round(float(t0), 4),
                    "interval_s": round(float(gap), 4),
                    "median_s": round(median, 4),
                    "ratio": round(gap / median, 2),
                    "note": "check the video -- a misstep and replant here "
                            "would not appear as a separate heel strike",
                })
    out.sort(key=lambda d: -d["ratio"])
    return out


def sort_events(events):
    """Chronological within a task, which is how a rater reads them."""
    return sorted(events, key=lambda e: (e["task"], e["time_s"]))


def walking_signals(marker_dict):
    """Return {r_calc_rel_x, l_calc_rel_x, r_toe_rel_x, l_toe_rel_x}."""
    m = marker_dict["markers"]

    r_calc_rel = m["r_calc_study"] - m["r.PSIS_study"]
    r_toe_rel = m["r_toe_study"] - m["r.PSIS_study"]
    l_calc_rel = m["L_calc_study"] - m["L.PSIS_study"]
    l_toe_rel = m["L_toe_study"] - m["L.PSIS_study"]

    mid_psis = (m["r.PSIS_study"] + m["L.PSIS_study"]) / 2
    mid_asis = (m["r.ASIS_study"] + m["L.ASIS_study"]) / 2
    mid_dir = mid_asis - mid_psis
    mid_dir_floor = np.copy(mid_dir)
    mid_dir_floor[:, 1] = 0
    mid_dir_floor = mid_dir_floor / np.linalg.norm(mid_dir_floor, axis=1,
                                                   keepdims=True)

    return {
        "r_calc_rel_x": np.einsum("ij,ij->i", mid_dir_floor, r_calc_rel),
        "l_calc_rel_x": np.einsum("ij,ij->i", mid_dir_floor, l_calc_rel),
        "r_toe_rel_x": np.einsum("ij,ij->i", mid_dir_floor, r_toe_rel),
        "l_toe_rel_x": np.einsum("ij,ij->i", mid_dir_floor, l_toe_rel),
    }


ORDER_FORWARD = {"rHS": "lTO", "lTO": "lHS", "lHS": "rTO", "rTO": "rHS"}
ORDER_BACKWARD = {"rTO": "lHS", "lHS": "lTO", "lTO": "rHS", "rHS": "rTO"}

# OpenCap's ladder, strictest first, descending only when the gate fails
PROMINENCES_OPENCAP = [0.3, 0.25, 0.2]

LOW_PROMINENCE_FIRST = True


def prominence_ladder():
    return (list(reversed(PROMINENCES_OPENCAP)) if LOW_PROMINENCE_FIRST
            else list(PROMINENCES_OPENCAP))


PROMINENCES = PROMINENCES_OPENCAP


REPLANT_WINDOW_HS_S = 0.70  # s. smallest genuine same-side HS gap on controls is 0.983
REPLANT_WINDOW_TO_S = 0.35  # s. tight: NEW_GX tug has a glitch TO 0.483s before the real one


def cluster_events(idx, time, max_gap_s, keep):
    """Collapse a replanted foot's repeated peaks into one event."""
    idx = np.asarray(idx, dtype=int)
    if len(idx) < 2:
        return idx
    # Cluster boundaries are the gaps long enough to be real steps.
    breaks = np.flatnonzero(np.diff(time[idx]) >= max_gap_s) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(idx)]))
    return np.array([idx[s] if keep == 'first' else idx[e - 1]
                     for s, e in zip(starts, ends)], dtype=int)


def detect_peaks(signals, time, prominence):
    """The class's detect_gait_peaks: HS from calc peaks, TO from toe troughs."""
    from scipy.signal import find_peaks
    rHS, _ = find_peaks(signals["r_calc_rel_x"], prominence=prominence)
    lHS, _ = find_peaks(signals["l_calc_rel_x"], prominence=prominence)
    rTO, _ = find_peaks(-signals["r_toe_rel_x"], prominence=prominence)
    lTO, _ = find_peaks(-signals["l_toe_rel_x"], prominence=prominence)
    hw, tw = REPLANT_WINDOW_HS_S, REPLANT_WINDOW_TO_S
    return {"rHS": cluster_events(rHS, time, hw, "last"),
            "lHS": cluster_events(lHS, time, hw, "last"),
            "rTO": cluster_events(rTO, time, tw, "first"),
            "lTO": cluster_events(lTO, time, tw, "first")}


def order_ok(peaks, expected_order):
    """Replica of the class's detect_correct_order, with a supplied chain."""
    vectors = {k: np.array(v) for k, v in peaks.items()}
    non_empty = {k: v for k, v in vectors.items() if len(v) > 0}
    if not non_empty:
        return True
    v1 = min(non_empty, key=lambda k: non_empty[k][0])
    while any(len(v) > 0 for v in vectors.values()):
        vectors[v1] = np.delete(vectors[v1], 0)
        non_empty = {k: v for k, v in vectors.items() if len(v) > 0}
        if not non_empty:
            break
        v2 = min(non_empty, key=lambda k: non_empty[k][0])
        if v2 != expected_order[v1]:
            return False
        v1 = v2
    return True


def load_trc(label, task):
    """Read a trial's .trc through OpenCap's own reader."""
    from utilsTRC import trc_2_dict
    path = os.path.join(session_dir(label), "MarkerData",
                        trial_name(label, task) + ".trc")
    return trc_2_dict(path)


def _window(marker_dict, t_start, t_end):
    time = marker_dict["time"]
    i0 = np.where(np.round(time - t_start, 6) <= 0)[0][-1] if t_start > 0 else 0
    if t_end > 0:
        i1 = np.where(np.round(time, 6) <= np.round(time[-1] - t_end, 6))[0][-1] + 1
    else:
        i1 = len(time)
    return ({"time": time[i0:i1],
             "markers": {k: v[i0:i1, :] for k, v in marker_dict["markers"].items()}},
            time[i0:i1])


def search_trim(marker_dict, expected_order, base_start=0.0, base_end=0.0,
                max_extra_start=5.0, max_end=3.0, step=0.25,
                min_heel_strikes=2):
    """Find the smallest (trimming_start, trimming_end) clearing the gate."""
    n_start = int(round(max_extra_start / step)) + 1
    n_end = int(round(max_end / step)) + 1
    combos = [(a * step, b * step) for a in range(n_start) for b in range(n_end)]
    combos.sort(key=lambda ab: (ab[0] + ab[1], ab[0]))

    for extra, extra_end in combos:
        start = base_start + extra
        end = base_end + extra_end
        win, time = _window(marker_dict, start, end)
        if len(time) < 60:
            continue
        sig = walking_signals(win)
        for prom in prominence_ladder():
            peaks = detect_peaks(sig, time, prom)
            if not order_ok(peaks, expected_order):
                continue
            if max(len(peaks["rHS"]), len(peaks["lHS"])) < min_heel_strikes:
                continue
            detail = (f"trimming_start {start:.3f}s "
                      f"(base {base_start:.3f}s + {extra:.2f}s), "
                      f"trimming_end {end:.3f}s "
                      f"(base {base_end:.3f}s + {extra_end:.2f}s), "
                      f"prominence {prom}")
            return start, end, prom, detail

    raise ValueError(
        f"no trim within +{max_extra_start:.1f}s start / {max_end:.1f}s end "
        f"clears the event order gate")


def dense_baseline(pos, rate=SAMPLE_RATE, speed_percentile=30, bin_m=0.05,
                   min_frames=30):
    """Neutral posture of a 3D marker track: the most-occupied still place."""
    d = np.diff(pos, axis=0, prepend=pos[:1])
    speed = np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2) * rate
    still = speed <= np.percentile(speed, speed_percentile)
    P = pos[still] if still.sum() >= min_frames else pos

    kx = np.round(P[:, 0] / bin_m).astype(int)
    kz = np.round(P[:, 2] / bin_m).astype(int)
    keys, counts = np.unique(np.stack([kx, kz], axis=1), axis=0,
                             return_counts=True)
    bx, bz = keys[int(np.argmax(counts))]

    sel = (np.abs(kx - bx) <= 1) & (np.abs(kz - bz) <= 1)
    if sel.sum() < min_frames:
        sel = np.ones(len(P), dtype=bool)
    return np.median(P[sel], axis=0)


def ground_frame(marker_dict, quiet_mask=None):
    """Return (forward, right) unit vectors, FIXED for the whole trial."""
    m = marker_dict["markers"]
    mid_psis = (m["r.PSIS_study"] + m["L.PSIS_study"]) / 2
    mid_asis = (m["r.ASIS_study"] + m["L.ASIS_study"]) / 2
    heading = mid_asis - mid_psis
    heading[:, 1] = 0
    heading = heading / np.linalg.norm(heading, axis=1, keepdims=True)

    if quiet_mask is not None and quiet_mask.sum() > 30:
        fwd = heading[quiet_mask].mean(axis=0)
    else:
        fwd = heading.mean(axis=0)
    fwd = fwd / np.linalg.norm(fwd)

    right = -np.cross([0.0, 1.0, 0.0], fwd)
    right = right / np.linalg.norm(right)
    return fwd, right


def steady_runs(time, x, window_s=1.0, sd_threshold=None, min_len_s=2.0,
                rate=SAMPLE_RATE):
    """Intervals where a 1D signal holds still."""
    import pandas as pd
    n = max(int(window_s * rate), 3)
    sd = pd.Series(x).rolling(n, center=True).std().values
    if sd_threshold is None:
        sd_threshold = max(np.nanpercentile(sd, 35), 1e-9)

    low = sd < sd_threshold
    runs = []
    i = 0
    while i < len(low):
        if low[i]:
            j = i
            while j + 1 < len(low) and low[j + 1]:
                j += 1
            if time[j] - time[i] >= min_len_s:
                runs.append((float(time[i]), float(time[j]),
                             float(np.nanmean(x[i:j + 1])), i, j))
            i = j + 1
        else:
            i += 1
    return runs, sd, sd_threshold


def movement_onset(sig, time, peak_idx, search_from=0, vel_fraction=0.05,
                   vel_floor=1e-4):
    """First movement before a peak, by walking back from the fastest change."""
    if peak_idx <= search_from:
        return int(search_from)

    vel = np.gradient(sig, time)
    seg = vel[search_from:peak_idx]
    if len(seg) == 0:
        return int(search_from)

    k = search_from + int(np.argmax(seg))
    thresh = max(vel_fraction * abs(float(vel[k])), vel_floor)

    j = k
    while j > search_from and vel[j] > thresh:
        j -= 1
    return int(j)


def excursion_bounds(sig, peak_idx, level, band):
    """(onset, return) indices bracketing an excursion peak."""
    thresh = level + band
    before = np.where(sig[:peak_idx] <= thresh)[0]
    after = np.where(sig[peak_idx:] <= thresh)[0]
    onset = int(before[-1]) if len(before) else None
    ret = int(peak_idx + after[0]) if len(after) else None
    return onset, ret


TURN_VEL_THRESHOLD = 30.0    # deg/s
TURN_MERGE_GAP = 0.30        # s
TURN_MIN_SWEEP = 90.0        # deg


def detect_turns(time, pelvis_rotation,
                 vel_threshold=TURN_VEL_THRESHOLD,
                 merge_gap=TURN_MERGE_GAP,
                 min_sweep=TURN_MIN_SWEEP):
    """Return [(start_time, end_time, signed_sweep_deg)] for each turn."""
    omega = np.gradient(pelvis_rotation, time)
    active = np.abs(omega) > vel_threshold

    # Contiguous runs above threshold.
    runs = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j + 1 < len(active) and active[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1

    if not runs:
        return []

    def sweep(a, b):
        return float(pelvis_rotation[b] - pelvis_rotation[a])

    # Join neighbouring runs only when close in time AND rotating the same way.
    merged = [list(runs[0])]
    for a, b in runs[1:]:
        pa, pb = merged[-1]
        close = (time[a] - time[pb]) < merge_gap
        same_sign = np.sign(sweep(a, b)) == np.sign(sweep(pa, pb))
        if close and same_sign:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    turns = []
    for a, b in merged:
        s = sweep(a, b)
        if abs(s) >= min_sweep:
            turns.append((float(time[a]), float(time[b]), s))
    return turns
