"""
Shared plumbing for the per-task segmentation scripts.
======================================================

Holds everything that is NOT segmentation logic: session/trial resolution,
the calibration trim, .mot reading, absolute frame/time bookkeeping, and the
event record format.

The segmentation logic itself lives in the per-task scripts and comes from
the OpenCap team's ActivityAnalyses modules, imported directly from the
opencap-processing checkout so the code that runs is byte-identical to
theirs. Nothing in this file reimplements a gait or sit-to-stand event.

FRAME AND TIME CONVENTION
    .trc and .mot share an identical time base and row count in every trial
    (verified: PR gait is 681 rows in both, 0.0 -> 11.3333333 s), so one
    (frame, second) pair is valid for both files. Frame numbers are the .trc
    file's own 1-based Frame# column, i.e. frame = round(t * rate) + 1.

    The OpenCap classes return indices into their internally TRIMMED arrays.
    Those indices are not comparable across trials with different trims, so
    every event here is recorded from the returned TIME and converted back to
    an absolute frame. Times coming out of the classes are already absolute
    because the trimmed time vector keeps real clock values.
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))


def _find_opencap():
    """Locate the opencap-processing checkout.

    Searched rather than hardcoded relative to this file, so the scripts can
    be dropped into one folder next to the checkout instead of only working
    from Data/Analysis/segmentation/. OPENCAP_PROCESSING_DIR overrides
    everything, for a checkout that lives somewhere else entirely.
    """
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

# opencap-processing's utils.py runs API_TOKEN = get_token() at import time,
# and utilsKinematics imports utils, so importing gait_analysis or
# sts_analysis triggers an interactive login to app.opencap.ai. get_token()
# checks the API_TOKEN environment variable before prompting, so setting it
# here makes the import work offline. The value is never used: it only
# reaches utils.py's download helpers, and every trial we read is already on
# disk. Set a real token in the environment beforehand if you ever do want
# the download paths to work.
os.environ.setdefault("API_TOKEN", "offline-local-data-only")

SESSION_PREFIXES = ["MG_Trial", "OLD_MG_Trial", "NEW_GX_Trial",
                    "OLD_GX_Trial", "PR_Trial"]

# Where to look for session folders. BASE (Data/) holds the controls. Extra
# roots come from the DCM_SESSION_ROOTS environment variable, colon-separated,
# so a session set that lives outside this tree can be segmented without its
# location being written down in the repo. session_dir also accepts a path to
# a session folder directly, which needs no configuration at all.
SESSION_ROOTS = [BASE] + [p for p in os.environ.get(
    "DCM_SESSION_ROOTS", "").split(os.pathsep) if p.strip()]

# The tasks the segmentation modules implement, in the order
# run_segmentation.MODULES runs them. Kept here rather than there so that
# manifest.py can read the list without importing the OpenCap classes.
TASKS_IN_SCOPE = [
    "standing-ec", "rtt", "semi-tandem", "tandem", "sls-ec",
    "5tsts", "march",
    "gait", "gait-start", "gait-head-turn", "gait-tandem",
    "gait-bckw", "gait-bckw-3s", "gait-pivot",
    "frt", "y-balance", "arms", "tug",
]

# Recorded tasks we deliberately do not segment, per your instructions. Listed
# so manifest.py can report "present but out of scope" separately from
# "present and unrecognised", which is the group that needs an alias.
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

# Per-session alias tables, matched by label prefix. Replaces the single
# hardcoded `if label.startswith("OLD_GX")` branch: any session that named its
# trials differently gets an entry here, and manifest.py reports which trials
# are still unmatched so the entry can be written.
SESSION_TRIAL_ALIASES = {
    "OLD_GX": TRIAL_ALIASES,
}

SAMPLE_RATE = 60.0


def session_dir(label):
    """Resolve a session label or folder path to its session folder.

    A label ('PR', 'OLD_GX') is matched as a prefix against the folder names
    in every root in SESSION_ROOTS. A path to a session folder is returned
    as-is, identified by its MarkerData subfolder, so a session outside
    SESSION_ROOTS can be segmented by passing its path.
    """
    if os.path.isdir(os.path.join(label, "MarkerData")):
        return os.path.abspath(label)

    # The controls are <LABEL>_Trial_OpenCapData_<uuid>; the DCM sessions are
    # DCM_###_OpenCapData_<uuid>, with no _Trial infix. Both are tried, so
    # 'PR' and 'DCM_001' both resolve without the caller knowing which.
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
    """Filename-safe short name, for a label or a session folder path.

    Output files are named events_<label>.csv, so a label that is a path has
    to be reduced to its folder name first.
    """
    if os.sep not in label:
        return label
    base = os.path.basename(os.path.abspath(label))
    return base.split("_OpenCapData")[0] or base


def trial_name(label, task):
    """Map a canonical task name to the on-disk trial name for a session."""
    for prefix, aliases in SESSION_TRIAL_ALIASES.items():
        if label.startswith(prefix):
            return aliases.get(task, task)
    return task


def trial_exists(label, task):
    sd = session_dir(label)
    trc = os.path.join(sd, "MarkerData", trial_name(label, task) + ".trc")
    return os.path.exists(trc)


# ---------------------------------------------------------------------------
# .mot reading
# ---------------------------------------------------------------------------

def load_mot(label, task):
    """Read a trial's joint-angle .mot as (time, {column: array}).

    Prefers <trial>_shoulder.mot where it exists (OLD_GX, OLD_MG), which is
    the harmonized 40-column shoulder-model file. For sessions processed
    natively on the shoulder model the plain <trial>.mot is already correct.

    Note this is deliberately independent of utilsKinematics, which hardcodes
    the plain <trial>.mot and would silently load the older 34-column
    arm_flex layout for the two OLD sessions.
    """
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


# ---------------------------------------------------------------------------
# Calibration trim
# ---------------------------------------------------------------------------
# Segmentation markers document, header: "Shoulder elevation peak visible in
# bottom left corner of each task. Segmentation markers should only be noted
# AFTER shoulder elevation drop-after-peak time. (exception for OLD_GX, no
# calibration arm raise done in that trial)"
#
# Two known exceptions, both user-confirmed: OLD_GX performed no calibration
# raise on any trial, and OLD_MG omitted it on standing-ec, rtt and
# semi-tandem. A trial whose peak elevation never reaches PEAK_MIN_DEG is
# treated as having no raise, which is what catches the OLD_MG cases.

PEAK_MIN_DEG = 90.0      # below this, the elevation peak is the task, not a raise
DROP_FRACTION = 0.25     # "drop-after-peak" = fall to this fraction of the peak
PEAK_PROMINENCE_DEG = 30.0   # to count as a distinct elevation peak

# Largest share of a trial the calibration trim may consume before it is
# treated as a misfire rather than a raise. Controls top out at 54.9%.
MAX_TRIM_FRACTION = 0.75

# Trials with no usable calibration arm raise, keyed by session label prefix.
# "*" means the whole session. Anything listed gets trim 0.0 instead of a
# detected peak.
#
# This has to be declared rather than detected, because a detector looking
# for a raise that is not there does not return nothing -- it returns
# whatever else crossed the threshold. DCM_001 skipped the raise on some
# tasks, and on its march trial an arm swing crossing 90 deg was read as the
# raise and trimmed 10.4 s off the front of a 29 s trial. The opposite case
# needs declaring too: DCM_002 began with the arms already overhead and
# lowered them, so there is no rise-then-fall for find_peaks to lock onto,
# the fallback lands at t=0 and the walk forward to the drop-after-peak eats
# the start of the task.
#
# Add entries as {"<session prefix>": {"<task>", ...}}. Per-trial, because
# the same subject may have done the raise on some tasks and not others.
NO_CALIBRATION = {
    "OLD_GX": {"*"},    # performed no calibration raise on any trial
}

# Per-session declaration file, read from the session folder itself.
#
# The patient labels are identifying and must not be committed to this
# repository, so a patient override cannot be a key in the dict above. The
# declaration travels with the session folder instead, which is already
# protected. One task name per line, or a single "*" for the whole session;
# blank lines and #-comments ignored.
NO_CALIBRATION_FILE = "no_calibration.txt"


def no_calibration(label, task):
    """True if this trial is declared to have no usable calibration raise.

    Two places to declare it:

        NO_CALIBRATION                   keyed by label, for the controls
        <session>/no_calibration.txt     one task per line, or "*"

    Matched on the SHORT label. run_all.py hands each step the session
    PATH rather than the label, so a raw prefix test against `label` sees
    "/.../<session>" and never fires -- a label-keyed override would be
    silently dead in the one way the pipeline is actually driven.
    """
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
    """Return (trimming_start_seconds, applied, detail).

    trimming_start is the time of the drop-after-peak, suitable for passing
    straight into the OpenCap classes' trimming_start argument.
    """
    if no_calibration(label, task):
        return 0.0, False, "declared: no usable calibration raise on this trial"

    time, cols, _ = load_mot(label, task)
    if "sh_elev_r" not in cols or "sh_elev_l" not in cols:
        return 0.0, False, "no sh_elev columns"

    elev = np.maximum(cols["sh_elev_r"], cols["sh_elev_l"])

    # The calibration raise is the FIRST qualifying elevation peak, not the
    # largest. Taking the global maximum is safe only where the calibration
    # raise is the trial's one overhead movement, which is true of every task
    # except arms -- and there it fails badly. PR's arms trial has six
    # elevation peaks; the calibration raise is at 4.15 s but the largest
    # elevation is 139.1 deg at 9.07 s, which is the task's parasagittal
    # raise. Trimming to the maximum therefore discarded the first task
    # raise and shifted every plane label by one, so scapular/frontal/
    # backward were reported as parasagittal/scapular/frontal.
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(elev, prominence=PEAK_PROMINENCE_DEG)
    qualifying = [int(p) for p in peaks if elev[p] >= PEAK_MIN_DEG]

    if qualifying:
        k = qualifying[0]
    else:
        # No prominent peak clears the bar; fall back to the global maximum
        # so a single broad raise is still caught.
        k = int(np.argmax(elev))

    peak = float(elev[k])
    if peak < PEAK_MIN_DEG:
        return 0.0, False, f"peak elevation {peak:.1f} deg < {PEAK_MIN_DEG:.0f}, no raise"

    thr = DROP_FRACTION * peak
    j = k
    while j < len(elev) - 1 and elev[j] > thr:
        j += 1

    # A trim that swallows the trial is never right, and it fails a long way
    # from its cause. DCM_002 began its 5tsts with the arms already overhead,
    # so there is no rise-then-fall for find_peaks to lock onto, the fallback
    # lands the "peak" at the very start, and the walk forward never finds a
    # drop below threshold -- j reaches the last frame, trimming_start
    # becomes the whole 21.98 s trial, and segmentation got a single sample.
    # That surfaced as "IndexError: index 1 is out of bounds for axis 0 with
    # size 1" at dt = timeVec[1] - timeVec[0], inside segment_sts.
    #
    # Across 69 control trials with a genuine raise, the drop-after-peak
    # never lands later than 54.9% of the trial (PR gait-start), so a cap at
    # MAX_TRIM_FRACTION rejects the runaway case without touching any of
    # them. A rejected trim means no trim, which is the same thing a trial
    # with no raise gets.
    if time[j] > MAX_TRIM_FRACTION * time[-1]:
        return (0.0, False,
                f"peak {peak:.1f} deg at {time[k]:.3f}s but the drop-after-"
                f"peak is at {time[j]:.3f}s of {time[-1]:.3f}s "
                f"({time[j] / time[-1]:.0%} of the trial) -- implausible as "
                f"a calibration raise, no trim applied")

    return (float(time[j]), True,
            f"peak {peak:.1f} deg at {time[k]:.3f}s, dropped below "
            f"{thr:.1f} deg at {time[j]:.3f}s")


# ---------------------------------------------------------------------------
# Event records
# ---------------------------------------------------------------------------

def frame_of(t, rate=SAMPLE_RATE):
    """Absolute 1-based .trc Frame# for an absolute time in seconds."""
    return int(round(float(t) * rate)) + 1


def exc_detail(exc):
    """"TypeError: msg  [at file.py:123 in func]" for a caught exception.

    A bare type-and-message is not enough to act on when a task fails on
    data that cannot be opened here. DCM_002's 5tsts reported only
    "IndexError: index 1 is out of bounds for axis 0 with size 1", which is
    true of a dozen places in the sit-to-stand chain and pinned down none of
    them. The originating line makes it one place.
    """
    import traceback
    tb = traceback.extract_tb(exc.__traceback__)
    if not tb:
        return f"{type(exc).__name__}: {exc}"
    last = tb[-1]
    return (f"{type(exc).__name__}: {exc}  [at "
            f"{os.path.basename(last.filename)}:{last.lineno} in {last.name}]")


def event(task, label, name, t, source, definition, cycle=None, side=None,
          anchor_leg=None):
    """One row of the output table.

    anchor_leg records which leg's heel strikes bracketed the cycle. It is
    only meaningful where a trial is segmented from both legs, so that the
    two overlapping cycle series stay separable.
    """
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
    """Verify no S(N+1) lands before its S(N).

    Three checks per task:
      within-cycle   inside one cycle, S-numbers must ascend in time
      across-cycle   cycle N's last event must precede cycle N+1's first
      unnumbered     events with no cycle (turns, the rise, the sit) must
                     have their S-numbers ascend in time

    Returns a list of violation dicts. An empty list means every task's
    labels run in the same order as the clock.
    """
    def snum(e):
        s = e["event"].lstrip("S")
        return int(s) if s.isdigit() else -1

    violations = []
    tasks = sorted({e["task"] for e in events})

    for task in tasks:
        rows = [e for e in events if e["task"] == task]

        # Cycles from different anchor legs overlap in time by design, so
        # they are checked as independent series.
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


# ---------------------------------------------------------------------------
# Stride-interval outliers
# ---------------------------------------------------------------------------
# A replant that the pelvis outruns is invisible to the heel-strike
# definition. Heel strike is a peak in the heel's fore-aft position RELATIVE
# TO THE PELVIS, and during a replant the pelvis keeps advancing faster than
# the foot moves forward, so the second contact lands on the falling limb of
# that signal instead of making a new peak. No prominence setting recovers a
# peak that is not there, and the replant collapse can only choose between
# peaks that exist.
#
# Measured on DCM_001's gait-head-turn: contact at 6.25 s with the heel
# 5.6 mm off the floor, foot lifted to 77 mm by 6.65 s, descent flattening
# at about 6.87 s -- a committed plant the event list puts at 6.267 s. No
# peak exists at 6.87 s at any prominence down to 0.05.
#
# What IS visible is the hole it leaves in the timing. The interval from the
# ipsilateral heel strike to the next contralateral toe-off is 0.25 s in
# every clean cycle of that trial and 0.80 s in the bad one. So rather than
# move the event, the trial is reported: every stride interval is compared
# with the median of the same event pair in the same task, and the outliers
# are named for a rater to check against the video.
#
# Report only. Nothing here changes an event.
#
# Scoped to the gait family, whose cycles share one S1-S4 stride structure.
# 5tsts and march were measured too and are not included: the 5tsts lean
# duration genuinely varies between repetitions (MG 0.78 s against a 0.17 s
# median, which is real), and march medians are small enough that a 41 ms
# difference reads as 2x.
#
# RATIO 2.5 -- across the five controls the largest gait-family ratio is
# 1.91 (NEW_GX gait-tandem), and the DCM_001 case above is 3.20, so the
# threshold sits between them with margin either side and flags 0 of the
# control intervals. FLOOR keeps a large ratio on a tiny interval quiet.

INTERVAL_TASKS = {"gait", "gait-start", "gait-head-turn", "gait-tandem",
                  "gait-bckw", "gait-bckw-3s", "gait-pivot", "tug"}
INTERVAL_RATIO = 2.5
INTERVAL_FLOOR_S = 0.15
INTERVAL_MIN_SAMPLES = 3


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


# ---------------------------------------------------------------------------
# Walking signals
# ---------------------------------------------------------------------------
# The four projections that OpenCap's segment_walking builds before it hunts
# for peaks. Copied from that method so the numbers here are the same numbers
# it works on. Used only where the class cannot be called for its own
# assembled output (the single-stride case in seg_gait) and for reporting.
#
# Document notes, verbatim:
#   Heading      = (r.ASIS + L.ASIS)/2 - (r.PSIS + L.PSIS)/2
#   r_calc_rel_x = (r_calc_study - r.PSIS_study) . heading
#   l_calc_rel_x = (L_calc_study - L.PSIS_study) . heading
#   r_toe_rel_x  = (r_toe_study  - r.PSIS_study) . heading
#   l_toe_rel_x  = (L_toe_study  - L.PSIS_study) . heading

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


# ---------------------------------------------------------------------------
# Order gate and trim search
# ---------------------------------------------------------------------------
# segment_walking raises ValueError when its event order gate fails at all
# three prominences, and the message says: "Consider trimming your trial
# using the trimming_start and trimming_end options." Finding those two
# numbers by hand per trial does not scale, so they are searched for here.
#
# The search runs on the same projections and the same prominence ladder the
# class uses, and applies a replica of the class's own gate, so a window that
# passes here passes there. The class is then constructed once with the
# winning trim and it -- not this code -- produces the events.
#
# Preference is the smallest total trim that both clears the gate and leaves
# a complete cycle, which keeps the most strides. On PR's gait-tandem that is
# 1.5 s off the end (4 right-led cycles); trimming 2.7 s also passes but
# drops to 3 left-led cycles because it removes the final heel strike.

ORDER_FORWARD = {"rHS": "lTO", "lTO": "lHS", "lHS": "rTO", "rTO": "rHS"}
ORDER_BACKWARD = {"rTO": "lHS", "lHS": "lTO", "lTO": "rHS", "rHS": "rTO"}

# OpenCap's ladder, walked strictest first and descending only when the order
# gate fails. See seg_walk_core for why the direction matters.
PROMINENCES_OPENCAP = [0.3, 0.25, 0.2]

# True  -> take the LOWEST prominence that still passes the gate, which
#          recovers genuine low-prominence events such as the compressed
#          first step out of a pivot turn.
# False -> OpenCap's behaviour, the strictest passing prominence.
# The trim search and the gait classes must agree on this, so both read it
# from here.
LOW_PROMINENCE_FIRST = True


def prominence_ladder():
    return (list(reversed(PROMINENCES_OPENCAP)) if LOW_PROMINENCE_FIRST
            else list(PROMINENCES_OPENCAP))


PROMINENCES = PROMINENCES_OPENCAP


# How close two detections of the same event have to be to count as one
# replanted contact rather than two separate steps.
#
# The two event types need different windows, and measuring them separately
# is what let the heel-strike window be widened at all. Across the five
# controls, eight gait tasks and all three prominences:
#
#     heel strikes  n=729  minimum same-side gap 0.983 s  (p1 1.019)
#     toe-offs      n=733  minimum same-side gap 0.483 s  (p1 0.967)
#
# So no genuine pair of heel strikes is anywhere near 0.70 s apart -- no
# control event changes at any HS window up to 0.8 s -- while the toe-off
# window has to stay tight, because NEW_GX's tug holds a glitch rTO at
# 14.500 s only 0.483 s before the real one at 14.983 s, and keep-first
# would retain the glitch if they merged.
#
# 0.70 s covers a misstep-then-replant heel strike 0.57 s apart, which is
# what DCM_001's gait-head-turn does at 6.26 -> 6.83 s.
REPLANT_WINDOW_HS_S = 0.70
REPLANT_WINDOW_TO_S = 0.35


def cluster_events(idx, time, max_gap_s, keep):
    """Collapse a replanted foot's repeated peaks into one event.

    A subject who is unsteady puts the foot down, picks it up and puts it
    down again, and each touch produces its own peak. Indices less than
    max_gap_s apart are treated as one event, and each cluster is reduced
    to a single index:

        keep='first'   foot-off      -- the moment the foot first left
        keep='last'    foot-contact  -- the committed plant

    The asymmetry is the convention: the airborne phase is bounded by the
    first lift and the last landing, so a hesitant multi-tap landing
    resolves to where the foot actually stayed.
    """
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
    """The class's detect_gait_peaks: HS from calc peaks, TO from toe troughs.

    The replant collapse is applied here, and not only in
    seg_walk_core.segment_walking, because this is the choke point every
    other gait path goes through: search_trim's order gate and the
    single-stride fallback both call it. Collapsing in only one of them
    makes the trim search and the segmenter disagree about what the peaks
    are, and the gate then trims away strides the segmenter would have
    kept -- NEW_GX tug lost the 13.93 s heel strike that way, because the
    gate still saw the duplicate lTO at 14.48 s breaking the chain and
    pushed trimming_start forward 0.75 s.
    """
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
    """Find the smallest (trimming_start, trimming_end) clearing the gate.

    Returns (trim_start, trim_end, prominence, detail) or raises ValueError.
    min_heel_strikes is the number of ipsilateral heel strikes needed for at
    least one complete cycle; set it to 1 to allow single-stride trials.

    base_start and base_end are floors, not offsets to search around: they
    are added to whatever extra trim the search picks. base_end is measured
    from the end of the trial, matching the class's trimming_end convention,
    so a walking window (t0, t1) inside a longer trial is expressed as
    base_start=t0, base_end=(trial_duration - t1).
    """
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


# ---------------------------------------------------------------------------
# Baselines, frames, plateaus and excursions
# ---------------------------------------------------------------------------
# Shared by the tasks that have no OpenCap equivalent (standing-ec, rtt,
# semi-tandem, tandem, sls-ec, frt, y-balance, arms). None of these
# reimplements a gait or sit-to-stand event; they implement the segmentation
# markers document's own plateau and excursion wording.

def dense_baseline(pos, rate=SAMPLE_RATE, speed_percentile=30, bin_m=0.05,
                   min_frames=30):
    """Neutral posture of a 3D marker track: the most-occupied still place.

    Two things this is NOT, both of which were tried and both of which fail:

    A plain median is not the neutral posture when a trial contains large
    excursions. It lands between neutral and the reach, so the neutral
    periods read as excursions in the opposite direction. On PR's frt that
    produced four spurious "backward" peaks interleaved with the three real
    reaches.

    Iteratively keeping the nearer half of the frames is not it either. Each
    pass halves the retained set, so it converges on the densest SMALL
    cluster, and in a reach task the longest stationary hold can be a reach
    rather than neutral. On PR's frt that left the median excursion at
    0.501 m of a 0.800 m maximum, i.e. the baseline sat inside a reach, and
    downstream both the rightward and leftward reaches collapsed onto one
    frame because the signal never returned inside the baseline band.

    So: keep the frames where the marker is nearly stationary, bin those
    positions on the horizontal plane, and take the median of the most
    populated bin and its neighbours. The subject visits neutral between
    every reach, so neutral is the most occupied still place even when any
    single reach hold lasts longer.
    """
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
    """Return (forward, right) unit vectors, FIXED for the whole trial.

    OpenCap's walking code projects onto a per-frame pelvis heading, which is
    right for walking because the heading is the direction of travel. It is
    wrong for a standing reach task: the subject rotates the pelvis into the
    reach, so a diagonal reach comes out looking forward. On PR's y-balance
    the pelvis heading swings 63.9 degrees from its neutral mean (p95 57.4),
    and with a per-frame frame the two second-set diagonal reaches read as
    -6.8 and -17.8 degrees, i.e. straight ahead. Against a frame fixed from
    the neutral periods they read as +55.5 and -64.5 degrees, which is what
    they are.

    'right' is signed anatomically: cross(up, heading) points to the
    subject's LEFT (verified against r.ASIS - L.ASIS on PR: dot -0.92 for
    y-balance, -0.97 for frt), so it is negated here. Positive lateral means
    the subject's right.
    """
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
    """Intervals where a 1D signal holds still.

    Returns [(start_s, end_s, mean_level, i0, i1)] for every run whose
    rolling standard deviation stays under sd_threshold for at least
    min_len_s. This is the "plateau" / "steady offset" wording in the
    document's standing-ec, rtt, semi-tandem, tandem and sls-ec rows.
    """
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
    """First movement before a peak, by walking back from the fastest change.

    A level crossing is not the first movement: the signal has already
    travelled some distance by the time it clears any threshold, so the onset
    always lands late, and how late depends on where the threshold sits. This
    instead anchors on the fastest rate of change between search_from and the
    peak, then walks back to where the signal was not yet moving -- the same
    construction OpenCap's startRisingIdx uses on the pelvis and that
    seg_sts_core uses for the trunk lean.

    The stop threshold is a fraction of the peak rate rather than an absolute
    velocity, so the same helper works on an arm raise of about 0.9 m and a
    reach excursion of about 0.7 m without retuning.

    search_from bounds the backward walk, and matters for the same reason it
    does in the sit-to-stand lean: without it the walk can run back through
    the previous repetition.
    """
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
    """(onset, return) indices bracketing an excursion peak.

    onset  = last frame at or below level+band before the peak
    return = first frame at or below level+band after the peak
    Either is None when the signal does not come back inside the band, which
    is how a reach that runs to the end of the trial is reported rather than
    silently clipped.
    """
    thresh = level + band
    before = np.where(sig[:peak_idx] <= thresh)[0]
    after = np.where(sig[peak_idx:] <= thresh)[0]
    onset = int(before[-1]) if len(before) else None
    ret = int(peak_idx + after[0]) if len(after) else None
    return onset, ret


# ---------------------------------------------------------------------------
# Turn detection (pelvis_rotation, from the .mot)
# ---------------------------------------------------------------------------
# Segmentation markers document, gait-pivot S5/S7 and tug S8/S10: "start of
# pelvis_rotation angular velocity increase / above 30 deg/s, at least 90
# degree rotation, consecutive angular-velocity bursts are joined only if
# their rotation changes have the same sign".
#
# The same-sign join is the load-bearing rule. Without it, the out-turn and
# the return turn merge into one event; with it, the gap tolerance barely
# matters (0.15, 0.30 and 0.80 s all give the same answer on these trials).

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
