"""
Timed Up and Go: tug.
=====================

The only task that needs all three methods at once: OpenCap's sts_analysis
for the rise and the sit, pelvis_rotation for the two turns, and OpenCap's
gait_analysis for the strides, run inside each straight window.

    S1   forward lean onset      sts_analysis forwardLeanIdx
    S2   start of rising         sts_analysis startRisingIdx
    S3   end of rising           sts_analysis endRisingIdx
    S4..S7  strides, outbound    gait_analysis in window 1
    S8   start of turn           pelvis_rotation
    S9   end of turn
    S4..S7  strides, return      gait_analysis in window 2
    S10  start of turn           pelvis_rotation
    S11  end of turn
    S12  sitting                 sts_analysis sittingIdx

WINDOWS
    window 1 = end of rising -> start of turn 1
    window 2 = end of turn 1 -> start of turn 2
    The sit-to-stand bounds the start of walking and the second turn bounds
    the end of it, which is why the three methods have to be composed rather
    than run independently: each one supplies the other's window edges.

FORWARD LEAN ONSET
    tug has exactly one repetition, so it is always the case that
    sts_analysis sets temp_sit_ind = 0 and searches backwards for the lean
    onset from lift-off across everything before it -- including the
    calibration arm raise and the initial sit-down. That is the first-
    repetition exposure that in 5tsts only affects cycle 1. trimming_start is
    passed from the calibration drop-after-peak time to bound it, which is
    what that parameter exists for.

BOTH ANCHOR LEGS, ONE SERIES
    Each window is segmented twice, once anchored on each leg, and the two
    results are POOLED into a single series -- not emitted side by side.
    leg='auto' picks whichever leg has the later final heel strike, which
    optimises the tail of the window and is blind to its head, so in a short
    window it drops the first stride: OLD_MG's return window holds lHS 11.58,
    rTO 11.78, rHS 12.20, lTO 12.38, lHS 12.75, rTO 12.93, rHS 13.28, auto
    picks right, and the left-anchored stride at 11.58 -- the first one after
    the turn -- is lost.

    Emitting both series instead put the same instant in the table twice
    under two S numbers, because one left heel strike is the left-anchored
    cycle's ipsilateral strike and the right-anchored cycle's contralateral
    one. So seg_gait.single_series pools the two into one set of typed
    events keyed by frame and lays one cycle series back over them, anchored
    on the leg whose first heel strike is earliest. See its docstring.

    Cycle numbers run straight through the task; the sts
    events carry no cycle number.
"""

import numpy as np

import seg_common as sc
import seg_gait
import seg_5tsts

from seg_sts_core import sts_analysis_leanfix as sts_analysis
from seg_sts_core import refine_end_rising

TASKS = ["tug"]

LOWPASS_STS = 6
EXPECTED_TURNS = 2

STRIDE_DEFS = [
    ("S4", "ips_hs", "ipsilateral heel strike", "peak in {ips}_calc_rel_x"),
    ("S5", "cont_to", "contralateral toe-off", "trough in {cont}_toe_rel_x"),
    ("S6", "cont_hs", "contralateral heel strike", "peak in {cont}_calc_rel_x"),
    ("S7", "ips_to", "ipsilateral toe-off", "trough in {ips}_toe_rel_x"),
]

TURN_DEFS = {
    "start": ("start of turn",
              "start of pelvis_rotation angular velocity increase above 30 "
              "deg/s, at least 90 degree rotation, consecutive "
              "angular-velocity bursts are joined only if their rotation "
              "changes have the same sign"),
    "end": ("end of turn",
            "pelvis_rotation angular velocity returns to baseline (~0)"),
}


def _emit_strides(task, cycles, ips_leg, first_cycle, direction):
    """Emit one pooled series' strides. Returns (events, cycle numbers used).

    No separate closing heel strike is appended: single_series gives every
    ipsilateral heel strike its own cycle, so the closing strike is already
    the next cycle's own anchor. Slots the window does not support are
    absent rather than guessed.
    """
    cont_leg = "l" if ips_leg == "r" else "r"
    events = []
    for n, c in enumerate(cycles, start=first_cycle):
        for key, field, name, definition in STRIDE_DEFS:
            if c[field] is None:
                continue
            side = ips_leg if "ips" in field else cont_leg
            events.append(sc.event(
                task, key, f"{name} ({direction})", c[field], "trc",
                definition.format(ips=ips_leg, cont=cont_leg),
                cycle=n, side=side, anchor_leg=ips_leg))
    return events, len(cycles)


def segment(label, task="tug", verbose=True):
    sd = sc.session_dir(label)
    tn = sc.trial_name(label, task)
    trim_start, trim_applied, trim_detail = sc.calibration_trim(label, task)

    # --- rise and sit -----------------------------------------------------
    # sts_analysis raises out of its own constructor when it finds no rise
    # ("No STS cycles found."), so the construction is guarded rather than
    # the cycle count tested afterwards.
    end_rising_detail = []
    try:
        sts = sts_analysis(
            sd, tn,
            lowpass_cutoff_frequency_for_coordinate_values=LOWPASS_STS,
            n_sts_cycles=-1,
            trimming_start=trim_start,
            trimming_end=0)
        sev = sts.stsEvents
        # The refined end of rising is also the left edge of the outbound
        # walking window below, so it has to be applied before the windows
        # are built.
        end_rising_detail = refine_end_rising(sts)
        n_sts = len(sev["startRisingIdx"])
        sts_error = None
    except Exception as exc:
        sev, n_sts = {}, 0
        sts_error = sc.exc_detail(exc)

    # A tug the subject only half completed still holds real events. Rather
    # than raise and discard the whole trial, each piece that cannot be
    # derived is skipped and named in missing_events, so "the patient never
    # sat back down" is reported as a result instead of a failure.
    missing = []
    events = []
    if n_sts < 1:
        missing += ["S1 forward lean onset", "S2 start of rising",
                    "S3 end of rising"]
        # Without a rise there is no bound on the start of walking, so the
        # first window opens at the calibration trim instead.
        end_rising = trim_start
    else:
        for key, time_key in (("S1", "forwardLeanTime"),
                              ("S2", "startRisingTime"),
                              ("S3", "endRisingTime")):
            name, definition = seg_5tsts.DEFS[key]
            events.append(sc.event(task, key, name, sev[time_key][0],
                                   "mot+model", definition))
        end_rising = sev["endRisingTime"][0]

    sit_times = sev.get("sittingTime", [])
    sitting = float(sit_times[-1]) if len(sit_times) else None
    if sitting is None:
        missing.append("S12 sitting")

    # --- turns ------------------------------------------------------------
    time, cols, mot_file = sc.load_mot(label, task)
    mask = time >= trim_start
    turns = sc.detect_turns(time[mask], cols["pelvis_rotation"][mask])
    if len(turns) > EXPECTED_TURNS:
        raise ValueError(f"expected {EXPECTED_TURNS} turns, detected "
                         f"{len(turns)}: "
                         + ", ".join(f"{a:.2f}-{b:.2f}s ({s:+.0f} deg)"
                                     for a, b, s in turns))

    # As in gait-pivot: N turns separate N+1 walking legs, and the leg after
    # the final turn is only walking when a turn is missing, i.e. when the
    # subject did not complete the course.
    legs, prev_end = [], end_rising
    for a, b, _ in turns:
        legs.append((prev_end, a))
        prev_end = b
    if len(turns) < EXPECTED_TURNS:
        legs.append((prev_end, float(time[-1])))
    directions = ["outbound", "return", "extra"]

    # --- strides, one straight window at a time ---------------------------
    meta_windows = []
    # One running counter for the whole task. Each window now emits a single
    # pooled series, and the windows do not overlap in time, so cycles can be
    # numbered straight through. The anchor leg is chosen per window and can
    # differ between them -- a per-leg counter would restart the numbering
    # mid-task whenever it switched.
    next_cycle = 1
    for wi, (w0, w1) in enumerate(legs, start=1):
        direction = directions[min(wi - 1, len(directions) - 1)]
        results = []
        for min_hs in (2, 1):
            results = seg_gait.cycles_in_window_both(
                label, task, w0, w1, min_heel_strikes=min_hs)
            if results:
                break
        if not results:
            try:
                seg_gait.cycles_in_window(label, task, w0, w1,
                                          min_heel_strikes=1)
                err = "no gait cycles for either anchor leg"
            except Exception as exc:
                err = sc.exc_detail(exc)
            meta_windows.append({
                "window": wi, "direction": direction,
                "span_s": [round(w0, 3), round(w1, 3)], "error": err})
            if verbose:
                print(f"    window {wi} ({direction}) {w0:.2f}-{w1:.2f}s: "
                      f"FAILED -- {err}")
            continue

        per_leg = {ips_leg: {"n_cycles": len(cy),
                             "first_anchor_s": round(float(cy[0]["ips_hs"]), 3),
                             "trim_search": detail}
                   for cy, ips_leg, detail in results}
        # The two anchor legs describe the same footfalls, so they are
        # pooled into one series; see seg_gait.single_series.
        cycles, ips_leg, pool_info = seg_gait.single_series(
            results, extra=seg_gait.window_peaks(label, task, w0, w1))
        ev, used = _emit_strides(task, cycles, ips_leg, next_cycle,
                                 direction)
        events += ev
        next_cycle += used
        if verbose:
            print(f"    window {wi} ({direction}) {w0:.2f}-{w1:.2f}s: "
                  f"pooled -> anchor {ips_leg}, {len(cycles)} cycles from "
                  f"{cycles[0]['ips_hs']:.2f}s, "
                  f"{pool_info.get('events_pooled')} distinct events")
        meta_windows.append({
            "window": wi, "direction": direction,
            "span_s": [round(w0, 3), round(w1, 3)],
            "n_cycles": len(cycles), "anchor_legs": sorted(per_leg),
            "per_leg": per_leg, "pooled": pool_info})

    # --- turn events and the sit ------------------------------------------
    # Turn i contributes the pair (S8, S9), (S10, S11), so a missing turn
    # leaves its pair unemitted.
    for i in range(EXPECTED_TURNS):
        start_key, end_key = f"S{8 + 2 * i}", f"S{9 + 2 * i}"
        if i >= len(turns):
            missing += [f"{start_key} start of turn {i + 1}",
                        f"{end_key} end of turn {i + 1}"]
            continue
        for key, which, t in ((start_key, "start", turns[i][0]),
                              (end_key, "end", turns[i][1])):
            name, definition = TURN_DEFS[which]
            events.append(sc.event(task, key, name, t, "mot", definition))

    if sitting is not None:
        name, definition = seg_5tsts.DEFS["S4"]
        events.append(sc.event(task, "S12", name, sitting, "mot+model",
                               definition))

    meta = {
        "task": task, "trial_file": tn, "mot_file": mot_file,
        "n_sts_cycles": n_sts, "sts_error": sts_error,
        "end_of_rising": end_rising_detail,
        "turns": [{"start_s": round(a, 3), "end_s": round(b, 3),
                   "sweep_deg": round(s, 1)} for a, b, s in turns],
        "n_turns_expected": EXPECTED_TURNS,
        "partial": bool(missing),
        "missing_events": missing,
        "windows": meta_windows,
        "trim_start_s": round(trim_start, 4), "trim_applied": trim_applied,
        "calibration_detail": trim_detail,
    }
    if verbose:
        rise = (f"rise {sev['startRisingTime'][0]:.2f}-{end_rising:.2f}s"
                if n_sts >= 1 else "no rise found")
        desc = " and ".join(f"{a:.2f}-{b:.2f}s ({s:+.0f} deg)"
                            for a, b, s in turns) or "none detected"
        sit = f"{sitting:.2f}s" if sitting is not None else "not found"
        print(f"  {task}: {rise}, turns {desc}, sit {sit}")
        if missing:
            print(f"    PARTIAL -- not emitted: {', '.join(missing)}")
    return events, meta


def run(label, tasks=None, verbose=True):
    tasks = tasks or TASKS
    all_events, all_meta = [], []
    for task in tasks:
        if not sc.trial_exists(label, task):
            print(f"  {task}: no trial in {label}, skipping")
            continue
        try:
            events, meta = segment(label, task, verbose=verbose)
            all_events += events
            all_meta.append(meta)
        except Exception as exc:
            print(f"  {task}: FAILED -- {sc.exc_detail(exc)}")
            all_meta.append({"task": task,
                             "error": sc.exc_detail(exc)})
    return all_events, all_meta
