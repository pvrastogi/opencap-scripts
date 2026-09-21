"""
Gait with a pivot turn: gait-pivot.
===================================

Two methods are composed. Turns come from pelvis_rotation in the .mot,
because OpenCap has no turn detector. Strides come from OpenCap's
gait_analysis, run separately inside each straight window, because it cannot
be run once over a trial containing a turn: the pelvis heading rotates
through 180 degrees during the pivot, and the order gate sees the strides
either side of it as one broken sequence.

    S1..S4  strides, outbound     OpenCap gait_analysis in window 1
    S5      start of turn         pelvis_rotation angular velocity increase
    S6      end of turn           angular velocity returns to baseline
    S1..S4  strides, return       OpenCap gait_analysis in window 2
    S7      start of turn
    S8      end of turn

TURN RULE
    Document: "above 30 deg/s, at least 90 degree rotation, consecutive
    angular-velocity bursts are joined only if their rotation changes have
    the same sign". The same-sign condition is what keeps the outbound and
    return turns from merging into one event; see seg_common.detect_turns.

WINDOWS
    window 1 = calibration trim  -> start of turn 1
    window 2 = end of turn 1     -> start of turn 2
    Subjects were told to do whatever they wanted at the turn -- some pause
    and step round, some pivot continuously -- so the windows are defined by
    the measured turn boundaries rather than by an assumed footfall pattern.

    The document's gait-pivot rows carry no "Sn, first body position over end
    tape" event, unlike the other gait variants, so none is emitted.

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

    Cycle numbers run straight through the task.
"""

import numpy as np

import seg_common as sc
import seg_gait

TASKS = ["gait-pivot"]

EXPECTED_TURNS = 2

TURN_DEFS = {
    "start": ("start of turn",
              "start of pelvis_rotation angular velocity increase above 30 "
              "deg/s, at least 90 degree rotation, consecutive "
              "angular-velocity bursts are joined only if their rotation "
              "changes have the same sign"),
    "end": ("end of turn",
            "pelvis_rotation angular velocity returns to baseline (~0)"),
}

STRIDE_DEFS = [
    ("S1", "ips_hs", "ipsilateral heel strike", "peak in {ips}_calc_rel_x"),
    ("S2", "cont_to", "contralateral toe-off", "trough in {cont}_toe_rel_x"),
    ("S3", "cont_hs", "contralateral heel strike", "peak in {cont}_calc_rel_x"),
    ("S4", "ips_to", "ipsilateral toe-off", "trough in {ips}_toe_rel_x"),
]


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


def segment(label, task="gait-pivot", verbose=True):
    trim_start, trim_applied, trim_detail = sc.calibration_trim(label, task)
    time, cols, mot_file = sc.load_mot(label, task)

    mask = time >= trim_start
    turns = sc.detect_turns(time[mask], cols["pelvis_rotation"][mask])

    if len(turns) > EXPECTED_TURNS:
        raise ValueError(f"expected {EXPECTED_TURNS} turns, detected "
                         f"{len(turns)}: "
                         + ", ".join(f"{a:.2f}-{b:.2f}s ({s:+.0f} deg)"
                                     for a, b, s in turns))

    # Fewer turns than expected is a half-done task, not a detector failure:
    # the subject walked out and did not pivot, or pivoted and did not walk
    # back. Raising there would discard the strides they did walk, so the
    # windows the turns do bound are segmented and the turn events that
    # cannot exist are recorded as missing instead. More turns than expected
    # is still an error, because that is the detector over-firing.
    #
    # N turns separate N+1 walking legs. On a complete trial the leg after
    # the final turn is the subject stopping, so it is dropped; when a turn
    # is missing, the trial ended mid-task and that last leg is real walking.
    legs, prev_end = [], trim_start
    for a, b, _ in turns:
        legs.append((prev_end, a))
        prev_end = b
    if len(turns) < EXPECTED_TURNS:
        legs.append((prev_end, float(time[-1])))
    directions = ["outbound", "return", "extra"]

    events = []
    meta_windows = []
    # One running counter for the whole task. Each window now emits a single
    # pooled series, and the windows do not overlap in time, so cycles can be
    # numbered straight through. The anchor leg is chosen per window and can
    # differ between them -- a per-leg counter would restart the numbering
    # mid-task whenever it switched.
    next_cycle = 1

    for wi, (w0, w1) in enumerate(legs, start=1):
        direction = directions[min(wi - 1, len(directions) - 1)]
        results = seg_gait.cycles_in_window_both(label, task, w0, w1)
        if not results:
            results = seg_gait.cycles_in_window_both(
                label, task, w0, w1, min_heel_strikes=1)
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
                print(f"    window {wi} ({direction}) "
                      f"{w0:.2f}-{w1:.2f}s: FAILED -- {err}")
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

    # Turn i contributes the pair (S5, S6), (S7, S8), ... so a missing turn
    # simply leaves its pair unemitted.
    missing = []
    for i in range(EXPECTED_TURNS):
        start_key, end_key = f"S{5 + 2 * i}", f"S{6 + 2 * i}"
        if i >= len(turns):
            missing += [f"{start_key} start of turn {i + 1}",
                        f"{end_key} end of turn {i + 1}"]
            continue
        for key, which, t in ((start_key, "start", turns[i][0]),
                              (end_key, "end", turns[i][1])):
            name, definition = TURN_DEFS[which]
            events.append(sc.event(task, key, name, t, "mot", definition))

    meta = {
        "task": task, "trial_file": sc.trial_name(label, task),
        "mot_file": mot_file,
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
        desc = " and ".join(f"{a:.2f}-{b:.2f}s ({s:+.0f} deg)"
                            for a, b, s in turns) or "none detected"
        print(f"  {task}: turns {desc}")
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
