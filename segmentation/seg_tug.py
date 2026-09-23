import numpy as np

import seg_common as sc
import seg_gait
import seg_5tsts

from seg_sts_core import sts_analysis_leanfix as sts_analysis
from seg_sts_core import refine_end_rising

TASKS = ["tug"]

LOWPASS_STS = 6      # Hz, OpenCap's own sts default
EXPECTED_TURNS = 2   # protocol: turn at the cone, turn at the chair

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
    """Emit one pooled series' strides. Returns (events, cycle numbers used)."""
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

    end_rising_detail = []
    try:
        sts = sts_analysis(
            sd, tn,
            lowpass_cutoff_frequency_for_coordinate_values=LOWPASS_STS,
            n_sts_cycles=-1,
            trimming_start=trim_start,
            trimming_end=0)
        sev = sts.stsEvents
        end_rising_detail = refine_end_rising(sts)
        n_sts = len(sev["startRisingIdx"])
        sts_error = None
    except Exception as exc:
        sev, n_sts = {}, 0
        sts_error = sc.exc_detail(exc)

    missing = []
    events = []
    if n_sts < 1:
        missing += ["S1 forward lean onset", "S2 start of rising",
                    "S3 end of rising"]
        # no rise means no bound, so open at the calibration trim
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

    legs, prev_end = [], end_rising
    for a, b, _ in turns:
        legs.append((prev_end, a))
        prev_end = b
    if len(turns) < EXPECTED_TURNS:
        legs.append((prev_end, float(time[-1])))
    directions = ["outbound", "return", "extra"]

    # --- strides, one straight window at a time ---------------------------
    meta_windows = []
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
        # both anchor legs describe the same footfalls; pool them
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
