import numpy as np

import seg_common as sc

from scipy.signal import find_peaks

TASKS = ["sls-ec"]

BASELINE_PERCENTILE = 10     # floor level; robust to a trial starting or ending mid-raise
BASELINE_BAND = 0.030        # m above baseline still counts as down
LIFT_PROMINENCE = 0.100      # m, minimum lift to count as a leg raise
MIN_HOLD_S = 2.0             # s, from lift-off to touchdown

REPLANT_GAP_S = 3.0          # s. foot down longer than this ended the attempt

# Plateau at mid-shin height.
PLATEAU_SD = 0.020
PLATEAU_MIN_S = 1.0          # s. raised holds run 13-17s, so easily cleared

# trailing number = first S label for that leg (S1-S5 right, S6-S10 left)
LEG_LABELS = {"r": ("right", "r_calc_study", 1),
              "l": ("left", "L_calc_study", 6)}


def _one_leg(task, t, y, leg):
    name, marker, base_label = LEG_LABELS[leg]
    base = float(np.percentile(y, BASELINE_PERCENTILE))

    raised = y > base + BASELINE_BAND
    runs = []
    i = 0
    while i < len(raised):
        if raised[i]:
            j = i
            while j + 1 < len(raised) and raised[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1

    qualifying = [(i0, i1) for i0, i1 in runs
                  if (t[i1] - t[i0]) >= MIN_HOLD_S
                  and (y[i0:i1 + 1].max() - base) >= LIFT_PROMINENCE]
    if not qualifying:
        return [], None

    i0, i1 = max(qualifying, key=lambda r: t[r[1]] - t[r[0]])
    absorbed = 0
    extending = True
    while extending:
        extending = False
        for a, b in runs:
            if a < i0 and t[i0] - t[b] < REPLANT_GAP_S:
                i0, extending, absorbed = a, True, absorbed + 1
            elif b > i1 and t[a] - t[i1] < REPLANT_GAP_S:
                i1, extending, absorbed = b, True, absorbed + 1

    onset = i0
    pk = i0 + int(np.argmax(y[i0:i1 + 1]))
    # The foot is still up at the last frame if the run reaches the end.
    truncated = i1 >= len(y) - 2
    ret = None if truncated else i1

    hi = (ret if ret is not None else len(y) - 1)
    runs, _, _ = sc.steady_runs(t[onset:hi + 1], y[onset:hi + 1],
                                window_s=0.5, sd_threshold=PLATEAU_SD,
                                min_len_s=PLATEAU_MIN_S)
    high = [r for r in runs if r[2] > base + BASELINE_BAND]
    if high:
        first_high = min(high, key=lambda r: r[0])
        last_high = max(high, key=lambda r: r[1])
        s2_time, plateau_level = first_high[0], first_high[2]
        descent_start = last_high[1]
        plateau_span = [round(first_high[0], 3), round(last_high[1], 3)]
    else:
        s2_time, plateau_level = float(t[pk]), float(y[pk])
        descent_start = float(t[pk])
        plateau_span = None

    b = base_label
    events = [
        sc.event(task, f"S{b}", f"{name} foot lift-off", t[onset], "trc",
                 f"{marker} vertical position begins sustained rise off "
                 f"baseline", side=leg, anchor_leg=leg),
        sc.event(task, f"S{b + 1}", f"{name} foot at mid-shin height",
                 s2_time, "trc",
                 f"{marker} vertical position plateaus "
                 f"({plateau_level:.3f} m)", side=leg, anchor_leg=leg),
    ]
    if ret is not None:
        events.append(sc.event(
            task, f"S{b + 3}",
            f"first sustained movement of {name} foot back to neutral",
            descent_start, "trc",
            f"{marker} vertical position begins sustained descent",
            side=leg, anchor_leg=leg))
        events.append(sc.event(
            task, f"S{b + 4}", f"{name} foot touchdown", t[ret], "trc",
            f"{marker} vertical position returns to baseline",
            side=leg, anchor_leg=leg))
    info = {"lift_off_s": round(float(t[onset]), 3),
            "touchdown_s": (round(float(t[ret]), 3) if ret is not None
                            else None),
            "truncated_by_end_of_trial": bool(ret is None),
            "missing_events": ([] if ret is not None else
                               [f"S{base_label + 3} descent onset",
                                f"S{base_label + 4} touchdown"]),
            # non-zero means the foot came down and back up mid-attempt
            "replants_absorbed": absorbed,
            "hold_duration_s": round(float(t[hi] - t[onset]), 2),
            "peak_height_m": round(float(y[pk] - base), 3),
            "plateau_level_m": round(plateau_level - base, 3),
            "plateau_span_s": plateau_span,
            "baseline_m": round(base, 4)}
    return events, info


def segment(label, task="sls-ec", verbose=True):
    trim_start, trim_applied, trim_detail = sc.calibration_trim(label, task)
    md = sc.load_trc(label, task)
    win, t = sc._window(md, trim_start, 0.0)

    events, per_leg = [], {}
    for leg in ("r", "l"):
        y = win["markers"][LEG_LABELS[leg][1]][:, 1]
        ev, info = _one_leg(task, t, y, leg)
        events += ev
        per_leg[leg] = info or {"error": "no qualifying leg raise found"}

    lifts = {k: v["lift_off_s"] for k, v in per_leg.items()
             if v and "lift_off_s" in v}
    first = min(lifts, key=lifts.get) if lifts else None

    meta = {
        "task": task, "trial_file": sc.trial_name(label, task),
        "first_leg": first, "per_leg": per_leg,
        "trim_start_s": round(trim_start, 4), "trim_applied": trim_applied,
        "calibration_detail": trim_detail,
        "skipped_manual": ["S3 eyes close (right)", "S8 eyes close (left)"],
    }
    if verbose:
        desc = ", ".join(
            f"{k} {v['hold_duration_s']:.1f}s at {v['plateau_level_m']:.3f} m"
            for k, v in per_leg.items() if "hold_duration_s" in v)
        print(f"  {task}: {first} leg first; holds [{desc}]")
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
