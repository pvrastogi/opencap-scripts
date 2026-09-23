import numpy as np

import seg_common as sc

from scipy.signal import find_peaks

TASKS = ["march"]

BASELINE_PERCENTILE = 10     # percent, the floor-contact level
BASELINE_BAND = 0.02         # m above baseline still counts as "flat"
STEP_PROMINENCE = 0.05       # m, minimum lift to count as a march step
DESCENT_VELOCITY = 0.05      # m/s downward for "sustained descent"
DESCENT_FRAMES = 3           # consecutive frames required

MIN_STEP_DURATION = 0.25     # s, from S1 to S4

REPLANT_WINDOW_S = 0.5   # s. PR stance can be 0.57s, longer chases the next step

RELIFT_MIN_S = 0.2       # s. PR has a 1-frame +95mm spike; real lifts ramp over many frames


def _foot_cycles(time, y, toe_y, side):
    """Find S1..S4 for one foot: steps from the heel's vertical trace, S1/S4
    from the toe's."""
    baseline = float(np.percentile(y, BASELINE_PERCENTILE))
    flat = baseline + BASELINE_BAND
    dt = float(np.mean(np.diff(time)))
    vel = np.gradient(y, time)
    toe_vel = np.gradient(toe_y, time)

    peaks, _ = find_peaks(y, prominence=STEP_PROMINENCE)

    cycles = []
    for pk in peaks:
        # S1: last frame at or below the flat band before the peak.
        below = np.where(y[:pk] <= flat)[0]
        if not len(below):
            continue
        s1 = int(below[-1])

        below = y[pk:] <= flat
        if not below.any():
            continue
        crossings = pk + np.flatnonzero(
            below & ~np.concatenate(([False], below[:-1])))
        s4 = int(crossings[0])
        for c in crossings[1:]:
            # Past REPLANT_WINDOW_S this is the foot's next march step, not
            # a replant of this one.
            if time[c] - time[s4] >= REPLANT_WINDOW_S:
                break
            lift = y[s4:c] - flat
            if (lift.max() >= STEP_PROMINENCE
                    and (lift > 0).sum() * dt >= RELIFT_MIN_S):
                s4 = int(c)

        # Duration gate on the off-floor span, before widening below.
        if time[s4] - time[s1] < MIN_STEP_DURATION:
            continue

        # Heel off the floor to the zero crossings, bounding the toe search.
        while s1 > 0 and vel[s1] > 0:
            s1 -= 1
        while s4 < len(vel) - 1 and vel[s4] < 0:
            s4 += 1

        # S3: first frame where downward velocity holds DESCENT_FRAMES in a row
        s3 = None
        for i in range(pk, max(pk, s4 - DESCENT_FRAMES) + 1):
            if np.all(vel[i:i + DESCENT_FRAMES] < -DESCENT_VELOCITY):
                s3 = i
                break
        if s3 is None:
            s3 = int(pk)

        # S1/S4 = toe vertical-velocity zero crossings, the StaBLE march
        # convention (Heigold et al., stable-analysis @ 3233a5c,
        # get_features_march.py): walk back from the toe's peak upward
        # velocity, forward from its peak downward velocity.
        s1 = s1 + int(np.argmax(toe_vel[s1:pk + 1]))
        while s1 > 0 and toe_vel[s1] > 0:
            s1 -= 1
        s4 = s3 + int(np.argmin(toe_vel[s3:s4 + 1]))
        while s4 < len(toe_vel) - 1 and toe_vel[s4] < 0:
            s4 += 1

        cycles.append({"side": side, "s1": s1, "s2": int(pk), "s3": s3,
                       "s4": s4, "height": float(y[pk] - baseline)})
    return cycles, baseline


def segment(label, task="march", verbose=True):
    trim_start, trim_applied, trim_detail = sc.calibration_trim(label, task)
    md = sc.load_trc(label, task)
    win, time = sc._window(md, trim_start, 0.0)

    marker = {"r": "r_calc_study", "l": "L_calc_study"}
    toe = {"r": "r_toe_study", "l": "L_toe_study"}
    all_cycles = []
    baselines = {}
    for side in ("r", "l"):
        y = win["markers"][marker[side]][:, 1]     # index 1 is vertical
        cycles, baseline = _foot_cycles(
            time, y, win["markers"][toe[side]][:, 1], side)
        all_cycles += cycles
        baselines[side] = baseline

    all_cycles.sort(key=lambda c: c["s1"])

    events = []
    for n, c in enumerate(all_cycles, start=1):
        m, tm = marker[c["side"]], toe[c["side"]]
        for key, idx, name, definition in (
            ("S1", c["s1"], "stepping-leg toe-off",
             f"{tm} vertical velocity zero crossing before its peak upward "
             f"velocity (StaBLE, Heigold et al.)"),
            ("S2", c["s2"], "peak heel height",
             f"{m} vertical position local maximum"),
            ("S3", c["s3"], "initiation of stepping foot return to neutral",
             f"{m} first sustained descent after that maximum"),
            ("S4", c["s4"], "stepping-foot toe contact (neutral)",
             f"{tm} vertical velocity zero crossing after its peak downward "
             f"velocity (StaBLE, Heigold et al.)"),
        ):
            events.append(sc.event(task, key, name, time[idx], "trc",
                                   definition, cycle=n, side=c["side"]))

    sides = [c["side"] for c in all_cycles]
    alternating = sum(1 for a, b in zip(sides, sides[1:]) if a != b)

    meta = {
        "task": task, "trial_file": sc.trial_name(label, task),
        "n_cycles": len(all_cycles),
        "n_right": sides.count("r"), "n_left": sides.count("l"),
        "alternation": f"{alternating}/{max(len(sides) - 1, 0)}",
        "peak_heights_m": [round(c["height"], 3) for c in all_cycles],
        "baseline_r_m": round(baselines["r"], 4),
        "baseline_l_m": round(baselines["l"], 4),
        "trim_start_s": round(trim_start, 4), "trim_applied": trim_applied,
        "calibration_detail": trim_detail,
    }
    if verbose:
        print(f"  {task}: {len(all_cycles)} steps "
              f"({sides.count('r')} right, {sides.count('l')} left), "
              f"alternation {meta['alternation']}, trim {trim_start:.3f}s")
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
