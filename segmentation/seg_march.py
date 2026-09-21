"""
March.
======

OpenCap has no marching analysis, so this is new code. It follows the
segmentation markers document's march rows, which are defined on heel
vertical position rather than on the heading projections used for walking:

    S1 stepping-leg heel lift-off   r/L_calc_study vertical position begins
                                    sustained rise off baseline
    S2 peak heel height             r/L_calc_study vertical position local
                                    maximum
    S3 initiation of stepping foot  r/L_calc_study first sustained descent
       return to neutral            after that maximum
    S4 foot flat on floor (neutral) r/L_calc_study vertical position returns
                                    to baseline
    repeat S1-S4 for remaining cycles, alternating sides

The document's note on these rows reads "but, i think the knee flexion and
hip flexion gave clean and crisp spikes!" -- the marker-based definitions are
what is implemented here; the joint-angle version stays available in the
.mot overlay for comparison at review time.

WHY VERTICAL POSITION AND NOT THE WALKING PROJECTIONS
    Marching is stepping in place. The pelvis barely translates, so
    calc_rel_x -- a fore-aft projection relative to the ipsilateral PSIS --
    has no stride-length excursion to find peaks in. The discriminating
    signal is the vertical lift of each heel, which is what the document
    specifies.

THRESHOLDS
    Baseline is the 10th percentile of each heel's vertical position over the
    analysed window, which is the floor-contact level, robust to the trial
    starting or ending mid-step. A heel is "off baseline" once it clears
    BASELINE_BAND above that. Steps are found as local maxima at least
    STEP_PROMINENCE above their surroundings, which separates a deliberate
    march step from postural sway. All three are module constants so they can
    be tuned in one place after the manual check.
"""

import numpy as np

import seg_common as sc

from scipy.signal import find_peaks

TASKS = ["march"]

BASELINE_PERCENTILE = 10     # percent, the floor-contact level
BASELINE_BAND = 0.02         # m above baseline still counts as "flat"
STEP_PROMINENCE = 0.05       # m, minimum lift to count as a march step
DESCENT_VELOCITY = 0.05      # m/s downward for "sustained descent"
DESCENT_FRAMES = 3           # consecutive frames required

# A real march step takes most of a second. Rejecting anything shorter kills
# small wobbles of the airborne foot without raising STEP_PROMINENCE, which
# would cost sensitivity in a patient who marches low. On PR this removes one
# 0.115 m, 0.15 s left-heel bump that occurred while the right foot was still
# in the air, taking the count from 16 steps (alternation 14/15) to 15
# (alternation 15/15). Real steps in that trial lift 0.49-0.56 m and last
# 0.68-0.75 s, so the margin either side of this threshold is wide.
MIN_STEP_DURATION = 0.25     # s, from S1 to S4

# How long after the first contact a re-lift still counts as replanting the
# same step rather than starting the next one. PR's left heel lands at
# 9.18 s and takes off again at 9.75 s, so stance can be as short as 0.57 s
# and the window has to stay clear of that.
REPLANT_WINDOW_S = 0.5

# A replant means the foot actually left the floor, so the re-lift has to
# clear STEP_PROMINENCE *and* last this long. Height alone is not enough:
# PR's L_calc has a one-frame +95 mm spike at 9.80 s while the foot is flat
# on the floor, which passes any height test but is over in 0.12 s. Real
# lifts ramp smoothly over many frames.
RELIFT_MIN_S = 0.2


def _foot_cycles(time, y, side):
    """Find S1..S4 for one heel's vertical trace."""
    baseline = float(np.percentile(y, BASELINE_PERCENTILE))
    flat = baseline + BASELINE_BAND
    dt = float(np.mean(np.diff(time)))
    vel = np.gradient(y, time)

    peaks, _ = find_peaks(y, prominence=STEP_PROMINENCE)

    cycles = []
    for pk in peaks:
        # S1: last frame at or below the flat band before the peak.
        below = np.where(y[:pk] <= flat)[0]
        if not len(below):
            continue
        s1 = int(below[-1])

        # S4: the committed plant. A subject who is unsteady sets the foot
        # down, lifts it and sets it down again, and the step is not
        # finished until the foot stays there, so S4 advances to the last
        # contact of the replant rather than the first.
        #
        # What makes a later crossing a replant and not noise is that the
        # foot genuinely came back up in between. Requiring STEP_PROMINENCE
        # of clearance is the guard: on the controls the heel marker
        # chatters across the band by 2-32 mm while the foot rests flat on
        # the floor, and without this test those blips drag S4 up to 0.63 s
        # past the real landing on three of the four sessions.
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

        if time[s4] - time[s1] < MIN_STEP_DURATION:
            continue

        # S3: first sustained descent after the maximum -- the first frame
        # where downward velocity holds for DESCENT_FRAMES in a row.
        s3 = None
        for i in range(pk, max(pk, s4 - DESCENT_FRAMES) + 1):
            if np.all(vel[i:i + DESCENT_FRAMES] < -DESCENT_VELOCITY):
                s3 = i
                break
        if s3 is None:
            s3 = int(pk)

        cycles.append({"side": side, "s1": s1, "s2": int(pk), "s3": s3,
                       "s4": s4, "height": float(y[pk] - baseline)})
    return cycles, baseline


def segment(label, task="march", verbose=True):
    trim_start, trim_applied, trim_detail = sc.calibration_trim(label, task)
    md = sc.load_trc(label, task)
    win, time = sc._window(md, trim_start, 0.0)

    marker = {"r": "r_calc_study", "l": "L_calc_study"}
    all_cycles = []
    baselines = {}
    for side in ("r", "l"):
        y = win["markers"][marker[side]][:, 1]     # index 1 is vertical
        cycles, baseline = _foot_cycles(time, y, side)
        all_cycles += cycles
        baselines[side] = baseline

    all_cycles.sort(key=lambda c: c["s1"])

    events = []
    for n, c in enumerate(all_cycles, start=1):
        m = marker[c["side"]]
        for key, idx, name, definition in (
            ("S1", c["s1"], "stepping-leg heel lift-off",
             f"{m} vertical position begins sustained rise off baseline"),
            ("S2", c["s2"], "peak heel height",
             f"{m} vertical position local maximum"),
            ("S3", c["s3"], "initiation of stepping foot return to neutral",
             f"{m} first sustained descent after that maximum"),
            ("S4", c["s4"], "foot flat on floor (neutral)",
             f"{m} vertical position returns to baseline"),
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
