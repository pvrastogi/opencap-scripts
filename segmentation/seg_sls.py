"""
Single-leg stance with eyes closed: sls-ec.
===========================================

Ten events, five per leg, all on calcaneus vertical position:

    S1 right foot lift-off              r_calc_study vertical position
                                        begins sustained rise off baseline
    S2 right foot at mid-shin height    r_calc_study vertical position
                                        plateaus
    S3 eyes close                       from video                  manual
    S4 first sustained movement of      r_calc_study vertical position
       right foot back to neutral       begins sustained descent
    S5 right foot touchdown             r_calc_study vertical position
                                        returns to baseline
    S6-S10  the same for the left leg on L_calc_study, with S8 eyes close

S3 and S8 are video-only and are not emitted.

The document's note on S8 reads "look at ankle dorsi/plantarflexion, could
be discriminatory" and on S10 "maybe hip flexion dropping significantly" --
both recorded as leads to follow, neither implemented as the marker.

RIGHT LEG FIRST, BUT NOT ASSUMED
    PR and the other four controls all raised the right leg first, so S1-S5
    do precede S6-S10 in practice. The module still assigns by which leg
    lifted rather than by order, so a trial that leads with the left is
    labelled correctly and simply reports S6-S10 earlier in time. The two
    legs are separate series for the consecutiveness check.

SCALE
    A clear, large signal: on PR the calcaneus lifts 0.399 m (right) and
    0.356 m (left) off a 0.07 m baseline, and the raised holds run 13-17 s.
    Nothing here is near the noise floor, unlike rtt.
"""

import numpy as np

import seg_common as sc

from scipy.signal import find_peaks

TASKS = ["sls-ec"]

BASELINE_PERCENTILE = 10
BASELINE_BAND = 0.030        # m above baseline still counts as down
LIFT_PROMINENCE = 0.100      # m, minimum lift to count as a leg raise
MIN_HOLD_S = 2.0             # s, from lift-off to touchdown

# Foot down for longer than this ends the attempt. Below it, the foot going
# back up is a replant within the same attempt, so the raise is bounded by
# the first lift-off and the last touchdown across the whole thing.
REPLANT_GAP_S = 3.0

# Plateau at mid-shin height.
PLATEAU_SD = 0.020
PLATEAU_MIN_S = 1.0

LEG_LABELS = {"r": ("right", "r_calc_study", 1),
              "l": ("left", "L_calc_study", 6)}


def _one_leg(task, t, y, leg):
    name, marker, base_label = LEG_LABELS[leg]
    base = float(np.percentile(y, BASELINE_PERCENTILE))

    # Raised periods by level crossing, not by find_peaks. A peak needs the
    # signal to come back down on both sides by the prominence, so a raise
    # that is still in progress when the trial ends is invisible to it: MG's
    # left leg lifts 0.280 m and holds it to the last frame, and find_peaks
    # returned nothing at all for that leg.
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

    # The attempt is the longest qualifying raise, extended over any
    # replants. A foot put down and lifted again within REPLANT_GAP_S is the
    # same attempt, so S1 is the FIRST lift-off and S5 the LAST touchdown of
    # the whole thing; a foot left down for longer than that ended the
    # attempt, and a later raise is a second attempt we do not report.
    #
    # Extending outwards from the longest qualifying run, rather than
    # merging every raised period, is what keeps a stray 3 cm blip near the
    # floor from dragging the onset backwards into a period the subject was
    # not actually balancing.
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
        # S2 is arrival at mid-shin height, so it is the start of the FIRST
        # raised plateau; S4 is the departure downwards, so it is the end of
        # the LAST one. Taking the longest plateau for both puts S4 early
        # whenever the raised leg wobbles into more than one plateau: on PR's
        # right leg that reported the descent at 19.117 s against a touchdown
        # at 25.117 s.
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
    # The descent and the touchdown are both unobservable while the foot is
    # still raised at the last frame: what looks like the end of the plateau
    # is only the data running out. PR's sls-ec truncated at 20 s puts the
    # descent at 19.117 s when the real one is at 24.500 s, so S4 was as
    # fabricated as S5 would have been -- it just was not being suppressed.
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
            # Non-zero means the foot came down and went back up inside the
            # attempt, so S1/S5 span more than one raised period.
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
