"""
Semi-tandem and tandem stance: semi-tandem, tandem.
===================================================

Both tasks share one signal and one eight-event structure. From the
document's notes:

    Heading      = (r.ASIS + L.ASIS)/2 - (r.PSIS + L.PSIS)/2
    r_calc_rel_x = (r_calc_study - r.PSIS_study) . heading
    l_calc_rel_x = (L_calc_study - L.PSIS_study) . heading
    foot_offset  = r_calc_rel_x - l_calc_rel_x

    S1 first sustained movement of right foot        foot_offset departs its
                                                     neutral plateau
    S2 right foot touchdown in position              foot_offset arrives at
                                                     its first steady offset
                                                     plateau
    S3 first sustained movement of right foot (back) foot_offset departs
                                                     that plateau
    S4 right foot touchdown back in neutral          foot_offset returns to
                                                     its neutral plateau
    S5-S8  the same for the left foot, "(opposite sign)"

WHY foot_offset AND NOT THE TWO SIGNALS SEPARATELY
    Both heel projections are taken relative to the ipsilateral PSIS, so
    each one still contains whatever the pelvis does. Subtracting them
    cancels the pelvis and leaves the anteroposterior heel-to-heel offset,
    which is the thing the task actually manipulates. Standing in
    semi-tandem or tandem with the right foot forward drives foot_offset
    positive; left foot forward drives it negative; feet level sits near
    zero.

LABELS COME FROM THE SIGN, NOT THE ORDER
    The document numbers the right foot first, but subjects were free to
    lead with either foot, and NEW_GX led with the other one. So the two
    holds are matched to S1-S4 and S5-S8 by the SIGN of their offset, not by
    which came first. A trial that leads with the left foot therefore
    reports S5-S8 before S1-S4 in time, which is correct and is why the
    consecutiveness check treats the two hold groups as separate series.

    The named foot steps in FRONT (user-confirmed), which is what fixes
    positive offset = right foot forward.

MEASURED SCALE ON PR
    semi-tandem: neutral +0.021 m, holds +0.128 m (9.6 s) and -0.085 m
    (9.5 s). tandem: neutral +0.005 m, holds +0.277 m (8.4 s) and -0.257 m
    (8.3 s). Rolling sd is 0.001-0.002 m while holding and 0.03-0.08 m
    during the transitions, so the plateaus separate cleanly and the two
    tasks differ mainly in how far the foot travels, as expected.
"""

import numpy as np

import seg_common as sc

TASKS = ["semi-tandem", "tandem"]

# m. Rolling-sd ceiling for "steady". Holds sit at 0.001-0.002, transitions
# at 0.03-0.08, so this is an order of magnitude clear of both. Used only to
# locate the NEUTRAL level now, not to find the holds.
PLATEAU_SD = 0.010
PLATEAU_MIN_S = 2.0

# m. A hold must sit at least this far from the neutral level to count as an
# offset hold rather than another spell of standing level. Smallest real hold
# measured is 0.059; largest neutral is 0.021.
OFFSET_MIN = 0.040

# s. Minimum time past OFFSET_MIN for the excursion to be a hold.
MIN_HOLD_S = 2.0

# s. A dip back inside OFFSET_MIN shorter than this is the subject adjusting
# mid-hold, not the end of the hold.
HOLD_MERGE_GAP_S = 1.0

# Fraction of the travel from neutral to the hold level that bounds the
# "in position" band. Touchdown is arrival inside it, departure is the last
# frame in it, which is the same level-crossing idiom as sls-ec touchdown
# and march S4.
ARRIVE_FRACTION = 0.25

# s. A dip out of the in-position band shorter than this is a wobble, not
# an arrival or a departure. Without it a single frame outside the band
# immediately after the hold's most extreme frame ends the hold: PR's tandem
# right hold runs 10.1-19.1 s and its departure was read as 10.100 s.
BAND_BRIDGE_S = 0.5

# m. Band around neutral that counts as back in neutral position. 0.020
# misses OLD_MG's tandem return, which settles 0.029 m off neutral; 0.030
# catches it and moves the other returns by at most 0.05 s. It stays below
# OFFSET_MIN so "back in neutral" and "in a hold" cannot both be true.
NEUTRAL_BAND = 0.030

# The protocol is two ten-second holds.
EXPECTED_HOLDS = 2


def _find_holds(t, foot_offset, neutral_level):
    """Sustained excursions from neutral, either sign, as [(i0, i1, sign)].

    Holds used to be found as steady plateaus, which asks the subject to
    stand STILL rather than to stand in position. A patient who holds
    semi-tandem while swaying never produces a run with rolling sd under
    0.010 m for 2 s, and the hold went missing entirely -- DCM_001's left
    foot was simply not segmented. Requiring the offset to be sustained,
    without requiring it to be quiet, is the frt idiom and detects the same
    two holds in all ten control trials.
    """
    out = []
    for sgn in (1.0, -1.0):
        above = sgn * (foot_offset - neutral_level) >= OFFSET_MIN
        spans, i = [], 0
        while i < len(above):
            if above[i]:
                j = i
                while j + 1 < len(above) and above[j + 1]:
                    j += 1
                spans.append([i, j])
                i = j + 1
            else:
                i += 1
        merged = []
        for lo, hi in spans:
            if merged and t[lo] - t[merged[-1][1]] < HOLD_MERGE_GAP_S:
                merged[-1][1] = hi
            else:
                merged.append([lo, hi])
        out += [(lo, hi, sgn) for lo, hi in merged
                if t[hi] - t[lo] >= MIN_HOLD_S]
    out.sort()
    return out


def _band_runs(inside, t):
    """Contiguous runs of inside, with brief dips out of the band bridged.

    Scanning either side of the hold's most extreme frame was tried first
    and is not robust: a lead-in spike just past the hold level makes that
    frame the extreme one, and the run around it is a fraction of a second.
    OLD_GX's semi-tandem right hold reported 2.967-3.150 s against a real
    6.0-12.4 s that way. Enumerating the runs and taking the longest one
    overlapping the hold has no such anchor to be wrong about.
    """
    runs, i, n = [], 0, len(inside)
    while i < n:
        if inside[i]:
            j = i
            while j + 1 < n and inside[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for lo, hi in runs:
        if merged and t[lo] - t[merged[-1][1]] <= BAND_BRIDGE_S:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    return merged


def segment(label, task, verbose=True):
    trim_start, trim_applied, trim_detail = sc.calibration_trim(label, task)
    md = sc.load_trc(label, task)
    win, t = sc._window(md, trim_start, 0.0)

    sig = sc.walking_signals(win)
    foot_offset = sig["r_calc_rel_x"] - sig["l_calc_rel_x"]

    runs, sd, sd_thr = sc.steady_runs(t, foot_offset, window_s=1.0,
                                      sd_threshold=PLATEAU_SD,
                                      min_len_s=PLATEAU_MIN_S)

    # Neutral is where the feet are level, which by construction is
    # foot_offset ~ 0 -- not merely the steadiest level closest to zero.
    #
    # Taking the closest-to-zero steady run fails whenever the subject does
    # not stand still in neutral for long enough to register a plateau. The
    # closest run is then one of the two holds, every offset is measured
    # from it, and the opposite hold is the only one far enough away to
    # qualify. That cost one of the two holds in five of the eight tandem
    # trials: MG tandem took +0.234 m as neutral, NEW_GX semi-tandem
    # -0.156 m, NEW_GX tandem +0.247 m, OLD_MG semi-tandem -0.089 m, and MG
    # semi-tandem +0.075 m.
    #
    # So a steady run only counts as neutral if it is actually near zero,
    # and if none is, zero itself is used and the trial is flagged.
    neutral_runs = [r for r in runs if abs(r[2]) < OFFSET_MIN]
    if neutral_runs:
        neutral = max(neutral_runs, key=lambda r: r[1] - r[0])
        neutral_level = neutral[2]
        neutral_found = True
    else:
        neutral = (float(t[0]), float(t[0]), 0.0, 0, 0)
        neutral_level = 0.0
        neutral_found = False

    holds = _find_holds(t, foot_offset, neutral_level)

    # One hold per foot: the longest. A second excursion on the same side is
    # not part of the documented structure, so it is reported rather than
    # allowed to overwrite the first or to consume the other foot's labels.
    best, extra = {}, []
    for h in holds:
        side = "right" if h[2] > 0 else "left"
        if side in best:
            shorter = min(best[side], h, key=lambda x: x[1] - x[0])
            best[side] = max(best[side], h, key=lambda x: x[1] - x[0])
            extra.append([round(float(t[shorter[0]]), 3),
                          round(float(t[shorter[1]]), 3)])
        else:
            best[side] = h
    chosen = sorted(best.items(), key=lambda kv: kv[1][0])

    events, assigned, missing = [], {}, []
    prev_return = 0
    for forward, (i0, i1, sgn) in chosen:
        base = 1 if forward == "right" else 5
        sign_note = "" if forward == "right" else " (opposite sign)"
        pos = ("semi-tandem position" if task == "semi-tandem"
               else "tandem position")

        s = sgn * (foot_offset - neutral_level)
        level = float(np.median(foot_offset[i0:i1 + 1]))
        band = abs(level - neutral_level) * ARRIVE_FRACTION

        # TOUCHDOWN IS ARRIVAL AT THE POSITION, NOT THE END OF THE WOBBLE.
        # These were the start and end of the steady plateau, which asks
        # when the subject stopped moving rather than when the foot landed.
        # On a patient the two are seconds apart -- DCM_001's tandem
        # touchdown at 4.35 s was reported at 7.68 s, and the departure at
        # 13 s was reported at 10.8 s, because the hold was unsteady at both
        # ends. Bracketing the hold level instead moves the controls
        # -0.45 to -0.83 s on arrival and +0.37 to +0.72 s on departure,
        # consistently and with no outliers across all ten trials.
        inside = np.abs(foot_offset - level) <= band
        overlapping = [r for r in _band_runs(inside, t)
                       if r[0] <= i1 and r[1] >= i0]
        if overlapping:
            arrive, depart = max(overlapping,
                                 key=lambda r: t[r[1]] - t[r[0]])
        else:
            arrive, depart = i0, i1

        back = np.where(
            np.abs(foot_offset[depart:] - neutral_level) <= NEUTRAL_BAND)[0]
        ret = int(depart + back[0]) if len(back) else None

        # S1 walks back from the arrival to where the foot was not yet
        # moving, bounded by the previous hold's return. The old level
        # crossing had no answer at all when the subject never stood inside
        # the neutral band before the hold, and fell back to the first frame
        # of the trial -- DCM_001's semi-tandem reported 0.000 s against a
        # movement that starts at 1.7-1.9 s.
        onset = sc.movement_onset(s, t, arrive, search_from=prev_return)
        prev_return = ret if ret is not None else depart

        assigned[forward] = (float(t[arrive]), float(t[depart]), level)

        events.append(sc.event(
            task, f"S{base}",
            f"first sustained movement of {forward} foot", float(t[onset]),
            "trc",
            f"walking back from the fastest change to the start of the "
            f"movement out of neutral{sign_note}",
            side=forward[0], anchor_leg=forward))
        events.append(sc.event(
            task, f"S{base + 1}",
            f"{forward} foot touchdown in {pos}", float(t[arrive]), "trc",
            f"foot_offset arrives within {band:.3f} m of its hold level "
            f"({level:+.3f} m){sign_note}",
            side=forward[0], anchor_leg=forward))
        # The mirror of the touchdown: the last frame the foot is still in
        # position. Anchoring this on the fastest return and walking back,
        # as frt does, was tried and is not robust here -- with no bound
        # above it when the subject never returns, the walk-back latches
        # onto whichever blip is fastest and lands mid-hold.
        events.append(sc.event(
            task, f"S{base + 2}",
            f"first sustained movement of {forward} foot back to neutral",
            float(t[depart]), "trc",
            f"foot_offset departs the band around its hold level"
            f"{sign_note}",
            side=forward[0], anchor_leg=forward))
        # The subject who never brings the foot back has no touchdown in
        # neutral to report. It used to be filled in with the last frame of
        # the trial, which is as fabricated as sls-ec's truncated descent
        # was -- DCM_001 never returns the left foot in tandem.
        if ret is None:
            missing.append(f"S{base + 3} {forward} foot touchdown back in "
                           f"neutral position")
        else:
            events.append(sc.event(
                task, f"S{base + 3}",
                f"{forward} foot touchdown back in neutral position",
                float(t[ret]), "trc",
                f"foot_offset returns within {NEUTRAL_BAND:.3f} m of "
                f"neutral{sign_note}",
                side=forward[0], anchor_leg=forward))

    for forward in ("right", "left"):
        if forward not in best:
            b = 1 if forward == "right" else 5
            missing.append(f"S{b}-S{b + 3} {forward} foot hold not found")

    lead = chosen[0][0] if chosen else None

    meta = {
        "task": task, "trial_file": sc.trial_name(label, task),
        "neutral_level_m": round(neutral_level, 4),
        "neutral_plateau_found": neutral_found,
        "neutral_span_s": [round(neutral[0], 3), round(neutral[1], 3)],
        "n_holds": len(assigned), "expected_holds": EXPECTED_HOLDS,
        "leading_foot": lead,
        "holds": {k: {"in_position_s": [round(v[0], 3), round(v[1], 3)],
                      "duration_s": round(v[1] - v[0], 2),
                      "level_m": round(v[2], 4)}
                  for k, v in assigned.items()},
        "extra_holds_s": extra,
        "n_steady_runs": len(runs),
        "partial": bool(missing), "missing_events": missing,
        "trim_start_s": round(trim_start, 4), "trim_applied": trim_applied,
        "calibration_detail": trim_detail,
    }
    warnings = []
    if len(assigned) != EXPECTED_HOLDS:
        warnings.append(f"{len(assigned)} offset hold(s) found, protocol "
                        f"expects {EXPECTED_HOLDS}")
    if not neutral_found:
        warnings.append("no steady plateau near zero offset; neutral taken "
                        "as 0 m (subject did not hold still between holds)")
    if warnings:
        meta["warning"] = "; ".join(warnings)

    if verbose:
        desc = ", ".join(f"{k} {v[2]:+.3f} m for {v[1] - v[0]:.1f}s"
                         for k, v in sorted(assigned.items()))
        print(f"  {task}: neutral {neutral_level:+.3f} m; {len(assigned)} "
              f"hold(s) [{desc}]; led with {lead}")
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
