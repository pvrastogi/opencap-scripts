"""
Y-Balance: y-balance.
=====================

    S1-S3    right foot forward          r_toe_study begins sustained
                                         excursion relative to the
                                         contralateral foot / maximum
                                         excursion / returned to neutral
    S4-S6    left foot forward           L_toe_study, same three
    S7-S9    right foot right diagonal
    S10-S12  left foot left diagonal
    S13-S15  right foot left diagonal
    S16-S18  left foot right diagonal

DETECTION VERSUS REPORTING
    The document measures excursion "relative to the contralateral foot".
    That relative vector is the right thing to report -- it cancels any
    whole-body drift -- but it cannot say WHICH foot moved, because
    (r_toe - L_toe) and (L_toe - r_toe) are the same displacement mirrored,
    with identical magnitude. So each reach is detected on the moving toe's
    own displacement from its neutral position, where the stance foot stays
    put, and the reported magnitude is taken relative to the contralateral
    toe.

DIRECTION IS MEASURED, NOT COUNTED
    The document distinguishes forward from the two diagonals by ordinal
    position only. Here the bearing of each reach is measured in a frame
    fixed from the neutral periods, and reaches are matched to the event
    whose name fits. This matters: PR's pelvis heading swings up to 63.9
    degrees during this task, so against OpenCap's per-frame heading the
    fifth and sixth reaches measure -6.8 and -17.8 degrees and look like
    repeat forward reaches. In a fixed frame they are +55.5 and -64.5
    degrees, which is what they are -- the second diagonal pair.

    Measured on PR, in order: right forward (+5.7 deg), left forward
    (+1.7 deg), right diagonal (-59.7), left diagonal (+49.8), right
    diagonal the other way (+55.5), left diagonal the other way (-64.5).
    Each foot's two diagonals come out with opposite sign, as they should,
    and the whole sequence matches the document's structure.

    Positive lateral is the subject's right; see seg_common.ground_frame for
    how that sign is fixed anatomically.

THE THIRD MARKER IS THE COMPLETED RETURN, AND IS NAMED THAT WAY
    It was called "first movement return to neutral" and its definition
    string said "begins sustained return", but it has always been computed
    as the toe arriving back inside the neutral band -- the END of the
    return. It is now named "returned to neutral", which is what it is.

THE RETURN IS CORROBORATED WITH KNEE FLEXION
    The toe getting back to its neutral position is not the whole of getting
    back to neutral: the subject is still bent over it. At the frame the toe
    comes back inside the band, the more-flexed knee is still at 9.6-52.2
    degrees on the five controls -- PR is at 35-52 degrees on all six
    reaches. So the event waits for both: the toe back inside the band AND
    max(knee_angle_r, knee_angle_l) back within KNEE_BAND_DEG of its
    standing level, whichever is later. Both legs are covered by the max,
    which is the more-flexed of the two, so the stance leg cannot report the
    subject upright while the reaching leg is still bent.

    Measured across the forty-two control reaches this moves the event
    +0.00 to +0.43 s later. Never earlier, by construction.

    THE BASELINE HAS TO COME FROM THE QUIET FRAMES. A low percentile of the
    knee trace looks like the same thing and is not: it lands on the single
    straightest instant of the trial (0.0-0.1 deg on four of the five
    controls), which the subject never returns to, and the search then runs
    3-12 s past the reach or off the end of the trial. Taking the median
    over the frames where NEITHER toe is displaced gives 0.0-8.4 deg, a
    level the subject actually stands at.

THE PEAK IS STILL THE MAXIMUM
    Two corroborations were tried for the peak and both were rejected on the
    controls. The contralateral-relative displacement -- the signal the
    document names -- puts the maximum anywhere from 0.72 s before to 1.10 s
    after the toe's own maximum. Peak stance-knee or stance-hip flexion is
    worse, scattering from 1.03 s before to 1.93 s after: the stance leg
    keeps sinking after the toe has stopped advancing. Neither is a clock
    for the other, so the peak stays the maximum excursion.
"""

import numpy as np

import seg_common as sc

from scipy.signal import find_peaks

TASKS = ["y-balance"]

# m. PR's six reaches all measure 0.93-1.09 m, and the stance foot's median
# displacement is 0.06-0.08 m, so this sits far from both.
REACH_PROMINENCE = 0.150

# m. Band around neutral that counts as returned.
BASELINE_BAND = 0.100

# deg. Band above the standing knee-flexion level that counts as upright.
# The controls' quiet-frame baselines are 0.0-8.4 deg and the knees are
# 9.6-52.2 deg at the toe's return, so this discriminates the two states
# without being tight enough to catch postural sway. At 5 deg one control
# reach (MG, left) waits an extra 1.13 s; at 8 that becomes 0.30 s.
KNEE_BAND_DEG = 8.0

# deg. A reach whose bearing is inside this cone counts as forward; outside
# it, the sign of the lateral component names the diagonal. PR's forward
# reaches measure +5.7 and +1.7 deg, the diagonals +-49.8 to +-64.5, so the
# gap either side of 30 is wide.
FORWARD_CONE_DEG = 30.0

FOOT = {"r": ("right", "r_toe_study", "L_toe_study"),
        "l": ("left", "L_toe_study", "r_toe_study")}

# (foot, which reach for that foot) -> first S number of the triplet
LABELS = {("r", 0): 1, ("l", 0): 4, ("r", 1): 7,
          ("l", 1): 10, ("r", 2): 13, ("l", 2): 16}

NAMES = {1: "right foot forward", 4: "left foot forward",
         7: "right foot right diagonal", 10: "left foot left diagonal",
         13: "right foot left diagonal", 16: "left foot right diagonal"}


def segment(label, task="y-balance", verbose=True):
    trim_start, trim_applied, trim_detail = sc.calibration_trim(label, task)
    md = sc.load_trc(label, task)
    win, t = sc._window(md, trim_start, 0.0)
    M = win["markers"]

    # Fixed frame from the quiet periods of the right toe.
    r_pos = M["r_toe_study"]
    r_base = sc.dense_baseline(r_pos)
    rd = r_pos - r_base
    r_h = np.sqrt(rd[:, 0] ** 2 + rd[:, 2] ** 2)
    quiet = r_h < np.percentile(r_h, 40)
    fwd_axis, right_axis = sc.ground_frame(win, quiet)

    # Both feet's displacement is needed before either is used, because the
    # frames that fix the knee baseline are the ones where NEITHER foot is
    # reaching.
    disp = {}
    for leg, (_, moving, _) in FOOT.items():
        dd = M[moving] - sc.dense_baseline(M[moving])
        disp[leg] = (dd, np.sqrt(dd[:, 0] ** 2 + dd[:, 2] ** 2))

    mot_t, mot_cols, mot_file = sc.load_mot(label, task)
    knee = np.maximum(np.interp(t, mot_t, mot_cols["knee_angle_r"]),
                      np.interp(t, mot_t, mot_cols["knee_angle_l"]))
    standing = (disp["r"][1] < BASELINE_BAND) & (disp["l"][1] < BASELINE_BAND)
    knee_base = (float(np.median(knee[standing])) if standing.any()
                 else float(np.percentile(knee, 10)))
    knee_level = knee_base + KNEE_BAND_DEG

    reaches = []
    for leg, (name, moving, stance) in FOOT.items():
        d, h = disp[leg]
        peaks, _ = find_peaks(h, prominence=REACH_PROMINENCE)
        for pk in peaks:
            rel = M[moving][pk] - M[stance][pk]
            rel_base = np.median(M[moving] - M[stance], axis=0)
            rel_d = rel - rel_base
            f = float(d[pk] @ fwd_axis)
            r = float(d[pk] @ right_axis)
            onset, toe_ret = sc.excursion_bounds(h, int(pk), 0.0,
                                                 BASELINE_BAND)
            # Whichever is later: the toe back inside the band, or the knee
            # back at its standing level. A knee that never comes back --
            # the subject stays bent to the end of the trial -- leaves the
            # toe's own return standing, and is reported.
            ret, knee_ret = toe_ret, None
            if toe_ret is not None:
                back = np.where(knee[toe_ret:] <= knee_level)[0]
                if len(back):
                    knee_ret = int(toe_ret + back[0])
                    ret = knee_ret
            reaches.append({
                "leg": leg, "peak_idx": int(pk), "onset": onset,
                "return": ret, "toe_return": toe_ret, "knee_return": knee_ret,
                "own_magnitude_m": float(h[pk]),
                "relative_magnitude_m": float(
                    np.sqrt(rel_d[0] ** 2 + rel_d[2] ** 2)),
                "forward_m": f, "right_m": r,
                "bearing_deg": float(np.degrees(np.arctan2(r, f))),
            })

    reaches.sort(key=lambda x: x["peak_idx"])
    for rec in reaches:
        b = rec["bearing_deg"]
        if abs(b) <= FORWARD_CONE_DEG:
            rec["direction"] = "forward"
        else:
            rec["direction"] = "right diagonal" if b > 0 else "left diagonal"

    # Per foot, in time order: reach 0 must be forward, then the two
    # diagonals in the order they occur.
    events, assigned = [], {}
    per_foot_count = {"r": 0, "l": 0}
    for rec in reaches:
        leg = rec["leg"]
        idx = per_foot_count[leg]
        if (leg, idx) not in LABELS:
            continue
        per_foot_count[leg] += 1
        base_label = LABELS[(leg, idx)]
        assigned[base_label] = rec
        name = NAMES[base_label]
        moving = FOOT[leg][1]

        if rec["onset"] is not None:
            events.append(sc.event(
                task, f"S{base_label}", f"first movement {name}",
                t[rec["onset"]], "trc",
                f"{moving} begins sustained excursion relative to the "
                f"contralateral foot", side=leg, anchor_leg=leg))
        events.append(sc.event(
            task, f"S{base_label + 1}", f"peak {name}",
            t[rec["peak_idx"]], "trc",
            f"maximum excursion of {moving} "
            f"({rec['relative_magnitude_m']:.3f} m relative to the "
            f"contralateral foot)", side=leg, anchor_leg=leg))
        if rec["return"] is not None:
            events.append(sc.event(
                task, f"S{base_label + 2}", "returned to neutral",
                t[rec["return"]], "trc",
                f"{moving} back within {BASELINE_BAND:.3f} m of its neutral "
                f"position and the more-flexed knee back within "
                f"{KNEE_BAND_DEG:.0f} deg of standing, whichever is later",
                side=leg, anchor_leg=leg))

    meta = {
        "task": task, "trial_file": sc.trial_name(label, task),
        "mot_file": mot_file,
        "n_reaches": len(assigned), "expected_reaches": 6,
        "pelvis_heading_swing_deg": None,
        "knee_standing_level_deg": round(knee_base, 1),
        "knee_return_level_deg": round(knee_level, 1),
        "reaches": {f"S{k}": {
            "name": NAMES[k], "leg": v["leg"],
            "peak_s": round(float(t[v["peak_idx"]]), 3),
            "toe_return_s": (round(float(t[v["toe_return"]]), 3)
                             if v["toe_return"] is not None else None),
            "return_s": (round(float(t[v["return"]]), 3)
                         if v["return"] is not None else None),
            "knee_held_flexed_s": (
                round(float(t[v["return"]] - t[v["toe_return"]]), 3)
                if v["toe_return"] is not None and v["knee_return"] is not None
                else None),
            "measured_direction": v["direction"],
            "bearing_deg": round(v["bearing_deg"], 1),
            "relative_magnitude_m": round(v["relative_magnitude_m"], 3),
            "own_magnitude_m": round(v["own_magnitude_m"], 3),
        } for k, v in sorted(assigned.items())},
        "trim_start_s": round(trim_start, 4), "trim_applied": trim_applied,
        "calibration_detail": trim_detail,
    }

    # Does the measured direction agree with the label the structure implies?
    mismatches = []
    for k, v in assigned.items():
        expected = ("forward" if "forward" in NAMES[k]
                    else "right diagonal" if "right diagonal" in NAMES[k]
                    else "left diagonal")
        if v["direction"] != expected:
            mismatches.append(f"S{k} labelled {expected}, measured "
                              f"{v['direction']} ({v['bearing_deg']:+.0f} deg)")
    if mismatches:
        meta["direction_mismatches"] = mismatches
    if len(assigned) != 6:
        meta["warning"] = f"{len(assigned)} reaches found, expected 6"

    if verbose:
        print(f"  {task}: {len(assigned)} reach(es); bearings "
              + ", ".join(f"{v['leg']}{v['bearing_deg']:+.0f}"
                          for _, v in sorted(assigned.items())))
        for m in mismatches:
            print(f"    NOTE {m}")
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
