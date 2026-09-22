import numpy as np

import seg_common as sc

from scipy.signal import find_peaks

TASKS = ["y-balance"]

# m. reaches are 0.93-1.09m, stance foot 0.06-0.08m, so far from both
REACH_PROMINENCE = 0.150

# m. Band around neutral that counts as returned.
BASELINE_BAND = 0.100

KNEE_BAND_DEG = 8.0      # deg. quiet-frame baselines are 0-8.4; knees 9.6-52.2 at toe return

FORWARD_CONE_DEG = 30.0  # deg. PR forward reaches +1.7/+5.7, diagonals +-49.8 to +-64.5

FOOT = {"r": ("right", "r_toe_study", "L_toe_study"),
        "l": ("left", "L_toe_study", "r_toe_study")}

# (foot, which reach for that foot) -> first S number of the triplet
LABELS = {("r", 0): 1, ("l", 0): 4, ("r", 1): 7,
          ("l", 1): 10, ("r", 2): 13, ("l", 2): 16}

# key = first S number of each reach triplet
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

    # per foot in time order: forward first, then the two diagonals
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
