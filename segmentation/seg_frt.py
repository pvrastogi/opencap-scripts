import numpy as np

import seg_common as sc

TASKS = ["frt"]

# m. neutral is ~0.15m and reaches ~0.86m on PR, so far from both
REACH_THRESHOLD = 0.300

# s. Minimum time above threshold for the excursion to be "sustained".
MIN_HOLD_S = 1.0

RETURN_BAND = 0.080     # m. bounding needs a lower bar than detection; 0.30 put events ~0.2s off

# Percentile of the reach magnitude taken as the neutral level.
NEUTRAL_PERCENTILE = 10

# direction -> (first S number, onset event name, onset definition, adjective)
DEFS = {
    "forward": (1, "first forward reach movement",
                "both wrist midpoint begins sustained excursion along heading",
                "forward"),
    "right": (5, "first rightward reach movement",
              "second sustained excursion of the wrist midpoint",
              "rightward"),
    "left": (9, "first left reach movement",
             "third sustained excursion of the wrist midpoint",
             "left"),
}


def segment(label, task="frt", verbose=True):
    trim_start, trim_applied, trim_detail = sc.calibration_trim(label, task)
    md = sc.load_trc(label, task)
    win, t = sc._window(md, trim_start, 0.0)
    M = win["markers"]

    rw = (M["r_lwrist_study"] + M["r_mwrist_study"]) / 2
    lw = (M["L_lwrist_study"] + M["L_mwrist_study"]) / 2
    bw = (rw + lw) / 2
    psis = (M["r.PSIS_study"] + M["L.PSIS_study"]) / 2

    fwd_axis, right_axis = sc.ground_frame(win)

    rel = bw - psis
    fwd_c = rel @ fwd_axis
    right_c = rel @ right_axis
    mag = np.sqrt(fwd_c ** 2 + right_c ** 2)

    neutral_level = float(np.percentile(mag, NEUTRAL_PERCENTILE))
    threshold = neutral_level + REACH_THRESHOLD

    near = mag <= neutral_level + REACH_THRESHOLD / 2
    neutral_vec = np.array([float(np.median(fwd_c[near])),
                            float(np.median(right_c[near]))]) if near.any() \
        else np.array([0.0, 0.0])

    # Contiguous runs above threshold, kept only if sustained.
    above = mag > threshold
    runs = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            if t[j] - t[i] >= MIN_HOLD_S:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1

    return_level = neutral_level + RETURN_BAND

    found = []
    prev_return = 0
    for i0, i1 in runs:
        pk = i0 + int(np.argmax(mag[i0:i1 + 1]))
        disp = np.array([fwd_c[pk], right_c[pk]]) - neutral_vec
        f, r = float(disp[0]), float(disp[1])
        direction = "forward" if abs(f) >= abs(r) else ("right" if r > 0
                                                        else "left")

        # walk back from the peak, bounded by the previous reach's return
        onset = sc.movement_onset(mag, t, pk, search_from=prev_return)

        # back at neutral, not merely below the detection threshold
        after = np.where(mag[pk:] <= return_level)[0]
        ret = int(pk + after[0]) if len(after) else len(mag) - 1

        back = sc.movement_onset(-mag, t, ret, search_from=pk)
        prev_return = ret

        found.append({"direction": direction, "onset": onset, "return": ret,
                      "first_return": back,
                      "peak_idx": pk, "magnitude_m": float(mag[pk]),
                      "forward_m": f, "right_m": r,
                      "hold_s": float(t[i1] - t[i0]),
                      "above_threshold_s": [round(float(t[i0]), 3),
                                            round(float(t[i1]), 3)],
                      "bearing_deg": float(np.degrees(np.arctan2(r, f)))})

    events, assigned = [], {}
    for rec in found:
        d = rec["direction"]
        if d in assigned:
            continue
        assigned[d] = rec
        base, name, definition, adj = DEFS[d]
        events.append(sc.event(task, f"S{base}", name, t[rec["onset"]],
                               "trc", definition, anchor_leg=d))
        events.append(sc.event(
            task, f"S{base + 1}", f"maximal {adj} extension",
            t[rec["peak_idx"]], "trc",
            f"maximum of the pelvis-relative wrist midpoint within that "
            f"excursion ({rec['magnitude_m']:.3f} m)", anchor_leg=d))
        events.append(sc.event(
            task, f"S{base + 2}", "first movement return to neutral",
            t[rec["first_return"]], "trc",
            "walking back from the fastest decrease between that maximum "
            "and the return, to the start of the return movement",
            anchor_leg=d))
        events.append(sc.event(
            task, f"S{base + 3}", "return to neutral", t[rec["return"]],
            "trc",
            f"both wrist midpoint back within {RETURN_BAND:.3f} m of "
            f"neutral", anchor_leg=d))

    observed = [r["direction"] for r in found]
    documented = ["forward", "right", "left"]

    meta = {
        "task": task, "trial_file": sc.trial_name(label, task),
        "neutral_level_m": round(neutral_level, 3),
        "reach_threshold_m": round(threshold, 3),
        "return_level_m": round(return_level, 3),
        "n_reaches": len(assigned), "n_runs": len(runs),
        "observed_order": observed, "documented_order": documented,
        "order_matches_document": observed[:3] == documented,
        "reaches": {k: {"onset_s": round(float(t[v["onset"]]), 3),
                        "peak_s": round(float(t[v["peak_idx"]]), 3),
                        "first_return_s": round(float(t[v["first_return"]]), 3),
                        "return_s": round(float(t[v["return"]]), 3),
                        "hold_s": round(v["hold_s"], 2),
                        "above_threshold_s": v["above_threshold_s"],
                        "magnitude_m": round(v["magnitude_m"], 3),
                        "forward_m": round(v["forward_m"], 3),
                        "right_m": round(v["right_m"], 3),
                        "bearing_deg": round(v["bearing_deg"], 1)}
                    for k, v in assigned.items()},
        "trim_start_s": round(trim_start, 4), "trim_applied": trim_applied,
        "calibration_detail": trim_detail,
    }
    missing = [d for d in documented if d not in assigned]
    if missing:
        meta["warning"] = f"no reach matched: {', '.join(missing)}"

    if verbose:
        desc = ", ".join(
            f"{k} {v['magnitude_m']:.3f} m held {v['hold_s']:.1f}s "
            f"({v['bearing_deg']:+.0f} deg)"
            for k, v in sorted(assigned.items(),
                               key=lambda kv: kv[1]["onset"]))
        print(f"  {task}: neutral {neutral_level:.3f} m; {len(assigned)} "
              f"reach(es) [{desc}]; order "
              f"{'matches' if meta['order_matches_document'] else 'DIFFERS from'}"
              f" the document")
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
