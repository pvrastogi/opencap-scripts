"""
Functional reach test: frt.
===========================

    S1 first forward reach movement     both wrist midpoint begins sustained
                                        excursion along heading        auto
    S2 maximal forward extension        maximum of the pelvis-relative wrist
                                        midpoint within that excursion
    S3 first movement return to neutral both wrist midpoint begins returning
                                        to baseline
    S4 return to neutral                both wrist midpoint is back at
                                        baseline
    S5-S8   the same for the rightward reach, second sustained excursion
    S9-S12  the same for the left reach, third sustained excursion

FOUR MARKERS PER REACH, NOT TWO
    The reach used to be bounded by its onset and its completion only, and
    the completion was named "first movement return to neutral" although it
    is computed as the wrist arriving back at baseline -- the END of the
    return, not its beginning. Both the maximum and the true first movement
    of the return are now emitted, and the completion carries the name it
    always described.

    The two new markers reuse definitions already validated on other tasks
    rather than inventing any:

    maximal extension            the maximum of the detection signal inside
                                 the sustained excursion, as march S2 takes
                                 the local maximum of heel height and
                                 y-balance takes the maximum toe excursion.

    first movement return        seg_common.movement_onset run on the
                                 NEGATED magnitude between the maximum and
                                 the completed return: anchor on the fastest
                                 decrease, then walk back to where the wrist
                                 was not yet returning. That is the same
                                 construction as the reach onset itself and
                                 as arms S1/S3/S5/S7, mirrored in time, so
                                 no new helper is needed.

    Measured across the fifteen control reaches every quadruple comes out
    correctly ordered, with holds of 0.18-3.47 s between the maximum and the
    first movement back and unwinds of 0.43-2.85 s after it.

NUMBERING
    Four markers per reach means S1-S4 forward, S5-S8 rightward, S9-S12
    left, so the numbers run in time order across the task as they do in
    every other module. The old S3 and S5 (the rightward and left onsets)
    are now S5 and S9.

    r_wrist midpoint = (r_lwrist_study + r_mwrist_study)/2
    l_wrist midpoint = (L_lwrist_study + L_mwrist_study)/2

"both wrist midpoint" is the mean of the two, following the document's note
that this is an upper-extremity analogue of OpenCap's ankle joint centre
idiom, (r_ankle_study + r_mankle_study)/2.

THE SIGNAL IS PELVIS-RELATIVE, NOT ABSOLUTE
    The reach is measured as the wrist midpoint relative to the mid-PSIS,
    projected onto a fixed ground frame -- the same construction OpenCap
    uses for gait, (marker - PSIS) . heading, and the same one the document
    writes out for calc_rel_x. Body-relative is what makes the neutral
    posture well defined: it removes whole-body sway and any step, so
    "returns to baseline" means the arm came back, not that the subject
    stopped drifting.

    Measured on PR, the magnitude of that vector is about 0.15 m at neutral
    and 0.85-0.87 m at all three reaches, so the two states are separated by
    a factor of five and the threshold sits nowhere near either.

WHY NOT A SPATIAL BASELINE
    Two spatial baselines were tried and both fail on this task, which is
    worth recording because they look reasonable:

    A plain median lands between neutral and the reach, so the neutral
    periods read as backward excursions -- four spurious peaks interleaved
    with the three real reaches.

    The most-occupied still position lands INSIDE a reach. PR holds each
    reach for 4-5 s and returns to neutral for only about 1 s, so the reach
    holds are both stiller and far longer than the neutral. That put the
    baseline 0.82 m from the neutral and collapsed the detection to a single
    peak. Restricting to the elevated part of the trial does not help, since
    the elevated part IS the reaches.

REACHES ARE MATCHED BY MEASURED DIRECTION, NOT ORDINAL POSITION
    The marker column identifies the second and third excursions by counting
    them, but the event names say "rightward" and "left", and direction is
    measurable, so each excursion is matched to the event whose name it fits
    and the observed order is recorded in the run log.

    On PR the two agree: forward at 7.6-12.1 s, rightward at 14.1-18.1 s,
    leftward at 19.6-24.1 s. That agreement was not obvious in advance --
    against OpenCap's per-frame pelvis heading the rightward reach appears
    leftward, because the pelvis rotates into the reach. See
    seg_common.ground_frame.

ONSET AND RETURN USE A LOWER THRESHOLD THAN DETECTION
    See RETURN_BAND below. The reach is detected with a high threshold so a
    transient cannot register as a reach, but the events themselves are
    bounded with a low one, because the movement starts before the wrist has
    travelled far and is not over until the wrist is back at neutral. The
    onset additionally walks back from the peak via
    seg_common.movement_onset rather than using any level crossing.

SUSTAINED MEANS SUSTAINED
    The excursion has to hold above threshold for MIN_HOLD_S. Without that
    the arms descending from the calibration raise register as a reach: on
    PR that transient reaches 0.447 m, just at the detection threshold, but
    lasts a fraction of a second against 4-5 s for every real reach.
"""

import numpy as np

import seg_common as sc

TASKS = ["frt"]

# m. Excursion of the pelvis-relative wrist midpoint above its neutral level
# that counts as a reach. Neutral is ~0.15 m and reaches are ~0.86 m on PR.
REACH_THRESHOLD = 0.300

# s. Minimum time above threshold for the excursion to be "sustained".
MIN_HOLD_S = 1.0

# m. Tight band around neutral used to bound the events, as opposed to
# REACH_THRESHOLD which only DETECTS the reach.
#
# Two different jobs need two different thresholds. Detection wants a high
# bar so a transient cannot register as a reach. Bounding the event wants a
# low bar, because the reach starts before the wrist has travelled 0.30 m and
# is not finished until the wrist is actually back at neutral. Using the
# detection threshold for both put the onset late and the completion of the
# return early: on PR, S1 at 7.200 s against a movement that starts at
# about 6.7 s, and S2 at 12.883 s against a return that completes at about
# 13.1 s.
RETURN_BAND = 0.080

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

        # Onset: walk back from the peak to where the wrist was not yet
        # moving, bounded by the previous reach's return.
        onset = sc.movement_onset(mag, t, pk, search_from=prev_return)

        # Return: the wrist is back when it is actually back at neutral, not
        # when it drops below the detection threshold.
        after = np.where(mag[pk:] <= return_level)[0]
        ret = int(pk + after[0]) if len(after) else len(mag) - 1

        # First movement OF the return: the onset construction run on the
        # negated magnitude, bounded by the maximum below and the completed
        # return above, so it cannot run back into the reach itself.
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
