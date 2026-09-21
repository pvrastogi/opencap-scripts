"""
Arm raises in four planes: arms.
================================

    S1 first movement in parasagittal plane  first peak in both
                                             wrist-midpoint height after
                                             calibration
    S2 returned to neutral                   first return to baseline of both
                                             wrist midpoint
    S3/S4  scapular plane, second peak
    S5/S6  frontal plane, third peak
    S7/S8  backwards, fourth peak

The odd-numbered events are the ONSET of each raise, found by walking
backwards from the peak, per your correction. The marker column identifies
each raise by its peak, and the peak is what selects and orders the four
raises, but the event name is "first movement in <plane>" and that is what
is emitted. Each peak time is still recorded in the run log.

The walk-back is seg_common.movement_onset: anchor on the fastest ascent
between the previous return to neutral and the peak, then walk back to where
the wrist was not yet rising. A level crossing was tried first and lands
late by construction, since the arm has already travelled some way before it
clears any threshold.

FOUR PEAKS AFTER CALIBRATION
    Per your instruction: count out four after the calibration raise, and an
    extra one may just be a swing back forward. OLD_GX performed no
    calibration raise on any trial, which is already handled upstream by
    seg_common.calibration_trim.

    This is the task that exposed a bug in the calibration trim itself.
    seg_common.calibration_trim used to take the GLOBAL maximum of shoulder
    elevation, which is safe only where the calibration raise is the trial's
    one overhead movement. In the arms trial it is not: PR's largest
    elevation, 139.1 deg, belongs to the task's parasagittal raise, not to
    the calibration raise at 4.15 s. Trimming to the maximum therefore
    discarded the first task raise and shifted every plane label by one, so
    scapular, frontal and backward were reported as parasagittal, scapular
    and frontal. calibration_trim now takes the first qualifying peak.

    With that fixed, PR's arms trial has five peaks after the trim: the four
    task raises at 9.03, 11.07, 13.17 and 15.17 s, plus a smaller one at
    16.43 s which is reported as extra -- the swing back forward you
    described.

THE PLANE LABELS ARE STILL POSITIONAL
    This is the one task where the label cannot be verified from the signal
    used to detect it: wrist-midpoint HEIGHT is the same whether the arm
    goes up in front, in the scapular plane or out to the side. So the
    plane names rest entirely on the subject having performed them in the
    documented order.

    Rather than leave that unchecked, the horizontal position of the wrist
    midpoint during the ascent is measured in the fixed frame and reported in the
    run log as forward_m and right_m. A parasagittal raise should show a
    forward offset with little lateral, a frontal raise the reverse, the
    scapular plane in between, and the backward raise a negative forward
    offset. That does not rename anything -- it gives a rater the numbers to
    confirm or reject the assumed order.
"""

import numpy as np

import seg_common as sc

from scipy.signal import find_peaks

TASKS = ["arms"]

# m. PR's four raises lift the wrist midpoint 0.92 m, and the same four
# peaks are found anywhere in 0.10-0.20; 0.25 loses the fourth.
PEAK_PROMINENCE = 0.150

# m above the resting height that still counts as "returned to baseline".
#
# 0.100 was five times wider than it needed to be: the resting spread of the
# wrist midpoint (2nd to 25th percentile of its height) is 5-22 mm on the
# five controls, so a 100 mm band called the arms "back" while they were
# still a hand's width up. 0.050 keeps a deliberate buffer over the 22 mm
# worst case for a patient whose arms do not settle as still, and moves the
# controls' returns 0.03-0.15 s later -- always later, never earlier.
#
# The cost of tightening is that a raise the subject never fully lowers no
# longer gets a return event at all. That is reported rather than guessed:
# the event is omitted and named in missing_events.
BASELINE_PERCENTILE = 10
BASELINE_BAND = 0.050

N_EXPECTED = 4

PLANES = ["parasagittal plane", "scapular plane", "frontal plane",
          "backwards"]
ORDINALS = ["first", "second", "third", "fourth"]


def segment(label, task="arms", verbose=True):
    trim_start, trim_applied, trim_detail = sc.calibration_trim(label, task)
    md = sc.load_trc(label, task)
    win, t = sc._window(md, trim_start, 0.0)
    M = win["markers"]

    rw = (M["r_lwrist_study"] + M["r_mwrist_study"]) / 2
    lw = (M["L_lwrist_study"] + M["L_mwrist_study"]) / 2
    bw = (rw + lw) / 2
    height = bw[:, 1]

    base = float(np.percentile(height, BASELINE_PERCENTILE))
    peaks, _ = find_peaks(height, prominence=PEAK_PROMINENCE)

    # Horizontal position at each peak, for verifying the plane labels.
    # Lateral motion of the MIDPOINT cancels in a symmetric bilateral raise:
    # frontal-plane abduction moves the right wrist right and the left wrist
    # left, so the mean barely moves sideways. On PR the midpoint's lateral
    # offset was under 0.05 m for all four planes, which discriminates
    # nothing. The separation BETWEEN the two wrists does discriminate, and
    # strongly: PR measures 0.51 m parasagittal, 1.01 scapular, 1.40 frontal
    # and 0.63 backward against 0.48 m at rest.
    separation = np.abs((rw - lw) @ sc.ground_frame(win)[1])

    horiz_base = sc.dense_baseline(bw)
    d = bw - horiz_base
    h = np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2)
    quiet = h < np.percentile(h, 40)
    fwd_axis, right_axis = sc.ground_frame(win, quiet)

    kept = list(peaks[:N_EXPECTED])
    extra = list(peaks[N_EXPECTED:])

    events, details = [], []
    prev_return = 0
    for n, pk in enumerate(kept):
        plane = PLANES[n]
        ordinal = ORDINALS[n]
        _, ret = sc.excursion_bounds(height, int(pk), base, BASELINE_BAND)

        # Walk back from the peak to the start of the raise, bounded by the
        # previous return to neutral so the search cannot run into the
        # preceding raise.
        onset = sc.movement_onset(height, t, int(pk), search_from=prev_return)

        events.append(sc.event(
            task, f"S{2 * n + 1}", f"first movement in {plane}",
            t[onset], "trc",
            f"walking back from the {ordinal} peak in both wrist-midpoint "
            f"height after calibration to the start of the rise",
            cycle=n + 1))
        if ret is not None:
            events.append(sc.event(
                task, f"S{2 * n + 2}", "returned to neutral", t[ret], "trc",
                f"{ordinal} return to baseline of both wrist midpoint",
                cycle=n + 1))
        prev_return = ret if ret is not None else int(pk)

        # The plane is distinguishable during the ASCENT, not at the peak: at
        # full elevation the wrist is overhead whichever plane it travelled
        # through, which is why measuring at the peak gave a lateral offset
        # of about zero for all four of PR's raises. Sample instead at
        # half-height on the way up, and also report the largest lateral
        # offset reached during the ascent.
        a0 = onset if onset is not None else max(int(pk) - int(1.0 * 60), 0)
        rise = height[a0:int(pk) + 1]
        if len(rise) > 2:
            half = base + 0.5 * (height[pk] - base)
            crossings = np.where(rise >= half)[0]
            mid = a0 + int(crossings[0]) if len(crossings) else int(pk)
            sep_span = separation[a0:int(pk) + 1]
            max_sep = float(sep_span.max())
        else:
            mid, max_sep = int(pk), float(separation[pk])

        details.append({
            "label": f"S{2 * n + 1}", "plane": plane,
            "peak_s": round(float(t[pk]), 3),
            "onset_s": round(float(t[onset]), 3),
            "lead_over_peak_s": round(float(t[pk] - t[onset]), 3),
            "return_s": round(float(t[ret]), 3) if ret is not None else None,
            "height_above_base_m": round(float(height[pk] - base), 3),
            "at_half_height_s": round(float(t[mid]), 3),
            "forward_m": round(float(d[mid] @ fwd_axis), 3),
            "right_m": round(float(d[mid] @ right_axis), 3),
            "wrist_separation_at_half_m": round(float(separation[mid]), 3),
            "max_wrist_separation_m": round(max_sep, 3),
        })

    meta = {
        "task": task, "trial_file": sc.trial_name(label, task),
        "n_peaks_found": len(peaks), "n_used": len(kept),
        "extra_peaks_s": [round(float(t[p]), 3) for p in extra],
        "baseline_height_m": round(base, 4),
        "peaks": details,
        "plane_labels_are_positional": True,
        "trim_start_s": round(trim_start, 4), "trim_applied": trim_applied,
        "calibration_detail": trim_detail,
    }
    if len(kept) < N_EXPECTED:
        meta["warning"] = (f"only {len(kept)} peak(s) after calibration, "
                           f"expected {N_EXPECTED}")

    if verbose:
        print(f"  {task}: {len(peaks)} peak(s) after calibration, using "
              f"{len(kept)}"
              + (f" (extra at {meta['extra_peaks_s']})" if extra else ""))
        for dd in details:
            print(f"    {dd['label']} {dd['plane']:<20s} onset {dd['onset_s']:6.2f}s"
                  f" peak {dd['peak_s']:6.2f}s"
                  f"  at half-height: fwd {dd['forward_m']:+.3f} "
                  f"right {dd['right_m']:+.3f}"
                  f"  wrist sep {dd['wrist_separation_at_half_m']:.3f}")
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
