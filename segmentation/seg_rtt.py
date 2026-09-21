"""
Rise to toes: rtt.
==================

    S1 first sustained upward movement of heels   r_calc_study and
                                                  L_calc_study vertical
                                                  position begin sustained
                                                  rise                    auto
    S2 peak stable heel height reached            plateau in calcaneus
                                                  vertical position (mean
                                                  over plateau)           auto
    S3 heels return flat on floor                 calcaneus vertical
                                                  position returns to
                                                  baseline                auto

The document specifies the heel height of BOTH heels for S1 and then says
"calcaneus vertical position" for S2 and S3 without naming a side, so the
signal here is the mean of the two calcaneus heights. A rise to toes is a
bilateral movement; averaging keeps a single-frame dropout on one heel from
moving the event, and the per-side rise heights are reported separately in
the run log so an asymmetric rise is still visible.

SCALE
    The heel rise in this task is small: on PR the calcaneus lifts 0.057 m
    (right) and 0.066 m (left) off a 0.05 m baseline. That is an order of
    magnitude less than march, where the same marker lifts about 0.5 m. So
    the thresholds here are deliberately tight and are NOT shared with
    seg_march.

REPETITIONS
    The document lists S1-S3 once with no "repeat" row. Every rise found is
    emitted with its own cycle number, so a single hold comes out as cycle 1
    and a trial that turns out to contain several is reported rather than
    silently reduced to its first.
"""

import numpy as np

import seg_common as sc

from scipy.signal import find_peaks

TASKS = ["rtt"]

# m. Baseline is the 10th percentile of the mean heel height (floor
# contact). A heel is off the floor once it clears this much above it.
BASELINE_PERCENTILE = 10
BASELINE_BAND = 0.010

# m. Minimum lift to count as a deliberate rise rather than postural sway.
RISE_PROMINENCE = 0.020

# Plateau detection on the raised hold.
PLATEAU_SD = 0.006
PLATEAU_MIN_S = 0.5

MIN_RISE_DURATION = 0.5     # s, from S1 to S3


def segment(label, task="rtt", verbose=True):
    trim_start, trim_applied, trim_detail = sc.calibration_trim(label, task)
    md = sc.load_trc(label, task)
    win, t = sc._window(md, trim_start, 0.0)

    r = win["markers"]["r_calc_study"][:, 1]
    l = win["markers"]["L_calc_study"][:, 1]
    y = (r + l) / 2.0

    base = float(np.percentile(y, BASELINE_PERCENTILE))
    peaks, _ = find_peaks(y, prominence=RISE_PROMINENCE)

    events, rises = [], []
    for pk in peaks:
        onset, ret = sc.excursion_bounds(y, int(pk), base, BASELINE_BAND)
        if onset is None or ret is None:
            continue
        if t[ret] - t[onset] < MIN_RISE_DURATION:
            continue
        rises.append((onset, int(pk), ret))

    # ONE CYCLE PER EXCURSION, NOT PER PEAK
    # find_peaks can return two maxima inside a single heel rise whenever the
    # raised hold wobbles, and excursion_bounds then hands back the SAME
    # (onset, return) for both -- the same frame at which the heels left the
    # floor and the same frame at which they came back. That emitted the
    # whole cycle twice: two S1 at one instant, two S2, two S3, under cycle
    # numbers 1 and 2. Collapsing on the bounds keeps genuinely separate
    # rises apart, because those have different bounds by construction, and
    # keeps the highest peak within each one.
    by_bounds = {}
    for onset, pk, ret in rises:
        prev = by_bounds.get((onset, ret))
        if prev is None or y[pk] > y[prev[1]]:
            by_bounds[(onset, ret)] = (onset, pk, ret)
    rises = [by_bounds[k] for k in sorted(by_bounds)]

    for n, (onset, pk, ret) in enumerate(rises, start=1):
        # S2 is the plateau at the top, not the single highest frame: the
        # document asks for the "peak stable heel height ... (mean over
        # plateau)". Fall back to the peak frame if no plateau holds.
        seg_runs, _, _ = sc.steady_runs(t[onset:ret + 1], y[onset:ret + 1],
                                        window_s=0.3,
                                        sd_threshold=PLATEAU_SD,
                                        min_len_s=PLATEAU_MIN_S)
        high = [rr for rr in seg_runs if rr[2] > base + BASELINE_BAND]
        if high:
            # Arrival at the raised hold, so the start of the FIRST raised
            # plateau rather than the highest one.
            first_high = min(high, key=lambda rr: rr[0])
            last_high = max(high, key=lambda rr: rr[1])
            s2_time, plateau_level = first_high[0], first_high[2]
            plateau_span = [round(first_high[0], 3), round(last_high[1], 3)]
        else:
            s2_time, plateau_level = float(t[pk]), float(y[pk])
            plateau_span = None

        events.append(sc.event(
            task, "S1", "first sustained upward movement of heels",
            t[onset], "trc",
            "r_calc_study and L_calc_study vertical position begin "
            "sustained rise", cycle=n))
        events.append(sc.event(
            task, "S2", "peak stable heel height reached", s2_time, "trc",
            f"plateau in calcaneus vertical position (mean over plateau "
            f"{plateau_level:.3f} m)", cycle=n))
        events.append(sc.event(
            task, "S3", "heels return flat on floor", t[ret], "trc",
            "calcaneus vertical position returns to baseline", cycle=n))
        rises[n - 1] = (onset, pk, ret, plateau_level, plateau_span)

    meta = {
        "task": task, "trial_file": sc.trial_name(label, task),
        "n_rises": len(rises),
        "n_peaks_found": len(peaks),
        "baseline_m": round(base, 4),
        "rise_height_r_m": round(float(r.max() - np.percentile(r, 10)), 3),
        "rise_height_l_m": round(float(l.max() - np.percentile(l, 10)), 3),
        "plateau_levels_m": [round(x[3], 3) for x in rises],
        "plateau_spans_s": [x[4] for x in rises],
        "trim_start_s": round(trim_start, 4), "trim_applied": trim_applied,
        "calibration_detail": trim_detail,
    }
    if verbose:
        print(f"  {task}: {len(rises)} heel rise(s), baseline {base:.3f} m, "
              f"rise R {meta['rise_height_r_m']:.3f} / "
              f"L {meta['rise_height_l_m']:.3f} m")
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
