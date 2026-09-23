import numpy as np

import seg_common as sc

from scipy.signal import find_peaks

TASKS = ["arms"]

# m. raises lift 0.92m; 0.10-0.20 all find four peaks, 0.25 loses one
PEAK_PROMINENCE = 0.150

BASELINE_PERCENTILE = 10     # resting wrist height; robust to a trial starting mid-raise
BASELINE_BAND = 0.050        # m. resting spread is 5-22mm on controls, so this has buffer

N_EXPECTED = 4               # four planes: parasagittal, scapular, frontal, backwards

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
