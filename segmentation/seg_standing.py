import numpy as np

import seg_common as sc

TASKS = ["standing-ec"]

# deg. PR holds ~93, arms-at-sides sessions 12-29
HANDS_ON_HIPS_MIN = 60.0

PLATEAU_SD = 1.0      # deg. elbow flexion holds far steadier than this while standing
PLATEAU_MIN_S = 3.0   # s. the stance is held ~30s, so this is a safe floor

# deg below the plateau level that counts as the hands having come off.
DROP_MARGIN = 15.0


def segment(label, task="standing-ec", verbose=True):
    trim_start, trim_applied, trim_detail = sc.calibration_trim(label, task)
    time, cols, mot_file = sc.load_mot(label, task)

    mask = time >= trim_start
    t = time[mask]
    if "elbow_flex_r" not in cols:
        raise ValueError(f"{mot_file} has no elbow_flex columns")
    elbow = np.maximum(cols["elbow_flex_r"][mask], cols["elbow_flex_l"][mask])

    runs, sd, sd_thr = sc.steady_runs(t, elbow, window_s=1.0,
                                      sd_threshold=PLATEAU_SD,
                                      min_len_s=PLATEAU_MIN_S)
    if not runs:
        raise ValueError("no stable elbow-flexion plateau found")

    # The stance plateau is the longest one.
    start_s, end_s, level, i0, i1 = max(runs, key=lambda r: r[1] - r[0])
    hands_on_hips = level >= HANDS_ON_HIPS_MIN

    events = [sc.event(
        task, "S1",
        "hands on hips" if hands_on_hips else "onset of stable stance",
        start_s, "mot",
        f"stable/plateau elbow flexion (level {level:.1f} deg)")]

    drop_time = None
    if hands_on_hips:
        # First frame after the plateau where flexion has fallen clear of it.
        after = np.where(elbow[i1:] < level - DROP_MARGIN)[0]
        if len(after):
            drop_time = float(t[i1 + after[0]])
            events.append(sc.event(
                task, "S3", "eyes reopen / hands off hip", drop_time, "mot",
                f"elbow flexion drops below {level - DROP_MARGIN:.1f} deg "
                f"(video needed for the eyes-reopen instant)"))

    meta = {
        "task": task, "trial_file": sc.trial_name(label, task),
        "mot_file": mot_file,
        "hands_on_hips": bool(hands_on_hips),
        "plateau_level_deg": round(level, 1),
        "plateau_span_s": [round(start_s, 3), round(end_s, 3)],
        "plateau_duration_s": round(end_s - start_s, 2),
        "hands_off_s": round(drop_time, 3) if drop_time else None,
        "n_plateaus": len(runs),
        "trim_start_s": round(trim_start, 4), "trim_applied": trim_applied,
        "calibration_detail": trim_detail,
        "skipped_manual": ["S2 eyes closure"] + ([] if hands_on_hips else
                                                 ["S3 eyes reopen"]),
    }
    if verbose:
        posture = ("hands on hips" if hands_on_hips
                   else "arms at sides, no hands-on-hips transition")
        print(f"  {task}: {posture}; plateau {start_s:.2f}-{end_s:.2f}s at "
              f"{level:.1f} deg"
              + (f", hands off at {drop_time:.2f}s" if drop_time else ""))
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
