"""
Standing with eyes closed: standing-ec.
=======================================

The only task in the document still defined on joint angles rather than
marker positions, per your decision ("standing-ec on joint angles is fine").

    S1 hands on hips   stable/plateau elbow flexion                    auto
    S2 eyes closure    eyes close in video                             manual
    S3 eyes reopen     eyes reopen in video, hands off hip via elbow
                       flexion drops                                   manual/auto

S2 is video-only and is not emitted. S3 has an automatable half -- the
elbow-flexion drop when the hands come off the hips -- and that is what is
emitted; the eyes-reopen instant itself still needs the video.

PROTOCOL DEVIATION THAT MATTERS HERE
    Only PR adopted the hands-on-hips posture (elbow flexion holding near
    93 degrees). MG, OLD_MG, NEW_GX and OLD_GX stood with arms at their
    sides, 12-29 degrees. For those four there is no hands-on-hips
    transition and no elbow-flexion drop to find, so S1 degrades to the
    onset of stable stance and S3 has no automatable component at all.

    Rather than emit a confidently wrong event, the module checks whether
    the elbow-flexion plateau is high enough to be a hands-on-hips posture.
    Below HANDS_ON_HIPS_MIN it reports hands_on_hips=False in the run log,
    labels S1 as the onset of the stable elbow-flexion plateau, and omits
    S3. The distinction is visible in the output rather than buried.
"""

import numpy as np

import seg_common as sc

TASKS = ["standing-ec"]

# deg. PR holds ~93; arms-at-sides sessions sit at 12-29. Anything above
# this is taken as a deliberate hands-on-hips posture.
HANDS_ON_HIPS_MIN = 60.0

# deg. Rolling-sd ceiling for "stable/plateau". PR's standing-ec has
# rolling sd p10 0.07, p50 0.17, p90 8.34 deg, so 1.0 sits well inside the
# quiet stance and well below the transitions.
PLATEAU_SD = 1.0
PLATEAU_MIN_S = 3.0

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
