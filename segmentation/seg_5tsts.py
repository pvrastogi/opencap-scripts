# seg_common puts opencap-processing on sys.path; import it first
import seg_common as sc

from seg_sts_core import sts_analysis_leanfix as sts_analysis
from seg_sts_core import refine_end_rising

TASKS = ["5tsts"]

# Hz. the class's own default
LOWPASS = 6

DEFS = {
    "S1": ("forward lean onset",
           "forwardLeanIdx: searching back from lift-off, last frame before "
           "torso_z angular velocity crosses -0.05 rad/s"),
    "S2": ("start of rising",
           "startRisingIdx: pelvis_ty velocity peak (>= 0.2 m/s, largest), "
           "walk back to < 0.1 m/s, plus 0.1 s delay"),
    "S3": ("end of rising",
           "endRisingIdx: forward from the velocity peak to where velocity "
           "< 0.1 m/s, then forward again to full knee extension -- the "
           "first frame within 3 deg of the minimum of max(knee_angle_r, "
           "knee_angle_l) before the knee re-flexes by 10 deg"),
    "S4": ("sitting",
           "sittingIdx: descent instant where pelvis returns within 5 cm of "
           "the height at delayed start"),
}


def segment(label, task="5tsts", n_sts_cycles=-1, verbose=True):
    sd = sc.session_dir(label)
    tn = sc.trial_name(label, task)
    trim_start, trim_applied, trim_detail = sc.calibration_trim(label, task)

    sts = sts_analysis(
        sd, tn,
        lowpass_cutoff_frequency_for_coordinate_values=LOWPASS,
        n_sts_cycles=n_sts_cycles,
        trimming_start=trim_start,
        trimming_end=0)

    ev = sts.stsEvents
    end_rising_detail = refine_end_rising(sts)
    n = len(ev["startRisingIdx"])

    events, missing = [], []
    for cycle in range(n):
        for key, time_key in (("S1", "forwardLeanTime"),
                              ("S2", "startRisingTime"),
                              ("S3", "endRisingTime"),
                              ("S4", "sittingTime")):
            times = ev.get(time_key, [])
            if cycle >= len(times):
                missing.append(f"{key} cycle {cycle + 1}")
                continue
            name, definition = DEFS[key]
            events.append(sc.event(task, key, name, times[cycle],
                                   "mot+model", definition, cycle=cycle + 1))

    meta = {
        "task": task, "trial_file": tn, "n_cycles": n,
        "end_of_rising": end_rising_detail,
        "partial": bool(missing), "missing_events": missing,
        "trim_start_s": round(trim_start, 4), "trim_applied": trim_applied,
        "calibration_detail": trim_detail,
    }
    if verbose:
        print(f"  {task}: {n} sit-to-stand cycles, trim {trim_start:.3f}s "
              f"({trim_detail})")
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
