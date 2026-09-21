"""
Sit-to-stand: 5tsts.
====================

Segmentation is done by the OpenCap team's sts_analysis class, imported
unmodified from opencap-processing/ActivityAnalyses. This file maps its
returned event lists onto the S1..S4 labels in the segmentation markers
document and repeats them per cycle.

EVENT MAPPING
    S1 forward lean onset   forwardLeanIdx / forwardLeanTime
    S2 start of rising      startRisingIdx / startRisingTime   (delayed +0.1 s)
    S3 end of rising        endRisingIdx   / endRisingTime
    S4 sitting              sittingIdx     / sittingTime
    repeated for all cycles, as the document specifies.

WHY THE CALIBRATION TRIM MATTERS HERE
    forwardLeanIdx searches backwards from lift-off for the last frame before
    the torso's angular velocity crosses -0.05 rad/s. For the first
    repetition the class sets temp_sit_ind = 0, so that backward window spans
    everything before the first stand -- including the calibration arm raise
    and the sit-down. trimming_start is what the parameter exists for, and it
    is passed here from the calibration drop-after-peak time, which is also
    what the segmentation markers document's header requires.

OPENSIM DEPENDENCY
    sts_analysis reads torso orientation and angular velocity in the ground
    frame via get_body_orientation / get_body_angular_velocity, so it needs
    the scaled .osim model, not just the .mot. That is why S1 is marked
    "requires OpenSim" in the document. The model resolves through
    sessionMetadata.yaml's openSimModel field.

    Note utilsKinematics hardcodes OpenSimData/Kinematics/<trial>.mot, so for
    the two OLD sessions it will load the older 34-column arm_flex layout
    rather than <trial>_shoulder.mot. That does not affect pelvis_ty or torso
    orientation, which is all this task reads, but it is worth knowing before
    reusing this module for a shoulder-dependent measure.
"""

# seg_common puts the opencap-processing checkout on sys.path, so it has to
# be imported before the OpenCap modules.
import seg_common as sc

from seg_sts_core import sts_analysis_leanfix as sts_analysis
from seg_sts_core import refine_end_rising

TASKS = ["5tsts"]

# The class's own default. Document: "pelvSignal = pelvis vertical position
# with trial minimum subtracted", filtered at this cutoff.
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

    # The class's four event lists are not guaranteed to be the same length.
    # A repetition the subject began but did not finish leaves a rise with no
    # matching sit, so sittingTime is shorter than startRisingIdx. Indexing
    # every list by the rise count then raises IndexError and the whole task
    # is discarded -- DCM_002 lost its 5tsts to "index 1 is out of bounds for
    # axis 0 with size 1". Each label is emitted only for the cycles that
    # have it, and the rest are named.
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
