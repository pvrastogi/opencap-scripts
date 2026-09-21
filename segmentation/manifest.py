"""
What is actually in a session folder, before segmenting anything.
=================================================================

    python3 manifest.py PR
    python3 manifest.py DCM_001
    python3 manifest.py /path/to/a_session_OpenCapData_folder

Sorts the trials on disk against the task names the segmentation modules
implement, into four groups:

    matched        a task we segment, with a trial file on disk
    missing        a task we segment, with no trial file
    out of scope   a trial file for a task we deliberately do not segment
    UNMATCHED      a trial file whose name is nothing we recognise

WHY THE LAST TWO GROUPS ARE SEPARATE
    run_segmentation.py cannot tell "missing" from "unmatched". Every
    module's run() calls sc.trial_exists(), and both cases return False and
    print the same "no trial in <label>, skipping". But they need opposite
    responses. A missing task is a task the subject did not attempt, and
    there is nothing to do about it. An unmatched trial is data we own and
    are discarding because the file is named something the code does not
    look for: it needs an entry in seg_common.SESSION_TRIAL_ALIASES, after
    which it segments normally.

WHAT IT READS
    Trial file names, and the .trc header line that carries NumFrames and
    DataRate. No marker coordinate or joint angle is read, so the output
    carries no kinematics. Durations are here because they are how you tell
    a half-done task from a whole one.
"""

import os
import sys

import seg_common as sc

# The static calibration pose. Every session has one and it is not a task.
NON_TASK_TRIALS = ["neutral"]

# Retakes the operator abandoned but left on disk under a suffix. Matched as
# substrings so both spellings seen so far -- "5tsts-deleted" and
# "tandem_delted" -- land here. They must not appear under UNMATCHED: that
# group is the one you have to act on, so anything already accounted for
# belongs elsewhere or it trains you to ignore the list.
DISCARDED_MARKERS = ["delet", "delt"]


def is_discarded(trial):
    low = trial.lower()
    return any(m in low for m in DISCARDED_MARKERS)


def trc_span(path):
    """"<n> fr <secs> s" from a .trc header, without reading the data.

    Line 3 of a .trc is the tab-separated header values, in the order named
    on line 2: DataRate, CameraRate, NumFrames, NumMarkers, Units, ...
    """
    try:
        with open(path) as fh:
            fh.readline()
            fh.readline()
            vals = fh.readline().strip().split("\t")
        rate, n = float(vals[0]), int(vals[2])
        return f"{n:5d} fr  {(n - 1) / rate:6.2f} s"
    except Exception as exc:
        return f"header unreadable: {type(exc).__name__}: {exc}"


def inspect(label):
    sd = sc.session_dir(label)
    marker_dir = os.path.join(sd, "MarkerData")
    kin_dir = os.path.join(sd, "OpenSimData", "Kinematics")

    print(f"=== {sc.short_label(label)}")
    print(f"folder: {sd}")
    if not os.path.isdir(marker_dir):
        print("\n  NO MarkerData FOLDER -- nothing can be segmented\n")
        return
    print(f"MarkerData: present"
          f"   Kinematics: {'present' if os.path.isdir(kin_dir) else 'MISSING'}")

    on_disk = sorted(f[:-4] for f in os.listdir(marker_dir)
                     if f.endswith(".trc"))
    mots = (sorted(f[:-4] for f in os.listdir(kin_dir) if f.endswith(".mot"))
            if os.path.isdir(kin_dir) else [])

    matched, missing, claimed = [], [], set()
    for task in sc.TASKS_IN_SCOPE:
        tn = sc.trial_name(label, task)
        if tn in on_disk:
            matched.append((task, tn))
            claimed.add(tn)
        else:
            missing.append(task)

    # Out-of-scope tasks get renamed by the alias table too -- OLD_GX's
    # dext-left is on disk as dexterity1 -- so they are matched through
    # trial_name as well, or they would land in UNMATCHED and read as a
    # naming problem when they are simply not in scope.
    oos_files = {sc.trial_name(label, t): t for t in sc.TASKS_OUT_OF_SCOPE}

    out_of_scope, non_task, discarded, unmatched = [], [], [], []
    for tn in on_disk:
        if tn in claimed:
            continue
        if tn in oos_files:
            task = oos_files[tn]
            out_of_scope.append(task if task == tn else f"{task} (file: {tn})")
        elif tn in NON_TASK_TRIALS:
            non_task.append(tn)
        elif is_discarded(tn):
            discarded.append(tn)
        else:
            unmatched.append(tn)

    print(f"\n--- matched ({len(matched)} of {len(sc.TASKS_IN_SCOPE)} tasks)")
    for task, tn in matched:
        # A task whose .trc is there but whose .mot is not will still fail in
        # the modules that read joint angles, so flag it here instead.
        has_mot = any(m in (tn, tn + "_shoulder") for m in mots)
        note = "" if has_mot else "   [no .mot]"
        alias = "" if tn == task else f"  (file: {tn})"
        print(f"  {task:16s} {trc_span(os.path.join(marker_dir, tn + '.trc'))}"
              f"{alias}{note}")

    print(f"\n--- missing, i.e. not attempted ({len(missing)})")
    print("  " + (", ".join(missing) if missing else "none"))

    print(f"\n--- out of scope, present but not segmented "
          f"({len(out_of_scope)})")
    print("  " + (", ".join(out_of_scope) if out_of_scope else "none"))

    if non_task:
        print(f"\n--- not a task ({len(non_task)})")
        print("  " + ", ".join(non_task))

    if discarded:
        print(f"\n--- discarded retakes, ignored by name ({len(discarded)})")
        print("  " + ", ".join(discarded))

    print(f"\n--- UNMATCHED, data we would silently discard ({len(unmatched)})")
    if not unmatched:
        print("  none")
    else:
        for tn in unmatched:
            print(f"  {tn:24s} "
                  f"{trc_span(os.path.join(marker_dir, tn + '.trc'))}")
        print("\n  Each of these is either a task above under a different "
              "name -- add it to\n  seg_common.SESSION_TRIAL_ALIASES -- or "
              "genuinely out of scope.")
    print()


def main():
    for label in (sys.argv[1:] or ["PR"]):
        try:
            inspect(label)
        except Exception as exc:
            print(f"=== {label}\n  FAILED -- {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
