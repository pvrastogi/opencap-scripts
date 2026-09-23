import os
import sys

import seg_common as sc

# The static calibration pose. Every session has one and it is not a task.
NON_TASK_TRIALS = ["neutral"]

DISCARDED_MARKERS = ["delet", "delt"]


def is_discarded(trial):
    low = trial.lower()
    return any(m in low for m in DISCARDED_MARKERS)


def trc_span(path):
    """"<n> fr <secs> s" from a .trc header, without reading the data."""
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
        # .trc without its .mot still fails later, so flag it here
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
