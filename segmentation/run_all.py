"""
Run the whole pipeline -- segmentation and every plot set -- on sessions.
========================================================================

    python3 run_all.py /path/to/folder_of_patient_sessions
    python3 run_all.py /path/to/one_session_OpenCapData_folder
    python3 run_all.py PR MG                 # labels already under Data/
    python3 run_all.py --in /path/to/folder DCM_001 DCM_003

Give it the PARENT folder and it finds every OpenCap session inside; give it
a single session folder and it runs just that one; give it labels and it
resolves them the usual way. Any mixture works.

--in ADDS A FOLDER TO THE LABEL SEARCH
    A bare label is looked up under Data/, which is where the controls are.
    --in <folder> adds another place to look, so a patient can be named by
    label instead of by its full <LABEL>_OpenCapData_<uuid> path. It is
    repeatable and can be mixed with labels that resolve the usual way:

        run_all.py --in "$PHI" DCM_001          one patient
        run_all.py --in "$PHI" DCM_001 DCM_003  several patients
        run_all.py PR                           one control
        run_all.py PR MG OLD_MG                 several controls
        run_all.py --in "$PHI" DCM_001 PR       any mixture

For each session it runs, in order:

    manifest.py                    what is on disk vs what we segment
    flag_spikes.py                 trials with physically impossible jumps
    run_segmentation.py            events_<label>.csv, run_log_<label>.txt,
                                   and the two annotated plot sets
    plot_opencap_trc_markers.py    marker_plots/     (.trc, unannotated)
    plot_opencap_all_coords.py     all_coords/       (.mot, unannotated)

Everything is written inside each session's own Outputs/ folder. Nothing is
written back into Data/.

WHY THIS EXISTS
    The same work as a four-line shell loop, minus the four ways that loop
    goes wrong: a $PY that is empty because the export ran in a different
    terminal, a DCM_SESSION_ROOTS that was never set, a label that was a
    placeholder rather than a real folder name, and a relative script path
    that only resolves from one working directory. This script uses its own
    interpreter and its own location for all four, so the only thing you
    have to get right is the folder you point it at.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)


def _script(name):
    """The plot scripts sit one folder up in the Data/Analysis layout and
    right here in the single-folder layout the SOP describes."""
    local = os.path.join(HERE, name)
    return local if os.path.exists(local) else os.path.join(ANALYSIS, name)

# (script path, label shown while it runs)
STEPS = [
    (os.path.join(HERE, "manifest.py"), "manifest"),
    # Straight after the manifest, so "what is in this session" and "which
    # of it looks glitchy" sit together, before any of it is segmented.
    (os.path.join(HERE, "flag_spikes.py"), "spike check"),
    (os.path.join(HERE, "run_segmentation.py"), "segmentation"),
    (_script("plot_opencap_trc_markers.py"), "marker plots"),
    (_script("plot_opencap_all_coords.py"), "coord plots"),
]


def is_session(path):
    return os.path.isdir(os.path.join(path, "MarkerData"))


def resolve(arg):
    """One argument -> [(label, session path)]."""
    if is_session(arg):
        path = os.path.abspath(arg)
        return [(os.path.basename(path).split("_OpenCapData")[0], path)]

    if os.path.isdir(arg):
        # A parent folder: every OpenCap session directly inside it.
        found = []
        for name in sorted(os.listdir(arg)):
            path = os.path.join(os.path.abspath(arg), name)
            if "_OpenCapData" in name and is_session(path):
                found.append((name.split("_OpenCapData")[0], path))
        if not found:
            print(f"[WARN] {arg} holds no *_OpenCapData_* session folders")
        return found

    # Not a path, so treat it as a label and let seg_common resolve it.
    import seg_common as sc
    return [(sc.short_label(arg), sc.session_dir(arg))]


def main():
    # --in <folder> is consumed here and added to seg_common's search roots;
    # everything else is a path or a label.
    args, roots, rest = [], [], list(sys.argv[1:])
    while rest:
        a = rest.pop(0)
        if a == "--in":
            if not rest:
                print("--in needs a folder")
                return 1
            roots.append(os.path.abspath(rest.pop(0)))
        else:
            args.append(a)

    if not args:
        print(__doc__)
        return 1

    if roots:
        import seg_common as sc
        sc.SESSION_ROOTS.extend(roots)

    sessions = []
    for arg in args:
        try:
            sessions += resolve(arg)
        except Exception as exc:
            print(f"[FAIL] {arg}: {type(exc).__name__}: {exc}")

    if not sessions:
        print("Nothing to run.")
        return 1

    print(f"{len(sessions)} session(s) to process:")
    for label, path in sessions:
        print(f"    {label:12s} {path}")
    print()

    failures = []
    for label, path in sessions:
        print("=" * 72)
        print(f"=== {label}")
        print("=" * 72)
        for script, name in STEPS:
            # Every step but the manifest spends its first ~30 s importing
            # numpy, OpenSim and the opencap modules before it prints a
            # thing, which reads as a hang. Say so rather than leave a
            # silent terminal.
            wait = ("" if name in ("manifest", "spike check")
                    else "  (~30 s of imports first)")
            print(f"\n--- {name} ---{wait}", flush=True)
            t0 = time.time()
            # The session PATH is passed, not the label, so nothing depends
            # on DCM_SESSION_ROOTS being set in this shell.
            r = subprocess.run([sys.executable, script, path])
            print(f"--- {name}: {time.time() - t0:.0f} s", flush=True)
            if r.returncode != 0:
                failures.append((label, name, r.returncode))
                print(f"[FAIL] {name} exited {r.returncode}")
        print()

    print("=" * 72)
    if failures:
        print(f"{len(failures)} step(s) failed:")
        for label, name, code in failures:
            print(f"    {label:12s} {name:14s} exit {code}")
    else:
        print(f"All steps completed for all {len(sessions)} session(s).")
    print("Outputs are in each session's Outputs/ folder.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
