"""Run segmentation and every plot set on one or more sessions.

    python run_all.py <session folder> [more folders ...]
    python run_all.py <parent folder of sessions>
    python run_all.py --in <folder> LABEL [LABEL ...]
    python run_all.py PR MG                  (labels under Data/)

Outputs go to each session's own Outputs/ folder.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)


def _script(name):
    """The plot scripts sit one folder up in the Data/Analysis layout and"""
    local = os.path.join(HERE, name)
    return local if os.path.exists(local) else os.path.join(ANALYSIS, name)

# (script path, label shown while it runs)
STEPS = [
    (os.path.join(HERE, "manifest.py"), "manifest"),
    # right after the manifest: what is here, and what looks glitchy
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
    # --in <folder> adds a label search root; the rest are paths or labels
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
            wait = ("" if name in ("manifest", "spike check")
                    else "  (~30 s of imports first)")
            print(f"\n--- {name} ---{wait}", flush=True)
            t0 = time.time()
            # pass the PATH, so nothing depends on DCM_SESSION_ROOTS here
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
