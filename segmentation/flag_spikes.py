"""
Flag physically impossible marker jumps, so a glitchy trial is obvious.
======================================================================

    python3 flag_spikes.py PR
    python3 flag_spikes.py /path/to/folder_of_sessions
    python3 flag_spikes.py DCM_001 DCM_002 DCM_003

Reports only. Nothing is filtered and no file is modified -- the point is to
tell you which trials to go and look at, not to decide for you.

WHAT A SPIKE IS
    One or two frames where a marker jumps a long way and comes straight
    back. Not noise in the ordinary sense: physically impossible motion. The
    markerless pipeline mis-detects a keypoint in a single video frame and
    that error propagates through triangulation into the 3D marker, so the
    heel appears to move half a metre in 33 ms and return.

HOW IT IS MEASURED
    Distance of each frame from the median of its own 5-frame neighbourhood.
    A median ignores one or two bad frames out of five, so a genuine fast
    movement -- which is smooth across neighbours -- scores near zero while
    an isolated jump scores its full size.

WHY THE THRESHOLD IS 50 mm
    Measured across control trials, that deviation has a median of 1.2 mm
    and a 99th percentile of 9.0 mm. The largest value from real movement
    seen so far is 27.8 mm (MG gait-tandem). Known glitches score 104 mm
    (PR gait, r_calc at 8.90 s), 118 mm (PR march, r_toe at 9.77 s), 308 mm
    (NEW_GX tug, r_calc at 14.92 s -- the one that invented a heel strike)
    and 476 mm (NEW_GX tug, L_calc at 14.48 s -- the one that breaks the
    order gate). 50 mm sits in the empty gap between those two populations.
"""

import os
import sys

import numpy as np
from scipy.ndimage import median_filter

import seg_common as sc

THRESHOLD_M = 0.050          # flag a frame deviating more than this
MEDIAN_FRAMES = 5            # neighbourhood the deviation is measured against

# Markers every gait, balance and reach event is defined on. Scanning all 63
# would mostly report markers nothing depends on.
MARKERS = ["r_calc_study", "L_calc_study", "r_toe_study", "L_toe_study",
           "r.PSIS_study", "L.PSIS_study", "r.ASIS_study", "L.ASIS_study",
           "r_wrist_study", "L_wrist_study"]


def spikes_in_trial(md):
    """[(deviation_m, marker, time_s)] for every frame over the threshold."""
    time = md["time"]
    found = []
    for name in MARKERS:
        p = md["markers"].get(name)
        if p is None or len(p) < MEDIAN_FRAMES:
            continue
        med = median_filter(p, size=(MEDIAN_FRAMES, 1), mode="nearest")
        dev = np.linalg.norm(p - med, axis=1)
        for i in np.flatnonzero(dev > THRESHOLD_M):
            found.append((float(dev[i]), name, float(time[i])))
    found.sort(reverse=True)
    return found


def inspect(label):
    sd = sc.session_dir(label)
    marker_dir = os.path.join(sd, "MarkerData")
    print(f"=== {sc.short_label(label)}")
    if not os.path.isdir(marker_dir):
        print("  no MarkerData folder\n")
        return

    rows = []
    for fname in sorted(os.listdir(marker_dir)):
        if not fname.endswith(".trc"):
            continue
        trial = fname[:-4]
        try:
            md = sc.load_trc(label, trial)
        except Exception as exc:
            print(f"  {trial:16s} unreadable: {type(exc).__name__}: {exc}")
            continue
        rows.append((trial, spikes_in_trial(md)))

    flagged = [(t, s) for t, s in rows if s]
    if not flagged:
        print(f"  clean: no marker frame deviates more than "
              f"{THRESHOLD_M * 1000:.0f} mm in any of {len(rows)} trials\n")
        return

    flagged.sort(key=lambda r: -r[1][0][0])
    print(f"  {len(flagged)} of {len(rows)} trials have at least one jump "
          f"over {THRESHOLD_M * 1000:.0f} mm\n")
    print(f"  {'trial':16s} {'frames':>6s}  {'worst':>9s}  "
          f"{'marker':16s} {'at':>8s}")
    for trial, sp in flagged:
        worst, marker, t = sp[0]
        print(f"  {trial:16s} {len(sp):6d}  {worst * 1000:6.0f} mm  "
              f"{marker:16s} {t:7.2f}s")
        # The rest of the distinct moments, so a trial with several separate
        # glitches does not look like one.
        seen = {round(t, 1)}
        for dev, m, tt in sp[1:]:
            if round(tt, 1) in seen:
                continue
            seen.add(round(tt, 1))
            if len(seen) > 4:
                print(f"  {'':16s} {'':6s}  ... and more")
                break
            print(f"  {'':16s} {'':6s}  {dev * 1000:6.0f} mm  "
                  f"{m:16s} {tt:7.2f}s")
    print()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    labels = []
    for a in args:
        if os.path.isdir(a) and not os.path.isdir(os.path.join(a, "MarkerData")):
            labels += [os.path.join(a, d) for d in sorted(os.listdir(a))
                       if os.path.isdir(os.path.join(a, d, "MarkerData"))]
        else:
            labels.append(a)
    for label in labels:
        try:
            inspect(label)
        except Exception as exc:
            print(f"=== {label}\n  FAILED -- {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
