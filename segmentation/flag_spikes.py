"""Report trials with physically impossible marker jumps.

    python flag_spikes.py <session folder or label>
"""

import os
import sys

import numpy as np
from scipy.ndimage import median_filter

import seg_common as sc

THRESHOLD_M = 0.050          # flag a frame deviating more than this
MEDIAN_FRAMES = 5            # neighbourhood the deviation is measured against

# markers every event is defined on; all 63 would mostly be noise.
# Both wrist markers per side: seg_arms and seg_frt define their events on the
# lwrist/mwrist midpoint, so a spike in either one moves the event.
MARKERS = ["r_calc_study", "L_calc_study", "r_toe_study", "L_toe_study",
           "r.PSIS_study", "L.PSIS_study", "r.ASIS_study", "L.ASIS_study",
           "r_lwrist_study", "r_mwrist_study",
           "L_lwrist_study", "L_mwrist_study"]


def spikes_in_trial(md):
    """([(deviation_m, marker, time_s)], [absent marker]) for one trial."""
    time = md["time"]
    found, absent = [], []
    for name in MARKERS:
        p = md["markers"].get(name)
        if p is None:
            # a misspelt name used to vanish here without a word
            absent.append(name)
            continue
        if len(p) < MEDIAN_FRAMES:
            continue
        med = median_filter(p, size=(MEDIAN_FRAMES, 1), mode="nearest")
        dev = np.linalg.norm(p - med, axis=1)
        for i in np.flatnonzero(dev > THRESHOLD_M):
            found.append((float(dev[i]), name, float(time[i])))
    found.sort(reverse=True)
    return found, absent


def inspect(label):
    sd = sc.session_dir(label)
    marker_dir = os.path.join(sd, "MarkerData")
    print(f"=== {sc.short_label(label)}")
    if not os.path.isdir(marker_dir):
        print("  no MarkerData folder\n")
        return

    rows, never_checked = [], None
    for fname in sorted(os.listdir(marker_dir)):
        if not fname.endswith(".trc"):
            continue
        trial = fname[:-4]
        try:
            md = sc.load_trc(label, trial)
        except Exception as exc:
            print(f"  {trial:16s} unreadable: {type(exc).__name__}: {exc}")
            continue
        spikes, absent = spikes_in_trial(md)
        rows.append((trial, spikes))
        # same in every trial of a session, so report it once
        if never_checked is None:
            never_checked = absent

    if never_checked:
        print(f"  [WARNING] not in this session's .trc, never checked: "
              f"{', '.join(never_checked)}")

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
        # distinct moments, so several glitches don't read as one
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
