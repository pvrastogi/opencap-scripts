"""
Backward walking group: gait-bckw, gait-bckw-3s.
================================================

Segmentation comes from seg_walk_core, which holds one verbatim copy of
OpenCap's segment_walking with the event chain and the prominence ladder
made overridable. Backward walking uses the chain rTO -> lHS -> lTO -> rHS,
the forward chain with the heel-strike and toe-off labels exchanged.

The detectors do not change: a peak in calc_rel_x still fills the "HS" slot
and a trough in toe_rel_x the "TO" slot. What changes is what those slots
mean. Walking backwards the planted foot drifts anteriorly relative to the
pelvis, so a calcaneus peak is heel-OFF and a toe trough is toe-STRIKE.

EVENT MAPPING
    From the ipsilateral heel-off anchor the chain runs
        ipsiHS -> ipsiTO -> contraHS -> contraTO -> ipsiHS
    giving the document's S1..S4:
        S1 ipsilateral toe-strike    trough in r/l_toe_rel_x  ipsilateralTime[:,1]
        S2 contralateral heel-off    peak in l/r_calc_rel_x   contralateralTime[:,1]
        S3 contralateral toe-strike  trough in l/r_toe_rel_x  contralateralTime[:,0]
        S4 ipsilateral heel-off      peak in r/l_calc_rel_x   ipsilateralTime[:,2]
    The trial's first ipsilateral heel-off (ipsilateralTime[0,0]) anchors
    cycle 1 but belongs to the stride before it, so it is emitted once as the
    S4 of cycle 0 rather than dropped.

WHY BOTH LEGS ARE SEGMENTED
    OpenCap's leg='auto' picks the leg whose FINAL heel strike is latest,
    because segment_walking assembles cycles backwards from the end of the
    trial (hsIps[-i-2] -> hsIps[-i-1]). That choice guarantees the last cycle
    closes on a real event instead of running off the end of the trial. It
    optimises the tail and is blind to the head.

    On PR's gait-bckw the consequence is a lost stride. Right-anchored
    cycles run [9.05, 10.62, 12.28] and left-anchored ones [8.40, 9.88,
    11.48] -- two cycles either way, but different coverage. auto picks right
    (12.28 > 11.48), so the first backward stride, the one whose left heel-off
    is at 8.40 s, belongs to no cycle and is never emitted. Forcing leg='l'
    would recover it and lose the stride ending at 12.28 instead.

    So both legs are segmented and the results POOLED into one series, which
    covers 8.40 and 12.28 without discarding either. The two series overlap
    in time by construction, so emitting both put one instant in the table
    twice under two S numbers; seg_gait.single_series instead pools them
    into one set of typed events keyed by frame and lays a single cycle
    series back over them. anchor_leg records which leg that series is
    anchored on. leg='auto' is still run and still reported, as the leg
    OpenCap would have chosen on its own.
"""

import numpy as np

import seg_common as sc
import seg_gait

from seg_walk_core import (gait_analysis_backward,
                           gait_analysis_backward_lowprom)

TASKS = ["gait-bckw", "gait-bckw-3s"]

LOWPASS = -1

# Segment from both legs and pool into one series, rather than taking only
# leg='auto'. See seg_gait.single_series.
BOTH_LEGS = True


def _bckw_class():
    return (gait_analysis_backward_lowprom if sc.LOW_PROMINENCE_FIRST
            else gait_analysis_backward)


def _marker_names(leg):
    p = "r" if leg == "r" else "L"
    return f"{p}_calc_study", f"{p}_toe_study"


def _emit(task, cycles, ips_leg):
    cont_leg = "l" if ips_leg == "r" else "r"
    ips_calc, ips_toe = _marker_names(ips_leg)
    cont_calc, cont_toe = _marker_names(cont_leg)
    defs = (
        ("S1", "ips_to", "ipsilateral toe-strike", ips_leg,
         f"trough in {ips_leg}_toe_rel_x ({ips_toe})"),
        ("S2", "cont_hs", "contralateral heel-off", cont_leg,
         f"peak in {cont_leg}_calc_rel_x ({cont_calc})"),
        ("S3", "cont_to", "contralateral toe-strike", cont_leg,
         f"trough in {cont_leg}_toe_rel_x ({cont_toe})"),
        # The closing heel-off, not the opening one: cycle n's closing
        # anchor is cycle n+1's opening anchor, so emitting it from
        # ips_hs_close is what keeps each instant to a single row.
        ("S4", "ips_hs_close", "ipsilateral heel-off", ips_leg,
         f"peak in {ips_leg}_calc_rel_x ({ips_calc})"),
    )

    events = []

    # The first cycle's opening heel-off belongs to the stride before it, so
    # it is emitted as the S4 of cycle 0 rather than dropped -- but only when
    # it plausibly belongs to the same walking bout. In gait-bckw the subject
    # walks forward before turning around and going backward, and a
    # calcaneus peak from the forward phase can end up as that opening
    # anchor: on PR the right-anchored series opens at 6.483 s while its own
    # first interior event is at 8.15 s, a 1.67 s gap across the turnaround,
    # against a stride span of 0.90 s. Requiring the gap to be no larger
    # than the cycle's own span drops that one and keeps the left-anchored
    # series' opener at 8.40 s (gap 0.45 s, span 1.03 s).
    first = cycles[0]
    if first["ips_to"] is not None and first["ips_hs_close"] is not None:
        lead_gap = float(first["ips_to"] - first["ips_hs"])
        cycle_span = float(first["ips_hs_close"] - first["ips_to"])
        if lead_gap <= cycle_span:
            events.append(sc.event(
                task, "S4", "ipsilateral heel-off", first["ips_hs"], "trc",
                f"peak in {ips_leg}_calc_rel_x ({ips_calc})",
                cycle=0, side=ips_leg, anchor_leg=ips_leg))

    for n, c in enumerate(cycles, start=1):
        for key, slot, name, side, definition in defs:
            if c[slot] is None:
                continue
            events.append(sc.event(task, key, name, c[slot], "trc",
                                   definition, cycle=n, side=side,
                                   anchor_leg=ips_leg))
    return events


def _run_one_leg(session_dir, trial, leg, trim_start, trim_end):
    ga = _bckw_class()(
        session_dir, trial,
        leg=leg,
        lowpass_cutoff_frequency_for_coordinate_values=LOWPASS,
        n_gait_cycles=-1,
        trimming_start=trim_start,
        trimming_end=trim_end)
    ev = ga.get_gait_events()
    return (np.atleast_2d(ev["ipsilateralTime"]),
            np.atleast_2d(ev["contralateralTime"]),
            ev["ipsilateralLeg"])


def segment(label, task, verbose=True):
    """Run the backward-order walking segmentation on one trial."""
    sd = sc.session_dir(label)
    tn = sc.trial_name(label, task)
    cal_start, trim_applied, cal_detail = sc.calibration_trim(label, task)
    md = sc.load_trc(label, task)

    # The standing-to-walking transition at the start of a backward trial
    # leaves an unpaired heel strike and a missing event, which breaks the
    # gate before steady-state walking begins. The search finds where the
    # chain becomes clean.
    trim_start, trim_end, prom, trim_search = sc.search_trim(
        md, sc.ORDER_BACKWARD, base_start=cal_start, min_heel_strikes=2)

    _, _, auto_leg = _run_one_leg(sd, tn, "auto", trim_start, trim_end)

    legs = ["r", "l"] if BOTH_LEGS else [auto_leg]
    results, per_leg = [], {}
    for leg in legs:
        try:
            ips_t, cont_t, ips_leg = _run_one_leg(sd, tn, leg,
                                                  trim_start, trim_end)
        except Exception as exc:
            per_leg[leg] = {"error": sc.exc_detail(exc)}
            if verbose:
                print(f"    anchor leg {leg}: FAILED -- {exc}")
            continue
        if ips_leg in per_leg:
            continue
        cyc = [{"ips_hs": ips_t[r, 0], "cont_to": cont_t[r, 0],
                "cont_hs": cont_t[r, 1], "ips_to": ips_t[r, 1],
                "ips_hs_close": ips_t[r, 2]}
               for r in np.argsort(ips_t[:, 0])]
        results.append((cyc, ips_leg, ""))
        per_leg[ips_leg] = {"n_cycles": len(cyc),
                            "first_anchor_s": round(float(ips_t[:, 0].min()), 3),
                            "last_anchor_s": round(float(ips_t[:, 2].max()), 3)}
        if verbose:
            print(f"    anchor leg {ips_leg}: {ips_t.shape[0]} cycles, "
                  f"{ips_t[:, 0].min():.2f}-{ips_t[:, 2].max():.2f}s")

    # Pool the two anchor legs into one series; see seg_gait.single_series.
    cycles, ips_leg, pool_info = seg_gait.single_series(results, backward=True)
    events = _emit(task, cycles, ips_leg) if cycles else []
    per_leg["pooled"] = pool_info
    if verbose and cycles:
        print(f"    pooled -> anchor {ips_leg}, {len(cycles)} cycles, "
              f"{pool_info.get('events_pooled')} distinct events")

    meta = {
        "task": task, "trial_file": tn,
        "opencap_auto_leg": auto_leg, "legs_segmented": legs,
        "per_leg": per_leg,
        "n_cycles": len(cycles),
        "trim_start_s": round(trim_start, 4), "trim_end_s": round(trim_end, 4),
        "prominence": prom, "low_prominence_first": sc.LOW_PROMINENCE_FIRST,
        "calibration_trim_s": round(cal_start, 4),
        "trim_applied": trim_applied, "calibration_detail": cal_detail,
        "trim_search": trim_search,
    }
    if verbose:
        print(f"  {task}: {meta['n_cycles']} cycles across "
              f"{len(legs)} anchor leg(s), OpenCap auto would pick "
              f"'{auto_leg}'; {trim_search}")
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
