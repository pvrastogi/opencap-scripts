import numpy as np

import seg_common as sc
import seg_gait

from seg_walk_core import (gait_analysis_backward,
                           gait_analysis_backward_lowprom)

TASKS = ["gait-bckw", "gait-bckw-3s"]

LOWPASS = -1   # -1 = no filtering, OpenCap's default

# pool both legs rather than taking only leg='auto'
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
        ("S4", "ips_hs_close", "ipsilateral heel-off", ips_leg,
         f"peak in {ips_leg}_calc_rel_x ({ips_calc})"),
    )

    events = []

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
