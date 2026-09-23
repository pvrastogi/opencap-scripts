import numpy as np

# seg_common puts opencap-processing on sys.path; import it first
import seg_common as sc

from seg_walk_core import gait_analysis_variant, gait_analysis_lowprom

TASKS = ["gait", "gait-start", "gait-head-turn", "gait-tandem"]

LOWPASS = -1        # -1 = no filtering, OpenCap's default

SINGLE_STRIDE_TAG = " [single stride, no full cycle]"

EDGE_PAD_S = 0.25   # s. prominence dies within ~0.15s of a window start; pad, then discard

BOTH_LEGS = True


def _gait_class():
    """Which walking class to run."""
    return (gait_analysis_lowprom if sc.LOW_PROMINENCE_FIRST
            else gait_analysis_variant)


def _marker_names(leg):
    """OpenCap marker names for a side. Right is lowercase, left is capital."""
    p = "r" if leg == "r" else "L"
    return f"{p}_calc_study", f"{p}_toe_study"


CHAIN_FORWARD = (("ips_hs", "A", "HS"), ("cont_to", "C", "TO"),
                 ("cont_hs", "C", "HS"), ("ips_to", "A", "TO"))
CHAIN_BACKWARD = (("ips_hs", "A", "HS"), ("ips_to", "A", "TO"),
                  ("cont_hs", "C", "HS"), ("cont_to", "C", "TO"))


def cycles_from_events(ev):
    """OpenCap's gait-event arrays as a list of cycle dicts."""
    ips_t = np.atleast_2d(ev["ipsilateralTime"])
    cont_t = np.atleast_2d(ev["contralateralTime"])
    cycles = [{"ips_hs": ips_t[row, 0], "cont_to": cont_t[row, 0],
               "cont_hs": cont_t[row, 1], "ips_to": ips_t[row, 1],
               "ips_hs_close": ips_t[row, 2]}
              for row in np.argsort(ips_t[:, 0])]
    return cycles, ev["ipsilateralLeg"]


def window_peaks(label, task, t0, t1):
    """Every gait event the detector finds inside a walking window."""
    md = sc.load_trc(label, task)
    duration = float(md["time"][-1])
    for min_hs in (2, 1):
        try:
            trim_start, trim_end, prom, _ = sc.search_trim(
                md, sc.ORDER_FORWARD, base_start=t0,
                base_end=max(duration - t1, 0.0),
                max_extra_start=2.0, max_end=2.0, min_heel_strikes=min_hs)
            break
        except ValueError:
            trim_start = None
    if trim_start is None:
        return []

    win, time = sc._window(md, max(trim_start - EDGE_PAD_S, 0.0), trim_end)
    peaks = sc.detect_peaks(sc.walking_signals(win), time, prom)
    return [(float(time[i]), key[0], key[1:])
            for key, idx in peaks.items() for i in idx
            if time[i] >= trim_start]


def single_series(results, backward=False, extra=()):
    """Pool both anchor legs' cycles into one non-redundant series."""
    pool = {}
    for cycles, ips_leg, _ in results:
        cont_leg = "l" if ips_leg == "r" else "r"
        for c in cycles:
            for slot, t in c.items():
                if t is None:
                    continue
                leg = ips_leg if slot.startswith("ips") else cont_leg
                kind = "HS" if "hs" in slot else "TO"
                frame = round(float(t) * sc.SAMPLE_RATE)
                pool.setdefault((leg, kind), {})[frame] = float(t)
    for t, leg, kind in extra:
        pool.setdefault((leg, kind), {})[round(float(t) * sc.SAMPLE_RATE)] = \
            float(t)

    def times(leg, kind):
        return sorted(pool.get((leg, kind), {}).values())

    legs = [lg for lg in ("r", "l") if times(lg, "HS")]
    if not legs:
        return [], None, {}

    anchor = min(legs, key=lambda lg: times(lg, "HS")[0])
    cont = "l" if anchor == "r" else "r"
    chain = CHAIN_BACKWARD if backward else CHAIN_FORWARD
    seqs = {slot: times(anchor if who == "A" else cont, kind)
            for slot, who, kind in chain}

    hs = seqs["ips_hs"]
    out = []
    for i, h in enumerate(hs):
        nxt = hs[i + 1] if i + 1 < len(hs) else None
        cyc = {"ips_hs": h, "cont_to": None, "cont_hs": None,
               "ips_to": None, "ips_hs_close": nxt}
        prev = h
        for slot, _, _ in chain[1:]:
            t = next((x for x in seqs[slot]
                      if x > prev and (nxt is None or x < nxt)), None)
            if t is not None:
                cyc[slot], prev = t, t
        out.append(cyc)

    before = sorted(t for slot, _, _ in chain[1:]
                    for t in seqs[slot] if t < hs[0])
    info = {"anchor_leg": anchor, "n_cycles": len(out),
            "events_pooled": sum(len(v) for v in pool.values()),
            "anchor_legs_available": sorted(
                {leg for _, leg, _ in results}),
            "dropped_before_first_heel_strike":
                [round(t, 3) for t in before]}
    return out, anchor, info


def _emit_cycles(task, cycles, ips_leg):
    """Emit S1-S4 per cycle from one pooled series."""
    cont_leg = "l" if ips_leg == "r" else "r"
    ips_calc, ips_toe = _marker_names(ips_leg)
    cont_calc, cont_toe = _marker_names(cont_leg)
    defs = (
        ("S1", "ips_hs", "ipsilateral heel strike", ips_leg,
         f"peak in {ips_leg}_calc_rel_x ({ips_calc})"),
        ("S2", "cont_to", "contralateral toe-off", cont_leg,
         f"trough in {cont_leg}_toe_rel_x ({cont_toe})"),
        ("S3", "cont_hs", "contralateral heel strike", cont_leg,
         f"peak in {cont_leg}_calc_rel_x ({cont_calc})"),
        ("S4", "ips_to", "ipsilateral toe-off", ips_leg,
         f"trough in {ips_leg}_toe_rel_x ({ips_toe})"),
    )
    events = []
    for n, c in enumerate(cycles, start=1):
        for key, slot, name, side, definition in defs:
            if c[slot] is None:
                continue
            events.append(sc.event(task, key, name, c[slot], "trc",
                                   definition, cycle=n, side=side,
                                   anchor_leg=ips_leg))
    return events


def _single_stride(task, marker_dict, trim_start, trim_end, prom, pad=0.0):
    """Read S1..S4 off the peak detection for a trial with one stride."""
    win, time = sc._window(marker_dict, max(trim_start - pad, 0.0), trim_end)
    peaks = sc.detect_peaks(sc.walking_signals(win), time, prom)
    peaks = {k: [i for i in v if time[i] >= trim_start]
             for k, v in peaks.items()}
    rHS, lHS, rTO, lTO = (peaks["rHS"], peaks["lHS"],
                          peaks["rTO"], peaks["lTO"])

    if not (len(rHS) and len(lHS) and len(rTO) and len(lTO)):
        raise ValueError("single-stride fallback: missing an event on one side")

    ips_leg = "r" if time[rHS[0]] < time[lHS[0]] else "l"
    cont_leg = "l" if ips_leg == "r" else "r"
    hs = {"r": rHS, "l": lHS}
    to = {"r": rTO, "l": lTO}

    ips_calc, ips_toe = _marker_names(ips_leg)
    cont_calc, cont_toe = _marker_names(cont_leg)

    def first_after(indices, after_idx):
        later = [int(i) for i in indices if i > after_idx]
        return later[0] if later else None

    a_hs = int(hs[ips_leg][0])
    b_to = first_after(to[cont_leg], a_hs)
    c_hs = first_after(hs[cont_leg], b_to) if b_to is not None else None
    d_to = first_after(to[ips_leg], c_hs) if c_hs is not None else None

    if None in (b_to, c_hs, d_to):
        raise ValueError("single-stride fallback: chain incomplete after the "
                         "anchoring heel strike")

    events = [
        sc.event(task, "S1", "ipsilateral heel strike", time[a_hs], "trc",
                 f"peak in {ips_leg}_calc_rel_x ({ips_calc})",
                 cycle=1, side=ips_leg, anchor_leg=ips_leg),
        sc.event(task, "S2", "contralateral toe-off", time[b_to], "trc",
                 f"trough in {cont_leg}_toe_rel_x ({cont_toe})",
                 cycle=1, side=cont_leg, anchor_leg=ips_leg),
        sc.event(task, "S3", "contralateral heel strike", time[c_hs], "trc",
                 f"peak in {cont_leg}_calc_rel_x ({cont_calc})",
                 cycle=1, side=cont_leg, anchor_leg=ips_leg),
        sc.event(task, "S4", "ipsilateral toe-off", time[d_to], "trc",
                 f"trough in {ips_leg}_toe_rel_x ({ips_toe})",
                 cycle=1, side=ips_leg, anchor_leg=ips_leg),
    ]
    return events, ips_leg


def segment(label, task, n_gait_cycles=-1, leg="auto", verbose=True):
    """Run OpenCap's walking segmentation on one trial."""
    sd = sc.session_dir(label)
    tn = sc.trial_name(label, task)
    cal_start, trim_applied, cal_detail = sc.calibration_trim(label, task)
    md = sc.load_trc(label, task)

    single = False
    try:
        trim_start, trim_end, prom, trim_search = sc.search_trim(
            md, sc.ORDER_FORWARD, base_start=cal_start, min_heel_strikes=2)
    except ValueError:
        # No window contains a complete cycle. Accept a single stride.
        trim_start, trim_end, prom, trim_search = sc.search_trim(
            md, sc.ORDER_FORWARD, base_start=cal_start, min_heel_strikes=1)
        single = True

    if not single:
        try:
            ga = _gait_class()(
                sd, tn,
                leg=leg,
                lowpass_cutoff_frequency_for_coordinate_values=LOWPASS,
                n_gait_cycles=n_gait_cycles,
                trimming_start=trim_start,
                trimming_end=trim_end)
        except Exception as exc:
            if "Not enough gait cycles" not in str(exc):
                raise
            single = True

    if single:
        # the fallback picks its own anchor, so only one series
        events, ips_leg = _single_stride(task, md, trim_start, trim_end, prom)
        auto_leg, legs_used = ips_leg, [ips_leg]
        per_leg = {ips_leg: {"n_cycles": 0, "single_stride": True}}
        n_cycles = 0
    else:
        ev = ga.get_gait_events()
        auto_leg = ev["ipsilateralLeg"]
        legs = ["r", "l"] if (BOTH_LEGS and leg == "auto") else [auto_leg]
        results, per_leg = [], {}
        for lg in legs:
            if lg == auto_leg:
                # auto already resolved here; rebuilding would be identical
                ev_lg = ev
            else:
                try:
                    ev_lg = _gait_class()(
                        sd, tn,
                        leg=lg,
                        lowpass_cutoff_frequency_for_coordinate_values=LOWPASS,
                        n_gait_cycles=n_gait_cycles,
                        trimming_start=trim_start,
                        trimming_end=trim_end).get_gait_events()
                except Exception as exc:
                    per_leg[lg] = {"error": sc.exc_detail(exc)}
                    if verbose:
                        print(f"    anchor leg {lg}: FAILED -- {exc}")
                    continue
            cyc_lg, ips_leg = cycles_from_events(ev_lg)
            if ips_leg in per_leg:
                continue
            results.append((cyc_lg, ips_leg, ""))
            per_leg[ips_leg] = {
                "n_cycles": len(cyc_lg),
                "first_anchor_s": round(float(cyc_lg[0]["ips_hs"]), 3),
                "last_anchor_s": round(float(cyc_lg[-1]["ips_hs_close"]), 3)}

        # both anchor legs describe the same footfalls; pool them
        cycles, ips_leg, pool_info = single_series(results)
        events = _emit_cycles(task, cycles, ips_leg) if cycles else []
        legs_used = [ips_leg] if ips_leg else []
        n_cycles = len(cycles)
        per_leg["pooled"] = pool_info
        if verbose:
            for lg, v in per_leg.items():
                if lg != "pooled" and "n_cycles" in v:
                    print(f"    anchor leg {lg}: {v['n_cycles']} cycles, "
                          f"{v['first_anchor_s']:.2f}-{v['last_anchor_s']:.2f}s")
            print(f"    pooled -> anchor {ips_leg}, {n_cycles} cycles, "
                  f"{pool_info.get('events_pooled')} distinct events")

    meta = {
        "task": task, "trial_file": tn, "opencap_auto_leg": auto_leg,
        "legs_segmented": legs_used, "per_leg": per_leg,
        "n_cycles": n_cycles, "single_stride": single,
        "low_prominence_first": sc.LOW_PROMINENCE_FIRST,
        "trim_start_s": round(trim_start, 4), "trim_end_s": round(trim_end, 4),
        "prominence": prom, "calibration_trim_s": round(cal_start, 4),
        "trim_applied": trim_applied, "calibration_detail": cal_detail,
        "trim_search": trim_search,
    }

    if verbose:
        if single:
            print(f"  {task}: 1 stride, no complete cycle -- events read from "
                  f"peak detection; ipsilateral leg = {auto_leg}; "
                  f"{trim_search}")
        else:
            print(f"  {task}: {n_cycles} cycles across {len(legs_used)} anchor "
                  f"leg(s), OpenCap auto would pick '{auto_leg}'; "
                  f"{trim_search}")

    return events, meta


def cycles_in_window(label, task, t0, t1, leg="auto", min_heel_strikes=2):
    """Run OpenCap's walking segmentation over one straight window."""
    sd = sc.session_dir(label)
    tn = sc.trial_name(label, task)
    md = sc.load_trc(label, task)
    duration = float(md["time"][-1])

    trim_start, trim_end, prom, detail = sc.search_trim(
        md, sc.ORDER_FORWARD,
        base_start=t0, base_end=max(duration - t1, 0.0),
        max_extra_start=2.0, max_end=2.0,
        min_heel_strikes=min_heel_strikes)

    ga, exc = None, None
    for pad in (EDGE_PAD_S, 0.0):
        try:
            ga = _gait_class()(
                sd, tn,
                leg=leg,
                lowpass_cutoff_frequency_for_coordinate_values=LOWPASS,
                n_gait_cycles=-1,
                trimming_start=max(trim_start - pad, 0.0),
                trimming_end=trim_end)
            break
        except Exception as e:
            exc = e

    if ga is None:
        if "Not enough gait cycles" not in str(exc):
            raise exc
        stride, ips_leg = _single_stride(task, md, trim_start, trim_end, prom,
                                         pad=EDGE_PAD_S)
        by_label = {e["event"]: e["time_s"] for e in stride}
        cycle = {"ips_hs": by_label["S1"], "cont_to": by_label["S2"],
                 "cont_hs": by_label["S3"], "ips_to": by_label["S4"],
                 "ips_hs_close": None}
        return [cycle], ips_leg, detail + SINGLE_STRIDE_TAG

    cycles, ips_leg = cycles_from_events(ga.get_gait_events())
    cycles = [c for c in
              ({k: (None if v is not None and v < trim_start else v)
                for k, v in c.items()} for c in cycles)
              if any(v is not None for v in c.values())]
    return cycles, ips_leg, detail


def cycles_in_window_both(label, task, t0, t1, min_heel_strikes=2):
    """cycles_in_window for BOTH anchor legs, merged."""
    best = {}
    for leg in ("r", "l"):
        try:
            cycles, ips_leg, detail = cycles_in_window(
                label, task, t0, t1, leg=leg,
                min_heel_strikes=min_heel_strikes)
        except Exception:
            continue
        if not cycles:
            continue
        single = detail.endswith(SINGLE_STRIDE_TAG)
        prev = best.get(ips_leg)
        if prev is None or (prev[2] and not single):
            best[ips_leg] = (cycles, detail, single)
    return [(c, leg, d) for leg, (c, d, _) in sorted(best.items())]


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
