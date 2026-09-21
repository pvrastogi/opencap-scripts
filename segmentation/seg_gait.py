"""
Forward walking group: gait, gait-start, gait-head-turn, gait-tandem.
=====================================================================

Segmentation is done by the OpenCap team's gait_analysis class, imported
unmodified from opencap-processing/ActivityAnalyses. This file maps the
class's returned event arrays onto the S1..Sn labels in the segmentation
markers document, and wraps two of the class's own documented escape hatches.

EVENT MAPPING
    gait_analysis.segment_walking returns, per gait cycle:
        ipsilateralTime   -> [HS, TO, HS]
        contralateralTime -> [TO, HS]
    and the order gate it enforces is rHS -> lTO -> lHS -> rTO -> rHS. So
    within one cycle, anchored on the ipsilateral heel strike, events run
    ipsiHS, contraTO, contraHS, ipsiTO -- exactly the document's S1..S4:
        S1 ipsilateral heel strike   peak in r/l_calc_rel_x  ipsilateralTime[:,0]
        S2 contralateral toe-off     trough in l/r_toe_rel_x contralateralTime[:,0]
        S3 contralateral heel strike peak in l/r_calc_rel_x  contralateralTime[:,1]
        S4 ipsilateral toe-off       trough in r/l_toe_rel_x ipsilateralTime[:,1]
    The cycle's closing ipsilateral heel strike (ipsilateralTime[:,2]) is the
    next cycle's S1; it is emitted once, as the S1 of a trailing cycle.

IPSILATERAL LEG
    leg='auto' is left exactly as OpenCap wrote it: the leg whose final heel
    strike comes later. It is recorded per trial in the output rather than
    forced to a side, so a left-led trial is reported as left-led.

CYCLE ORDER
    segment_walking assembles cycles backwards from the end of the trial, so
    its rows arrive in reverse chronological order. Cycles are renumbered
    forwards here, so cycle 1 is the earliest stride.

PROMINENCE LADDER
    OpenCap's ladder is [0.3, 0.25, 0.2] walked strictest-first, descending
    only when the order gate fails, so it settles on the strictest passing
    value. With seg_common.LOW_PROMINENCE_FIRST set it is walked from the
    other end and settles on the lowest passing value, which recovers genuine
    events that are merely less prominent than 0.3 m. The prominence actually
    used is recorded per trial in the run log.

TRIM SEARCH
    When the order gate fails at every prominence, segment_walking raises
    ValueError and tells you to trim the trial. That happens when the subject
    takes an extra unrelated step after the walk (PR's gait-tandem has a 2 s
    pause and then a stray step, which puts rTO before lHS). Rather than
    hardcode a per-trial constant, seg_common.search_trim finds the smallest
    (trimming_start, trimming_end) that clears the gate and leaves a complete
    cycle. Trimming the least keeps the most cycles.

SINGLE-STRIDE TRIALS
    A gait cycle needs two consecutive ipsilateral heel strikes. A gait
    initiation trial can contain only one per foot, in which case the class
    raises 'Not enough gait cycles found.' even though all four events are
    present and correctly ordered. In that case the four events are read
    straight off the peak detection instead of the cycle assembly, using the
    same signals, the same prominence ladder and the same order gate. This is
    the one place in this file where events do not come out of OpenCap's own
    assembly step, and it is reported as such in the run log.
"""

import numpy as np

# seg_common puts the opencap-processing checkout on sys.path, so it has to
# be imported before the OpenCap modules.
import seg_common as sc

from seg_walk_core import gait_analysis_variant, gait_analysis_lowprom

TASKS = ["gait", "gait-start", "gait-head-turn", "gait-tandem"]

LOWPASS = -1

SINGLE_STRIDE_TAG = " [single stride, no full cycle]"

# Lead-in given to peak detection at the start of a walking window, and
# discarded again afterwards. find_peaks measures prominence against the
# troughs either side of a peak, so a heel strike landing within ~0.15 s of
# the window start has its rising trough outside the window: the prominence
# is amputated below threshold and the event vanishes. That is invisible in
# a whole-trial task and only bites the windowed ones, where a turn defines
# the boundary and the subject plants immediately after it.
#
# On the tug return window PR plants 0.300 s after the turn and was never
# affected; MG (0.150 s), OLD_MG (0.000 s) and NEW_GX (0.150 s) all lost
# that first heel strike. 0.25 s recovers every one of them and 0.50 s finds
# nothing further, so the pad is set at the smaller value.
EDGE_PAD_S = 0.25

# Segment every trial on both anchor legs, then POOL the two results into one
# series via single_series -- do not emit them side by side. leg='auto' picks
# whichever leg has the later final heel strike, which optimises the tail of
# the trial and is blind to its head, so it can drop the first stride. Running
# both legs recovers that; pooling is what keeps one physical instant to one
# row. Set False to fall back to OpenCap's choice of a single leg.
BOTH_LEGS = True


def _gait_class():
    """Which walking class to run.

    Both are OpenCap's segment_walking with the forward chain; they differ
    only in the direction the prominence ladder is walked, which is switched
    from seg_common.LOW_PROMINENCE_FIRST. See seg_walk_core.
    """
    return (gait_analysis_lowprom if sc.LOW_PROMINENCE_FIRST
            else gait_analysis_variant)


def _marker_names(leg):
    """OpenCap marker names for a side. Right is lowercase, left is capital."""
    p = "r" if leg == "r" else "L"
    return f"{p}_calc_study", f"{p}_toe_study"


# Temporal order of the four slots inside one cycle. Forward walking runs
# ipsilateral heel strike -> contralateral toe-off -> contralateral heel
# strike -> ipsilateral toe-off. Backward walking visits the same detector
# slots in the order its own chain does, rTO -> lHS -> lTO -> rHS, which puts
# the ipsilateral toe-off second rather than last. "A" is the anchor leg,
# "C" the contralateral one.
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
    """Every gait event the detector finds inside a walking window.

    single_series can only pool what the class assembled into cycles, so a
    real event past the last cycle is invisible to it: recovering the first
    heel strike after a turn moved MG's tug series earlier and silently cost
    the trailing left toe-off at 15.000 s. These are the same peaks over the
    same window, with the same edge pad, so the pooling step can complete
    the trailing partial cycle instead of stopping at the last full one.
    """
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
    """Pool both anchor legs' cycles into one non-redundant series.

    Segmenting on both legs returns two cycle series that overlap in time,
    and the same physical instant fills a different slot in each: one left
    heel strike is the left-anchored cycle's ipsilateral strike and the
    right-anchored cycle's contralateral one, so it is emitted twice under
    two different S numbers. Across the four controls that was about a
    fifth of every gait event.

    Running both legs is still worth doing. Each anchor is blind to
    whatever falls outside its own first and last ipsilateral heel strike,
    so the pooled set of distinct physical events is up to two larger than
    either leg alone. What has to stop is emitting two *series*.

    So the two are pooled here into one set of typed events, keyed by frame
    so the duplicates collapse exactly, and a single cycle series is laid
    back over them. The anchor is the leg whose first heel strike is
    earliest: that puts any leftover event after the last full cycle, where
    the trailing partial cycle absorbs it, rather than before the first
    where it would have nowhere to go.

    Returns (cycles, anchor_leg, info). Every cycle has the same five keys
    as cycles_from_events, and any slot the trial does not support is None.
    """
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
    """Emit S1-S4 per cycle from one pooled series.

    No trailing S1 is appended: single_series already gives every
    ipsilateral heel strike its own cycle, the last of which is partial, so
    the closing strike is that cycle's own S1. Slots the trial does not
    support are absent rather than guessed.
    """
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
    """Read S1..S4 off the peak detection for a trial with one stride.

    pad opens the detection window early for prominence context only; peaks
    inside it are dropped, as in cycles_in_window.
    """
    win, time = sc._window(marker_dict, max(trim_start - pad, 0.0), trim_end)
    peaks = sc.detect_peaks(sc.walking_signals(win), time, prom)
    peaks = {k: [i for i in v if time[i] >= trim_start]
             for k, v in peaks.items()}
    rHS, lHS, rTO, lTO = (peaks["rHS"], peaks["lHS"],
                          peaks["rTO"], peaks["lTO"])

    if not (len(rHS) and len(lHS) and len(rTO) and len(lTO)):
        raise ValueError("single-stride fallback: missing an event on one side")

    # The class picks the ipsilateral leg as the one with the LATER final heel
    # strike, because it assembles cycles backwards from the end of the trial.
    # With a single stride that rule anchors S1 on the last event and pushes
    # S3 and S4 in front of it, so the stride is anchored on the EARLIER heel
    # strike instead and S1..S4 come out chronological. On PR's gait-start
    # this is the difference between S1 at 9.283 s (left, events out of order)
    # and S1 at 6.650 s-style ordering (right, rHS -> lTO -> lHS -> rTO).
    ips_leg = "r" if time[rHS[0]] < time[lHS[0]] else "l"
    cont_leg = "l" if ips_leg == "r" else "r"
    hs = {"r": rHS, "l": lHS}
    to = {"r": rTO, "l": lTO}

    ips_calc, ips_toe = _marker_names(ips_leg)
    cont_calc, cont_toe = _marker_names(cont_leg)

    # Follow the chain forwards from the anchor rather than taking the first
    # event of each kind in the window. Taking the first of each is only
    # safe when the window opens before the anchoring heel strike, which is
    # true of a gait-initiation trial but not of a window carved out of the
    # middle of a trial: there the first ipsilateral toe-off can precede the
    # contralateral heel strike, which puts S4 before S3. That is what
    # happened on MG's tug return window, S7 at 13.783 s against S6 at
    # 14.817 s.
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
        # The fallback reads the stride off the peak detection and picks its
        # own anchor, so there is only ever one series to emit.
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
                # leg='auto' already resolved to this leg, so the run above is
                # the run for it; constructing it again would be identical.
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

        # The two anchor legs describe the same footfalls, so they are pooled
        # into one series rather than emitted side by side.
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
    """Run OpenCap's walking segmentation over one straight window.

    Used by the composite tasks (gait-pivot, tug), where the trial contains
    turns. The gait segmenter cannot be run once over such a trial: the
    pelvis heading rotates through 180 degrees during a turn, and the order
    gate sees the strides either side of it as one broken sequence. So the
    turn detector defines the straight windows first and the segmenter runs
    inside each one via trimming_start / trimming_end.

    Returns (cycles, ipsilateral_leg, detail), where each cycle is a dict of
    the five event times: ips_hs, cont_to, cont_hs, ips_to, ips_hs_close.
    """
    sd = sc.session_dir(label)
    tn = sc.trial_name(label, task)
    md = sc.load_trc(label, task)
    duration = float(md["time"][-1])

    trim_start, trim_end, prom, detail = sc.search_trim(
        md, sc.ORDER_FORWARD,
        base_start=t0, base_end=max(duration - t1, 0.0),
        max_extra_start=2.0, max_end=2.0,
        min_heel_strikes=min_heel_strikes)

    # The pad is meant as prominence context, but the class gates on
    # everything it can see, so a stray peak sitting inside the pad can break
    # the event chain and lose the whole window -- NEW_GX's tug return has a
    # glitch rTO at 14.500 s, 0.03 s inside the pad, which does exactly that.
    # Falling back to the unpadded window means the pad can only ever add
    # events, never cost them.
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
        # A straight window between two turns can be short enough to hold
        # only one heel strike per foot, and a cycle needs two on the same
        # foot. The four events are still present and correctly ordered, so
        # read the single stride off the peak detection rather than
        # returning nothing. This is the return leg of tug in MG, OLD_MG and
        # NEW_GX, whose windows are 2.0, 1.7 and 2.5 s long.
        stride, ips_leg = _single_stride(task, md, trim_start, trim_end, prom,
                                         pad=EDGE_PAD_S)
        by_label = {e["event"]: e["time_s"] for e in stride}
        cycle = {"ips_hs": by_label["S1"], "cont_to": by_label["S2"],
                 "cont_hs": by_label["S3"], "ips_to": by_label["S4"],
                 "ips_hs_close": None}
        return [cycle], ips_leg, detail + SINGLE_STRIDE_TAG

    cycles, ips_leg = cycles_from_events(ga.get_gait_events())
    # The pad was context for the peak detector only. Anything inside it
    # belongs to the turn, not to this walking leg, so those slots are
    # blanked and any cycle left with nothing is dropped.
    cycles = [c for c in
              ({k: (None if v is not None and v < trim_start else v)
                for k, v in c.items()} for c in cycles)
              if any(v is not None for v in c.values())]
    return cycles, ips_leg, detail


def cycles_in_window_both(label, task, t0, t1, min_heel_strikes=2):
    """cycles_in_window for BOTH anchor legs, merged.

    leg='auto' picks the leg whose final heel strike is latest, which
    guarantees the last cycle closes on a real event but is blind to the
    start of the window. In a short window carved out between two turns that
    costs the first stride: OLD_MG's gait-pivot return window holds
    lHS 11.58, rTO 11.78, rHS 12.20, lTO 12.38, lHS 12.75, rTO 12.93,
    rHS 13.28, and auto picks right, so the cycle anchored on the left heel
    strike at 11.58 -- the first stride after the turn -- is never emitted.

    Returns [(cycles, ips_leg, detail)], one entry per anchor leg that
    produced anything. Callers tag the events with anchor_leg so the two
    overlapping series stay separable.
    """
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
        # The single-stride fallback reads the stride off the peak detection
        # and chooses its own anchor, ignoring the leg asked for. So the two
        # requests can collide on one leg: both may return that same stride,
        # and a fallback fired for one leg can name the other leg, whose own
        # request found real cycles. Keep at most one series per anchor leg,
        # and never let a fallback stride displace full cycles.
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
