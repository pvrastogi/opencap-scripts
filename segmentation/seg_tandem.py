import numpy as np
from scipy.signal import butter, filtfilt

import seg_common as sc

TASKS = ["semi-tandem", "tandem"]

PLATEAU_SD = 0.010      # m. holds sit at 0.001-0.002, transitions at 0.03-0.08
PLATEAU_MIN_S = 2.0     # s. protocol holds are ~10s

OFFSET_MIN = 0.040      # m. smallest real hold 0.059, largest neutral 0.021

# s. Minimum time past OFFSET_MIN for the excursion to be a hold.
MIN_HOLD_S = 2.0

# s. a shorter dip is the subject adjusting, not the hold ending
HOLD_MERGE_GAP_S = 1.0

ARRIVE_FRACTION = 0.25  # fraction of travel; the 'in position' band

BAND_BRIDGE_S = 0.5     # s. one frame out otherwise collapsed PR's 9s tandem hold

NEUTRAL_BAND = 0.030    # m. 0.020 missed OLD_MG's return at 0.029; stays under OFFSET_MIN

# The protocol is two ten-second holds.
EXPECTED_HOLDS = 2

# s. With no neutral plateau, neutral = median offset over the trial's first
# NEUTRAL_START_S. Controls stand still there (sd <= 4 mm) within 0.011 m of
# the plateau where one exists; DCM_001 semi-tandem's first move is at 1.7 s.
NEUTRAL_START_S = 0.5

# S2/S6 = first frame, from band arrival on, that the stepping heel is
# planted: horizontal speed under PLANT_SPEED for PLANT_HOLD_S. Band arrival
# alone is the foot passing into the band (NEW_GX tandem right 4.23 s,
# planted 4.85 s); foot-distance stillness (StaBLE) waits out the sway
# (DCM_001 tandem right 6.25 s against a 4.35 s touchdown). On controls the
# plant time is the same at 0.05 or 0.10 m/s and 0.5 or 1.0 s in 19/20 holds.
PLANT_SPEED = 0.10      # m/s
PLANT_HOLD_S = 0.5      # s
PLANT_FILTER_HZ = 6.0   # Hz, heel position before differentiating


def _find_holds(t, foot_offset, neutral_level):
    """Sustained excursions from neutral, either sign, as [(i0, i1, sign)]."""
    out = []
    for sgn in (1.0, -1.0):
        above = sgn * (foot_offset - neutral_level) >= OFFSET_MIN
        spans, i = [], 0
        while i < len(above):
            if above[i]:
                j = i
                while j + 1 < len(above) and above[j + 1]:
                    j += 1
                spans.append([i, j])
                i = j + 1
            else:
                i += 1
        merged = []
        for lo, hi in spans:
            if merged and t[lo] - t[merged[-1][1]] < HOLD_MERGE_GAP_S:
                merged[-1][1] = hi
            else:
                merged.append([lo, hi])
        out += [(lo, hi, sgn) for lo, hi in merged
                if t[hi] - t[lo] >= MIN_HOLD_S]
    out.sort()
    return out


def _band_runs(inside, t):
    """Contiguous runs of inside, with brief dips out of the band bridged."""
    runs, i, n = [], 0, len(inside)
    while i < n:
        if inside[i]:
            j = i
            while j + 1 < n and inside[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for lo, hi in runs:
        if merged and t[lo] - t[merged[-1][1]] <= BAND_BRIDGE_S:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    return merged


def _heel_speed(m, forward, t):
    heel = m["r_calc_study" if forward == "right" else "L_calc_study"]
    fs = 1.0 / float(np.mean(np.diff(t)))
    b, a = butter(4, PLANT_FILTER_HZ / (0.5 * fs), btype="low")
    v = np.gradient(filtfilt(b, a, heel, axis=0), t, axis=0)
    return np.hypot(v[:, 0], v[:, 2])


def _planted(speed, t, start, stop):
    """First frame in [start, stop) that begins PLANT_HOLD_S of planted heel."""
    n = max(int(round(PLANT_HOLD_S / float(np.mean(np.diff(t))))), 1)
    slow = speed < PLANT_SPEED
    for j in range(start, stop):
        if slow[j:j + n].all() and j + n <= len(slow):
            return j
    return None


def segment(label, task, verbose=True):
    trim_start, trim_applied, trim_detail = sc.calibration_trim(label, task)
    md = sc.load_trc(label, task)
    win, t = sc._window(md, trim_start, 0.0)

    sig = sc.walking_signals(win)
    foot_offset = sig["r_calc_rel_x"] - sig["l_calc_rel_x"]

    runs, sd, sd_thr = sc.steady_runs(t, foot_offset, window_s=1.0,
                                      sd_threshold=PLATEAU_SD,
                                      min_len_s=PLATEAU_MIN_S)

    neutral_runs = [r for r in runs if abs(r[2]) < OFFSET_MIN]
    if neutral_runs:
        neutral = max(neutral_runs, key=lambda r: r[1] - r[0])
        neutral_level = neutral[2]
        neutral_found = True
    else:
        start = np.flatnonzero(t - t[0] <= NEUTRAL_START_S)
        neutral_level = float(np.median(foot_offset[start]))
        neutral = (float(t[0]), float(t[start[-1]]), neutral_level, 0,
                   int(start[-1]))
        neutral_found = False

    holds = _find_holds(t, foot_offset, neutral_level)

    best, extra = {}, []
    for h in holds:
        side = "right" if h[2] > 0 else "left"
        if side in best:
            shorter = min(best[side], h, key=lambda x: x[1] - x[0])
            best[side] = max(best[side], h, key=lambda x: x[1] - x[0])
            extra.append([round(float(t[shorter[0]]), 3),
                          round(float(t[shorter[1]]), 3)])
        else:
            best[side] = h
    chosen = sorted(best.items(), key=lambda kv: kv[1][0])

    events, assigned, missing, unplanted = [], {}, [], []
    prev_return = 0
    for forward, (i0, i1, sgn) in chosen:
        base = 1 if forward == "right" else 5
        sign_note = "" if forward == "right" else " (opposite sign)"
        pos = ("semi-tandem position" if task == "semi-tandem"
               else "tandem position")

        s = sgn * (foot_offset - neutral_level)
        level = float(np.median(foot_offset[i0:i1 + 1]))
        band = abs(level - neutral_level) * ARRIVE_FRACTION

        inside = np.abs(foot_offset - level) <= band
        overlapping = [r for r in _band_runs(inside, t)
                       if r[0] <= i1 and r[1] >= i0]
        if overlapping:
            arrive, depart = max(overlapping,
                                 key=lambda r: t[r[1]] - t[r[0]])
        else:
            arrive, depart = i0, i1

        band_arrive = arrive
        plant = _planted(_heel_speed(win["markers"], forward, t), t,
                         arrive, depart)
        if plant is not None:
            arrive = plant
        else:
            unplanted.append(forward)

        back = np.where(
            np.abs(foot_offset[depart:] - neutral_level) <= NEUTRAL_BAND)[0]
        ret = int(depart + back[0]) if len(back) else None

        onset = sc.movement_onset(s, t, arrive, search_from=prev_return)
        prev_return = ret if ret is not None else depart

        assigned[forward] = (float(t[arrive]), float(t[depart]), level)

        events.append(sc.event(
            task, f"S{base}",
            f"first sustained movement of {forward} foot", float(t[onset]),
            "trc",
            f"walking back from the fastest change to the start of the "
            f"movement out of neutral{sign_note}",
            side=forward[0], anchor_leg=forward))
        events.append(sc.event(
            task, f"S{base + 1}",
            f"{forward} foot touchdown in {pos}", float(t[arrive]), "trc",
            (f"first frame after foot_offset arrives within {band:.3f} m of "
             f"its hold level ({level:+.3f} m, {t[band_arrive]:.2f} s) with "
             f"the heel planted: speed < {PLANT_SPEED} m/s for "
             f"{PLANT_HOLD_S} s{sign_note}") if forward not in unplanted else
            (f"heel never planted for {PLANT_HOLD_S} s; foot_offset arrives "
             f"within {band:.3f} m of its hold level ({level:+.3f} m)"
             f"{sign_note}"),
            side=forward[0], anchor_leg=forward))
        events.append(sc.event(
            task, f"S{base + 2}",
            f"first sustained movement of {forward} foot back to neutral",
            float(t[depart]), "trc",
            f"foot_offset departs the band around its hold level"
            f"{sign_note}",
            side=forward[0], anchor_leg=forward))
        if ret is None:
            missing.append(f"S{base + 3} {forward} foot touchdown back in "
                           f"neutral position")
        else:
            events.append(sc.event(
                task, f"S{base + 3}",
                f"{forward} foot touchdown back in neutral position",
                float(t[ret]), "trc",
                f"foot_offset returns within {NEUTRAL_BAND:.3f} m of "
                f"neutral{sign_note}",
                side=forward[0], anchor_leg=forward))

    for forward in ("right", "left"):
        if forward not in best:
            b = 1 if forward == "right" else 5
            missing.append(f"S{b}-S{b + 3} {forward} foot hold not found")

    lead = chosen[0][0] if chosen else None

    meta = {
        "task": task, "trial_file": sc.trial_name(label, task),
        "neutral_level_m": round(neutral_level, 4),
        "neutral_plateau_found": neutral_found,
        "neutral_span_s": [round(neutral[0], 3), round(neutral[1], 3)],
        "n_holds": len(assigned), "expected_holds": EXPECTED_HOLDS,
        "leading_foot": lead,
        "holds": {k: {"in_position_s": [round(v[0], 3), round(v[1], 3)],
                      "duration_s": round(v[1] - v[0], 2),
                      "level_m": round(v[2], 4)}
                  for k, v in assigned.items()},
        "extra_holds_s": extra,
        "n_steady_runs": len(runs),
        "partial": bool(missing), "missing_events": missing,
        "trim_start_s": round(trim_start, 4), "trim_applied": trim_applied,
        "calibration_detail": trim_detail,
    }
    warnings = []
    if len(assigned) != EXPECTED_HOLDS:
        warnings.append(f"{len(assigned)} offset hold(s) found, protocol "
                        f"expects {EXPECTED_HOLDS}")
    if unplanted:
        warnings.append(f"{'/'.join(unplanted)} heel never planted for "
                        f"{PLANT_HOLD_S} s in the hold; touchdown is band "
                        f"arrival")
    if not neutral_found:
        warnings.append(f"no steady plateau near zero offset; neutral taken "
                        f"from the first {NEUTRAL_START_S} s "
                        f"({neutral_level:+.3f} m)")
    if warnings:
        meta["warning"] = "; ".join(warnings)

    if verbose:
        desc = ", ".join(f"{k} {v[2]:+.3f} m for {v[1] - v[0]:.1f}s"
                         for k, v in sorted(assigned.items()))
        print(f"  {task}: neutral {neutral_level:+.3f} m; {len(assigned)} "
              f"hold(s) [{desc}]; led with {lead}")
        if missing:
            print(f"    PARTIAL -- not emitted: {', '.join(missing)}")
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
