"""
sts_analysis with the forward-lean onset re-anchored.
=====================================================

OpenCap's segment_sts finds the lean onset by stepping BACKWARDS from
lift-off and stopping at the first frame whose torso angular velocity is
above lean_threshold (-0.05 rad/s), i.e. the first frame not rotating
forward. That marks the start of the last uninterrupted forward-rotation
burst, and it is fragile in one specific way: if the torso has already
stopped rotating forward at lift-off -- which happens whenever the subject
completes the lean and pauses before pushing up -- the search terminates on
its very first frame and the onset collapses onto lift-off.

Measured on PR's 5tsts, the lead of the lean onset over the start of rising
was 0.950, 0.083, 0.083, 0.350, 0.267 s across the five repetitions. Cycles
2 and 3 are the collapsed ones: torso_z_vel at lift-off is -0.015 rad/s for
cycle 2, already above the -0.05 threshold.

THE CHANGE
    Anchor on the fastest forward rotation instead of on lift-off, then walk
    back from there. This is the pattern startRisingIdx already uses for the
    pelvis:

        startRisingIdx      find the pelvis vertical velocity PEAK, then walk
                            back until velocity drops below velSeated
        _forward_lean_onset find the torso forward angular velocity PEAK,
                            then walk back until velocity rises above
                            lean_threshold

    Same threshold as OpenCap, same walk-back, different anchor. A pause at
    lift-off can no longer truncate the search, because the search no longer
    starts there.

WHY THE WINDOW IS BOUNDED BY THE SEATED PERIOD, NOT temp_sit_ind
    temp_sit_ind is computed as maxIdxOld + argmin(pelvVel[...]), i.e. the
    instant of FASTEST PELVIS DESCENT -- the middle of the previous sit-down,
    not the point at which the subject is seated. Anchoring on a velocity
    extremum inside that window lets the search look into the previous
    repetition's sit, where the trunk also flexes forward. Left unbounded
    that produced onsets before the previous cycle's sitting event: PR's
    5tsts cycle 2 reported 12.650 s against cycle 1 sitting at 12.750 s.

    So the window start is advanced from temp_sit_ind to the first frame
    where the pelvis has stopped descending (pelvVel >= -velSeated), reusing
    OpenCap's own velSeated constant. The lean is then searched for strictly
    within the seated period.

WHAT THIS DOES NOT CHANGE
    The lean is still measured on the torso segment in the ground frame
    (pelvis orientation plus lumbar), not on hip flexion. Those genuinely
    differ: in PR's tug the trunk rotates about 20 degrees before hip flexion
    begins, because the early lean is lumbar. If the intended event is the
    hip-flexion onset, that is a different signal, not a tuning of this one.
"""

import numpy as np

# seg_common puts the opencap-processing checkout on sys.path, so it has to
# be imported before the OpenCap modules.
import seg_common as sc

from scipy import signal
import matplotlib.pyplot as plt

from sts_analysis import sts_analysis


# ---------------------------------------------------------------------------
# End of rising, corroborated with knee extension
# ---------------------------------------------------------------------------
# OpenCap sets endRisingIdx to the first frame after the pelvis velocity peak
# at which the pelvis is rising slower than velStanding (0.1 m/s). A subject
# who pushes up, stalls and then completes trips that threshold at the stall,
# so the rise is reported as finished while they are still bent: DCM_001's
# 5tsts cycle 1 reported 5.233 s against a true 5.98 s, and cycles 4 and 5
# the same way, while cycles 2 and 3 -- the ones without a stall -- were
# right.
#
# The refinement never moves the event earlier. It walks FORWARD from
# OpenCap's frame to the most extended frame of the same rise:
#
#     signal   max(knee_angle_r, knee_angle_l), the MORE FLEXED knee, so
#              both legs are included and the trailing leg governs. A
#              subject who brings one foot under them last cannot be called
#              upright until that leg has straightened too.
#     window   forward until the signal rises END_RISE_REFLEX_DEG above its
#              running minimum -- the subject has started to bend again,
#              which in tug is the first step -- capped at END_RISE_CAP_S
#              and at the same cycle's sitting frame.
#     event    the first frame within END_RISE_TOL_DEG of the window
#              minimum, so it is arrival at full extension rather than the
#              single most extended frame of a plateau.
#
# Measured over the thirty rises in the five controls' 5tsts and tug, the
# shift is 0.000 to +0.350 s, median +0.10 s; 5tsts alone maxes at +0.15 s.
# Results are identical at caps of 1.5 and 2.5 s, so it is the re-flexion
# rule and not the cap that binds.
#
# THREE ALTERNATIVES WERE MEASURED AND REJECTED, all of them plausible:
#
#   Bounding the window by the pelvis-height peak instead. Wrong for tug,
#   where the pelvis keeps rising through the walk -- PR's peak is at
#   20.6 s, 6.7 s after standing -- and the knee's minimum in that window
#   belongs to a gait cycle. It moved OLD_MG's tug +1.10 s.
#
#   Walking forward through the whole extension burst, stopping where the
#   knee stops extending. Drifts on a subject standing still, because
#   postural sway keeps the velocity under any threshold: OLD_GX's 5tsts
#   cycle 3 ran +1.85 s for 5.6 deg of extension gained.
#
#   Waiting for the knee to return to a standing baseline taken as a low
#   percentile of the trial. Fine on 5tsts, catastrophic on tug, where
#   there is no standing period for the percentile to represent: MG +7.13 s,
#   NEW_GX +4.32 s.
#
# Adding hip flexion to the signal also works and costs a little more --
# controls shift up to +0.47 s rather than +0.35 s -- so the knee alone is
# what is used and both hips are reported alongside in the run log.

END_RISE_REFLEX_DEG = 10.0   # deg of re-flexion that ends the rise
END_RISE_CAP_S = 2.0         # s, hard ceiling on how far it can advance
END_RISE_TOL_DEG = 3.0       # deg of the window minimum that counts as there


def refine_end_rising(sts):
    """Advance each endRising event to full knee extension.

    Mutates sts.stsEvents in place and returns the per-cycle detail for the
    run log. Returns [] when the class found no cycles.
    """
    ev = sts.stsEvents
    if not ev.get("endRisingIdx"):
        return []

    t = np.asarray(sts.markerDict["time"])
    cv = sts.coordinateValues
    knee = np.maximum(np.asarray(cv["knee_angle_r"]),
                      np.asarray(cv["knee_angle_l"]))
    hip = np.maximum(np.asarray(cv["hip_flexion_r"]),
                     np.asarray(cv["hip_flexion_l"]))
    cap_frames = int(END_RISE_CAP_S * sc.SAMPLE_RATE)

    idx, times, detail = [], [], []
    for i, raw in enumerate(ev["endRisingIdx"]):
        e = int(raw)
        hi = min(e + cap_frames, len(t) - 1)
        if i < len(ev["sittingIdx"]):
            hi = min(hi, int(ev["sittingIdx"][i]))
        hi = max(hi, e)

        # Forward to the first sustained re-flexion.
        j, run_min = e, knee[e]
        while j < hi:
            if knee[j] >= run_min + END_RISE_REFLEX_DEG:
                break
            run_min = min(run_min, knee[j])
            j += 1

        window = knee[e:j + 1]
        floor = float(window.min())
        k = e + int(np.argmax(window <= floor + END_RISE_TOL_DEG))

        idx.append(k)
        times.append(float(t[k]))
        detail.append({
            "cycle": i + 1,
            "opencap_s": round(float(t[e]), 3),
            "refined_s": round(float(t[k]), 3),
            "advanced_s": round(float(t[k] - t[e]), 3),
            "search_ended_s": round(float(t[j]), 3),
            "capped": bool(j >= e + cap_frames),
            "knee_at_opencap_deg": round(float(knee[e]), 1),
            "knee_at_event_deg": round(float(knee[k]), 1),
            "hip_at_event_deg": round(float(hip[k]), 1),
        })

    ev["endRisingIdx"] = idx
    ev["endRisingTime"] = times
    return detail


class sts_analysis_leanfix(sts_analysis):
    """sts_analysis with _forward_lean_onset in place of the inline search."""

    # rad/s. OpenCap's own lean_threshold default, reused as the "no longer
    # rotating forward" level so the walk-back criterion is unchanged.
    LEAN_STOP = -0.05

    # s. Brief interruptions shorter than this do not end a lean.
    #
    # Swept 0.05 / 0.10 / 0.15 across all four sessions: the value changes
    # nothing anywhere except PR's tug, where the trunk drifts slowly
    # forward from 11.30 s with two short pauses before the fast lean at
    # 12.4 s. At 0.15 s the bridge joins the drift to the lean and reports
    # 11.25 s; at 0.05 s the two stay separate and it reports 12.25 s, which
    # is the value cross-checked against that trial earlier.
    LEAN_GAP_S = 0.05

    def _seated_start(self, pelvVel, sit_idx, lift_idx, velSeated):
        """First frame after the LAST sit-down preceding lift-off.

        temp_sit_ind is the instant of fastest pelvis descent, and for the
        first repetition the class sets it to 0 outright. Neither is the
        moment the subject is seated. Advancing only while the pelvis is
        descending AT that index does not help when the descent comes later
        in the window, which is the first-repetition case: OLD_MG's 5tsts
        starts with temp_sit_ind = 0 while the sit-down runs 9.28-9.88 s, so
        the window still contained the whole descent.
        """
        window = pelvVel[sit_idx:lift_idx]
        descending = np.where(window < -velSeated)[0]
        if len(descending) == 0:
            return sit_idx
        k = sit_idx + int(descending[-1])
        while k < lift_idx and pelvVel[k] < -velSeated:
            k += 1
        return k

    def _forward_lean_onset(self, torso_z_vel, pelvVel, sit_idx, lift_idx,
                            lean_threshold=None, velSeated=0.1):
        """Onset of the forward lean immediately preceding lift-off.

        Three steps:

        1. Bound the search below by the end of the last sit-down before
           lift-off, so the trunk flexion that happens while dropping into
           the chair can never be mistaken for the lean. OLD_MG's 5tsts
           reported 9.367 s without this -- the descent itself, at which
           point the torso swung from -24.3 to -39.0 degrees.

        2. Walk back from lift-off to the most recent frame that is actually
           rotating forward. OpenCap stops at the first frame that is NOT,
           which collapses the onset onto lift-off whenever the subject
           finishes the lean and pauses before pushing up.

        3. Walk back through that burst to its start, bridging interruptions
           shorter than LEAN_GAP_S.

        Anchoring on the fastest forward rotation anywhere in the window --
        which is what this did before -- picks whichever trunk movement is
        largest rather than the one that leads into the rise. NEW_GX's 5tsts
        reported 6.633 s that way, a seated trunk excursion to -53.7 degrees
        that the subject came back up from well before standing at 9.2 s.
        """
        if lean_threshold is None:
            lean_threshold = self.LEAN_STOP
        if lift_idx <= sit_idx:
            return lift_idx

        lo = self._seated_start(pelvVel, sit_idx, lift_idx, velSeated)
        if lift_idx <= lo:
            return lift_idx

        # Most recent frame rotating forward.
        j = lift_idx
        while j > lo and torso_z_vel[j] >= lean_threshold:
            j -= 1
        if j <= lo:
            # Never rotated forward while seated; nothing to anchor on.
            return lift_idx

        # Back to the start of that burst, bridging brief interruptions.
        gap = max(int(self.LEAN_GAP_S * 60), 1)
        k = j
        while k > lo:
            if torso_z_vel[k] < lean_threshold:
                k -= 1
                continue
            look = max(lo, k - gap)
            if np.any(torso_z_vel[look:k] < lean_threshold):
                k -= 1
            else:
                break
        return k

    def _apply_lean_floor(self, leanInds, sitInds, leanWindows, torso_z_vel,
                          pelvVel, timeVec, lean_threshold, velSeated):
        """Forbid a lean onset from starting before the previous cycle's sit.

        _seated_start already bounds the search at the end of the last pelvis
        descent, which lands close to the sitting event, so on the five
        controls this changes nothing -- the smallest margin between a lean
        and the preceding sit is 0.100 s. It is here as a hard guarantee
        rather than an emergent one: the two are derived from different
        signals (end of descent vs pelvis back within 5 cm of the seated
        height), and nothing so far forces them to agree on a subject who
        lowers into the chair slowly or in stages.

        sittingIdx is not available inside the segmentation loop -- OpenCap
        computes it afterwards, in the periodicity block, and cycle i-1's
        value depends on cycle i's rising indices -- so the leans are
        recomputed here instead, against a floor, for any cycle that violates
        it. The first cycle has no preceding sit and keeps floor 0.
        """
        out = []
        for i, idx in enumerate(leanInds):
            floor = int(sitInds[i - 1][1]) if i > 0 else 0
            if int(idx) >= floor:
                out.append(int(idx))
                continue
            sit_idx, lift_idx = leanWindows[i]
            out.append(int(self._forward_lean_onset(
                torso_z_vel, pelvVel, max(int(sit_idx), floor), int(lift_idx),
                lean_threshold, velSeated)))
        return out, [timeVec[i].tolist() for i in out]

    def segment_sts(self, n_sts_cycles=-1, velSeated=0.1, velStanding=0.1,
                    visualize=False, delay=0.1, lean_threshold=-0.05):
        # Extract pelvis height and time
        pelvis_ty = self.coordinateValues['pelvis_ty']
        timeVec = self.markerDict['time']
        dt = timeVec[1] - timeVec[0]

        # Extract torso lean angle and angular velocity
        torso_z = self.body_angles['torso_z']
        torso_z_vel = self.body_angular_velocity['torso_z']

        # Normalize pelvis height signal
        pelvSignal = np.array(pelvis_ty - np.min(pelvis_ty))
        pelvVel = np.diff(pelvSignal, append=0) / dt

        # Find peaks in pelvis vertical position (STS max points)
        idxMaxPelvTy, _ = signal.find_peaks(pelvSignal - np.min(pelvSignal), distance=.9 / dt, height=.12, prominence=.12)

        # Initialize storage
        maxIdxOld = 0
        startFinishInds = []
        forwardLeanInds = []
        leanWindows = []

        for i, maxIdx in enumerate(idxMaxPelvTy):
            # Identify velocity peak before pelvis peak
            vels = pelvVel[maxIdxOld:maxIdx]
            velPeak, peakVals = signal.find_peaks(vels, distance=.9 / dt, height=.2)
            velPeak = velPeak[np.argmax(peakVals['peak_heights'])] + maxIdxOld

            # Find sitting and standing transitions
            velsLeftOfPeak = np.flip(pelvVel[maxIdxOld:velPeak])
            velsRightOfPeak = pelvVel[velPeak:]

            slowingIndLeft = np.argwhere(velsLeftOfPeak < velSeated)[0]
            startIdx = velPeak - slowingIndLeft
            slowingIndRight = np.argwhere(velsRightOfPeak < velStanding)[0]
            endIdx = velPeak + slowingIndRight

            startFinishInds.append([startIdx[0], endIdx[0]])

            # Define a temporary seating index for better forward lean detection
            if i == 0:
                temp_sit_ind = maxIdxOld
            else:
                temp_sit_ind = maxIdxOld + np.argmin(pelvVel[maxIdxOld:startIdx[0]])

            # ================= CHANGE FROM OpenCap 1 of 2 =================
            forwardLeanIdx = self._forward_lean_onset(
                torso_z_vel, pelvVel, int(temp_sit_ind), int(startIdx[0]),
                lean_threshold, velSeated)
            leanWindows.append((int(temp_sit_ind), int(startIdx[0])))
            # ==============================================================

            forwardLeanInds.append(forwardLeanIdx)
            maxIdxOld = np.copy(maxIdx)

        # Convert times
        risingTimes = [timeVec[i].tolist() for i in startFinishInds]
        forwardLeanTimes = [timeVec[i].tolist() for i in forwardLeanInds]

        # Apply delay correction
        sf = 1 / np.round(np.mean(np.round(timeVec[1:] - timeVec[:-1], 2)), 16)
        startFinishIndsDelay = []
        for i in startFinishInds:
            c_i = []
            for c_j, j in enumerate(i):
                if c_j == 0:
                    c_i.append(j + int(delay * sf))
                else:
                    c_i.append(j)
            startFinishIndsDelay.append(c_i)
        risingTimesDelayedStart = [
            timeVec[i].tolist() for i in startFinishIndsDelay]

        # Adjust for periodicity
        startFinishIndsDelayPeriodic = []
        for i, val in enumerate(startFinishIndsDelay):
            pelvVal_up = pelvSignal[val[0]]
            val_down = np.argwhere(pelvSignal[val[1] + 1:] < (pelvVal_up+0.05))[0][0] + val[1] + 1 # 5cm threshold above where the pelvis started rising

            original_val_down = val_down
            last_index = startFinishIndsDelay[i+1][1] if i < len(startFinishIndsDelay) - 1 else len(pelvSignal) - 1
            while val_down < last_index:   # Don't go past end of trial
                if pelvSignal[val_down] <= pelvVal_up or pelvVel[val_down] >= 0:
                    break
                val_down += 1


            if val_down >= last_index:
                val_down = original_val_down
            else:
                val_down -=1
            startFinishIndsDelayPeriodic.append([val[0], val_down])

        # ================= CHANGE FROM OpenCap 2 of 2 =================
        # sittingIdx only exists now, so the lean floor is applied here.
        forwardLeanInds, forwardLeanTimes = self._apply_lean_floor(
            forwardLeanInds, startFinishIndsDelayPeriodic, leanWindows,
            torso_z_vel, pelvVel, timeVec, lean_threshold, velSeated)
        # ==============================================================

        risingSittingTimesDelayedStartPeriodicEnd = [timeVec[i].tolist() for i in startFinishIndsDelayPeriodic]

        if visualize:
            plt.figure()
            plt.plot(pelvSignal)
            for c_v, val in enumerate(startFinishInds):
                plt.plot(val, pelvSignal[val], marker='o', markerfacecolor='k',
                         markeredgecolor='none', linestyle='none',
                         label='Rising phase')
                val2 = startFinishIndsDelay[c_v][0]
                plt.plot(val2, pelvSignal[val2], marker='o',
                         markerfacecolor='r', markeredgecolor='none',
                         linestyle='none', label='Delayed start')
                val3 = startFinishIndsDelayPeriodic[c_v][1]
                plt.plot(val3, pelvSignal[val3], marker='o',
                         markerfacecolor='g', markeredgecolor='none',
                         linestyle='none',
                         label='Periodic end corresponding to delayed start')
                val4 = forwardLeanInds[c_v]
                plt.plot(val4, pelvSignal[val4], marker='o',
                         markerfacecolor='b', markeredgecolor='none',
                         linestyle='none', label='Forward Lean Start')
                if c_v == 0:
                    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.40), ncol=2)
            plt.xlabel('Frames')
            plt.ylabel('Position [m]')
            plt.title('Vertical pelvis position')
            plt.tight_layout()
            plt.show()

        # Ensure correct STS cycle count
        actual_cycles = len(risingSittingTimesDelayedStartPeriodicEnd)
        if actual_cycles < n_sts_cycles or n_sts_cycles == -1:
            n_sts_cycles = actual_cycles

        if n_sts_cycles < 1:
            raise Exception('No STS cycles found.')
        else:
            print('Found and Processing', n_sts_cycles, 'STS cycles.')

        # Build output dictionary
        stsEvents = {
            'startRisingIdx': [startFinishIndsDelay[i][0] for i in range(n_sts_cycles)],
            'startRisingTime': [risingTimesDelayedStart[i][0] for i in range(n_sts_cycles)],
            'endRisingIdx': [startFinishInds[i][1] for i in range(n_sts_cycles)],
            'endRisingTime': [risingTimes[i][1] for i in range(n_sts_cycles)],
            'sittingIdx': [startFinishIndsDelayPeriodic[i][1] for i in range(n_sts_cycles)],
            'sittingTime': [risingSittingTimesDelayedStartPeriodicEnd[i][1] for i in range(n_sts_cycles)],
            'forwardLeanIdx': [forwardLeanInds[i] for i in range(n_sts_cycles)],
            'forwardLeanTime': [forwardLeanTimes[i] for i in range(n_sts_cycles)]
        }

        return stsEvents


