import numpy as np

# seg_common puts opencap-processing on sys.path; import it first
import seg_common as sc

from scipy import signal
import matplotlib.pyplot as plt

from sts_analysis import sts_analysis


END_RISE_REFLEX_DEG = 10.0   # deg of re-flexion that ends the rise
END_RISE_CAP_S = 2.0         # s, hard ceiling on how far it can advance
END_RISE_TOL_DEG = 3.0       # deg of the window minimum that counts as there


def refine_end_rising(sts):
    """Advance each endRising event to full knee extension."""
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

    # rad/s. OpenCap's own lean_threshold, reused unchanged
    LEAN_STOP = -0.05

    LEAN_GAP_S = 0.05

    def _seated_start(self, pelvVel, sit_idx, lift_idx, velSeated):
        """First frame after the LAST sit-down preceding lift-off."""
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
        """Onset of the forward lean immediately preceding lift-off."""
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
        """Forbid a lean onset from starting before the previous cycle's sit."""
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


