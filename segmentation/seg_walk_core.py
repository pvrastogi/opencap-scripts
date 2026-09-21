"""
One verbatim copy of OpenCap's segment_walking, with two lines parameterised.
============================================================================

gait_analysis.segment_walking hardcodes two things we need to vary:

    expectedOrder = {'rHS': 'lTO', ...}     the forward event chain
    prominences   = [0.3, 0.25, 0.2]        the retry ladder

Backward walking needs a different chain, and recovering low-prominence
events needs the ladder walked from the other end. Rather than keep a
separate full copy of the method per combination, the method is copied once
from opencap-processing/ActivityAnalyses/gait_analysis.py lines 819-1053 with
those two expressions replaced by class attributes:

    expectedOrder = self.EXPECTED_ORDER
    prominences   = self.PROMINENCES

Every other line is byte-identical to OpenCap's, including the projections,
the cycle assembly, the contralateral search and the silent frame-zero
behaviour when an ipsilateral toe-off search fails. Run
    python3 verify_verbatim.py
to re-check that claim against the source.

THE FOUR VARIANTS

    gait_analysis_variant            forward chain, OpenCap's ladder
    gait_analysis_lowprom            forward chain, ladder reversed
    gait_analysis_backward           backward chain, OpenCap's ladder
    gait_analysis_backward_lowprom   backward chain, ladder reversed

WHY REVERSE THE LADDER
    OpenCap's ladder is failure-driven: it takes the FIRST prominence whose
    peaks satisfy the order gate, which is the strictest one, and only
    descends when the gate fails. So a genuine event that is merely less
    prominent than 0.3 m is never recovered as long as the strict setting
    happens to produce a self-consistent order. PR's gait-pivot is the case:
    the first step out of the turn peaks at 0.428 m in r_calc_rel_x, which
    0.3 m rejects, and because the remaining events ordered correctly the
    ladder never retried. Reversing it to [0.2, 0.25, 0.3] makes "first
    passing" mean "lowest passing", which recovers that heel strike.

    The risk is the mirror image: a lower prominence admits noise peaks that
    happen to preserve a valid order. That is why the ladder direction is a
    module-level switch and why the run log records which prominence was
    actually used per trial.

BACKWARD CHAIN
    Forward walking runs rHS -> lTO -> lHS -> rTO. Backward walking is that
    same chain with the heel-strike and toe-off labels exchanged,
    rTO -> lHS -> lTO -> rHS, which is the chain in the segmentation markers
    document's gait-bckw row. The detectors do not change: a peak in
    calc_rel_x still fills the "HS" slot, but walking backwards the planted
    foot drifts anteriorly relative to the pelvis, so that peak is heel-OFF
    and a trough in toe_rel_x is toe-STRIKE.
"""

import numpy as np

# seg_common puts the opencap-processing checkout on sys.path, so it has to
# be imported before the OpenCap modules.
import seg_common as sc

import copy
import pandas as pd
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

from gait_analysis import gait_analysis


class gait_analysis_variant(gait_analysis):
    """gait_analysis with the event chain and retry ladder made overridable."""

    EXPECTED_ORDER = sc.ORDER_FORWARD
    PROMINENCES = [0.3, 0.25, 0.2]

    def _collapse_replants(self, rHS, lHS, rTO, lTO):
        """Reduce a replanted foot's repeated peaks to one event each.

        A heel strike is a foot going down, so a cluster keeps its LAST
        member: the contact the subject committed to. A toe-off is a foot
        coming up, so a cluster keeps its FIRST: the moment it left.

        This runs before detect_correct_order, which matters more than the
        event placement does. A replant injects an extra HS between the real
        HS and the next TO, which breaks the rHS -> lTO -> lHS -> rTO chain,
        so the order gate fails, the prominence ladder is walked to the end
        and segment_walking raises. Collapsing first is what lets a trial
        with a replanted step segment at all.
        """
        t = self.markerDict['time']
        hw, tw = sc.REPLANT_WINDOW_HS_S, sc.REPLANT_WINDOW_TO_S
        return (sc.cluster_events(rHS, t, hw, 'last'),
                sc.cluster_events(lHS, t, hw, 'last'),
                sc.cluster_events(rTO, t, tw, 'first'),
                sc.cluster_events(lTO, t, tw, 'first'))

    def segment_walking(self, n_gait_cycles=-1, leg='auto', visualize=False):

        # n_gait_cycles = -1 finds all accessible gait cycles. Otherwise, it 
        # finds that many gait cycles, working backwards from end of trial.
           
        # Helper functions
        def detect_gait_peaks(r_calc_rel_x,
                              l_calc_rel_x,
                              r_toe_rel_x,
                              l_toe_rel_x,
                              prominence = 0.3):
            # Find HS.
            rHS, _ = find_peaks(r_calc_rel_x, prominence=prominence)
            lHS, _ = find_peaks(l_calc_rel_x, prominence=prominence)
        
            # Find TO.
            rTO, _ = find_peaks(-r_toe_rel_x, prominence=prominence)
            lTO, _ = find_peaks(-l_toe_rel_x, prominence=prominence)

            # ================= CHANGE FROM OpenCap 3 of 3 =================
            return self._collapse_replants(rHS,lHS,rTO,lTO)
            # ==============================================================
    
        def detect_correct_order(rHS, rTO, lHS, lTO):
            # checks if the peaks are in the right order
                
            expectedOrder = self.EXPECTED_ORDER
                
            # Identify vector that has the smallest value in it. Put this vector name
            # in vName1
            vectors = {'rHS': rHS, 'rTO': rTO, 'lHS': lHS, 'lTO': lTO}
            non_empty_vectors = {k: v for k, v in vectors.items() if len(v) > 0}
    
            # Check if there are any non-empty vectors
            if not non_empty_vectors:
                return True  # All vectors are empty, consider it correct order
    
            vName1 = min(non_empty_vectors, key=lambda k: non_empty_vectors[k][0])
    
            # While there are any values in any of the vectors (rHS, rTO, lHS, or lTO)
            while any([len(vName) > 0 for vName in vectors.values()]):
                # Delete the smallest value from the vName1
                vectors[vName1] = np.delete(vectors[vName1], 0)
    
                # Then find the vector with the next smallest value. Define vName2 as the
                # name of this vector
                non_empty_vectors = {k: v for k, v in vectors.items() if len(v) > 0}
            
                # Check if there are any non-empty vectors
                if not non_empty_vectors:
                    break  # All vectors are empty, consider it correct order
    
                vName2 = min(non_empty_vectors, key=lambda k: non_empty_vectors[k][0])
    
                # If vName2 != expectedOrder[vName1], return False
                if vName2 != expectedOrder[vName1]:
                    return False
    
                # Set vName1 equal to vName2 and clear vName2
                vName1, vName2 = vName2, ''
    
            return True
    
        # Subtract sacrum from foot.
        # It looks like the position-based approach will be more robust.        
        r_calc_rel = (
            self.markerDict['markers']['r_calc_study'] - 
            self.markerDict['markers']['r.PSIS_study'])
    
        r_toe_rel = (
            self.markerDict['markers']['r_toe_study'] - 
            self.markerDict['markers']['r.PSIS_study'])
        r_toe_rel_x = r_toe_rel[:,0]
        # Repeat for left.
        l_calc_rel = (
            self.markerDict['markers']['L_calc_study'] - 
            self.markerDict['markers']['L.PSIS_study'])
        l_toe_rel = (
            self.markerDict['markers']['L_toe_study'] - 
            self.markerDict['markers']['L.PSIS_study'])
    
        # Identify which direction the subject is walking.
        mid_psis = (self.markerDict['markers']['r.PSIS_study'] + self.markerDict['markers']['L.PSIS_study'])/2
        mid_asis = (self.markerDict['markers']['r.ASIS_study'] + self.markerDict['markers']['L.ASIS_study'])/2
        mid_dir = mid_asis - mid_psis
        mid_dir_floor = np.copy(mid_dir)
        mid_dir_floor[:,1] = 0
        mid_dir_floor = mid_dir_floor / np.linalg.norm(mid_dir_floor,axis=1,keepdims=True)
    
        # Dot product projections   
        r_calc_rel_x = np.einsum('ij,ij->i', mid_dir_floor,r_calc_rel)
        l_calc_rel_x = np.einsum('ij,ij->i', mid_dir_floor,l_calc_rel)
        r_toe_rel_x = np.einsum('ij,ij->i', mid_dir_floor,r_toe_rel)
        l_toe_rel_x = np.einsum('ij,ij->i', mid_dir_floor,l_toe_rel)
    
        # Old Approach that does not take the heading direction into account.
        # r_psis_x = self.markerDict['markers']['r.PSIS_study'][:,0]
        # r_asis_x = self.markerDict['markers']['r.ASIS_study'][:,0]
        # r_dir_x = r_asis_x-r_psis_x
        # position_approach_scaling = np.where(r_dir_x > 0, 1, -1)        
        # r_calc_rel_x = r_calc_rel[:,0] * position_approach_scaling
        # r_toe_rel_x = r_toe_rel[:,0] * position_approach_scaling
        # l_calc_rel_x = l_calc_rel[:,0] * position_approach_scaling
        # l_toe_rel_x = l_toe_rel[:,0] * position_approach_scaling
                   
        # Detect peaks, check if they're in the right order, if not reduce prominence.
        # the peaks can be less prominent with pathological or slower gait patterns
        prominences = self.PROMINENCES
    
        for i,prom in enumerate(prominences):
            rHS,lHS,rTO,lTO = detect_gait_peaks(r_calc_rel_x=r_calc_rel_x,
                                  l_calc_rel_x=l_calc_rel_x,
                                  r_toe_rel_x=r_toe_rel_x,
                                  l_toe_rel_x=l_toe_rel_x,
                                  prominence=prom)
            if not detect_correct_order(rHS=rHS, rTO=rTO, lHS=lHS, lTO=lTO):
                if prom == prominences[-1]:
                    raise ValueError('The ordering of gait events is not correct. Consider trimming your trial using the trimming_start and trimming_end options.')
                else:
                    print('The gait events were not in the correct order. Trying peak detection again ' +
                      'with prominence = ' + str(prominences[i+1]) + '.')
            else:
                # everything was in the correct order. continue.
                break
    
        if visualize:
            import matplotlib.pyplot as plt
            plt.close('all')
            plt.figure(1)
            plt.plot(self.markerDict['time'],r_toe_rel_x,label='toe')
            plt.plot(self.markerDict['time'],r_calc_rel_x,label='calc')
            plt.scatter(self.markerDict['time'][rHS], r_calc_rel_x[rHS], color='red', label='rHS')
            plt.scatter(self.markerDict['time'][rTO], r_toe_rel_x[rTO], color='blue', label='rTO')
            plt.legend()

            plt.figure(2)
            plt.plot(self.markerDict['time'],l_toe_rel_x,label='toe')
            plt.plot(self.markerDict['time'],l_calc_rel_x,label='calc')
            plt.scatter(self.markerDict['time'][lHS], l_calc_rel_x[lHS], color='red', label='lHS')
            plt.scatter(self.markerDict['time'][lTO], l_toe_rel_x[lTO], color='blue', label='lTO')
            plt.legend()

        # Find the number of gait cycles for the foot of interest.
        if leg=='auto':
            # Find the last HS of either foot.
            if rHS[-1] > lHS[-1]:
                leg = 'r'
            else:
                leg = 'l'
    
        # Find the number of gait cycles for the foot of interest.
        if leg == 'r':
            hsIps = rHS
            toIps = rTO
            hsCont = lHS
            toCont = lTO
        elif leg == 'l':
            hsIps = lHS
            toIps = lTO
            hsCont = rHS
            toCont = rTO

        if len(hsIps)-1 < n_gait_cycles:
            print('You requested {} gait cycles, but only {} were found. '
                  'Proceeding with this number.'.format(n_gait_cycles,len(hsIps)-1))
            n_gait_cycles = len(hsIps)-1
        if n_gait_cycles == -1:
            n_gait_cycles = len(hsIps)-1
            print('Processing {} gait cycles, leg: '.format(n_gait_cycles) + leg + '.')
        
        # Ipsilateral gait events: heel strike, toe-off, heel strike.
        gaitEvents_ips = np.zeros((n_gait_cycles, 3),dtype=int)
        # Contralateral gait events: toe-off, heel strike.
        gaitEvents_cont = np.zeros((n_gait_cycles, 2),dtype=int)
        if n_gait_cycles <1:
            raise Exception('Not enough gait cycles found.')

        for i in range(n_gait_cycles):
            # Ipsilateral HS, TO, HS.
            gaitEvents_ips[i,0] = hsIps[-i-2]
            gaitEvents_ips[i,2] = hsIps[-i-1]
        
            # Iterate in reverse through ipsilateral TO, finding the one that
            # is within the range of gaitEvents_ips.
            toIpsFound = False
            for j in range(len(toIps)):
                if toIps[-j-1] > gaitEvents_ips[i,0] and toIps[-j-1] < gaitEvents_ips[i,2] and not toIpsFound:
                    gaitEvents_ips[i,1] = toIps[-j-1]
                    toIpsFound = True

            # Contralateral TO, HS.
            # Iterate in reverse through contralateral HS and TO, finding the
            # one that is within the range of gaitEvents_ips
            hsContFound = False
            toContFound = False
            for j in range(len(toCont)):
                if toCont[-j-1] > gaitEvents_ips[i,0] and toCont[-j-1] < gaitEvents_ips[i,2] and not toContFound:
                    gaitEvents_cont[i,0] = toCont[-j-1]
                    toContFound = True
                
            for j in range(len(hsCont)):
                if hsCont[-j-1] > gaitEvents_ips[i,0] and hsCont[-j-1] < gaitEvents_ips[i,2] and not hsContFound:
                    gaitEvents_cont[i,1] = hsCont[-j-1]
                    hsContFound = True
        
            # Skip this step if no contralateral peaks fell within ipsilateral events
            # This can happen with noisy data with subject far from camera. 
            if not toContFound or not hsContFound:                   
                print('Could not find contralateral gait event within ' + 
                               'ipsilateral gait event range ' + str(i+1) + 
                               ' steps until the end. Skipping this step.')
                gaitEvents_cont[i,:] = -1
                gaitEvents_ips[i,:] = -1
    
        # Remove any nan rows
        mask_ips = (gaitEvents_ips == -1).any(axis=1)
        if all(mask_ips):
            raise Exception('No good steps for ' + leg + ' leg.')
        gaitEvents_ips = gaitEvents_ips[~mask_ips]
        gaitEvents_cont = gaitEvents_cont[~mask_ips]
        
        # Convert gaitEvents to times using self.markerDict['time'].
        gaitEventTimes_ips = self.markerDict['time'][gaitEvents_ips]
        gaitEventTimes_cont = self.markerDict['time'][gaitEvents_cont]
                        
        gaitEvents = {'ipsilateralIdx':gaitEvents_ips,
                      'contralateralIdx':gaitEvents_cont,
                      'ipsilateralTime':gaitEventTimes_ips,
                      'contralateralTime':gaitEventTimes_cont,
                      'eventNamesIpsilateral':['HS','TO','HS'],
                      'eventNamesContralateral':['TO','HS'],
                      'ipsilateralLeg':leg}
    
        return gaitEvents

class gait_analysis_lowprom(gait_analysis_variant):
    """Forward chain, ladder walked from the lowest prominence up."""
    PROMINENCES = [0.2, 0.25, 0.3]


class gait_analysis_backward(gait_analysis_variant):
    """Backward chain, OpenCap's ladder."""
    EXPECTED_ORDER = sc.ORDER_BACKWARD


class gait_analysis_backward_lowprom(gait_analysis_variant):
    """Backward chain, ladder walked from the lowest prominence up."""
    EXPECTED_ORDER = sc.ORDER_BACKWARD
    PROMINENCES = [0.2, 0.25, 0.3]
