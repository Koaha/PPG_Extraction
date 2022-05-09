from scipy.signal import detrend, find_peaks

import numpy as np
import pandas as pd
import sys

from RRest.common.rpeak_detection import (
	PeakDetector
	)
from RRest.preprocess.band_filter import BandpassFilter

def preprocess_signal(sig,fs,filter_type="bessel",highpass=5,lowpass=10):
	#Prepare and filter signal
	hp_cutoff_order = [highpass, 1]
	lp_cutoff_order = [lowpass, 1]
	filt = BandpassFilter(band_type=filter_type, fs=fs)
	filtered_segment = filt.signal_highpass_filter(sig, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
	filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=lp_cutoff_order[0], order=lp_cutoff_order[1])
	cutoff = np.quantile(np.abs(filtered_segment),0.9)
	filtered_segment[np.abs(filtered_segment)<cutoff]=0
	return filtered_segment

def get_troughs(sig, rr_index):
    trough_indices = []
    for i,j in zip(rr_index[:-2],rr_index[1:]):
        t_index = np.argmin(np.abs(np.gradient(sig[i:j])))
        trough_indices.append(i+t_index)
    return np.array(trough_indices)

def get_rr(sig, fs, preprocess=True):
	if preprocess:
		sig = preprocess_signal(sig,fs)
	# R-R peaks detection
	rr, _ = find_peaks(sig, distance=40, height=0.5)
	# remove the local minimum
	unified_rr = np.delete(rr, np.argmin(sig[rr]))
	troughs = get_troughs(sig, unified_rr)

	# find relevant peaks and troughs
	q3 = np.quantile(sig[rr], 0.75)
	thresh = q3 * 0.9

	rel_peaks = unified_rr[sig[unified_rr] > thresh]
	rel_troughs = troughs[sig[troughs] < 0]

	# find valid breathing cycles. start with  a peak.
	cycle_duration = []
	for i, j in zip(rel_peaks[:-2], rel_peaks[1:]):
		# cycles_rel_troughs = rel_troughs[(rel_troughs > i) and (rel_troughs<j) ]
		cycles_rel_troughs = rel_troughs[np.where((rel_troughs > i) & (rel_troughs < j))]
		if len(cycles_rel_troughs) == 1:
			cycle_duration.append((j - i) * 1000 / fs)

	average_breath_duration = np.mean(cycle_duration)
	resp_res = 60 / average_breath_duration * 1000
	return resp_res
