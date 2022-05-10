from scipy.signal import detrend, find_peaks

import numpy as np
import pandas as pd
import sys

from RRest.common.rpeak_detection import (
	PeakDetector
	)
from RRest.preprocess.band_filter import BandpassFilter
from scipy import signal

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

	# Step 1 preprocess with butterworth filter - 0.1-0.5 -> depend on the device
	if preprocess:
		sig = preprocess_signal(sig,fs)

	# Step 2
	local_max = signal.argrelmax(sig,order=2) # if the diff is greater than 2 continuous points
	local_min = signal.argrelmin(sig,order=2)

	# Step 3 define the local max threshold by taking the 0.75 quantile
	max_threshold = np.quantile(local_max,0.75)*0.2

	print(max_threshold)
	return
