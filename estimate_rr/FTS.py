from scipy.signal import detrend, find_peaks
from plotly import express as px
import plotly.graph_objects as go
import numpy as np
from scipy import signal
from RRest.preprocess.band_filter import BandpassFilter

def preprocess_signal(sig,fs,filter_type="butterworth",highpass=0.1,lowpass=0.5,degree =1,cutoff=False,cutoff_quantile=0.9):
	#Prepare and filter signal
	hp_cutoff_order = [highpass, degree]
	lp_cutoff_order = [lowpass, degree]
	filt = BandpassFilter(band_type=filter_type, fs=fs)
	filtered_segment = filt.signal_highpass_filter(sig, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
	filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=lp_cutoff_order[0], order=lp_cutoff_order[1])
	if cutoff:
		cutoff = np.quantile(np.abs(filtered_segment),cutoff_quantile)
		filtered_segment[np.abs(filtered_segment)<cutoff]=0
	return filtered_segment
