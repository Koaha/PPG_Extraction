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

# def get_troughs(sig, rr_index):
#     trough_indices = []
#     for i,j in zip(rr_index[:-2],rr_index[1:]):
#         t_index = np.argmin(np.abs(np.gradient(sig[i:j])))
#         trough_indices.append(i+t_index)
#     return np.array(trough_indices)

# def get_rr(sig, fs, preprocess=True):
# 	if preprocess:
# 		sig = preprocess_signal(sig,fs)
# 	# R-R peaks detection
# 	rr, _ = find_peaks(sig, distance=40, height=0.5)
# 	# remove the local minimum
# 	unified_rr = np.delete(rr, np.argmin(sig[rr]))
# 	troughs = get_troughs(sig, unified_rr)
#
# 	# find relevant peaks and troughs
# 	q3 = np.quantile(sig[rr], 0.75)
# 	thresh = q3 * 0.9
#
# 	rel_peaks = unified_rr[sig[unified_rr] > thresh]
# 	rel_troughs = troughs[sig[troughs] < 0]
#
# 	# find valid breathing cycles. start with  a peak.
# 	cycle_duration = []
# 	for i, j in zip(rel_peaks[:-2], rel_peaks[1:]):
# 		# cycles_rel_troughs = rel_troughs[(rel_troughs > i) and (rel_troughs<j) ]
# 		cycles_rel_troughs = rel_troughs[np.where((rel_troughs > i) & (rel_troughs < j))]
# 		if len(cycles_rel_troughs) == 1:
# 			cycle_duration.append((j - i) * 1000 / fs)
#
# 	average_breath_duration = np.mean(cycle_duration)
# 	resp_res = 60 / average_breath_duration * 1000
# 	return resp_res

def get_rr(sig,fs,preprocess=True):
	# Step 1 preprocess with butterworth filter - 0.1-0.5 -> depend on the device
	if preprocess:
		# pro_sig = preprocess_signal(sig, fs)
		sig = preprocess_signal(sig, fs)
	# fig = go.Figure()
	# fig.add_trace(go.Scatter(x=np.arange(len(sig)),y=sig,line=dict(color='blue')))
	# fig.add_trace(go.Scatter(x=np.arange(len(pro_sig)), y=pro_sig,line=dict(color='crimson')))
	# fig.show()
	# Step 2
	local_max = signal.argrelmax(sig, order=1)[0] # if the diff is greater than 2 continuous points
	local_min = signal.argrelmin(sig, order=1)[0]

	# Step 3 define the local max threshold by taking the 0.75 quantile
	max_threshold = np.quantile(sig[local_max], 0.75) * 0.2

	#Step 4 find the valid resp cycle
	resp_markers = get_valid_rr(sig,local_min,local_max,max_threshold)
	print(len(resp_markers))

def get_valid_rr(sig,local_min,local_max,thres):
	extrema_indices = np.sort(list(local_min)+list(local_max))
	resp_markers = []
	rel_peaks = local_max[sig[local_max]>thres]
	rel_troughs = local_min[sig[local_min]<0]
	for i in range(len(rel_peaks)-1):
		cyc_rel_troughs = (np.where((rel_troughs > rel_peaks[i]) & \
						   (rel_troughs < rel_peaks[i + 1])))[0]
		if len(cyc_rel_troughs) == 1:
			resp_markers.append((rel_peaks[i], rel_peaks[i + 1]))
	return resp_markers