import scipy
import numpy as np
import pandas as pd
from common.rpeak_detection import (
	PeakDetector
	)
from preprocess.band_filter import BandpassFilter
from scipy import signal
import plotly.express as px
import plotly.graph_objects as go

train_data = np.loadtxt('dataset/Khoa1waves.asc', dtype=None, delimiter='\t',skiprows=2)
df = pd.DataFrame(train_data,columns=["Time","ECG1","Pleth","Resp"])
#Prepare filter and filter signal
sampling_rate = 300
hp_cutoff_order = [5, 1]
lp_cutoff_order = [10, 1]
primary_peakdet = 7
filt = BandpassFilter(band_type='bessel', fs=sampling_rate)
filtered_segment = filt.signal_highpass_filter(df["Resp"], cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=lp_cutoff_order[0], order=lp_cutoff_order[1])
# filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
# filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])

examined_segment = filtered_segment[90000:108000]

cutoff = np.quantile(np.abs(examined_segment),0.9)
examined_segment[np.abs(examined_segment)<cutoff]=0

df_plot = pd.DataFrame(dict(x=np.arange(len(filtered_segment))[90000:108000],y=examined_segment))
print(df_plot["x"])
fig = px.line(df_plot,x="x",y="y")
# fig = go.Figure()
# fig.add_trace(go.scatter(x=np.arange(len(filtered_segment)), y=filtered_segment, name='Pleth',
#                          line=dict(color='firebrick', width=4,
#                               dash='dash') # dash options include 'dash', 'dot', and 'dashdot'
# ))
fig.show()

def count_resp_impedance(inp):
	cutoff = np.quantile(np.abs(inp),0.9)
	inp[np.abs(inp) < cutoff] = 0

	local_maxima = signal.argrelmax(inp)[0]
	local_minima = signal.argrelmin(inp)[0]

	peak_threshold = np.quantile(inp[local_maxima], 0.75) * 0.2
	trough_threshold = np.quantile(inp[local_minima], 0.25) * 0.2

	peak_shortlist = np.array([optima for optima in local_maxima
							   if inp[optima] >= peak_threshold])
	trough_shortlist = np.array([optima for optima in local_minima
								 if inp[optima] <= trough_threshold])

	peak_finalist = []
	through_finalist = []
	left_trough = trough_shortlist[0]

	# while len(peak_shortlist)>0 and len(trough_shortlist)>0:


	for i in range(1, len(trough_shortlist)):

		right_trough = trough_shortlist[i]
		peaks = [peak for peak in peak_shortlist
				 if peak < right_trough and peak > left_trough]
		if len(peaks) == 0:
			left_trough = np.array([left_trough, right_trough])
			[np.argmin([inp[left_trough], inp[right_trough]])]
		else:
			peak = peaks[np.argmax(inp[peaks])]
			peak_finalist.append(peak)
			through_finalist.append(left_trough)
			left_trough = right_trough
	through_finalist.append(right_trough)
	return peak_finalist, through_finalist


peak_finalist, through_finalist = count_resp_impedance(examined_segment)

#Prepare primary peak detector and perform peak detection
# detector = PeakDetector()
# peak_list, trough_list = detector.ppg_detector(examined_segment, primary_peakdet)

print(peak_list)
print(trough_list)