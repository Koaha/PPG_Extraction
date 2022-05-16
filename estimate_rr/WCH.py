import numpy as np
from scipy import signal
from scipy.signal import resample,detrend
import scipy
from RRest.preprocess.band_filter import BandpassFilter
import plotly.graph_objects as go

def preprocess_signal(sig,fs,filter_type="butterworth",highpass=0.1,lowpass=1.5,
					  degree =5,resampling_rate = 6,cutoff=False,cutoff_quantile=0.9):
	# ranging from 6 - 90 breathpm. Resampling to 6Hz -> better trend detection
	#Prepare and filter signal
	hp_cutoff_order = [highpass, degree]
	lp_cutoff_order = [lowpass, degree]
	filt = BandpassFilter(band_type=filter_type, fs=fs)
	filtered_segment = filt.signal_highpass_filter(sig, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
	filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=lp_cutoff_order[0], order=lp_cutoff_order[1])
	if cutoff:
		cutoff = np.quantile(np.abs(filtered_segment),cutoff_quantile)
		filtered_segment[np.abs(filtered_segment)<cutoff]=0

	filtered_segment = resample(filtered_segment,int(resampling_rate/fs*len(filtered_segment)))
	filtered_segment = detrend(filtered_segment,overwrite_data=True)
	return filtered_segment

def get_rr(sig,fs,preprocess=True,downsample_fs = 4):
	if preprocess:
		sig = preprocess_signal(sig,fs)
	# Find the welch periodogram
	segment_length = min(1024,len(sig))#np.power(2,downsample_fs)
	overlap = int(segment_length/2)
	f,Pxx  = signal.welch(sig,fs,nperseg=1024,noverlap=overlap)
	print(Pxx)
	fig = go.Figure()
	# fig.add_trace(go.Scatter(x=np.arange(len(sig)),y=sig,line=dict(color='blue')))
	fig.add_trace(go.Scatter(x=f, y=Pxx, line=dict(color='crimson')))
	# fig.show()
	valid_peaks = find_spectral_peak(spectral_power=Pxx,frequency=f)
	print(f)
	return f[valid_peaks]

def find_spectral_peak(spectral_power, frequency):
	cand_els = []
	# fig = go.Figure()
	# fig.add_trace(go.Scatter(x=np.arange(len(spectral_power)), y=spectral_power, line=dict(color='crimson')))
	# fig.show()

	spectral_peaks = scipy.signal.argrelmax(spectral_power, order=1)[0]
	power_dev = spectral_power-np.min(spectral_power)

	valid_signal = np.where((frequency[spectral_peaks] > 0) & (frequency[spectral_peaks] < 2))
	return frequency[spectral_peaks[valid_signal]]
