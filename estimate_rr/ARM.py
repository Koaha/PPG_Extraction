import scipy.signal
from scipy.signal import detrend, find_peaks,resample
from plotly import express as px
import plotly.graph_objects as go
from spectrum import pburg,arburg
import numpy as np
from RRest.preprocess.band_filter import BandpassFilter
from scipy.signal import detrend, resample
from mne.filter import filter_data, resample
from statsmodels.tsa.ar_model import AutoReg

#EF3 Auto-regressive spectral analysis using the median spectrum for model orders 2–20 (Shah et al 2015).

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


#======================================================================
def get_rr(sig, fs, upperbound_freq=6,preprocess=True):
	if preprocess:
		sig = preprocess_signal(sig,fs,resampling_rate=upperbound_freq)

	fig = go.Figure()
	# fig.add_trace(go.Scatter(x=np.arange(len(sig)),y=sig,line=dict(color='blue')))
	fig.add_trace(go.Scatter(x=np.arange(len(sig)), y=sig,line=dict(color='crimson')))
	fig.show()

	# apply power spectrum autoregression

	nfft = 1024

	psd_list = []
	for order in range(2,21):
		# arburg(sig, order,NFFT=nfft, scale_by_freq=True)
		p = pburg(sig, order,NFFT=nfft, scale_by_freq=True)
		psd = p.psd
		psd_list.append(psd)

	p_F = np.arange(0,upperbound_freq,upperbound_freq/(nfft*0.5))

	spectral_power = np.median(np.array(psd_list),axis=0)
	spectral_peak = find_spectral_peak(p,spectral_power,p_F)
	return spectral_peak

def find_spectral_peak(p,spectral_power, frequency):
	cand_els = []
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=np.arange(len(spectral_power)), y=spectral_power, line=dict(color='crimson')))
	fig.show()
	# spectral_peaks = scipy.signal.find_peaks(spectral_power)
	spectral_peaks = scipy.signal.argrelmax(spectral_power, order=1)[0]
	power_dev = spectral_power-np.min(spectral_power)

	ar= p.ar
	angles = np.angle(ar)
	valid_signal = np.where((frequency[spectral_peaks] > 0) & (frequency[spectral_peaks] < 2))
	return len(valid_signal[0])
