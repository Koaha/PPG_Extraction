import scipy.signal
from scipy.signal import detrend, find_peaks
from spectrum import pburg,arburg
import numpy as np
from RRest.preprocess.band_filter import BandpassFilter
from scipy.signal import detrend, resample
from mne.filter import filter_data, resample
from statsmodels.tsa.ar_model import AutoReg

def preprocess_signal(sig,fs,filter_type="bessel",highpass=5,lowpass=10):
	# Convert the source frequency to frequency in second unit
	# by resampling. Or convert the cutoff threshold.
	# fs sample-> 1s
	#Prepare and filter signal
	hp_cutoff_order = [highpass, 1]
	lp_cutoff_order = [lowpass, 1]
	filt = BandpassFilter(band_type=filter_type, fs=fs)
	filtered_segment = filt.signal_highpass_filter(sig, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
	filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=lp_cutoff_order[0], order=lp_cutoff_order[1])
	cutoff = np.quantile(np.abs(filtered_segment),0.9)
	filtered_segment[np.abs(filtered_segment)<cutoff]=0
	return filtered_segment

#======================================================================
def get_rr(sig, fs, preprocess=True):
	if preprocess:
		sig = preprocess_signal(sig,fs)

	#detrend & downsampling
	detrend(sig,overwrite_data=True)
	sig = resample(sig)

	# apply power spectrum autoregression

	nfft = 1024

	psd_list = []
	p_F = np.arange(0, 0.5, 1 / nfft)
	for order in range(2,21):
		# arburg(sig, order,NFFT=nfft, scale_by_freq=True)
		p = pburg(sig, order,NFFT=nfft, scale_by_freq=True)
		psd = p.psd
		psd_list.append(psd)

	p_F = np.arange(0,0.5,1/nfft)

	spectral_power = np.median(np.array(psd_list),axis=0)
	spectral_peak = find_spectral_peak(spectral_power,p_F)
	return spectral_peak

def find_spectral_peak(spectral_power, frequency):
	cand_els = []
	spectral_peaks = scipy.signal.find_peaks(spectral_power)
	power_dev = spectral_power-np.min(spectral_power)

	return len(spectral_peaks[0])