import scipy.signal
from scipy.signal import detrend, find_peaks
from spectrum import pburg,arburg
import numpy as np
from RRest.preprocess.band_filter import BandpassFilter
from scipy.signal import detrend, resample
from mne.filter import filter_data, resample
from statsmodels.tsa.ar_model import AutoReg

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


def find_spectral_peak(spectral_power, frequency):
	cand_els = []
	spectral_peaks = scipy.signal.find_peaks(spectral_power)
	power_dev = spectral_power-np.min(spectral_power)

	return len(spectral_peaks[0])

#TODO
def get_rr(sig,fs,order=7,preprocess=True):
    if preprocess:
        sig = preprocess_signal(sig,fs)
    detrend(sig,overwrite_data=True)
    sig = resample(sig)

    ar = arburg(sig, order)
    if len(ar) > 1:
        poles = np.roots(ar[0])
    angles = np.angle(poles) * (fs/(2*np.pi))*60 #in bpm
    # find rel poles
    # check if angle in the positive part - see the paper
    rr_range = [np.pi*-0.25, np.pi*0.25]
    rel_pole_els = angles[np.where((angles> rr_range[0])& (angles < rr_range[1]))]

    model_angle = np.angle(rel_pole_els)
    model_pole_mag = np.abs(poles)
    #select the pole with the greatest magnitude as the respiratory pole

    return