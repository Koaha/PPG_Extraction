from scipy.signal import find_peaks,detrend
import plotly.express as px
from plotly import graph_objects as go
import numpy as np
from scipy import signal
from RRest.preprocess.band_filter import BandpassFilter

def preprocess_signal(sig,fs,filter_type="butterworth",
                      highpass=0.1,lowpass=0.25,degree=1,
                      cutoff=False,cutoff_quantile=0.8):
    hp_cutoff_order = [highpass, degree]
    lp_cutoff_order = [lowpass, degree]
    filt = BandpassFilter(band_type=filter_type, fs=fs)
    filtered_segment = filt.signal_highpass_filter(sig, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
    filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=lp_cutoff_order[0], order=lp_cutoff_order[1])
    if cutoff:
        cutoff = np.quantile(np.abs(filtered_segment), cutoff_quantile)
        filtered_segment[np.abs(filtered_segment) < cutoff] = 0
    return filtered_segment

def get_rr(sig,fs,preprocess=True):
    if preprocess:
        sig = preprocess_signal(sig,fs)
    local_max = signal.argrelmax(sig)[0]
    local_min = signal.argrelmin(sig)[0]

    #Step 1
    #elimination of peaks less than the mean,
    # and troughs greater than the mean
    threshold_max = np.mean(sig[local_max])
    #local_max_list indices
    rel_peaks_indices = np.where(sig[local_max]>threshold_max)[0]

    threshold_min = np.mean(sig[local_min])
    rel_troughs_indices= np.where(sig[local_min]<threshold_min)[0]

    #Step 2
    #elimination of peaks (and troughs) within 0.5s of the previous peak (or trough)
    peaks_interval = np.diff(local_max[rel_peaks_indices])
    rel_peaks_indices = np.delete(rel_peaks_indices,np.where(peaks_interval<0.5*fs)[0]+1)
    troughs_interval = np.diff(local_min[rel_troughs_indices])
    rel_troughs_indices = np.delete(rel_troughs_indices, np.where(troughs_interval < 0.5 * fs)[0] + 1)

    #Step 3
    #elimination of peaks (and troughs) which are immediately followed by a peak (or trough)
    #combine index of rel_peaks & rel_trough
    extrema_indices = np.sort(list(local_min[rel_troughs_indices])+list(local_max[rel_peaks_indices]))
    remove_indices = []
    sign = 0
    for i in range(len(extrema_indices)):
        if sign == 0:
            sign = get_sign(extrema_indices[i],local_min[rel_troughs_indices],local_max[rel_peaks_indices])
        elif sign == get_sign(extrema_indices[i],local_min[rel_troughs_indices],local_max[rel_peaks_indices]):
            # remove_indices.append(extrema_indices[i-1])
            remove_indices.append(i - 1)
            sign = 0
        else:
            sign = (-1)*sign
    rel_extrema = np.delete(extrema_indices,remove_indices)

    final_peaks = np.intersect1d(local_max[rel_peaks_indices],rel_extrema)
    final_troughs = np.intersect1d(local_min[rel_troughs_indices], rel_extrema)
    return final_peaks



def get_sign(extrema_indices,trough_indices,peak_indices):
    if extrema_indices in trough_indices:
        return -1
    return 1
