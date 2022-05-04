from scipy.signal import detrend, find_peaks

import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from common.rpeak_detection import (
	PeakDetector
	)
from preprocess.band_filter import BandpassFilter

train_data = np.loadtxt('../dataset/Khoa1waves.asc', dtype=None, delimiter='\t',skiprows=2)
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



# Interpolate and compute HR
def interp_cubic_spline(rri, sf_up=4):
	"""
    Interpolate R-R intervals using cubic spline.
    Taken from the `hrv` python package by Rhenan Bartels.

    Parameters
    ----------
    rri : np.array
        R-R peak interval (in ms)
    sf_up : float
        Upsampling frequency.

    Returns
    -------
    rri_interp : np.array
        Upsampled/interpolated R-R peak interval array
    """
	rri_time = np.cumsum(rri) / 1000.0
	time_rri = rri_time - rri_time[0]
	time_rri_interp = np.arange(0, time_rri[-1], 1 / float(sf_up))
	tck = splrep(time_rri, rri, s=0)
	rri_interp = splev(time_rri_interp, tck, der=0)
	return rri_interp

def get_rr():

    return

# Load and preprocess data
df_ecg = df["ECG1"]
ecg = df_ecg[90000:508000]
sf_ori = sampling_rate
sf = 100
# sf = 100
dsf = sf / sf_ori
ecg = resample(ecg, dsf)
ecg = filter_data(ecg, sf, 2, 30, verbose=0)

# Select only a 20 sec window
window = 60
start = 1000
ecg = ecg[int(start*sf):int((start+window)*sf)]


# R-R peaks detection
rr, _ = find_peaks(ecg, distance=40, height=0.5)
# remove the local minimum
unified_rr = np.delete(rr,np.argmin(ecg[rr]))

def get_troughs(sig, rr_index):
    trough_indices = []
    for i,j in zip(rr_index[:-2],rr_index[1:]):
        t_index = np.argmin(np.abs(np.gradient(sig[i:j])))
        trough_indices.append(i+t_index)
    return np.array(trough_indices)

troughs = get_troughs(ecg,unified_rr)

# find relevant peaks and troughs
q3 = np.quantile(ecg[rr],0.75)
thresh = q3*0.9

rel_peaks = unified_rr[ecg[unified_rr]>thresh]
rel_troughs = troughs[ecg[troughs]<0]

# find valid breathing cycles. start with  a peak.
cycle_duration = []
for i,j in zip(rel_peaks[:-2],rel_peaks[1:]):
	# cycles_rel_troughs = rel_troughs[(rel_troughs > i) and (rel_troughs<j) ]
	cycles_rel_troughs = rel_troughs[np.where((rel_troughs > i) & (rel_troughs<j))]
	if len(cycles_rel_troughs) == 1:
		cycle_duration.append((j-i)*1000/sf)

average_breath_duration = np.mean(cycle_duration)
resp_res = 60/average_breath_duration*1000

# R-R interval in ms
rr = (rr / sf) * 1000
rri = np.diff(rr)

sf_up = 4
rri_interp = interp_cubic_spline(rri, sf_up)
hr = 1000 * (60 / rri_interp)
print(hr)
print('Mean HR: %.2f bpm' % np.mean(hr))

# Detrend and normalize
edr = detrend(hr)
edr = (edr - edr.mean()) / edr.std()

hp_cutoff_order = [8, 1]
lp_cutoff_order = [60, 1]
primary_peakdet = 7
filt = BandpassFilter(band_type='bessel', fs=sampling_rate)
filtered_segment = filt.signal_highpass_filter(edr, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=lp_cutoff_order[0], order=lp_cutoff_order[1])

# Find respiratory peaks
resp_peaks, _ = find_peaks(filtered_segment, height=0, distance=sf_up)

# Convert to seconds
resp_peaks = resp_peaks
resp_peaks_diff = np.diff(resp_peaks) / sf_up

print(resp_peaks)
breath_rate = 60/np.diff(resp_peaks)
print(breath_rate)
print(len(breath_rate))

# Plot the EDR waveform
plt.plot(filtered_segment, '-')
plt.plot(resp_peaks, filtered_segment[resp_peaks], 'o')
_ = plt.title('ECG derived respiration')
plt.show()