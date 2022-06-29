import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from mne.filter import filter_data, resample
from scipy.signal import detrend, find_peaks
import numpy as np
import pandas as pd
from RRest.preprocess.band_filter import BandpassFilter

train_data = np.loadtxt('dataset/Khoa1waves.asc', dtype=None, delimiter='\t', skiprows=2)
df = pd.DataFrame(train_data, columns=["Time", "ECG1", "Pleth", "Resp"])
# Prepare filter and filter signal
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

cutoff = np.quantile(np.abs(examined_segment), 0.9)
examined_segment[np.abs(examined_segment) < cutoff] = 0

# Load and preprocess data
df_ecg = np.array(df["ECG1"])
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
ecg = ecg[int(start * sf):int((start + window) * sf)]

# R-R peaks detection
rr, _ = find_peaks(ecg, distance=40, height=0.5)

# plt.plot(ecg)
# plt.plot(rr, ecg[rr], 'o')
# plt.title('ECG signal')
# plt.xlabel('Samples')
# _ =plt.ylabel('Voltage')

# R-R interval in ms
rr = (rr / sf) * 1000
rri = np.diff(rr)


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
breath_rate = 60 / np.diff(resp_peaks)
print(breath_rate)
print(len(breath_rate))

# Plot the EDR waveform
plt.plot(filtered_segment, '-')
plt.plot(resp_peaks, filtered_segment[resp_peaks], 'o')
_ = plt.title('ECG derived respiration')
plt.show()
