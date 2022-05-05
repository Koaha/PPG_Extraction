import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.interpolate import splrep, splev
from mne.filter import filter_data, resample
from scipy.signal import detrend, find_peaks

import scipy
import numpy as np
import pandas as pd
from common.rpeak_detection import (
	PeakDetector
	)
from RRest.preprocess.band_filter import BandpassFilter
from RRest.estimate_rr import CtO

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
ecg = ecg[int(start*sf):int((start+window)*sf)]


resp = CtO.get_rr(ecg,sf)
print(resp)
