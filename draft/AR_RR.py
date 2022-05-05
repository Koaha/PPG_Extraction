import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.interpolate import splrep, splev
from mne.filter import filter_data, resample
from scipy.signal import detrend, find_peaks
from statsmodels.tsa.ar_model import AutoReg

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

examined_segment = filtered_segment[100000:208000]

cutoff = np.quantile(np.abs(examined_segment),0.9)
examined_segment[np.abs(examined_segment)<cutoff]=0

# Load and preprocess data
df_ecg = np.array( df["ECG1"])
ecg = df_ecg[90000:508000]
sf_ori = sampling_rate
sf = 100
# sf = 100
dsf = sf / sf_ori
ecg = resample(ecg, dsf)
ecg = filter_data(ecg, sf, 2, 30, verbose=0)

# Select only a 60 sec window
window = 60
start = 12
ecg = ecg[int(start*sf):int((start+window)*sf)]

#======================================================================

fs = sf;


# resample the time series @ 2Hz
fs_down = 2;
y = resample(ecg, dsf)
# y = interp1(1:numel(thorax), thorax, 1: 1 / 2:20, 'spline');
y = y - np.mean(y);

# % Applying the Autoregressive Model method model y using AR order 10
# a = arburg(y, 10);
ar_model = AutoReg(y, lags=10).fit()
ar = ar_model.predict()

# % obtain the poles of this AR
ar = np.nan_to_num(ar,nan=0)
r = np.roots(ar);

print(r)
filtered_r = [i for i in r if (np.angle(i)>=10/60*2*np.pi/fs_down)]
filtered_r = [i for i in filtered_r if (np.angle(i)<25/60*2*np.pi/fs_down)]
print(len(filtered_r))
# % searching for poles only between 10 Hz to 25 Hz
# r(angle(r) <= f_low / 60 * 2 * pi / fs_down) = [];
# r(angle(r) > f_high / 60 * 2 * pi / fs_down) = [];
# r = sort(r, 'descend');
# # % plot(r, 'o')
#
# # % Determine the respiratory rate
RR = 60*np.angle(np.max(filtered_r)) * fs_down /2/np.pi
print(RR)

