from mne.filter import filter_data, resample

import scipy
import numpy as np
import pandas as pd
from RRest.estimate_rr import ARM, CtO, Interpolate_RR,AR_RR,ARM,ARP

train_data = np.loadtxt('dataset/Khoa1waves.asc', dtype=None, delimiter='\t',skiprows=2)
df = pd.DataFrame(train_data,columns=["Time","ECG1","Pleth","Resp"])
#Prepare filter and filter signal

# Load and preprocess data
df_ecg = np.array(df["ECG1"])
ecg = df_ecg[90000:508000]
sf_ori = 300
sf = 100
# sf = 100
dsf = sf / sf_ori
ecg = resample(ecg, dsf)
ecg = filter_data(ecg, sf, 2, 30, verbose=0)

# Select only a 20 sec window
window = 60
start = 1000
ecg = ecg[int(start*sf):int((start+window)*sf)]

# resp = CtO.get_rr(ecg,sf)
resp = ARP.get_rr(ecg,sf)
print(resp)
