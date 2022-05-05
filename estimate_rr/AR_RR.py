import numpy as np
from RRest.preprocess.band_filter import BandpassFilter
from mne.filter import filter_data, resample
from statsmodels.tsa.ar_model import AutoReg

def preprocess_signal(sig,fs,filter_type="bessel",highpass=5,lowpass=10):
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
    sf_ori = fs
    sf = 100
    # sf = 100
    dsf = sf / sf_ori
    ecg = resample(sig, dsf)
    ecg = filter_data(ecg, sf, 2, 30, verbose=0)

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
    return RR

