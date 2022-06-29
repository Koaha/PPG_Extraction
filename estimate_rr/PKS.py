from scipy.signal import find_peaks
import plotly.graph_objects as go
import numpy as np
from scipy import signal
from RRest.preprocess.band_filter import BandpassFilter


def preprocess_signal(sig, fs, filter_type="butterworth", highpass=0.1, lowpass=0.5, degree=1, cutoff=False,
                      cutoff_quantile=0.9):
    # Prepare and filter signal
    hp_cutoff_order = [highpass, degree]
    lp_cutoff_order = [lowpass, degree]
    filt = BandpassFilter(band_type=filter_type, fs=fs)
    filtered_segment = filt.signal_highpass_filter(sig, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
    filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=lp_cutoff_order[0], order=lp_cutoff_order[1])
    if cutoff:
        cutoff = np.quantile(np.abs(filtered_segment), cutoff_quantile)
        filtered_segment[np.abs(filtered_segment) < cutoff] = 0
    return filtered_segment


def get_rr(sig, fs, preprocess=True):
    if preprocess:
        sig = preprocess_signal(sig, fs)
    local_max = signal.argrelmax(sig)
    thres = np.quantile(sig[local_max], 0.75) * 0.5
    peaks = find_peaks(sig, height=thres)[0]
    # fig = go.Figure()
    # fig.add_traces(go.Scatter())
    peaks_t = np.diff(peaks) * (1 / fs)
    breath_peaks = signal.argrelmax(peaks_t)[0]

    fig = go.Figure()
    fig.add_traces(go.Scatter(x=np.arange(len(peaks_t)), y=peaks_t, line=dict(color='blue')))
    fig.add_traces(go.Scatter(mode='markers', x=breath_peaks, y=peaks_t[breath_peaks],
                              marker=dict(color='crimson', size=4)))
    fig.show()
    return breath_peaks
