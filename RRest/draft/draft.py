"""Signal quality indexes based on dynamic template matching
"""
import numpy as np
from RRest.common.generate_template import (
    ppg_absolute_dual_skewness_template,
    ppg_dual_double_frequency_template,
    ppg_nonlinear_dynamic_system_template,
    ecg_dynamic_template
)
from scipy.spatial.distance import euclidean
from scipy.signal import resample,spectrogram
from sklearn.preprocessing import MinMaxScaler
import librosa
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
import pandas as pd
import os
from tqdm import tqdm
from RRest.common.rpeak_detection import PeakDetector

DATASET_FOLDER = "../../dataset/"
G_FOLDER = "../../dataset/G"
NG_1_FOLDER = "../../dataset/NG_1"
NG_2_FOLDER = "../../dataset/NG_2"
PPG_FOLDER = "../../dataset/ppg"

good_files = [".".join(x.split(".")[:-1])+".csv" for x in next(os.walk(G_FOLDER), (None, None, []))[2]]
ng_1_files = [".".join(x.split(".")[:-1])+".csv" for x in next(os.walk(NG_1_FOLDER), (None, None, []))[2]]
ng_2_files = [".".join(x.split(".")[:-1])+".csv" for x in next(os.walk(NG_2_FOLDER), (None, None, []))[2]]



def dtw_sqi(s, template_type = 1, template_size=100):
    """
    Euclidean distance between signal and its template

    Parameters
    ----------
    s :
        array_like, signal containing int or float values.

    template_sequence :
        array_like, signal containing int or float values.

    Returns
    -------

    """
    s = resample(s, template_size)
    scaler = MinMaxScaler()
    s = scaler.fit_transform(s.reshape(-1,1)).reshape(-1)
    # reference_1= ppg_nonlinear_dynamic_system_template(template_size).reshape(-1)
    # reference_2 = ppg_dual_double_frequency_template(template_size)
    # reference_3 = ppg_absolute_dual_skewness_template(template_size)
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x = np.arange(len(s)), y=s, name="signal"
    # ))
    # fig.add_trace(go.Scatter(
    #     x=np.arange(len(s)), y=reference_1, name="template_1"
    # ))
    # fig.add_trace(go.Scatter(
    #     x=np.arange(len(s)), y=reference_2, name="template_2"
    # ))
    # fig.add_trace(go.Scatter(
    #     x=np.arange(len(s)), y=reference_3, name="template_3"
    # ))
    # fig.show()
    # return
    if template_type == 1:
        reference = ppg_nonlinear_dynamic_system_template(template_size).reshape(-1)
    elif template_type == 2:
        reference = ppg_dual_double_frequency_template(template_size)
    else:
        reference = ppg_absolute_dual_skewness_template(template_size)

    dtw_distances = np.ones((template_size, template_size)) * \
                    np.inf
    # first matching sample is set to zero
    dtw_distances[0, 0] = 0
    cost = 0
    for i in range(template_size):
        cost = cost + euclidean(s[i], reference[i])
    return cost / template_size


def plot_spectrogram_librosa(s,template_size=100):
    fs = 100
    reference_1 = ppg_nonlinear_dynamic_system_template(template_size).reshape(-1)
    reference_2 = ppg_dual_double_frequency_template(template_size)
    reference_3 = ppg_absolute_dual_skewness_template(template_size)

    window_size = template_size//4
    window = np.hanning(window_size)
    stft = librosa.core.spectrum.stft(np.array(s), n_fft=window_size,
                                      window=window)
    out = 2 * np.abs(stft) / np.sum(window)
    spec_data = librosa.amplitude_to_db(out, ref=np.max)
    # For plotting headlessly
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(
        spec_data, sr=template_size,ax=ax, y_axis='hz', x_axis='s')
    return

def plot_spectrogram_scipy(data,
                           fs=100,window_size=4,
                           spec_file_name = 'spec',
                           plot_image=False
                           ):
    window = np.hanning(window_size)
    fig, ax = plt.subplots()

    f, t, Sxx = spectrogram(np.array(data),
                            nfft=80,
                            nperseg=window_size,
                            fs=fs, window=window)
    if plot_image:
        color_norm = matplotlib.colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max())
        pc = ax.pcolormesh(t, f, Sxx, norm=color_norm, cmap='inferno')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Time')
        fig.colorbar(pc)
        fig.savefig(spec_file_name+'.png')
        fig.show()

    return f, t, Sxx

def get_template(template_type=1):
    if template_type == 1:
        reference = ppg_nonlinear_dynamic_system_template(template_size).reshape(-1)
    elif template_type == 2:
        reference = ppg_dual_double_frequency_template(template_size)
    else:
        reference = ppg_absolute_dual_skewness_template(template_size)
    return reference

for file_name in tqdm(good_files[3:4]):
    file_path = os.path.join(PPG_FOLDER,file_name)
    ppg_stable = np.array(pd.read_csv(os.path.join(os.getcwd(), file_path), header=None)).reshape(-1)
    detector = PeakDetector(wave_type='ppg')
    peak_milestones, trough_milestones = detector.ppg_detector(ppg_stable)

    beats = [ppg_stable[trough_milestones[i]:
                        trough_milestones[i+1]] for i in range(len(trough_milestones)-1)]

    template_size = 100
    template_type = 1
    resampling = []
    resampling_ref = []
    for beat in beats:
        beat = resample(beat, template_size)
        scaler = MinMaxScaler()
        beat = scaler.fit_transform(beat.reshape(-1, 1)).reshape(-1)
        resampling = resampling + list(beat)
        reference = get_template(template_type)
        resampling_ref = resampling_ref + list(reference)

    f, t, Sxx = plot_spectrogram_scipy(beat)

    f_ref, t_ref, Sxx_ref = plot_spectrogram_scipy(reference)

    # Sxx_diff = np.abs(np.log(Sxx_ref)-np.log(Sxx))
    Sxx_diff = np.abs((Sxx_ref) - (Sxx))

    # fig, ax = plt.subplots()
    # color_norm = matplotlib.colors.LogNorm(vmin=Sxx_diff.min(), vmax=Sxx_diff.max())
    # pc = ax.pcolormesh(t, f, Sxx_diff, norm=color_norm, cmap='inferno')
    # ax.set_ylabel('Frequency')
    # ax.set_xlabel('Time')
    # fig.colorbar(pc)
    # fig.show()


    #==============================================================

    # x, y = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10))
    # y += 2 * np.sin(x * 2 * np.pi / 10)
    # z = np.exp(-x / 10)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=t,
                               y=f,
                               z=Sxx_diff ))
    fig.add_trace()
    fig.show()

    print("done")