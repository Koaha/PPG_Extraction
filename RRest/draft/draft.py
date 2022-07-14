"""Signal quality indexes based on dynamic template matching
"""
import numpy as np
from RRest.common.generate_template import (
    ppg_absolute_dual_skewness_template,
    ppg_dual_double_frequency_template,
    ppg_nonlinear_dynamic_system_template,
    ecg_dynamic_template
)
from scipy.signal import find_peaks
from RRest.preprocess.preprocess_signal import smooth
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
from librosa import segment

DATASET_FOLDER = "../../dataset/"
G_FOLDER = "../../dataset/G"
NG_1_FOLDER = "../../dataset/NG_1"
NG_2_FOLDER = "../../dataset/NG_2"
PPG_FOLDER = "../../dataset/ppg"

good_files = [".".join(x.split(".")[:-1])+".csv" for x in next(os.walk(G_FOLDER), (None, None, []))[2]]
ng_1_files = [".".join(x.split(".")[:-1])+".csv" for x in next(os.walk(NG_1_FOLDER), (None, None, []))[2]]
ng_2_files = [".".join(x.split(".")[:-1])+".csv" for x in next(os.walk(NG_2_FOLDER), (None, None, []))[2]]


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
                           nfft=None,noverlap=None,
                           plot_image=False
                           ):
    window = np.hanning(window_size)
    fig, ax = plt.subplots()

    f, t, Sxx = \
        spectrogram(np.array(data),nfft=nfft,nperseg=window_size,
                    fs=fs, noverlap=noverlap,window=window)
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


    # 1-beat      beat_length                         beat_length/fs(s)
    # r-beat      fs                                  1(s)
    #
    # after scale
    # 1-beat      template_size                       beat_length/fs(s)
    #             template_size/(beat_length/fs)      1s


def get_critical_points(s):

    trough_start = np.argmin(s[:int(len(s)/2)])
    trough_end = np.argmin(s[int(len(s)/2):])+int(len(s)/2)
    systolic_peak = np.argmax(s)

    critical_points = [trough_start,trough_end,systolic_peak]

    ds = np.gradient(s)
    dss = np.gradient(smooth(ds))
    scaler = MinMaxScaler(feature_range=(min(s), max(s)))
    ds = scaler.fit_transform(ds.reshape(-1, 1)).reshape(-1)
    dss = scaler.fit_transform(dss.reshape(-1, 1)).reshape(-1)

    # second derivative -> shift back 2 indices
    diastolic_peak = int(np.median(find_peaks(-dss[systolic_peak:])[0])) + systolic_peak-2
    critical_points.append(diastolic_peak)

    # second derivative -> shift forward 2 indices
    systolic_diastolic_connector = find_peaks(dss[systolic_peak:])[0]
    if len(systolic_diastolic_connector) > 0:
        systolic_diastolic_connector = systolic_diastolic_connector[0] + systolic_peak+2
    else:
        systolic_diastolic_connector = int(np.mean([systolic_peak,diastolic_peak]))
    critical_points.append(systolic_diastolic_connector)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(systolic_diastolic_connector,len(s)),
        y=s[systolic_diastolic_connector:],
        fill='tozeroy',
        # textposition='inside',
        # text=str(systolic_area)
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(systolic_diastolic_connector+1),
        y=s[:systolic_diastolic_connector+1],
        fill='tozeroy',
        # textposition='inside',
        # text=str(diastolic_area)
    ))
    fig.add_trace(go.Scatter(
        x=critical_points,
        y=s[critical_points],
        mode='markers',
        marker=dict(size=12,color='MediumPurple',
                    line=dict(width=2,color='DarkSlateGrey'))
    ))

    # fig.show()

    return np.sort(critical_points)


systolic_area_list = []
diastolic_area_list = []

for file_name in tqdm(good_files[:15]):
    file_path = os.path.join(PPG_FOLDER,file_name)
    ppg_stable = np.array(pd.read_csv(os.path.join(os.getcwd(), file_path), header=None)).reshape(-1)
    detector = PeakDetector(wave_type='ppg')
    peak_milestones, trough_milestones = detector.ppg_detector(ppg_stable)

    beats = [ppg_stable[trough_milestones[i]:
                        trough_milestones[i+1]] for i in range(len(trough_milestones)-1)]


    template_size = 100
    template_type = 2
    resampling = []
    resampling_ref = []
    beat_list = []
    reference_list = []
    beat_length_list = []
    for beat in beats:
        beat_length_list.append(len(beat))
        beat = resample(beat, template_size)
        scaler = MinMaxScaler()
        beat = scaler.fit_transform(beat.reshape(-1, 1)).reshape(-1)
        resampling = resampling + list(beat)
        reference = get_template(template_type)
        resampling_ref = resampling_ref + list(reference)

        beat_list.append(beat)
        reference_list.append(reference)

    fs = 100
    beat_length = int(np.mean(beat_length_list))
    beat_list = np.array(beat_list)
    beat = np.apply_along_axis(np.mean,axis=0,arr=beat_list)
    f, t, Sxx = plot_spectrogram_scipy(beat, nfft=8096,
                                           noverlap = 2,
                                           fs= int(template_size/(beat_length/fs)))

    f_ref, t_ref, Sxx_ref = \
    plot_spectrogram_scipy(reference,nfft=8096,
                                   noverlap = 2,
                                    fs= int(template_size/(beat_length/fs)))

    xsim = segment.cross_similarity(beat,reference,mode='distance')
    xsim_aff = librosa.segment.cross_similarity(beat,reference, mode='affinity')
    hop_length=1024
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    imgsim = librosa.display.specshow(xsim, x_axis='s', y_axis='s',cmap='magma_r',
                                          hop_length=hop_length, ax=ax[0])
    ax[0].set(title='Binary cross-similarity (symmetric)')
    imgaff = librosa.display.specshow(xsim_aff, x_axis='s', y_axis='s',
                                          cmap='magma_r', hop_length=hop_length, ax=ax[1])
    ax[1].set(title='Cross-affinity')
    ax[1].label_outer()
    fig.colorbar(imgsim, ax=ax[0], orientation='horizontal', ticks=[0, 1])
    fig.colorbar(imgaff, ax=ax[1], orientation='horizontal')
    # fig.show()

        # scale beat
    f = f[f < 5]
    f_ref = f_ref[f_ref < 5]

    # beat_resample_length = int(len(t)*beat_length/fs)
    beat_resample_length = int(len(t))
    beat = resample(beat,beat_resample_length)
    scaler = MinMaxScaler(feature_range=(min(f), max(f)))
    beat = scaler.fit_transform(beat.reshape(-1, 1)).reshape(-1)

    # scale beat
    reference = resample(reference, beat_resample_length)
    scaler = MinMaxScaler(feature_range=(min(f_ref), max(f_ref)))
    reference = scaler.fit_transform(reference.reshape(-1, 1)).reshape(-1)


    critical_points = get_critical_points(beat)
    trough_start = critical_points[0]
    trough_end = critical_points[-1]
    systolic_peak = critical_points[1]
    systolic_diastolic_connector = critical_points[2]
    diastolic_peak = critical_points[3]

    systolic_area = np.trapz(y=beat[trough_start:systolic_diastolic_connector])

    diastolic_area = np.trapz(y=beat[systolic_diastolic_connector:trough_end])

    # ratio from the peak to the connector
    ratio_con_sys = beat[systolic_diastolic_connector] / beat[systolic_peak]

    # ratio between the diastolic peak and the connector
    ratio_con = beat[systolic_diastolic_connector] / beat[systolic_peak]

    # the width of the systolic peak
    width_start = np.where(beat > beat[systolic_diastolic_connector])[0][0]
    upper_systolic_area = np.trapz(beat[width_start:systolic_diastolic_connector]
                                         -beat[systolic_diastolic_connector])
    systolic_area_ratio = upper_systolic_area / systolic_area
    # width_ratio = t[systolic_diastolic_connector] - t[width_start]



    systolic_area_list.append(systolic_area)
    diastolic_area_list.append(diastolic_area)


    t = t[:len(beat)]
    Sxx = Sxx[:len(f), :len(beat)]

    t_ref = t_ref[:len(reference)]
    Sxx_ref = Sxx_ref[:len(f_ref), :len(reference)]

    Sxx_diff = np.abs((Sxx_ref) - (Sxx))


    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=t,
                             y=f,
                             z=np.log(Sxx_diff)))
    fig.add_trace(go.Scatter(
        x=t,
        y=beat
    ))
    fig.add_trace(go.Scatter(
        x=t_ref,
        y=reference
    ))
    # fig.show()

    # fig = go.Figure()
    # fig.add_trace(go.Heatmap(x=t,
    #                          y=f,
    #                          z=np.log(Sxx)))
    # fig.add_trace(go.Scatter(
    #     x=t,
    #     y=beat
    # ))
    # fig.show()
    #
    # fig = go.Figure()
    # fig.add_trace(go.Heatmap(x=t_ref,
    #                          y=f_ref,
    #                          z=np.log(Sxx_ref)))
    # fig.add_trace(go.Scatter(
    #     x=t_ref,
    #     y=reference
    # ))
    # fig.show()

print("done")