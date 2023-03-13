from scipy.signal import resample
import numpy as np
from scipy import signal
import plotly.graph_objects as go
import matplotlib as plt

from scipy import sparse
from RRest.preprocess.band_filter import BandpassFilter
import pandas  as pd
import ast,os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from datetime import datetime
from ..common.rpeak_detection import PeakDetector
import warnings
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

def synthesize_nonlinear_dynamic_system(duration, trend_frequency=None,
                                        noise_scale=0.05, noise_mean=0.2,
                                        noise_density=0.2,extend_rate=0.2,
                                        resample_rate=2):
    """
    EXPOSE
    :param width:
    :return:
    """
    x1 = 0.15
    x2 = 0.15
    u = 0.5
    beta = 1
    gamma1 = -0.25
    gamma2 = 0.25
    x1_list = [x1]
    x2_list = [x2]

    dt = 0.1
    num_dt = duration/dt

    trough_idx = []

    for t in np.arange(1, num_dt, dt):
        y1 = 0.5 * (np.abs(x1 + 1) - np.abs(x1 - 1))
        y2 = 0.5 * (np.abs(x2 + 1) - np.abs(x2 - 1))
        dx1 = -x1 + (1 + u) * y1 - beta * y2 + gamma1
        dx2 = -x2 + (1 + u) * y2 + beta * y1 + gamma2

        x1 = x1 + dx1 * dt
        x2 = x2 + dx2 * dt

        if np.random.rand()<extend_rate:
            if np.random.rand()<extend_rate:
                num = np.random.randint(6,12)
            else:
                num = np.random.randint(4, 6)
            x2_ = (signal.resample([x2_list[-1],x2],num,signal.windows.hamming(10))[0]).tolist()[:int(num/2)]
            x2_list = x2_list + x2_
        else:
            x2_list.append(x2)
        # x2_list.append(x2)
        x1_list.append(x1)

    trend_t = np.linspace(1, duration, len(x2_list))
    noise_std = (np.max(x2_list) - np.min(x2_list))*noise_scale
    noise_series = np.random.normal(noise_mean, noise_std, len(trend_t))

    noise_sparse = sparse.random(1,len(trend_t),density=noise_density)
    noise_sparse.data[:] = 1
    noise_series = np.multiply(noise_sparse.toarray().reshape(-1), noise_series)

    trend_volts = 2 * np.sin(trend_t / (2*trend_frequency * np.pi)) + noise_series

    trend_list =  trend_volts + x2_list

    # x1_list = signal.normalize(x1_list,[1]*len(x1_list))
    processed_sig = signal.detrend(trend_list)
    bandFilter = BandpassFilter()
    processed_sig = bandFilter.signal_lowpass_filter(processed_sig,4)
    processed_sig = bandFilter.signal_highpass_filter(processed_sig, 0.5)
    processed_sig = signal.resample(processed_sig, int(len(processed_sig) / resample_rate))

    # processed_sig = synthesize_nonlinear_dynamic_system(duration=30, trend_frequency=200,
    #                                                     noise_scale=0.1, noise_mean=0.5,
    #                                                     noise_density=0.3, extend_rate=0.2,
    #                                                     resample_rate=2.5
    #                                                     )
    scaler = MinMaxScaler(
        feature_range=(min(processed_sig), max(processed_sig))
    )
    file_df = pd.read_csv("v2smartcare.csv", warn_bad_lines=True, error_bad_lines=False)
    df = pd.DataFrame()
    pleth_data = []
    file_df['PLETH'].apply(get_flat, flat=pleth_data)
    df['TIMESTAMP_MS'] = np.arange(0, len(pleth_data)) * 10
    df['PLETH'] = pleth_data
    pleth_data = scaler.fit_transform(np.array(pleth_data[:len(processed_sig)]).reshape(-1, 1)).reshape(-1)

    trough_real = signal.find_peaks(-pleth_data,width=15)[0]
    trough_syn = signal.find_peaks(-processed_sig,width=15)[0]




    #=====================================================================
    #               COMBINE SYN AND REAL
    #====================================================================

    ratio_real_syn = 0.2
    ratio_real_syn_idx = (int) (ratio_real_syn*len(trough_syn))
    insertion_idx = np.random.permutation(trough_syn)




    # =====================================================================
    #               END SYN AND REAL
    # ====================================================================
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=np.arange(len(processed_sig)),
            y=pleth_data[:len(processed_sig)],
            mode="lines",
            name="real_data"
        )
    )

    fig.add_trace(go.Scatter(
        x=trough_real,
        y=pleth_data[trough_real],
        marker=dict(color="crimson", size=12),
        mode="markers"
    ))
    # fig.add_traces(go.Scatter(
    #     x=np.arange(len(trend_volts)),
    #     y=trend_volts, mode="lines",
    #     name="trend_t"
    # ))
    fig.add_traces(go.Scatter(
        x=np.arange(len(processed_sig)),
        y=processed_sig, mode="lines",
        name="processed_sig"
    ))
    fig.add_trace(go.Scatter(
        x=trough_syn,
        y=processed_sig[trough_syn],
        marker=dict(color="orange", size=12),
        mode="markers"
    ))

    fig.show()

    return processed_sig

def get_flat(x,flat):
    flat += ast.literal_eval(x)


def plot_processed_signal(processed_sig):
    scaler = MinMaxScaler(
        feature_range=(min(processed_sig), max(processed_sig))
    )
    file_df = pd.read_csv("v2smartcare.csv", warn_bad_lines=True, error_bad_lines=False)
    df = pd.DataFrame()
    pleth_data = []
    file_df['PLETH'].apply(get_flat,flat=pleth_data)
    df['TIMESTAMP_MS'] = np.arange(0,len(pleth_data))*10
    df['PLETH'] = pleth_data
    pleth_data = scaler.fit_transform(np.array(pleth_data[:len(processed_sig)]).reshape(-1, 1)).reshape(-1)


    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
                x=np.arange(len(processed_sig)),
                y=pleth_data[:len(processed_sig)],
                mode="lines",
                name="real_data"
            )
    )
    # fig.add_traces(go.Scatter(
    #     x=np.arange(len(trend_volts)),
    #     y=trend_volts, mode="lines",
    #     name="trend_t"
    # ))
    fig.add_traces(go.Scatter(
        x=np.arange(len(processed_sig)),
        y=processed_sig, mode="lines",
        name="processed_sig"
    ))

    fig.show()

    # peak_detection = signal.argrelmax()


    print("ahihi")
    # local_minima = argrelextrema(np.array(x2_list), np.less)[0]
    # s = np.array(x2_list[local_minima[-2]:local_minima[-1] + 1])
    #
    # rescale_signal = resample(s,width)
    #
    # window = signal.windows.cosine(len(rescale_signal), 0.5)
    # signal_data_tapered = np.array(window) * (rescale_signal - min(rescale_signal))
    #
    # out_scale = MinMaxScaler().fit_transform(
    #     np.array(signal_data_tapered).reshape(-1, 1))
    # return out_scale.reshape(-1)
    return


#
# def extract_ppg_features(fname,patient_id,analysis_duration,feat_PATH):
#     # file_in = os.path.join(PATH, fname)
#
#     feats = {}
#
#     feats['start_time'] = []
#     feats['end_time'] = []
#     feats['systolic_area_list'] = []
#     feats['diastolic_area_list'] = []
#     feats['systolic_area_ratio_list'] = []
#     feats['ratio_con_sys'] = []
#     feats['ratio_con_dias'] = []
#     feats['amplitude_sys'] = []
#     feats['amplitude_dias'] = []
#     feats['rr'] = []
#
#
#     df_sqis = pd.DataFrame()
#     feats['Sxx_list'] = []
#     feat_dict = {}
#     event_time_start = []
#     event_time_end = []
#     df_beat = []
#     df_beat_spo2 = []
#     ppg_files = []
#     for idx,ppg_file in enumerate(tqdm(ppg_files)):
#         try:
#             df_ppg =  pd.read_csv(os.path.join(ppg_path,ppg_file),usecols=range(1,4))
#             if total_duration == 0:
#                 df_segment = df_ppg
#             else:
#                 df_segment = df_segment.append(df_ppg)
#             format = "%Y-%m-%d %H:%M:%S.%f"
#
#             # start_time_converted = datetime.strptime(df_ppg['timestamps'].iloc[0], format)
#             current_start_time = datetime.strptime(df_ppg['timestamps'].iloc[0], format)
#             end_start_time = datetime.strptime(df_ppg['timestamps'].iloc[-1], format)
#
#             next_ppg = pd.read_csv(os.path.join(ppg_path,ppg_files[idx+1]),usecols=range(1,3))
#             next_start_time = datetime.strptime(next_ppg['timestamps'].iloc[0], format)
#
#             total_duration = total_duration + np.round((next_start_time-current_start_time).total_seconds())
#             if total_duration >= analysis_duration:
#                 feat_dict,df_sqis,beat = get_all_features((df_segment[["timestamps","PLETH"]]),feats, df_sqis)
#
#                 start = df_segment['timestamps'].iloc[0]
#                 end = df_segment['timestamps'].iloc[-1]
#                 event_time_start.append(start)
#                 event_time_end.append(end)
#                 df_beat.append(beat)
#
#                 beat_spo2 = df_segment["SPO2"]
#                 df_beat_spo2.append(np.min(beat_spo2))
#
#                 # critical_points = signal.find_peaks(beat)[0]
#                 # if len(critical_points)>1:
#                 #     if (beat[critical_points[0]] < beat[critical_points[1]]):
#                 #         start = df_segment['timestamps'].iloc[0]
#                 #         end = df_segment['timestamps'].iloc[-1]
#                 #         event_time_start.append(start)
#                 #         event_time_end.append(end)
#
#                 plt.plot(beat)
#
#                 detector = PeakDetector(wave_type='ppg')
#                 peak_list, trough_list = detector.ppg_detector(np.array(df_segment["PLETH"]))
#                 if len(peak_list) < 2:
#                     warnings.warn("Peak Detector cannot find more than 2 peaks to process")
#                     peak_list = []
#                 rr_list = np.diff(peak_list) * (1000 / 100)  # 1000 milisecond
#                 rr_list = list(map(int,rr_list))
#                 feats['rr'].append(rr_list)
#                 feats['start_time'].append(df_segment["timestamps"].iloc[0])
#                 feats['end_time'].append(df_segment["timestamps"].iloc[-1])
#                 total_duration = 0
#         except Exception as err:
#             print(err)
#
#
#     # apply covariance outlier detection
#     # ee = EllipticEnvelope(contamination=0.01)
#     # yhat = ee.fit_predict(df_beat)
#     lof = LocalOutlierFactor()
#     oneclass_svm = OneClassSVM()
#     gmm = GaussianMixture(n_components=2)
#     knn = KMeans(n_clusters=2)
#     isolation_forest = IsolationForest()
#     try:
#         yhat_lof = lof.fit_predict(np.array(df_beat))
#         yhat_svm = oneclass_svm.fit_predict(np.array(df_beat))
#         yhat_iso = isolation_forest.fit_predict(np.array(df_beat))
#         yhat_gmm = gmm.fit_predict(np.array(df_beat))
#         yhat_knn = knn.fit_predict(np.array(df_beat))
#
#     except Exception as err:
#         print(err)
#         yhat_lof = np.ones(len(df_beat))
#         yhat_svm = np.ones(len(df_beat))
#         yhat_iso = np.ones(len(df_beat))
#         yhat_gmm = np.ones(len(df_beat))
#         yhat_knn = np.ones(len(df_beat))
#
#     mask_lof = yhat_lof == -1
#     mask_svm = yhat_svm == -1
#     mask_iso = yhat_iso == -1
#
#     values_gmm,count_gmm = np.unique(yhat_gmm,return_counts=True)
#     if len(values_gmm) < 2:
#         mask_gmm = yhat_gmm == np.inf
#     else:
#         outlier_gmm_value = values_gmm[np.argmin(count_gmm)]
#         mask_gmm = yhat_gmm == outlier_gmm_value
#
#     values_knn, count_knn = np.unique(yhat_knn, return_counts=True)
#     if len(values_gmm) < 2:
#         mask_knn = yhat_knn == np.inf
#     else:
#         outlier_knn_value = values_knn[np.argmin(count_knn)]
#         mask_knn = yhat_knn == outlier_knn_value
#
#     mask = mask_lof & mask_iso & (mask_gmm | mask_knn | mask_svm)
#     # mask = (mask_gmm & mask_knn)
#
#     log_idx = np.arange(len(df_beat))[mask]
#
#     sus_spo2 = np.array(df_beat_spo2)[log_idx]
#     mask_sus = sus_spo2 < 96
#     log_idx = log_idx[mask_sus]
#     # ng_spo2_log_idx =
#
#     img_location = os.path.join(img_PATH, ".".join(fname.split("."))[:-2] + ".png")
#     if not os.path.exists(os.path.join(img_PATH, "logs")):
#         os.makedirs(os.path.join(img_PATH, "logs"))
#     img_log_file = os.path.join(img_PATH,"logs", ".".join(fname.split("."))[:-2] + ".csv")
#     df_log = pd.DataFrame()
#     df_log['start_time'] = np.array(event_time_start)[log_idx]
#     df_log['end_time'] = np.array(event_time_end)[log_idx]
#     df_log['spo2'] = np.array(df_beat_spo2)[log_idx]
#     plt.savefig(img_location)
#     df_log.to_csv(img_log_file)
#
#     outlier_img_location = os.path.join(img_PATH, "logs",".".join(fname.split("."))[:-2] + ".png")
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     for idx in log_idx:
#         outlier_beat = df_beat[idx]
#         label = str(np.array(event_time_start)[idx]) +" - "+ str(np.array(df_beat_spo2)[idx])
#         ax.plot(outlier_beat,label = label)
#     lgd = ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.5),ncol=2)
#     fig.savefig(outlier_img_location,bbox_extra_artist=(lgd,),bbox_inches='tight')
#     pd.DataFrame(feats).to_csv(os.path.join(feat_PATH, ".".join(fname.split("."))[:-2] + "rr.csv"))
#
#     # feat_dict.update(df_sqis.to_dict('list'))
#     # pd.DataFrame(feat_dict).to_csv(os.path.join(feat_PATH,".".join(fname.split("."))[:-2]+"feats.csv"))
#
#     return


def combine_synthesize_data():
    processed_sig = synthesize_nonlinear_dynamic_system(duration=30,trend_frequency=200,
                                    noise_scale = 0.1, noise_mean = 0.5,
                                    noise_density = 0.3,extend_rate=0.2,
                                    resample_rate = 2.5
                                    )
    scaler = MinMaxScaler(
        feature_range=(min(processed_sig), max(processed_sig))
    )
    file_df = pd.read_csv("v2smartcare.csv", warn_bad_lines=True, error_bad_lines=False)
    df = pd.DataFrame()
    pleth_data = []
    file_df['PL' \
            'ETH'].apply(get_flat, flat=pleth_data)
    df['TIMESTAMP_MS'] = np.arange(0, len(pleth_data)) * 10
    df['PLETH'] = pleth_data
    pleth_data = scaler.fit_transform(np.array(pleth_data[:len(processed_sig)]).reshape(-1, 1)).reshape(-1)

    peak_real = signal.find_peaks_cwt(-pleth_data)

    return


synthesize_nonlinear_dynamic_system(duration=30,trend_frequency=200,
                                    noise_scale = 0.1, noise_mean = 0.5,
                                    noise_density = 0.3,extend_rate=0.2,
                                    resample_rate = 2.5
                                    )


