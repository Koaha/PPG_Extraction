import numpy as np
from scipy import signal
from RRest.estimate_rr import CtO,AR_RR,FTS

def get_fusion(sig,fs):
    #one method from BW -> method using the envelope of signal : AR methods
    #one method using AM -> the magnitude from the peak to the prior trough: Fourier method
    #one method using FM -> the change in the interval between the subsequent peak: CtO method
    bw_est = CtO.get_rr(sig,fs)
    am_est = AR_RR.get_rr(sig,fs)
    fm_est = FTS.get_rr(sig,fs)

    # find the range where all 3 algo detect the breathing signal


    mutual_times = intersect(intersect(data.bw.t, data.fm.t), data.am.t);

    for mod = {'bw', 'am', 'fm'}
    eval(['temp_data = data.' mod{1, 1} ';']);
    [rel_times, rel_els, ~] = intersect(temp_data.t, mutual_times);

    eval([mod{1, 1} '_est = data.' mod{1, 1} '.v(rel_els);']);
