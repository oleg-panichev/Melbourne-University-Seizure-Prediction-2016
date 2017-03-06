import os
import time
import numpy as np
import pandas as pd
from math import *
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from decimal import Decimal
import threading

def list_files(path):
    files = []
    for fname in os.listdir(path):
        if os.path.isfile(os.path.join(path, fname)):
            files.append(fname)
    return files


def mat_to_data(path):
    mat = loadmat(path, verify_compressed_data_integrity=False)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata


def corr(data, type_corr):
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    w, v = np.linalg.eig(C)
    x = np.sort(w)
    x = np.real(x)
    return x


def fcalc_corrc(data, type_corr):
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    n = len(C)
    N = (n * n - n) / 2
    X = np.empty(N)
    idx = 0
    for i in range(0, n, 1):
        for j in range(i + 1, n, 1):
            X[idx] = C[i, j]
            idx += 1
    return X


def calculate_starter_v0_features(f, epoch_len):
    fs = f['iEEGsamplingRate'][0, 0]
    eegData = f['data']
    [nt, nc] = eegData.shape
    # print((nt, nc))
    subsampLen = int(floor(fs * epoch_len))  # epoch length
    numSamps = int(floor(nt / subsampLen))  # Num of 1-min samples
    sampIdx = range(0, (numSamps + 1) * subsampLen, subsampLen)

    for i in range(1, numSamps + 1):
        # if i % 5 == 0:
        #    print('processing file {} epoch {}'.format(file_name, i))
        epoch = eegData[sampIdx[i - 1]:sampIdx[i], :]
        epoch_features = []

        # compute Shannon's entropy, spectral edge and correlation matrix
        # segments corresponding to frequency bands
        lvl = np.array([0.2, 4, 8, 12, 30, 70, 180])  # Frequency levels in Hz
        lseg = np.round(nt / fs * lvl).astype('int')
        D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))
        D[0, :] = 0  # set the DC component to zero
        D /= D.sum()  # Normalize each channel

        dspect = np.zeros((len(lvl) - 1, nc))
        for j in range(len(dspect)):
            dspect[j, :] = 2 * np.sum(D[lseg[j]:lseg[j + 1], :], axis=0)

        # Find the shannon's entropy
        spentropy = -1 * np.sum(np.multiply(dspect, np.log(dspect)), axis=0)

        # Find the spectral edge frequency
        sfreq = fs
        tfreq = 40
        ppow = 0.5

        topfreq = int(round(nt / sfreq * tfreq)) + 1
        A = np.cumsum(D[:topfreq, :])
        B = A - (A.max() * ppow)
        spedge = np.min(np.abs(B))
        spedge = (spedge - 1) / (topfreq - 1) * tfreq

        # Calculate correlation matrix and its eigenvalues (b/w channels)
        data = pd.DataFrame(data=epoch)
        type_corr = 'pearson'
        lxchannels = corr(data, type_corr)

        # Calculate correlation matrix and its eigenvalues (b/w freq)
        data = pd.DataFrame(data=dspect)
        lxfreqbands = corr(data, type_corr)

        # Spectral entropy for dyadic bands
        # Find number of dyadic levels
        ldat = int(floor(nt / 2.0))
        no_levels = int(floor(log(ldat, 2.0)))
        seg = floor(ldat / pow(2.0, no_levels - 1))

        # Find the power spectrum at each dyadic level
        dspect = np.zeros((no_levels, nc))
        for j in range(no_levels - 1, -1, -1):
            dspect[j, :] = 2 * np.sum(D[int(floor(ldat / 2.0)) + 1:ldat, :], axis=0)
            ldat = int(floor(ldat / 2.0))

        # Find the Shannon's entropy
        spentropyDyd = -1 * np.sum(np.multiply(dspect, np.log(dspect)), axis=0)

        # Find correlation between channels
        data = pd.DataFrame(data=dspect)
        lxchannelsDyd = corr(data, type_corr)

        # Fractal dimensions
        no_channels = nc
        # fd = np.zeros((2,no_channels))
        # for j in range(no_channels):
        #    fd[0,j] = pyeeg.pfd(epoch[:,j])
        #    fd[1,j] = pyeeg.hfd(epoch[:,j],3)
        #    fd[2,j] = pyeeg.hurst(epoch[:,j])

        # [mobility[j], complexity[j]] = pyeeg.hjorth(epoch[:,j)
        # Hjorth parameters
        # Activity
        activity = np.var(epoch, axis=0)
        # print('Activity shape: {}'.format(activity.shape))
        # Mobility
        mobility = np.divide(
            np.std(np.diff(epoch, axis=0)),
            np.std(epoch, axis=0))
        # print('Mobility shape: {}'.format(mobility.shape))
        # Complexity
        complexity = np.divide(np.divide(
            # std of second derivative for each channel
            np.std(np.diff(np.diff(epoch, axis=0), axis=0), axis=0),
            # std of second derivative for each channel
            np.std(np.diff(epoch, axis=0), axis=0))
            , mobility)
        # print('Complexity shape: {}'.format(complexity.shape))
        # Statistical properties

        # Skewness
        sk = skew(epoch)
        # print sk.shape
        # print sk

        # Kurtosis
        kurt = kurtosis(epoch)

        corrc = fcalc_corrc(data, type_corr)

        # compile all the features
        epoch_features = np.concatenate((np.array([i]),
                                         spentropy.ravel(),
                                         spedge.ravel(),
                                         lxchannels.ravel(),
                                         lxfreqbands.ravel(),
                                         spentropyDyd.ravel(),
                                         lxchannelsDyd.ravel(),
                                         # fd.ravel(),
                                         activity.ravel(),
                                         mobility.ravel(),
                                         complexity.ravel(),
                                         sk.ravel(),
                                         kurt.ravel(),
                                         corrc.ravel()
                                         ))

        if np.isnan(epoch_features[1]):
            for nf in range(1, len(epoch_features)):
                epoch_features[nf] = 0

        if not 'features' in locals():
            features = np.empty((numSamps, len(epoch_features)))

        features[i - 1] = epoch_features.view()

    return features

def calculate_starter_features(f, epoch_len):
    # f = mat_to_data(file_name)
    fs = f['iEEGsamplingRate'][0, 0]
    eegData = f['data']
    [nt, nc] = eegData.shape
    # print((nt, nc))
    subsampLen = int(floor(fs * epoch_len))  # epoch length
    numSamps = int(floor(nt / subsampLen))  # Num of 1-min samples
    sampIdx = range(0, (numSamps + 1) * subsampLen, subsampLen)

    for i in range(1, numSamps + 1):
        # if i % 5 == 0:
        #    print('processing file {} epoch {}'.format(file_name, i))
        epoch = eegData[sampIdx[i - 1]:sampIdx[i], :]
        epoch_features = []

        # compute Shannon's entropy, spectral edge and correlation matrix
        # segments corresponding to frequency bands
        lvl = np.array([0.1, 4, 8, 12, 30, 70, 180])  # Frequency levels in Hz
        lseg = np.round(nt / fs * lvl).astype('int')
        D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))
        D[0, :] = 0  # set the DC component to zero
        D /= D.sum()  # Normalize each channel

        dspect = np.zeros((len(lvl) - 1, nc))
        for j in range(len(dspect)):
            dspect[j, :] = 2 * np.sum(D[lseg[j]:lseg[j + 1], :], axis=0)

        # Find the shannon's entropy
        spentropy = -1 * np.sum(np.multiply(dspect, np.log(dspect)), axis=0)

        # Find the spectral edge frequency
        sfreq = fs
        tfreq = 40
        ppow = 0.5

        topfreq = int(round(nt / sfreq * tfreq)) + 1
        A = np.cumsum(D[:topfreq, :])
        B = A - (A.max() * ppow)
        spedge = np.min(np.abs(B))
        spedge = (spedge - 1) / (topfreq - 1) * tfreq

        # Calculate correlation matrix and its eigenvalues (b/w channels)
        data = pd.DataFrame(data=epoch)
        type_corr = 'pearson'
        lxchannels = corr(data, type_corr)

        # Calculate correlation matrix and its eigenvalues (b/w freq)
        data = pd.DataFrame(data=dspect)
        lxfreqbands = corr(data, type_corr)

        # Spectral entropy for dyadic bands
        # Find number of dyadic levels
        ldat = int(floor(nt / 2.0))
        no_levels = int(floor(log(ldat, 2.0)))
        seg = floor(ldat / pow(2.0, no_levels - 1)) # * 999999999999999999999999999999999999999999999999

        # Find the power spectrum at each dyadic level
        dspect = np.zeros((no_levels, nc))
        for j in range(no_levels - 1, -1, -1):
            dspect[j, :] = 2 * np.sum(D[int(floor(ldat / 2.0)) + 1:ldat, :], axis=0)
            ldat = int(floor(ldat / 2.0))

        # Find the Shannon's entropy
        spentropyDyd = -1 * np.sum(np.multiply(dspect, np.log(dspect)), axis=0)

        # Find correlation between channels
        data = pd.DataFrame(data=dspect)
        lxchannelsDyd = corr(data, type_corr)

        # Fractal dimensions
        no_channels = nc
        # fd = np.zeros((2,no_channels))
        # for j in range(no_channels):
        #    fd[0,j] = pyeeg.pfd(epoch[:,j])
        #    fd[1,j] = pyeeg.hfd(epoch[:,j],3)
        #    fd[2,j] = pyeeg.hurst(epoch[:,j])

        # [mobility[j], complexity[j]] = pyeeg.hjorth(epoch[:,j)
        # Hjorth parameters
        # Activity
        activity = np.var(epoch, axis=0)
        # print('Activity shape: {}'.format(activity.shape))
        # Mobility
        mobility = np.divide(
            np.std(np.diff(epoch, axis=0)),
            np.std(epoch, axis=0))
        # print('Mobility shape: {}'.format(mobility.shape))

        # Complexity
        complexity = np.divide(np.divide(
            # std of second derivative for each channel
            np.std(np.diff(np.diff(epoch, axis=0), axis=0), axis=0),
            # std of second derivative for each channel
            np.std(np.diff(epoch, axis=0), axis=0))
            , mobility)
        # print('Complexity shape: {}'.format(complexity.shape))

        # Statistical properties
        # Skewness
        sk = skew(epoch)

        # Kurtosis
        kurt = kurtosis(epoch)

        # compile all the features
        epoch_features = np.concatenate((np.array([i]),
                                         spentropy.ravel(),
                                         spedge.ravel(),
                                         lxchannels.ravel(),
                                         lxfreqbands.ravel(),
                                         spentropyDyd.ravel(),
                                         lxchannelsDyd.ravel(),
                                         # fd.ravel(),
                                         activity.ravel(),
                                         mobility.ravel(),
                                         complexity.ravel(),
                                         sk.ravel(),
                                         kurt.ravel()
                                         ))

        if not 'features' in locals():
            features = np.empty((numSamps, len(epoch_features)))

        features[i - 1] = epoch_features.view()

    return features


def calculate_spectral_features(f, epoch_len):
    fs = f['iEEGsamplingRate'][0, 0]
    eegData = f['data']
    [nt, nc] = eegData.shape
    subsampLen = int(floor(fs * epoch_len))  # epoch length
    numSamps = int(floor(nt / subsampLen))  # Num of 1-min samples
    sampIdx = range(0, (numSamps + 1) * subsampLen, subsampLen)

    # Allocate spaces
    p_total = np.empty(nc)
    p_d = np.empty(nc)
    p_t = np.empty(nc)
    p_a = np.empty(nc)
    p_b = np.empty(nc)
    # p_gl = np.empty(nc)
    # p_gh = np.empty(nc)

    # p_dm = np.empty(nc)
    # p_tm = np.empty(nc)
    # p_am = np.empty(nc)
    # p_bm = np.empty(nc)
    # p_glm = np.empty(nc)
    # p_ghm = np.empty(nc)

    p_dt = np.empty(nc)
    p_tt = np.empty(nc)
    p_at = np.empty(nc)
    p_bt = np.empty(nc)
    # p_glt = np.empty(nc)
    # p_ght = np.empty(nc)

    p_dt = np.empty(nc)
    p_da = np.empty(nc)
    p_db = np.empty(nc)
    # p_dgl = np.empty(nc)
    # p_dgh = np.empty(nc)

    p_td = np.empty(nc)
    p_ta = np.empty(nc)
    p_tb = np.empty(nc)
    # p_tgl = np.empty(nc)
    # p_tgh = np.empty(nc)

    p_ad = np.empty(nc)
    p_at = np.empty(nc)
    p_ab = np.empty(nc)
    # p_agl = np.empty(nc)
    # p_agh = np.empty(nc)

    p_bd = np.empty(nc)
    p_bt = np.empty(nc)
    p_ba = np.empty(nc)
    # p_bgl = np.empty(nc)
    # p_bgh = np.empty(nc)

    # p_gld = np.empty(nc)
    # p_glt = np.empty(nc)
    # p_gla = np.empty(nc)
    # p_glb = np.empty(nc)
    # p_glgh = np.empty(nc)

    # p_ghd = np.empty(nc)
    # p_ght = np.empty(nc)
    # p_gha = np.empty(nc)
    # p_ghb = np.empty(nc)
    # p_ghgl = np.empty(nc)

    # sep10 = np.empty(nc)
    # sep20 = np.empty(nc)
    # sep30 = np.empty(nc)
    # sep40 = np.empty(nc)
    # sep50 = np.empty(nc)
    # sep50 = np.empty(nc)
    # sep60 = np.empty(nc)
    # sep70 = np.empty(nc)
    # sep80 = np.empty(nc)
    # sep90 = np.empty(nc)


    for i in range(1, numSamps + 1):
        # if i % 5 == 0:
        #    print('processing file {} epoch {}'.format(file_name, i))
        epoch = eegData[sampIdx[i - 1]:sampIdx[i], :]

        epoch_features = []

        for j in range (0, nc):
            S = abs(np.fft.fft(epoch[:, j] - np.mean(epoch[:, j])))/epoch.shape[0]

            p_total[j] = np.sum(S[round(epoch_len*0.0):epoch_len*30])
            p_d[j] = np.sum(S[round(epoch_len*0.0):epoch_len*3])
            p_t[j] = np.sum(S[epoch_len*3:epoch_len*8])
            p_a[j] = np.sum(S[epoch_len*8:epoch_len*14])
            p_b[j] = np.sum(S[epoch_len*14:epoch_len*30])

            p_dt[j] = p_d[j] / p_total[j]
            p_tt[j] = p_t[j] / p_total[j]
            p_at[j] = p_a[j] / p_total[j]
            p_bt[j] = p_b[j] / p_total[j]

            p_dt[j] = p_d[j] / p_t[j]
            p_da[j] = p_d[j] / p_a[j]
            p_db[j] = p_d[j] / p_b[j]

            p_td[j] = p_t[j] / p_d[j]
            p_ta[j] = p_t[j] / p_a[j]
            p_tb[j] = p_t[j] / p_b[j]

            p_ad[j] = p_a[j] / p_d[j]
            p_at[j] = p_a[j] / p_t[j]
            p_ab[j] = p_a[j] / p_b[j]

            p_bd[j] = p_b[j] / p_d[j]
            p_bt[j] = p_b[j] / p_t[j]
            p_ba[j] = p_b[j] / p_a[j]

        # compile all the features
        epoch_features = np.concatenate((np.array([i]),
                                         p_total.ravel(),
                                         p_d.ravel(),
                                         p_t.ravel(),
                                         p_a.ravel(),
                                         p_b.ravel(),

                                         p_dt.ravel(),
                                         p_tt.ravel(),
                                         p_at.ravel(),
                                         p_bt.ravel(),

                                         p_dt.ravel(),
                                         p_da.ravel(),
                                         p_db.ravel(),

                                         p_td.ravel(),
                                         p_ta.ravel(),
                                         p_tb.ravel(),

                                         p_ad.ravel(),
                                         p_at.ravel(),
                                         p_ab.ravel(),

                                         p_bd.ravel(),
                                         p_bt.ravel(),
                                         p_ba.ravel()
                                         ))

        if not 'features' in locals():
            features = np.empty((numSamps, len(epoch_features)))

        features[i - 1] = epoch_features.view()

    return features


def round_to_zero(d):
    # Rounding small values to zero
    fsz = d.shape
    for i in range(0, fsz[0], 1):
        for j in range(0, fsz[1], 2):
            if abs(d[i, j]) < 0.00001:
                d[i, j] = 0

    return d


def extract(input_path, output_path, output_label, safe_files_list):
    # Properties
    calc_starter_v0_flag = 1
    calc_starter_flag = 1
    calc_spectral_flag = 1

    float_fmt = ',%.8f'

    epoch_len = 30 # Seconds

    if len(safe_files_list) > 0:
        data = np.array(safe_files_list)
        files_list = data[:, 0]
        files_class = data[:, 1]
        files_safe = data[:, 2]

        files_list_safe = []
        files_class_safe = []
        for i in range(0, len(files_safe)):
            if files_safe[i] == 1:
                files_list_safe.append(files_list[i])
                files_class_safe.append(files_class[i])

        files_list = files_list_safe
        files_class = files_class_safe
    else:
        files_list = list_files(input_path)

    if calc_starter_v0_flag > 0:
        starter_v0_features_file = open(output_path + output_label + '_starter_old.csv', "w")
        starter_v0_str_buf = ''

    if calc_starter_flag > 0:
        starter_features_file = open(output_path + output_label + '_starter.csv', "w")
        starter_str_buf = ''

    if calc_spectral_flag > 0:
        spectral_features_file = open(output_path + output_label + '_spectral_v0.csv', "w")
        spectral_str_buf = ''

    cnt = 0
    # print files_list
    for idx, fname in enumerate(files_list):
        # print(input_path[0:-1] + ': ' + fname)
        f = mat_to_data(input_path + fname)

        # Starter V0 features ---------------------------------------------------------------------------------------------
        if calc_starter_v0_flag > 0:
            features = calculate_starter_v0_features(f, epoch_len)

            # Rounding small values to zero
            fsz = features.shape
            features = round_to_zero(features)


            if len(safe_files_list) > 0:
                fclass_str = str(files_class[idx])
            else:
                fclass_str = []

            for i in range(0, fsz[0], 1):
                starter_v0_str_buf += fname
                # print str_buf
                for j in range(0, fsz[1], 1):
                    if j == 0:
                        starter_v0_str_buf += ',%.0f' % features[i, j]
                    else:
                        starter_v0_str_buf += float_fmt % features[i, j]
                if len(fclass_str) > 0:
                    starter_v0_str_buf += ',' + fclass_str

                starter_v0_str_buf += '\n'

            if idx % 10 == 0 or idx == (len(files_list) - 1):
                starter_v0_features_file.write("%s" % starter_v0_str_buf)
                # print len(starter_str_buf)
                if idx % 10 == 0:
                    print('starter_v0: writing data to file (' + input_path + '): ' + str(idx+1) + '/' + str(len(files_list)))

                starter_v0_str_buf = ''

        # Starter features ---------------------------------------------------------------------------------------------
        if calc_starter_flag > 0:
            features = calculate_starter_features(f, epoch_len)

            # Rounding small values to zero
            fsz = features.shape
            features = round_to_zero(features)

            if len(safe_files_list) > 0:
                fclass_str = str(files_class[idx])
            else:
                fclass_str = []

            for i in range(0, fsz[0], 1):
                starter_str_buf += fname
                # print str_buf
                for j in range(0, fsz[1], 1):
                    if j == 0:
                        starter_str_buf += ",%.0f" % features[i, j]
                    else:
                        starter_str_buf += float_fmt % features[i, j]
                if len(fclass_str) > 0:
                    starter_str_buf += ',' + fclass_str

                starter_str_buf += '\n'

            if idx % 10 == 0 or idx == (len(files_list) - 1):
                starter_features_file.write("%s" % starter_str_buf)
                # print len(starter_str_buf)
                if idx % 10 == 0:
                    print('starter: writing data to file (' + input_path + '): ' + str(idx+1) + '/' + str(len(files_list)))
                starter_str_buf = ''

        # Spectral features --------------------------------------------------------------------------------------------
        if calc_spectral_flag > 0:
            features = calculate_spectral_features(f, epoch_len)
            fsz = features.shape

            features = round_to_zero(features)

            if len(safe_files_list) > 0:
                fclass_str = str(files_class[idx])
            else:
                fclass_str = []

            for i in range(0, fsz[0], 1):
                spectral_str_buf += fname
                # print str_buf
                for j in range(0, fsz[1], 1):
                    if j == 0:
                        spectral_str_buf += ",%.0f" % features[i, j]
                    else:
                        spectral_str_buf += float_fmt % features[i, j]
                if len(fclass_str) > 0:
                    spectral_str_buf += ',' + fclass_str
                spectral_str_buf += '\n'

            # print len(str_buf)
            # print str_buf
            if idx % 10 == 0 or idx == (len(files_list) - 1):
                spectral_features_file.write("%s" % spectral_str_buf)
                # print len(starter_str_buf)
                if idx % 10 == 0 or idx == (len(files_list) - 1):
                    print('spectral: writing data to file (' + input_path + '): ' + str(idx+1) + '/' + str(len(files_list)))
                spectral_str_buf = ''

        

    if calc_starter_v0_flag > 0:
        starter_v0_features_file.close()

    if calc_starter_flag > 0:
        starter_features_file.close()

    if calc_spectral_flag > 0:
        spectral_features_file.close()

## ---------------------------------------------------------------------------------------------------------------------
## Config
# db_path = 'E:/DB/NIH_sp/'
db_path = '../input/'
# files_path = ['test_1/']
# files_path = ['train_3/']
# files_path = ['train_1/', 'train_2/']
# files_path = ['test_1/', 'test_2/', 'test_3/']
# files_path = ['test_1_new/', 'test_2_new/', 'test_3_new/']
# files_path = ['train_1/', 'train_2/', 'train_3/', 'test_1/', 'test_2/', 'test_3/', \
#               'test_1_new/', 'test_2_new/', 'test_3_new/']
files_path = ['train_1/', 'train_2/', 'train_3/', 'test_1/', 'test_2/', 'test_3/', 'test_1_new/', 'test_2_new/', 'test_3_new/']
output_path = ''
use_thread_flag = 0 # if == 0 - threads aren't used

threads = []

now = time.strftime("%c")
print ('Started at ' + now)

if use_thread_flag == 0:
    for fpath in files_path:
        fname = db_path + 'safe_' + fpath[:-1] + '.csv'
        if os.path.isfile(fname):
            safe_files_list = pd.read_csv(fname)
        else:
            safe_files_list = []

        output_label = fpath[0:-1]

        extract(db_path + fpath, output_path, output_label, safe_files_list)
else:
    threads = []
    for fpath in files_path:
        fname = db_path + 'safe_' + fpath[:-1] + '.csv'
        if os.path.isfile(fname):
            safe_files_list = pd.read_csv(fname)
        else:
            safe_files_list = []

        output_label = fpath[0:-1]

        t = threading.Thread(target=extract, args=(db_path + fpath, output_path, output_label, safe_files_list))
        threads.append(t)
        t.start()

    for t in threads:
        t.join():
    t.join()

now = time.strftime("%c")
print ('Done at ' + now)
