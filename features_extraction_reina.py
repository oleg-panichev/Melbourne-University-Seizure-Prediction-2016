import os
from timeit import default_timer as timer
import time
import numpy as np
import pandas as pd
from math import *
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from decimal import Decimal
import threading

# Supporting functions -------------------------------------------------------------------------------------------------
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

# Features -------------------------------------------------------------------------------------------------------------
'''
Calculates the FFT of the epoch signal. Removes the DC component and normalizes the area to 1
'''
def calcNormalizedFFT(epoch, lvl, nt, fs):
    lseg = np.round(nt / fs * lvl).astype('int')
    D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))
    D[0, :] = 0  # set the DC component to zero
    D /= D.sum()  # Normalize each channel

    return D


def defineEEGFreqs():
    '''
    EEG waveforms are divided into frequency groups. These groups seem to be related to mental activity.
    alpha waves = 8-13 Hz = Awake with eyes closed
    beta waves = 14-30 Hz = Awake and thinking, interacting, doing calculations, etc.
    gamma waves = 30-45 Hz = Might be related to conciousness and/or perception (particular 40 Hz)
    theta waves = 4-7 Hz = Light sleep
    delta waves < 3.5 Hz = Deep sleep

    There are other EEG features like sleep spindles and K-complexes, but I think for this analysis
    we are just looking to characterize the waveform based on these basic intervals.
    '''
    return (np.array([0.1, 4, 8, 14, 30, 45, 70, 180]))  # Frequency levels in Hz


def calcDSpect(epoch, lvl, nt, nc, fs):
    D = calcNormalizedFFT(epoch, lvl, nt, fs)
    lseg = np.round(nt / fs * lvl).astype('int')

    dspect = np.zeros((len(lvl) - 1, nc))
    for j in range(len(dspect)):
        dspect[j, :] = 2 * np.sum(D[lseg[j]:lseg[j + 1], :], axis=0)

    return dspect


'''
Computes Shannon Entropy
'''
def calcShannonEntropy(epoch, lvl, nt, nc, fs):
    # compute Shannon's entropy, spectral edge and correlation matrix
    # segments corresponding to frequency bands
    dspect = calcDSpect(epoch, lvl, nt, nc, fs)

    # Find the shannon's entropy
    spentropy = -1 * np.sum(np.multiply(dspect, np.log(dspect)), axis=0)

    return spentropy


'''
Compute spectral edge frequency
'''
def calcSpectralEdgeFreq(epoch, lvl, nt, nc, fs):
    # Find the spectral edge frequency
    sfreq = fs
    tfreq = 40
    ppow = 0.5

    topfreq = int(round(nt / sfreq * tfreq)) + 1
    D = calcNormalizedFFT(epoch, lvl, nt, fs)
    A = np.cumsum(D[:topfreq, :], axis=0)
    B = A - (A.max() * ppow)
    spedge = np.min(np.abs(B), axis=0)
    spedge = (spedge - 1) / (topfreq - 1) * tfreq

    return spedge


'''
Calculate cross-correlation matrix
'''
def corr(data, type_corr):
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0  # Replace any NaN with 0
    C[np.isinf(C)] = 0  # Replace any Infinite values with 0
    w, v = np.linalg.eig(C)
    # print(w)
    x = np.sort(w)
    x = np.real(x)
    return x


'''
Compute correlation matrix across channels
'''
def calcCorrelationMatrixChan(epoch):
    # Calculate correlation matrix and its eigenvalues (b/w channels)
    data = pd.DataFrame(data=epoch)
    type_corr = 'pearson'

    lxchannels = corr(data, type_corr)

    return lxchannels


'''
Calculate correlation matrix across frequencies
'''
def calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs):
    # Calculate correlation matrix and its eigenvalues (b/w freq)
    dspect = calcDSpect(epoch, lvl, nt, nc, fs)
    data = pd.DataFrame(data=dspect)

    type_corr = 'pearson'

    lxfreqbands = corr(data, type_corr)

    return lxfreqbands


def calcActivity(epoch):
    '''
    Calculate Hjorth activity over epoch
    '''

    # Activity
    activity = np.var(epoch, axis=0)

    return activity


def calcMobility(epoch):
    '''
    Calculate the Hjorth mobility parameter over epoch
    '''

    # Mobility
    # N.B. the sqrt of the variance is the standard deviation. So let's just get std(dy/dt) / std(y)
    mobility = np.divide(
        np.std(np.diff(epoch, axis=0)),
        np.std(epoch, axis=0))

    return mobility


def calcComplexity(epoch):
    '''
    Calculate Hjorth complexity over epoch
    '''

    # Complexity
    complexity = np.divide(
        calcMobility(np.diff(epoch, axis=0)),
        calcMobility(epoch))

    return complexity


def hjorthFD(X, Kmax=3):
    """ Compute Hjorth Fractal Dimension of a time series X, kmax
     is an HFD parameter. Kmax is basically the scale size or time offset.
     So you are going to create Kmax versions of your time series.
     The K-th series is every K-th time of the original series.
     This code was taken from pyEEG, 0.02 r1: http://pyeeg.sourceforge.net/
    """
    L = []
    x = []
    N = len(X)
    for k in range(1, Kmax):
        Lk = []

        for m in range(k):
            Lmk = 0
            for i in range(1, int(floor((N - m) / k))):
                Lmk += np.abs(X[m + i * k] - X[m + i * k - k])

            Lmk = Lmk * (N - 1) / floor((N - m) / k) / k
            Lk.append(Lmk)

        L.append(np.log(np.mean(Lk)))  # Using the mean value in this window to compare similarity to other windows
        x.append([np.log(float(1) / k), 1])

    (p, r1, r2, s) = np.linalg.lstsq(x, L)  # Numpy least squares solution

    return p[0]


def petrosianFD(X, D=None):
    """Compute Petrosian Fractal Dimension of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed by first_order_diff(X) function of pyeeg

    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.

    This code was taken from pyEEG, 0.02 r1: http://pyeeg.sourceforge.net/
    """

    # If D has been previously calculated, then it can be passed in here
    #  otherwise, calculate it.
    if D is None:  ## Xin Liu
        D = np.diff(X)  # Difference between one data point and the next

    # The old code is a little easier to follow
    N_delta = 0;  # number of sign changes in derivative of the signal
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            N_delta += 1

    n = len(X)

    # This code is a little more compact. It gives the same
    # result, but I found that it was actually SLOWER than the for loop
    # N_delta = sum(np.diff(D > 0))

    return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta))


def katzFD(epoch):
    '''
    Katz fractal dimension
    '''

    L = np.abs(epoch - epoch[0]).max()
    d = len(epoch)

    return (np.log(L) / np.log(d))



def hurstFD(epoch):

    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(epoch[lag:], epoch[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


def logarithmic_n(min_n, max_n, factor):
    """
    Creates a list of values by successively multiplying a minimum value min_n by
    a factor > 1 until a maximum value max_n is reached.

    Non-integer results are rounded down.

    Args:
    min_n (float): minimum value (must be < max_n)
    max_n (float): maximum value (must be > min_n)
    factor (float): factor used to increase min_n (must be > 1)

    Returns:
    list of integers: min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
                      without duplicates
    """
    assert max_n > min_n
    assert factor > 1

    # stop condition: min * f^x = max
    # => f^x = max/min
    # => x = log(max/min) / log(f)

    max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
    ns = [min_n]

    for i in range(max_i + 1):
        n = int(np.floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)

    return ns


def dfa(data, nvals=None, overlap=True, order=1, debug_plot=False, plot_file=None):
    total_N = len(data)
    if nvals is None:
        nvals = logarithmic_n(4, 0.1 * total_N, 1.2)

    # create the signal profile (cumulative sum of deviations from the mean => "walk")
    walk = np.cumsum(data - np.mean(data))
    fluctuations = []

    for n in nvals:
        # subdivide data into chunks of size n
        if overlap:
            # step size n/2 instead of n
            d = np.array([walk[i:i + n] for i in range(0, len(walk) - n, n // 2)])
        else:
            # non-overlapping windows => we can simply do a reshape
            d = walk[:total_N - (total_N % n)]
            d = d.reshape((total_N // n, n))

        # calculate local trends as polynomes
        x = np.arange(n)
        tpoly = np.array([np.polyfit(x, d[i], order) for i in range(len(d))])
        trend = np.array([np.polyval(tpoly[i], x) for i in range(len(d))])

        # calculate standard deviation ("fluctuation") of walks in d around trend
        flucs = np.sqrt(np.sum((d - trend) ** 2, axis=1) / n)

        # calculate mean fluctuation over all subsequences
        f_n = np.sum(flucs) / len(flucs)
        fluctuations.append(f_n)

    fluctuations = np.array(fluctuations)
    # filter zeros from fluctuations
    nonzero = np.where(fluctuations != 0)
    nvals = np.array(nvals)[nonzero]
    fluctuations = fluctuations[nonzero]
    if len(fluctuations) == 0:
        # all fluctuations are zero => we cannot fit a line
        poly = [np.nan, np.nan]
    else:
        poly = np.polyfit(np.log(nvals), np.log(fluctuations), 1)
    if debug_plot:
        plot_reg(np.log(nvals), np.log(fluctuations), poly, "log(n)", "std(X,n)", fname=plot_file)

    return poly[0]


def higuchiFD(epoch, Kmax=8):
    '''
    Ported from https://www.mathworks.com/matlabcentral/fileexchange/30119-complete-higuchi-fractal-dimension-algorithm/content/hfd.m
    by Salai Selvam V
    '''

    N = len(epoch)

    Lmk = np.zeros((Kmax, Kmax))

    # TODO: I think we can use the Katz code to refactor resampling the series
    for k in range(1, Kmax + 1):

        for m in range(1, k + 1):

            Lmki = 0

            maxI = int(floor((N - m) / k))

            for i in range(1, maxI + 1):
                Lmki = Lmki + np.abs(epoch[m + i * k - 1] - epoch[m + (i - 1) * k - 1])

            normFactor = (N - 1) / (maxI * k)
            Lmk[m - 1, k - 1] = normFactor * Lmki

    Lk = np.zeros((Kmax, 1))

    # TODO: This is just a mean. Let's use np.mean instead?
    for k in range(1, Kmax + 1):
        Lk[k - 1, 0] = np.sum(Lmk[range(k), k - 1]) / k / k

    lnLk = np.log(Lk)
    lnk = np.log(np.divide(1., range(1, Kmax + 1)))

    fit = np.polyfit(lnk, lnLk, 1)  # Fit a line to the curve

    return fit[0]  # Grab the slope. It is the Higuchi FD


def calcFractalDimension(epoch):
    '''
    Calculate fractal dimension
    '''

    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []

    for j in range(no_channels):
        fd.append([petrosianFD(epoch[:, j])  # Petrosan fractal dimension
                      , hjorthFD(epoch[:, j], 3)  # Hjorth exponent
                      , hurstFD(epoch[:, j])  # Hurst fractal dimension
                      , katzFD(epoch[:, j])  # Katz fractal dimension
                      , higuchiFD(epoch[:, j])  # Higuchi fractal dimension
                   # , dfa(epoch[:,j])    # Detrended Fluctuation Analysis - This takes a long time!
                   ])

    return pd.DataFrame(fd, columns=['Petrosian FD', 'Hjorth FD', 'Hurst FD', 'Katz FD', 'Higuichi FD'])
    # return pd.DataFrame(fd, columns=['Petrosian FD', 'Hjorth FD', 'Hurst FD', 'Katz FD', 'Higuichi FD', 'DFA'])

'''
Calculate Petrosian fractal dimension
'''
def calcPetrosianFD(epoch):
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = np.empty(no_channels)

    for j in range(no_channels):
        fd[j] = petrosianFD(epoch[:, j])  # Petrosian fractal dimension

    return fd


'''
Calculate Hjorth fractal dimension
'''
def calcHjorthFD(epoch):
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = np.empty(no_channels)

    for j in range(no_channels):
        fd[j] = hjorthFD(epoch[:, j], 3)  # Hjorth exponent

    return fd


'''
Calculate Hurst fractal dimension
'''
def calcHurstFD(epoch):
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = np.empty(no_channels)

    for j in range(no_channels):
        fd[j] = hurstFD(epoch[:, j])  # Hurst fractal dimension

    return fd


def calcHiguchiFD(epoch):
    '''
    Calculate Higuchi fractal dimension
    '''

    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = np.empty(no_channels)

    for j in range(no_channels):
        fd[j] = higuchiFD(epoch[:, j])  # Higuchi fractal dimension

    return fd


def calcKatzFD(epoch):
    '''
    Calculate Katz fractal dimension
    '''

    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = np.empty(no_channels)

    for j in range(no_channels):
        fd[j] = katzFD(epoch[:, j])  # Katz fractal dimension

    return fd


def calcDFA(epoch):
    '''
    Calculate Detrended Fluctuation Analysis
    '''

    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = np.empty(no_channels)

    for j in range(no_channels):
        fd[j] = dfa(epoch[:, j])  # DFA

    return fd


def calcSkewness(epoch):
    '''
    Calculate skewness
    '''
    # Statistical properties
    # Skewness
    sk = skew(epoch)

    return sk


def calcKurtosis(epoch):
    '''
    Calculate kurtosis
    '''
    # Kurtosis
    kurt = kurtosis(epoch)

    return kurt


'''
Computes Shannon Entropy for the Dyads
'''
def calcShannonEntropyDyad(epoch, lvl, nt, nc, fs):
    dspect = calcDSpectDyad(epoch, lvl, nt, nc, fs)

    # Find the Shannon's entropy
    spentropyDyd = -1 * np.sum(np.multiply(dspect, np.log(dspect)), axis=0)

    return spentropyDyd


def calcDSpectDyad(epoch, lvl, nt, nc, fs):
    # Spectral entropy for dyadic bands
    # Find number of dyadic levels
    ldat = int(floor(nt / 2.0))
    no_levels = int(floor(log(ldat, 2.0)))
    seg = floor(ldat / pow(2.0, no_levels - 1))

    D = calcNormalizedFFT(epoch, lvl, nt, fs)

    # Find the power spectrum at each dyadic level
    dspect = np.zeros((no_levels, nc))
    for j in range(no_levels - 1, -1, -1):
        dspect[j, :] = 2 * np.sum(D[int(floor(ldat / 2.0)) + 1:ldat, :], axis=0)
        ldat = int(floor(ldat / 2.0))

    return dspect


def calcXCorrChannelsDyad(epoch, lvl, nt, nc, fs):
    dspect = calcDSpectDyad(epoch, lvl, nt, nc, fs)

    # Find correlation between channels
    data = pd.DataFrame(data=dspect)
    type_corr = 'pearson'
    lxchannelsDyd = corr(data, type_corr)

    return lxchannelsDyd


def calculate_features(f, epoch_len):
    # f = mat_to_data(file_name)

    fs = f['iEEGsamplingRate'][0, 0]
    eegData = f['data']
    [nt, nc] = eegData.shape
    # print('EEG shape = ({} timepoints, {} channels)'.format(nt, nc))

    lvl = defineEEGFreqs()

    subsampLen = int(floor(fs * epoch_len))  # Grabbing EPOCH_LEN-second epochs from within the time series
    numSamps = int(floor(nt / subsampLen))  # Num of 1-min samples
    sampIdx = range(0, (numSamps + 1) * subsampLen, subsampLen)

    functions = {
        'shannon entropy': 'calcShannonEntropy(epoch, lvl, nt, nc, fs)'
        , 'spectral edge frequency': 'calcSpectralEdgeFreq(epoch, lvl, nt, nc, fs)'
        , 'correlation matrix (channel)': 'calcCorrelationMatrixChan(epoch)'
        , 'correlation matrix (frequency)': 'calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs)'
        , 'shannon entropy (dyad)': 'calcShannonEntropyDyad(epoch, lvl, nt, nc, fs)'
        , 'crosscorrelation (dyad)': 'calcXCorrChannelsDyad(epoch, lvl, nt, nc, fs)'
        , 'hjorth activity': 'calcActivity(epoch)'
        , 'hjorth mobility': 'calcMobility(epoch)'
        , 'hjorth complexity': 'calcComplexity(epoch)'
        , 'skewness': 'calcSkewness(epoch)'
        , 'kurtosis': 'calcKurtosis(epoch)'
        , 'Petrosian FD': 'calcPetrosianFD(epoch)'
        , 'Hjorth FD': 'calcHjorthFD(epoch)'
        , 'Katz FD': 'calcKatzFD(epoch)'
        , 'Higuchi FD': 'calcHiguchiFD(epoch)'
                 # , 'Detrended Fluctuation Analysis' : 'calcDFA(epoch)'  # DFA takes a long time!
                 }

    # Initialize a dictionary of pandas dataframes with the features as keys
    # feat = {key[0]: pd.DataFrame() for key in functions.items()}

    # feat = []
    feat = np.empty([numSamps, len(functions)*16+1])

    for i in range(1, numSamps + 1):

        # print('processing file {} epoch {}'.format(file_name, i))
        epoch = eegData[sampIdx[i - 1]:sampIdx[i], :]
        feat[i-1, 0] = i

        # startt = timer()
        idx = 1
        for key in functions.items():
            f = np.asarray(eval(key[1]))
            feat[i-1, idx:idx+len(f)] = f
            idx += len(f)

    return feat

def round_to_zero(d):
    # Rounding small values to zero
    fsz = d.shape
    for i in range(0, fsz[0], 1):
        for j in range(0, fsz[1], 2):
            if abs(d[i, j]) < 0.00001:
                d[i, j] = 0

    return d


def extract(input_path, output_path, output_label, safe_files_list):
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

    reina_features_file = open(output_path + output_label + '_reina_e' + str(epoch_len) + '.csv', "w")
    reina_str_buf = ''

    for idx, fname in enumerate(files_list):
        # print(input_path[0:-1] + ': ' + fname)
        f = mat_to_data(input_path + fname)

        # Reina features ---------------------------------------------------------------------------------------------
        features = calculate_features(f, epoch_len)

        # Rounding small values to zero
        fsz = features.shape
        features = round_to_zero(features)

        # startt = timer()
        if len(safe_files_list) > 0:
            fclass_str = str(files_class[idx])
        else:
            fclass_str = []

        for i in range(0, fsz[0], 1):
            reina_str_buf += fname
            # print str_buf
            for j in range(0, fsz[1], 1):
                if j == 0:
                    reina_str_buf += ',%.0f' % features[i, j]
                else:
                    reina_str_buf += float_fmt % features[i, j]
            if len(fclass_str) > 0:
                reina_str_buf += ',' + fclass_str

            reina_str_buf += '\n'

        if idx % 10 == 0 or idx == (len(files_list) - 1):
            reina_features_file.write("%s" % reina_str_buf)
            print('reina: writing data to file (' + input_path + '): ' + str(idx+1) + '/' + str(len(files_list)))

            reina_str_buf = ''

        # endt = timer()
        # print 'Write file: ' + str(endt-startt)

    reina_features_file.close()

## ---------------------------------------------------------------------------------------------------------------------
## Config
db_path = 'E:/DB/NIH_sp/'
# files_path = ['train_3/']
# files_path = ['train_2/']
# files_path = ['train_1/', 'test_1/', 'test_2/', 'test_3/']
# files_path = ['test_1_new/', 'test_2_new/', 'test_3_new/']
files_path = ['train_1/', 'train_2/', 'train_3/', 'test_1/', 'test_2/', 'test_3/', 'test_1_new/', 'test_2_new/', 'test_3_new/']
output_path = ''
use_thread_flag = 0 # if == 0 - threads aren't used

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
        t.join()

now = time.strftime("%c")
print ('Done at ' + now)
