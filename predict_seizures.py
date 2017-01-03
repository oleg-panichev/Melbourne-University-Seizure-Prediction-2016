# coding: utf-8
import os
import time
import math
import pandas as pd
import numpy as np
from scipy import interp, float64
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import cycle
import random

from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Classifiers:
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import xgboost

# Serialize models
import pickle


def model_fit(classifier, x, y):
    classifier.fit(x, y)
    return classifier


def model_predict(classifier, x_test):
    p = classifier.predict_proba(x_test)
    return p


def model_evaluate(model_name, x, y, epoch_num):
    cv = StratifiedKFold(n_splits=6, shuffle=False)
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    print 'Cross validation'
    i = 0

    i_reduced = range(0, len(y), epoch_num)
    x_r = x[i_reduced]
    y_r = y[i_reduced]

    probabilities = np.empty(len(y_r))
    probabilities_epoch = np.empty([len(y_r), epoch_num])

    # plt.figure()
    roc_auc_max = 0
    for (train, test), color in zip(cv.split(x_r, y_r), colors):
        # Recalculating indexes to make all observations from one signal be present in train or test
        train_full = np.zeros(len(train) * epoch_num, dtype='int')
        for k in range(0, len(train), 1):
            for j in range(0, epoch_num, 1):
                train_full[k*epoch_num + j] = train[k]*epoch_num + j

        test_full = np.zeros(len(test) * epoch_num, dtype='int')
        for k in range(0, len(test), 1):
            for j in range(0, epoch_num, 1):
                test_full[k*epoch_num + j] = test[k]*epoch_num + j

        # print 'Model fitting...'
        classifier = model_create(model_name)
        classifier = model_fit(classifier, x[train_full], y[train_full])

        # print 'Predicting...'
        probas = model_predict(classifier, x[test_full])

        p, p_x = prob_decide(probas[:, 1], epoch_num)

        probabilities[test] = p
        probabilities_epoch[test, :] = p_x

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_r[test], p)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        if roc_auc > roc_auc_max:
            roc_auc_max = roc_auc
            best_classifier = classifier
            # print 'Best classifier found!'
        print 'Iteration #' + str(i) + ': AUC = ' + str(roc_auc)
        # plt.plot(fpr, tpr, lw=lw, color=color,
        #          label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
        
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
    #          label='Luck')

    mean_tpr /= cv.get_n_splits(x, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print 'Mean AUC = ', mean_auc

    # plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
    #          label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # # plt.show()
    # plt.draw()

    return mean_auc, best_classifier, probabilities, y_r, probabilities_epoch


def prob_decide(p, epoch_num):
    N = p.shape
    N = N[0]

    M = N/epoch_num
    p_res = np.zeros(M)
    p_x = np.zeros((M, epoch_num))

    idx = 0
    for i in range(0, N - N%epoch_num, epoch_num):
        p_res[idx] = np.mean(p[i:i+epoch_num])

        for j in range(0, epoch_num, 1):
            p_x[idx, j] = p[i+j]

        idx += 1
        
    return p_res, p_x


def load_test(feature_names, patient_i, path):
    x_test = []
    for feature_name in feature_names:
        # Load train data (train.csv)
        fname_test = path + 'test_' + str(patient_i) + '_new_' + feature_name + '.csv'
        data = pd.read_csv(fname_test, header=None)
        data.replace([np.inf, -np.inf], np.NaN)
        data = data.fillna(0)
        data = np.array(data)

        if len(x_test) == 0:
            x_test = data[:, 2:] 
            fnames = data[:, 0]
        else:
            x_test = np.concatenate((x_test, data[:, 2:]), axis=1)

    print 'Test: Number of observations = ' + str(x_test.shape[0]) + ' and number of features = ' + str(x_test.shape[1])

    return x_test, fnames


class ModelDescriptor:
    def __init__(self, feature_names, model_name):
        self.feature_names = feature_names
        self.model_name = model_name


# path = 'updated_features/'
# path = 'E:/GoogleDrive/nih_features/'
# path = '/media/usr/54532AD595ED8860/GoogleDrive/nih_features/'
path = '/users/oleg/Google Drive/nih_features/' # path where features are stored
spath = 'submissions/' # path to store submission files
mpath = 'models/' # path to store models
ppath = 'fprop/' # path to store mean, std and indexes of valid features from packs
pat_list = [1, 2, 3]

md_list = []

# # Model 1:
# feature_names = ['starter']
# model_name = 'gb'
# md_list.append(ModelDescriptor(feature_names, model_name))

# # Model 2:
# feature_names = ['starter']
# model_name = 'xgb'
# md_list.append(ModelDescriptor(feature_names, model_name))

# # Model 3:
# feature_names = ['starter']
# model_name = 'vot'
# md_list.append(ModelDescriptor(feature_names, model_name))

# # Model 4:
# feature_names = ['starter_old', 'spectral_v0']
# model_name = 'ada55_0'
# md_list.append(ModelDescriptor(feature_names, model_name))

# # Model 5:
# feature_names = ['starter_old', 'spectral_v0']
# model_name = 'rf_0'
# md_list.append(ModelDescriptor(feature_names, model_name))

# # Model 6:
# feature_names = ['starter_old', 'spectral_v0']
# model_name = 'gs'
# md_list.append(ModelDescriptor(feature_names, model_name))

# Model 7:
feature_names = ['reina_e30']
model_name = 'rfreina'
md_list.append(ModelDescriptor(feature_names, model_name))

epoch_num = 20
eval_flag = 0

now = time.strftime("%c")
print ('Started at ' + now)

for md in md_list:
    print('********************* ' + md.model_name + ' *********************')
    feature_names = md.feature_names
    model_name = md.model_name

    auc_list = []
    Y = []
    P = []
    F = []
    test_predictions = []
    for pat in pat_list:
        print 'Patient = ' + str(pat)
        x_test, fnames = load_test(feature_names, pat, path)
        
        print 'Test: Number of -Inf = ' + str(len(x_test[x_test>9999999])) + ', number if -Inf = ' + \
              str(len(x_test[x_test<-9999999]))

        # Remove Inf and -Inf from data
        x_test[x_test > 9999999] = 0
        x_test[x_test < -9999999] = 0

        # Features normalization and validation

        # load x_mean, x_std and valid_features for each patient

        x_mean = pickle.load(open(ppath + 'xmean_' + model_name.split('_')[0] + '_pat' + str(pat), 'rb'))
        x_std = pickle.load(open(ppath + 'xstd_' + model_name.split('_')[0] + '_pat' + str(pat), 'rb'))
        valid_features_idx = pickle.load(open(ppath + 'validf_' + model_name.split('_')[0] + '_pat' + str(pat), 'rb'))

        # valid_features_idx = np.zeros(x.shape[1], dtype='bool')
        # for i in range(0, x.shape[1]):
        #     if x_std[i] > 0.01:
        #         x_test[:, i] = (x_test[:, i] - x_mean[i]) / x_std[i]
        #     else:
        #         x[:, i] = (x[:, i] - x_mean[i])

        # # Features normalization
        for i in range(0, x_test.shape[1]):
            x_std = np.std(x_test[:, i])
            if x_std > 0.01:
                x_test[:, i] = (x_test[:, i] - np.mean(x_test[:, i]))/np.std(x_test[:, i])
            # else:
            #     x_test[:, i] = (x_test[:, i] - np.mean(x_test[:, i]))

        # Remove non-valid features
        x_test = x_test[:, valid_features_idx]

        print 'Test: Number of observations = ' + str(x_test.shape[0]) + ' and number of features = ' + str(x_test.shape[1])

        # Model test path (Kaggle)
        classifier = pickle.load(open(mpath + 'model_' + model_name + '_pat' + str(pat), 'rb'))
        probas = model_predict(classifier, x_test)

        # Evaluate probability for class basing on epochs classification
        pfile, probabilities_test = prob_decide(probas[:, 1], epoch_num)

        # Create list of file names for submission
        M = len(fnames) / epoch_num
        ffile = np.empty(M, dtype='S20')

        idx = 0
        for i in range(0, len(fnames), epoch_num):
            ffile[idx] = fnames[i]
            idx += 1

        # Save results to file 
        df = pd.DataFrame({'File': ffile, 'Class': pfile}, columns=['File', 'Class'], index=None)
        df.to_csv(spath + 'submission_' + model_name + '_pat_' + str(pat) + '.csv', sep=',', header=True, float_format='%.8f', index=False)

        test_predictions.append(df)

    # Overall AUC estimations:
    if eval_flag:
        print '*** Overall Mean AUC = ' + str(np.mean(auc_list)) + ' ***'

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        fpr, tpr, thresholds = roc_curve(Y, P)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        print '*** Overall AUC = ' + str(roc_auc) + ' ***'

    print 'Saving results to files'
    result = pd.concat(test_predictions)

    if not os.path.exists(spath):
        os.makedirs(spath)
    result.to_csv(spath + 'submission_' + model_name + '.csv', sep=',', header=True, index=False)

    # plt.show()

    now = time.strftime("%c")
    print ('Done at ' + now)
