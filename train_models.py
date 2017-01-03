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
from sklearn.ensemble import BaggingClassifier
import xgboost

# Serialize models
import pickle

def model_create(model_name):
    if model_name == 'ada55':
        classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                         algorithm="SAMME",
                         n_estimators=5)
    elif model_name == 'xgb':
        classifier = xgboost.XGBClassifier(n_estimators=800)
    elif model_name == 'gb':
        classifier = GradientBoostingClassifier(n_estimators=1000) 
    elif model_name == 'rf':
        classifier = RandomForestClassifier() 
    elif model_name == 'vot':
        param_grid = {"base_estimator__criterion" : ["gini"], "base_estimator__splitter" :   ["best"],  "n_estimators": [3,5, 6]}
        DTC = DecisionTreeClassifier(max_depth=5)
        ABC = AdaBoostClassifier(base_estimator = DTC, algorithm="SAMME", learning_rate=1, n_estimators=5)
        clf1 = GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc')
        clf2 = GradientBoostingClassifier(n_estimators=1000)
        clf3 = BaggingClassifier()
        classifier = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('rtf', clf3)], voting='soft') 
    elif model_name == 'gs':
        clf1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME")
        param_grid = {'n_estimators': [4, 5, 6]}
        classifier = GridSearchCV(clf1, param_grid=param_grid, scoring='roc_auc')
    elif model_name == 'rfreina':
        classifier = RandomForestClassifier(n_estimators=50) 
    else:
        raise ValueError('There are no model of type \'' + model_name + '\'')

    return classifier


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


def model_run(model_name, x, y, x_test):
    print 'Creating model on whole train set...'
    classifier = model_create(model_name)

    print 'Model fitting...'
    classifier = model_fit(classifier, x, y)

    print 'Predicting...'
    p = model_predict(classifier, x_test)
    return classifier, p


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

def load_train_competition(feature_names, patient_i, path):
    x_train1 = []
    for feature_name in feature_names:
        # Load train data (train.csv)
        fname_train1 = path + 'train_' + str(patient_i) + '_' + feature_name + '.csv'
        data = pd.read_csv(fname_train1, header = None)
        data = data.fillna(0)
        data = np.array(data)

        if len(x_train1) == 0:
            fnames_train1 = data[:, 0]
            x_train1 = data[:, 2:-1] 
            y_train1 = data[:, -1]
            y_train1 = np.asarray(y_train1, dtype="int")
        else:
            x_train1 = np.concatenate((x_train1, data[:, 2:-1]), axis=1)

    print 'Train (based on train): Number of observations = ' + str(
        x_train1.shape[0]) + ' and number of features = ' + str(x_train1.shape[1]) + \
          ', number of Y(1) = ' + str(np.sum(y_train1))

    x_train2 = []
    for feature_name in feature_names:
        # Load train data (train.csv)
        fname_train2 = path + 'test_' + str(patient_i) + '_' + feature_name + '.csv'
        data = pd.read_csv(fname_train2, header = None)
        data = data.fillna(0)
        data = np.array(data)

        if len(x_train2) == 0:
            fnames_train2 = data[:, 0]
            x_train2 = data[:, 2:-1]
            y_train2 = data[:, -1]
            y_train2 = np.asarray(y_train2, dtype="int")
        else:
            x_train2 = np.concatenate((x_train2, data[:, 2:-1]), axis=1)

    print 'Train (based on test): Number of observations = ' + str(
        x_train2.shape[0]) + ' and number of features = ' + str(x_train2.shape[1]) + \
          ', number of Y(1) = ' + str(np.sum(y_train2))

    x = np.concatenate((x_train1, x_train2))
    y = np.concatenate((y_train1, y_train2))
    fnames_train = np.concatenate((fnames_train1, fnames_train2))

    print 'Train (after merge): Number of observations = ' + str(
    x.shape[0]) + ' and number of features = ' + str(x.shape[1]) + \
      ', number of Y(1) = ' + str(np.sum(y))

    return x, y, fnames_train


def load_train(feature_names, patient_i, path):
    x = []
    for feature_name in feature_names:
        # Load train data (train.csv)
        fname = path + 'train_' + str(patient_i) + '_' + feature_name + '.csv'
        data = pd.read_csv(fname, header = None)
        data = data.fillna(0)
        data = np.array(data)

        if len(x) == 0:
            fnames = data[:, 0]
            x = data[:, 2:-1] 
            y = data[:, -1]
            y = np.asarray(y, dtype="int")
        else:
            x = np.concatenate((x, data[:, 2:-1]), axis=1)

    print 'Train (based on train): Number of observations = ' + str(
        x.shape[0]) + ' and number of features = ' + str(x.shape[1]) + \
          ', number of Y(1) = ' + str(np.sum(y))

    return x, y, fnames


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


# path = 'E:/GoogleDrive/nih_features/'
path = '/users/oleg/Google Drive/nih_features/' # path where features are stored
spath = 'submissions/' # path to store submission files
mpath = 'models/' # path to store models
ppath = 'fprop/' # path to store mean, std and indexes of valid features from packs
pat_list = [1, 2, 3]

md_list = []

# Model 1:
feature_names = ['starter']
model_name = 'gb'
md_list.append(ModelDescriptor(feature_names, model_name))

# Model 2:
feature_names = ['starter']
model_name = 'xgb'
md_list.append(ModelDescriptor(feature_names, model_name))

# Model 3:
feature_names = ['starter']
model_name = 'vot'
md_list.append(ModelDescriptor(feature_names, model_name))

# Model 4:
feature_names = ['starter_old', 'spectral_v0']
model_name = 'ada55_0'
md_list.append(ModelDescriptor(feature_names, model_name))

model_name = 'ada55_1'
md_list.append(ModelDescriptor(feature_names, model_name))

model_name = 'ada55_2'
md_list.append(ModelDescriptor(feature_names, model_name))

# Model 5:
feature_names = ['starter_old', 'spectral_v0']
model_name = 'rf_0'
md_list.append(ModelDescriptor(feature_names, model_name))

model_name = 'rf_1'
md_list.append(ModelDescriptor(feature_names, model_name))

model_name = 'rf_2'
md_list.append(ModelDescriptor(feature_names, model_name))

# Model 6:
feature_names = ['starter_old', 'spectral_v0']
model_name = 'gs'
md_list.append(ModelDescriptor(feature_names, model_name))

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
        x, y, fnames_train = load_train_competition(feature_names, pat, path)
        # x, y, fnames_train = load_train(feature_names, pat, path) # uncomment this to work without loading data from old test set in competition

        x_test, fnames = load_test(feature_names, pat, path)
        
        print 'Train: Number of -Inf = ' + str(len(x[x > 9999999])) + ', number if -Inf = ' + \
              str(len(x[x < -9999999]))
        print 'Test: Number of -Inf = ' + str(len(x_test[x_test>9999999])) + ', number if -Inf = ' + \
              str(len(x_test[x_test<-9999999]))

        # Remove Inf and -Inf from data
        x[x > 9999999] = 0
        x[x < -9999999] = 0
        x_test[x_test > 9999999] = 0
        x_test[x_test < -9999999] = 0

        # Features normalization and validation
        valid_features_idx = np.zeros(x.shape[1], dtype='bool')
        x_mean = np.zeros(x.shape[1])
        x_std = np.zeros(x.shape[1])
        for i in range(0, x.shape[1]):
            x_mean[i] = np.mean(x[:, i])
            x_std[i] = np.std(x[:, i])
            # print x_std

            if x_std[i] > 0.01:
                x[:, i] = (x[:, i] - x_mean[i]) / x_std[i]
                x_test[:, i] = (x_test[:, i] - x_mean[i]) / x_std[i]
            # else:
            #     x[:, i] = (x[:, i] - x_mean[i])
            #     x_test[:, i] = (x_test[:, i] - x_mean[i]) 

            if x_std[i] > 0:
                valid_features_idx[i] = 1

        if not os.path.exists(ppath):
            os.makedirs(ppath)
        pickle.dump(x_mean, open(ppath + 'xmean_' + model_name.split('_')[0] + '_pat' + str(pat), 'wb'))
        pickle.dump(x_std, open(ppath + 'xstd_' + model_name.split('_')[0] + '_pat' + str(pat), 'wb'))
        pickle.dump(valid_features_idx, open(ppath + 'validf_' + model_name.split('_')[0] + '_pat' + str(pat), 'wb'))

        # Features normalization
        for i in range(0, x_test.shape[1]):
            x_std = np.std(x_test[:, i])
            if x_std > 0.01:
                x_test[:, i] = (x_test[:, i] - np.mean(x_test[:, i]))/np.std(x_test[:, i])
            # else:
            #     x_test[:, i] = (x_test[:, i] - np.mean(x_test[:, i]))

        # Remove non-valid features
        x = x[:, valid_features_idx]
        x_test = x_test[:, valid_features_idx]

        print 'Train: Number of observations = ' + str(x.shape[0]) + ' and number of features = ' + str(x.shape[1])
        print 'Test: Number of observations = ' + str(x_test.shape[0]) + ' and number of features = ' + str(x_test.shape[1])

        if eval_flag:
            # Model evaluation path
            mean_auc, best_classifier, probabilities, y_r, probabilities_epoch = model_evaluate(model_name.split('_')[0], x, y, epoch_num)
            auc_list.append(mean_auc)
            Y = np.concatenate((Y, y_r))
            P = np.concatenate((P, probabilities))
            fnames_train = fnames_train[0:len(fnames_train):epoch_num]
            F = np.concatenate((F, fnames_train))

        # Model test path (Kaggle)
        classifier, probas = model_run(model_name.split('_')[0], x, y, x_test)

        # Evaluate probability for class basing on epochs classification
        pfile, probabilities_test = prob_decide(probas[:, 1], epoch_num)

        print 'Saving the model...'
        if not os.path.exists(mpath):
            os.makedirs(mpath)
        pickle.dump(classifier, open(mpath + 'model_' + model_name + '_pat' + str(pat), 'wb'))

        # Create list of file names for submission
        M = len(fnames) / epoch_num
        ffile = np.empty(M, dtype='S20')

        idx = 0
        for i in range(0, len(fnames), epoch_num):
            ffile[idx] = fnames[i]
            idx += 1

        # Save results to file 
        if not os.path.exists(spath):
            os.makedirs(spath)
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

        # df = pd.DataFrame({'File': F, 'Class': P, 'RealClass': Y}, columns=['File', 'Class', 'RealClass'], index=None)
        # df.to_csv('train_submission.csv', sep=',', header=True, index=False)

    print 'Saving results to files'
    # df1 = pd.read_csv('pat_1.csv', header = 0)
    # df2 = pd.read_csv('pat_2.csv', header = 0)
    # df3 = pd.read_csv('pat_3.csv', header = 0)

    # frames = [df1, df2, df3]
    # result = pd.concat(frames)

    result = pd.concat(test_predictions)

    if not os.path.exists(spath):
        os.makedirs(spath)
    result.to_csv(spath + 'submission_' + model_name + '.csv', sep=',', header=True, index=False)

    # plt.show()

now = time.strftime("%c")
print ('Done at ' + now)
