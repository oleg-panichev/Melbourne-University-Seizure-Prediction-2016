# coding: utf-8
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
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import xgboost


from sklearn.pipeline import Pipeline

from sklearn.calibration import CalibratedClassifierCV

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

# Outliers:
from sklearn.ensemble import IsolationForest

from sklearn.base import BaseEstimator, ClassifierMixin
class CustomEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers=None):
        self.classifiers = classifiers

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)

    def predict_proba(self, X):
        self.predictions_ = list()
        for classifier in self.classifiers:
            self.predictions_.append(classifier.predict_proba(X))

        return np.mean(self.predictions_, axis=0)

def model_create(type):
    if type == 'ada55':
        classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                         algorithm="SAMME",
                         n_estimators=5)
    elif type == 'xgb':
        classifier = xgboost.XGBClassifier(n_estimators=800)
    elif type == 'gb':
        classifier = GradientBoostingClassifier(n_estimators=1000) 
    elif type == 'rf':
        classifier = RandomForestClassifier() 
    elif type == 'gs1':
        param_grid = {"base_estimator__criterion" : ["gini"], "base_estimator__splitter" :   ["best"],  "n_estimators": [3,5, 6]}
        DTC = DecisionTreeClassifier(max_depth=5)
        ABC = AdaBoostClassifier(base_estimator = DTC, algorithm="SAMME", learning_rate=1, n_estimators=5)
        clf1 = GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc')
        clf2 = GradientBoostingClassifier(n_estimators=1000)
        clf3 = BaggingClassifier()
        classifier = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('rtf', clf3)], voting='soft') 
    elif type == 'gs2':
        clf1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME")
        param_grid = {'n_estimators': [4, 5, 6]}
        classifier = GridSearchCV(clf1, param_grid=param_grid, scoring='roc_auc')
    elif type == 'rf_reina':
        classifier = RandomForestClassifier(n_estimators=50) 


    # classifier = svm.SVC(kernel='rbf', probability=True,
    #                      random_state=None)

    # classifier = svm.NuSVC(probability=True)

    # classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
    #                      algorithm="SAMME",
    #                      n_estimators=5)

    # classifier = AdaBoostClassifier(LogisticRegression(),
    #                                 algorithm="SAMME",
    #                                 n_estimators=100)

    # classifier = CalibratedClassifierCV(classifier, method='sigmoid', cv=5)

    # classifier = AdaBoostClassifier(GaussianNB(),
    #                                 algorithm="SAMME",
    #                                 n_estimators=5)

    # classifier = BaggingClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
    #                      algorithm="SAMME",
    #                      n_estimators=5),
    #                    max_samples = 0.5, max_features = 0.5, n_jobs= 1)

    # classifier = GaussianNB()

    # classifier = BernoulliNB()

    # classifier = DecisionTreeClassifier(max_depth=5)

    # classifier = RandomForestClassifier(max_depth=1, n_estimators=50, n_jobs=4)#, max_features=100)
    classifier = RandomForestClassifier()

    # classifier = ExtraTreesClassifier(n_estimators=20, max_depth=2, min_samples_split = 2, random_state = 0, max_features=100)

    # classifier = KNeighborsClassifier(n_neighbors=1)

    # classifier = LogisticRegression()

    # classifier = LogisticRegressionCV()

    # classifier = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)

    # classifier = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')

    # classifier = QuadraticDiscriminantAnalysis()

    # classifier = MLPClassifier(alpha=0.5)

    # classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes = (30, 20, 10), random_state = 1)

    # classifier = GradientBoostingClassifier(n_estimators=1) #, learning_rate=0.5, max_depth = 5, random_state = 0)
    # classifier = GradientBoostingClassifier(n_estimators=100)

    # classifier = xgboost.XGBClassifier(n_estimators=800)

    # classifier = Pipeline([
    #     ('feature_selection', SelectFromModel(LinearSVC())),
    #     ('classification', AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
    #                      algorithm="SAMME",
    #                      n_estimators=5))
    # ])

    # classifier = xgboost.XGBClassifier(silent=1, nthread=3, max_depth=5)

    # classifier = TfMultiLayerPerceptron(eta=0.5,
    #                              epochs=20,
    #                              hidden_layers=[10],
    #                              activations=['logistic'],
    #                              optimizer='gradientdescent',
    #                              print_progress=3,
    #                              minibatches=1,
    #                              random_seed=1)

    # # clf1 = GaussianNB()
    # clf2 = DecisionTreeClassifier(max_depth=6)
    # clf2_1 = DecisionTreeClassifier(max_depth=5)
    # clf2_2 = DecisionTreeClassifier(max_depth=7)
    # # clf3 = LogisticRegression()
    # # clf4 = KNeighborsClassifier()
    # # clf4_1 = KNeighborsClassifier(n_neighbors=1)
    # # clf5 = ExtraTreesClassifier()
    # # clf5_1 = ExtraTreesClassifier(n_estimators=20)
    # clf6 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
    #                                                 algorithm="SAMME",
    #                                                 n_estimators=5)
    # clf6_1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
    #                           algorithm="SAMME",
    #                           n_estimators=4)
    # clf6_2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
    #                           algorithm="SAMME",
    #                           n_estimators=6)
    # clf6_3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=6),
    #                           algorithm="SAMME",
    #                           n_estimators=5)
    # clf6_4 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
    #                           algorithm="SAMME",
    #                           n_estimators=5)
    #
    # clf7 = RandomForestClassifier()
    # clf7_1 = RandomForestClassifier(n_estimators=20)
    # clf7_2 = RandomForestClassifier(n_estimators=5)
    # clf8 = GradientBoostingClassifier()
    # clf8_1 = GradientBoostingClassifier(n_estimators=200)
    # clf9 = xgboost.XGBClassifier()
    # clf9_1 = xgboost.XGBClassifier(n_estimators=200)
    # clf9_2 = xgboost.XGBClassifier(n_estimators=50)
    # classifier = CustomEnsembleClassifier([#clf1,
    #                                        clf2, clf2_1, clf2_2,
    #                                        #clf3,
    #                                        #clf4, clf4_1,
    #                                        #clf5, clf5_1,
    #                                        clf6, clf6_1, clf6_2, clf6_3, clf6_4,
    #                                        clf7, clf7_1, clf7_2,
    #                                        clf8, clf8_1,
    #                                        clf9, clf9_1, clf9_2
    # ])


    # clf3 = GradientBoostingClassifier(n_estimators=1000)
    # classifier = VotingClassifier([('ada5', clf1), ('dt5', clf2), ('gb1000', clf3)], weights=[1, 1], voting='soft')

    # clf1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME")
    # param_grid = {'n_estimators': [4, 5, 6]}
    # classifier = GridSearchCV(clf1, param_grid=param_grid, scoring='roc_auc')

    return classifier


def model_fit(classifier, x, y, valid_signal_flags_train):
    classifier.fit(x, y)
    return classifier


def model_predict(classifier, x_test):
    p = classifier.predict_proba(x_test)
    return p


def model_evaluate(x, y, valid_signal_flags_train, epoch_num):
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
        classifier = model_create()
        if len(valid_signal_flags_train) > 0:
            classifier = model_fit(classifier, x[train_full], y[train_full], valid_signal_flags_train[train_full])
        else:
            classifier = model_fit(classifier, x[train_full], y[train_full], valid_signal_flags_train)

        # print 'Predicting...'
        probas = model_predict(classifier, x[test_full])

        if len(valid_signal_flags_train) > 0:
            p, p_x = prob_decide(probas[:, 1], valid_signal_flags_train[test_full], epoch_num)
        else:
            p, p_x = prob_decide(probas[:, 1], valid_signal_flags_train, epoch_num)

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


def model_run(x, y, valid_signal_flags_train, x_test):
    print 'Creating model on whole train set...'
    classifier = model_create()

    print 'Model fitting...'
    classifier = model_fit(classifier, x, y, valid_signal_flags_train)

    print 'Predicting...'
    p = model_predict(classifier, x_test)
    return p


def prob_decide(p, valid_signal_flags, epoch_num):
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


# path = 'updated_features/'
# path = 'E:/GoogleDrive/nih_features/'
# path = '/media/usr/54532AD595ED8860/GoogleDrive/nih_features/'
path = '/users/oleg/Google Drive/nih_features/'
spath = 'submits_oldschool/'
pat_list = [1, 2, 3]
# feature_names = ['spectral_v1']#'['starter', 'corrc', 'spectral_v0']#, 'spectral_psd_v1']
# feature_names = ['starter', 'corrc', 'spectral_v0']
# feature_names = ['reina_e30', 'spectral_v0']#['starter_old', 'spectral_v0']
# feature_names = ['ada5p']
# feature_names = ['starter_v1_e60', 'spectral_v3_e60']
feature_names = ['starter_old', 'spectral_v0']#, 'spectral_psd_v1']#_old', 'spectral_v0']
# feature_names = ['reina_e30']
valid_signal_fname = ''#''valid_signal'
# all feature names = ['starter', 'corrc', 'spectral_v0','spectral_v1]
epoch_num = 20
eval_flag = 0

now = time.strftime("%c")
print ('Started at ' + now)

auc_list = []
Y = []
P = []
F = []
test_predictions = []
for pat in pat_list:
    print 'Patient = ' + str(pat)
    x, y, fnames_train = load_train_competition(feature_names, pat, path)
    # x, y, fnames_train = load_train(feature_names, pat, path) # uncomment this to work without loading data from old test set in competition

    x_test, fnames = load_test(feature_names, patient_i, path)
    
    print 'Train: Number of -Inf = ' + str(len(x[x > 9999999])) + ', number if -Inf = ' + \
          str(len(x[x < -9999999]))
    print 'Test: Number of -Inf = ' + str(len(x_test[x_test>9999999])) + ', number if -Inf = ' + \
          str(len(x_test[x_test<-9999999]))

    # Remove Inf and -Inf from data
    x[x > 9999999] = 0
    x[x < -9999999] = 0
    x_test[x_test > 9999999] = 0
    x_test[x_test < -9999999] = 0
    print 'Train: Number of -Inf = ' + str(len(x[x > 9999999])) + ', number if -Inf = ' + \
          str(len(x[x < -9999999]))
    print 'Test: Number of -Inf = ' + str(len(x_test[x_test > 9999999])) + ', number if -Inf = ' + \
          str(len(x_test[x_test < -9999999]))

    # Features normalization and validation
    valid_features_idx = np.zeros(x.shape[1], dtype='bool')
    for i in range(0, x.shape[1]):
        x_mean = np.mean(x[:, i])
        x_std = np.std(x[:, i])
        # print x_std

        if x_std > 0.01:
            x[:, i] = (x[:, i] - x_mean) / x_std
            # x_test[:, i] = (x_test[:, i] - x_mean) / x_std
        else:
            x[:, i] = (x[:, i] - np.mean(x[:, i]))

        if x_std > 0:
            valid_features_idx[i] = 1

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
        mean_auc, best_classifier, probabilities, y_r, probabilities_epoch = model_evaluate(x, y, valid_signal_flags_train, epoch_num)
        auc_list.append(mean_auc)
        Y = np.concatenate((Y, y_r))
        P = np.concatenate((P, probabilities))
        fnames_train = fnames_train[0:len(fnames_train):epoch_num]
        F = np.concatenate((F, fnames_train))

    # Model test path (Kaggle)
    probas = model_run(x, y, valid_signal_flags_train, x_test)

    # Evaluate probability for class basing on epochs classification
    pfile, probabilities_test = prob_decide(probas[:, 1], valid_signal_flags_test, epoch_num)

    # Create list of file names for submission
    M = len(fnames) / epoch_num
    ffile = np.empty(M, dtype='S20')

    idx = 0
    for i in range(0, len(fnames), epoch_num):
        ffile[idx] = fnames[i]
        idx += 1

    # Save results to file 
    # output_fname = 'pat_' + str(pat) + '.csv'
    df = pd.DataFrame({'File': ffile, 'Class': pfile}, columns=['File', 'Class'], index=None)
    # df.to_csv(output_fname, sep=',', header=True, float_format='%.8f', index=False)

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

result.to_csv(spath + 'submission.csv', sep=',', header=True, index=False)

# plt.show()

now = time.strftime("%c")
print ('Done at ' + now)
