"""
This module provides functions directly for AFM and AFM+S so they can be
called in loops etc.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import LabelKFold

from util import invlogit
from online_logistic import OnlineLogistic
from custom_logistic import CustomLogistic
from bounded_logistic import BoundedLogistic

from matplotlib import pyplot as plt


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def online_afm(kcs, opps, actuals, stu, student_label, item_label):
    """
    Executes AFM on the provided data and returns model fits and parameter estimates
    """
    sv = DictVectorizer()
    qv = DictVectorizer()
    ov = DictVectorizer()

    new_opps = []
    for o in opps:
        new_o = {}
        for e in o:
            new_o[e] = o[e]
            new_o['general_opp'] = o[e]
        new_opps.append(new_o)

    S = sv.fit_transform(stu)
    Q = qv.fit_transform(kcs)
    O = ov.fit_transform(opps)

    X = hstack((S, Q, O))
    y = np.array(actuals)

    l2 = [1.0 for i in range(S.shape[1])] 
    l2 += [0.0 for i in range(Q.shape[1])] 
    l2 += [0.0 for i in range(O.shape[1])]

    bounds = [(None, None) for i in range(S.shape[1])] 
    bounds += [(None, None) for i in range(Q.shape[1])] 
    bounds += [(0, None) for i in range(O.shape[1])]

    X = X.toarray()
    X2 = Q.toarray()

    model = OnlineLogistic(bounds=bounds, l2=l2, fit_intercept=True)
    # model = CustomLogistic(bounds=bounds, l2=l2, fit_intercept=False)

    scores = []

    best_error = float('inf')
    best_beta1 = None
    best_beta2 = None
    best_alpha = None

    for _ in range(1000):
        beta1 = np.random.beta(1, 4)
        beta2 = np.random.beta(1, 4)
        alpha = np.random.beta(4, 1)
        model = OnlineLogistic(bounds=bounds, l2=l2,
                 fit_intercept=True, alpha=alpha, beta1=beta1,
                 beta2=beta2)
        score = []
        for i in range(len(y)):
            score.append(model.mean_squared_error(X[i:i+1], y[i:i+1]))
            model.partial_fit(X[i:i+1], y[i:i+1])
        error = np.mean(np.sqrt(score))
        if error < best_error:
            print('--------------------')
            print('Found better', error)
            print('alpha =', alpha)
            print('beta1 =', beta1)
            print('beta2 =', beta2)
            print('--------------------')
            print()
            best_error = error
            best_alpha = alpha
            best_beta1 = beta1
            best_beta2 = beta2

    print('best_error', best_error)
    print('best_alpha', best_alpha)
    print('best_beta1', best_beta1)
    print('best_beta2', best_beta2)
    model = OnlineLogistic(bounds=bounds, l2=l2,
             fit_intercept=True, alpha=best_alpha, beta1=best_beta1,
             beta2=best_beta2)
    score = []
    for i in range(len(y)):
        score.append(model.mean_squared_error(X[i:i+1], y[i:i+1]))
        model.partial_fit(X[i:i+1], y[i:i+1])

    avg = movingaverage(np.sqrt(score), 300)
    # plt.plot(np.sqrt(score))
    plt.plot(avg)
    print(np.mean(np.sqrt(score)))
    plt.show()
    scores.append(np.mean(np.sqrt(score)))

    coef_s = model.coef_[0:S.shape[1]]
    coef_s = [[k, v, invlogit(v)] for k, v in sv.inverse_transform([coef_s])[0].items()]
    coef_q = model.coef_[S.shape[1]:S.shape[1]+Q.shape[1]]
    coef_qint = qv.inverse_transform([coef_q])[0]
    coef_o = model.coef_[S.shape[1]+Q.shape[1]:S.shape[1]+Q.shape[1]+O.shape[1]]
    coef_qslope = ov.inverse_transform([coef_o])[0]

    kc_vals = []
    all_kcs = set(coef_qint).union(set(coef_qslope))

    for kc in all_kcs:
        kc_vals.append([kc, coef_qint.setdefault(kc, 0.0),
                        invlogit(coef_qint.setdefault(kc, 0.0)),
                        coef_qslope.setdefault(kc, 0.0)])

    return scores, kc_vals, coef_s


def afm(kcs, opps, actuals, stu, student_label, item_label, nfolds=3, seed=None):
    """
    Executes AFM on the provided data and returns model fits and parameter estimates
    """
    sv = DictVectorizer()
    qv = DictVectorizer()
    ov = DictVectorizer()

    S = sv.fit_transform(stu)
    Q = qv.fit_transform(kcs)
    O = ov.fit_transform(opps)

    X = hstack((S, Q, O))
    y = np.array(actuals)

    l2 = [1.0 for i in range(S.shape[1])] 
    l2 += [0.0 for i in range(Q.shape[1])] 
    l2 += [0.0 for i in range(O.shape[1])]

    bounds = [(None, None) for i in range(S.shape[1])] 
    bounds += [(None, None) for i in range(Q.shape[1])] 
    bounds += [(0, None) for i in range(O.shape[1])]

    X = X.toarray()
    X2 = Q.toarray()

    model = CustomLogistic(bounds=bounds, l2=l2, fit_intercept=False)
    model.fit(X, y)

    coef_s = model.coef_[0:S.shape[1]]
    coef_s = [[k, v, invlogit(v)] for k, v in sv.inverse_transform([coef_s])[0].items()]
    coef_q = model.coef_[S.shape[1]:S.shape[1]+Q.shape[1]]
    coef_qint = qv.inverse_transform([coef_q])[0]
    coef_o = model.coef_[S.shape[1]+Q.shape[1]:S.shape[1]+Q.shape[1]+O.shape[1]]
    coef_qslope = ov.inverse_transform([coef_o])[0]

    kc_vals = []
    all_kcs = set(coef_qint).union(set(coef_qslope))

    for kc in all_kcs:
        kc_vals.append([kc, coef_qint.setdefault(kc, 0.0),
                        invlogit(coef_qint.setdefault(kc, 0.0)),
                        coef_qslope.setdefault(kc, 0.0)])

    cvs = [KFold(len(y), n_folds=nfolds, shuffle=True, random_state=seed),
           StratifiedKFold(y, n_folds=nfolds, shuffle=True, random_state=seed),
           LabelKFold(student_label, n_folds=nfolds),
           LabelKFold(item_label, n_folds=nfolds)]

    scores = []
    for cv in cvs:
        score = []
        for train_index, test_index in cv:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            score.append(model.mean_squared_error(X_test, y_test))
        scores.append(np.mean(np.sqrt(score)))

    return scores, kc_vals, coef_s

def afms (kcs, opps, actuals, stu, student_label, item_label, nfolds=3, seed=None):
    """
    Executes AFM+S on the provided data and returns model fits and parameter estimates
    """
    sv = DictVectorizer()
    qv = DictVectorizer()
    ov = DictVectorizer()

    S = sv.fit_transform(stu)
    Q = qv.fit_transform(kcs)
    O = ov.fit_transform(opps)

    X = hstack((S, Q, O))
    y = np.array(actuals)

    l2 = [1.0 for i in range(S.shape[1])]
    l2 += [0.0 for i in range(Q.shape[1])]
    l2 += [0.0 for i in range(O.shape[1])]

    bounds = [(None, None) for i in range(S.shape[1])]
    bounds += [(None, None) for i in range(Q.shape[1])]
    bounds += [(0, None) for i in range(O.shape[1])]

    X = X.toarray()
    X2 = Q.toarray()

    model = BoundedLogistic(first_bounds=bounds, first_l2=l2)
    model.fit(X, X2, y)
    coef_s = model.coef1_[0:S.shape[1]]
    coef_s = [[k, v, invlogit(v)] for k, v in sv.inverse_transform([coef_s])[0].items()]
    coef_q = model.coef1_[S.shape[1]:S.shape[1]+Q.shape[1]]
    coef_qint = qv.inverse_transform([coef_q])[0]
    coef_o = model.coef1_[S.shape[1]+Q.shape[1]:S.shape[1]+Q.shape[1]+O.shape[1]]
    coef_qslope = ov.inverse_transform([coef_o])[0]
    coef_qslip = qv.inverse_transform([model.coef2_])[0]
    
    kc_vals = []
    all_kcs = set(coef_qint).union(set(coef_qslope)).union(set(coef_qslip))
    for kc in all_kcs:
        kc_vals.append([kc,
                        coef_qint.setdefault(kc, 0.0),
                        invlogit(coef_qint.setdefault(kc, 0.0)),
                        coef_qslope.setdefault(kc, 0.0),
                        coef_qslip.setdefault(kc, 0.0)])

    cvs = [KFold(len(y), n_folds=nfolds, shuffle=True, random_state=seed),
           StratifiedKFold(y, n_folds=nfolds, shuffle=True, random_state=seed),
           LabelKFold(student_label, n_folds=nfolds),
           LabelKFold(item_label, n_folds=nfolds)]

    # scores_header = []
    scores = []
    for cv in cvs:
        score = []
        for train_index, test_index in cv:
            X_train, X_test = X[train_index], X[test_index]
            X2_train, X2_test = X2[train_index], X2[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, X2_train, y_train)
            score.append(model.mean_squared_error(X_test, X2_test, y_test))
        # scores_header.append(cv_name)
        scores.append(np.mean(np.sqrt(score)))

    return scores, kc_vals, coef_s
