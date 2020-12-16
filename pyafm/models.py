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
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold

from pyafm.util import invlogit
from pyafm.custom_logistic import CustomLogistic
from pyafm.bounded_logistic import BoundedLogistic


def afm(kcs, opps, actuals, stu, student_label, item_label, nfolds=3,
        seed=None):
    """
    Executes AFM on the provided data and returns model fits and parameter
    estimates
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
    coef_s = [[k, v, invlogit(v)]
              for k, v in sv.inverse_transform([coef_s])[0].items()]
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

    cvs = [KFold(n_splits=nfolds, shuffle=True, random_state=seed).split(X),
           StratifiedKFold(n_splits=nfolds, shuffle=True,
                           random_state=seed).split(X, y),
           GroupKFold(n_splits=nfolds).split(X, y, student_label),
           GroupKFold(n_splits=nfolds).split(X, y, item_label)]

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


def afms(kcs, opps, actuals, stu, student_label, item_label, nfolds=3, seed=None):
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
    coef_s = [[k, v, invlogit(v)]
              for k, v in sv.inverse_transform([coef_s])[0].items()]
    coef_q = model.coef1_[S.shape[1]:S.shape[1]+Q.shape[1]]
    coef_qint = qv.inverse_transform([coef_q])[0]
    coef_o = model.coef1_[S.shape[1]+Q.shape[1]                          :S.shape[1]+Q.shape[1]+O.shape[1]]
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

    # cvs = [KFold(len(y), n_splits=nfolds, shuffle=True, random_state=seed),
    #        StratifiedKFold(y, n_splits=nfolds, shuffle=True, random_state=seed),
    #        GroupKFold(student_label, n_splits=nfolds),
    #        GroupKFold(item_label, n_splits=nfolds)]

    cvs = [KFold(n_splits=nfolds, shuffle=True, random_state=seed).split(X),
           StratifiedKFold(n_splits=nfolds, shuffle=True,
                           random_state=seed).split(X, y),
           GroupKFold(n_splits=nfolds).split(X, y, student_label),
           GroupKFold(n_splits=nfolds).split(X, y, item_label)]

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
