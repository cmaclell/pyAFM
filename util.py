from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from math import log
from math import exp

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer

def log_one_plus_exp(z):
    """
    This function returns log(1 + exp(z)) where it rewrites the terms to reduce
    floating point errors.
    """
    if z > 0:
        return log(1 + exp(-z)) + z
    else:
        return log(1 + exp(z))

def invlogit(z):
    """
    This function return 1 / (1 + exp(-z)) where it rewrites the terms to
    reduce floating point errors.
    """
    if z > 0:
        return 1 / (1 + exp(-z))
    else:
        return exp(z) / (1 + exp(z))

invlogit_vect = np.vectorize(invlogit)
log_one_plus_exp_vect = np.vectorize(log_one_plus_exp)


def avg_y_by_x(x, y):
    x = np.array(x)
    y = np.array(y)

    xs = sorted(list(set(x)))

    xv = []
    yv = []

    for v in xs:
        ys = [y[i] for i,e in enumerate(x) if e == v]
        if len(ys) > 0:
            xv.append(v)
            yv.append(sum(ys) / len(ys))

    return xv, yv

def opps_to_vecs(kcs, opps, y, stu):
    sv = DictVectorizer()
    qv = DictVectorizer()
    ov = DictVectorizer()
    S = sv.fit_transform(stu)
    Q = qv.fit_transform(kcs)
    O = ov.fit_transform(opps)
    X = hstack((S, Q, O))
    y = np.array(y)

    # Regularize the student intercepts
    l2 = [1.0 for i in range(S.shape[1])] 
    l2 += [0.0 for i in range(Q.shape[1])] 
    l2 += [0.0 for i in range(O.shape[1])]

    # Bound the learning rates to be positive
    bounds = [(None, None) for i in range(S.shape[1])] 
    bounds += [(None, None) for i in range(Q.shape[1])] 
    bounds += [(0, None) for i in range(O.shape[1])]
    
    X = X.toarray()
    X2 = Q.toarray()

    return X, y, bounds, l2, X2