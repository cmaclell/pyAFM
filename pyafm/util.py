from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from math import log
from math import exp

import numpy as np

def afm_predict(student_step_rollup, kc_model_name):
    ssr_file = student_step_rollup

    kcs, opps, y, stu, student_label, item_label = read_datashop_student_step(
        ssr_file, kc_model_name)

    # Get everything in the right matrix format
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

    afm = CustomLogistic(bounds=bounds, l2=l2, fit_intercept=False)
    afm.fit(X, y)
    yAFM = afm.predict_proba(X)

    return yAFM

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


