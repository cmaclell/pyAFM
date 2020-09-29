import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer

from pyafm.process_datashop import read_datashop_student_step
from pyafm.custom_logistic import CustomLogistic


def afm_predict(student_step_rollup):
    with open(student_step_rollup, 'r') as ssr_file:
        kcs, opps, y, stu, student_label, item_label = read_datashop_student_step(
            ssr_file)

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
