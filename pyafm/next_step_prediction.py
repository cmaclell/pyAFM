import argparse
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer

from pyafm.roll_up import transaction_to_student_step
from pyafm.process_datashop import read_datashop_student_step
from pyafm.custom_logistic import CustomLogistic
from pyafm.bounded_logistic import BoundedLogistic


def afm_predict_next_step(ssr_file, model_type):
    results = read_datashop_student_step(ssr_file)
    kcs, opps, y, stu, student_label, item_label = results

    # Get students
    next_step_opps = {}
    for i in range(len(stu)):
        s = list(stu[i])[0]
        if s not in next_step_opps:
            next_step_opps[s] = {}
        for kc in kcs[i]:
            if (kc not in next_step_opps[s] or
                    next_step_opps[s][kc] < opps[i][kc] + 1):
                next_step_opps[s][kc] = opps[i][kc] + 1

    # Get everything in the right matrix format
    sv = DictVectorizer()
    qv = DictVectorizer()
    ov = DictVectorizer()
    S = sv.fit_transform(stu)
    Q = qv.fit_transform(kcs)
    OP = ov.fit_transform(opps)
    X = hstack((S, Q, OP))
    y = np.array(y)

    # Regularize the student intercepts
    l2 = [1.0 for i in range(S.shape[1])]
    l2 += [0.0 for i in range(Q.shape[1])]
    l2 += [0.0 for i in range(OP.shape[1])]

    # Bound the learning rates to be positive
    bounds = [(None, None) for i in range(S.shape[1])]
    bounds += [(None, None) for i in range(Q.shape[1])]
    bounds += [(0, None) for i in range(OP.shape[1])]

    X = X.toarray()

    if model_type == "AFM":
        afm = CustomLogistic(bounds=bounds, l2=l2, fit_intercept=False)
        afm.fit(X, y)
    elif model_type == "AFM+S":
        X2 = Q.toarray()
        afms = BoundedLogistic(first_bounds=bounds, first_l2=l2)
        afms.fit(X, X2, y)

    # Get everything in the right matrix format for predicting next steps.
    new_stu = []
    new_kcs = []
    new_opps = []
    for s in next_step_opps:
        for kc in next_step_opps[s]:
            new_stu.append({s: 1})
            new_kcs.append({kc: 1})
            new_opps.append({kc: next_step_opps[s][kc]})
    NS = sv.transform(new_stu)
    NQ = qv.transform(new_kcs)
    NOP = ov.transform(new_opps)
    NX = hstack((NS, NQ, NOP))
    NX = NX.toarray()

    if model_type == "AFM":
        nextY = afm.predict_proba(NX)
    elif model_type == "AFM+S":
        NX2 = NQ.toarray()
        nextY = afms.predict_proba(NX, NX2)

    with open('next_step_predictions.txt', 'w') as fout:
        fout.write('student,kc,opportunity,prediction\n')
        for i in range(len(new_stu)):
            s = list(new_stu[i])[0]
            kc = list(new_kcs[i])[0]
            opps = new_opps[i][kc]
            fout.write("{},{},{},{}\n".format(s, kc, opps, nextY[i]))


def main():
    parser = argparse.ArgumentParser(description='Process datashop file.')
    parser.add_argument('-ft', choices=["student_step", "transaction"],
                        help='the type of file to load '
                        '(default="student_step")', default="student_step")
    parser.add_argument('-m', choices=["AFM", "AFM+S"],
                        help='the type of model to plot',
                        default="AFM+S")
    parser.add_argument('student_data', type=argparse.FileType('r'),
                        help="the student data file in datashop format")
    args = parser.parse_args()

    if args.ft == "transaction":
        ssr_file = transaction_to_student_step(args.student_data)
        ssr_file = open(ssr_file, 'r')
    else:
        ssr_file = args.student_data

    afm_predict_next_step(ssr_file, args.m)


if __name__ == "__main__":
    main()
