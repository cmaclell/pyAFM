import sys
import argparse

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import LabelKFold


from custom_logistic import CustomLogistic
from bounded_logistic import BoundedLogistic

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process datashop file.')
    parser.add_argument('student_step_file', type=argparse.FileType('r'),
                        help="the student step export from datashop")
    #parser.add_argument('-m', choices=["AFM", "AFM+S", "PFA", "PFA+S"], 
    #                   help='the model to use (default="AFM+S")',
    #                    default="AFM+S")
    #parser.add_argument('-score', choices=['unstratifiedCV', 'stratifiedCV',
    #                                       'itemCV', 'studentCV'], 
    #                    default="stratifiedCV",
    #                   help='the type of RMSE cross validation to use (default="stratifiedCV").')
    #parser.add_argument('-nfolds', type=int, default=3, 
    #                    help="the number of cross validation folds, when using cv (default=3).")
    #parser.add_argument('-nrepeats', type=int, default=5, 
    #                    help="the number of times to repeat cross validation, when using cv (default=5).")
    #parser.add_argument('-cv_o', type=argparse.FileType('w'), 
    #                   help="file to output cv results")
    #parser.add_argument('-pred_o', type=argparse.FileType('w'), 
    #                   help="file to output predictions")
    #parser.add_argument('-param_o', type=argparse.FileType('w'), 
    #                   help="file to output estimated parameters")

    args = parser.parse_args()

    #if not(args.cv_o or args.pred_o or args.param_o):
    #    raise Exception("At least one output needs to be provided (cv_o, pred_o or param_o)")

    #fname = sys.argv[1]
    f = args.student_step_file
    header = {v: i for i,v in enumerate(f.readline().split('\t'))}
    #print(header)

    kcs = [v[4:-1] for v in header if v[0:2] == "KC"]
    kcs.sort()

    for i,v in enumerate(kcs):
        print("(%i) %s" % (i+1, v))
    modelId = int(input("Which KC model? "))-1
    model = "KC (%s)" % (kcs[modelId])
    opp = "Opportunity (%s)" % (kcs[modelId])

    allStudents = set()
    allKCs = set()
    allProblems = set()

    row_ids = []
    kcs = []
    opps = []
    pos_opps = []
    neg_opps = []
    y = []
    stu = []
    student_label = []
    item_label = []

    counter = {}

    for line in f:
        data = line.split('\t')

        kc_labels = [kc for kc in data[header[model]].split("~~") if kc != ""]

        if not kc_labels:
            continue

        kcs.append({kc: 1 for kc in kc_labels})

        kc_opps = [o for o in data[header[opp]].split("~~") if o != ""]
        opps.append({kc+" opportunities": int(kc_opps[i]) for i,kc in enumerate(kc_labels)})

        if data[header['First Attempt']] == "correct":
            y.append(1)
        else:
            y.append(0)

        student = data[header['Anon Student Id']]
        stu.append({student: 1})
        student_label.append(student)

        pos = {}
        nos = {}

        if data[header['First Attempt']] == "correct":
            for kc in kc_labels:
                po_name = student + "-" + kc + "-pos-opp"
                pos_opp = counter.setdefault(po_name, 1)
                counter[po_name] = pos_opp + 1
                pos[po_name] = pos_opp
        else:
            for kc in kc_labels:
                no_name = student + "-" + kc + "-neg-opp"
                neg_opp = counter.setdefault(no_name, 1)
                counter[no_name] = neg_opp + 1
                nos[no_name] = neg_opp

        pos_opps.append(pos)
        neg_opps.append(nos)

        item = data[header['Problem Name']] + "##" + data[header['Step Name']]
        item_label.append(item)

        row_id = data[header['Row']]
        row_ids.append(row_id)

    v = DictVectorizer()
    S = v.fit_transform(stu)
    Q = v.fit_transform(kcs)
    O = v.fit_transform(opps)
    PO = v.fit_transform(pos_opps)
    NO = v.fit_transform(neg_opps)

    #print(list(np.add(np.sum(PO.toarray(), axis=1), np.sum(NO.toarray(),
    #                                                       axis=1))))
    #print(list(np.sum(O.toarray(), axis=1)))

    #print(PO.shape)
    #print(NO.shape)

    #print(Q)
    #print(O)

    ## AFM
    X = hstack((S, Q, O))
    y = np.array(y)
    l2 = [1.0 for i in range(S.shape[1])] 
    l2 += [0.0 for i in range(Q.shape[1])] 
    l2 += [0.0 for i in range(O.shape[1])]

    bounds = [(None, None) for i in range(S.shape[1])] 
    bounds += [(None, None) for i in range(Q.shape[1])] 
    bounds += [(0, None) for i in range(O.shape[1])]
    
    # PFA
    #X = hstack((Q, PO, NO))
    #y = np.array(y)
    #l2 = 0.01

    #bounds = [(None, None) for i in range(Q.shape[1])] 
    #bounds += [(0, None) for i in range(PO.shape[1])]
    #bounds += [(0, None) for i in range(NO.shape[1])]

    X = X.toarray()
    X2 = Q.toarray()

    cv = KFold(len(y), n_folds=3, shuffle=True)
    #cv = StratifiedKFold(y, n_folds=3, shuffle=True)
    #cv = LabelKFold(student_label, n_folds=10)
    ##cv = LabelKFold(item_label, n_folds=3)
    est = BoundedLogistic(first_bounds=bounds, first_l2=l2)
    est2 = CustomLogistic(bounds=bounds, l2=l2)
    scores = []
    scores2 = []
    for train_index, test_index in cv:
        X_train, X_test = X[train_index], X[test_index]
        X2_train, X2_test = X2[train_index], X2[test_index]
        y_train, y_test = y[train_index], y[test_index]
        est.fit(X_train, X2_train, y_train)
        est2.fit(X_train, y_train)
        scores.append(est.mean_squared_error(X_test, X2_test, y_test))
        scores2.append(est2.mean_squared_error(X_test, y_test))
    print('AFM+S RMSE CV:', np.mean(np.sqrt(scores)))
    print('AFM RMSE CV:', np.mean(np.sqrt(scores2)))

