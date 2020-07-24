from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
import argparse

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer

from pyafm.custom_logistic import CustomLogistic
from pyafm.bounded_logistic import BoundedLogistic

def read_datashop_student_step(step_file, kc_model):
    headers = step_file.readline().rstrip().split('\t')
    header = {v: i for i,v in enumerate(headers)}

    model = "KC (%s)" % kc_model
    opp = "Opportunity (%s)" % kc_model

    original_headers = [h for h in headers 
                        if (("Predicted Error Rate" not in h) and 
                            (h == model or "KC (" not in h) and
                            (h == opp or "Opportunity (" not in h))]
    cols_to_keep = set([header[h] for h in original_headers])

    kcs = []
    opps = []
    y = []
    stu = []
    student_label = []
    item_label = []
    original_step_data = []

    for line in step_file:
        data = line.rstrip().split('\t')
        original_data = [d for i,d in enumerate(data) if i in cols_to_keep]
        original_step_data.append(original_data)

        kc_labels = [kc for kc in data[header[model]].split("~~") if kc != ""]

        if not kc_labels:
            continue

        kcs.append({kc: 1 for kc in kc_labels})

        kc_opps = [o for o in data[header[opp]].split("~~") if o != ""]
        opps.append({kc: int(kc_opps[i])-1 for i,kc in enumerate(kc_labels)})

        if data[header['First Attempt']] == "correct":
            y.append(1)
        else:
            y.append(0)

        student = data[header['Anon Student Id']]
        stu.append({student: 1})
        student_label.append(student)

        item = data[header['Problem Name']] + "##" + data[header['Step Name']]
        item_label.append(item)
    return (kcs, opps, y, stu, student_label, item_label, original_headers, original_step_data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process datashop file.')
    parser.add_argument('-m', choices=["AFM", "AFM+S"], 
                       help='the model to use (default="AFM+S")',
                        default="AFM+S")

    parser.add_argument('student_data', type=argparse.FileType('r'),
                        help="the datashop file in student step format")
    parser.add_argument('kc_model', type=str, 
                       help='the KC model that you would like to use; e.g., "Item"')
    parser.add_argument('output_data', type=argparse.FileType('w'),
                        help="the location of the output student step file")
    args = parser.parse_args()

    ssr_file = args.student_data

    kcs, opps, y, stu, student_label, item_label, original_headers, original_step_data = read_datashop_student_step(ssr_file, args.kc_model)

    sv = DictVectorizer()
    qv = DictVectorizer()
    ov = DictVectorizer()
    S = sv.fit_transform(stu)
    Q = qv.fit_transform(kcs)
    O = ov.fit_transform(opps)

    # AFM
    X = hstack((S, Q, O))
    y = np.array(y)
    l2 = [1.0 for i in range(S.shape[1])] 
    l2 += [0.0 for i in range(Q.shape[1])] 
    l2 += [0.0 for i in range(O.shape[1])]

    bounds = [(None, None) for i in range(S.shape[1])] 
    bounds += [(None, None) for i in range(Q.shape[1])] 
    bounds += [(0, None) for i in range(O.shape[1])]
    
    X = X.toarray()
    X2 = Q.toarray()

    if args.m == "AFM":
        m = CustomLogistic(bounds=bounds, l2=l2, fit_intercept=False)
        m.fit(X, y)
        yHat = m.predict_proba(X)
    elif args.m == "AFM+S":
        m = BoundedLogistic(first_bounds=bounds, first_l2=l2)
        m.fit(X, X2, y)
        yHat = m.predict_proba(X, X2)
    else:
        raise ValueError("Model type not supported")
        
    headers = original_headers + ["Predicted Error Rate (%s)" % args.kc_model]
    args.output_data.write("\t".join(headers) + "\n")
    for i, row in enumerate(original_step_data):
        d = row + ["%0.4f" % yHat[i]]
        args.output_data.write("\t".join(d) + "\n")
