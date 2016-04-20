import argparse

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt

from custom_logistic import CustomLogistic
from bounded_logistic import BoundedLogistic
from process_datashop import read_datashop_student_step

def avg_y_by_x(x,y):
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process datashop file.')
    parser.add_argument('student_step_file', type=argparse.FileType('r'),
                        help="the student step export from datashop")
    args = parser.parse_args()

    f = args.student_step_file

    kcs, opps, y, stu, student_label, item_label = read_datashop_student_step(args.student_step_file)

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

    afms = BoundedLogistic(first_bounds=bounds, first_l2=l2)
    afms.fit(X, X2, y)
    yAFMS = afms.predict_proba(X, X2)

    #plotkcs = ['All Knowledge Components']
    plotkcs = list(set([kc for row in kcs for kc in row])) + ['All Knowledge Components']

    #f, subplots = plt.subplots(len(plotkcs))
    for plot_id, plotkc in enumerate(plotkcs):

        plt.figure(plot_id+1)

        #if len(plotkcs) > 1:
        #    p = subplots[plot_id]
        #else:
        #    p = subplots
        xs = []
        y1 = []
        y2 = []
        y3 = []
        for i in range(len(y)):
            for kc in opps[i]:
                if not (kc == plotkc or plotkc == 'All Knowledge Components'):
                    continue
                xs.append(opps[i][kc])
                y1.append(y[i])
                y2.append(yAFM[i])
                y3.append(yAFMS[i])

        x, y1 = avg_y_by_x(xs, y1)
        x, y2 = avg_y_by_x(xs, y2)
        x, y3 = avg_y_by_x(xs, y3)

        y1 = [1-v for v in y1]
        y2 = [1-v for v in y2]
        y3 = [1-v for v in y3]

        human_line, = plt.plot(x, y1, color='red', label="Actual Data")
        afm_line, = plt.plot(x, y2, color='blue', label="AFM")
        afms_line, = plt.plot(x, y3, color='green', label="AFM+S")
        plt.legend(handles=[human_line, afm_line, afms_line])
        plt.title(plotkc)
        plt.xlabel("Opportunities")
        plt.ylabel("Error")
        plt.ylim(0,1)
        #plt.show()

        #p.plot(x, y1)
        #p.plot(x, y2)
        #p.plot(x, y3)
        #p.set_title(plotkc)
        #p.set_xlabel("Opportunities")
        #p.set_ylabel("Error")
        #p.set_ylim(0,1)

    plt.show()

        


