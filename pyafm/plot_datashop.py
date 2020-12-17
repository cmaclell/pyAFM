from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
import argparse

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
from scipy.stats import beta

from pyafm.custom_logistic import CustomLogistic
from pyafm.bounded_logistic import BoundedLogistic
from pyafm.process_datashop import read_datashop_student_step
from pyafm.roll_up import transaction_to_student_step


def avg_y_by_x(x, y):
    x = np.array(x)
    y = np.array(y)

    xs = sorted(list(set(x)))

    xv = []
    yv = []
    lcb = []
    ucb = []
    n_obs = []

    for v in xs:
        ys = [y[i] for i, e in enumerate(x) if e == v]
        if len(ys) > 0:
            xv.append(v)
            yv.append(sum(ys) / len(ys))
            n_obs.append(len(ys))

            unique, counts = np.unique(ys, return_counts=True)
            counts = dict(zip(unique, counts))

            if 0 not in counts:
                counts[0] = 0
            if 1 not in counts:
                counts[1] = 0

            ci = beta.interval(0.95, 0.5 + counts[0], 0.5 + counts[1])
            lcb.append(ci[0])
            ucb.append(ci[1])

    return xv, yv, lcb, ucb, n_obs

def main():
    parser = argparse.ArgumentParser(description='Process datashop file.')
    parser.add_argument('-ft', choices=["student_step", "transaction"],
                        help='the type of file to load (default="student_'
                        'step")', default="student_step")
    parser.add_argument('-m', choices=["AFM", "AFM+S", "both"],
                        help='the type of model to plot',
                        default="both")
    parser.add_argument('-p', choices=["overall", "individual_kcs", "both"],
                        help='the type of graph to generate',
                        default="overall")
    parser.add_argument('student_data', type=argparse.FileType('r'),
                        help="the student data file in datashop format")
    args = parser.parse_args()

    if args.ft == "transaction":
        ssr_file = transaction_to_student_step(args.student_data)
        ssr_file = open(ssr_file, 'r')
    else:
        ssr_file = args.student_data

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

    afms = BoundedLogistic(first_bounds=bounds, first_l2=l2)
    afms.fit(X, X2, y)
    yAFMS = afms.predict_proba(X, X2)

    plotkcs = []

    if args.p == "overall" or args.p == "both":
        plotkcs += ['All Knowledge Components']
    if args.p == "individual_kcs" or args.p == "both":
        plotkcs += list(set([kc for row in kcs for kc in row]))

    # f, subplots = plt.subplots(len(plotkcs))
    for plot_id, plotkc in enumerate(plotkcs):

        # plt.figure(plot_id+1)

        # if len(plotkcs) > 1:
        #    p = subplots[plot_id]
        # else:
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

        x, y1, lcb, ucb, n_obs = avg_y_by_x(xs, y1)
        x, y2, _, _, _ = avg_y_by_x(xs, y2)
        x, y3, _, _, _ = avg_y_by_x(xs, y3)

        y1 = [1-v for v in y1]
        y2 = [1-v for v in y2]
        y3 = [1-v for v in y3]

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        human_line, = axs[0].plot(x, y1, color='red', label="Actual Data")
        # human_line, = plt.plot(x, y1, color='red', label="Actual Data")
        axs[0].fill_between(x, lcb, ucb, color='red', alpha=.1)

        lines = [human_line]

        if args.m == "AFM" or args.m == "both":
            afm_line, = axs[0].plot(x, y2, color='blue', label="AFM")
            lines.append(afm_line)
        if args.m == "AFM+S" or args.m == "both":
            afms_line, = axs[0].plot(x, y3, color='green', label="AFM+S")
            lines.append(afms_line)
        axs[0].legend(handles=lines)
        axs[0].set_title(plotkc)
        axs[1].set_xlabel("Opportunities")
        axs[0].set_ylabel("Error")
        axs[0].set_ylim(0, 1)

        # dropout = [0] + [n_obs[i] - n_obs[i+1]for i in range(len(n_obs)-1)]
        axs[1].bar([i for i in range(len(n_obs))], n_obs)
        axs[1].set_ylabel("# of Obs.")
        # plt.show()

        # p.plot(x, y1)
        # p.plot(x, y2)
        # p.plot(x, y3)
        # p.set_title(plotkc)
        # p.set_xlabel("Opportunities")
        # p.set_ylabel("Error")
        # p.set_ylim(0,1)

    plt.show()


if __name__ == "__main__":
    main()
