from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
import argparse

from tabulate import tabulate

from pyafm.roll_up import transaction_to_student_step
from pyafm.models import afm
from pyafm.models import afms


def read_datashop_student_step(step_file, model_id=None):
    header = {v: i for i, v in enumerate(
        step_file.readline().rstrip().split('\t'))}

    kc_mods = [v[4:-1] for v in header if v[0:2] == "KC"]
    kc_mods.sort()

    if model_id is None:
        print()
        print('Found these KC models:')
        for i, val in enumerate(kc_mods):
            print("  (%i) %s" % (i+1, val))
        print()
        model_id = int(input("Enter the number of which one you want to use: "))-1
    model = "KC (%s)" % (kc_mods[model_id])
    opp = "Opportunity (%s)" % (kc_mods[model_id])

    kcs = []
    opps = []
    y = []
    stu = []
    student_label = []
    item_label = []

    for line in step_file:
        data = line.rstrip().split('\t')

        kc_labels = [kc for kc in data[header[model]].split("~~") if kc != ""]

        if not kc_labels:
            continue

        kcs.append({kc: 1 for kc in kc_labels})

        kc_opps = [o for o in data[header[opp]].split("~~") if o != ""]
        opps.append({kc: int(kc_opps[i])-1 for i, kc in enumerate(kc_labels)})

        if data[header['First Attempt']] == "correct":
            y.append(1)
        else:
            y.append(0)

        student = data[header['Anon Student Id']]
        stu.append({student: 1})
        student_label.append(student)

        item = data[header['Problem Name']] + "##" + data[header['Step Name']]
        item_label.append(item)
    return (kcs, opps, y, stu, student_label, item_label)


def main():
    parser = argparse.ArgumentParser(description='Process datashop file.')
    parser.add_argument('-ft', choices=["student_step", "transaction"],
                        help='the type of file to load (default="student_step")',
                        default="student_step")
    parser.add_argument('student_data', type=argparse.FileType('r'),
                        help="the student data file in datashop format")
    parser.add_argument('-m', choices=["AFM", "AFM+S"],
                        help='the model to use (default="AFM+S")',
                        default="AFM+S")
    parser.add_argument('-nfolds', type=int, default=3,
                        help="the number of cross validation folds, when using cv (default=3).")
    parser.add_argument('-seed', type=int, default=None,
                        help='the seed used for shuffling in cross validation to ensure comparable'
                        'folds between runs (default=None).')
    parser.add_argument('-report', choices=['all', 'cv', 'kcs', 'kcs+stu'], default='all',
                        help='model values to report after fitting (default=all).')
    args = parser.parse_args()

    if args.ft == "transaction":
        ssr_file = transaction_to_student_step(args.student_data)
        ssr_file = open(ssr_file, 'r')
    else:
        ssr_file = args.student_data

    kcs, opps, y, stu, student_label, item_label = read_datashop_student_step(
        ssr_file)

    if args.m == "AFM":

        scores, kc_vals, coef_s = afm(kcs, opps, y, stu,
                                      student_label, item_label, args.nfolds, args.seed)
        print()
        if args.report in ['all', 'cv']:
            print(tabulate([scores], ['Unstratified CV', 'Stratified CV', 'Student CV', 'Item CV'],
                           floatfmt=".3f"))
            print()

        if args.report in ['all', 'kcs', 'kcs+stu']:
            print(tabulate(sorted(kc_vals), ['KC Name', 'Intercept (logit)',
                                             'Intercept (prob)', 'Slope'],
                           floatfmt=".3f"))
            print()

        if args.report in ['all', 'kcs+stu']:
            print(tabulate(sorted(coef_s), ['Anon Student Id', 'Intercept (logit)',
                                            'Intercept (prob)'],
                           floatfmt=".3f"))

    elif args.m == "AFM+S":

        scores, kc_vals, coef_s = afms(kcs, opps, y, stu,
                                       student_label, item_label, args.nfolds, args.seed)
        print()
        if args.report in ['all', 'cv']:
            print(tabulate([scores], ['Unstratified CV', 'Stratified CV', 'Student CV', 'Item CV'],
                           floatfmt=".3f"))
            print()

        if args.report in ['all', 'kcs', 'kcs+stu']:
            print(tabulate(sorted(kc_vals), ['KC Name', 'Intercept (logit)',
                                             'Intercept (prob)', 'Slope'],
                           floatfmt=".3f"))
            print()

        if args.report in ['all', 'kcs+stu']:
            print(tabulate(sorted(coef_s), ['Anon Student Id', 'Intercept (logit)',
                                            'Intercept (prob)'],
                           floatfmt=".3f"))

    else:
        raise ValueError("Model type not supported")


if __name__ == "__main__":
    main()
