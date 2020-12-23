from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
import csv
from dateutil.parser import parse


def write_problem(steps, problem_views, kc_ops, row_count, kc_model_names,
                  out):

    # variable to store rolled up steps
    rollup = []

    for s in steps:
        # sort transactions within a step by time (should be sorted already,
        # but just in case)
        steps[s].sort(key=lambda x: x['time'])

        # update variables for first attempt
        student = steps[s][0]['anon student id']
        problem_name = steps[s][0]['problem name']
        step_name = s
        step_start_time = steps[s][0]['time']
        first_transaction_time = steps[s][0]['time']
        correct_transaction_time = ""
        step_end_time = steps[s][0]['time']
        first_attempt = steps[s][0]['outcome'].lower()
        incorrects = 0
        corrects = 0
        hints = 0
        kc_sets = {kc_mod: set() for kc_mod in kc_model_names}

        # update variables for non-first attempt transactions
        for t in steps[s]:
            step_end_time = t['time']
            if t['outcome'].lower() == 'correct':
                correct_transaction_time = t['time']
                corrects += 1
            elif t['outcome'].lower() == 'incorrect':
                incorrects += 1
            elif t['outcome'].lower() == 'hint':
                hints += 1

            for kc_mod in kc_model_names:
                for kc in t[kc_mod].split("~~"):
                    kc_sets[kc_mod].add(kc)

        # for each rolled up step, we need to increment the KC counts.
        kc_to_write = []
        for kc_mod in kc_model_names:
            model_name = kc_mod[4:-1]
            kcs = list(kc_sets[kc_mod])
            kc_to_write.append("~~".join(kcs))

            if model_name not in kc_ops:
                kc_ops[model_name] = {}

            ops = []
            for kc in kcs:
                if kc not in kc_ops[model_name]:
                    kc_ops[model_name][kc] = 0
                kc_ops[model_name][kc] += 1
                ops.append(str(kc_ops[model_name][kc]))
            kc_to_write.append("~~".join(ops))

        # add rolled up step to rollup
        rolled_up_step = [str(row_count),
                          student,
                          problem_name,
                          str(problem_views),
                          step_name,
                          step_start_time,
                          first_transaction_time,
                          correct_transaction_time,
                          step_end_time,
                          first_attempt,
                          str(incorrects),
                          str(corrects),
                          str(hints)]
        rolled_up_step.extend(kc_to_write)

        row_count += 1
        rollup.append(rolled_up_step)

    # sort the rolled up steps by step start time
    rollup.sort(key=lambda x: x[5])

    for line_to_write in rollup:
        out.write('\t'.join(line_to_write)+'\n')

    return row_count


def transaction_to_student_step(datashop_file):
    out_file = datashop_file.name[:-4]+'-rollup.txt'
    students = {}
    header = None

    for row in csv.reader(datashop_file, delimiter='\t'):
        if header is None:
            header = row
            continue

        line = {}
        kc_mods = {}

        for i, h in enumerate(header):
            if h[:4] == 'KC (':
                line[h] = row[i]
                if h not in kc_mods:
                    kc_mods[h] = []
                if line[h] != "":
                    kc_mods[h].append(line[h])
                continue
            else:
                h = h.lower()
                line[h] = row[i]

        if 'step name' in line:
            pass
        elif 'selection' in line and 'action' in line:
            line['step name'] = line['selection'] + ' ' + line['action']
        else:
            raise Exception(
                'No fields present to make step names, either add a "Step'
                ' Name" column or "Selection" and "Action" columns.')

        if 'step name' in line and 'problem name' in line:
            line['prob step'] = line['problem name'] + ' ' + line['step name']

        for km in kc_mods:
            line[km] = '~~'.join(kc_mods[km])

        if line['anon student id'] not in students:
            students[line['anon student id']] = []
        students[line['anon student id']].append(line)

    kc_model_names = list(set(kc_mods))
    row_count = 0

    with open(out_file, 'w') as out:

        new_head = ['Row',
                    'Anon Student Id',
                    'Problem Name',
                    'Problem View',
                    'Step Name',
                    'Step Start Time',
                    'First Transaction Time',
                    'Correct Transaction Time',
                    'Step End Time',
                    'First Attempt',
                    'Incorrects',
                    'Corrects',
                    'Hints', ]

        out.write('\t'.join(new_head))

        for km in kc_model_names:
            out.write('\t'+km+'\tOpportunity ('+km[4:])

        out.write('\n')

        stu_list = list(students.keys())
        sorted(stu_list)

        for stu in stu_list:
            transactions = students[stu]
            transactions = sorted(transactions, key=lambda k: parse(k['time']))
            problem_views = {}
            kc_ops = {}
            row_count = 0
            steps = {}
            problem_name = ""

            # Start iterating through the stuff.
            for i, t in enumerate(transactions):
                if problem_name != t['problem name']:

                    # we don't need to write the first row, because we don't
                    # have anything yet.
                    if i != 0:

                        if problem_name not in problem_views:
                            problem_views[problem_name] = 0
                        problem_views[problem_name] += 1

                        row_count = write_problem(steps,
                                                  problem_views[problem_name],
                                                  kc_ops, row_count,
                                                  kc_model_names, out)
                        steps = {}

                if t['step name'] not in steps:
                    steps[t['step name']] = []
                steps[t['step name']].append(t)

                problem_name = t['problem name']

            # need to write the last problem
            if problem_name not in problem_views:
                problem_views[problem_name] = 0
            problem_views[problem_name] += 1

            row_count = write_problem(steps, problem_views[problem_name],
                                      kc_ops, row_count, kc_model_names, out)
            steps = {}

    print('transaction file rolled up into:', out_file)
    return out_file
