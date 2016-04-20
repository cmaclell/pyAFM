import csv

def transaction_to_student_step(datashop_file):
    students = {}
    #header = {v: i for i,v in enumerate(datashop_file.readline().split('\t'))}
    out_file = datashop_file.name[:-4]+'-rollup.txt'
    
    header = None

    for row in csv.reader(datashop_file,delimiter='\t'):

        if header is None:
            header = row
            continue
        
        line = {}
        kc_mods = {}

        for i in range(len(header)):
            h = header[i]

            if h[:4] == 'KC (':
                line[h] = row[i]
                if h not in kc_mods:
                    kc_mods[h] = []
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
            raise Exception('No fields present to make step names, either add a "Step Name" column or "Selection" and "Action" columns.')

        if 'step name' in line and 'problem name' in line:
            line['prob step'] = line['problem name'] + ' ' + line['step name']

        for km in kc_mods:
            line[km] = '~~'.join(kc_mods[km])

        if line['anon student id'] not in students:
            students[line['anon student id']] = []

        students[line['anon student id']].append(line)

    touched = set()
    kc_model_names = [km for km in header if km[:4] == 'KC (' and not (km in touched or touched.add(km))]
    row_count = 0

    with open(out_file,'w') as out:

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
                    'Hints',]

        out.write('\t'.join(new_head))
        
        for km in kc_model_names:
            out.write('\t'+km+'\tOpportunity ('+km[4:])

        out.write('\n')

        stu_list = list(students.keys())
        sorted(stu_list)

        for stu in stu_list:
            transactions = students[stu]
            transactions = sorted(transactions, key=lambda k: k['time'])

            problem_views = {}
            kc_ops = {}
            atmp_counts = {}
            last_prob = None
            last_step = None
            step_stats = {}
            pre_t = None
            cur_t = None
            nex_t = None


            for i, t in enumerate(transactions):

                problem_name = t['problem name']
                student = t['anon student id']
                step_name = t['step name']

                if t['problem name'] != last_prob:
                    if t['problem name'] not in problem_views:
                        problem_views[t['problem name']] = 0
                    problem_views[t['problem name']] += 1
                    last_step = None
                
                if step_name != last_step:
                    step_stats = {}
                    step_stats['step_start_time'] = t['time']
                    step_stats['step_end_time'] = t['time']
                    step_stats['first_transaction_time'] = t['time']
                    step_stats['first_attempt'] = t['outcome'].lower()

                    step_stats['correct_transaction_time'] = None
                    step_stats['corrects'] = 0
                    step_stats['incorrects'] = 0
                    step_stats['hints'] = 0
                    
                    if t['outcome'].lower() == 'correct':
                        step_stats['correct_transaction_time'] = t['time']
                        step_stats['corrects'] = 1
                    elif t['outcome'].lower() == 'incorrect':
                        step_stats['incorrects'] = 1
                    elif t['outcome'].lower() == 'hint':
                        step_stats['hints'] = 1
                    else:
                        raise Exception('Unkown outcome type: ',t['outcome'])
                    t['step end time'] = None
                    trans_time = t['time']

                kc_to_write = []    

                if step_name == last_step and problem_name == last_prob:
                    step_stats['step_end_time'] = t['time']
                    if t['outcome'].lower() == 'correct':
                        step_stats['corrects'] += 1
                        if step_stats['correct_transaction_time'] is None:
                            step_stats['correct_transaction_time'] = t['time']
                    elif t['outcome'].lower() == 'incorrect':
                        step_stats['incorrects'] += 1
                    elif t['outcome'].lower() == 'hint':
                        step_stats['hints'] += 1

              #  if step_name != last_step:
                for kc_mod in kc_model_names:
                    value = t[kc_mod]
                    mod_name = kc_mod[4:-1]
                    if mod_name not in kc_ops:
                        kc_ops[mod_name] = {}
                    if '~~' in value:
                        kc_to_write.append(value)
                        kcs = value.split('~~')
                        ops_to_write = []
                        for kc in kcs:
                            if kc not in kc_ops[mod_name]:
                                kc_ops[mod_name][kc] = 0
                            kc_ops[mod_name][kc] += 1
                            ops_to_write.append(kc_ops[mod_name][kc])
                        kc_to_write.append('~~'.join(ops_to_write))
                    else:
                        if value not in kc_ops[mod_name]:
                            kc_ops[mod_name][value] = 0
                        kc_ops[mod_name][value] += 1
                        kc_to_write.append(value)
                        kc_to_write.append(str(kc_ops[mod_name][value]))

                last_step = step_name
                last_prob = problem_name

                if len(kc_to_write) > 0 and (i == len(transactions)-1 or t['prob step'] != transactions[i+1]['prob step']):

                    row_count += 1
                    line_to_write = [str(row_count),
                                    student,
                                    problem_name,
                                    str(problem_views[problem_name]),
                                    step_name,
                                    step_stats['step_start_time'],
                                    step_stats['first_transaction_time'],
                                    str(step_stats['correct_transaction_time']),
                                    step_stats['step_end_time'],
                                    step_stats['first_attempt'],
                                    str(step_stats['incorrects']),
                                    str(step_stats['corrects']),
                                    str(step_stats['hints'])]
                    line_to_write.extend(kc_to_write)
                    out.write('\t'.join(line_to_write)+'\n')

    return out_file
