from collections import Counter
from collections import defaultdict
import csv
import pickle

import numpy as np
from tqdm import tqdm
import pandas as pd
 
def flatten(ll):
    return [item for sublist in ll for item in sublist]
    
def is_match(code, code_set):
    for i in range(1, len(code)+1):
        if code[:i] in code_set:
            return True
    return False
    
def overlap(code_set_1, code_set_2):
    for c in code_set_1:
        if is_match(c, code_set_2):
            return True
    for c in code_set_2:
        if is_match(c, code_set_1):
            return True
    return False
            

problem_snomed, problem_icd10, problem_icd9 = defaultdict(list), defaultdict(list), defaultdict(list)
print("loading problem definitions")
with open('data/problem_codes_all.csv') as f:
    r = csv.reader(f)
    header = next(r)
    for row in r:
        if row[1] == 'icd10':
            problem_icd10[row[0]].append(row[2])
        if row[1] == 'icd9':
            problem_icd9[row[0]].append(row[2])
        if row[1] == 'snomed':
            problem_snomed[row[0]].append(row[2])

df = pd.read_csv('data/test_probs.csv')
adf = pd.read_csv('data/all.csv', names=['problem', 'relationType', 'target', 'label'])
meds = sorted([med.split('_')[1] for med in adf[adf['relationType'] == 'medication']['target'].unique()])
print(f"num all meds: {len(set(meds))}")
print(f"num rxnorm meds: {len(set([med for med in meds if 'XXXXX' not in med]))}")

med_may_treat = {'snomed': defaultdict(list), 'icd9': defaultdict(list), 'icd10': defaultdict(list)}
with open('data/med_may_treat.csv') as f:
    r = csv.reader(f)
    for row in r:
        med, dx, dx_type = row
        med_may_treat[dx_type][med].append(dx)

print("processing NEGATIVES")
num_negs = Counter()
num_pos_negs = Counter()
for ix, triple in enumerate(df.itertuples()):
    if triple.label == 1:
        if triple.relationType != 'medication':
            continue
        target = triple.target.split('_')[1]
        m_sno = flatten(med_may_treat['snomed'][target])
        m_icd9 = flatten(med_may_treat['icd9'][target])
        m_icd10 = flatten(med_may_treat['icd10'][target])
        p_sno = problem_snomed[triple.problem]
        p_icd9 = problem_icd9[triple.problem]
        p_icd10 = problem_icd10[triple.problem]
        num_negs[(triple.problem, triple.relationType)] += 1
        if len(set(m_sno).intersection(set(p_sno))) > 0:
            num_pos_negs[(triple.problem, triple.relationType)] += 1
        elif overlap(m_icd10, p_icd10):
            num_pos_negs[(triple.problem, triple.relationType)] += 1
        elif overlap(m_icd9, p_icd9):
            num_pos_negs[(triple.problem, triple.relationType)] += 1
            
print("\n\nprocessing POSITIVES")
ranks = []
matches = 0

def rx_prob_match(target, problem):
    p_sno = problem_snomed[problem]
    p_icd9 = problem_icd9[problem]
    p_icd10 = problem_icd10[problem]
    m_sno = med_may_treat['snomed'][target]
    m_icd9 = med_may_treat['icd9'][target]
    m_icd10 = med_may_treat['icd10'][target]
    if len(set(m_sno).intersection(set(p_sno))) > 0:
        return 'snomed'
    elif overlap(m_icd10, p_icd10):
        return 'icd10'
    elif overlap(m_icd9, p_icd9):
        return 'icd9'
    return None
    

for triple in df.itertuples():
    if triple.label != 2 or triple.relationType != 'medication':
        continue
    target = triple.target.split('_')[1]
    nn = num_negs[(triple.problem, triple.relationType)]
    npn = num_pos_negs[(triple.problem, triple.relationType)]
    
    route = rx_prob_match(target, triple.problem)
    if route != None:
        rank = npn / 2 + 1
        matches += 1
    else:
        # no match
        rank = np.median(np.arange(npn+1, nn+1+1))
    probs = [prob for prob in problem_icd10 if rx_prob_match(target, prob) is not None]
    ranks.append(rank)
    
ranks = np.array(ranks)
mr = np.mean(ranks)
mrr = np.mean(1./ranks)
hits_at_1 = np.mean([rank <= 1 for rank in ranks])
hits_at_5 = np.mean([rank <= 5 for rank in ranks])
hits_at_10 = np.mean([rank <= 10 for rank in ranks])
hits_at_30 = np.mean([rank <= 30 for rank in ranks])

print(f"MR: {mr}, MRR: {mrr}, H@1: {hits_at_1}, H@5: {hits_at_5}, H@10: {hits_at_10}, H@30: {hits_at_30}")
