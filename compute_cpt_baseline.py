from collections import Counter
from collections import defaultdict
import csv
import pickle
import re
import typing as t
import requests as rq
from umls_api_auth import Authentication

import numpy as np
from tqdm import tqdm
import pandas as pd

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


CPT_TO_DISCIPLINE = {
        # surgeries
        "1010115": "Audiology", #        "Surgical Procedures on the Auditory System"
        "1006056": "Circulatory", #        "Surgical Procedures on the Cardiovascular System"
        "1006964": "Gastroenterology", #        "Surgical Procedures on the Digestive System"
        "1009024": "Endocrinology", #        "Surgical Procedures on the Endocrine System"
        "1009727": "Ophthalmology", #        "Surgical Procedures on the Eye and Ocular Adnexa"
        "1008681": "Genitourinary", #        "Surgical Procedures on the Female Genital System"
        "1006843": "Hematology", #         "Surgical Procedures on the Hemic and Lymphatic Systems"
        "1003148": "Dermatology", #        "Surgical Procedures on the Integumentary System"
        "1008470": "Genitourinary", #        "Surgical Procedures on the Male Genital System"
        "1006933": "Respiratory", #         "Surgical Procedures on the Mediastinum and Diaphragm"
        "1003679": "Musculoskeletal", #         "Surgical Procedures on the Musculoskeletal System"
        "1009068": "Neurology", #        "Surgical Procedures on the Nervous System"
        "1005690": "Respiratory", #         "Surgical Procedures on the Respiratory System"
        "1008061": "Genitourinary", #         "Surgical Procedures on the Urinary System"
        # radiology procedures
        "1015091": "Musculoskeletal", #        "Bone/Joint Studies"
        "1010334": "Respiratory", #        Diagnostic Radiology (Diagnostic Imaging) Procedures of the Chest
        "1010537": "Gastroenterology", # Diagnostic Radiology (Diagnostic Imaging) Procedures of the Gastrointestinal Tract
        "1010594": "Circulatory", # Diagnostic Radiology (Diagnostic Imaging) Procedures of the Heart
        "1010602": "Circulatory", # Diagnostic Radiology (Diagnostic Imaging) Procedures of the Vascular System
        "1010574": "Genitourinary", # Diagnostic Radiology (Diagnostic Imaging) Procedures of the Urinary Tract
        "1015090": "Genitourinary", # Breast, Mammography
        "1010843": "Oncology", # Radiation Oncology Treatment
        # medicine services and procedures
        "1013263": "Immunology", # Allergy and Clinical Immunology Procedures
        "1012974": "Circulatory", # Cardiovascular Procedures
        "1013417": "Mental Health", # Central Nervous System Assessments/Tests (eg, Neuro-Cognitive, Mental Status, Speech Testing)
        "1013565": "Musculoskeletal", # Chiropractic Manipulative Treatment Procedures
        "1012740": "Genitourinary", # Dialysis Services and Procedures
        "1013306": "Endocrinology", # Endocrinology Services
        "1012764": "Gastroenterology", # Gastroenterology Procedures
        "1013309": "Neurology", # Neurology and Neuromuscular Procedures
        "1012793": "Ophthalmology", # Ophthalmology Services and Procedures
        "1013483": "Musculoskeletal", # Physical Medicine and Rehabilitation Evaluations
        "1012681": "Mental Health", # Psychiatry Services and Procedures
        "1013214": "Respiratory", # Pulmonary Procedures
        "1013471": "Dermatology", # Special Dermatological Procedures
        }

class ICDPrefix(t.NamedTuple):
    lowLetter: str
    lowNumber: int
    highLetter: str
    highNumber: int

ICD_10_PREF_TO_DISCIPLINE = {
        ICDPrefix("A", 0, "B", 99): "Infectious Disease", 
        ICDPrefix("C", 0, "D", 49): "Oncology", 
        ICDPrefix("D", 0, "D", 89): "Hematology", 
        ICDPrefix("E", 0, "E", 89): "Endocrinology", 
        ICDPrefix("F", 1, "F", 99): "Mental Health", 
        ICDPrefix("G", 0, "G", 99): "Neurology", 
        ICDPrefix("H", 0, "H", 59): "Ophthalmology", 
        ICDPrefix("H", 60, "H", 95): "Otology", 
        ICDPrefix("I", 00, "I", 99): "Circulatory", 
        ICDPrefix("J", 0, "J", 99): "Respiratory", 
        ICDPrefix("K", 0, "K", 14): "Dentistry", 
        ICDPrefix("K", 20, "K", 95): "Gastroenterology", 
        ICDPrefix("L", 0, "L", 99): "Dermatology", 
        ICDPrefix("M", 0, "M", 99): "Musculoskeletal", 
        ICDPrefix("N", 0, "N", 99): "Genitourinary", 
        ICDPrefix("R", 0, "R", 9): "Circulatory", 
        ICDPrefix("R", 10, "R", 19): "Gastroenterology", 
        ICDPrefix("R", 20, "R", 23): "Dermatology", 
        ICDPrefix("R", 25, "R", 29): "Musculoskeletal", 
        ICDPrefix("R", 30, "R", 39): "Genitourinary", 
        ICDPrefix("R", 40, "R", 46): "Mental Health", 
        ICDPrefix("S", 0, "T", 98): "External", 
        ICDPrefix("Z", 0, "Z", 29): "Health Maintenance", 
        }

ICD_9_PREF_TO_DISCIPLINE = {
        ICDPrefix("", 1, "", 139):   "Infectious Disease",
        ICDPrefix("", 140, "", 239): "Oncology",
        ICDPrefix("", 240, "", 279): "Endocrinology",
        ICDPrefix("", 280, "", 289): "Hematology",
        ICDPrefix("", 290, "", 319): "MentalHealth",
        ICDPrefix("", 320, "", 389): "Neurology",
        ICDPrefix("", 390, "", 459): "Circulatory",
        ICDPrefix("", 460, "", 519): "Respiratory",
        ICDPrefix("", 520, "", 529): "Dentistry",
        ICDPrefix("", 530, "", 579): "Gastroenterology",
        ICDPrefix("", 580, "", 629): "Genitourinary",
        ICDPrefix("", 630, "", 679): "Other",
        ICDPrefix("", 680, "", 709): "Dermatology",
        ICDPrefix("", 710, "", 739): "Musculoskeletal",
        ICDPrefix("", 740, "", 759): "Other",
        ICDPrefix("", 760, "", 779): "Other",
        ICDPrefix("", 780, "", 799): "Other",
        ICDPrefix("", 800, "", 999): "External",
        ICDPrefix("V", 1, "V", 91):  "Health Maintenance",
        ICDPrefix("E", 0, "E", 999): "External",
        }

def cpt_to_discipline(px_code, tgt):
    # get discipline for a CPT code using our manually created map
    disciplines = []
    # take part of code before modifier
    code = px_code.split('.')[0]
    route = f'/content/current/source/CPT/{code}/ancestors'
    query = {'ticket': auth.getst(tgt)}
    anc = rq.get(URI+route, params=query)
    if anc.status_code == 200:
        disciplines.extend([CPT_TO_DISCIPLINE[a['ui']] for a in anc.json()['result'] if a['ui'] in CPT_TO_DISCIPLINE])
    return set(disciplines)

def icd9_to_disc(dx_code):
    prefix = dx_code[:3]
    if dx_code[0].isnumeric():
        prefix = int(prefix)
        for pref, disc in ICD_9_PREF_TO_DISCIPLINE.items():
            if prefix_number >= pref.lowNumber and prefix_number <= pref.highNumber:
                return disc
    else:
        prefix_letter = dx_code[0]
        # just manually do this
        if prefix_letter == "V":
            return "Health Maintenance"
        elif prefix_letter == "E":
            return "External"
    return

def icd10_to_disc(dx_code):
    prefix = dx_code[:3]
    prefix_letter = prefix[0]
    prefix_number = int(prefix[1:])
    for pref, disc in ICD_10_PREF_TO_DISCIPLINE.items():
        if prefix_letter >= pref.lowLetter and prefix_letter <= pref.highLetter and prefix_number >= pref.lowNumber and prefix_number <= pref.highNumber:
            return disc
    return

def dxs_to_discipline(dx_codes, dx_systems):
    # get discipline for ICD code using manually created maps
    disciplines = []
    for code, system in zip(dx_codes, dx_systems):
        if system == "ICD-9-CM":
            icd9_disc = icd9_to_disc(code)
            disciplines.append(icd9_disc)
        elif system == "ICD-10-CM":
            icd10_disc = icd10_to_disc(code)
            disciplines.append(icd10_disc)
    return set(disciplines)

def px_to_prob_match_by_discipline(px, problem):
    tgt = auth.gettgt()
    px_discs = cpt_to_discipline(px, tgt)
    dx_discs = dxs_to_discipline(problem_icd10[problem], ["ICD-10-CM"] * len(problem_icd10[problem]))
    return len(px_discs.intersection(dx_discs)) > 0

def is_cpt(code):
    return re.match("^[A-Z]\d{4}(\.\d+)?", code) is not None or re.match("^\d{4}[\dFMTU](\.\d+)?$", code) is not None

auth = Authentication('MY-SECRET-KEY')
tgt = auth.gettgt()

URI = "https://uts-ws.nlm.nih.gov/rest"

df = pd.read_csv('data/test_probs.csv')
adf = pd.read_csv('data/all.csv', names=['problem', 'relationType', 'target', 'label'])
procs = sorted([proc.split('_')[1] for proc in adf[adf['relationType'] == 'procedure']['target'].unique()])
print(f"num all procs: {len(set(procs))}")
cpt_procs = [p for p in procs if is_cpt(p)]
print(f"num cpt procs: {len(set(cpt_procs))}")

num_negs = Counter()
num_pos_negs = Counter()
for ix, triple in enumerate(df.itertuples()):
    if triple.label == 1:
        if triple.relationType != 'procedure':
            continue
        target = triple.target.split('_')[1]
        num_negs[(triple.problem, triple.relationType)] += 1
        if is_cpt(target):
            if px_to_prob_match_by_discipline(target, triple.problem):
                num_pos_negs[(triple.problem, triple.relationType)] += 1
            
ranks = []
matches = 0
for triple in df.itertuples():
    if triple.label != 2 or triple.relationType != 'procedure':
        continue
    target = triple.target.split('_')[1]
    nn = num_negs[(triple.problem, triple.relationType)]
    npn = num_pos_negs[(triple.problem, triple.relationType)]
    if is_cpt(target):
        if px_to_prob_match_by_discipline(target, triple.problem):
            rank = npn / 2 + 1
        else:
            rank = np.median(np.arange(npn+1, nn+1+1))
    else:
        rank = np.median(np.arange(npn+1, nn+1+1))
    ranks.append(rank)
    
ranks = np.array(ranks)
mr = np.mean(ranks)
mrr = np.mean(1./ranks)
hits_at_1 = np.mean([rank <= 1 for rank in ranks])
hits_at_5 = np.mean([rank <= 5 for rank in ranks])
hits_at_10 = np.mean([rank <= 10 for rank in ranks])
hits_at_30 = np.mean([rank <= 30 for rank in ranks])

print(f"MR: {mr}, MRR: {mrr}, H@1: {hits_at_1}, H@5: {hits_at_5}, H@10: {hits_at_10}, H@30: {hits_at_30}")
