import argparse
from collections import defaultdict
import csv
from io import BytesIO
import pickle, boto3
import random

from gensim.models import KeyedVectors, Word2Vec
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser()
parser.add_argument("embed_file", type=str, help="path to file containing pre-trained embeddings")
parser.add_argument("embed_file_type", type=str, choices=['w2v', 'bin'], help="specifies the file format")
parser.add_argument("embed_size", type=int, help="embedding size")
args = parser.parse_args()

print("loading dx code embeddings")
if args.embed_file_type == 'w2v':
    dx_embed = KeyedVectors.load_word2vec_format(args.embed_file)
elif args.embed_file_type == 'bin':
    dx_embed = Word2Vec.load(args.embed_file).wv

print("loading cui embeddings")
cui_embed = KeyedVectors.load_word2vec_format('embeddings/claims_cuis_hs_300.txt')
print("loading CPT, LOINC code embeddings")
code_embed = KeyedVectors.load_word2vec_format('embeddings/claims_codes_hs_300.txt')
rx_embed = cui_embed
px_embed = code_embed
lab_embed = code_embed

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

print("building vocab")
icd9_vocab = set([item for sublist in problem_icd9.values() for item in sublist])
dx_icd9_freq = {row[0]: float(row[1]) for row in csv.reader(open('data/site_icd9_relative_freqs.csv'))}

vocab = [line.strip() for line in open('vocab.txt')]
dvocab = {word:ix for ix,word in enumerate(vocab)}

df = pd.read_csv('data/all.csv', names=['problem', 'relationType', 'target', 'label'])
rxns = [v for v in vocab if v.startswith('RX_')]
procs = [v for v in vocab if v.startswith('PX_')]
loincs = [v for v in vocab if v.startswith('LAB_')]

        
rxn2cuis = defaultdict(set)
with open('data/rxn2cuis.txt') as f:
    for line in f:
        rxn, cui = line.strip().split(',')
        rxn2cuis[rxn].add(cui)
        
embeds = np.zeros((len(vocab), args.embed_size))

print("loading random init embeddings for missing codes")
missing_wv = KeyedVectors.load_word2vec_format('rand_init_missing.w2v')

def add_back_dot(icd9):
    if icd9.isnumeric() or icd9.startswith('V'):
        if len(icd9) > 3:
            return icd9[:3] + '.' + icd9[3:]
        else:
            return icd9
    elif icd9.startswith('E'):
        if len(icd9) > 4:
            return icd9[:4] + '.' + icd9[4:]
        else:
            return icd9

for prob in problem_icd10:
    # circumvent dumb data error
    if prob == 'kidney_stones':
        continue
    ix = dvocab[prob]
    avg_embed = np.zeros(args.embed_size)
    num = 0
    total = sum([dx_icd9_freq.get(f'{icd9}', 0) for icd9 in problem_icd9[prob]])
    for icd9 in problem_icd9[prob]:
        icd9_a = f'IDX_{add_back_dot(icd9)}'
        if icd9_a in dx_embed and icd9 in dx_icd9_freq:
            avg_embed += dx_embed[icd9_a] * dx_icd9_freq[icd9]
            num += 1
    if num > 0:
        avg_embed /= total
        embeds[ix,:] = avg_embed
 
num_ = 0
for rxn in rxns:
    ix = dvocab[rxn]
    avg_embed = np.zeros(args.embed_size)
    num = 0
    raw_code = rxn.split('_')[1]
    for cui in rxn2cuis[raw_code]:
        if cui in rx_embed:
            avg_embed += rx_embed[cui]
            num += 1
    if num > 0:
        avg_embed /= num
        avg_embed = normalize(avg_embed.reshape(1,-1))
        embeds[ix,:] = avg_embed
        num_ += 1
    else:
        embeds[ix,:] = missing_wv[rxn]
 
print(f"frac of rxn codes with embeddings: {num_}/{len(rxns)}")

num = 0
for loinc in loincs:
    ix = dvocab[loinc]
    raw_code = loinc.split('_')[1]
    if f'L_{raw_code}' in lab_embed:
        embeds[ix,:] = normalize(lab_embed[f'L_{raw_code}'].reshape(1,-1))
        num +=1 
    else:
        embeds[ix,:] = missing_wv[loinc]

print(f"frac of lab codes with embeddings: {num}/{len(loincs)}")

num = 0
for proc in procs:
    ix = dvocab[proc]
    raw_code = proc.split('_')[1]
    if f'C_{raw_code}' in px_embed:
        embeds[ix,:] = normalize(px_embed[f'C_{raw_code}'].reshape(1,-1))
        num += 1
    else:
        embeds[ix,:] = missing_wv[proc]

print(f"frac of CPT codes with embeddings: {num}/{len(loincs)}")

inv_vocab = {ix:item for item,ix in dvocab.items()}
with open(f'embeddings/clinicalml.txt', 'w') as of:
    of.write(f"{len(embeds)} {len(embeds[0])}\n")
    for ix, embed in enumerate(embeds):
        of.write(inv_vocab[ix].replace(' ', '_') + " " + " ".join(map(str,embed)) + "\n")

