import argparse
from collections import Counter, defaultdict
import copy
import json
import os
import pickle
import sys
import random
import time

from gensim.models import KeyedVectors
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NUM_RELATION_TYPES = 3

def build_embed_type_and_suffix(args):
    embed_type = "Choi et al" if 'clinicalml' in args.embed_file[0] else "Site-specific"
    suffix = "clinicalml" 
    embed_type += " (ensemble)" if len(args.embed_file) > 1 else ""
    suffix += "_ensemble" if len(args.embed_file) > 1 else ""
    if args.max_epochs == 0:
        embed_type += " (frozen)"
        suffix += "_frozen"
    if args.freeze_order:
        if args.freeze_problem:
            embed_type += " \\textsc{RelationOnly}"
            suffix += "_relationOnly"
        elif args.freeze_relation:
            embed_type += " \\textsc{ProblemOnly}"
            suffix += "_problemOnly"
        else:
            embed_type += " \\textsc{Problem+Relation}"
            suffix += "_problemRelation"
    elif args.freeze_problem:
        if args.freeze_relation:
            embed_type += " \\textsc{TargetOnly}"
            suffix += "_targetOnly"
        elif args.freeze_order:
            embed_type += " \\textsc{RelationOnly}"
            suffix += "_relationOnly"
        else:
            embed_type += " \\textsc{Relation+Target}"
            suffix += "_relationTarget"
    elif args.freeze_relation:
        embed_type += " \\textsc{Problem+Target}"
        suffix += "_problemTarget"

    suffix += (f"_{args.tag}" if args.tag is not None else "")
    return embed_type, suffix

def choose_model(args, dicts, train=True):
    if len(args.embed_file) > 1:
        if train:
            model = DistMultEnsemble(args.embed_file, dicts, args.num_neg_samples, args.freeze_problem, args.freeze_relation, args.freeze_order, args.random_init, args.use_negs, args.weight_decay, args.dropout)
        else:
            model = DistMultEnsemble(args.embed_file, dicts, 0, False, False, False, False, True, 0, 0)
    else:
        if train:
            model = DistMult(args.embed_file[0], dicts, args.num_neg_samples, args.freeze_problem, args.freeze_relation, args.freeze_order, args.random_init, args.use_negs, args.weight_decay, args.dropout)
        else:
            model = DistMult(args.embed_file[0], dicts, 0, False, False, False, False, True, 0, 0)
    return model

def build_dicts(args):
    ix2code = {ix:line.strip() for ix,line in enumerate(open(args.vocab_file))}
    ix2prob = {ix:code for ix,code in enumerate(filter(str.islower, ix2code.values()))}
    ix2ord = {ix:code for ix,code in enumerate(filter(lambda s: not s.islower(), ix2code.values()))}
    prob2ix = {prob:ix for ix,prob in ix2prob.items()}
    ord2ix = {ord:ix for ix,ord in ix2ord.items()}
    rel2ix = {'medication': 0, 'procedure': 1, 'lab': 2}
    ix2rel = {ix:rel for rel,ix in rel2ix.items()}
    dicts = {'ix2code': ix2code, 'prob2ix': prob2ix, 'ix2prob': ix2prob, 'ord2ix': ord2ix, 'ix2ord': ix2ord, 'rel2ix': rel2ix, 'ix2rel': ix2rel}
    return dicts

class DistMult(nn.Module):
    def __init__(self, embed_file, dicts, num_neg_samples, freeze_problem, freeze_relation, freeze_order, random_init, use_negs, lmbda=0, dropout=0.0):
        super(DistMult, self).__init__()
        """
            read pretrained embeddings
            pull out problem embeddings into its own thing
            pull out ord embeddings into its own thing
            Find ixs where ord embeddings are 0
        """
        # problems have all lowercase names
        problem_ixs = [ix for ix,code in dicts['ix2code'].items() if code.islower()]
        ord_ixs = [ix for ix,code in dicts['ix2code'].items() if not code.islower()]

        if random_init:
            self.embed_size = 300
            self.prob_embed = nn.Embedding(len(problem_ixs), self.embed_size)
            self.ord_embed = nn.Embedding(len(ord_ixs), self.embed_size)
            self.rel_embed = nn.Embedding(NUM_RELATION_TYPES, self.embed_size)
        else:
            wv = KeyedVectors.load_word2vec_format(embed_file)
            self.embed_size = wv.vectors.shape[1]

            self.prob_embed = nn.Embedding(len(problem_ixs), self.embed_size)
            self.prob_embed = self.prob_embed.from_pretrained(torch.Tensor(wv.vectors[problem_ixs,:]), freeze=freeze_problem)

            self.ord_embed = nn.Embedding(len(ord_ixs), self.embed_size)
            self.ord_embed = self.ord_embed.from_pretrained(torch.Tensor(wv.vectors[ord_ixs,:]))

            # fill in missing vecs with xavier init
            missing_ixs = [ix for ix,vec in enumerate(wv.vectors[ord_ixs]) if vec.sum() == 0]
            missing_init = torch.Tensor(len(missing_ixs), self.embed_size)
            nn.init.xavier_uniform_(missing_init)
            self.ord_embed.weight[missing_ixs] = missing_init
            if not freeze_order:
                self.ord_embed.weight.requires_grad = True

            # initialize relation embeddings
            self.rel_embed = nn.Embedding(NUM_RELATION_TYPES, self.embed_size)
            # identity
            self.rel_embed.weight.data = torch.ones(NUM_RELATION_TYPES, self.embed_size)
            if freeze_relation:
                self.rel_embed.weight.requires_grad = False

        self.num_neg_samples = num_neg_samples
        self.use_negs = use_negs
        self.lmbda = lmbda
        self.dropout = nn.Dropout(p=dropout)
        self.ix2ord = dicts['ix2ord']

    def forward(self, problems, rels, targets, labels):
        """
            a batch is a set of examples along with negative samples
            if num_negs = 2, batch is (true_example, neg, neg, true, neg, neg,...)
        """
        scores = self.infer_scores(problems, rels, targets)
        if self.use_negs:
            pos = scores[labels == 1]
            negs = scores[labels != 1]
            margins = negs.reshape(-1,1) - pos
            losses = F.relu(margins+1).sum(1)
            if len(pos) == 0:
                return None
        else:
            # subtract score for true samples from those for negative samples
            scores = scores.reshape(-1, 1+self.num_neg_samples)
            margins = scores[:,1:] - scores[:,:1]
            # add one and sum to get loss for each example
            losses = F.relu(margins + 1).sum(1)
        loss = losses.mean()
        return loss

    def infer_scores(self, problems, rels, targets):
        #import pdb; pdb.set_trace()
        probs = self.dropout(self.prob_embed(problems))
        rels = self.dropout(self.rel_embed(rels))
        targets = self.dropout(self.ord_embed(targets))
        scores = probs.mul(rels).mul(targets).sum(1)
        return scores

    def compute_margins_and_loss(self, scores, labels):
        if self.use_negs:
            pos = scores[labels == 1]
            negs = scores[labels != 1]
            margins = negs.reshape(-1,1) - pos
            losses = F.relu(margins+1).sum(1)
            if len(pos) == 0:
                return None
        else:
            # subtract score for true samples from those for negative samples
            scores = scores.reshape(-1, 1+self.num_neg_samples)
            margins = scores[:,1:] - scores[:,:1]
            # add one and sum to get loss for each example
            losses = F.relu(margins + 1).sum(1)
        return losses.mean()


class DistMultEnsemble(nn.Module):
    def __init__(self, embed_files, dicts, num_neg_samples, freeze_problem, freeze_relation, freeze_order, random_init, use_negs, lmbda=0, dropout=0):
        super(DistMultEnsemble, self).__init__()
        # problems have all lowercase names
        problem_ixs = [ix for ix,code in dicts['ix2code'].items() if code.islower()]
        ord_ixs = [ix for ix,code in dicts['ix2code'].items() if not code.islower()]
        self.prob_embeds = nn.ModuleList([])
        self.ord_embeds = nn.ModuleList([])
        self.rel_embeds = nn.ModuleList([])

        if random_init:
            self.embed_size = 300
            for i in range(len(embed_files)):
                self.prob_embeds.append(nn.Embedding(len(problem_ixs), self.embed_size))
                self.ord_embeds.append(nn.Embedding(len(ord_ixs), self.embed_size))
                self.rel_embeds.append(nn.Embedding(NUM_RELATION_TYPES, self.embed_size))
        else:
            for embed_file in embed_files:
                wv = KeyedVectors.load_word2vec_format(embed_file)
                embed_size = wv.vectors.shape[1]

                prob_embed = nn.Embedding(len(problem_ixs), embed_size)
                prob_embed = prob_embed.from_pretrained(torch.Tensor(wv.vectors[problem_ixs,:]), freeze=freeze_problem)

                ord_embed = nn.Embedding(len(ord_ixs), embed_size)
                ord_embed = ord_embed.from_pretrained(torch.Tensor(wv.vectors[ord_ixs,:]))

                # fill in missing vecs with xavier init
                missing_ixs = [ix for ix,vec in enumerate(wv.vectors[ord_ixs]) if vec.sum() == 0]
                missing_init = torch.Tensor(len(missing_ixs), embed_size)
                nn.init.xavier_uniform_(missing_init)
                ord_embed.weight[missing_ixs] = missing_init
                if not freeze_order:
                    ord_embed.weight.requires_grad = True

                # initialize relation embeddings
                rel_embed = nn.Embedding(NUM_RELATION_TYPES, embed_size)
                # identity
                rel_embed.weight.data = torch.ones(NUM_RELATION_TYPES, embed_size)
                if freeze_relation:
                    rel_embed.weight.requires_grad = False
                self.prob_embeds.append(prob_embed)
                self.ord_embeds.append(ord_embed)
                self.rel_embeds.append(rel_embed)

        self.num_neg_samples = num_neg_samples
        self.use_negs = use_negs
        self.lmbda = lmbda
        self.num_predictors = len(embed_files)
        self.final = nn.Linear(self.num_predictors, 1, bias=False)
        self.final.weight.data = torch.ones(1,self.num_predictors) / self.num_predictors

    def forward(self, problems, rels, targets, labels):
        """
            a batch is a set of examples along with negative samples
            if num_negs = 2, batch is (true_example, neg, neg, true, neg, neg,...)
        """
        scores = self.infer_scores(problems, rels, targets)
        loss = self.compute_margins_and_loss(scores, labels)
        return loss

    def compute_margins_and_loss(self, scores, labels):
        if self.use_negs:
            pos = scores[labels == 1]
            negs = scores[labels != 1]
            margins = negs.reshape(-1,1) - pos
            losses = F.relu(margins+1).sum(1)
            if len(pos) == 0:
                return None
        else:
            # subtract score for true samples from those for negative samples
            scores = scores.reshape(-1, 1+self.num_neg_samples)
            margins = scores[:,1:] - scores[:,:1]
            # add one and sum to get loss for each example
            losses = F.relu(margins + 1).sum(1)
        return losses.mean()

    def infer_scores(self, in_problems, in_rels, in_targets):
        scores = []
        for i in range(self.num_predictors):
            probs = self.prob_embeds[i](in_problems)
            rels = self.rel_embeds[i](in_rels)
            targets = self.ord_embeds[i](in_targets)
            score = probs.mul(rels).mul(targets).sum(1)
            scores.append(score.unsqueeze(1))
        scores = torch.cat(scores, dim=1)
        final_scores = self.final(scores).squeeze()
        return final_scores

        
class TripleGenerator():
    def __init__(self, fname, prob2ix, ord2ix, rel2ix, batch_size, num_neg_samples=2, rxn_codes=None, loinc_codes=None, px_codes=None, prob_code_weights=None, scorer_fn=None, temperature=1.0, use_negs=False):
        self.triples = pd.read_csv(fname)
        self.triple_set = set([(row.problem, row.target) for row in self.triples.itertuples()])
        self.target_set = set([tup[1] for tup in self.triple_set])
        self.target_cnt = Counter([tup[1] for tup in self.triple_set])
        self.pos_set = set([(row.problem, row.target) for row in self.triples.itertuples() if row.label == 2])
        self.pos_target_set = set([tup[1] for tup in self.pos_set])
        self.pos_target_cnt = Counter([tup[1] for tup in self.pos_set])
        self.neg_set = set([(row.problem, row.target) for row in self.triples.itertuples() if row.label == 1])
        self.neg_target_set = set([tup[1] for tup in self.neg_set])
        self.num_neg_samples = num_neg_samples
        self.prob2ix = prob2ix
        self.ix2prob = {i:p for p,i in self.prob2ix.items()}
        self.ord2ix = ord2ix
        self.ix2ord = {i:p for p,i in self.ord2ix.items()}
        self.rel2ix = rel2ix
        self.ix2rel = {i:p for p,i in self.rel2ix.items()}
        self.batch_size = batch_size
        self.rxn_codes = set()
        self.loinc_codes = set()
        self.px_codes = set()
        if rxn_codes is not None:
            self.rxn_codes.update([line.strip() for line in open(rxn_codes)])
        if loinc_codes is not None:
            self.loinc_codes.update([line.strip() for line in open(loinc_codes)])
        if px_codes is not None:
            self.px_codes.update([line.strip() for line in open(px_codes)])
        self.scorer_fn = scorer_fn
        self.temperature = temperature
        self.use_negs = use_negs

    def __len__(self):
        return len(self.triples)

    def precompute_sample_weights(self):
        # can pre-compute weights for each (problem, relationType) pair = 32*3 = 96 pairs. 
        # then when in use just have to ignore the true target if it shows up
        sample_weights = defaultdict(np.array)
        for relationType, prefix, ord_codes in zip(sorted(self.rel2ix.keys()), ['LAB_', 'RX_', 'PX_'], [self.loinc_codes, self.rxn_codes, self.px_codes]):
            for problem in self.prob2ix.keys():
                neg_targets = [ord for ord in self.ord2ix.keys() if ord.startswith(prefix)]
                if self.scorer_fn is not None:
                    problems = torch.LongTensor([self.prob2ix[problem]] * len(neg_targets))
                    rels = torch.LongTensor([self.rel2ix[relationType]] * len(neg_targets))
                    targets = torch.LongTensor([self.ord2ix[nt] for nt in neg_targets])
                    scores = self.scorer_fn(problems, rels, targets)
                    weights = F.softmax(scores/self.temperature, dim=0).data.numpy()
                    # numerical manipulation to make it sum to 1 according to numpy
                    if weights.sum() < 1:
                        resid = 1 - weights.sum()
                        weights += resid / len(weights)
                    elif weights.sum() > 1:
                        resid = weights.sum() - 1
                        weights -= resid / len(weights)
                    sample_weights[(problem, relationType)] = weights
                else:
                    sample_weights[(problem, relationType)] = np.ones(len(neg_targets)) / len(neg_targets)
        self.sample_weights = sample_weights

    def negative_sample(self, problem, relationType, target):
        neg_problem = random.choice(list(set(self.prob2ix.keys()) - set([problem])))
        while (neg_problem, target) in self.triple_set:
            neg_problem = random.choice(list(set(self.prob2ix.keys()) - set([problem])))

        if target.startswith('RX_'):
            neg_relationType = 'medication'
            neg_targets = [ord for ord in self.ord2ix.keys() if ord.startswith('RX_')]
        elif target.startswith('LAB_'):
            neg_relationType = 'lab'
            neg_targets = [ord for ord in self.ord2ix.keys() if ord.startswith('LAB_')]
        elif target.startswith('PX_'):
            neg_relationType = 'procedure'
            neg_targets = [ord for ord in self.ord2ix.keys() if ord.startswith('PX_')]

        weights = self.sample_weights[(problem, relationType)]
        neg_target = np.random.choice(neg_targets, p=weights)
        while (problem, neg_target) in self.triple_set or neg_target == target:
            neg_target = np.random.choice(neg_targets, p=weights)

        return [(neg_problem, relationType, target), (problem, neg_relationType, neg_target)]

    def skip(self, target, relationType):
        # this should only happen when training on problemlist.org data rather than our annotated data
        if target not in self.ord2ix:
            return True

    def skip_eval(self, target, relationType):
        if target not in self.ord2ix:
            return True
        if relationType == 'medication' and (target not in self.rxn_codes and len(self.rxn_codes) != 0):
            return True
        elif relationType == 'lab' and (target not in self.loinc_codes and len(self.loinc_codes) != 0):
            return True
        elif relationType == 'procedure' and (target not in self.px_codes and len(self.px_codes) != 0):
            return True

    def generate(self, seed):
        self.triples = self.triples.sample(frac=1, random_state=seed)

        samples = []
        num_missing = 0
        cur_problem = ''
        cur_relation = ''
        for ix, triple in enumerate(self.triples.itertuples()):
            target = triple.target
            if self.skip(target, triple.relationType):
                num_missing += 1
                continue

            if self.use_negs:
                # subtract one because data is in 1-2 format instead of 0-1 lol
                samples.append((self.prob2ix[triple.problem], self.rel2ix[triple.relationType], self.ord2ix[target], triple.label-1))
                if len(samples) >= self.batch_size:
                    yield samples
                    samples = []
                    cur_problem = triple.problem
                    cur_relation = triple.relationType
            else:
                samples.append((self.prob2ix[triple.problem], self.rel2ix[triple.relationType], self.ord2ix[target], 1))
                if self.num_neg_samples == 0:
                    yield samples
                    samples = []
                for i in range(self.num_neg_samples // 2):
                    negs = self.negative_sample(triple.problem, triple.relationType, target)
                    for neg in negs:
                        samples.append((self.prob2ix[neg[0]], self.rel2ix[neg[1]], self.ord2ix[neg[2]], 0))
                        if len(samples) >= self.batch_size * (self.num_neg_samples+1):
                            yield samples
                            samples = []
        if len(samples) > 1:
            yield samples

    def generate_dev(self, train_triples):
        if self.use_negs:
            # create common set of negatives for each (problem, relationType) pair
            negs = defaultdict(list)
            for ix, triple in enumerate(self.triples.itertuples()):
                if triple.label == 1:
                    negs[(triple.problem, triple.relationType)].append(triple.target)
        samples = []
        for ix, triple in enumerate(self.triples.itertuples()):
            target = triple.target
            if self.skip_eval(target, triple.relationType):
                continue
            if self.use_negs:
                # skip negatives because we already gathered them for comparison to the positives
                if triple.label != 2:
                    continue
                samples.append((self.prob2ix[triple.problem], self.rel2ix[triple.relationType], self.ord2ix[target]))
                for neg in negs[(triple.problem, triple.relationType)]:
                    samples.append((self.prob2ix[triple.problem], self.rel2ix[triple.relationType], self.ord2ix[neg]))
                if len(samples) > 1:
                    yield samples
                samples = []
            else:
                samples.append((self.prob2ix[triple.problem], self.rel2ix[triple.relationType], self.ord2ix[target]))
                # add ALL other negative targets of same type
                for ord in self.ord2ix.keys():
                    if ord[:3] == target[:3]:
                        if not self.skip_eval(ord, triple.relationType):
                            if (triple.problem, ord) not in train_triples and ord != target:
                                samples.append((self.prob2ix[triple.problem], self.rel2ix[triple.relationType], self.ord2ix[ord]))
                yield samples
                samples = []
        yield samples

def create_examples(dev_data, dicts, scorer_fn, out_dir, train_triples, suffix=""):
    if out_dir is not None:
        of = open(f'{out_dir}/html_examples.txt', 'w')
    dev_probs = set([prob for prob, target in dev_data.triple_set])
    for prob in dev_probs:
        prob_str = prob.replace('_', ' ')
        prob_str = prob_str[0].upper() + prob_str[1:]
        of.write(f"<table><tr><td>{prob_str}</td></tr>\n")
        of.write(f"<table><tr><td>Medication</td><td>Procedure</td><td>Lab</td></tr>\n")
        latex_meds = []
        latex_procs = []
        latex_labs = []
        for rel, rel_ix in dev_data.rel2ix.items():
            # target codes are all codes of that type in the dev set
            suffix = 'RX_' if rel == 'medication' else ('PX_' if rel == 'procedure' else 'LAB_')
            target_ixs = [ord_ix for ord, ord_ix in dev_data.ord2ix.items() if ord.startswith(suffix) and (prob, ord) not in train_triples and (prob, ord) in dev_data.triple_set]
            if len(target_ixs) == 0:
                continue
            problems = torch.LongTensor([dev_data.prob2ix[prob]] * len(target_ixs))
            rels = torch.LongTensor([rel_ix] * len(target_ixs))
            targets = torch.LongTensor(target_ixs)

            scores = scorer_fn(problems, rels, targets)
            sorted_ixs = np.argsort(scores.numpy())[::-1]

            top10_strs = []
            for trank, ix in enumerate(sorted_ixs[:30]):
                target = targets[ix].item()
                target_code = dicts['ix2ord'][target]

                if target_code.startswith('RX_'):
                    raw_code = target_code[3:]
                    target_str = raw_code
                elif target_code.startswith('PX_'):
                    raw_code = target_code[3:]
                    target_str = raw_code
                elif target_code.startswith('LAB_'):
                    raw_code = target_code[4:]
                    target_str = raw_code
                try:
                    target_str = target_str[0].upper() + target_str[1:].lower()
                except:
                    continue
                if trank < 10:
                    top10_strs.append((target_str, (prob, target_code) in dev_data.pos_set))

            examples = [s if not trueExample else '<span style="color: blue; font-weight: 600">' + s + '</span>' for s,trueExample in top10_strs]
            if rel == 'medication':
                latex_meds = examples
            if rel == 'procedure':
                latex_procs = examples
            if rel == 'lab':
                latex_labs = examples

        for m,p,l in zip(latex_meds, latex_procs, latex_labs):
            of.write(f"<tr><td>{m}</td><td>{p}</td><td>{l}</td></tr>\n")
        of.write("</table>")

def eval_dev(dev_data, train_data, dicts, scorer_fn, out_dir, epoch, embed_type, split_type, force_print=False, save_examples=False, is_test=False, save_fig=False):
    if out_dir is not None and not is_test:
        of = open(f'{out_dir}/dv_preds.jsonl', 'w')
    ranks = []
    rand_ranks = []
    dev_losses = []

    rx_ranks = []
    lab_ranks = []
    px_ranks = []

    examples = []
    if save_examples:
        base_name = out_dir.split('/')[-1]
        of = open(f'{out_dir}/examples_{base_name}.txt', 'w')

    examples_written = set() 

    matrix_vals = defaultdict(list)
    in_train_ranks = []
    not_in_train_ranks = []
    pos_train_ranks = []
    neg_train_ranks = []
    both_train_ranks = []
    for ix, batch in enumerate(dev_data.generate_dev(train_data.triple_set)):
        if len(batch) == 0:
            continue
        problems, rels, targets = list(zip(*batch))
        problems = torch.LongTensor(problems)
        rels = torch.LongTensor(rels)
        targets = torch.LongTensor(targets)

        scores = scorer_fn(problems, rels, targets)
        margins = scores[1:] - scores[0]
        loss = F.relu(margins + 1).sum()

        dev_losses.append(loss)
        sorted_ixs = np.argsort(scores.numpy())[::-1]
        rank = np.where(sorted_ixs == 0)[0][0]
        ranks.append(rank)
        matrix_vals[(dicts['ix2prob'][batch[0][0]], dicts['ix2rel'][batch[0][1]])].append(rank)
        target_code = dicts['ix2ord'][batch[0][2]]
        if target_code in train_data.target_set:
            in_train_ranks.append(rank)
        else:
            not_in_train_ranks.append(rank)
        if target_code in train_data.pos_target_set:
            pos_train_ranks.append(rank)
            if target_code in train_data.neg_target_set:
                both_train_ranks.append(rank)
        elif target_code in train_data.neg_target_set:
            neg_train_ranks.append(rank)

        rand_ranks.append(random.choice(range(len(scores))))

        target_code = dicts['ix2ord'][targets[0].item()]
        if target_code.startswith('RX_'):
            rx_ranks.append(rank)
        elif target_code.startswith('PX_'):
            px_ranks.append(rank)
        elif target_code.startswith('LAB_'):
            lab_ranks.append(rank)

        if save_examples:
            prob = dicts['ix2prob'][problems[0].item()]
            rel = dicts['ix2rel'][rels[0].item()]
            if (prob, rel) not in examples_written:
                if target_code.startswith('RX_'):
                    target_str = target_code[3:]
                elif target_code.startswith('PX_'):
                    target_str = target_code[3:]
                elif target_code.startswith('LAB_'):
                    target_str = target_code[4:]

                of.write(prob + " - " + rel + " - " + target_str + f"({target_code})" + "\n")
                for trank, ix in enumerate(sorted_ixs[:10]):
                    target = targets[ix].item()
                    target_code = dicts['ix2ord'][target]

                    if target_code.startswith('RX_'):
                        target_str = target_code[3:]
                    elif target_code.startswith('PX_'):
                        target_str = target_code[3:]
                    elif target_code.startswith('LAB_'):
                        target_str = target_code[4:]
                        
                    of.write(f"rank {trank+1}: {target_str} ({target_code})\n")

                of.write(f"rank of true target: {rank}/{len(targets)}\n\n")
            examples_written.add((prob, rel))

    ranks = np.array(ranks)
    in_train_ranks = np.array(in_train_ranks)
    not_in_train_ranks = np.array(not_in_train_ranks)
    pos_train_ranks = np.array(pos_train_ranks)
    neg_train_ranks = np.array(neg_train_ranks)
    both_train_ranks = np.array(both_train_ranks)
    rx_ranks = np.array(rx_ranks)
    px_ranks = np.array(px_ranks)
    lab_ranks = np.array(lab_ranks)
    rand_ranks = np.array(rand_ranks)
    dev_loss = np.mean(dev_losses).astype(np.float64)
    if save_examples:
        of.close()

    mr = np.mean(ranks+1)
    mrr = np.mean(1./(ranks+1))
    hits_at_10 = np.mean([rank < 10 for rank in ranks])
    hits_at_30 = np.mean([rank < 30 for rank in ranks])
    hits_at_1 = np.mean([rank < 1 for rank in ranks])
    hits_at_5 = np.mean([rank < 5 for rank in ranks])

    rx_mr = np.mean(rx_ranks+1)
    rx_mrr = np.mean(1./(rx_ranks+1))
    rx_hits_at_10 = np.mean([rank < 10 for rank in rx_ranks])
    rx_hits_at_30 = np.mean([rank < 30 for rank in rx_ranks])
    rx_hits_at_1 = np.mean([rank < 1 for rank in rx_ranks])
    rx_hits_at_5 = np.mean([rank < 5 for rank in rx_ranks])

    px_mr = np.mean(px_ranks+1)
    px_mrr = np.mean(1./(px_ranks+1))
    px_hits_at_10 = np.mean([rank < 10 for rank in px_ranks])
    px_hits_at_30 = np.mean([rank < 30 for rank in px_ranks])
    px_hits_at_1 = np.mean([rank < 1 for rank in px_ranks])
    px_hits_at_5 = np.mean([rank < 5 for rank in px_ranks])

    lab_mr = np.mean(lab_ranks+1)
    lab_mrr = np.mean(1./(lab_ranks+1))
    lab_hits_at_10 = np.mean([rank < 10 for rank in lab_ranks])
    lab_hits_at_30 = np.mean([rank < 30 for rank in lab_ranks])
    lab_hits_at_1 = np.mean([rank < 1 for rank in lab_ranks])
    lab_hits_at_5 = np.mean([rank < 5 for rank in lab_ranks])

    in_rank_mrr = np.mean(1./(in_train_ranks+1))
    not_in_rank_mrr = np.mean(1./(not_in_train_ranks+1))
    in_rank_hits_at_5 = np.mean([rank < 5 for rank in in_train_ranks])
    not_in_rank_hits_at_5 = np.mean([rank < 5 for rank in not_in_train_ranks])

    pos_rank_mrr = np.mean(1./(pos_train_ranks+1))
    pos_rank_hits_at_5 = np.mean([rank < 5 for rank in pos_train_ranks])
    neg_rank_mrr = np.mean(1./(neg_train_ranks+1))
    neg_rank_hits_at_5 = np.mean([rank < 5 for rank in neg_train_ranks])
    both_rank_mrr = np.mean(1./(both_train_ranks+1))
    both_rank_hits_at_5 = np.mean([rank < 5 for rank in both_train_ranks])

    if force_print:
        print("METRICS")
        if split_type == 'problems':
            print("MR,MRR,RX_MRR,RX_H@5,PX_MRR,PX_H@5,LAB_MRR,LAB_H@5")
            print(f"{embed_type},{mr:.2f},{mrr:.3f},{rx_mrr:.3f},{rx_hits_at_5:.3f},{px_mrr:.3f},{px_hits_at_5:.3f},{lab_mrr:.3f},{lab_hits_at_5:.3f}")
            print()
        elif split_type == 'triplets':
            print("MR,MRR,RX_MRR,RX_H@1,PX_MRR,PX_H@1,LAB_MRR,LAB_H@1")
            print(f"{embed_type},{mr:.2f},{mrr:.3f},{rx_mrr:.3f},{rx_hits_at_1:.3f},{px_mrr:.3f},{px_hits_at_1:.3f},{lab_mrr:.3f},{lab_hits_at_1:.3f}")
            print()


    if is_test and split_type == 'problems':
        print("MATRIX")
        for (prob, relType), vals in sorted(matrix_vals.items(), key=lambda x: x[0][1]):
            hits_at_5 = np.mean([rank < 5 for rank in vals])
            print(f"{relType}, {prob}: {hits_at_5:.3f}")
        print()


    metrics = {}
    metrics['mr'] = mr
    metrics['mrr'] = mrr
    metrics['hits@10'] = hits_at_10
    metrics['hits@30'] = hits_at_30
    metrics['hits@1'] = hits_at_1
    metrics['hits@5'] = hits_at_5
    metrics['dev_loss'] = dev_loss

    metrics['rx_mr'] = rx_mr
    metrics['rx_mrr'] = rx_mrr
    metrics['rx_hits@10'] = rx_hits_at_10
    metrics['rx_hits@30'] = rx_hits_at_30
    metrics['rx_hits@1'] = rx_hits_at_1
    metrics['rx_hits@5'] = rx_hits_at_5

    metrics['px_mr'] = px_mr
    metrics['px_mrr'] = px_mrr
    metrics['px_hits@10'] = px_hits_at_10
    metrics['px_hits@30'] = px_hits_at_30
    metrics['px_hits@1'] = px_hits_at_1
    metrics['px_hits@5'] = px_hits_at_5

    metrics['lab_mr'] = lab_mr
    metrics['lab_mrr'] = lab_mrr
    metrics['lab_hits@10'] = lab_hits_at_10
    metrics['lab_hits@30'] = lab_hits_at_30
    metrics['lab_hits@1'] = lab_hits_at_1
    metrics['lab_hits@5'] = lab_hits_at_5
    return metrics

def check_best_model_and_save(model, metrics_hist, criterion, out_dir):
    is_best = False
    if criterion == 'mr':
        if np.nanargmin(metrics_hist[criterion]) == len(metrics_hist[criterion]) - 1:
            # save model
            sd = model.state_dict()
            torch.save(sd, f'{out_dir}/model_best_{criterion}.pth')
            is_best = True
    else:
        if np.nanargmax(metrics_hist[criterion]) == len(metrics_hist[criterion]) - 1:
            # save model
            sd = model.state_dict()
            torch.save(sd, f'{out_dir}/model_best_{criterion}.pth')
            is_best = True
    return is_best


def save_metrics(metrics_hist, out_dir):
    # save predictions
    if out_dir is not None and not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(f'{out_dir}/metrics.json', 'w') as of:
        json.dump(metrics_hist, of, indent=1)
    # make and save plot
    #for metric in metrics_hist:
    #    if metric[:3] not in ['lab', 'px_', 'rx_']:
    #        plt.figure()
    #        plt.plot(metrics_hist[metric])
    #        plt.xlabel('epoch')
    #        plt.ylabel(metric)
    #        plt.title(f"dev {metric} vs. epochs")
    #        plt.savefig(f'{out_dir}/dev_{metric}_plot.png')
    #        plt.close()

def early_stop(metrics_hist, criterion, patience):
    if len(metrics_hist[criterion]) >= patience:
        if criterion in ['mr', 'dev_loss']:
            return np.nanargmin(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
        else:
            return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        return False

def main(args):
    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dicts = build_dicts(args)
    model = choose_model(args, dicts)

    scorer_fn = None
    if args.weighted_sample:
        init_model = copy.deepcopy(model)
        scorer_fn = init_model.embed_scores

    train_data = TripleGenerator(args.train_file, dicts['prob2ix'], dicts['ord2ix'], dicts['rel2ix'], args.batch_size, rxn_codes=args.rxn_codes, loinc_codes=args.loinc_codes, px_codes=args.px_codes, scorer_fn=scorer_fn, temperature=args.temperature, use_negs=args.use_negs)
    dev_file = args.train_file.replace('train', 'dev')
    dev_data = TripleGenerator(dev_file, dicts['prob2ix'], dicts['ord2ix'], dicts['rel2ix'], 1, num_neg_samples=0, rxn_codes=args.rxn_codes, loinc_codes=args.loinc_codes, px_codes=args.px_codes, use_negs=True)

    embed_type, suffix = build_embed_type_and_suffix(args)
    timestamp = time.strftime('%b_%d_%H:%M:%S', time.localtime())
    out_dir = f'results/distmult_{suffix}_{timestamp}'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    losses = []
    num_batches = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    metrics_hist = defaultdict(list)
    stop_training = False
    best_epoch = 0
    for epoch in range(args.max_epochs):
        print(". ", end='')
        epoch_losses = []
        if epoch == 0:
            train_data.precompute_sample_weights()
        for batch_ix, batch in enumerate(train_data.generate(args.seed)):
            problems, rels, targets, labels = list(zip(*batch))
            problems = torch.LongTensor(problems)
            rels = torch.LongTensor(rels)
            targets = torch.LongTensor(targets)
            labels = torch.FloatTensor(labels)

            model.zero_grad()
            #import pdb; pdb.set_trace()
            loss = model(problems, rels, targets, labels)
            if loss is not None and not torch.isnan(loss):
                epoch_losses.append(loss.item())

                loss.backward()
                optimizer.step()

            num_batches += 1
            
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)

        # eval on dev after every epoch as well
        with torch.no_grad():
            model.eval()
            metrics = eval_dev(dev_data, train_data, dicts, model.infer_scores, out_dir, epoch, embed_type, args.split_type, force_print=args.verbose, save_examples=True)
            for name, metric in metrics.items():
                metrics_hist[name].append(metric)
            save_metrics(metrics_hist, out_dir)
            is_best = check_best_model_and_save(model, metrics_hist, args.criterion, out_dir)
            if is_best:
                best_epoch = epoch
            if early_stop(metrics_hist, args.criterion, args.patience):
                print("!!! early stopping hit !!!")
                stop_training = True
                break
            create_examples(dev_data, dicts, model.infer_scores, out_dir, train_data.triple_set)

        if stop_training:
            break

    # save args
    with open(f'{out_dir}/args.json', 'w') as of:
        of.write(json.dumps(args.__dict__, indent=2) + "\n")

    if args.max_epochs > 0:
        # save the model at the end
        sd = model.state_dict()
        torch.save(sd, out_dir + "/model.pth")

        # reload the best model
        print(f"\nReloading and evaluating model with best {args.criterion} (epoch {best_epoch})")
        sd = torch.load(f'{out_dir}/model_best_{args.criterion}.pth')
        model.load_state_dict(sd)

    # eval on dev at end
    with torch.no_grad():
        model.eval()
        all_metrics = eval_dev(dev_data, train_data, dicts, model.infer_scores, out_dir, 0, embed_type, args.split_type, force_print=args.verbose, save_examples=True)
        create_examples(dev_data, dicts, model.infer_scores, out_dir, train_data.triple_set)
        if args.run_test:
            print()
            print("RUNNING TEST")
            test_file = args.train_file.replace('train', 'test')
            test_data = TripleGenerator(test_file, dicts['prob2ix'], dicts['ord2ix'], dicts['rel2ix'], 1, num_neg_samples=0, rxn_codes=args.rxn_codes, loinc_codes=args.loinc_codes, px_codes=args.px_codes, use_negs=True)
            eval_dev(test_data, train_data, dicts, model.infer_scores, out_dir, 0, embed_type, args.split_type, force_print=True, save_examples=True, is_test=True, save_fig=True)
            create_examples(test_data, dicts, model.infer_scores, out_dir, train_data.triple_set, suffix="test")
    print(f"THIS RUN'S RESULT DIR IS: {out_dir}")
    print("\n\n")
        


if __name__ == "__main__":
    print("starting!")
    print("COMMAND: " + ' '.join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument("embed_file", nargs="+", help="path to embedding file (already consolidated into problem-level)")
    parser.add_argument("vocab_file", type=str, help="path to vocab file (already consolidated into problem-level)")
    parser.add_argument("train_file", type=str, help="path to train file (dev file path will be assumed from formatting)")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for training")
    parser.add_argument("--num_neg_samples", type=int, default=2, help="num neg samples to use (must be divisible by 2)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for adam")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for embeddings")
    parser.add_argument("--weight_decay", type=float, default=0, help="l2 regularization strength")
    parser.add_argument("--max_epochs", type=int, default=100, help="batch size for training")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for softmax for random sampling - a float between 0 and 1. Smaller values make spikier distributions, larger values make smoother distributions")
    parser.add_argument("--embed_size", required=False, type=int, help="size of embeddings")
    parser.add_argument("--seed", required=False, type=int, default=11, help="random seed")
    parser.add_argument("--split_type", required=False, type=str, choices=['problems', 'triplets'], default='problems', help="which type of split (problems or triplets)")
    parser.add_argument("--rxn_codes", required=False, type=str, help="path to pickled rxnorm codes set")
    parser.add_argument("--loinc_codes", required=False, type=str, help="path to pickled loinc codes set")
    parser.add_argument("--px_codes", required=False, type=str, help="path to pickled px codes set")
    parser.add_argument("--tag", required=False, type=str, help="tag to put at end of output dir name for findability")
    parser.add_argument("--freeze_problem", action="store_true", help="set to freeze problem embedding weights")
    parser.add_argument("--freeze_relation", action="store_true", help="set to freeze relation embedding weights")
    parser.add_argument("--freeze_order", action="store_true", help="set to freeze order (med/lab/proc) embedding weights")
    parser.add_argument("--random_init", action="store_true", help="set to randomly initialize all embeddings (ignores freeze arguments)")
    parser.add_argument("--verbose", action="store_true", help="set to print more")
    parser.add_argument("--weighted_sample", action="store_true", help="set to sample negatives proportionally to the model's score")
    parser.add_argument("--use_negs", action="store_true", help="set not sample negatives in training and instead use the negatives in the dataset provided")
    parser.add_argument("--run_test", action="store_true", help="set to run on test too after running on dev at the end")
    parser.add_argument("--criterion", type=str, default='mrr', required=False, help="Which metric to use for early stopping (default: mrr)")
    parser.add_argument("--patience", type=int, default=5, required=False, help="How many epochs to wait for improved criterion metric before early stopping (default: 5)")
    parser.add_argument("--print_every", type=int, default=5, required=False, help="How many epochs to wait between loss printouts (default: 5)")
    args = parser.parse_args()
    if args.use_negs:
        args.weighted_sample = False
    main(args)
