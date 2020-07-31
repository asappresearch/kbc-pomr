# kbc-pomr

Code for the [paper](https://arxiv.org/abs/2004.12905) "Knowledge Base Completion for Constructing Problem-Oriented Medical Records" at MLHC 2020

# Data

All annotations can be found at `data/all.csv`. 
Each row lists a problem, a relation type (also the data type of the target), and the target code, along with the annotated label (1 = negative, 2 = positive).

In the `data/` directory, we also have the `train`, `dev`, and `test` splits for both experiments conducted in the paper.
`*_probs.csv` files contains data splits, separated by problem type (Table 3), and `*_rand.csv` files contains data splits, separated at random (Table 2).

We also provide: 
- `data/med_may_treat.csv` - An auxiliary lookup to find the SNOMED/ICD diagnosis codes that an RxNorm code may be related to, which we constructed by going through NDF-RT's "MayTreat" and "MayPrevent" relations
- `data/problem_codes_all.csv` - A file with our problem definitions
- `data/site_icd9_relative_freqs.csv` - A file with the relative frequencies of ICD-9 codes computed from our EHR dataset, to properly initialize problem embeddings
- `intersect_*.txt`: the lists of codes for each data type that we evaluate on, which we constructed by taking the intersection with the set of site-specific codes.
- `vocab.txt` - the vocabulary used (site-specific codes censored with X's)
- `embeddings/claims_codes_hs_300.txt` and `embeddings/claims_cuis_hs_300.txt` - the code and CUI embeddings from Choi et al

# Reproduction of results

## Download open-source embeddings

First, download and extract (with `gunzip`) the embeddings for codes ([here](https://github.com/clinicalml/embeddings/blob/master/claims_codes_hs_300.txt.gz)) and CUIs ([here](https://github.com/clinicalml/embeddings/blob/master/claims_cuis_hs_300.txt.gz)) from prior work, and put the files in the `embeddings/` directory.

## Environment setup

To set up the proper dependencies using conda, run:

`conda create -n POMR python=3.7`

`conda activate POMR`

`pip install -r requirements.txt`

## Reproducing experiments

The jupyter notebook `Reproduction.ipynb` gives full instructions to reproduce the results from the paper, specifically line 4 ("Choi et al") in Table 2 and lines 1 ("Ontology baseline") and 5 ("Choi et al") in Table 3.

At a high level, the steps are:

- Construct RxNorm-to-CUI lookup using UMLS, so we can use Choi et al's medication embeddings
- Pre-compute problem and target embeddings to use to initialize models.
- Train on the held-out triplets data splits (`*_rand.csv`) to reproduce Table 2
- Train on the held-out problems data splits (`*_newprobs.csv`) to reproduce Table 3

# Citation

If you use this repository, please cite our paper:

```
@inproceedings{mullenbach2020knowledge,
  title={Knowledge Base Completion for Constructing Problem-Oriented Medical Records},
  author={Mullenbach, James and Swartz, Jordan and McKelvey, T Greg and Dai, Hui and Sontag, David},
  booktitle={Machine Learning for Healthcare Conference},
  year={2020}
}
```
