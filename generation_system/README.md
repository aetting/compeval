# CompEval Generation System

This is the code for the generation system described in Assessing Composition in Sentence Vector Representations (COLING 2018), by Allyson Ettinger, Ahmed Elgohary, Colin Phillips, and Philip Resnik.

# Citation

```
@inproceedings{ettinger2018assessing,
  author = {Allyson Ettinger and Ahmed Elgohary and Colin Phillips and Philip Resnik},
  title = {Assessing Composition in Sentence Vector Representations},
  booktitle = {Proceedings of COLING},
  year = {2018},
 }
```

# Prerequisites

NumPy
NLTK

# Instructions

## Preparing vocab

If you want to change the vocabulary, you can modify 'vocabulary.json'.

## Config files

config1.example.json

config2.example.json

config2.example.json

## Generating sentences

python3 gen_from_meaning.py --setname xy --setdir ../../dataset/sets --configfile config2.example.json --mpo 5

python gen_from_meaning.py --setname xy --setdir ../../dataset/sets --configfile config1.example.json --mpo 5 --adv 4
