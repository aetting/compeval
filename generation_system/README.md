# CompEval Generation System

This is the code for the generation system described in 'Assessing Composition in Sentence Vector Representations' (COLING 2018), by Allyson Ettinger, Ahmed Elgohary, Colin Phillips, and Philip Resnik.

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
* NumPy
* NLTK

# Instructions

## Config files

config1.example.json

config2.example.json

config2.example.json

## Generating sentences

python3 gen_from_meaning.py --setname xy --setdir ../../dataset/sets --configfile config2.example.json --mpo 5

python gen_from_meaning.py --setname xy --setdir ../../dataset/sets --configfile config1.example.json --mpo 5 --adv 4

## Modifying vocabulary

You can modify the vocabulary that the system draws on in the following way.

Modify `lexical/vocabulary.json` to reflect the lemmas that you want to use. The lemmas are divided into three categories: nouns, transitive verbs, intransitive verbs, and adverbs. Be sure to put your new words in the correct category. Also ensure that you use lemmas (dictionary forms - e.g., 'sleep') and NOT inflected forms (e.g., 'sleeps'). The next step will get the inflections.

Get the inflections and build the variables that the system will use by running `get_lexicon.py`.

```
python get_lexicon.py
```

This will write a new `gensys_lexvars.json` with an inflection dictionary and other lexical variables for the system to use, based on the new vocabulary.
