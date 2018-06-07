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

Config files are used to specify a) the input constraints for the sentences to be generated, and b) possible structures for the sentences.

The three example config files give some illustrations of how you can specify constraints.

`config1.example.json` specifies no input constraints. Sentences can vary freely within the built-in parameters of the system.

`config2.example.json` uses all of the categories of input constraint, to illustrate usage of each.
* With `needEv`, it specifies an event that all sentences must include: in this case, an event with *lawyer* as AGENT of *recommend*. This is a JSON object.
* With `avoidEv`, it specifies an event that no sentences can include: in this case, an event with *lawyer* as AGENT of *shout*. This is a JSON object.
* With `needList`, it specifies lemmas that need to be present in every sentence. This is a JSON object with keys for 'noun','transitive' (verb), and 'intransitive' (verb), and arrays of lemmas as values.
* With `avoidList`, it specifies lemmas that cannot be present in any sentence. This is a JSON object with keys for 'noun','transitive' (verb), and 'intransitive' (verb), and arrays of lemmas as values.

`config3.example.json` demonstrates a more complex usage of `needEv` (this usage extends also to `avoidEv`).

EVENT objects can specify the following:
* `name`: the lemma describing the event (verb)
* `tense`: tense of the event (`past` or `pres`, e.g., 'sleeps' vs 'slept')
* `aspect`: aspect of the event (`neut` or `prog`, e.g., 'sleeps' vs 'is sleeping')
* `voice`: voice of the event (`active` or `passive`, e.g., 'x chased y' vs 'y was chased by x')
* `pol`: polarity of the event (`pos` or `neg`, e.g., 'slept' vs 'did not sleep')
* `frame`: transitivity of the event (`transitive` or `intransitive`)
* `participants`: participants in the event - an object with possible keys of `agent` and `patient`

The participant keys (`agent` or `patient`) map to CHARACTER objects, within which you can specify the following:
* `name`: the lemma describing the participant (noun)
* `num`: number of the event (`sg` or `pl`, e.g., 'student' vs 'students')
* `attributes`: attributes of the participant - another object. Currently the only supported attribute is a relative clause, the key for which is `rc`, and the value for which is another object.

The RC object can specify the following:
* `rtype`: object-relative or subject-relative (`orc` or `src`)
* `event`: the event described by the relative clause. The value here will be another EVENT object, as described above.


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

`get_lexicon.py` uses the [XTAG morphology database](https://www.cis.upenn.edu/~xtag/swrelease.html) to look up inflections, and will remove any lemmas from the vocabulary that are missing inflections in that lookup.
