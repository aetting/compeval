from __future__ import print_function

import random
import pickle
import os
import gzip
import math
import ast
import operator
import argparse
import itertools
import shutil
import numpy as np
from gen_traintestfolds import read_set
from classify_lreg import load_embdict
from collections import Counter
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score


# collect all sentences, shuffle
def get_sents(setfilename):
    sents = []
    levdict = read_set(setfilename)
    for lev1 in levdict:
        level1dict = ast.literal_eval(lev1)
        rcstatuses = ['transitive','intransitive']
        if 'patient' in level1dict and level1dict['patient'] in rcstatuses and level1dict['agent'] in rcstatuses:
            continue
        # if level1dict['agent'] in rcstatuses or ('patient' in level1dict and level1dict['patient'] in rcstatuses):
        #     continue
        for lev2 in levdict[lev1]:
            for lev3 in levdict[lev1][lev2]:
                for lev4 in levdict[lev1][lev2][lev3]:
                    sents += levdict[lev1][lev2][lev3][lev4]

    sentdict = {}
    random.shuffle(sents)
    # sents_to_use = sents[:numsents]

    for line in sents:
        id,sent = line.split('\t')
        sentdict[id] = sent

    return sentdict

#get the words in each sentence
def get_words_from_annot(annotDict):
    words = []
    words.append((annotDict['name'],annotDict['surface']))
    for part in annotDict['participants']:
        if annotDict['participants'][part]['surface']:
            words.append((annotDict['participants'][part]['name'],annotDict['participants'][part]['surface']))
        if 'rc' in annotDict['participants'][part]['attributes']:
            rcdict = annotDict['participants'][part]['attributes']['rc']['event']
            rcwords = get_words_from_annot(rcdict)
            words += rcwords
    words = list(set(words))
    return words

def get_rels_from_annot(annotDict):
    rels = {}
    rels['agent'] = []
    rels['patient'] = []
    mainev = annotDict['name']
    for part in annotDict['participants']:
        rels[part].append((annotDict['participants'][part]['name'],mainev))
        if 'rc' in annotDict['participants'][part]['attributes']:
            rcdict = annotDict['participants'][part]['attributes']['rc']['event']
            rcrels = get_rels_from_annot(rcdict)
            rels['agent'] += rcrels['agent']
            rels['patient'] += rcrels['patient']
    return rels

def get_pols_from_annot(annotDict):
    pols = {}
    pols['neg'] = []
    pols['pos'] = []
    pols[annotDict['pol']].append(annotDict['name'])
    for part in annotDict['participants']:
        if 'rc' in annotDict['participants'][part]['attributes']:
            rcdict = annotDict['participants'][part]['attributes']['rc']['event']
            rcpols = get_pols_from_annot(rcdict)
            pols['neg'] += rcpols['neg']
            pols['pos'] += rcpols['pos']
    return pols

def get_neg_scope_from_annot(annotDict):
    #if main verb pol = neg: main vb is negated
    #if main vb active and negated and patient has rc: patient-rc vb is negated
    #if main vb passive and agent has rc: agent-rc vb negated
    polscope = {}
    polscope['neg'] = []
    polscope['pos'] = []
    polscope[annotDict['pol']].append(annotDict['name'])
    for part in annotDict['participants']:
        if 'rc' in annotDict['participants'][part]['attributes']:
                rcdict = annotDict['participants'][part]['attributes']['rc']['event']
                if annotDict['pol'] == 'neg' or rcdict['pol'] == 'neg':
                    polscope['neg'].append(rcdict['name'])
                else:
                    polscope['pos'].append(rcdict['name'])

    # for part in annotDict['participants']:
    #     if 'rc' in annotDict['participants'][part]['attributes']:
    #         rcdict = annotDict['participants'][part]['attributes']['rc']['event']
    #         if annotDict['pol'] == 'neg':
    #             if (part == 'agent' and annotDict['voice'] == 'passive') or (part == 'patient' and annotDict['voice'] == 'active'):
    #                 polscope['neg'].append(rcdict['name'])
    #             else:
    #                 if rcdict['pol'] == 'neg':
    #                     polscope['neg'].append(rcdict['name'])
    #                 elif rcdict['pol'] == 'pos':
    #                     polscope['pos'].append(rcdict['name'])
    #         else:
    #             if rcdict['pol'] == 'neg':
    #                 polscope['neg'].append(rcdict['name'])
    #             elif rcdict['pol'] == 'pos':
    #                 polscope['pos'].append(rcdict['name'])
    return polscope



def sents2items(sentwordlist,outfilepath,vocab,labdict):
    items = []
    tot = 0
    true = 0
    for id,sent,words in sentwordlist:
        vwords = [e for e in words if e in vocab]
        abswords = [e for e in vocab if e not in words]
#             random.shuffle(abswords)
        for w in vwords:
            items.append(['TRUE',id,sent,w])
            labdict[w]['TRUE'] += 1
            true += 1
            tot += 1
        i = 0
        for w in abswords:
            i += 1
            if i > len(vwords): break
            items.append(['FALSE',id,sent,w])
            labdict[w]['FALSE'] += 1
            tot += 1
    random.shuffle(items)
    with open(outfilepath,'w') as out:
        for it in items: out.write('\t'.join(it) + '\n')
    return labdict,tot,true


def get_xypairs(words,vocabn,vocabv,holdoutpairs):
    lems,surfs = zip(*words)
    in_nouns = [e for e in lems if e in vocabn]
    in_verbs = [e for e in lems if e in vocabv]
    pos_pairs = list(itertools.product(in_nouns,in_verbs))

    absnouns = [e for e in vocabn if e not in lems]
    absverbs = [e for e in vocabv if e not in lems]
    n_vabs = list(itertools.product(in_nouns,absverbs))
    v_nabs = list(itertools.product(absnouns,in_verbs))
    # other_out = [p for p in itertools.product(vocabn,vocabv) if p not in pos_pairs + n_vabs + v_nabs]

    for l in [n_vabs,v_nabs]:
        random.shuffle(l)

    #make list of negative pairs, starting with n_vabs and v_nabs interleaved
    neg_pairs = []
    for i in range(max(len(n_vabs),len(v_nabs))):
        for l in [n_vabs,v_nabs]:
            if i < len(l): neg_pairs.append(l[i])
    # neg_pairs += other_out
    pos_pairs = [e for e in pos_pairs if e not in holdoutpairs]
    neg_pairs = [e for e in neg_pairs if e not in holdoutpairs]

    maxlen = min(len(pos_pairs),len(neg_pairs))

    for l in [pos_pairs,neg_pairs]:
        random.shuffle(l)
    pos_pairs = pos_pairs[:maxlen]
    neg_pairs = neg_pairs[:maxlen]

    return pos_pairs,neg_pairs

#take all agent relations
def get_rel_pairs(targetrel,rels,words,vocabn,vocabv,holdoutpairs):
    pos_pairs = []
    neg_pairs = []
    for r in rels[targetrel]:
        pos_pairs.append(r)
    lems,surfs = zip(*words)
    in_nouns = [e for e in lems if e in vocabn]
    in_verbs = [e for e in lems if e in vocabv]
    neg_pairs = [p for p in itertools.product(in_nouns,in_verbs) if p not in pos_pairs]

    pos_pairs = [e for e in pos_pairs if e not in holdoutpairs]
    neg_pairs = [e for e in neg_pairs if e not in holdoutpairs]

    maxlen = min(len(pos_pairs),len(neg_pairs))

    for l in [pos_pairs,neg_pairs]:
        random.shuffle(l)
    pos_pairs = pos_pairs[:maxlen]
    neg_pairs = neg_pairs[:maxlen]

    return pos_pairs,neg_pairs

def get_order_pairs(sent,words,vocabn,vocabv,holdoutpairs):
    pos_pairs = []
    neg_pairs = []
    # or maybe instead of only putting (nv) pairs, also include corresponding (vn)?
    # or do we need to include (nn) and (vv) pairs to make it
    lems,surfs = zip(*words)
    order = {e:i for i,e in enumerate(sent.split())}
    all_pairs = list(itertools.permutations(range(len(lems)),2))
    nv_pairs = list(itertools.product(vocabn,vocabv))
    for xi,yi in all_pairs:
        if (lems[xi],lems[yi]) not in nv_pairs: continue
        if order[surfs[xi]] < order[surfs[yi]]:
            pos_pairs.append((lems[xi],lems[yi]))
        else:
            neg_pairs.append((lems[xi],lems[yi]))
    pos_pairs = [e for e in pos_pairs if e not in holdoutpairs]
    neg_pairs = [e for e in neg_pairs if e not in holdoutpairs]
    maxlen = min(len(pos_pairs),len(neg_pairs))

    for l in [pos_pairs,neg_pairs]:
        random.shuffle(l)
    pos_pairs = pos_pairs[:maxlen]
    neg_pairs = neg_pairs[:maxlen]

    return pos_pairs,neg_pairs

def get_negwords(pols):
    pos_words = pols['pos']
    neg_words = pols['neg']

    maxlen = min(len(pos_words),len(neg_words))

    for l in [pos_words,neg_words]:
        random.shuffle(l)
    pos_words = pos_words[:maxlen]
    neg_words = neg_words[:maxlen]

    return pos_words,neg_words

def get_negscopewords(polscope):
    pos_words = polscope['pos']
    neg_words = polscope['neg']

    maxlen = max(len(pos_words),len(neg_words))

    for l in [pos_words,neg_words]:
        random.shuffle(l)
    pos_words = pos_words[:maxlen]
    neg_words = neg_words[:maxlen]

    return pos_words,neg_words


def get_cont1words(words,vocabv):
    lems,surfs = zip(*words)
    in_vs = [w for w in lems if w in vocabv]
    out_vs = [w for w in vocabv if w not in in_vs]

    for l in [in_vs,out_vs]:
        random.shuffle(l)

    maxlen = min(len(in_vs),len(out_vs))

    in_vs = in_vs[:maxlen]
    out_vs = out_vs[:maxlen]

    return in_vs,out_vs

def get_veridwords(words,vocabn,vocabv,rels,pols,targetrel,holdoutpairs):
    pos_pairs = []
    neg_pairs = []
    negation_pairs = []
    for n,v in rels[targetrel]:
        if v not in pols['neg']:
            pos_pairs.append((n,v))
        else: negation_pairs.append((n,v))
    lems,surfs = zip(*words)
    in_nouns = [e for e in lems if e in vocabn]
    in_verbs = [e for e in lems if e in vocabv]
    neg_pairs = [p for p in itertools.product(in_nouns,in_verbs) if p not in pos_pairs and p not in negation_pairs]

    pos_pairs = [e for e in pos_pairs if e not in holdoutpairs]
    neg_pairs = [e for e in neg_pairs if e not in holdoutpairs]
    negation_pairs = [e for e in negation_pairs if e not in holdoutpairs]

    maxlen = min(len(pos_pairs),len(neg_pairs + negation_pairs))

    for l in [pos_pairs,neg_pairs,negation_pairs]:
        random.shuffle(l)
    pos_pairs = pos_pairs[:maxlen]
    neg_pairs = negation_pairs+neg_pairs
    neg_pairs = neg_pairs[:maxlen]

    return pos_pairs,neg_pairs

def init_labdict(voc):
    labdict = {}
    for w in voc:
        labdict[w] = {}
        labdict[w]['TRUE'] = 0
        labdict[w]['FALSE'] = 0
    for p in ['%s-%s'%(w1,w2) for w1,w2 in itertools.permutations(voc,2)]:
        labdict[p] = {}
        labdict[p]['TRUE'] = 0
        labdict[p]['FALSE'] = 0
    return labdict

def extract_items_for_set(subtask,sentwordrels,vocabn,vocabv,max_items,holdoutpairs=[],used_ids=[]):
    items = []
    ids = []
    true = 0
    targetrel = 'agent'
    for id,sent,words,rels,pols,polscope in sentwordrels:
        if id in used_ids: continue
        if len(items) >= max_items: break
        if subtask == 'order':
            pos_items,neg_items = get_order_pairs(sent,words,vocabn,vocabv,holdoutpairs)
        elif subtask == 'sr':
            pos_items,neg_items = get_rel_pairs(targetrel,rels,words,vocabn,vocabv,holdoutpairs)
        elif subtask == 'cont2':
            pos_items,neg_items = get_xypairs(words,vocabn,vocabv,holdoutpairs)
        elif subtask == 'neg':
            pos_items,neg_items = get_negwords(pols)
        elif subtask == 'negscope':
            pos_items,neg_items = get_negscopewords(polscope)
        elif subtask == 'cont1':
            pos_items,neg_items = get_cont1words(words,vocabv)
        elif subtask == 'verid':
            pos_items,neg_items = get_veridwords(words,vocabn,vocabv,rels,pols,targetrel,holdoutpairs)

        if len(pos_items) > 0:
            print(sent)
            print(pos_items)
            print(neg_items)

        true += len(pos_items)
        if len(pos_items) > 0: ids.append(id)

        if subtask in ('sr','order','cont2','verid'):
            for lab,pairlist in [('TRUE',pos_items),('FALSE',neg_items)]:
                for w1,w2 in pairlist:
                    items.append([lab,id,sent,w1,w2])

        elif subtask in ('neg','cont1'):
            for lab,wordlist in [('TRUE',pos_items),('FALSE',neg_items)]:
                for w in wordlist:
                    items.append([lab,id,sent,w])

    return items,ids,true

#takes annotdict and writes to train and test files
def make_input_pairs(annotDicts,traintestdir,subtask,traintestnums):
    sentwordrels = []
    trainnum,testnum = traintestnums
    for id in annotDicts:
        sent,aDict = annotDicts[id]
        words = get_words_from_annot(aDict)
        rels = get_rels_from_annot(aDict)
        pols = get_pols_from_annot(aDict)
        polscope = get_neg_scope_from_annot(aDict)
        sentwordrels.append((id,sent,words,rels,pols,polscope))

    testpairs = [('professor','help'),('man','call'),('woman','meet'),('scientist','recommend'),('lawyer','follow'),('doctor','help'),('student','call')]
    trainpairs = [p for p in itertools.product(vocabn,vocabv) if p not in testpairs]

    random.shuffle(sentwordrels)

    used_ids = []
    testitems,testids,testtrue = extract_items_for_set(subtask,sentwordrels,vocabn,vocabv,testnum,holdoutpairs=trainpairs,used_ids=used_ids,)
    used_ids += testids
    trainitems,trainids,traintrue = extract_items_for_set(subtask,sentwordrels,vocabn,vocabv,trainnum,holdoutpairs=testpairs,used_ids=used_ids)

    for l in [trainitems,testitems]:
        random.shuffle(l)

    with open(os.path.join(traintestdir,'train.txt'),'w') as out:
        for it in trainitems: out.write('\t'.join(it) + '\n')
    with open(os.path.join(traintestdir,'test.txt'),'w') as out:
        for it in testitems: out.write('\t'.join(it) + '\n')

    traintot = len(trainitems)
    testtot = len(testitems)
    train_true_perc = traintrue/float(traintot)
    test_true_perc = testtrue/float(testtot)

    print('Percent TRUE in train: %s'%train_true_perc)
    print('Percent TRUE in test: %s'%test_true_perc)

    print('Number of items in train: %s'%traintot)
    print('Number of items in test: %s'%testtot)
    print('Size vocab: %s'%len(vocab))
    # print(trainlabdict)
    # print(testlabdict)


#takes annotdict and writes to train and test files
def make_input_pairs_oneset(annotDicts,traintestdir,subtask,name,num_items):
    sentwordrels = []
    trainnum,testnum = traintestnums
    for id in annotDicts:
        sent,aDict = annotDicts[id]
        words = get_words_from_annot(aDict)
        rels = get_rels_from_annot(aDict)
        pols = get_pols_from_annot(aDict)
        polscope = get_neg_scope_from_annot(aDict)
        sentwordrels.append((id,sent,words,rels,pols,polscope))

    random.shuffle(sentwordrels)

    items,ids,true = extract_items_for_set(subtask,sentwordrels,vocabn,vocabv,num_items)

    random.shuffle(items)

    with open(os.path.join(traintestdir,'%s.txt'%name),'w') as out:
        for it in items: out.write('\t'.join(it) + '\n')

    tot = len(items)
    true_perc = true/float(tot)

    print('Percent TRUE in %s: %s'%(name,true_perc))

    print('Number of items in %s: %s'%(name,tot))
    # print('Size vocab: %s'%len(vocab))
    # # print(trainlabdict)
    # # print(testlabdict)
    # print(vocab)



def get_mdict(filepref):
    with open(filepref+'-dicts.pkl') as annotfile:
        mdict = pickle.load(annotfile)
    # mdict = {k:mdict[k] for k in mdict if k in get_sents(filepref+'.txt')}
    return mdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('subtask')
    parser.add_argument('trainitems',type=int)
    parser.add_argument('testitems',type=int)
    args = parser.parse_args()

    datadir = args.datadir
    subtask = args.subtask
    traintestnums = (args.trainitems,args.testitems)

    # print('NUM SENTENCES: %s'%numsents)
    has_train_test = False
    task = 'xy'
    # if subtask in ('neg','cont1'):
    #     setname = 'xy_neg'
    if subtask == 'neg':
        has_train_test = True
        setname_train = 'neg_train'
        setname_test = 'neg_test'
        setnamelist = [(setname_train,'train',args.trainitems),(setname_test,'test',args.testitems)]
    else:
        setname_all = 'xy_pos'
    traintest = os.path.join(datadir,task,subtask)
    # if not os.path.isdir(traintest): os.mkdir(traintest)
    if os.path.isdir(traintest): shutil.rmtree(traintest)
    os.mkdir(traintest)

    vocabn = ['professor','student','woman','man','doctor','scientist','lawyer']
    vocabv = ['dance','sleep','call','help','follow','recommend','meet']
    vocab = vocabn + vocabv
    with open(os.path.join(traintest,'vocab.txt'),'w') as vocfile:
        for w in vocab: vocfile.write(w + '\n')

    if not has_train_test:
        sentpref = os.path.join(datadir,task,setname_all)
        # print('Getting sents')
        # sentdict = get_sents(sentpref+'.txt')
        print('Getting mdict')
        mdict = get_mdict(sentpref)
        print('Getting input pairs')
        make_input_pairs(mdict,traintest,subtask,traintestnums)
    else:
        for setname,traintestname,item_num in setnamelist:
            sentpref = os.path.join(datadir,task,setname)
            print('Getting mdict')
            mdict = get_mdict(sentpref)
            print('Getting input pairs')
            make_input_pairs_oneset(mdict,traintest,subtask,traintestname,item_num)
