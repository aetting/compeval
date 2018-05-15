from __future__ import print_function

import random
import pickle
import os
import gzip
import math
import ast
import operator
import argparse
import numpy as np
from collections import Counter
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

def load_word_embeddings(path, vocab):
    embeddings = Counter()
    if path.endswith('.gz'):
        hndl = gzip.open(path, 'rU')
    else:
        hndl = open(path, 'rU')
#         hndl = open(path)
    for wordvec in hndl:
        parts = wordvec[:-1].split()
        if parts[0] in vocab:
            embeddings[parts[0]] = [float(v) for v in parts[1:]]
    hndl.close()
    print('loaded embeddings file %s' % path)
    return embeddings

def load_embdict(embfile):
    embdict = {}
    if embfile.endswith('.gz'):
        file = gzip.open(embfile, 'rU')
    else:
        file = open(embfile, 'rU')
    for line in file:
        if line[0] == '{' or len(line.strip()) < 1: continue
        id,sent,emb = line.strip().split('\t')
        embdict[id] = (sent,emb)
    file.close()
    return embdict

def read_set(setfilename,inds=[0,1,2,3]):
    with open(setfilename) as setfile:
        prevlinestart = None
        linelists = []
        linelist = None
        for line in setfile.readlines():
            if len(line.strip()) == 0 or line[0] == 'N': continue
            if line[0] == '{' and line[0] != prevlinestart:
                if linelist: linelists.append(linelist)
                linelist = []
            linelist.append(line.strip())
            prevlinestart = line[0]
        linelists.append(linelist)
        sentdict = {}
        for l in linelists:
            l1 = l[inds[0]]
            l2 = l[inds[1]]
            l3 = l[inds[2]]
            l4 = l[inds[3]]
            sents = l[4:]
            if l1 not in sentdict: sentdict[l1] = {}
            if l2 not in sentdict[l1]: sentdict[l1][l2] = {}
            if l3 not in sentdict[l1][l2]: sentdict[l1][l2][l3] = {}
            sentdict[l1][l2][l3][l4] = sents
    return sentdict

def get_xyclass_ftrs(file_path,sentembdict,wordembdict,word2id):
    X_loc = []
    X_emb = []
    X_randprobe = []
    X_randsent = []
    y = []
    ids = []
    sents = []
    allprobes = []
    cnt = 0
    probeemb = []
    with open(file_path) as fh:
        for ln in fh.readlines():
            label, id , sent = ln.strip().split('\t')[:3]
            probes = ln.strip().split('\t')[3:]
            sent2,sentemb = sentembdict[id]
            if sent != sent2:
                raise Exception('NOT EQUAL: %s || %s'%(sent,sent2))
            sentemb = [float(v) for v in sentemb.split()]
            # print(sent)
            # print(id)
            # print(sentemb[:5])

            probeemb = []
            probe1hot = []
            randprobe = []

            for probe in probes:
                # probeemb += wordembdict[probe]

                probeid = word2id[probe]
                sg_probe1hot = [0] * len(word2id)
                sg_probe1hot[probeid] = 1
                probe1hot += sg_probe1hot

                sg_randprobe = [0] * len(word2id)
                randid = np.random.randint(len(word2id))
                sg_randprobe[randid] = 1
                randprobe += sg_randprobe
#             randprobe = list(len(word2id))

            randsent = list(np.random.randn(len(sentemb)))

            y.append(label)
            locftrs = sentemb + probe1hot
            embftrs = sentemb + probeemb
            randpftrs = sentemb + randprobe
            randsftrs = randsent + probe1hot

            X_loc.append(locftrs)
            X_emb.append(embftrs)
            X_randprobe.append(randpftrs)
            X_randsent.append(randsftrs)

            ids.append(id)
            sents.append(sent)
            allprobes.append(probes)
            cnt += 1
#         X_shuf = [e for e in enumerate(X)]
#         random.shuffle(X_shuf)
#         shuf_inds,X_shuf = zip(*X_shuf)
#         X = X_shuf
#         y = [y[i] for i in shuf_inds]
#         ids = [ids[i] for i in shuf_inds]
#         sents = [sents[i] for i in shuf_inds]
    return (np.asarray(X_loc),np.asarray(X_emb),np.asarray(X_randprobe),np.asarray(X_randsent),np.asarray(y),ids,sents,allprobes)

def xy_classify(train_X,train_y,test_X,test_y,report_f,test_ids,test_sents,test_probes,cltype):

    #encoding class labels
    le = preprocessing.LabelEncoder()
    le.fit(train_y)
    train_y = np.asarray(le.transform(train_y))

    if cltype == 'lreg':
        print("Training and tuning logistic regression\n")
        tuned_parameters = [{'C': [0.00001,0.001,0.01,1, 10, 100, 1000]}]
        clf = GridSearchCV(LogisticRegression(penalty='l2'), tuned_parameters, cv=5, scoring='f1_weighted', verbose=1)  # tuning using 5-fold cross validation
    elif cltype == 'mlp':
        # print("Training and tuning MLP\n")
        # f1_average = 'micro'
        hidden_layer_sizes =(train_X.shape[1]) #***so here we can just repeat that argument for num of layers we want
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation='relu',max_iter=700)

    clf.fit(train_X, train_y)


    if cltype == 'lreg':
        print("Best parameters set:")
        print(clf.best_params_)

    # print("Classification Report:")

    y_pred = le.inverse_transform(clf.predict(test_X))
    dist_pred = clf.predict_proba(test_X)


    report = classification_report(test_y, y_pred)
    acc = accuracy_score(test_y, y_pred)
    accuracy = 'classification accuracy: %0.4f' % acc
    # print(report)
    # print(accuracy)
    with open(report_f,'w') as out:
        out.write('REPORT\n')
        out.write(report + '\n\n')
        out.write(accuracy + '\n\n')
        out.write('\t'.join(['ID','PRED','TRUTH','CORR','SENT'])+'\n\n')
        for i in range(len(test_y)):
            pred = y_pred[i]
            truelab = test_y[i]
            id = test_ids[i]
            sent = test_sents[i]
            probes = test_probes[i]
            if pred == truelab:
                predcorr = '1'
            else:
                predcorr = '0'

            out.write('\t'.join([id,pred,truelab,predcorr,sent] + probes)+'\n')
    return report,accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cltype')
    parser.add_argument('subtask')
    parser.add_argument('datadir')
    parser.add_argument('embdir')
    parser.add_argument('wordembs')
    parser.add_argument('desc')
    parser.add_argument('--embmeths',nargs='+')
    args = parser.parse_args()

    cltype = args.cltype
    subtask = args.subtask
    datadir = args.datadir
    embdir = args.embdir
    wordembs = args.wordembs
    wordembs = args.desc
    embmeths = args.embmeths

    s = ''
    s += 'TRAIN/TEST: %s\n'%subtask
    s += 'CLTYPE: %s\n'%cltype

    task = 'xy'
    if subtask in ('cont1','cont1simp'):
        setname = 'xy_neg'
    elif subtask == 'neg':
        setname = 'neg'
    else:
        setname = 'xy_pos'
    sentpref = os.path.join(datadir,task,setname)
    traintest = os.path.join(datadir,task,subtask+'-'+desc)

    vocab = []
    with open(os.path.join(traintest,'vocab.txt')) as vocfile:
        for line in vocfile:
            vocab.append(line.strip())
    word2id = {}
    for i,word in enumerate(vocab): word2id[word] = i

    for method in embmeths:
        s += 'METHOD: %s'%method

        ttmethdir = os.path.join(traintest,method)
        if not os.path.isdir(ttmethdir): os.mkdir(ttmethdir)

        sent_embeddings_path = os.path.join(embdir,'%s-embs'%task,method,'%s.txt'%setname)

        sentembdict = load_embdict(sent_embeddings_path)
        wordembdict = None
        # wordembdict = load_word_embeddings(wordembs,vocab)

        X_train_loc, X_train_emb,X_train_randp,X_train_rands,y_train,_,_,_ = get_xyclass_ftrs(os.path.join(traintest,'train.txt'),sentembdict,wordembdict,word2id)
        X_test_loc, X_test_emb,X_test_randp,X_test_rands,y_test,test_ids,test_sents,test_probes = get_xyclass_ftrs(os.path.join(traintest,'test.txt'),sentembdict,wordembdict,word2id)


        s += '\nLOCALIST CLASSIFICATION\n'
        rep,acc = xy_classify(X_train_loc,y_train,X_test_loc,y_test,os.path.join(ttmethdir,'loc_results.txt'),test_ids,test_sents,test_probes,cltype)
        s += '%s\n%s\n'%(rep,acc)
        # s += '\nEMBEDDING CLASSIFICATION\n'
        # rep,acc = xy_classify(X_train_emb,y_train,X_test_emb,y_test,os.path.join(ttmethdir,'emb_results.txt'),test_ids,test_sents,cltype)
        # s += '%s\n%s\n'%(rep,acc)
        s += '\nRANDOM PROBEVEC CLASSIFICATION\n'
        rep,acc = xy_classify(X_train_randp,y_train,X_test_randp,y_test,os.path.join(ttmethdir,'randp_results.txt'),test_ids,test_sents,test_probes,cltype)
        s += '%s\n%s\n'%(rep,acc)
        s += '\nRANDOM SENTVEC CLASSIFICATION\n'
        rep,acc = xy_classify(X_train_rands,y_train,X_test_rands,y_test,os.path.join(ttmethdir,'rands_results.txt'),test_ids,test_sents,test_probes,cltype)
        s += '%s\n%s\n'%(rep,acc)

        print(s)
        with open(os.path.join(ttmethdir,'full_results.txt'),'w') as out:
            out.write(s)
