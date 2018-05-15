from __future__ import division
import sys
import numpy as np
from collections import Counter
import os
import collections
import string
import math
import gzip
import torch
import time
import argparse


# sdae_path = '/Users/allysonettinger/Desktop/meaning_cc/composition/dataset/modelsforcomp/sdae/sdae_models'
# skip_thought_path = '/Users/allysonettinger/Desktop/meaning_cc/composition/dataset/modelsforcomp/skip-thoughts'

sdae_path = '/fs/clip-cognitive/composition/NAACL2018/trained_models/SDAE'
skip_thought_path = '/fs/clip-cognitive/composition/NAACL2018/trained_models/SkipThought'

sdae_model = os.path.join(sdae_path,'model_rnn_book_noemb_possiblebest.npz')
sdae_dictionary = os.path.join(sdae_path,'wiki_dictionary.pkl')

# word_embeddings_path = '/Users/allysonettinger/Desktop/meaning_cc/modeling/models/pretrained_embeddings/glove/glove-Wik-Gig/glove.6B.50d.txt.gz'
# word_embd_size = 50
word_embeddings_path = '/fs/clip-cognitive/composition/NAACL2018/trained_models/W2V/tbc-w2v.txt.gz'
word_embd_size = 2400

# paragram_phrases_path = '/fs/clip-summarization/composition/trained_models/words/en.paragram-phrase-XXL.txt'
# paragram_sl999_path = '/fs/clip-summarization/composition/trained_models/words/en.paragram_300_sl999.txt'

infersent_glove_path = '/fs/clip-summarization/Composition-Eval/infersent/InferSent/dataset/GloVe/glove.840B.300d.txt'
infersent_snli2400_model_path = '/fs/clip-cognitive/composition/NAACL2018/trained_models/InferSent/model2400.pickle.encoder'
infersent_snli4800_model_path = '/fs/clip-cognitive/composition/NAACL2018/trained_models/InferSent/model4800.pickle.encoder'

paragram_embd_size = 300

sys.path.append(sdae_path)
sys.path.append(skip_thought_path)

#import get_desent
#import skipthoughts
"""
skipthought_embd_size = 4800 #  consisting of the concatenation of the vectors from uni-skip and bi-skip

sys.path.insert(0, skip_thought_path)
"""

class SkipThought_Embedding(object):
    def __init__(self, type_):
        import skipthoughts
        self.model = skipthoughts.load_model()
        self.type = type_
#todo: do all sentences together
    def embd_sent(self,sent):
        import skipthoughts
        if self.type == 'uni':
            return skipthoughts.encode(self.model, [sent])[0][0:2400]
        elif self.type == 'bi':
            return skipthoughts.encode(self.model, [sent])[0][2400:]
        else:
            return skipthoughts.encode(self.model, [sent])[0]
    def embd_multiple_sents(self,sents):
        import skipthoughts
        if self.type == 'combined':
            return skipthoughts.encode(self.model, sents)
        elif self.type == 'uni':
            ix_from = 0
            ix_to = 2400
        else:
            ix_from = 2400
            ix_to = 4800
        return np.asarray([v[ix_from:ix_to] for v in skipthoughts.encode(self.model, sents)])


class Bow_Embedding(object):

    def load_embeddings(self,path, vocab):
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

    def corpus_to_vocab(self,corpuslist):
        vocab = set()
        for infile in corpuslist:
            with open(infile) as inh:
                for ln in inh:
                    if len(ln.strip()) < 1 or ln.startswith('{') or ln.startswith('Number') : continue
    #                 for w in ln.strip().split('\t')[2].split(): vocab.add(w)
                    for w in ln.strip().split('\t')[1].split(): vocab.add(w)
        return vocab
    def __init__(self,embd_path,fallback_embd_path,embd_size, corpuslist=None):
        self.embd_size = embd_size
        if corpuslist is not None:
            vocab = self.corpus_to_vocab(corpuslist)
            self.embds = self.load_embeddings(embd_path, vocab)
        else:
            self.embds = self.load_embeddings(embd_path)
        self.fallback_embds = self.load_embeddings(fallback_embd_path) if fallback_embd_path is not None else None

    def embd_sent(self, sent):
        embedding = [0]*self.embd_size
        numwords = 0
        for w in sent.split():
            if self.embds[w] != 0:
                numwords += 1
                embedding = np.add(embedding,self.embds[w])
            elif self.fallback_embds is not None and self.fallback_embds[w] !=0:
                numwords += 1
                embedding = np.add(embedding,self.fallback_embds[w])
            else:
                print('**NOT FOUND '+w+' **')
        if numwords == 0: return embedding

        embedding = np.divide(embedding,numwords)
        return embedding

    def embd_multiple_sents(self,sents):
        output = []
        for sent in sents:
            output.append(self.embd_sent(sent))
        return output

class SDAE_Embedding(object):

    def __init__(self,model,dictionary):
        import get_desent as gd
        self.model = model
        self.dictionary = dictionary
        self.model_options, self.tparams, self.f_ctx, self.dictionary, self.wv_embs = gd.load_model(self.model, self.dictionary)

    def embd_multiple_sents(self,sents):
        import get_desent as gd
        embs = gd.get_desent(self.model_options, self.f_ctx, sents, self.dictionary, wv_embs=self.wv_embs)
        return embs

class InferSent_Embeddings(object):
    def __init__(self, model_path, glove_path, vocab_size=100000):
        self.model = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.set_glove_path(glove_path)
        self.model.build_vocab_k_words(K=vocab_size)

    def embd_multiple_sents(self, sentences):
        # gpu mode : ~1000 sentences/s
        # cpu mode : ~75 sentences/s
        return self.model.encode(sentences, bsize=128, tokenize=False, verbose=True)

# def embd_file_inbatches_old(embder,in_file, out_file, batchsize=500):
#     print 'computing embeddings of %s and storing the results in %s' % (in_file, out_file)
#     outh = open(out_file,'w')
#     inh = open(in_file)
#     sents_batch = []
#     labels_batch = []
#     ids_batch = []
#     batchcount = 0
#
#     for ln in inh.readlines():
#         lbl, id, sent = ln.strip().split('\t')
#         sents_batch.append(sent)
#         labels_batch.append(lbl)
#         ids_batch.append(id)
#         batchcount += 1
#         if batchcount == batchsize:
#             #start_time = time.time()
#             embds = embder.embd_multiple_sents(sents_batch)
#             #print("--- %s seconds ---" % (time.time() - start_time))
#             ix = 0
#             for e in embds:
#                 s = ' '.join([str(v) for v in e])+'\t'+labels_batch[ix]+'\t'+ids_batch[ix]+'\n'
#                 outh.write(' '.join([str(v) for v in e])+'\t'+labels_batch[ix]+'\t'+ids_batch[ix]+'\n')
#                 ix += 1
#
#             outh.flush()
#             sents_batch = []
#             labels_batch = []
#             ids_batch = []
#             batchcount = 0
#
#     embds = embder.embd_multiple_sents(sents_batch)
#     ix = 0
#     for e in embds:
#         outh.write(' '.join([str(v) for v in e])+'\t'+labels_batch[ix]+'\t'+ids_batch[ix]+'\n')
#         ix += 1
#
#     outh.close()
#     inh.close()

def embd_file_inbatches(embder,in_file_list, out_file, batchsize=500):
    outh = open(out_file,'w')

    for in_file in in_file_list:
        with open(in_file) as inh:
            print('computing embeddings of %s and storing the results in %s' % (in_file, out_file))
            sent_batch_list = []
            ids_batch_list = []
            batchcount = 0

            catlist = []
            indexlist = []

        #     sentind = 0

            cat = []
            sent_batch = []
            ids_batch = []
            prevlinestart = None
            for ln in inh.readlines():
                if len(ln.strip()) < 1 or ln.startswith('Number'): continue

                if ln.startswith('{'):
                    if ln[0] != prevlinestart:
                        if prevlinestart:
                            catlist.append(cat)
        #                     indexlist.append(indices)
                            sent_batch_list.append(sent_batch)
                            ids_batch_list.append(ids_batch)
                            cat = []
                            sent_batch = []
                            ids_batch = []
        #                     indices = []
                            cat.append(ln)
                        else:
                            cat.append(ln)
                    elif ln[0] == prevlinestart:
                        cat.append(ln)
                else:
                    id, sent = ln.strip().split('\t')
                    sent_batch.append(sent)
                    ids_batch.append(id)
        #             indices.append(sentind)
        #             sentind += 1
                prevlinestart = ln[0]
            catlist.append(cat)
            sent_batch_list.append(sent_batch)
            ids_batch_list.append(ids_batch)
        #     indexlist.append(indices)
            embed_batch_sz = 500
            for cat,sents,ids in zip(catlist,sent_batch_list,ids_batch_list):
                for catlevel in cat:
        #             if sys.version_info[0] < 3:
                    outh.write(catlevel)
        #             else:
        #                 outh.write(catlevel.encode('utf-8'))
                for ii in range(0,len(sents),embed_batch_sz):
                    sent_minibatch = sents[ii:ii+embed_batch_sz]
                    id_minibatch = ids[ii:ii+embed_batch_sz]
                    embds = embder.embd_multiple_sents(sent_minibatch)
                    for ei,emb in enumerate(embds):
                        linestring = id_minibatch[ei]+'\t'+ sent_minibatch[ei]+'\t'+' '.join([str(v) for v in emb])+'\n'
        #                 if sys.version_info[0] < 3:
                        outh.write(linestring)
        #                 else:
        #                     outh.write(linestring.encode('utf-8'))
                        outh.flush()

    outh.close()



def embd_main(input_file_list,output_file,embd_method):
    # inout = zip(input_file_list,output_file_list)
    if embd_method == 'skipthoughts':
        st = SkipThought_Embedding('combined')
        embd_file_inbatches(st,input_file_list, output_file)
    elif embd_method == 'skipthoughts-bi':
        st = SkipThought_Embedding('bi')
        embd_file_inbatches(st,input_file_list, output_file)
    elif embd_method == 'skipthoughts-uni':
        st = SkipThought_Embedding('uni')
        embd_file_inbatches(st,input_file_list, output_file)
    elif embd_method == 'bow':
        bow = Bow_Embedding(word_embeddings_path, None, word_embd_size, input_file_list)
        embd_file_inbatches(bow,input_file_list, output_file)
    elif embd_method == 'sdae':
        sd = SDAE_Embedding(sdae_model,sdae_dictionary)
        embd_file_inbatches(sd,input_file_list, output_file)
    elif embd_method == 'infersent':
        infersent = InferSent_Embeddings(infersent_snli2400_model_path,infersent_glove_path)
        embd_file_inbatches(infersent,input_file_list, output_file)
    elif embd_method == 'infersent48':
        infersent = InferSent_Embeddings(infersent_snli4800_model_path,infersent_glove_path)
        embd_file_inbatches(infersent,input_file_list, output_file)

def all_methods(task,setname,datadir,embdir,embd_methods,traintest=True):
    taskembdir = os.path.join(embdir,'%s-embs'%task)
    if not os.path.isdir(taskembdir): os.mkdir(taskembdir)
    if traintest:
        pref = os.path.join(datadir,task,setname)
        inputlist = [pref+'_train.txt',pref+'_test.txt']
    else:
        pref = os.path.join(datadir,task,setname)
        inputlist = [pref+'.txt']
    for method in embd_methods:
        embmethdir = os.path.join(taskembdir,method)
        if not os.path.isdir(embmethdir): os.mkdir(embmethdir)
        outputfile = os.path.join(embmethdir,setname+'.txt')
        embd_main(inputlist,outputfile,method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task')
    parser.add_argument('setname')
    parser.add_argument('datadir')
    parser.add_argument('embdir')
    parser.add_argument('--embmeths',nargs='+')
    args = parser.parse_args()

    all_methods(args.task,args.setname,args.datadir,args.embdir,args.embmeths,traintest=False)


#     embd_main('/fs/clip-cognitive/composition/sets/profhelp_pos.txt','/fs/clip-cognitive/composition/sets/profhelppostestembs','infersent')
