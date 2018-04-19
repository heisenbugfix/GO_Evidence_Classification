import numpy as np
import codecs #utf-8
import time
import nltk
import json
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
import cPickle as pickle

def tokenize(model, abstract):
    tokenized = word_tokenize(abstract)
    vecs = []
    for token in tokenized:
        try:
            vecs.append(model[token])
        except Exception:
            0
    return vecs
"""
data = json.load(open("GO_Evidence_Classification/train_dev_test/pubmed_latest.json"))
keys = data.keys()
with codecs.open("GO_Evidence_Classification/data/pubmed_tiny.txt", "w+", "UTF-8") as wf:
    for i in range(0, 5):
        split = data[keys[i]]["abstract"].split("\t")
        if len(split) > 1:
            wf.write(split[1])
            wf.write("\n")
        elif len(split) == 0:
            wf.write(split[1])
            wf.write("\n")
        else:
            print 0
"""
model = KeyedVectors.load_word2vec_format('GO_Evidence_Classification/data/PubMed-w2v.bin', binary=True)
pmid_to_abstract = {}
with codecs.open("GO_Evidence_Classification/data/pubmed_output.txt", "r", "UTF-8") as rf:
    #ignore mesh ids for now
    #TODO: What to do with title data? (Split[2])
    print "started pmid map"
    for key in data.keys():
        split = data[key]["abstract"].split("\t")
        if len(split) > 1:
            pmid_to_abstract[key] = np.mean(tokenize(model," ".join([split[0], split[1]])), axis=0)[None,:]
        elif len(split) == 0:
            pmid_to_abstract[key] = np.mean(tokenize(model," ".join([split[0], split[1]])), axis=0)[None,:]
        else:
            print 0
    abstract_lengths = {}
    print "pmid file parsed"
    print "all keys added"
pickle.dump(pmid_to_abstract, open("pmid_latest_to_wordemb.p", "wb"))
