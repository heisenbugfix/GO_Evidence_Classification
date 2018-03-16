import numpy as np
import codecs #utf-8
import time
import nltk
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import word_tokenize
nltk.download('punkt')
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


model = KeyedVectors.load_word2vec_format('GO_Evidence_Classification/data/PubMed-w2v.bin', binary=True)
pmid_to_abstract = {}
with codecs.open("GO_Evidence_Classification/data/pubmed_output.txt", "r", "UTF-8") as rf:
    #ignore mesh ids for now
    #TODO: What to do with title data? (Split[2])
    print "started pmid map"
    for line in rf:
        split = line.split("\t")
        pmid_to_abstract[split[0]] = np.mean(tokenize(model," ".join([split[2], split[3]])), axis=0)[None,:]
    abstract_lengths = {}
    print "pmid file parsed"
    print "all keys added"
pickle.dump(pmid_to_abstract, open("pmid_to_wordemb.p", "wb"))
