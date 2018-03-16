import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss
import gensim
import codecs
import cPickle as pickle
from random import shuffle

def write(filepath):
    data = []
    ct = 0
    already_seen_keys = []
    pmid_to_wordemb = pickle.load(open("pmid_to_wordemb.p", "rb"))
    lines = []
    with codecs.open(filepath, "r") as rf:
        for line in rf:
            split = line.strip().split("\t")
            split_pmid = split[5].split(":")
            if split_pmid[0] == "PMID" and split_pmid[1] in pmid_to_wordemb.keys() and split_pmid[1] not in already_seen_keys:
                lines.append(split)
        with codecs.open("goa_full_70_train.gaf", "w+") as wf:
            with codecs.open("goa_full_10_dev.gaf", "w+") as devf:
                with codecs.open("goa_full_20_test.gaf", "w+") as testf:
                    shuffle(lines)
                    for line in lines:
                        split = line#.strip().split("\t")
                        split_pmid = split[5].split(":")
                        if split_pmid[0] == "PMID" and split_pmid[1] in pmid_to_wordemb.keys() and split_pmid[1] not in already_seen_keys:
                            if ct < 11088:
                                wf.write("\t".join(split) + "\n")
                            elif ct < 12672:
                                devf.write("\t".join(split)+ "\n")
                            else:
                                testf.write("\t".join(split)+ "\n")
                            ct += 1
                            already_seen_keys.append(split_pmid[1])
                    wf.flush()
                    wf.close()
                    devf.flush()
                    devf.close()
                    testf.flush()
                    testf.close()

def count(filepath):
    data = []
    ct = 0
    already_seen_keys = []
    pmid_to_wordemb = pickle.load(open("pmid_to_wordemb.p", "rb"))
    with codecs.open(filepath, "r") as rf:
        for line in rf:
            split = line.strip().split("\t")
            split_pmid = split[5].split(":")
            if split_pmid[0] == "PMID" and split_pmid[1] in pmid_to_wordemb.keys() and split_pmid[1] not in already_seen_keys:
                ct +=1
                already_seen_keys.append(split_pmid[1])

        print ct

#count("goa_uniprot_all_noiea.gaf")
write("goa_uniprot_all_noiea.gaf")


#Import word vectors
#Make model bidirectional LSTM followed by feedforward layer

'''

def get_toy_data(filepath):
    data = []
    with open(filepath, "r") as rf:
        for line in rf:
            split = line.strip().split("\t")
            if split[5].split(":")[0] == "PMID":
                data.append([split[3], split[5].split(":")[1], split[6], split[7]])
        """
        split[0] = DB
        split[1] = DB Code
        split[2] = DB Object Symbol
        split[3] = "Qualifier" (probably not useful)
        split[4] = Go ID
        split[5] = DB Ref, potentially PubMed ID
        split[6] = Evidence code
        split[7] = "Evidence with" (optional, required for some evidence codes)
        split[8] = P (biological process) or C (cellular component) or F (molecular function)
        split[9] = Gene full name
        split[10] = Gene synonyms
        split[11] = gene object type
        """
    return data


model = gensim.models.KeyedVectors.load_word2vec_format('GO_Evidence_Classification/data/PubMed-w2v.bin', binary=True)
get_pubmed_map("GO_Evidence_Classification/data/pubmed_output.txt")
get_toy_data("goa_data_small.gaf")

doc = []
for i in pmid_to_abstract["24823393"][0]:
    try:
        doc.append(model[i])
    except Exception:
        0
for i in pmid_to_abstract["24823393"][1]:
    try:
        doc.append(model[i])
    except Exception:
        0
print np.asarray(doc).shape

print pmid_to_abstract["24823393"]
'''
