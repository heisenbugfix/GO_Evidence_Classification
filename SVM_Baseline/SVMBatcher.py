import numpy as np
import codecs #utf-8
import time
import nltk
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import cPickle as pickle

class Batcher(object):
    def __init__(self, config, abstracts_file, data_file, batch_size = 100, dev = False, return_one_epoch=False, shuffle=True, triplet=True):
        print "started batcher"
        self.config  = config
        self.batch_size = batch_size
        self.dev = dev
        self.shuffle = shuffle
        self.return_one_epoch = return_one_epoch
        self.start_index = 0
        #self.abtract_lengths = {}
        self.pmid_to_abstract = pickle.load(open("pmid_to_wordemb.p", "rb"))
        #self.get_pubmed_map(abstracts_file)
        self.max_abstract_len = config.max_abstract_len
        self.load_data(abstracts_file, data_file)

    def get_pubmed_map(self, filepath):
        with codecs.open(filepath, "r", "UTF-8") as rf:
            #ignore mesh ids for now
            #TODO: What to do with title data? (Split[2])
            print "started pmid map"
            for line in rf:
                split = line.split("\t")
                self.pmid_to_abstract[split[0]] = self.tokenize(" ".join([split[2], split[3]]))
            self.abstract_lengths = {}
            print "pmid file parsed"
            keys = self.pmid_to_abstract.keys()
            for key in keys:
                self.abstract_lengths[key] = len(self.pmid_to_abstract.keys())
            max_len = max(self.abstract_lengths)
            print "lengths gathered"
            for key in keys:
                doc = self.pmid_to_abstract[key]
                if len(doc) >= self.config.max_abstract_len:
                    doc = doc[0:self.config.max_abstract_len]
                self.pmid_to_abstract[key] = np.asarray(doc)
            print "all keys added"



    def get_next_batch(self):
        """
        returns the next batch
        TODO(rajarshd): move the if-check outside the loop, so that conditioned is not checked every time. the conditions are suppose to be immutable.
        """
        while True:
            #print "next batch is triplet"
            #print self.sources.shape, self.positives.shape, self.negatives.shape
            if self.dev == True:
                self.batch_size = len(self.gene_ids)
            if self.start_index > self.num_examples - self.batch_size:
                if self.return_one_epoch:
                    return  # stop after returning one epoch
                self.start_index = 0
                if self.shuffle:
                    self.shuffle_data()
            else:
                num_data_returned = min(self.batch_size, self.num_examples - self.start_index)
                assert num_data_returned > 0
                end_index = self.start_index + num_data_returned
                yield self.gene_ids[self.start_index:end_index], self.abstract_encodings[self.start_index:end_index], self.evidence_labels[self.start_index:end_index], self.aspects[self.start_index:end_index]
                self.start_index = end_index

    def shuffle_data(self):
        """
        Shuffles maintaining the same order.
        """
        perm = np.random.permutation(len(self.gene_ids))
        assert len(perm) == len(self.gene_ids)
        self.abstract_encodings = self.abstract_encodings[perm]
        self.evidence_labels = self.evidence_labels[perm]
        self.aspects = self.aspects[perm]
        self.gene_ids = self.gene_ids[perm]

    def reset(self):
        self.start_index = 0

    def get_toy_data(self, filepath):
        gene_ids = []
        gene_pmids = []
        evidence_codes = []
        aspects = []
        with codecs.open(filepath, "r", "UTF-8") as rf:
            for line in rf:
                split = line.strip().split("\t")
                if split[5].split(":")[0] == "PMID":
                    #data.append([split[3], split[5].split(":")[1], split[6], split[7]])
                    gene_ids.append(split[4].split(":")[1].encode("ascii", "ignore"))
                    gene_pmids.append(split[5].split(":")[1])
                    evidence_codes.append(split[6])
                    aspects.append(split[8])
        return gene_ids, gene_pmids, evidence_codes, aspects

    def aspects_to_onehot(self, aspects):
        asp_onehots = []
        for x in aspects:
            if x == "F":
                asp_onehots.append([1, 0, 0])
            elif x == "C":
                asp_onehots.append([0, 1, 0])
            elif x == "P":
                asp_onehots.append([0, 0, 1])
        return np.asarray(asp_onehots)

    def evidence_codes_to_onehot(self, evidence_codes):
        EV_TO_ONEHOT_DICT = {}
        EV_TO_ONEHOT_DICT["EXP"] = [1] + [0] * 19
        EV_TO_ONEHOT_DICT["IDA"] = [0] * 1 + [1] + [0] * 20
        EV_TO_ONEHOT_DICT["IPI"] = [0] * 2 + [1] + [0] * 19
        EV_TO_ONEHOT_DICT["IMP"] = [0] * 3 + [1] + [0] * 18
        EV_TO_ONEHOT_DICT["IGI"] = [0] * 4 + [1] + [0] * 17
        EV_TO_ONEHOT_DICT["IEP"] = [0] * 5 + [1] + [0] * 16
        EV_TO_ONEHOT_DICT["ISS"] = [0] * 6 + [1] + [0] * 15
        EV_TO_ONEHOT_DICT["ISO"] = [0] * 7 + [1] + [0] * 14
        EV_TO_ONEHOT_DICT["ISA"] = [0] * 8 + [1] + [0] * 13
        EV_TO_ONEHOT_DICT["ISM"] = [0] * 9 + [1] + [0] * 12
        EV_TO_ONEHOT_DICT["ISC"] = [0] * 10 + [1] + [0] * 11
        EV_TO_ONEHOT_DICT["IBA"] = [0] * 11 + [1] + [0] * 10
        EV_TO_ONEHOT_DICT["IBD"] = [0] * 12 + [1] + [0] * 9
        EV_TO_ONEHOT_DICT["IKR"] = [0] * 13 + [1] + [0] * 8
        EV_TO_ONEHOT_DICT["IRD"] = [0] * 14 + [1] + [0] * 7
        EV_TO_ONEHOT_DICT["RCA"] = [0] * 15 + [1] + [0] * 6
        EV_TO_ONEHOT_DICT["TAS"] = [0] * 16 + [1] + [0] * 5
        EV_TO_ONEHOT_DICT["NAS"] = [0] * 17 + [1] + [0] * 4
        EV_TO_ONEHOT_DICT["IGC"] = [0] * 18 + [1] + [0] * 3
        EV_TO_ONEHOT_DICT["HDA"] = [0] * 19 + [1] + [0] * 2
        EV_TO_ONEHOT_DICT["IC"] = [0] * 20 + [1] + [0] * 1
        EV_TO_ONEHOT_DICT["ND"] = [0] * 21 + [1]
        evidence_onehots = np.asarray([EV_TO_ONEHOT_DICT[code] for code in evidence_codes])
        evidence_nums = np.asarray([np.argmax(i) for i in evidence_onehots])

        return evidence_onehots, evidence_nums

    def load_data(self, abstracts_file, data_file):
        #self.get_toy_data(data_file)
        #abstracts_raw = self.get_toy_data(abstracts_file)
        #abstracts_tokenized = []
        #for pm_abstract_raw in abstracts_raw:
        #    abstracts_tokenized.append(self.tokenize(pm_abstract_raw))
        #        self.abstracts = abstracts_tokenized
        #        self.abstracts = np.asarray(abstracts_tokenized)
        #self.num_examples = len(self.abstracts)
        self.gene_ids, self.gene_pmids, self.evidence_codes, self.aspects = self.get_toy_data(data_file)
        self.gene_ids = np.asarray(self.gene_ids)
        self.num_examples = len(self.gene_ids)
        self.abstract_encodings = np.asarray([self.pmid_to_abstract[i] for i in self.gene_pmids])
        self.abstract_encodings = np.squeeze(self.abstract_encodings, axis=1)
        self.evidence_onehots, self.evidence_labels = self.evidence_codes_to_onehot(self.evidence_codes)
        self.aspects = self.aspects_to_onehot(self.aspects)

    def tokenize(self, abstract):
        tokenized = word_tokenize(abstract)
        vecs = []
        for token in tokenized:
            try:
                0#vecs.append(self.model[token])
            except Exception:
                0
        return vecs

'''        with codecs.open(self.input_file, "r", "UTF-8", errors="ignore") as inp:
                sources = []
                sources_lengths = []
                positives = []
                pos_lengths = []
                negatives = []
                neg_lengths = []
                ct = -1
                for line in inp:
                    ct += 1
                    line = line.strip()
                    split = line.split("\t") #source, pos, negative
                    if len(split) < 3:
                        print(split, len(split), ct)
                    else:
                        sources.append(np.asarray(self.vocab.to_ints(split[0])))
                        sources_lengths.append([min(self.config.max_string_len,len(split[0])) - 1])
                        positives.append(np.asarray(self.vocab.to_ints(split[1])))
                        pos_lengths.append([min(self.config.max_string_len,len(split[1])) - 1])
                        negatives.append(np.asarray(self.vocab.to_ints(split[2])))
                        neg_lengths.append([min(self.config.max_string_len,len(split[2])) - 1])
                self.sources = np.asarray(sources)
                self.positives = np.asarray(positives)
                self.negatives = np.asarray(negatives)
                self.source_lens = np.asarray(sources_lengths)
                self.pos_lens = np.asarray(pos_lengths)
                self.neg_lens = np.asarray(neg_lengths)
                print("length of data", len(sources))
'''
