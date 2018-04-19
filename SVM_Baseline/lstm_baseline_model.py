import numpy as np
import torch
import torch.nn as nn
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import gensim
import torchwordemb

class LSTMModel(torch.nn.Module):
    def __init__(self,config,lstm_dim=100):
        super(LSTMModel, self).__init__()
        WORD2VEC_EMB_DIM = 200 #TODO: FIX
        self.config = config
        lstm_bidirectional = False
        #self.model = gensim.models.KeyedVectors.load_word2vec_format('GO_Evidence_Classification/data/PubMed-w2v.bin', binary=True)
        self.vocab, vec = torchwordemb.load_word2vec_bin('GO_Evidence_Classification/data/PubMed-w2v.bin')
        vec = torch.cat([vec, torch.zeros((1, 200))], dim=0)
        self.emb = nn.Embedding(vec.size()[0], vec.size(1))
        self.emb.weight = nn.Parameter(vec)
        #self.emb = self.emb_layer(self.model, trainable=False)
        self.lstm = nn.LSTM(WORD2VEC_EMB_DIM, lstm_dim, 1, bidirectional = lstm_bidirectional, batch_first = True)
        self.feedforward1 = nn.Parameter(torch.randn(lstm_dim, 20),requires_grad=True)
        self.loss = CrossEntropyLoss()
        self.batch_size = config.batch_size
        self.num_directions = 1
        self.h0 = Variable(torch.zeros(self.num_directions, self.batch_size, lstm_dim), requires_grad=False)
        self.c0 = Variable(torch.zeros(self.num_directions, self.batch_size, lstm_dim), requires_grad=False)



#TODO: Tokenize  word embeddings by converting strings to ints using word2vec.keys() as a vocab map.

    def emb_layer(self, keyed_vectors, trainable=False):
        """Create an Embedding layer from the supplied gensim keyed_vectors."""
        emb_weights = Tensor(keyed_vectors.syn0)
        emb = nn.Embedding(*emb_weights.shape)
        emb.weight = nn.Parameter(emb_weights)
        emb.weight.requires_grad = trainable
        print emb
        return emb


    def compute_loss(self,abstracts, abstract_lengths,  labels):
        """ Compute the loss (CE) for a batch of examples
        :return:
        """
        print abstracts.shape
        abst = torch.LongTensor(abstracts)
        embedding_output = self.embed(abst, abstract_lengths)
        loss = self.loss(embedding_output,labels)
        return loss

    def embed(self,abstracts, abstract_lengths):
        """
        :param string_mat: Batch_size by max_string_len
        :return: batch_size by embedding dim
        """
        emb_abs = self.emb(Variable(abstracts))
        print emb_abs
        string_len = torch.from_numpy(abstract_lengths)#.cuda()
        lengths = Variable(torch.FloatTensor(abstract_lengths).squeeze() * self.batch_size)
        final_emb, final_hn = self.lstm(emb_abs, (self.h0, self.c0))
        reshaped = final_emb.resize(self.batch_size * self.config.max_abstract_len, self.config.lstm_hidden_size * self.num_directions)
        #offset = Variable(torch.cuda.LongTensor([x for x in range(0, self.config.batch_size)]))
        offset = Variable(torch.LongTensor([x for x in range(0, self.config.batch_size)]))
        lookup = lengths + offset
        last_state = reshaped.index_select(0,lookup)
        return self.feedforward1(last_state)
