import datetime
import os
import sys
import numpy as np
from shutil import copyfile, copytree
import random

import torch
import torch.optim as optim

from lstm_baseline_model import LSTMModel
from SVMBatcher import Batcher
from Config import Config
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support



def train_model(config, dataset_name,model_name, f1_type):
    """ Train based on the given config, model / dataset

    :param config: config object
    :param dataset_name: name of dataset
    :param model_name: name of model
    :return:
    """
    now = datetime.datetime.now()
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                            now.second)
    experiment_out_dir = os.path.join("exp_out", dataset_name, model_name, ts)

    # Set up output dir
    output_dir = experiment_out_dir
    os.makedirs(output_dir)

    torch.manual_seed(config.random_seed)
    print "creating model"
    model = SGDClassifier()
    print "model created"

    # Set up batcher
    print "creating batcher"
    #batcher = Batcher(config, "GO_Evidence_Classification/data/pubmed_output.txt", "GO_Evidence_Classification/train_dev_test/xtrain_unique.txt", dev=True)
    batcher = Batcher(config, "blah", "GO_Evidence_Classification/train_dev_test/xtrain_all.txt", dev=True)
    dev_batcher = Batcher(config, "blah", "GO_Evidence_Classification/train_dev_test/xtest_all.txt", dev=True)
    for gene_ids, abstracts, labels, aspects in batcher.get_next_batch():
        break
    labels_ct = [0] * 22
    for i in labels:
        if i > 22:
            print i
        labels_ct[i] += 1
    for i in range(0, len(labels_ct)):
        labels_ct[i] = labels_ct[i] * 1.0 / len(labels)
    #dev_batcher = Batcher(config, "GO_Evidence_Classification/data/pubmed_output.txt", "GO_Evidence_Classification/train_dev_test/xtest_unique.txt", dev=True)
    for dev_gene_ids, dev_abstracts, dev_labels, dev_aspects in dev_batcher.get_next_batch():
        break
    print "batcher created"
    #model.cuda()

    print("Begin Training")
    sys.stdout.flush()

    # Training loop
    counter = 0
    dev_predicts = [np.random.choice(range(0, 22), p=labels_ct) for i in range(0, len(dev_labels))]
    print(precision_recall_fscore_support(dev_predicts, dev_labels, average=f1_type))

if __name__ == "__main__":

    # Set up the config
    config = Config(sys.argv[1])
    dataset_name = sys.argv[2]
    model_name = sys.argv[3]
    f1_type = sys.argv[4]
    train_model(config, dataset_name,model_name, f1_type)
