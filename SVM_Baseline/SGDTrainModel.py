import datetime
import os
import sys
import numpy as np
from shutil import copyfile, copytree

import torch
import torch.optim as optim

from lstm_baseline_model import LSTMModel
from SVMBatcher import Batcher
from Config import Config
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support



def train_model(config, dataset_name,model_name):
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
    batcher = Batcher(config, "GO_Evidence_Classification/data/pubmed_output.txt", "goa_full_70_train.gaf", dev=True)
    dev_batcher = Batcher(config, "GO_Evidence_Classification/data/pubmed_output.txt", "goa_full_10_dev.gaf", dev=True)
    for dev_gene_ids, dev_abstracts, dev_labels, dev_aspects in dev_batcher.get_next_batch():
        break
    print "batcher created"
    #model.cuda()

    print("Begin Training")
    sys.stdout.flush()

    # Training loop
    counter = 0
    for gene_ids, abstracts, labels, aspects in batcher.get_next_batch():
        counter = counter + 1
        model.partial_fit(abstracts, labels, classes=[i for i in range(0, 21)])

        if counter % 100 == 0:
            #predict_dev
            predicts = model.predict(dev_abstracts)
            #print("Processed {} batches, Accuracy of batch {}: {}.".format(counter, counter, predict_dev(predicts, dev_labels)))
            print("Processed {} batches, Accuracy of batch {}: {}.".format(counter, counter, precision_recall_fscore_support(predicts, dev_labels, average="weighted")))
            sys.stdout.flush()
        #if counter == config.num_minibatches:
        #    break

def predict_dev(predicted_labels, labels):
    num_correct = 0
    for i in range(0, len(labels)):
        if predicted_labels[i] != labels[i]:
            num_correct += 1
    return num_correct * 1.0 / len(labels)

if __name__ == "__main__":

    # Set up the config
    config = Config(sys.argv[1])
    dataset_name = sys.argv[2]
    model_name = sys.argv[3]
    train_model(config, dataset_name,model_name)
