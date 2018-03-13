import datetime
import os
import sys
from shutil import copyfile, copytree

import torch
import torch.optim as optim

from lstm_baseline_model import LSTMModel
from Batcher import Batcher
from Config import Config


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
    model = LSTMModel(config)
    print "model created"

    # Set up batcher
    print "creating batcher"
    batcher = Batcher(config, "GO_Evidence_Classification/data/pubmed_output.txt", "goa_700_train.gaf", model.vocab)
    print "batcher created"
    #model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001,
                           weight_decay=0)

    # Stats
    best_map = 0
    counter = 0
    sum_loss = 0.0

    print("Begin Training")
    sys.stdout.flush()

    # Training loop
    for gene_ids, abstracts, abstract_lengths, labels, aspects in batcher.get_next_batch():
        counter = counter + 1
        optimizer.zero_grad()

        loss = model.compute_loss(abstracts, abstract_lengths, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), config.clip) #TODO: do i still need this?
        optimizer.step()

        if counter % 100 == 0:
            # print("p-n:{}".format(model.print_loss(source,pos,neg,source_len,pos_len,neg_len)))
            this_loss = loss.cpu().data.numpy()[0]
            sum_loss += this_loss
            print("Processed {} batches, Loss of batch {}: {}. Average loss: {}".format(counter, counter, this_loss,
                                                                                        sum_loss / (counter / 100)))
            sys.stdout.flush()

        #if counter % config.eval_every == 0:
        #    dev_batcher = Batcher("dev") #todo: put in dev set
            '''scores = ""
            map_score = float(eval_map_file(prediction_filename))
            hits_at_1 = float(eval_hits_at_k_file(prediction_filename, 1))
            hits_at_10 = float(eval_hits_at_k_file(prediction_filename, 10))
            hits_at_50 = float(eval_hits_at_k_file(prediction_filename, 50))
            scores += "{}\t{}\t{}\tMAP\t{}\n".format(config.model_name, config.dataset_name, counter, map_score)
            scores += "{}\t{}\t{}\tHits@1\t{}\n".format(config.model_name, config.dataset_name, counter, hits_at_1)
            scores += "{}\t{}\t{}\tHits@10\t{}\n".format(config.model_name, config.dataset_name, counter, hits_at_10)
            scores += "{}\t{}\t{}\tHits@50\t{}\n".format(config.model_name, config.dataset_name, counter, hits_at_50)
            print(scores)
            score_obj = {"samples": counter, "map": map_score, "hits_at_1": hits_at_1, "hits_at_10": hits_at_10, "hits_at_50": hits_at_50,
                         "config": config.__dict__}
            print(score_obj)
            #save_dict_to_json(score_obj, os.path.join(output_dir, 'dev.scores.{}.json'.format(counter)))
            with open(os.path.join(output_dir, 'dev.scores.{}.tsv'.format(counter)), 'w') as fout:
                fout.write(scores)
            if map_score > best_map:
                print("New best MAP!")
                print("Saving Model.....")
                torch.save(model, os.path.join(output_dir,
                                               'model_{}_{}_{}.torch'.format(config.model_name, config.dataset_name,
                                                                             counter)))
                best_map = map_score'''
            sys.stdout.flush()
        #if counter == config.num_minibatches:
        #    break

if __name__ == "__main__":

    # Set up the config
    config = Config(sys.argv[1])
    dataset_name = sys.argv[2]
    model_name = sys.argv[3]
    train_model(config, dataset_name,model_name)
