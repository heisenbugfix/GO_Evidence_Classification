import tensorflow as tf
import numpy as np
import json
import time
import logging
import pickle as pkl
from accuracy_score import model_evaluation


def HAN_model_1(session, config, logger, restore=False):
    """Hierarhical Attention Network"""
    try:
        from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, DropoutWrapper
    except ImportError:
        MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
        GRUCell = tf.nn.rnn_cell.GRUCell
    from bn_lstm import BNLSTMCell
    from han_model import HANClassifierModel

    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    cell = GRUCell(50)
    if config["cell"] == 0:
        cell = GRUCell(50)
        logger.info("Using GRU")
    elif config["cell"] == 1:
        cell = BNLSTMCell(80, is_training)  # h-h batchnorm LSTMCell
        logger.info("Using batch Normalized LSTM")
    else:
        logger.info("Using GRU")

    # elif config["cell"] == 2:
    #     cell = MultiRNNCell([cell] * 5)
    #     logger.info("Using multi RNN cells")

    model = HANClassifierModel(
        vocab_size=config["vocab_size"],
        embedding_size=config["embedding_size"],
        go_size=config["go_size"],
        go_embedding_size=config["go_embedding_size"],
        classes=config["classes"],
        word_cell=cell,
        sentence_cell=cell,
        word_output_size=config["word_output_size"],
        sentence_output_size=config["sentence_output_size"],
        max_grad_norm=config["max_grad_norm"],
        dropout_keep_proba=config["dropout_keep_prob"],
        is_training=is_training
    )

    saver = tf.train.Saver(tf.global_variables())
    checkpoint = tf.train.get_checkpoint_state(config["checkpoint_dir"])
    print(checkpoint)
    if checkpoint and not config["is_training"]:
        print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
        logger.info("Reading model parameters from",checkpoint.model_checkpoint_path)
        saver.restore(session, checkpoint.model_checkpoint_path)
    else:
        print("Created model with fresh parameters")
        session.run(tf.global_variables_initializer())
    return model, saver


model_fn = HAN_model_1


def batch_iterator(dataset, batch_size, max_epochs):
    for i in range(max_epochs):
        xb = []
        yb = []
        for ex in dataset:
            x, y = ex
            xb.append(x)
            yb.append(y)
            if len(xb) == batch_size:
                yield xb, yb
                xb, yb = [], []


def train_test(configuration):
    tf.reset_default_graph()
    logger = logging.getLogger("HAN")
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as s:
        model, saver = model_fn(s, configuration, logger)
        ########
        # Written only for sanity check of the model
        # fd = {
        #     model.is_training: True,
        #     model.inputs: [[
        #         [5, 4, 1, 0],
        #         [3, 3, 6, 7],
        #         [6, 7, 0, 0]
        #     ],
        #         [
        #             [2, 2, 1, 0],
        #             [3, 3, 6, 7],
        #             [0, 0, 0, 0]
        #         ]],
        #     model.word_lengths: [
        #         [3, 4, 2],
        #         [3, 4, 0],
        #     ],
        #     model.sentence_lengths: [3, 2],
        #     model.labels: [[0, 1, 0], [1, 1, 1]],
        #     model.aspect:[[0,1,0],[1,0,0]],
        #     model.go_inputs:[1,2]
        # }
        ########
        if configuration["is_training"]:
            # LOAD WORD EMBEDDINGS
            with open(configuration["w_emb_path"], 'rb') as f:
                w_emb = pkl.load(f)
            s.run(model.word_emb_init, feed_dict={model.word_emb_placeholder: w_emb})
            summary_writer = tf.summary.FileWriter(configuration["tflog_dir"], graph=tf.get_default_graph())
            # Loading train data
            with open(configuration["train_data_path"], 'rb')as f:
                data = pkl.load(f)
            logger.info("Loaded Data")
            full_batch = configuration["full_batch"]
            for i in range(1, configuration["epochs"] + 1):
                fd, _ = model.get_feed_data(data, full_batch=full_batch)
                t0 = time.clock()
                step, summaries, loss, _ = s.run([
                    model.global_step,
                    model.summary_op,
                    model.loss,
                    model.train_op,
                ], fd)
                td = time.clock() - t0
                logger.info("STEP: %7d | Loss: %.8f  | Time: %f" % (step, loss, td))
                print("STEP: %7d | Loss: %.8f  | Time: %f" % (step, loss, td))
                if configuration["dump_log"]:
                    summary_writer.add_summary(summaries, global_step=step)

                if step % configuration["dump_after_every_x_epochs"] == 0:
                    logger.info('checkpoint & graph meta')
                    saver.save(s, configuration["checkpoint_dir"], global_step=step)
                    logger.info('checkpoint done')
        else:
            # Load test data:
            with open(configuration["test_data_path"], 'rb') as f:
                data = pkl.load(f)
            logger.info("Loaded Test Data")
            test_data = model.get_feed_data_for_test(data, max_batchsize=1000)
            print("CALCUALTED TRUE")
            curr_true = None
            curr_pred = None
            for i in range(len(test_data)):
                logger.info("Loaded Test Data")
                fd, y_true = test_data[i][0], test_data[i][1]
                #creating feeddict:
                feed_d = {}
                feed_d = {model.inputs:fd["abstract"],
                          model.sentence_lengths:fd['doc_len'],
                          model.word_lengths:fd["sent_len"],
                          model.go_inputs:fd["go_inputs"],
                          model.aspect:fd["aspect"],
                          model.is_training:False
                          }
                sigmoids = s.run(model.prediction, feed_d)
                print("CALCUALTED PRED")
                predictions = sigmoids > 0.5
                y_pred = predictions.astype(int)
                if curr_true is None:
                    curr_true = y_true
                else:
                    curr_true = np.vstack((curr_true,y_true))
                if curr_pred is None:
                    curr_pred = y_pred
                else:
                    curr_pred = np.vstack((curr_pred,y_pred))
            #dumping the predicted data
            if configuration["dump_eval"]:
                with open("predict_data.pkl",'wb') as f:
                    pkl.dump([curr_true,curr_pred],f)
            evaluator = model_evaluation(curr_true)
            acc = evaluator.compute_accuracy_score(curr_true, curr_pred)
            print("ACCURACY OF THE MODEL IS %f",acc)
            logger.info("ACCURACY OF THE MODEL IS %f",acc)
            f1_mac, f1_mic, f1_weighted, precision, recall = evaluator.binary_class_model(curr_true, curr_pred)
            print("f1_macro : ",f1_mac)
            print("###############################")
            print("f1_micro     :", f1_mic)
            print("###############################")
            print("f1_weighted     :", f1_weighted)
            print("###############################")
            print("precision     :", precision)
            print("###############################")
            print("recall     :", recall)
            print("###############################")

            logger.info("f1_macro : "+ str(f1_mac))
            logger.info("###############################")
            logger.info("f1_micro     :"+ str(f1_mic))
            logger.info("###############################")
            logger.info("f1_weighted     :"+ str(f1_weighted))
            logger.info("###############################")
            logger.info("precision     :"+ str(precision))
            logger.info("###############################")
            logger.info("recall     :"+ str(recall))
            logger.info("###############################")

            # print(y_pred, y_true)
            # calculate precision recall f1


def main():
    with open("config.json") as f:
        config = json.load(f)

        if config["is_training"]:
            try:
                logfile = config["train_log_filename"]
                with open(logfile, 'w') as f:
                    pass
            except:
                logfile = "log_train.txt"
            logging.basicConfig(filename=logfile,
                                filemode='a',
                                level=logging.DEBUG)

            logging.info("Starting training of Attention Model")
        else:
            try:
                logfile = config["test_log_filename"]
                with open(logfile, 'w') as f:
                    pass
            except:
                logfile = "log_test.txt"
            logging.basicConfig(filename=logfile,
                                filemode='a',
                                level=logging.DEBUG)

            logging.info("Starting testing of Attention Model")
        train_test(config)


if __name__ == '__main__':
    main()
