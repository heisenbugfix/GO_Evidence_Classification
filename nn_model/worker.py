import tensorflow as tf
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
    if checkpoint and not config["is_training"]:
        print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
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
            fd, y_true = model.get_feed_data(data, is_training=False, full_batch=True)
            sigmoids = s.run(model.prediction, fd)
            predictions = sigmoids > 0.5
            y_pred = predictions.astype(int)
            evaluator = model_evaluation(y_true)
            acc = evaluator.compute_accuracy_score(y_true, y_pred)
            print("ACCURACY OF THE MODEL IS %f",acc)
            logger.info("ACCURACY OF THE MODEL IS %f",acc)
            f1_mac, f1_mic, f1_weighted, precision, recall = evaluator.binary_class_model(y_true, y_pred)
            print("f1_mac:%0.4f , f1_mic:%0.4f , f1_weighted:%0.4f , precision:%0.4f , recall:%0.4f"%(f1_mac, f1_mic, f1_weighted, precision, recall))
            logger.info("f1_mac:%0.4f , f1_mic:%0.4f , f1_weighted:%0.4f , precision:%0.4f , recall:%0.4f"%(f1_mac, f1_mic, f1_weighted, precision, recall))
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
