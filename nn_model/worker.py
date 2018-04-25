import tensorflow as tf
import json
import time

def HAN_model_1(session, config, restore=False):
    """Hierarhical Attention Network"""
    try:
        from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, DropoutWrapper
    except ImportError:
        MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
        GRUCell = tf.nn.rnn_cell.GRUCell
    from nn_model.bn_lstm import BNLSTMCell
    from nn_model.han_model import HANClassifierModel

    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    cell = GRUCell(30)
    if config["cell"]==0:
        cell = GRUCell(30)
    elif config["cell"]==1:
        cell = BNLSTMCell(80, is_training)  # h-h batchnorm LSTMCell
    elif config["cell"]==2:
        cell = MultiRNNCell([cell] * 5)

    model = HANClassifierModel(
        vocab_size=10,
        embedding_size=5,
        go_size=10,
        go_embedding_size=10,
        classes=3,
        word_cell=GRUCell(10),
        sentence_cell=GRUCell(10),
        word_output_size=10,
        sentence_output_size=10,
        max_grad_norm=5.0,
        dropout_keep_proba=0.5
        # is_training = config["is_training"]
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


def train(configuration):
    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as s:
        model, saver = model_fn(s, configuration)
        if configuration["is_training"]:
            summary_writer = tf.summary.FileWriter(configuration["tflog_dir"], graph=tf.get_default_graph())

            ########
            fd = {
                model.is_training: True,
                model.inputs: [[
                    [5, 4, 1, 0],
                    [3, 3, 6, 7],
                    [6, 7, 0, 0]
                ],
                    [
                        [2, 2, 1, 0],
                        [3, 3, 6, 7],
                        [0, 0, 0, 0]
                    ]],
                model.word_lengths: [
                    [3, 4, 2],
                    [3, 4, 0],
                ],
                model.sentence_lengths: [3, 2],
                model.labels: [[0, 1, 0], [1, 1, 1]],
                model.aspect:[[0,1,0],[1,0,0]],
                model.go_inputs:[1,2]
            }
            ########
            # for i, (x, y) in enumerate(batch_iterator(task.read_trainset(epochs=3), args.batch_size, 300)):
            for i in range(2):
                # fd = model.get_feed_data(x, y, class_weights=class_weights)
                t0 = time.clock()
                step, summaries, loss, _ = s.run([
                    model.global_step,
                    model.summary_op,
                    model.loss,
                    model.train_op,
                ], fd)
                td = time.clock() - t0

                if configuration["dump_log"]:
                    summary_writer.add_summary(summaries, global_step=step)

                if step != 0:
                    print('checkpoint & graph meta')
                    saver.save(s, configuration["checkpoint_dir"], global_step=step)
                    print('checkpoint done')



def main():
    with open("config.json") as f:
        config = json.load(f)
        train(config)


if __name__ == '__main__':
    main()
