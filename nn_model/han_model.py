import tensorflow as tf
import random
import tensorflow.contrib.layers as layers
import math
from model_components import task_specific_attention, bidirectional_rnn
import itertools



class HANClassifierModel():
    """ Implementation of document classification model described in
      `Hierarchical Attention Networks for Document Classification (Yang et al., 2016)`
      (https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)"""

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 go_size,
                 go_embedding_size,
                 classes,
                 word_cell,
                 sentence_cell,
                 word_output_size,
                 sentence_output_size,
                 max_grad_norm,
                 dropout_keep_proba,
                 is_training=None,
                 learning_rate=1e-4,
                 device='/cpu:0',
                 scope=None):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.go_size = go_size
        self.go_embedding_size = go_embedding_size
        self.classes = classes
        self.word_cell = word_cell
        self.word_output_size = word_output_size
        self.sentence_cell = sentence_cell
        self.sentence_output_size = sentence_output_size
        self.max_grad_norm = max_grad_norm
        self.dropout_keep_proba = dropout_keep_proba

        with tf.variable_scope(scope or 'go_evidence') as scope:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            if is_training is not None:
                self.is_training = is_training
            else:
                self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

            self.sample_weights = tf.placeholder(shape=(None,), dtype=tf.float32, name='sample_weights')

            # [GO_term]
            self.go_inputs = tf.placeholder(shape=(None,), dtype=tf.int32, name="GO_inputs")

            # [Aspect]
            self.aspect = tf.placeholder(shape=(None, 3), dtype=tf.float32, name="Aspect")

            # [document x sentence x word]
            self.inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='inputs')

            # [document x sentence]
            self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths')

            # [document]
            self.sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')

            # [document]
            self.labels = tf.placeholder(shape=(None, self.classes), dtype=tf.float32, name='labels')

            (self.document_size,
             self.sentence_size,
             self.word_size) = tf.unstack(tf.shape(self.inputs))
            # embeddings cannot be placed on GPU
            with tf.device(device):
                self._init_embedding(scope)

            self._init_body(scope)

        with tf.variable_scope('train'):
            # self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            # self.loss = tf.reduce_mean(tf.multiply(self.cross_entropy, self.sample_weights))
            self.loss = tf.reduce_mean(self.cross_entropy)
            tf.summary.scalar('loss', self.loss)

            # self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32))
            # tf.summary.scalar('accuracy', self.accuracy)

            tvars = tf.trainable_variables()

            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            opt = tf.train.AdamOptimizer(learning_rate)

            self.train_op = opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)

            self.summary_op = tf.summary.merge_all()

    def get_fully_connected_layer(self, input, classes, namescope):
        with tf.name_scope(namescope):
            output = layers.fully_connected(input, classes, activation_fn=None)
            return output

    def _init_embedding(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding") as scope:
                self.embedding_matrix = tf.get_variable(
                    name="embedding_matrix",
                    shape=[self.vocab_size, self.embedding_size],
                    initializer=layers.xavier_initializer(),
                    dtype=tf.float32, trainable=False)
                self.inputs_embedded = tf.nn.embedding_lookup(
                    self.embedding_matrix, self.inputs)
                self.word_emb_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size])
                self.word_emb_init = self.embedding_matrix.assign(self.word_emb_placeholder)

                self.go_term_embedding_matrix = tf.get_variable(name="go_term_embedding_matrix",
                                                                shape=[self.go_size, self.go_embedding_size],
                                                                initializer=layers.xavier_initializer(),
                                                                dtype=tf.float32)
                self.inputs_go_embedded = tf.nn.embedding_lookup(
                    self.embedding_matrix, self.go_inputs)

    def _init_body(self, scope):
        with tf.variable_scope(scope):
            word_level_inputs = tf.reshape(self.inputs_embedded, [
                self.document_size * self.sentence_size,
                self.word_size,
                self.embedding_size
            ])
            word_level_lengths = tf.reshape(
                self.word_lengths, [self.document_size * self.sentence_size])

            with tf.variable_scope('word') as scope:
                word_encoder_output, _ = bidirectional_rnn(
                    self.word_cell, self.word_cell,
                    word_level_inputs, word_level_lengths,
                    scope=scope)

                with tf.variable_scope('attention') as scope:
                    word_level_output = task_specific_attention(
                        word_encoder_output,
                        self.word_output_size,
                        scope=scope)

                with tf.variable_scope('dropout'):
                    word_level_output = layers.dropout(
                        word_level_output, keep_prob=self.dropout_keep_proba,
                        is_training=self.is_training,
                    )

            # sentence_level

            sentence_inputs = tf.reshape(
                word_level_output, [self.document_size, self.sentence_size, self.word_output_size])

            with tf.variable_scope('sentence') as scope:
                sentence_encoder_output, _ = bidirectional_rnn(
                    self.sentence_cell, self.sentence_cell, sentence_inputs, self.sentence_lengths, scope=scope)

                with tf.variable_scope('attention') as scope:
                    sentence_level_output = task_specific_attention(
                        sentence_encoder_output, self.sentence_output_size, scope=scope)

                with tf.variable_scope('dropout'):
                    sentence_level_output = layers.dropout(
                        sentence_level_output, keep_prob=self.dropout_keep_proba,
                        is_training=self.is_training,
                    )
            #Concatenation of features
            concatenated_sentence_level_out = tf.concat((self.inputs_go_embedded, sentence_level_output, self.aspect), axis=1)

            with tf.variable_scope('classifier'):
                self.logits = layers.fully_connected(
                    concatenated_sentence_level_out, self.classes, activation_fn=None)

                self.prediction = tf.sigmoid(self.logits)

    def get_feed_data(self, data, is_training=True, full_batch=False,B=1000):
        inputs = data['abstract']
        sentence_lengths = data["doc_len"]
        word_lengths = data["sent_len"]
        go_inputs = data["go_inputs"]
        labels = data["label"]
        aspect = data["aspect"]
        if not full_batch:
            indx = random.sample(range(0, data["abstract"].shape[0]), B)
            inputs = inputs[indx]
            sentence_lengths = sentence_lengths[indx]
            word_lengths = word_lengths[indx]
            go_inputs = go_inputs[indx]
            labels = labels[indx]
            aspect = aspect[indx]
        fd = {
            self.inputs: inputs,
            self.sentence_lengths: sentence_lengths,
            self.word_lengths: word_lengths,
            self.go_inputs: go_inputs,
            self.aspect: aspect
        }
        if is_training:
            fd[self.labels] = labels
        fd[self.is_training] = is_training
        if is_training:
            return fd, None
        else:
            return fd, labels




    def get_feed_data_for_test(self, data, max_batchsize=40000):
        data_len = len(data['abstract'])
        splits = int(math.ceil(data_len/max_batchsize))
        inputs = data['abstract']
        sentence_lengths = data["doc_len"]
        word_lengths = data["sent_len"]
        go_inputs = data["go_inputs"]
        labels = data["label"]
        aspect = data["aspect"]
        data_list = []
        for i in range(splits):
            dic = {}
            if i < splits-1:
                dic['abstract'] = inputs[i*max_batchsize: (i+1)*max_batchsize]
                dic['doc_len'] = sentence_lengths[i*max_batchsize: (i+1)*max_batchsize]
                dic["sent_len"] = word_lengths[i*max_batchsize: (i+1)*max_batchsize]
                dic["go_inputs"] = go_inputs[i*max_batchsize: (i+1)*max_batchsize]
                dic["label"] = labels[i*max_batchsize: (i+1)*max_batchsize]
                dic["aspect"] = aspect[i*max_batchsize: (i+1)*max_batchsize]
            else:
                dic['abstract'] = inputs[i*max_batchsize: ]
                dic['doc_len'] = sentence_lengths[i*max_batchsize: ]
                dic["sent_len"] = word_lengths[i*max_batchsize: ]
                dic["go_inputs"] = go_inputs[i*max_batchsize: ]
                dic["label"] = labels[i*max_batchsize: ]
                dic["aspect"] = aspect[i*max_batchsize: ]
            data_list.append(dic)
        return data_list

    def get_feed_data_test_2(self,data,max_batchsize=40000)
        n=len(data)/3
        i=iter(data.items())
        d1=dict(itertools.islice(i,n))
        d1=dict(itertools.islice(i,n))
        d2=dict(i)  
        l1=d1['label']
        l2=d2['label']
        l3=d3['label']
        return [[d1,l1],[d2,l2],[d3,l3]]      



if __name__ == '__main__':
    try:
        from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
    except ImportError:
        LSTMCell = tf.nn.rnn_cell.LSTMCell
        LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
        GRUCell = tf.nn.rnn_cell.GRUCell

    tf.reset_default_graph()
    with tf.Session() as session:
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
            dropout_keep_proba=0.5,
        )
        session.run(tf.global_variables_initializer())

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

        sigmoids = (session.run(model.prediction, fd))
        predictions = sigmoids > 0.5
        print(predictions.astype(float))
        print(session.run([model.train_op, model.loss], fd))


