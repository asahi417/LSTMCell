import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import xavier_initializer

import numpy as np

from cells import CustomLSTMCell


def _batch_size(inputs):
    """ Dynamic batch size. https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn.py """
    while nest.is_sequence(inputs):
        inputs = inputs[0]
    return array_ops.shape(inputs)[0]


def _full_connected(x, weight_shape, scope=None, bias=True, initializer=xavier_initializer(seed=0)):
    """ fully connected layer
    - weight_shape: input size, output size
    - priority: batch norm (remove bias) > dropout and bias term
    """
    # scope = "fully connected" if scope is None else scope
    with tf.variable_scope(scope or "fully_connected"):
        w = tf.get_variable("weight", shape=weight_shape, initializer=initializer, dtype=tf.float32)
        x = tf.matmul(x, w)
        if bias:
            b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
            return tf.add(x, b)
        else:
            return x


class LSTMLanguageModel(object):
    """ LSTM based Neural Language Model
            * recurrent dropout and layer norm for LSTM
            * batch norm for full connect
        - input -> LSTM x 3 -> output unit -> FC -> output
    """

    def __init__(self, config, load_model=None,
                 learning_rate=0.0001, gradient_clip=None, batch_norm=None, keep_prob=1.0,
                 layer_norm=False):
        """
        :param dict config: network config. Suppose dictionary with following elements
            num_steps: truncated sequence size
            vocab_size: vocabulary size
            embedding_size: embedding dimension
            n_hidden_1:
            n_hidden_2:
            n_hidden_3:
        :param float learning_rate: default 0.001
        :param float gradient_clip: (option) clipping gradient value
        :param float keep_prob: (option) keep probability of dropout
        :param bool layer_norm: (option) If True, use layer normalization for LSTM cell
        :param float batch_norm: (option) decay for batch norm for full connected layer. 0.95 is preferred.
                                 https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        :param str load_model: (option) load saved model. It is needed to provide same config.
        """
        self._config = config
        self._lr = learning_rate
        self._clip = gradient_clip
        self._batch_norm_decay = batch_norm
        self._layer_norm = layer_norm
        self._keep_prob = keep_prob

        # Create network
        self._build_model()
        # Launch the session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        # Load model
        if load_model:
            tf.reset_default_graph()
            self.saver.restore(self.sess, load_model)

    def _build_model(self):
        """ Create Network, Define Loss Function and Optimizer """

        self.inputs = tf.placeholder(tf.int32, [None, self._config["num_steps"]], name="inputs")
        self.targets = tf.placeholder(tf.int32, [None, self._config["num_steps"]], name="output")

        self.is_train = tf.placeholder_with_default(False, [])
        __keep_prob = tf.where(self.is_train, self._keep_prob, 1.0)
        __keep_prob_r = tf.where(self.is_train, self._keep_prob, 1.0)

        # onehot and embedding
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self._config["vocab_size"], self._config["embedding_size"]])
            inputs = tf.nn.embedding_lookup(embedding, self.inputs)
            batch_size = _batch_size(inputs)  # dynamic batch size

        inputs = tf.nn.dropout(inputs, __keep_prob)

        # build stacked LSTM instance
        with tf.variable_scope("stacked_lstm"):
            cells = []
            for i in range(1, 4):
                cell = CustomLSTMCell(num_units=self._config["n_hidden_%i" % i], dropout_keep_prob=__keep_prob_r,
                                      layer_norm=self._layer_norm)
                # cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self._config["n_hidden_%i" % i])
                cells.append(cell)
            cells = tf.nn.rnn_cell.MultiRNNCell(cells)

            outputs = []
            state = cells.zero_state(batch_size=batch_size, dtype=tf.float32)
            for time_step in range(self._config["num_steps"]):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cells(inputs[:, time_step, :], state)
                # print(cell_output.shape)
                outputs.append(cell_output)

            # weight is shared sequence direction (in addition to batch direction),
            # so reshape to treat sequence direction as batch direction
            # output shape: (batch, num_steps, last hidden size) -> (batch x num_steps, last hidden size)
            outputs = tf.stack(outputs, axis=1)
            outputs = tf.reshape(outputs, [-1, self._config["n_hidden_3"]])

            self._final_state = state  # currently, not used.

        # Prediction and Loss
        with tf.variable_scope("fully_connected"):
            layer = tf.nn.dropout(outputs, __keep_prob)
            weight = [self._config["n_hidden_3"], self._config["vocab_size"]]
            if self._batch_norm_decay is not None:
                layer = _full_connected(layer, weight, bias=False, scope="fc")
                logit = tf.contrib.layers.batch_norm(layer, decay=self._batch_norm_decay, is_training=self.is_train,
                                                     updates_collections=None)
            else:
                logit = _full_connected(layer, weight, bias=True, scope="fc")
            # Reshape logit to be a 3-D tensor for sequence loss
            logit = tf.reshape(logit, [batch_size, self._config["num_steps"], self._config["vocab_size"]])

            self._prediction = tf.nn.softmax(logit)

        # optimization
        with tf.variable_scope("optimization"):
            weight = tf.ones([batch_size, self._config["num_steps"]], dtype=tf.float32)
            loss = tf.contrib.seq2seq.sequence_loss(logits=logit, targets=self.targets, weights=weight,
                                                    average_across_timesteps=False, average_across_batch=True)
            self._loss = tf.reduce_sum(loss)

            # Define optimizer and learning rate: lr = lr/lr_decay
            self.lr_decay = tf.placeholder_with_default(1.0, [])  # learning rate decay
            optimizer = tf.train.AdamOptimizer(self._lr / self.lr_decay)
            if self._clip is not None:
                _var = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, _var), self._clip)
                self._train_op = optimizer.apply_gradients(zip(grads, _var))
            else:
                self._train_op = optimizer.minimize(self._loss)

        # trainable variables
        variables = tf.trainable_variables()
        self._n_var = np.prod(variables[0].get_shape().as_list())
        for var in variables[1:]:
            sh = var.get_shape().as_list()
            # print(var.name, sh)
            self._n_var += np.prod(sh)
        print(self._n_var, 'total variables')

        # saver
        self._saver = tf.train.Saver()

    @property
    def final_state(self):
        return self._final_state

    @property
    def prediction(self):
        return self._prediction

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def saver(self):
        return self._saver

    @property
    def total_variable_number(self):
        return self._n_var


if __name__ == '__main__':
    import os

    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    net = {
        "num_steps": 35,
        "vocab_size": 10000,
        "embedding_size": 64,
        "n_hidden_1": 64, "n_hidden_2": 128, "n_hidden_3": 256
    }
    LSTMLanguageModel(net, gradient_clip=10, batch_norm=0.95, keep_prob=0.8, layer_norm=True)
