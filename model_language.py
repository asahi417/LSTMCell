import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
import numpy as np

from cells import HyperLSTMCell
from cells import CustomLSTMCell
from cells import CustomRNNCell
from cells import KVPAttentionWrapper


def dynamic_batch_size(inputs):
    """ Dynamic batch size. https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn.py """
    while nest.is_sequence(inputs):
        inputs = inputs[0]
    return array_ops.shape(inputs)[0]


def full_connected(x, weight_shape, scope=None, bias=True):
    """ fully connected layer
    - weight_shape: input size, output size
    - priority: batch norm (remove bias) > dropout and bias term
    """
    # scope = "fully connected" if scope is None else scope
    with tf.variable_scope(scope or "fully_connected", reuse=None):
        w = tf.get_variable("weight", shape=weight_shape, dtype=tf.float32)
        x = tf.matmul(x, w)
        if bias:
            b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
            return tf.add(x, b)
        else:
            return x


class LSTMLanguageModel:
    """ Neural Language Model with following regularization."""

    def __init__(self,
                 config,
                 config_lstm,
                 logger,
                 ini_scale=0.01,
                 type_of_lstm=None,
                 load_model=None,
                 learning_rate=0.0001,
                 gradient_clip=None,
                 batch_norm=None,
                 keep_prob=1.0,
                 keep_prob_r=1.0,
                 weight_decay=0.0,
                 weight_tying=False,
                 layer_norm=False,
                 optimizer='sgd',
                 batch_size=50):
        """
        :param dict config: network config. Suppose dictionary with following elements
            num_steps: truncated sequence size
            vocab_size: vocabulary size
            embedding_size: embedding dimension
            n_unit: number of hidden unit
        :param float learning_rate: default 0.001
        :param float gradient_clip: (option) clipping gradient value
        :param float keep_prob: (option) keep probability of dropout
        :param float weight_decay: (option) weight decay (L2 regularization)
        :param float weight_tying: (option) weight tying
        :param bool layer_norm: (option) If True, use layer normalization for LSTM cell
        :param float batch_norm: (option) decay for batch norm for full connected layer. 0.95 is preferred.
                                 https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        :param str load_model: (option) load saved model. It is needed to provide same config.
        """
        self._config = config
        self._config_lstm = config_lstm
        self._lr = learning_rate
        self._clip = gradient_clip
        self._batch_norm_decay = batch_norm
        self._layer_norm = layer_norm
        self._keep_prob = keep_prob
        self._keep_prob_r = keep_prob_r
        self._weight_decay = weight_decay
        self._weight_tying = weight_tying
        self._type_of_lstm = type_of_lstm if type_of_lstm else 'lstm'
        self._logger = logger
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._ini_scale = ini_scale

        if type_of_lstm in ["hypernets"]:
            self._LSTMCell = HyperLSTMCell
        elif type_of_lstm in ["lstm", "kvp"]:
            self._LSTMCell = CustomLSTMCell
        elif type_of_lstm in ["rhn", "hsg"]:
            self._LSTMCell = CustomRNNCell
        else:
            raise ValueError("unknown lstm")

        initializer = tf.random_uniform_initializer(-ini_scale, ini_scale, seed=0)
        self._logger.info('BUILD GRAPH')
        with tf.variable_scope("language_model_%s" % type_of_lstm, initializer=initializer, reuse=None):
            self.__build_graph()
            self._sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        if load_model:  # Load trained model to use
            self._saver.restore(self._sess, load_model)
        else:  # Initializing the tensor flow variables
            self._sess.run(tf.global_variables_initializer())

    def __build_graph(self):
        """ Create Network, Define Loss Function and Optimizer """

        self.inputs = tf.placeholder(tf.int32, [None, self._config["num_steps"]], name="inputs")
        self.targets = tf.placeholder(tf.int32, [None, self._config["num_steps"]], name="output")

        self.is_train = tf.placeholder_with_default(False, [])
        __keep_prob = tf.where(self.is_train, self._keep_prob, 1.0)
        __keep_prob_r = tf.where(self.is_train, self._keep_prob_r, 1.0)
        __weight_decay = tf.where(self.is_train, self._weight_decay, 0)

        # onehot and embedding
        with tf.device("/cpu:0"):  # with tf.variable_scope("embedding"):
            embedding = tf.get_variable("embedding", [self._config["vocab_size"], self._config["embedding_size"]])
            inputs = tf.nn.embedding_lookup(embedding, self.inputs)
            batch_size = dynamic_batch_size(inputs)  # dynamic batch size

        inputs = tf.nn.dropout(inputs, __keep_prob)

        with tf.variable_scope("RNNCell"):
            if self._type_of_lstm == "attention":
                # build stacked LSTM layer
                cells = []
                for i in range(1, 3):
                    self._config["dropout_keep_prob"] = __keep_prob_r
                    cell = self._LSTMCell(**self._config_lstm)
                    cells.append(cell)
                cells = tf.nn.rnn_cell.MultiRNNCell(cells)

                attention_layer = KVPAttentionWrapper(cells,
                                                      self._config["attention_window"],
                                                      self._config["num_steps"],
                                                      mode=self._config["attention_mode"])

                self._initial_state = cells.zero_state(batch_size=batch_size, dtype=tf.float32)
                outputs, self._final_state = attention_layer(inputs, self._initial_state)
                self._alignment_history_size = attention_layer.alignment_history_size
                self._alignment_history = attention_layer.alignment_history
                n_hidden = attention_layer.n_hidden

            else:
                if self._type_of_lstm in ["rhn", "hsg"]:
                    # build single RNN layer
                    self._config["dropout_keep_prob"] = __keep_prob_r
                    cells = self._LSTMCell(**self._config_lstm)
                else:
                    # build stacked LSTM layer
                    cells = []
                    for i in range(1, 3):
                        self._config["dropout_keep_prob"] = __keep_prob_r
                        cell = self._LSTMCell(**self._config_lstm)
                        cells.append(cell)
                    cells = tf.nn.rnn_cell.MultiRNNCell(cells)

                outputs = []
                self._initial_state = cells.zero_state(batch_size=batch_size, dtype=tf.float32)
                state = self._initial_state
                for time_step in range(self._config["num_steps"]):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cells(inputs[:, time_step, :], state)
                    # print(cell_output.shape)
                    outputs.append(cell_output)
                outputs = tf.stack(outputs, axis=1)
                self._final_state = state
                n_hidden = self._config_lstm["num_units"]
        # weight is shared sequence direction (in addition to batch direction),
        # so reshape to treat sequence direction as batch direction
        # output shape: (batch, num_steps, last hidden size) -> (batch x num_steps, last hidden size)
        outputs = tf.reshape(outputs, [-1, n_hidden])

        outputs = tf.nn.dropout(outputs, __keep_prob)

        # Prediction and Loss
        with tf.variable_scope("fully_connected", reuse=None):
            weight = [n_hidden, self._config["vocab_size"]]
            if self._batch_norm_decay is not None:
                layer = full_connected(outputs, weight, bias=False, scope="fc")
                logit = tf.contrib.layers.batch_norm(layer, decay=self._batch_norm_decay, is_training=self.is_train,
                                                     updates_collections=None)
            else:
                logit = full_connected(outputs, weight, bias=True, scope="fc")
            # Reshape logit to be a 3-D tensor for sequence loss
            logit = tf.reshape(logit, [batch_size, self._config["num_steps"], self._config["vocab_size"]])

            self._prediction = tf.nn.softmax(logit)

        # optimization
        with tf.variable_scope("optimization"):
            loss = tf.contrib.seq2seq.sequence_loss(
                logits=logit,
                targets=self.targets,
                weights=tf.ones([batch_size, self._config["num_steps"]], dtype=tf.float32),
                average_across_timesteps=False, average_across_batch=True)

            t_vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in t_vars])  # weight decay

            self._loss = tf.reduce_sum(loss) + __weight_decay * l2_loss

            # Define optimizer and learning rate: lr = lr/lr_decay
            self.lr_decay = tf.placeholder_with_default(1.0, [])  # learning rate decay
            if self._optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self._lr * self.lr_decay)
            elif self._optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self._lr * self.lr_decay)
            else:
                raise ValueError('unknown optimizer %s' % self._optimizer)
            if self._clip is not None:
                grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, t_vars), self._clip)
                self._train_op = optimizer.apply_gradients(zip(grads, t_vars))
            else:
                self._train_op = optimizer.minimize(self._loss)

        # count trainable variables
        self._n_var = 0
        for var in t_vars:
            sh = var.get_shape().as_list()
            sh = int(np.prod(sh))
            self._logger.info('%s: %i' % (var.name, sh))
            self._n_var += sh
        self._logger.info('total variables:%i' % self._n_var)

        # saver
        self._saver = tf.train.Saver()

    @property
    def initial_state(self):
        return self._initial_state

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
    def sess(self):
        return self._sess

    @property
    def total_variable_number(self):
        return self._n_var

    @property
    def batch_size(self):
        return self._batch_size
