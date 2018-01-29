
"""
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn_cell_impl.py
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/rnn/python/ops/rnn_cell.py

BasicLSTMCell (and other RNN based cell) only for input with (batch, time).
Dynamic RNN cell can be handle input with (batch, time, input size) and dynamic sequence length.
"""

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl

from . import CustomLSTMCell

_EPSILON = 10**-4


class HyperLSTMCell(rnn_cell_impl.RNNCell):
    """
    Hypernets cell with recurrent dropout and layer normalization.
    """

    def __init__(self, num_units, num_units_hyper, embedding_dim,
                 forget_bias=1.0, activation=None, reuse=None,
                 layer_norm=False, norm_shift=0.0, norm_gain=1.0,  # layer normalization
                 dropout_keep_prob=1.0, dropout_prob_seed=None  # recurrent dropout
                 ):
        """ Initialize the Hyper LSTM cell.
        :param int num_units: The number of units in the LSTM cell.
        :param int num_units_hyper: The number of units in the Hyper LSTM cell (smaller than `num_units`).
        :param int embedding_dim: The number of embedding in the Hyper LSTM cell (smaller than `num_units_hyper`).
        :param float forget_bias: The bias added to forget gates (see above).
                Must set to `0.0` manually when restoring from CudnnLSTM-trained checkpoints.
        :param activation: Activation function of the inner states.  Default: `tanh`.
        :param reuse: (optional) Python boolean describing whether to reuse variables
                in an existing scope.  If not `True`, and the existing scope already has
                the given variables, an error is raised.
                When restoring from CudnnLSTM-trained checkpoints, must use
                CudnnCompatibleLSTMCell instead.
        :param bool layer_norm: Layer normalization
        :param float norm_shift: Layer normalization (shift)
        :param float norm_gain: Layer normalization (gain)
        :param float dropout_keep_prob: Recurrent dropout
        :param dropout_prob_seed:
        """
        super(HyperLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._num_units_hyper = num_units_hyper
        self._embedding_dim = embedding_dim
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

        self._layer_norm = layer_norm
        self._g = norm_gain
        self._b = norm_shift

        self._keep_prob = dropout_keep_prob
        self._seed = dropout_prob_seed

        self._hyper_lstm_cell = CustomLSTMCell(
            num_units_hyper, forget_bias=forget_bias, activation=activation, reuse=reuse,
            layer_norm=layer_norm, norm_shift=norm_shift, norm_gain=norm_gain, dropout_keep_prob=1.0)

    @property
    def state_size(self):
        full_num_units = self._num_units + self._num_units_hyper
        return rnn_cell_impl.LSTMStateTuple(full_num_units, full_num_units)

    @property
    def output_size(self):
        return self._num_units + self._num_units_hyper

    @staticmethod
    def _linear(x, weight_shape, bias=True, scope=None):
        """ linear projection (weight_shape: input size, output size) """
        with vs.variable_scope(scope or "linear"):
            w = vs.get_variable("kernel", shape=weight_shape)
            x = math_ops.matmul(x, w)
            if bias:
                b = vs.get_variable("bias", initializer=[0.0] * weight_shape[-1])
                return nn_ops.bias_add(x, b)
            else:
                return x

    def _layer_normalization(self, inputs, scope=None):
        """
        :param inputs: (batch, shape)
        :param scope:
        :return : layer normalized inputs (batch, shape)
        """
        shape = inputs.get_shape()[-1:]
        with vs.variable_scope(scope or "layer_norm"):
            # Initialize beta and gamma for use by layer_norm.
            gain = vs.get_variable("gain", shape=shape, initializer=init_ops.constant_initializer(self._g))
            scale = vs.get_variable("shift", shape=shape, initializer=init_ops.constant_initializer(self._b))
        m, v = nn_impl.moments(inputs, [1], keep_dims=True)  # (batch,)
        normalized_input = (inputs - m) / math_ops.sqrt(v + _EPSILON)  # (batch, shape)
        return normalized_input * gain + scale

    def _embedding(self, arg, h_hyper, scope):
        """ Hyper LSTM projection layer

        :param arg: hidden unit `h` or input `x`
        :param h_hyper: hyper hidden unit `h_hyper`
        :return: ["input", "transform", "forget", "output"]
        """
        with vs.variable_scope(scope):
            z = self._linear(h_hyper, [self._num_units_hyper, 4 * self._embedding_dim], scope="z")
            z = array_ops.split(value=z, num_or_size_splits=4, axis=1)
            cells = []
            for i, name in enumerate(["i", "j", "f", "o"]):
                d = self._linear(z[i], [self._embedding_dim, self._num_units], scope="d_%s" % name, bias=False)
                d = d * arg
                d = self._linear(d, [self._num_units, self._num_units], scope="d_linear_%s" % name, bias=False)
                cells.append(d)
        return cells

    def _embedding_bias(self, h_hyper, scope):
        """ Hyper LSTM projection layer for bias term

        :param h_hyper: hyper hidden unit `h_hyper`
        :return: ["input", "transform", "forget", "output"]
        """
        with vs.variable_scope(scope):
            z = self._linear(h_hyper, [self._num_units_hyper, 4 * self._embedding_dim], scope="z", bias=False)
            z = array_ops.split(value=z, num_or_size_splits=4, axis=1)
            cells = []
            for i, name in enumerate(["i", "j", "f", "o"]):
                cells.append(self._linear(z[i], [self._embedding_dim, self._num_units], scope="d_%s" % name))
        return cells

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM) with Hypernets.

        Layer norm for hyperLSTM and mainLSTM
        Recurrent dropout for mainLSTM

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size (num_unit + num_unit_hyper) ]`
            This state include [LSTM, hyperLSTM]
        Returns:
          A pair containing the new hidden state, and the new state.
        """
        c_concat, h_concat = state  # memory cell, hidden unit
        c, h = c_concat[:, 0:self._num_units], h_concat[:, 0:self._num_units]
        c_hyper, h_hyper = c_concat[:, self._num_units:], h_concat[:, self._num_units:]

        with vs.variable_scope("hyper_lstm"):
            inputs_hyper = array_ops.concat([inputs, h], 1)
            state_hyper = rnn_cell_impl.LSTMStateTuple(c_hyper, h_hyper)
            output_hyper, state_hyper = self._hyper_lstm_cell(inputs_hyper, state_hyper)
            (c_hyper, h_hyper) = state_hyper

        # embedding hidden state
        h_embed = self._embedding(h, h_hyper, scope="h")
        x_embed = self._embedding(inputs, h_hyper, scope="x")
        b_embed = self._embedding_bias(h_hyper, scope="b")
        cells = []
        for i, name in enumerate(["i", "j", "f", "o"]):
            cell = h_embed[i] + x_embed[i] + b_embed[i]
            if self._layer_norm:
                cell = self._layer_normalization(cell, scope="layer_norm_%s" % name)
            cells.append(cell)
        i, j, f, o = cells
        g = self._activation(j)  # gating

        # recurrent dropout (dropout gating cell)
        if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
            g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

        gated_in = math_ops.sigmoid(i) * g
        memory = c * math_ops.sigmoid(f + self._forget_bias)

        c = memory + gated_in
        h = self._activation(c) * math_ops.sigmoid(o)

        c_concat = array_ops.concat([c, c_hyper], 1)
        h_concat = array_ops.concat([h, h_hyper], 1)
        state = rnn_cell_impl.LSTMStateTuple(c_concat, h_concat)
        return h, state


if __name__ == '__main__':
    _cell = HyperLSTMCell(256, 128, 4)

