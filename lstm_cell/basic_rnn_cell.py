
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl

_EPSILON = 10**-4


class CustomRNNCell(rnn_cell_impl.RNNCell):
    """Customized RNN with several additional regularization
    - variational dropout (inputs and state: currently same dropout prob,  per-sample masking)
    - recurrent highway network
    - layer normalization
    """

    def __init__(self,
                 num_units: int,
                 forget_bias: float=-2.0,
                 activation: str=None,
                 reuse: bool=None,
                 layer_norm: bool=False,
                 norm_shift: float=0.0,
                 norm_gain: float=1.0,  # layer normalization
                 dropout_keep_prob_in: float=1.0,
                 dropout_keep_prob_h: float=1.0,
                 variational_dropout: bool=True,
                 dropout_prob_seed: int=None,  # dropout
                 recurrent_highway: bool=False,
                 highway_state_gate: bool=False,  # if true use HSG
                 recurrence_depth: int=4,  # depth for recurrent highway
                 coupling_gate: bool=True  # coupling parameter for RHN and HSG
                 ):
        """Initialize the basic RNN cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          layer_norm: (optional) If True, apply layer normalization.
          norm_shift: (optional) Shift parameter for layer normalization.
          norm_gain: (optional) Gain parameter for layer normalization.
          dropout_keep_prob_h: (optional) keep probability of variational dropout for recurrent unit
          dropout_keep_prob_in: (optional) keep probability of dropout for input
          dropout_prob_seed: (optional) seed value for dropout random variable
          recurrent_highway: (optional)
          recurrence_depth: (optional)
          highway_state_gate: (optional)
          coupling_gate: valid for hsg, rhn. coupling c = 1-t
          forget_bias: valid for hsg, rhn. forget gate initial bias
          variational_dropout: bool, if use variational dropout (apply dropout between state)
        """
        super(CustomRNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

        self._layer_norm = layer_norm
        self._g = norm_gain
        self._b = norm_shift

        self._keep_prob_in = dropout_keep_prob_in
        self._keep_prob_h = dropout_keep_prob_h
        self._seed = dropout_prob_seed

        self._highway = recurrent_highway
        self._highway_state_gate = highway_state_gate
        self._recurrence_depth = recurrence_depth
        self._coupling_gate = coupling_gate
        self._variational_dropout = variational_dropout

    @property
    def state_size(self):
        if self._highway_state_gate:
            return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)
        else:
            return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _layer_normalization(self, inputs, scope=None):
        """
        :param inputs: (batch, shape)
        :param scope:
        :return : layer normalized inputs (batch, shape)
        """
        shape = inputs.get_shape()[-1:]
        with vs.variable_scope(scope or "layer_norm"):
            # Initialize beta and gamma for use by layer_norm.
            g = vs.get_variable("gain", shape=shape, initializer=init_ops.constant_initializer(self._g))  # (shape,)
            s = vs.get_variable("shift", shape=shape, initializer=init_ops.constant_initializer(self._b))  # (shape,)
        m, v = nn_impl.moments(inputs, [1], keep_dims=True)  # (batch,)
        normalized_input = (inputs - m) / math_ops.sqrt(v + _EPSILON)  # (batch, shape)
        return normalized_input * g + s

    @staticmethod
    def _linear(x, weight_shape, bias=True, scope=None, bias_ini=0.0):
        """ linear projection (weight_shape: input size, output size) """
        with vs.variable_scope(scope or "linear"):
            w = vs.get_variable("kernel", shape=weight_shape)
            x = math_ops.matmul(x, w)
            if bias:
                b = vs.get_variable("bias", initializer=[bias_ini] * weight_shape[-1])
                return nn_ops.bias_add(x, b)
            else:
                return x

    def call(self, inputs, state):
        """ RNN cell """

        # Highway state gate
        if self._highway and self._highway_state_gate:
            state_hsg, state = state
        else:
            state_hsg = None

        # variational dropout for hidden unit (recurrent unit)
        if self._variational_dropout:
            state = nn_ops.dropout(state, self._keep_prob_h, seed=self._seed)

        # dropout for input
        inputs = nn_ops.dropout(inputs, self._keep_prob_in, seed=self._seed)

        if self._highway or self._recurrence_depth > 1:
            # Recurrent Highway Cell (state is last intermediate output of previous time step)
            inter_out = state
            for r in range(1, self._recurrence_depth+1):
                with vs.variable_scope("recurrent_depth_%i" % r):
                    if r == 1:
                        args = array_ops.concat([inputs, inter_out], 1)
                        h = self._linear(args, [args.get_shape()[-1], self._num_units], scope="h")
                        t = self._linear(
                            args, [args.get_shape()[-1], self._num_units], scope="t", bias_ini=self._forget_bias)
                        if not self._coupling_gate:
                            c = self._linear(args, [args.get_shape()[-1], self._num_units], scope="c")
                    else:

                        # variational dropout for recurrent unit
                        if self._variational_dropout:
                            inter_out = nn_ops.dropout(inter_out, self._keep_prob_h, seed=self._seed)

                        h = self._linear(inter_out, [self._num_units, self._num_units], scope="h")
                        t = self._linear(
                            inter_out, [self._num_units, self._num_units], scope="t", bias_ini=self._forget_bias)
                        if not self._coupling_gate:
                            c = self._linear(inter_out, [self._num_units, self._num_units], scope="c")

                    # layer normalization
                    if self._layer_norm:
                        h = self._layer_normalization(h, "layer_norm_h")
                        t = self._layer_normalization(t, "layer_norm_t")
                        if not self._coupling_gate:
                            c = self._layer_normalization(c, "layer_norm_c")

                    h = self._activation(h)
                    t = math_ops.sigmoid(t)

                    if not self._coupling_gate:
                        c = math_ops.sigmoid(c)
                        inter_out = h * t + inter_out * c
                    else:  # c = 1 - t
                        inter_out = (h - inter_out) * t + inter_out

            # Highway state gate
            if self._highway_state_gate:
                args = array_ops.concat([state_hsg, inter_out], 1)
                g = self._linear(args, [args.get_shape()[-1], self._num_units], scope="highway_state_gate")
                g = math_ops.sigmoid(g)
                output = g * state_hsg + (1 - g) * inter_out
                state = rnn_cell_impl.LSTMStateTuple(output, inter_out)
            else:
                state = inter_out
                output = inter_out

        else:
            # Most basic RNN: output = new_state = act(W * input + U * state + B).
            args = array_ops.concat([inputs, state], 1)
            linear = self._linear(args, [args.get_shape()[-1], self._num_units])

            # layer normalization
            if self._layer_norm:
                linear = self._layer_normalization(linear, "layer_norm")
            state = self._activation(linear)
            output = self._activation(linear)

        return output, state

