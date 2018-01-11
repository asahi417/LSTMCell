
"""
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn_cell_impl.py
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/rnn/python/ops/rnn_cell.py
"""

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest


import logging


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class BasicLSTMCell(rnn_cell_impl.RNNCell):
    """Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units, forget_bias=1.0,
                 state_is_tuple=True, activation=None, reuse=None):
        """Initialize the basic LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          When restoring from CudnnLSTM-trained checkpoints, must use
          CudnnCompatibleLSTMCell instead.
        """
        super(BasicLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warning("%s: Using a concatenated state is slower and will soon be "
                            "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._linear = None

    @property
    def state_size(self):
        return (rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).
        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.
        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        Pep8 inspection appears since this signature is not same as `call` in tensorflow/python/layers/base.
            https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/layers/base.py
        """
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        if self._linear is None:
            self._linear = _Linear([inputs, h], 4 * self._num_units, True)
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)

        new_c = c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j)
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state


class _Linear(object):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of weight variable.
      dtype: data type for variables.
      build_bias: boolean, whether to build a bias variable.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      kernel_initializer: starting value to initialize the weight.
    Raises:
      ValueError: if inputs_shape is wrong.
    """

    def __init__(self, args, output_size, build_bias, bias_initializer=None, kernel_initializer=None):
        self._build_bias = build_bias

        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):  # list of Tensor or not
            args = [args]
            self._is_sequence = False  # not list
        else:
            self._is_sequence = True  # list of batch

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError("linear is expecting 2D arguments: %s" % shapes)
            if shape[1].value is None:
                raise ValueError("linear expects shape[1] to be provided for shape %s, but saw %s" % (shape, shape[1]))
            else:
                total_arg_size += shape[1].value

        dtype = [a.dtype for a in args][0]

        scope = vs.get_variable_scope()  # return current variable scope
        with vs.variable_scope(scope) as outer_scope:
            self._weights = vs.get_variable(
                _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype, initializer=kernel_initializer)
            if build_bias:
                with vs.variable_scope(outer_scope) as inner_scope:
                    inner_scope.set_partitioner(None)
                    if bias_initializer is None:
                        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
                    self._biases = vs.get_variable(
                        _BIAS_VARIABLE_NAME, [output_size], dtype=dtype, initializer=bias_initializer)

    def __call__(self, args):
        if not self._is_sequence:
            args = [args]

        if len(args) == 1:
            res = math_ops.matmul(args[0], self._weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), self._weights)
        if self._build_bias:
            res = nn_ops.bias_add(res, self._biases)
        return res
