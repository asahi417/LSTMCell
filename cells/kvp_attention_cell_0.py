"""Key-Value-Predict Attention Wrapper for LSTM cell

Daniluk, Michał, et al. "Frustratingly short attention spans in neural language modeling.
" Proceedings of International Conference on Learning Representations (ICLR) 2017.
https://arxiv.org/abs/1702.04521

- Basic attention: Key = Value = Predict = Output
- Key-Value attention: [Key, Value=Predict] = Output
- Key-Value-Predict attention: [Key, Value, Predict] = Output


Need to do
- bidirectional version (mainly for classification)
"""

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.util import nest


_EPSILON = 10**-4


class KVPAttentionWrapper:
    """
    Key-Value-Predict attention wrapper for (stacked) LSTM cell.
    """

    def __init__(self, cell, attention_window, sequence_length, mode="kvp", bidirectional=False):

        self._cell = cell
        self._attention_window = attention_window
        self._sequence_length = sequence_length
        self._mode = mode
        self._bidirectional = bidirectional
        # self._alignment_history = alignment_history
        # if alignment_history:
        #     self._alignments = []

    @staticmethod
    def _attention(output_sequence, output_target, mode):
        """ Get context vector based on attention and
        weighted output target by context vector derived by attention mechanism.

        :param output_sequence: tensor shaped (batch, attention window, hidden unit num)
        :param output_target: tensor shaped (batch, hidden unit num)
        :param mode: None for basic one, `kvp` for key-value-predict attention
        :return: new output, context vector derived by attention mechanism
        """

        n_window, n_hidden = output_sequence.shape.as_list()[1:]

        if mode is None or mode == "k":  # basic attention
            os_k = os_v = os_p = output_sequence
            ot_k = ot_v = ot_p = output_target
        elif mode == "kv":  # key-value attention
            os_k, os_v = array_ops.split(value=output_sequence, num_or_size_splits=2, axis=2)
            ot_k, ot_v = array_ops.split(value=output_target, num_or_size_splits=2, axis=1)
            ot_p, os_p = ot_v, os_v
            if n_hidden % 2 != 0:
                raise ValueError("for `kv` mode, `n_hidden` should be even.")
            n_hidden = int(n_hidden / 2)
        elif mode == "kvp":  # key-value-prediction attention
            os_k, os_v, os_p = array_ops.split(value=output_sequence, num_or_size_splits=3, axis=2)
            ot_k, ot_v, ot_p = array_ops.split(value=output_target, num_or_size_splits=3, axis=1)
            if n_hidden % 3 != 0:
                raise ValueError("for `kvp` mode, `n_hidden` should be able to be divided by 3.")
            n_hidden = int(n_hidden / 3)
        else:
            raise ValueError("unknown mode")

        with vs.variable_scope("context_vector"):
            a = []  # alpha of attention mechanism
            w_h = vs.get_variable("w_h", shape=[n_hidden, n_hidden])  # weight for target
            w_y = vs.get_variable("w_y", shape=[n_hidden, n_hidden])  # weight for sequence
            w = vs.get_variable("w", shape=[n_hidden, 1])  # weight for attention

        logit_h = math_ops.matmul(ot_k, w_h)  # (batch, hidden)

        for n_w in range(n_window):
            logit_y = math_ops.matmul(os_k[:, n_w, :], w_y)  # (batch, hidden)
            logit = logit_h + logit_y
            m = math_ops.tanh(logit)  # M of attention mechanism
            a.append(math_ops.matmul(m, w))  # (batch, 1)

        a = nn_ops.softmax(array_ops.stack(a, axis=1), axis=1)  # (batch, window, 1)
        r = math_ops.reduce_sum(os_v * a, axis=1)  # context vector (batch, hidden)

        with vs.variable_scope("weighted_output"):  # derive attention weighted output
            w_h = vs.get_variable("w_h", shape=[n_hidden, n_hidden])  # weight for original target
            w_r = vs.get_variable("w_r", shape=[n_hidden, n_hidden])  # weight for context vector

        logit = math_ops.matmul(ot_p, w_h) + math_ops.matmul(r, w_r)
        output = math_ops.tanh(logit)
        # new output (batch, hidden or hidden/2 (kv) or hidden/3 (kvp))
        return output, r, n_hidden

    @staticmethod
    def _split_output(mode, alignment):
        n_hidden = alignment.shape.as_list()[1]
        if mode is None or mode == "k":  # basic attention
            outputs = alignment
        elif mode == "kv":  # key-value attention
            outputs, _ = array_ops.split(value=alignment, num_or_size_splits=2, axis=1)
            if n_hidden % 2 != 0:
                raise ValueError("for `kv` mode, `n_hidden` should be even.")
        elif mode == "kvp":  # key-value-prediction attention
            outputs, _, _ = array_ops.split(value=alignment, num_or_size_splits=3, axis=1)
            if n_hidden % 3 != 0:
                raise ValueError("for `kvp` mode, `n_hidden` should be able to be divided by 3.")
        else:
            raise ValueError("unknown mode")
        return outputs

    def __call__(self, inputs, initial_state, scope=None, alignment_history=None):
        """

        alignment_history を placeholder で入れるようにする

        :param inputs: tensor shaped (batch, time, feature_size)
        :param initial_state:
        :return:
        """

        outputs = []
        state = initial_state
        # alignments = [] if alignment_history is None else [alignment_history]

        with vs.variable_scope(scope or "kvp_attention"):

            for time_step in range(self._sequence_length):
                with vs.variable_scope("rnn_cell", reuse=None if time_step == 0 else True):
                    (cell_output, state) = self._cell(inputs[:, time_step, :], state)
                alignments.append(cell_output)
                if len(alignments) < self._attention_window:
                    output = self._split_output(self._mode, cell_output)
                else:
                    reuse = None if len(alignments) == self._attention_window else True
                    with vs.variable_scope("attention", reuse=reuse):
                        alignments = alignments[-self._attention_window:]
                        output, context_vector, self._n_hidden =\
                            self._attention(output_sequence=array_ops.stack(alignments, axis=1),
                                            output_target=cell_output, mode=self._mode)
                outputs.append(output)

            outputs = array_ops.stack(outputs, axis=1)

        if self._alignment_history:
            self._alignments = alignments

        return outputs, state

    @property
    def n_hidden(self):
        return self._n_hidden

    @property
    def alignment_history_size(self):
        return len(self._alignments) if self._alignment_history else 0

    @property
    def alignment_history(self):
        return self._alignments if self._alignment_history else None

