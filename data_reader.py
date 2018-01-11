

"""
This reader is based on https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py
"""

import collections
import os
import tensorflow as tf


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".
    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.
    The PTB dataset comes from Tomas Mikolov's webpage:
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.
    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


"""
Batch feeding instance for list sequence
"""


class BatchFeeder:
    """ Batch feeding iterator for train neural language model."""

    def __init__(self, batch_size, num_steps, sequence):
        """
        :param int batch_size: batch size
        :param int num_steps: truncation size of sequence
        :param list sequence: list of index (int) of word
        """
        self._index = 0
        self._batch_size = batch_size
        self._num_steps = num_steps

        self._seq = sequence

        self._n = len(self._seq)

    def __iter__(self):
        return self

    def __next__(self):
        """ next batch for train data (size is `self._batch_size`)
        :return (inputs, outputs): list (batch, num_steps)
        """

        if self._index + self._batch_size + self._num_steps >= len(self._seq):
            self._index = 0
            raise StopIteration

        inputs, outputs = [], []
        for _b in range(self._batch_size):
            self._index += _b
            inputs.append(self._seq[self._index:self._index + self._num_steps])
            outputs.append(self._seq[self._index + 1:self._index + self._num_steps + 1])
        self._index += 1
        return inputs, outputs

    @property
    def data_size(self):
        return self._n

    @property
    def num_steps(self):
        return self._num_steps

    @property
    def batch_size(self):
        return self._batch_size


if __name__ == '__main__':
    a = [i for i in range(10)]
    bf = BatchFeeder(2, 3, a)
    print("first")
    for i in bf:
        print(i)

    print("second")
    for i in bf:
        print(i)