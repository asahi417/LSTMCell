

"""
This reader is based on https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py
"""

import collections
import os
import tensorflow as tf
import numpy as np


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
        :param int batch_size: data number in a batch
        :param int num_steps: truncation size of sequence
        :param list sequence: list of index (int) of word
        """
        self._index = 0
        self._batch_size = batch_size
        self._num_steps = num_steps
        seq = np.array(sequence, dtype=np.int32)

        self._n = len(seq)
        # batch_number: number of batch chunk in a full sequence
        batch_number = self._n // self._batch_size
        # sequence -> data
        # [1, 2, ..., 10] -> [[1, 2], [3, 4], ..., [9, 10]] for batch size (5)
        self._data = np.zeros([self._batch_size, batch_number], dtype=np.int32)
        for i in range(self._batch_size):
            self._data[i] = seq[batch_number * i:batch_number * (i + 1)]
        # iteration number of batch data in an epoch
        self._iteration_number = (batch_number - 1) // self._num_steps

    def __iter__(self):
        return self

    def __next__(self):
        """ next batch for train data (size is `self._batch_size`)
        loop for self._iteration_number
        :return (inputs, outputs): list (batch_size, num_steps)
        """
        if self._index == self._iteration_number:
            self._index = 0
            raise StopIteration
        x = self._data[:, self._index * self._num_steps:(self._index + 1) * self._num_steps]
        y = self._data[:, self._index * self._num_steps + 1:(self._index + 1) * self._num_steps + 1]
        self._index += 1
        return x, y

    @property
    def data_size(self):
        return self._n

    @property
    def iteration_number(self):
        return self._iteration_number

    @property
    def num_steps(self):
        return self._num_steps

    @property
    def batch_size(self):
        return self._batch_size


if __name__ == '__main__':
    bf = BatchFeeder(batch_size=5, num_steps=6, sequence=[i for i in range(100)])
    print("first")
    for i in bf:
        print(i)

    # print("second")
    # for i in bf:
    #     print(i)
