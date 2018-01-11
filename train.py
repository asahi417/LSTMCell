
import os
import logging
import tensorflow as tf
import numpy as np

from data_reader import ptb_raw_data, BatchFeeder
from model import LSTMLanguageModel


def _create_log(name):
    """Logging."""
    if os.path.exists(name):
        os.remove(name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # handler for logger file
    handler1 = logging.FileHandler(name)
    handler1.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
    # handler for standard output
    handler2 = logging.StreamHandler()
    handler2.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def train(epoch, model,
          iter_train_data,
          iter_valid_data,
          iter_test_data,
          save_path="./log", lr_decay=None, test=False):
    """ Train model based on mini-batch of input data.

    :param int epoch:
    :param model: Model instance.
    :param iter_train_data: Data iterator.
    :param iter_valid_data: Data iterator.
    :param iter_test_data: Data iterator.
    :param str save_path: Path to save
    :param float lr_decay: learning rate will be divided by lr_decay each 100 epoch
    :param bool test: Show loss in each iteration.
    """
    data_size = iter_train_data.data_size
    num_steps = iter_train_data.num_steps
    batch_size = iter_train_data.batch_size

    # logger
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    logger = _create_log("%s/log" % save_path)
    logger.info(model.__doc__)
    logger.info("train: epoch (%i), size (%i), batch size(%i)" % (epoch, data_size, batch_size))

    # Initializing the tensor flow variables
    model.sess.run(tf.global_variables_initializer())

    result = []
    for _e in range(epoch):

        # Train
        length, loss = 0, 0.0
        for _in, _out in iter_train_data:
            feed_dict = dict(((model.inputs, _in), (model.targets, _out), (model.is_train, True)))
            if lr_decay is not None and lr_decay != 1.0:
                feed_dict[model.lr_decay] = lr_decay ** (np.ceil(_e / 100) - 1)
            _loss, _ = model.sess.run([model.loss, model.train_op], feed_dict=feed_dict)
            loss += _loss
            length += num_steps
            if test:
                print(np.exp(loss / length))
        perplexity = np.exp(loss / length)

        # Valid
        length, loss = 0, 0.0
        for _in, _out in iter_valid_data:
            feed_dict = dict(((model.inputs, _in), (model.targets, _out), (model.is_train, False)))
            _loss = model.sess.run([model.loss], feed_dict=feed_dict)
            loss += _loss
            length += num_steps
        perplexity_valid = np.exp(loss / length)

        # Test
        length, loss = 0, 0.0
        for _in, _out in iter_test_data:
            feed_dict = dict(((model.inputs, _in), (model.targets, _out), (model.is_train, False)))
            _loss = model.sess.run([model.loss], feed_dict=feed_dict)
            loss += _loss
            length += num_steps
        perplexity_test = np.exp(loss / length)

        logger.info("epoch %i, perplexity: train %0.3f, valid %0.3f, test %0.3f"
                    % (_e, perplexity, perplexity_valid, perplexity_test))

        result = [perplexity, perplexity_valid, perplexity_test]
        if _e % 50 == 0:
            model.saver.save(model.sess, "%s/progress-%i-model.ckpt" % (save_path, _e))
            np.savez("%s/progress-%i-acc.npz" % (save_path, _e), loss=np.array(result))
    model.saver.save(model.sess, "%s/model.ckpt" % save_path)
    np.savez("%s/statistics.npz" % save_path, loss=np.array(result))


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    _batch_size = 20
    _num_steps = 35

    raw_train, raw_validation, raw_test, vocab = ptb_raw_data("./simple-examples/data")

    iterators = dict()
    for raw_data, key in zip([raw_train, raw_validation, raw_test],
                             ["iter_train_data", "iter_valid_data", "iter_test_data"]):
        iterators[key] = BatchFeeder(batch_size=_batch_size, num_steps=_num_steps, sequence=raw_data)

    config = {
        "num_steps": _num_steps, "vocab_size": vocab,
        "embedding_size": 256, "n_hidden_1": 256, "n_hidden_2": 256, "n_hidden_3": 256
    }
    _model = LSTMLanguageModel(config, learning_rate=0.01)

    train(20, _model, test=False, **iterators)
