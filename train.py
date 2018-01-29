"""
This script is to check the implementation of LSTM cells, by language modeling task for PTB corpus.
Following parameter is fixed:
    - Epoch: 100
    - Learning rate: 0.5 (decay by 0.8 after 10 epoch)
    - batch: 20
    - back step: 35
    - hidden unit: 650
    - keep prob: 0.75 for embedding and output, 0.5 for recurrent state and input
"""


import os
import logging
import argparse

import tensorflow as tf
import numpy as np
from time import time

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


def train(model, max_max_epoch,
          iter_train_data, iter_valid_data, iter_test_data,
          save_path="./log", lr_decay=None, verbose=False):
    """ Train model based on mini-batch of input data.

    :param int max_max_epoch: max epoch
    :param model: Model instance.
    :param iter_train_data: Data iterator.
    :param iter_valid_data: Data iterator.
    :param iter_test_data: Data iterator.
    :param str save_path: Path to save
    :param float lr_decay: learning rate will be divided by lr_decay each 100 epoch
    :param bool verbose: Show loss in each iteration.
    """

    num_gpu = 0  # number of gpu

    max_epoch = int(np.ceil(max_max_epoch / 10))

    # logger
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    logger = _create_log("%s/log" % save_path)
    # logger.info(model.__doc__)
    logger.info("epoch (%i), sequence length (%i), batch size(%i), total variables(%i)" %
                (max_max_epoch, iter_train_data.data_size, iter_train_data.batch_size, model.total_variable_number))

    # Initializing the tensor flow variables
    model.sess.run(tf.global_variables_initializer())

    initial_state = None
    result = []
    for _e in range(max_max_epoch):

        # Train
        length, loss = 0, 0.0
        start_time = time()
        for step, (_in, _out) in enumerate(iter_train_data):
            feed_dict = dict(((model.inputs, _in), (model.targets, _out), (model.is_train, True)))

            if initial_state is not None:
                feed_dict[model.initial_state] = initial_state

            if _e >= max_epoch:
                feed_dict[model.lr_decay] = lr_decay

            val = model.sess.run([model.loss, model.final_state, model.train_op], feed_dict=feed_dict)

            loss += val[0]
            initial_state = val[1]
            length += iter_train_data.num_steps

            if verbose and step % (iter_train_data.iteration_number // 10) == 10:
                wps = length * iter_train_data.batch_size * max(1, num_gpu) / (time() - start_time)
                logger.info("epoch %i-%i/%i perplexity: %.3f, speed: %.3f wps"
                            % (_e, step, iter_train_data.iteration_number, np.exp(loss / length), wps))

        perplexity = np.exp(loss / length)

        # Valid
        length, loss = 0, 0.0
        for _in, _out in iter_valid_data:
            feed_dict = dict(((model.inputs, _in), (model.targets, _out), (model.is_train, False)))
            val = model.sess.run([model.loss], feed_dict=feed_dict)
            loss += val[0]
            length += iter_train_data.num_steps
        perplexity_valid = np.exp(loss / length)

        logger.info("epoch %i, perplexity: train %0.3f, valid %0.3f" % (_e, perplexity, perplexity_valid))

        result.append([perplexity, perplexity_valid])
        if _e % 50 == 0:
            model.saver.save(model.sess, "%s/progress-%i-model.ckpt" % (save_path, _e))
            np.savez("%s/progress-%i-acc.npz" % (save_path, _e), loss=np.array(result))

    # Test
    length, loss = 0, 0.0
    for _in, _out in iter_test_data:
        feed_dict = dict(((model.inputs, _in), (model.targets, _out), (model.is_train, False)))
        val = model.sess.run([model.loss], feed_dict=feed_dict)
        loss += val[0]
        length += iter_train_data.num_steps
    perplexity_test = np.exp(loss / length)

    model.saver.save(model.sess, "%s/model.ckpt" % save_path)
    np.savez("%s/statistics.npz" % save_path, loss=np.array(result), perplexity_test=perplexity_test)


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('lstm', help='LSTM type. (default: None)', default=None, type=str, **share_param)
    # parser.add_argument('-e', '--epoch', help='Epoch (default: 100)', default=100, type=int, **share_param)
    # parser.add_argument('-b', '--batch', help='Batch (default: 20)', default=20, type=int, **share_param)
    # parser.add_argument('-s', '--step', help='Num steps (default: 35)', default=35, type=int, **share_param)
    # parser.add_argument('-lr', '--lr', help='Learning rate (default: 0.5)', default=0.5, type=float, **share_param)
    # parser.add_argument('-c', '--clip', help='Gradient clip. (default: 10)', default=10, type=float, **share_param)
    # parser.add_argument('-k', '--keep', help='Keep rate. (default: 1.0)', default=1.0, type=float, **share_param)
    # parser.add_argument('-d', '--lrd', help='LR decay (default: 0.8)', default=0.8, type=float, **share_param)
    parser.add_argument('-wd', '--wd', help='Weight decay. (default: 0.0)', default=0.0, type=float, **share_param)
    parser.add_argument('-wt', '--wt', help='Weight tying. (default: False)', default=False, type=float, **share_param)
    parser.add_argument('-ln', '--ln', help='Layer norm. (default: False)', default=False, type=bool, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    _parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    args = get_options(_parser)

    path = "./log/%s" % "_".join(["_".join([key, str(value)]) for key, value in vars(args).items()])

    raw_train, raw_validation, raw_test, vocab = ptb_raw_data("./simple-examples/data")

    iterators = dict()
    for raw_data, key in zip([raw_train, raw_validation, raw_test],
                             ["iter_train_data", "iter_valid_data", "iter_test_data"]):
        iterators[key] = BatchFeeder(batch_size=20, num_steps=35, sequence=raw_data)

    config = {
        "num_steps": 35, "vocab_size": vocab,
        "n_hidden_hyper": 64, "n_embedding_hyper": 4,  # for hypernets
        "recurrent_highway": True, "recurrence_depth": 4,  # for RHN
        "embedding_size": 650, "n_hidden": 650
        }

    _model = LSTMLanguageModel(config, learning_rate=0.5, gradient_clip=10, keep_prob_r=0.5, keep_prob=0.75,
                               type_of_lstm=args.lstm, weight_decay=args.wd, weight_tying=args.wt, layer_norm=args.ln)

    train(_model, max_max_epoch=100, verbose=True, save_path=path, lr_decay=0.8, **iterators)
