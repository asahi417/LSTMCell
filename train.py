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
import argparse
import toml

import numpy as np
from time import time

from data_reader import ptb_raw_data, BatchFeeder
from model_language import LSTMLanguageModel
from util import checkpoint_version


def train(model,
          max_max_epoch,
          max_epoch,
          iter_train_data,
          iter_valid_data,
          iter_test_data,
          save_path,
          logger,
          lr_decay=None,
          verbose=False):
    """ Train model based on mini-batch of input data.

    :param int max_max_epoch: max epoch
    :param model: Model instance.
    :param iter_train_data: Data iterator.
    :param iter_valid_data: Data iterator.
    :param iter_test_data: Data iterator.
    :param str save_path: Path to save
    :param logger: logging instance
    :param float lr_decay: learning rate will be divided by lr_decay each 100 epoch
    :param bool verbose: Show loss in each iteration.
    """

    num_gpu = 0  # number of gpu
    logger.info("epoch (%i), sequence length (%i), batch size(%i), total variables(%i)" %
                (max_max_epoch, iter_train_data.data_size, iter_train_data.batch_size, model.total_variable_number))

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

    # Test
    length, loss = 0, 0.0
    for _in, _out in iter_test_data:
        feed_dict = dict(((model.inputs, _in), (model.targets, _out), (model.is_train, False)))
        val = model.sess.run([model.loss], feed_dict=feed_dict)
        loss += val[0]
        length += iter_train_data.num_steps
    perplexity_test = np.exp(loss / length)

    logger.info("test perplexity %0.3f" % perplexity_test)

    model.saver.save(model.sess, "%s/model.ckpt" % save_path)
    np.savez("%s/statistics.npz" % save_path, loss=np.array(result), perplexity_test=perplexity_test)


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-m', '--model', help='LSTM type. (default: None)', required=True, type=str, **share_param)
    parser.add_argument('-e', '--epoch', help='Epoch. (default: 50)', default=50, type=int, **share_param)
    parser.add_argument('-t', '--type', help='Problem type', required=True, type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # checkpoint
    _parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    args = get_options(_parser)
    config = toml.load(open('hyperparameters/%s/%s.toml' % (args.type, args.model)))
    checkpoint, _logger = checkpoint_version(config, 'checkpoint/%s' % args.model)

    # data
    raw_train, raw_validation, raw_test, vocab = ptb_raw_data("./simple-examples/data")

    iterators = dict()
    for raw_data, key in zip([raw_train, raw_validation, raw_test], ["iter_train_data", "iter_valid_data", "iter_test_data"]):
        iterators[key] = BatchFeeder(batch_size=config['batch_size'],
                                     num_steps=config['config']['num_steps'],
                                     sequence=raw_data)

    model_instance = LSTMLanguageModel(**config, logger=_logger)  # if args.t == 'language' else None
    train(model_instance,
          max_max_epoch=args.epoch,
          max_epoch=10,  # int(np.ceil(max_max_epoch / 10))
          verbose=True,
          save_path=checkpoint,
          lr_decay=0.1,
          logger=_logger,
          **iterators)
