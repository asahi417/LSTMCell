import os
import argparse
import toml
import json
from glob import glob
from data_reader import ptb_raw_data, BatchFeeder
from model_language import LanguageModel


def checkpoint_version(checkpoint_dir: str,
                       config: dict = None,
                       version: int = None):
    """ Checkpoint versioner:
    - return checkpoint dir, which has same hyper parameter (config)
    - return checkpoint dir, which corresponds to the version
    - return new directory
    :param config:
    :param checkpoint_dir: `./checkpoint/lam
    :param version: number of checkpoint
    :return:
    """

    if version is not None:
        checkpoints = glob('%s/v%i/hyperparameters.json' % (checkpoint_dir, version))
        if len(checkpoints) == 0:
            raise ValueError('No checkpoint: %s, %s' % (checkpoint_dir, version))
        elif len(checkpoints) > 1:
            raise ValueError('Multiple checkpoint found: %s, %s' % (checkpoint_dir, version))
        else:
            parameter = json.load(open(checkpoints[0]))
            target_checkpoints_dir = checkpoints[0].replace('/hyperparameters.json', '')
            return target_checkpoints_dir, parameter

    elif config is not None:
        # check if there are any checkpoints with same hyperparameters
        target_checkpoints = []
        for i in glob('%s/*' % checkpoint_dir):
            json_dict = json.load(open('%s/hyperparameters.json' % i))
            if config == json_dict:
                target_checkpoints.append(i)
        if len(target_checkpoints) == 1:
            return target_checkpoints[0], config
        elif len(target_checkpoints) == 0:
            new_checkpoint_id = len(glob('%s/*' % checkpoint_dir))
            new_checkpoint_path = '%s/v%i' % (checkpoint_dir, new_checkpoint_id)
            os.makedirs(new_checkpoint_path, exist_ok=True)
            with open('%s/hyperparameters.json' % new_checkpoint_path, 'w') as outfile:
                json.dump(config, outfile)
            return new_checkpoint_path, config
        else:
            raise ValueError('Checkpoints are duplicated')


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-m', '--model', help='LSTM type', required=True, type=str, **share_param)
    parser.add_argument('--max_max_epoch', help='max max epoch', required=True, type=int, **share_param)
    parser.add_argument('--max_epoch', help='max epoch', type=int, **share_param)
    parser.add_argument('--decay', help='max epoch', type=float, **share_param)
    parser.add_argument('--lr', help='learning rate', type=float, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # checkpoint
    _parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    args = get_options(_parser)
    _parameter = toml.load(open('./hyperparameters/%s.toml' % args.model))
    _checkpoint_dir, _ = checkpoint_version('./checkpoint/%s' % args.model, _parameter)

    # data
    raw_train, raw_validation, raw_test, vocab = ptb_raw_data("./simple-examples/data")

    iterators = dict()
    for raw_data, key in zip([raw_train, raw_validation, raw_test], ["batcher_train", "batcher_valid", "batcher_test"]):
        iterators[key] = BatchFeeder(batch_size=_parameter['batch_size'],
                                     num_steps=_parameter['config']['num_steps'],
                                     sequence=raw_data)

    model = LanguageModel(checkpoint_dir=_checkpoint_dir, **_parameter)
    model.train(max_max_epoch=args.max_max_epoch,
                max_epoch=args.max_epoch if args.max_epoch is not None else args.max_max_epoch,
                verbose=True,
                learning_rate=args.lr,
                lr_decay=args.decay,
                **iterators)
