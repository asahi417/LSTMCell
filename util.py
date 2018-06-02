import logging
import json
import os
from glob import glob


def create_log(name=None, save=False):
    """Logging. If `name` is None, only show in terminal and else save log file in `name`
     Usage
    -------------------
    logger.info(message)
    logger.error(error)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if save and name is not None:  # handler for logger file
        if name is not None and os.path.exists(name):
            os.remove(name)
        handler = logging.FileHandler(name)
        handler.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
        logger.addHandler(handler)
    else:  # handler for standard output
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
        logger.addHandler(handler)
    # logger.info('%i logger handlers' % len(logger.handlers))
    return logger


def checkpoint_version(config: dict,
                       path_to_checkpoints: str):
    """ Checkpoint versioner
    :param config:
    :param path_to_checkpoints: `./checkpoint/lam
    :return:
    """
    checkpoints = glob('%s/*' % path_to_checkpoints)

    # check if there are any checkpoints with same hyperparameters
    target_checkpoints = None
    for i in checkpoints:
        with open('%s/hyperparameters.json' % i, 'r') as f:
            json_dict = json.load(f)
        if config == json_dict:
            if target_checkpoints is not None:
                raise ValueError('Checkpoints are duplicated `%s`' % i)
            else:
                target_checkpoints = i

    if target_checkpoints is None:
        new_checkpoint_id = len(glob('%s/*' % path_to_checkpoints))
        new_checkpoint_path = '%s/%i' % (path_to_checkpoints, new_checkpoint_id)
        os.makedirs(new_checkpoint_path, exist_ok=True)
        with open('%s/hyperparameters.json' % new_checkpoint_path, 'w') as outfile:
            json.dump(config, outfile)
    else:
        new_checkpoint_path = target_checkpoints
    create_log('%s/logger.log' % new_checkpoint_path, True)
    logger = create_log('%s/logger.log' % new_checkpoint_path, False)
    return new_checkpoint_path, logger
