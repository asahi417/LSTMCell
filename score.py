import numpy as np
import pandas as pd
from glob import glob


def score():
    df = pd.DataFrame(index=['train', 'valid', 'epoch'])
    for _ckpt in glob('./checkpoint/*/*/statistics.npz'):
        _model = _ckpt.split('/')[2]
        _id = _ckpt.split('/')[3]
        name = '%s (%s)' % (_model, _id)
        stats = np.load(_ckpt)
        train = stats['loss'][:, 0].min()
        valid = stats['loss'][:, 1].min()
        epoch = np.argmin(stats['loss'][:, 1])
        df[name] = [train, valid, epoch]
    return df.T


if __name__ == '__main__':
    _df = score()
    print(_df)
