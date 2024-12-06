from __future__ import absolute_import
import warnings

from .cuhk02_DG import CUHK02_DG
from .cuhk03 import CUHK03
from .cuhk03_DG import CUHK03_DG
from .market1501 import Market1501
from .market1501_DG import Market1501_DG
from .msmt17 import MSMT17
from .msmt17_DG import MSMT17_DG
from .cuhksysu import CUHKSYSU
from .grid import GRID
from .ilids import iLIDS
from .prid import PRID
from .viper import VIPeR


__factory = {
    'cuhk02dg': CUHK02_DG, # full (train+test) dataset for training
    'cuhk03': CUHK03,
    'cuhk03dg': CUHK03_DG, # full (train+test) dataset for training
    'market1501': Market1501,
    'market1501dg': Market1501_DG, # full (train+test) dataset for training
    'msmt17': MSMT17,
    'msmt17dg': MSMT17_DG, # full (train+test) dataset for training
    'cuhksysu': CUHKSYSU,
    'grid': GRID,
    'ilids': iLIDS,
    'prid': PRID,
    'viper': VIPeR,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. 
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
