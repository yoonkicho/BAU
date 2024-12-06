from __future__ import absolute_import

from .model import *
from .memory import *

__factory = {
    'resnet50': resnet50,
    'mobilenetv2': mobilenetv2,
    'vitbase': vit_base_patch16,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'resnet50', 'mobilenetv2', 'vitbase'.
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    pretrained : bool, optional
        If True, will load imagenet pre-trained weights. Default: True
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)