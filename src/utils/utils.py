import distutils.util
import random

import numpy as np
import torch

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def print_arguments(args=None, configs=None):
    if args:
        logger.info('----------- 额外配置参数 -----------')
        for arg, value in sorted(vars(args).items()):
            logger.info(f'{arg}: {value}')
        logger.info('------------------------------------------------')
    if configs:
        logger.info('----------- 配置文件参数 -----------')
        for arg, value in sorted(configs.items()):
            if isinstance(value, dict):
                logger.info(f'{arg}:')
                for a, v in sorted(value.items()):
                    if isinstance(v, dict):
                        logger.info(f'\t{a}:')
                        for a1, v1 in sorted(v.items()):
                            logger.info(f'\t\t{a1}: {v1}')
                    else:
                        logger.info(f'\t{a}: {v}')
            else:
                logger.info(f'{arg}: {value}')
        logger.info('------------------------------------------------')


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument('--' + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = Dict()
    for k, v in dict_obj.items():
        inst[k] = dict_to_object(v)
    return inst


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
                