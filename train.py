import argparse
import functools
import warnings

from src.trainer import TacoTronTrainer
from src.utils.utils import add_arguments, print_arguments, set_seed
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('configs',           str,    'configs/Tacotron2.yml',     '配置文件')
    add_arg('save_model_path',   str,    'models/',                   '模型保存的路径')
    add_arg('seed',              int,    1211,                        '种子值')
    add_arg("use_gpu",           bool,   True,                       '是否使用GPU训练')
    add_arg('resume_model',      str,    None,                        '恢复训练，当为None则不使用预训练模型')
    add_arg('pretrained_model',  str,    None,                        '预训练模型的路径，当为None则不使用预训练模型')
    args = parser.parse_args()
    print_arguments(args=args)

    set_seed(args.seed)
    trainer = TacoTronTrainer(configs=args.configs, use_gpu=args.use_gpu)
    trainer.train(save_model_path=args.save_model_path,
                  resume_model=args.resume_model,
                  pretrained_model=args.pretrained_model)


if __name__ == '__main__':
    main()
