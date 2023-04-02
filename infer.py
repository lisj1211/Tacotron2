import argparse
import functools
import warnings

from src.predictor import Tacotron2Predictor
from src.utils.utils import add_arguments, print_arguments
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('configs',    str,  'configs/Tacotron2.yml',       "配置文件")
    add_arg('use_gpu',    bool, False,                         "是否使用GPU预测")
    add_arg('model_path', str,  'models/Tacotron2/best_model', "预测模型文件路径")
    add_arg('enhance',    bool, True,                          "对生成的语音是否去噪")
    args = parser.parse_args()
    print_arguments(args=args)

    predictor = Tacotron2Predictor(configs=args.configs,
                                   model_path=args.model_path,
                                   use_gpu=args.use_gpu)
    text = ''
    out_path = './1.wav'
    predictor.predict(sentence=text, output_path=out_path, enhancement=args.enhance)


if __name__ == "__main__":
    main()
