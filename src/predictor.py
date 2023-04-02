import os

import librosa
import numpy as np
import torch
import yaml
import soundfile as sf

from src.infer_utils.utils import generate_text_code, speech_enhance
from src.models.model import Tacotron2
from src.utils.logger import setup_logger
from src.utils.utils import dict_to_object, print_arguments

logger = setup_logger(__name__)


class Tacotron2Predictor:
    def __init__(self,
                 configs=None,
                 model_path=None,
                 use_gpu=True):
        """
        TTS预测工具
        :param configs: 配置文件路径
        :param model_path: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        if not isinstance(configs, str) or not os.path.exists(configs):
            raise ValueError('configs文件不存在')
        with open(configs, 'r', encoding='utf-8') as f:
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        print_arguments(configs=configs)

        self.configs = dict_to_object(configs)
        self.dic_phoneme = None
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.__init_model(model_path)

    def __init_model(self, model_path):
        """加载预训练模型"""
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'model.pt')
        assert os.path.exists(model_path), f"{model_path} 模型不存在！"
        self.model = Tacotron2(self.configs.model_conf)
        model_state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(model_path))
        self.model.eval()

        self.dic_phoneme = {}
        with open(self.configs.dataset_conf.vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                word, index = line.strip().split()
                index = int(index)
                self.dic_phoneme[word] = index

    def predict(self, sentence: str, output_path: str, enhancement=True):
        """
        :param sentence: 待预测文本
        :param output_path: .wav文件输出路径
        :param enhancement: 是否进行去噪处理
        """
        coded_text = generate_text_code(sentence, self.dic_phoneme)
        # text_in = torch.from_numpy(coded_text)
        text_in = torch.tensor(coded_text)
        text_in = text_in.unsqueeze(0).to(self.device)

        with torch.no_grad():
            eval_outputs = self.model.inference(text_in)
            mel_out = eval_outputs[1]
            mel_out = mel_out.squeeze(0)
            mel_out = mel_out.cpu().detach().numpy()

        # 加载统计信息
        file_static = os.path.join(self.configs.dataset_conf.mel_manifest_dir, 'static.npy')
        static_mel = np.load(file_static, allow_pickle=True)

        mean_mel = np.float64(static_mel[0])
        std_mel = np.float64(static_mel[1])

        # 反正则
        generated_mel = mel_out * std_mel + mean_mel

        # 进行解码
        inv_fbank = librosa.db_to_power(generated_mel)
        inv_wav = librosa.feature.inverse.mel_to_audio(inv_fbank,
                                                       sr=self.configs.preprocess_conf.fs,
                                                       n_fft=self.configs.preprocess_conf.n_fft,
                                                       win_length=self.configs.preprocess_conf.win_length,
                                                       hop_length=self.configs.preprocess_conf.hop_length,
                                                       fmin=self.configs.preprocess_conf.fmin,
                                                       fmax=self.configs.preprocess_conf.fmax)
        inv_wav = inv_wav / max(inv_wav)
        if enhancement:
            inv_wav = speech_enhance(wave_data=inv_wav,
                                     n_fft=self.configs.preprocess_conf.n_fft,
                                     hop_length=self.configs.preprocess_conf.hop_length,
                                     win_length=self.configs.preprocess_conf.win_length,
                                     noise_frame=30)
        inv_wav, _ = librosa.effects.trim(inv_wav)
        sf.write(output_path, inv_wav, self.configs.preprocess_conf.fs)
