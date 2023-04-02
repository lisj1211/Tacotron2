import argparse
import glob
import os

import numpy as np
import librosa
from tqdm import tqdm

from src.utils.utils import print_arguments
from src.data_utils.utils import pinyin_2_phoneme


def wav2feature(wav_file, args):
    wav, _ = librosa.load(wav_file, sr=None, mono=True)
    fbank = librosa.feature.melspectrogram(y=wav,
                                           sr=args.fs,
                                           n_fft=args.n_fft,
                                           win_length=args.win_length,
                                           hop_length=args.hop_length,
                                           n_mels=args.n_mels,
                                           fmin=args.fmin,
                                           fmax=args.fmax)
    log_fbank = librosa.power_to_db(fbank, ref=np.max)
    return log_fbank


def processing_wavs(wav_files, args):
    feats = []
    ids = []
    for file in tqdm(wav_files, desc='featurizer'):
        id_wav = os.path.split(file)[-1][:-4]
        fea = wav2feature(file, args)
        feats.append(fea)
        ids.append(id_wav)

    # 计算特征的均值和方差
    fea_array = np.concatenate(feats, axis=1)  # fea的维度 D*T
    fea_mean = np.mean(fea_array, axis=1, keepdims=True)
    fea_std = np.std(fea_array, axis=1, keepdims=True)

    mel_save_path = os.path.join(args.output_dir, 'mel_features')
    os.makedirs(mel_save_path, exist_ok=True)

    # 对所有的特征进行正则, 并保存
    for feat, id_wav in zip(feats, ids):
        norm_fea = (feat - fea_mean) / fea_std
        fea_name = os.path.join(mel_save_path, id_wav + '.npy')
        np.save(fea_name, norm_fea)

    static_name = os.path.join(mel_save_path, 'static.npy')
    np.save(static_name, np.array([fea_mean, fea_std], dtype=object))


def trans_prosody(args):
    file_trans = os.path.join(args.input_dir, 'ProsodyLabeling', '000001-010000.txt')
    dic_phoneme = {}
    with open(args.vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, index = line.strip().split()
            index = int(index)
            dic_phoneme[word] = index
    is_sentid_line = True
    with open(file_trans, encoding='utf-8') as f, \
            open(os.path.join(args.output_dir, 'train.txt'), 'w') as fw:
        for line in tqdm(f, desc='prosody transition'):
            if is_sentid_line:
                sent_id = line.split()[0]
                words = line.split('\t')[1].strip()
            else:
                sent_phonemes = pinyin_2_phoneme(line, words)

                sent_sent_phonemes_index = ''
                for phonemes in sent_phonemes.split():
                    sent_sent_phonemes_index = sent_sent_phonemes_index + str(dic_phoneme[phonemes]) + ' '

                sent_sent_phonemes_index = sent_sent_phonemes_index + str(dic_phoneme['~'])  # 添加eos
                fw.writelines('|'.join([sent_id, sent_phonemes, sent_sent_phonemes_index]) + '\n')
            is_sentid_line = not is_sentid_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./data', help='raw data dir')
    parser.add_argument('--output_dir', type=str, default='./data', help='data output dir')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab', help='vocabulary path')
    parser.add_argument('--fs', type=int, default=48000, help='sampling_rate')
    parser.add_argument('--n_fft', type=int, default=4096)
    parser.add_argument('--win_length', type=int, default=int(48000 * 0.05))
    parser.add_argument('--hop_length', type=int, default=int(48000 * 0.0125))
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--fmin', type=float, default=0.0)
    parser.add_argument('--fmax', type=float, default=48000 / 2)
    args = parser.parse_args()
    print_arguments(args=args)

    waves = glob.glob(args.input_dir + '/Wave/*.wav')
    processing_wavs(waves, args)
    trans_prosody(args)
