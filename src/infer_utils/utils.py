import os.path
from typing import Union

import librosa
import numpy as np
from xpinyin import Pinyin

from src.data_utils.utils import pinyin_2_phoneme


def generate_text_code(words, dic_phoneme):
    new_words = words.replace('#', '')

    new_words = ''.join([i for i in new_words if not i.isdigit()])
    p = Pinyin()
    out_pinyin = p.get_pinyin(new_words, ' ', tone_marks='numbers')
    sent_phonemes = pinyin_2_phoneme(out_pinyin, words)
    coded_text = [dic_phoneme[phonemes] for phonemes in sent_phonemes.split()]
    coded_text.append(dic_phoneme['~'])  # 添加eos
    return coded_text


def speech_enhance(*,
                   wave_data: Union[str, np.ndarray],
                   n_fft: int,
                   hop_length: int,
                   win_length: int,
                   noise_frame: int = 30,
                   alpha: int = 4,
                   beta: float = 0.0001,
                   gamma: int = 1,
                   method: int = 3):
    """
    语音增强，减小生成语音文件的噪声
    Args:
        wave_data: 语音文件路径，或已经读取完毕的numpy格式
        n_fft: 傅里叶变换参数
        hop_length: 傅里叶变换参数
        win_length: 傅里叶变换参数
        noise_frame: 前多少帧当作噪音信号
        alpha: 过减法参数
        beta: 过减法参数
        gamma: 过减法参数
        method: 1 表示谱减法，2 表示过减法，3 表示平滑法，default=3

    Returns:
        enhanced_wav: np.ndarray
    """
    if isinstance(wave_data, str):
        if not os.path.exists(wave_data):
            raise FileNotFoundError(f'Input wav file path is incorrect')
        noisy_wav, _ = librosa.load(wave_data, sr=None)
    elif isinstance(wave_data, np.ndarray):
        noisy_wav = wave_data
    else:
        raise ValueError(f'Wave_data only support `[str, np.ndarray]`')

    if method == 1:
        enhanced_wav = _sub_spec1(noisy_wav, n_fft, hop_length, win_length, noise_frame)
    elif method == 2:
        enhanced_wav = _sub_spec2(noisy_wav, n_fft, hop_length, win_length, noise_frame, alpha, beta, gamma)
    elif method == 3:
        enhanced_wav = _sub_spec3(noisy_wav, n_fft, hop_length, win_length, noise_frame, alpha, beta, gamma)
    else:
        raise ValueError(f'method only support `[1, 2, 3]`')

    return enhanced_wav


def _sub_spec1(wav_data, n_fft, hop_length, win_length, noise_frame):
    spec_raw = librosa.stft(wav_data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)  # D x T
    D, T = np.shape(spec_raw)
    mag_raw = np.abs(spec_raw)
    phase_raw = np.angle(spec_raw)
    power_raw = mag_raw ** 2
    assert noise_frame < T
    mag_noise = np.mean(np.abs(spec_raw[:, :noise_frame]), axis=1, keepdims=True)
    power_noise = mag_noise ** 2
    power_noise = np.tile(power_noise, [1, T])

    power_enhanced = power_raw - power_noise
    power_enhanced[power_enhanced < 0] = 0
    mag_enhanced = np.sqrt(power_enhanced)

    spec_enhanced = mag_enhanced * np.exp(1j * phase_raw)
    enhanced_wav = librosa.istft(spec_enhanced, hop_length=hop_length, win_length=win_length)
    return enhanced_wav


def _sub_spec2(wav_data, n_fft, hop_length, win_length, noise_frame, alpha, beta, gamma):
    spec_raw = librosa.stft(wav_data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)  # D x T
    D, T = np.shape(spec_raw)
    mag_raw = np.abs(spec_raw)
    phase_raw = np.angle(spec_raw)
    power_raw = mag_raw ** 2
    assert noise_frame < T
    mag_noise = np.mean(np.abs(spec_raw[:, :noise_frame]), axis=1, keepdims=True)
    power_noise = mag_noise ** 2
    power_noise = np.tile(power_noise, [1, T])

    power_enhanced = np.power(power_raw, gamma) - alpha * np.power(power_noise, gamma)
    power_enhanced = np.power(power_enhanced, 1 / gamma)
    # 对于过小的值用 beta* Power_noise 替代
    mask = (power_enhanced >= beta * power_noise) - 0
    power_enhanced = mask * power_enhanced + beta * (1 - mask) * power_noise
    mag_enhanced = np.sqrt(power_enhanced)

    spec_enhanced = mag_enhanced * np.exp(1j * phase_raw)
    enhanced_wav = librosa.istft(spec_enhanced, hop_length=hop_length, win_length=win_length)
    return enhanced_wav


def _sub_spec3(wav_data, n_fft, hop_length, win_length, noise_frame, alpha, beta, gamma):
    spec_raw = librosa.stft(wav_data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)  # D x T
    D, T = np.shape(spec_raw)
    mag_raw = np.abs(spec_raw)
    phase_raw = np.angle(spec_raw)
    assert noise_frame < T
    mag_noise = np.mean(np.abs(spec_raw[:, :noise_frame]), axis=1, keepdims=True)
    power_noise = mag_noise ** 2
    power_noise = np.tile(power_noise, [1, T])

    # 平滑
    mag_smoothed = np.copy(mag_raw)
    k = 1
    for t in range(k, T - k):
        mag_smoothed[:, t] = np.mean(mag_raw[:, t - k:t + k + 1], axis=1)
    power_smoothed = mag_smoothed ** 2

    # 过减法去噪
    power_enhanced = np.power(power_smoothed, gamma) - alpha * np.power(power_noise, gamma)
    power_enhanced = np.power(power_enhanced, 1 / gamma)
    mask = (power_enhanced >= beta * power_noise) - 0
    power_enhanced = mask * power_enhanced + beta * (1 - mask) * power_noise
    mag_enhanced = np.sqrt(power_enhanced)

    # 计算最大噪声残差
    max_residual_error = np.max(np.abs(spec_raw[:, :noise_frame]) - mag_noise, axis=1)
    mag_enhanced_new = np.copy(mag_enhanced)
    k = 1
    for t in range(k, T - k):
        index = np.where(mag_enhanced[:, t] < max_residual_error)[0]
        temp = np.min(mag_enhanced[:, t - k:t + k + 1], axis=1)
        mag_enhanced_new[index, t] = temp[index]
        
    spec_enhanced = mag_enhanced_new * np.exp(1j * phase_raw)
    enhanced_wav = librosa.istft(spec_enhanced, hop_length=hop_length, win_length=win_length)
    return enhanced_wav
