## Introduction

基于Pytorch的Tacotron2 TTS模型实现

## Requirements

* python 3.7
* torch = 1.12.1
* numpy
* tqdm
* yaml
* librosa
* xpinyin
* soundfile

## DataSet

数据集为[biaobei](https://www.data-baker.com/open_source.html)数据集, 包含10000句中文标准女声音库

## Train

* 训练config文件位于`configs/Tacotron2.ymal`, 可根据需要修改相关参数

* 数据预处理, 下载数据压缩文件后解压至`data`文件夹下

```
    python preprocess.py
```

* 训练模型

```
    python train.py
```

* 模型预测

```
    python infer.py
```

## Analysis

测试结果总体效果还行, 经过音频增强后仍然存在少量杂音. 另外因为训练设置每一timestep生成3frame, 虽然训练速度快,
但最后声音会存在拉长, 如果不在乎训练速度可调整每一timestep生成一frame. 另外生成梅尔频谱后可换更好的vocoder. 

## Reference

[1] [TTS-Tacotron2](https://gitee.com/yuhong-ldu/speech-processing/tree/master/TTS-Tacotron2)
