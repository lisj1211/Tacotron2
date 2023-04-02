import json
import os
import shutil
import time
from datetime import timedelta

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_utils.dataset import Tacotron2Dataset, TextMelCollate
from src.models.loss_function import Tacotron2Loss
from src.optimizer.scheduler import WarmupLR, NoamHoldAnnealing, CosineWithWarmup
from src.utils.logger import setup_logger
from src.utils.utils import dict_to_object, print_arguments
from src.models.model import Tacotron2

logger = setup_logger(__name__)


class TacoTronTrainer:
    """TTS Framework"""

    def __init__(self, configs, use_gpu=True):
        """
        :param configs: 配置文件路径或者是yaml读取到的配置参数
        :param use_gpu: 是否使用GPU训练模型
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        # 读取配置文件
        if os.path.exists(configs):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f, Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        else:
            raise ValueError('当前config文件不存在')
        self.configs = dict_to_object(configs)
        self.use_gpu = use_gpu
        self.model = None

    def __setup_dataloader(self):
        """获取训练数据"""
        collate_fn = TextMelCollate(self.configs.model_conf.n_frames_per_step)
        self.train_dataset = Tacotron2Dataset(self.configs.dataset_conf.train_manifest,
                                              self.configs.dataset_conf.mel_manifest_dir)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.configs.train_conf.batch_size,
                                                        collate_fn=collate_fn,
                                                        shuffle=True,
                                                        num_workers=self.configs.train_conf.num_workers,
                                                        drop_last=True)

    def __print_model_params(self):
        """打印模型参数"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info('total params: ' + str(total_params))
        logger.info('trainable params: ' + str(trainable_params))

    def __setup_model(self, is_train=False):
        """获取模型"""
        self.model = Tacotron2(self.configs.model_conf)
        self.model.to(self.device)
        if is_train:
            self.__print_model_params()
            self.criterion = Tacotron2Loss()
            if self.configs.train_conf.enable_amp:
                self.amp_scaler = torch.cuda.amp.GradScaler(init_scale=1024)
            # 获取优化方法
            optimizer = self.configs.optimizer_conf.optimizer
            if optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                                  lr=float(self.configs.optimizer_conf.learning_rate),
                                                  weight_decay=float(self.configs.optimizer_conf.weight_decay))
            elif optimizer == 'AdamW':
                self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                                   lr=float(self.configs.optimizer_conf.learning_rate),
                                                   weight_decay=float(self.configs.optimizer_conf.weight_decay))
            elif optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                                 momentum=self.configs.optimizer_conf.momentum,
                                                 lr=float(self.configs.optimizer_conf.learning_rate),
                                                 weight_decay=float(self.configs.optimizer_conf.weight_decay))
            else:
                raise ValueError(f'不支持优化方法：{optimizer}')
            # 学习率衰减
            scheduler_conf = self.configs.optimizer_conf.scheduler_conf
            scheduler = self.configs.optimizer_conf.scheduler
            if scheduler == 'WarmupLR':
                self.scheduler = WarmupLR(optimizer=self.optimizer, **scheduler_conf)
            elif scheduler == 'NoamHoldAnnealing':
                self.scheduler = NoamHoldAnnealing(optimizer=self.optimizer, **scheduler_conf)
            elif scheduler == 'CosineWithWarmup':
                self.scheduler = CosineWithWarmup(optimizer=self.optimizer, **scheduler_conf)
            else:
                raise Exception(f'不支持学习率衰减方法：{scheduler}')

    def __load_pretrained(self, pretrained_model):
        """加载预训练模型"""
        if pretrained_model is not None:
            if os.path.isdir(pretrained_model):
                pretrained_model = os.path.join(pretrained_model, 'model.pt')
            assert os.path.exists(pretrained_model), f'{pretrained_model} 模型不存在！'
            model_dict = self.model.state_dict()
            model_state_dict = torch.load(pretrained_model)
            # 特征层
            for name, weight in model_dict.items():
                if name in model_state_dict.keys():
                    if list(weight.shape) != list(model_state_dict[name].shape):
                        logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                       format(name, list(model_state_dict[name].shape), list(weight.shape)))
                        model_state_dict.pop(name, None)
                else:
                    logger.warning('Lack weight: {}'.format(name))
            self.model.load_state_dict(model_state_dict, strict=False)
            logger.info(f'成功加载预训练模型：{pretrained_model}')

    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        best_error_rate = 1e3
        save_model_name = f'{self.configs.use_model}'
        last_model_dir = os.path.join(save_model_path, save_model_name, 'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pt'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pt'))):
            # 判断从指定resume_model恢复训练，还是last_model恢复训练
            if resume_model is None:
                resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pt')), '模型参数文件不存在！'
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pt')), '优化方法参数文件不存在！'
            state_dict = torch.load(os.path.join(resume_model, 'model.pt'))
            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(torch.load(os.path.join(resume_model, 'optimizer.pt')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                if 'test_loss' in json_data.keys():
                    best_error_rate = abs(json_data['test_loss'])
            logger.info(f'成功恢复模型参数和优化方法参数：{resume_model}')
        return last_epoch, best_error_rate

    def __save_checkpoint(self, save_model_path, epoch_id, test_loss, best_model=False):
        """保存模型"""
        save_model_name = f'{self.configs.use_model}'
        if best_model:
            model_path = os.path.join(save_model_path, save_model_name, 'best_model')
        else:
            model_path = os.path.join(save_model_path, save_model_name, 'epoch_{}'.format(epoch_id))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pt'))
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pt'))
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            f.write('{{"last_epoch": {}, "test_loss": {}}}'.format(epoch_id, test_loss))
        if not best_model:
            last_model_path = os.path.join(save_model_path, save_model_name, 'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型
            old_model_path = os.path.join(save_model_path, save_model_name, 'epoch_{}'.format(epoch_id - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        logger.info('已保存模型：{}'.format(model_path))

    def __train_epoch(self, epoch_id):
        accum_grad = self.configs.train_conf.accum_grad
        grad_clip = self.configs.train_conf.grad_clip
        train_times, reader_times, batch_times, batch_losses = [], [], [], []
        start = time.time()
        self.model.train()
        for batch_id, batch in enumerate(tqdm(self.train_loader, desc=f'epoch:{epoch_id}')):
            text_padded = batch[0].to(self.device)
            text_lengths = batch[1].to(self.device)
            target_mel = batch[2].to(self.device)
            target_gate = batch[3].to(self.device)
            mel_lengths = batch[4].to(self.device)
            reader_times.append((time.time() - start) * 1000)
            start_step = time.time()

            # 执行模型计算，是否开启自动混合精度
            with torch.cuda.amp.autocast(enabled=self.configs.train_conf.enable_amp):
                outputs = self.model(text_padded, text_lengths, target_mel, mel_lengths)
                loss = self.criterion(outputs, [target_mel, target_gate])
            loss = loss / accum_grad
            batch_losses.append(loss.cpu().detach().numpy())
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                # loss缩放，乘以系数loss_scaling
                scaled = self.amp_scaler.scale(loss)
                scaled.backward()
            else:
                loss.backward()
            # 执行一次梯度计算
            if batch_id % accum_grad == 0:
                # 是否开启自动混合精度
                if self.configs.train_conf.enable_amp:
                    self.amp_scaler.unscale_(self.optimizer)
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    if torch.isfinite(grad_norm):
                        self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.train_step += 1
            batch_times.append((time.time() - start_step) * 1000)

            train_times.append((time.time() - start) * 1000)
            if batch_id % self.configs.train_conf.log_interval == 0:
                logger.info(f'loss: {loss.cpu().detach().numpy():.5f}, '
                            f'learning_rate: {self.scheduler.get_last_lr()[0]:>.8f}, '
                            f'reader_cost: {(sum(reader_times) / len(reader_times) / 1000):.4f}, '
                            f'batch_cost: {(sum(batch_times) / len(batch_times) / 1000):.4f}, ')
                train_times = []
            start = time.time()
        return float(sum(batch_losses) / len(batch_losses))

    def train(self,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None):
        """
        训练模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        """
        # 获取数据
        self.__setup_dataloader()
        logger.info(f'训练数据大小：{len(self.train_dataset)}')
        # 获取模型
        self.__setup_model(is_train=True)
        self.__load_pretrained(pretrained_model=pretrained_model)
        # 加载恢复模型
        last_epoch, best_error_rate = self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)

        test_step, self.train_step = 0, 0
        last_epoch += 1
        # 开始训练
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            epoch_id += 1
            start_epoch = time.time()
            epoch_loss = self.__train_epoch(epoch_id=epoch_id)
            logger.info('=' * 70)
            logger.info('Train result: epoch: {}, time/epoch: {}, loss: {:.5f}'.format(
                epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), epoch_loss))
            logger.info('=' * 70)
            test_step += 1
            # 保存最优模型
            if epoch_loss < best_error_rate:
                best_error_rate = epoch_loss
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id,
                                       test_loss=epoch_loss, best_model=True)
            # 保存模型
            self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, test_loss=epoch_loss)
