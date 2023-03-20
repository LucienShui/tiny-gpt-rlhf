"""
As blog of ChatGPT said, there are 3 steps to train a human friendly GPT:
    1. Train a supervised policy, named Supervised fine-tuning model, SFT for simplicity.
    2. Tell model what's good and bad, then train a reward model to follow human's instruction.
    3. Using PPO reinforcement learning algorithm and reward model to make SFT unsupervised learning.
"""
import logging
import os
import pandas as pd
from typing import List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from transformers import GPT2LMHeadModel, BertTokenizer

from training import Net


def init() -> logging.Logger:
    """
    日志写入文件
    :return:
    """
    log_dir: str = 'log'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logging.basicConfig(level=logging.INFO)

    file_handler = logging.FileHandler(os.path.join(log_dir, 'logger.txt'), encoding='utf-8')

    # 让 pytorch lightning 的日志也写入文件，以下代码为了解决一次日志写两遍的问题
    for name, logger in logging.Logger.manager.loggerDict.items():
        if 'pytorch_lightning' in name:
            if isinstance(logger, logging.Logger):
                if len(logger.handlers) > 0 and file_handler not in logger.handlers:
                    logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)
    return logger


def train(dataset: List[str], pretrained: str = None) -> None:
    checkpoint_path: str = 'resource/model/checkpoint'
    trainer_root_path: str = 'resource/model/trainer'
    model_path: str = 'resource/model/saved_model'

    n_epoch: int = 5
    batch_size: int = 2
    # eval_interval: int = 500
    val_examples: int = 100
    config_path: str = 'resource/config/model_config_tiny.json'
    # config_path: str = 'resource/model_config_small.json'
    vocab_path: str = 'gpt_chinese/vocab/vocab.txt'
    warmup_steps: int = 0
    lr: float = 1e-5

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        verbose=True,
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )
    learning_rate_callback = LearningRateMonitor()
    trainer = pl.Trainer(
        default_root_dir=trainer_root_path,
        gradient_clip_val=1,
        max_epochs=n_epoch,
        devices='auto',
        accelerator='auto',
        val_check_interval=None,
        callbacks=[learning_rate_callback, checkpoint_callback],
        precision=16 if torch.cuda.is_available() else 32,
    )

    net = Net(
        dataset,
        batch_size,
        n_epoch,
        config_path=config_path,
        pretrained=pretrained,
        vocab_path=vocab_path,
        warm_up_steps=warmup_steps,
        lr=lr,
        # additional_special_tokens=Preprocessor.additional_special_tokens
    )

    trainer.fit(net)

    net.model.save_pretrained(model_path)
    net.tokenizer.save_pretrained(model_path)


def get_dataset() -> Tuple[List[str], List[str]]:
    """
    外卖评论数据集，来自 https://github.com/SophonPlus/ChineseNlpCorpus 中的 waimai_10k.csv
    包含正向 4000 和负向 8000 条
    将其分割为 2 部分：
        1. 用于预训练，正向 3000 条，负向 8000 条
        2. 用来训练 SFT 模型，正向 1000 条，无负向
    :return: (List[str], List[str])
    """
    df = pd.read_csv('ChineseNlpCorpus/datasets/waimai_10k/waimai_10k.csv')
    sft_df = df.query('label == 1').sample(n=1000)
    pretrain_df = df.drop(index=sft_df.index)

    pretrain_str_list: List[str] = pretrain_df['review'].values.tolist()
    sft_str_list: List[str] = sft_df['review'].values.tolist()
    return pretrain_str_list, sft_str_list


def main():
    # pretrain_str_list, sft_str_list = get_dataset()
    pretrain_str_list, sft_str_list = ['测试'] * 1024, ['测试'] * 1024
    train(pretrain_str_list, 'resource/model/')


if __name__ == '__main__':
    main()
