from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def parse_args():
    parser = argparse.ArgumentParser(description='Train an AttnGAN network (CPU only)')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def gen_example(wordtoix, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = f"{cfg.DATA_DIR}/example_filenames.txt"
    data_dic = {}
    with open(filepath, "r", encoding='utf8') as f:
        filenames = f.read().splitlines()
        for name in filenames:
            if not name:
                continue
            txt_path = "../data/birds/example_captions.txt"
            with open(txt_path, "r", encoding='utf8') as ft:
                print('Load from:', name)
                sentences = ft.read().splitlines()
                captions, cap_lens = [], []
                for sent in sentences:
                    if not sent:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if not tokens:
                        continue
                    idxs = [wordtoix[t] for t in tokens if t in wordtoix]
                    if idxs:
                        captions.append(idxs)
                        cap_lens.append(len(idxs))
            max_len = np.max(cap_lens)
            sorted_indices = np.argsort(cap_lens)[::-1]
            captions = [captions[i] for i in sorted_indices]
            cap_lens = np.array([cap_lens[i] for i in sorted_indices])
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i, cap in enumerate(captions):
                cap_array[i, :len(cap)] = cap
            key = os.path.basename(name)
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


if __name__ == "__main__":
    args = parse_args()
    # Load configuration
    if args.cfg_file:
        cfg_from_file(args.cfg_file)
    # Force CPU
    cfg.CUDA = False
    cfg.GPU_ID = -1

    if args.data_dir:
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    # Set seeds
    manual_seed = args.manualSeed if args.manualSeed is not None else 100
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # Setup output directory
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join('..', 'output', f"{cfg.DATASET_NAME}_{cfg.CONFIG_NAME}_{timestamp}")

    # Determine split
    split_dir = 'train' if cfg.TRAIN.FLAG else 'test'
    shuffle = cfg.TRAIN.FLAG

    # Image transforms
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip(),
    ])

    # Data loader
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle, drop_last=True, num_workers=int(cfg.WORKERS)
    )

    # Initialize trainer (CPU)
    device = torch.device('cpu')
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        if cfg.B_VALIDATION:
            algo.sampling(split_dir)
        else:
            gen_example(dataset.wordtoix, algo)
    print('Total time:', time.time() - start_t)
