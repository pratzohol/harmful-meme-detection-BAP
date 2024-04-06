import argparse
import os
import wandb
from dataset import CustomCollator, load_dataset
from engine import CLIPClassifier
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import torch
from tqdm import trange
from tqdm import tqdm
import yaml
from datetime import date
import random

import warnings
warnings.filterwarnings("ignore")


def main(args):
    # load dataset
    if args.dataset in ['fb-meme', 'prop']:
        dataset_train = load_dataset(args=args, split='train')
        dataset_val = load_dataset(args=args, split='val')
        dataset_test = load_dataset(args=args, split='val')
    elif args.dataset == 'tamil':
        dataset_train = load_dataset(args=args, split='train')
        dataset_val = load_dataset(args=args, split='test')

    print("Number of training examples:", len(dataset_train))
    print("Number of validation examples:", len(dataset_val))

    if args.dataset == 'fb-meme':
        print("Number of test examples:", len(dataset_test))
        print("Number of validation examples :", len(dataset_val))
        print("Number of test examples :", len(dataset_test))
    elif args.dataset == 'prop':
        print("Number of test examples:", len(dataset_test))

    print("Sample item:", dataset_train[0])
    print("Image size:", dataset_train[0]['image'].size)


    num_cpus = min(args.batch_size, 16)

    collator = CustomCollator(args)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_cpus, collate_fn=collator)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator)
    if args.dataset in ['fb-meme', 'prop']:
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator)

    # create model
    model = CLIPClassifier(args)

    if args.dataset == 'prop':
        monitor = "val/f1"
        project = "meme-prop-v2"
    elif args.dataset == 'tamil':
        monitor = "val/f1"
        project = "meme-tamil-v2"
    else:
        monitor = "val/auroc"
        project = "meme-v2"

    wandb_logger = WandbLogger(project=project, config=args)
    wandb_logger.experiment.name = args.exp_name
    num_args = {f'param_{n}': p.numel() for n, p in model.named_parameters() if p.requires_grad}
    print("Number of parameters:", num_args)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints', filename=wandb_logger.experiment.name+'-{epoch:02d}',
        monitor=monitor, mode='max', verbose=True, save_weights_only=True, save_top_k=1, save_last=False
    )
    trainer = Trainer(
        devices=args.devices, max_epochs=args.max_epochs, max_steps=args.max_steps, gradient_clip_val=args.gradient_clip_val,
        logger=wandb_logger, log_every_n_steps=args.log_every_n_steps, check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=[checkpoint_callback], deterministic=True
    )

    trainer.fit(model, train_dataloaders=dataloader_train,val_dataloaders=dataloader_val)
    if args.dataset == 'fb-meme':
        trainer.test(ckpt_path='best', dataloaders=[dataloader_val, dataloader_test])
    elif args.dataset == 'tamil':
        trainer.test(ckpt_path='best', dataloaders=[dataloader_val, dataloader_val])
    elif args.dataset == 'prop':
        trainer.test(ckpt_path='best', dataloaders=[dataloader_val, dataloader_test])


if __name__ == '__main__':
    # Using config.yaml to get args
    with open("config.yaml", "r") as f:
        args = yaml.safe_load(f)

    # control random seed
    if args["seed"] is not None:
        SEED = args['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    args = argparse.Namespace(**args)
    main(args)
