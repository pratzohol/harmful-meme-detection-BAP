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

    dataset_train = load_dataset(args=args, split='train')
    dataset_val = load_dataset(args=args, split='val')
    dataset_test = load_dataset(args=args, split='test')

    num_cpus = min(args.batch_size, 6)
    collator = CustomCollator(args)

    if not args.eval_mode:
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_cpus, collate_fn=collator)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator)
    else:
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator)


    # create model
    model = CLIPClassifier(args)

    wandb_logger = WandbLogger(project="meme-v2", config=args)
    wandb_logger.experiment.name = args.exp_name
    num_args = {f'param_{n}': p.numel() for n, p in model.named_parameters() if p.requires_grad}
    print("Number of parameters:", num_args)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints', filename=wandb_logger.experiment.name+'-{epoch:02d}',
        monitor="val/auroc", mode='max', verbose=True, save_weights_only=True, save_top_k=1, save_last=False
    )
    trainer = Trainer(
        devices=[args.devices], max_epochs=args.max_epochs, max_steps=args.max_steps, gradient_clip_val=args.gradient_clip_val,
        logger=wandb_logger, log_every_n_steps=args.log_every_n_steps, check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=[checkpoint_callback], deterministic=True
    )
    if not args.eval_mode:
        trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
        trainer.test(ckpt_path='best', dataloaders=dataloader_test)
    else:
        trainer.test(model, ckpt_path=args.checkpoint, dataloaders=dataloader_test)


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
