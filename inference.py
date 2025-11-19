"""
Script that loads in a sg, preprocesses the data and runs GRID inference on it.
"""

from typing import Any, Optional
import math
import torch
import random
from torch import optim, nn
from torchmetrics import Accuracy
import torch.nn.functional as F
import time
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.distributed as dist
from dataloader.dataset import InstructSG_Dataset
from dataloader.dataset_wo_pre import InstructSG
import warnings
import numpy as np
from utils import visualize as vis
from utils import timer
from multiprocessing import set_start_method
import matplotlib.pyplot as plt
import os
from dataloader.data_preprocessor import data_preprocessor as data_pre
from lightning.pytorch.profilers import PyTorchProfiler as PTProfiler
from utils.logging import get_logger
from utils import get_accuracy
from arguments import create_parser, Config
import logging
from train import LitGRID, GRIDLoss
from dataloader.categories import categories as categories_
from dataloader.categories import actions as actions_




time_f=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())

data_path = "./dataset/GRID_Dataset-master/Mini_Dataset/data_reduced"

config_ = Config("hparams.cfg")

def main():

    dataset = InstructSG(config_, data_path=data_path, process_node_feature_method='tokenize')

    indices = torch.arange(len(dataset))

    val_indices = indices[:5]

    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    val_loader = DataLoader(val_dataset, batch_size=config_.batch_size, shuffle=False, drop_last=False, num_workers=config_.num_workers)

    model = LitGRID(config_)

    # create trainer, which we will use only to predict

    trainer = pl.Trainer(max_epochs=config_.max_epoch, 
                            accelerator='gpu', 
                            devices='auto'
                            )

    ckpt_path = "logs/test/version_21/checkpoints/epoch=551.ckpt"


    out = trainer.predict(model=model, dataloaders=val_loader, ckpt_path=ckpt_path, return_predictions=True)


    print(dataset)



if __name__ == "__main__":
    # optional but explicit on macOS:
    # import multiprocessing as mp
    # mp.set_start_method("spawn", force=True)

    main()