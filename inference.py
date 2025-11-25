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
from dataloader.data_preprocessor import data_preprocessor
from arguments import create_parser




time_f=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())

data_path = "./dataset/GRID_Dataset-master/Mini_Dataset/data_reduced"

config_ = Config("hparams.cfg")

arg_, config_ = create_parser(type="inference", print_config_flag=True)

def main():
    config_.batch_size = 4
    config_.preprocessed_language = True
    config_.dataset_size = 4
    #preprocessed_data_path= "preprocess_data"


    dataset = InstructSG(arg_ ,config_, data_path=data_path, process_node_feature_method='tokenize')
    dataset._load_data()
    dataset.preprocess_text(dataset.robot_graph, dataset.scene_graph, dataset.instruct)
    
    print(dataset.action_encoder)
    print(dataset.object_dict)
    print(dataset.object_id_mask)
    
    #dataset = InstructSG_Dataset(config=config_, data_path=preprocessed_data_path)

    indices = torch.arange(len(dataset))

    val_indices = indices[:5]

    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    val_loader = DataLoader(val_dataset, batch_size=config_.batch_size, shuffle=False, drop_last=False, num_workers=0)#config_.num_workers)
    ckpt_path = "logs/test/version_21/checkpoints/epoch=551.ckpt"
    model = LitGRID(config_)

    model.load_from_checkpoint(ckpt_path, config=config_,map_location=torch.device("cpu"))
    # create trainer, which we will use only to predict

    trainer = pl.Trainer(max_epochs=config_.max_epoch, 
                            accelerator=arg_.accelerator, 
                            devices='auto', val_check_interval=1000
                            )

    out = trainer.predict(model=model, dataloaders=val_loader, 
                                ckpt_path=ckpt_path)
    
    action, object = out[0]

    

    act_probs = F.softmax(action, dim=1)    # softmax over classes
    act_preds = torch.argmax(act_probs, dim=1)  # shape [32], one class per sample

    obj_probs = F.softmax(object, dim=1)    # softmax over classes
    obj_preds = torch.argmax(obj_probs, dim=1)  # shape [32], one class per sample

    print(act_preds)
    print(obj_preds)


    



if __name__ == "__main__":
    # optional but explicit on macOS:
    # import multiprocessing as mp
    # mp.set_start_method("spawn", force=True)

    main()