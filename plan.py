"""
Script that loads in a sg, preprocesses the data and runs GRID inference on it.
"""

from typing import Any, Optional
import math
import torch
import random
import json
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


import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_gpu_mem(tag=""):
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available")
        return
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / 1024**2   # MB
    reserved  = torch.cuda.memory_reserved(device)  / 1024**2   # MB
    print(f"[{tag}] allocated: {allocated:.1f} MB, reserved: {reserved:.1f} MB")



def predictions_to_labels():
    pass

def predict_next_step():
    pass


def main():



    config_.batch_size = 1
    config_.preprocessed_language = True
    config_.dataset_size = 1
    #preprocessed_data_path= "preprocess_data"

    data_path = "./dataset/GRID_Dataset-master/Mini_Dataset/data"
    
    print_gpu_mem("start")
    dataset = InstructSG(arg_ ,config_, data_path=data_path, process_node_feature_method='tokenize')
    print_gpu_mem("after loading INSTRUCTOR")

    while True:
        scene_id = int(input("Scene ID"))
        instr_id = int(input("Instruction ID"))
        sub_instr_id = int(input("Step ID"))


        # load single I, sg and rg
        instruct_file = os.path.join(data_path, f'scene.{scene_id}.instr.json')
        instr_json = json.load(open(instruct_file, 'r'))
        sg_json = json.load(open(os.path.join(data_path, f'scene.{scene_id}.graphs', f'scene.{scene_id}.instr.{instr_id}.sg.{sub_instr_id}.json'), 'r'))
        rg_json = json.load(open(os.path.join(data_path, f'scene.{scene_id}.graphs', f'scene.{scene_id}.instr.{instr_id}.rg.{sub_instr_id}.json'), 'r'))
        instr = instr_json['commands'][instr_id]['high']

        # sg_json = json.load(open("dataset/grid_scene_scene4.json", 'r'))
        # rg_json = json.load(open("dataset/grid_robot_scene4.json", 'r'))

        # instr = "Please grab a pillow"

        print(instr)
        
        dataset._load_input_data(rg_json, sg_json, instr)
        # before loading INSTRUCTOR
        
        

        # just before encode
        torch.cuda.reset_peak_memory_stats(device)
        print_gpu_mem("before INSTRUCTOR encode")

        dataset.preprocess_text(in_rg=dataset.robot_graph, in_sg=dataset.scene_graph, in_instruct=dataset.instruct)
        print_gpu_mem("after INSTRUCTOR encode")
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[INSTRUCTOR] peak memory during encode: {peak_mb:.1f} MB")  
        

        
        #dataset = InstructSG_Dataset(config=config_, data_path=preprocessed_data_path)

        indices = torch.arange(len(dataset))

        val_indices = indices

        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        val_loader = DataLoader(val_dataset, batch_size=config_.batch_size, shuffle=False, drop_last=False, num_workers=0)#config_.num_workers)
        ckpt_path = "logs/test/version_21/checkpoints/epoch=551.ckpt"
        
        #model = LitGRID(config_)
        print_gpu_mem("before loading GRID")
        model = LitGRID.load_from_checkpoint(ckpt_path, config=config_)#,map_location=torch.device("cpu"))  # TODO should i change the locaton of device?
        # create trainer, which we will use only to predict
        print_gpu_mem("after loading GRID")
        trainer = pl.Trainer(max_epochs=config_.max_epoch, 
                                accelerator=arg_.accelerator, 
                                devices='auto', val_check_interval=1000
                                )
        
        model.skip_predict_epoch_end = True # tells model not to run accuracy metrics on prediction at end 


        torch.cuda.reset_peak_memory_stats(device)
        print_gpu_mem("before trainer.predict")
        out = trainer.predict(model=model, dataloaders=val_loader, 
                                    ckpt_path=ckpt_path)
        print_gpu_mem("after trainer.predict")
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[GRID] peak memory during predict: {peak_mb:.1f} MB")
        
        action, object = out[0]

        

        act_probs = F.softmax(action, dim=1)    # softmax over classes
        act_preds = torch.argmax(act_probs, dim=1)  # shape [32], one class per sample

        obj_probs = F.softmax(object, dim=1)    # softmax over classes
        obj_preds = torch.argmax(obj_probs, dim=1)  # shape [32], one class per sample

        print(act_preds)
        print(obj_preds)
        idx = act_preds.item()  # -> 3
        label = dataset.action_encoder.categories[0][idx]
        print(label) 

        node = sg_json['nodes'][obj_preds.item()]
        node_label = (' ').join([node['type'].replace('_', ' ').strip().lower(),
                                    node['attributes']['color'].replace('_', ' ').strip().lower(),
                                    node['attributes']['label'].replace('_', ' ').strip().lower()]).strip()
        print(node_label)

       


    



if __name__ == "__main__":
    # optional but explicit on macOS:
    # import multiprocessing as mp
    # mp.set_start_method("spawn", force=True)
    time_f=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())

    data_path = "./dataset/GRID_Dataset-master/Mini_Dataset/data_reduced"

    config_ = Config("hparams.cfg")

    arg_, config_ = create_parser(type="inference", print_config_flag=True)

    main()