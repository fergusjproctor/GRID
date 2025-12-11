"""
Script that loads in a sg, preprocesses the data and runs GRID inference on it, refactored from original plan to make as lightweight as possible before tryng to integrate with VH
"""

import sys
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from flask import Flask, request, jsonify

from dataloader.dataset_wo_pre import InstructSG
from arguments import create_parser, Config
from train import LitGRID 





DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    else:
        return x  # leave non-tensors (ints, strings, etc.) alone



def predictions_to_labels(out, raw_input, instructor_model):
    action, object = out

    act_probs = F.softmax(action, dim=1)  
    act_preds = torch.argmax(act_probs, dim=1)  

    obj_probs = F.softmax(object, dim=1)   
    obj_preds = torch.argmax(obj_probs, dim=1)  

   
    action_idx = act_preds.item() 
    action_label = instructor_model.action_encoder.categories[0][action_idx]

    obj_idx = obj_preds.item()
    node = raw_input["sg_json"]['nodes'][obj_idx]
    node_label = node['attributes']['label'].replace('_', ' ').strip().lower()
    node_label_long = (' ').join([node['type'].replace('_', ' ').strip().lower(),
                                node['attributes']['color'].replace('_', ' ').strip().lower(),
                                node['attributes']['label'].replace('_', ' ').strip().lower()]).strip()
   
    return action_label, node_label, obj_idx


def load_GRID_model(ckpt_path, config):
    model = LitGRID.load_from_checkpoint(ckpt_path, config=config, map_location=DEVICE)
    model.eval()
    return model


def load_INSTRUCTOR_model(config_, arg_):
    instructor_model = InstructSG(arg_ ,config_, data_path=None, process_node_feature_method='tokenize')
    return instructor_model

def load_inference():
    config_ = Config("hparams.cfg")

    arg_, config_ = create_parser(type="inference", print_config_flag=True)
    config_.batch_size = 1
    config_.dataset_size = 1
    
    instructor_model = load_INSTRUCTOR_model(config_, arg_)

    ckpt_path = "logs/test/version_21/checkpoints/epoch=551.ckpt"
    GRID_model = load_GRID_model(ckpt_path, config_)

    return instructor_model, GRID_model


@torch.no_grad()
def run_inference(instructor_model, GRID_model, raw_input):
    # preprocess with instructor
    instructor_model._load_input_data(raw_input["rg_json"], raw_input["sg_json"], raw_input["instr"])
    instructor_model.preprocess_text(in_rg=instructor_model.robot_graph, in_sg=instructor_model.scene_graph, in_instruct=instructor_model.instruct)
    batch = instructor_model.__getitem__(torch.tensor(0))
    batch = default_collate([batch])  # unsure 'batched' even if we're running batch size 1
    batch = move_to_device(batch, DEVICE)

    # run inference with transformer
    outputs = GRID_model.grid(batch)

    # decode into labels
    labels = predictions_to_labels(outputs, raw_input, instructor_model)

    return labels


def main():
    config_ = Config("hparams.cfg")

    arg_, config_ = create_parser(type="inference", print_config_flag=True)
    config_.batch_size = 1
    config_.preprocessed_language = True
    config_.dataset_size = 1

    
    data_path = "./dataset/GRID_Dataset-master/Mini_Dataset/data"


    instructor_model, GRID_model = load_inference()


    while True:
        # user input option 
        scene_id = int(input("Scene ID"))
        instr_id = int(input("Instruction ID"))
        sub_instr_id = int(input("Step ID"))


        # load single I, sg and rg
        instruct_file = os.path.join(data_path, f'scene.{scene_id}.instr.json')
        instr_json = json.load(open(instruct_file, 'r'))
        sg_json = json.load(open(os.path.join(data_path, f'scene.{scene_id}.graphs', f'scene.{scene_id}.instr.{instr_id}.sg.{sub_instr_id}.json'), 'r'))
        rg_json = json.load(open(os.path.join(data_path, f'scene.{scene_id}.graphs', f'scene.{scene_id}.instr.{instr_id}.rg.{sub_instr_id}.json'), 'r'))
        instr = instr_json['commands'][instr_id]['high']


        # Fixed option for debugging
        # sg_json = json.load(open("dataset/grid_scene_scene4.json", 'r'))
        # rg_json = json.load(open("dataset/grid_robot_scene4.json", 'r'))
        # instr = "Please grab a pillow"


        raw_input = {"instr": instr, "rg_json": rg_json, "sg_json": sg_json}
        action, object, object_id = run_inference(instructor_model, GRID_model, raw_input)

        print(f"Predicted action: {action}, object: {object}, object id: {object_id}")
        


if __name__ == "__main__":
    main()

    