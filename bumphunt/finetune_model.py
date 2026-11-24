import flexibledataset as fd
import bump_hunt as bh

import racnn_model as rcn_m
import racnn_trainer as rcn_t

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import itertools
import time
import os
import sys
import random
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader,TensorDataset

device = bh.device
#NOISE = True

def _load_ft_split(data_path, mr_type, split):
    """Load <data_path>/<mr_type>/<split>.h5 (release format), else the legacy <mr_type>_ft_<split>_dataset.pt."""
    h5 = os.path.join(data_path, mr_type, f"{split}.h5")
    return fd.load_dataset(h5 if os.path.exists(h5) else os.path.join(data_path, f"{mr_type}_ft_{split}_dataset.pt"))


def _load_state(run_dir, name="best_model"):
    """Load a state_dict from <run_dir>/<name>.safetensors (release format), else <name>.pt."""
    st = os.path.join(run_dir, f"{name}.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file
        return load_file(st, device=str(device))
    return torch.load(os.path.join(run_dir, f"{name}.pt"), weights_only=False, map_location=device)


if __name__ == "__main__":
    if len(sys.argv) < 9:
        print("Usage: python finetune_model.py <model_type> <data_path> <save_path> <noise> <PSL> <loaded_model_path> <mr_type> <freeze>")
        sys.exit(1)

    
    model_type = sys.argv[1]
    data_path  = sys.argv[2]
    save_path  = sys.argv[3]
    NOISE      = eval(sys.argv[4])
    PSL        = eval(sys.argv[5])
    load_path  = sys.argv[6]
    mr_type    = sys.argv[7]
    freeze     = eval(sys.argv[8])   # was the raw string, so "False" was truthy and every original run was frozen
    assert isinstance(freeze, bool), freeze

    assert save_path != load_path
    assert mr_type in ["mrd","mrs","mr1s","mr100s",'mr1s_high','mr15s_high','mr100s_high','mr15d_high','mr1s_low','mr3s_low','mr5s_low','mr10s_low','mr1s_high','mr3s_high','mr5s_high','mr10s_high','mr50s_low','mr3d_low','mr5d_low','mr10d_low','mr50d_low']
    
    print("NOISE MODE :", NOISE)
    print("PSL MODE   :", PSL)
    print("load_path  :", load_path)
    print("mr_type    :", mr_type)
    print("freeze     :", freeze)
    
    train_dataset  = _load_ft_split(data_path, mr_type, "train")
    val_dataset    = _load_ft_split(data_path, mr_type, "val")

    train_loader   = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader     = DataLoader(val_dataset, batch_size=2048)

    epochs           = 500      # 10 
    warmup_epochs    = 10       # 5   
    flow_eval_epochs = 30       # 3   
    patience         = 30       # 10  

    if model_type.lower() == "CNNFlows".lower():
        cnnflows     = rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50).double()
        cnnflows.load_state_dict(_load_state(load_path, "best_model"))
        if freeze : 
            for param in cnnflows.cnn_head.parameters():
                param.requires_grad = False
        optimizer    = torch.optim.Adam(filter(lambda p: p.requires_grad, cnnflows.parameters()), lr=0.001)
        for name, param in cnnflows.named_parameters():
            if not param.requires_grad:
                print("Frozen:", name)
            else:
                print("Okay:", name)

        #data_np = np.stack(list(val_dataset.flow_outputs.values()))
        #data_np = data_np[:, np.all(np.isfinite(data_np), axis=0)]  # keep only columns without NaNs/Infs
        #data_tensor = torch.tensor(data_np.T).to("cpu")
        #cnnflows.pdf.init_params(data=data_tensor)

        cnnflows, out_dict = rcn_t.train_loop_nf(
        cnnflows,
        optimizer,
        train_loader,
        epochs=epochs,
        val_loader=val_loader,
        clipnorm=1,
        patience=patience,
        save_dir_path=save_path,
        flow_eval_epochs = flow_eval_epochs,
        plot=False,
        cnn_mode=True,
        noisy=NOISE,
        psl=PSL)

    elif model_type.lower() == "RACNNFlows".lower():
        racnnflows     = rcn_m.RACNNFlows(aux_input_size=3, cond_dim=25,hidden_size=25,hidden_size2=50).double()
        racnnflows.load_state_dict(_load_state(load_path, "best_model"))
        if freeze : 
            for param in racnnflows.cnn_head.parameters():
                param.requires_grad = False
            for param in racnnflows.aux_head.parameters():
                param.requires_grad = False                
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, racnnflows.parameters()), lr=0.001)
        for name, param in racnnflows.named_parameters():
            if not param.requires_grad:
                print("Frozen:", name)
            else:
                print("Okay:", name)

        #data_np = np.stack(list(val_dataset.flow_outputs.values()))
        #data_np = data_np[:, np.all(np.isfinite(data_np), axis=0)]  # keep only columns without NaNs/Infs
        #data_tensor = torch.tensor(data_np.T).to("cpu")
        #racnnflows.pdf.init_params(data=data_tensor)

        racnnflows, out_dict = rcn_t.train_loop_nf(
        racnnflows,
        optimizer,
        train_loader,
        epochs=epochs,
        val_loader=val_loader,
        clipnorm=1,
        patience=patience,
        save_dir_path=save_path,
        flow_eval_epochs = flow_eval_epochs,
        plot=False,
        cnn_mode=False,
        noisy=NOISE,
        psl=PSL)

    elif model_type.lower() == "CNNClass".lower():
        cnn_class       = rcn_m.CNNClass(cond_dim=25,hidden_size=25,hidden_size2=50).double()
        cnn_class.load_state_dict(_load_state(load_path, "best_model"))
        if freeze : 
            for param in cnn_class.cnn_head.parameters():
                param.requires_grad = False
        optimizer_class = torch.optim.Adam(filter(lambda p: p.requires_grad, cnn_class.parameters()), lr=0.001)
        for name, param in cnn_class.named_parameters():
            if not param.requires_grad:
                print("Frozen:", name)
            else:
                print("Okay:", name)
        cnn_class, out_dict_cnn = rcn_t.train_loop_class(
        cnn_class,
        optimizer_class,
        train_loader,
        epochs=epochs,
        val_loader=val_loader,
        clipnorm=1,
        patience=patience,
        save_dir_path=save_path,
        cnn_mode=True,
        psl=PSL)

    elif model_type.lower() == "RACNNClass".lower():
        racnn_class     = rcn_m.RACNNClass(aux_input_size=3, cond_dim=25,hidden_size=25,hidden_size2=50).double()
        racnn_class.load_state_dict(_load_state(load_path, "best_model"))
        if freeze : 
            for param in racnn_class.cnn_head.parameters():
                param.requires_grad = False
            for param in racnn_class.aux_head.parameters():
                param.requires_grad = False                  
        optimizer_class = torch.optim.Adam(filter(lambda p: p.requires_grad, racnn_class.parameters()), lr=0.001)
        for name, param in racnn_class.named_parameters():
            if not param.requires_grad:
                print("Frozen:", name)
            else:
                print("Okay:", name)
        racnn_class, out_dict1 = rcn_t.train_loop_class(
        racnn_class,
        optimizer_class,
        train_loader,
        epochs=epochs,
        val_loader=val_loader,
        clipnorm=1,
        patience=30,
        save_dir_path=save_path,
        cnn_mode=False,
        psl=PSL)

    elif model_type.lower() == "CNNClassiFlows".lower():
        cnnclassiflows     = rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50).double()
        cnnclassiflows.load_state_dict(_load_state(load_path, "best_model"))
        if freeze : 
            for param in cnnclassiflows.cnn_head.parameters():
                param.requires_grad = False
        optimizer_cf = torch.optim.Adam(filter(lambda p: p.requires_grad, cnnclassiflows.parameters()), lr=0.001)
        for name, param in cnnclassiflows.named_parameters():
            if not param.requires_grad:
                print("Frozen:", name)
            else:
                print("Okay:", name)

        #data_np = np.stack(list(val_dataset.flow_outputs.values()))
        #data_np = data_np[:, np.all(np.isfinite(data_np), axis=0)]  # keep only columns without NaNs/Infs
        #data_tensor = torch.tensor(data_np.T).to("cpu")
        #cnnclassiflows.pdf.init_params(data=data_tensor)


        cnnclassiflows, out_dict = rcn_t.train_loop_merged_model(
        cnnclassiflows,
        optimizer_cf,
        train_loader,
        epochs=epochs,
        val_loader=val_loader,
        clipnorm=1,
        patience=patience,
        save_dir_path=save_path,
        warmup_epochs = warmup_epochs ,
        flow_eval_epochs = flow_eval_epochs,
        plot=False,
        cnn_mode=True,
        noisy=NOISE,
        psl=PSL
        )

    elif model_type.lower() == "RACNNClassiFlows".lower():
        racnnclassiflows        = rcn_m.RACNNClassiFlows(aux_input_size=3, cond_dim=25,hidden_size=25,hidden_size2=50).double()
        racnnclassiflows.load_state_dict(_load_state(load_path, "best_model"))
        if freeze : 
            for param in racnnclassiflows.cnn_head.parameters():
                param.requires_grad = False
            for param in racnnclassiflows.aux_head.parameters():
                param.requires_grad = False     
            for param in racnnclassiflows.mlp_hidden_layer.parameters():
                param.requires_grad = False
            for param in racnnclassiflows.mlp_hidden_layer2.parameters():
                param.requires_grad = False
            for param in racnnclassiflows.mlp_output_layer.parameters():
                param.requires_grad = False
            for param in racnnclassiflows.merger_layer.parameters():
                param.requires_grad = False
            for param in racnnclassiflows.hidden1.parameters():
                param.requires_grad = False
            for param in racnnclassiflows.hidden2.parameters():
                param.requires_grad = False
            warmup_epochs = 0 # need to mod loss for classi term?

        optimizer_cf    = torch.optim.Adam(filter(lambda p: p.requires_grad, racnnclassiflows.parameters()), lr=0.001)
        for name, param in racnnclassiflows.named_parameters():
            if not param.requires_grad:
                print("Frozen:", name)
            else:
                print("Okay:", name)

        #data_np = np.stack(list(val_dataset.flow_outputs.values()))
        #data_np = data_np[:, np.all(np.isfinite(data_np), axis=0)]  # keep only columns without NaNs/Infs
        #data_tensor = torch.tensor(data_np.T).to("cpu")
        #racnnclassiflows.pdf.init_params(data=data_tensor)
        
        racnnclassiflows, out_dict = rcn_t.train_loop_merged_model(
        racnnclassiflows,
        optimizer_cf,
        train_loader,
        epochs=epochs,
        val_loader=val_loader,
        clipnorm=1,
        patience=patience,
        save_dir_path=save_path,
        warmup_epochs = warmup_epochs ,
        flow_eval_epochs = flow_eval_epochs,
        plot=False,
        cnn_mode=False,
        noisy=NOISE,
        psl=PSL)

    else:
        raise KeyError
