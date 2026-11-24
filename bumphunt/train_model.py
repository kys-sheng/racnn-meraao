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

def _load_split(data_path, split):
    """Load <data_path>/<split>.h5 (release format), else the legacy <split>_dataset.pt."""
    h5 = os.path.join(data_path, f"{split}.h5")
    return fd.load_dataset(h5 if os.path.exists(h5) else os.path.join(data_path, f"{split}_dataset.pt"))


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print("Usage: python train_model.py <model_type> <data_path> <save_path> <noise> [PSL]")
        sys.exit(1)

    model_type = sys.argv[1]
    data_path  = sys.argv[2]
    save_path  = sys.argv[3]
    NOISE      = eval(sys.argv[4])
    PSL        = eval(sys.argv[5]) if len(sys.argv) == 6 else False
    
    print("NOISE MODE:", NOISE)
    print("PSL MODE  :", PSL)
    
    train_dataset  = _load_split(data_path, "train")
    val_dataset    = _load_split(data_path, "val")

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=2048)

    epochs           = 500      # 10 
    warmup_epochs    = 10       # 5   
    flow_eval_epochs = 30       # 3   
    patience         = 30       # 10  

    if model_type.lower() == "CNNFlows".lower():
        cnnflows     = rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50).double()
        optimizer    = torch.optim.Adam(cnnflows.parameters(), lr=0.001)

        data_np = np.stack(list(val_dataset.flow_outputs.values()))
        data_np = data_np[:, np.all(np.isfinite(data_np), axis=0)]  # keep only columns without NaNs/Infs
        data_tensor = torch.tensor(data_np.T).to("cpu")
        cnnflows.pdf.init_params(data=data_tensor)

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
        optimizer = torch.optim.Adam(racnnflows.parameters(), lr=0.001)

        data_np = np.stack(list(val_dataset.flow_outputs.values()))
        data_np = data_np[:, np.all(np.isfinite(data_np), axis=0)]  # keep only columns without NaNs/Infs
        data_tensor = torch.tensor(data_np.T).to("cpu")
        racnnflows.pdf.init_params(data=data_tensor)

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
        optimizer_class = torch.optim.Adam(cnn_class.parameters(), lr=0.001)
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
        optimizer_class = torch.optim.Adam(racnn_class.parameters(), lr=0.001)
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
        optimizer_cf = torch.optim.Adam(cnnclassiflows.parameters(), lr=0.001)

        data_np = np.stack(list(val_dataset.flow_outputs.values()))
        data_np = data_np[:, np.all(np.isfinite(data_np), axis=0)]  # keep only columns without NaNs/Infs
        data_tensor = torch.tensor(data_np.T).to("cpu")
        cnnclassiflows.pdf.init_params(data=data_tensor)


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
        optimizer_cf    = torch.optim.Adam(racnnclassiflows.parameters(), lr=0.001)

        data_np = np.stack(list(val_dataset.flow_outputs.values()))
        data_np = data_np[:, np.all(np.isfinite(data_np), axis=0)]  # keep only columns without NaNs/Infs
        data_tensor = torch.tensor(data_np.T).to("cpu")
        racnnclassiflows.pdf.init_params(data=data_tensor)
        
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

    elif model_type.lower() == "NECNNFlows".lower():
        necnnflows     = rcn_m.NECNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50).double()
        optimizer = torch.optim.Adam(necnnflows.parameters(), lr=0.001)

        data_np = np.stack(list(val_dataset.flow_outputs.values()))
        data_np = data_np[:, np.all(np.isfinite(data_np), axis=0)]  # keep only columns without NaNs/Infs
        data_tensor = torch.tensor(data_np.T).to("cpu")
        necnnflows.pdf.init_params(data=data_tensor)

        necnnflows, out_dict = rcn_t.train_loop_nf(
        necnnflows,
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

    elif model_type.lower() == "NECNNClass".lower():
        necnn_class     = rcn_m.NECNNClass(cond_dim=25,hidden_size=25,hidden_size2=50).double()
        optimizer_class = torch.optim.Adam(necnn_class.parameters(), lr=0.001)
        necnn_class, out_dict1 = rcn_t.train_loop_class(
        necnn_class,
        optimizer_class,
        train_loader,
        epochs=epochs,
        val_loader=val_loader,
        clipnorm=1,
        patience=30,
        save_dir_path=save_path,
        cnn_mode=False,
        psl=PSL)


    elif model_type.lower() == "NECNNClassiFlows".lower():
        necnnclassiflows  = rcn_m.NECNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50).double()
        optimizer_cf      = torch.optim.Adam(necnnclassiflows.parameters(), lr=0.001)

        data_np = np.stack(list(val_dataset.flow_outputs.values()))
        data_np = data_np[:, np.all(np.isfinite(data_np), axis=0)]  # keep only columns without NaNs/Infs
        data_tensor = torch.tensor(data_np.T).to("cpu")
        necnnclassiflows.pdf.init_params(data=data_tensor)
        
        necnnclassiflows, out_dict = rcn_t.train_loop_merged_model(
        necnnclassiflows,
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