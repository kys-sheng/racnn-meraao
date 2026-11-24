import flexibledataset as fd


import racnn_model_mx as rcn_m
import racnn_trainer_mx as rcn_t
import racnn_eval_mx as rcn_e

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

device = rcn_e.device


def _load_split(data_path, split):
    """Load <data_path>/<split>.h5 (release format), else the legacy <split>_dataset.pt."""
    h5 = os.path.join(data_path, f"{split}.h5")
    return fd.load_dataset(h5 if os.path.exists(h5) else os.path.join(data_path, f"{split}_dataset.pt"))


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 8:
        print("Usage: python train_model.py <model_type> <data_path> <save_path> <noise> [<model_size=1 or 2>] [PSL] [JAMMY_LAYERS]")
        sys.exit(1)

    model_type = sys.argv[1]
    data_path  = sys.argv[2]
    save_path  = sys.argv[3]
    NOISE      = eval(sys.argv[4])
    
    if sys.argv[5]: model_size = eval(sys.argv[5])
    else          : model_size = 1 
    assert ((model_size == 1) or (model_size == 2)), "Model size can only be 1 or 2 so far."

    if sys.argv[6]: PSL = eval(sys.argv[6])
    else          : PSL = False
    assert ((PSL == True) or (PSL == False)), "PSL can only be True or False."
    
    if sys.argv[7]: JAMMY_LAYERS = sys.argv[7]
    else          : JAMMY_LAYERS = "gggt"

    print("model_type    :", model_type)
    print("data_path     :", data_path)
    print("save_path     :", save_path)
    print("NOISE MODE    :", NOISE)
    print("model_size    :", model_size)
    print("PSL           :", PSL)
    print("JAMMY_LAYERS  :", JAMMY_LAYERS)
    if "r" in JAMMY_LAYERS: manifold = "i1_-0.01_1.02+i1_-0.01_1.02"
    else                  : manifold = "e2"
    train_dataset  = _load_split(data_path, "train")
    val_dataset    = _load_split(data_path, "val")

    batch_size     = 512
    train_loader   = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader     = DataLoader(val_dataset, batch_size=2048, shuffle=True)

    epochs           = 50   # 500      # 10 
    warmup_epochs    = 3  # 10       # 5   
    flow_eval_epochs = 1  # 30       # 3   
    patience         = 10  # 30  #10       # 30  
    aux_input_size   = 11  # 11       # or 11 
    clipnorm         = 1000  # 1000

    print("batch_size       : ",batch_size)
    print("epochs           : ",epochs)
    print("warmup_epochs    : ",warmup_epochs)
    print("flow_eval_epochs : ",flow_eval_epochs)
    print("patience         : ",patience)
    print("aux_input_size   : ",aux_input_size)
    print("clipnorm         : ",clipnorm)

    if model_type.lower() == "CNNFlows".lower():
        if   model_size == 1 : cnnflows     = rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers=JAMMY_LAYERS,manifold=manifold).double()
        elif model_size == 2 : cnnflows     = rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers=JAMMY_LAYERS,manifold=manifold).double()
        else                 : raise KeyError(f"Model Size weird! {model_size}")
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
        clipnorm=clipnorm,
        patience=patience,
        save_dir_path=save_path,
        flow_eval_epochs = flow_eval_epochs,
        plot=False,
        cnn_mode=True,
        noisy=NOISE,
        psl=PSL)

    elif model_type.lower() == "RACNNFlows".lower():
        if   model_size == 1 : racnnflows     = rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers=JAMMY_LAYERS,manifold=manifold).double()
        elif model_size == 2 : racnnflows     = rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers=JAMMY_LAYERS,manifold=manifold).double()
        else                 : raise KeyError(f"Model Size weird! {model_size}")
        
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
        clipnorm=clipnorm,
        patience=patience,
        save_dir_path=save_path,
        flow_eval_epochs = flow_eval_epochs,
        plot=False,
        cnn_mode=False,
        noisy=NOISE,
        psl=PSL)

    elif model_type.lower() == "CNNClass".lower():
        if   model_size == 1 : cnn_class       = rcn_m.CNNClass(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double()
        elif model_size == 2 : cnn_class       = rcn_m.CNNClass(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double()
        else                 : raise KeyError(f"Model Size weird! {model_size}")

        optimizer_class = torch.optim.Adam(cnn_class.parameters(), lr=0.001)
        cnn_class, out_dict_cnn = rcn_t.train_loop_class(
        cnn_class,
        optimizer_class,
        train_loader,
        epochs=epochs,
        val_loader=val_loader,
        clipnorm=clipnorm,
        patience=patience,
        save_dir_path=save_path,
        cnn_mode=True,
        psl=PSL)

    elif model_type.lower() == "RACNNClass".lower():
        if   model_size == 1 : racnn_class     = rcn_m.RACNNClass(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double()
        elif model_size == 2 : racnn_class     = rcn_m.RACNNClass(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double()
        else                 : raise KeyError(f"Model Size weird! {model_size}")
        
        optimizer_class = torch.optim.Adam(racnn_class.parameters(), lr=0.001)
        racnn_class, out_dict1 = rcn_t.train_loop_class(
        racnn_class,
        optimizer_class,
        train_loader,
        epochs=epochs,
        val_loader=val_loader,
        clipnorm=clipnorm,
        patience=30,
        save_dir_path=save_path,
        cnn_mode=False,
        psl=PSL)

    elif model_type.lower() == "CNNClassiFlows".lower():
        if   model_size == 1 : cnnclassiflows     = rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers=JAMMY_LAYERS,manifold=manifold).double()
        elif model_size == 2 : cnnclassiflows     = rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers=JAMMY_LAYERS,manifold=manifold).double()
        else                 : raise KeyError(f"Model Size weird! {model_size}")
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
        clipnorm=clipnorm,
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
        if   model_size == 1 : racnnclassiflows        = rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers=JAMMY_LAYERS,manifold=manifold).double()
        elif model_size == 2 : racnnclassiflows        = rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers=JAMMY_LAYERS,manifold=manifold).double()
        else                 : raise KeyError(f"Model Size weird! {model_size}")
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
        clipnorm=clipnorm,
        patience=patience,
        save_dir_path=save_path,
        warmup_epochs = warmup_epochs ,
        flow_eval_epochs = flow_eval_epochs,
        plot=False,
        cnn_mode=False,
        noisy=NOISE,
        psl=PSL)

    elif model_type.lower() == "NECNNFlows".lower():
            rcn_t.NECNN_MODE = True
            if   model_size == 1 : necnnflows     = rcn_m.RACNNFlows(aux_input_size=3,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers=JAMMY_LAYERS,manifold=manifold).double()
            elif model_size == 2 : necnnflows     = rcn_m.RACNNFlows(aux_input_size=3,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers=JAMMY_LAYERS,manifold=manifold).double()
            else                 : raise KeyError(f"Model Size weird! {model_size}")
            
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
            clipnorm=clipnorm,
            patience=patience,
            save_dir_path=save_path,
            flow_eval_epochs = flow_eval_epochs,
            plot=False,
            cnn_mode=False,
            noisy=NOISE,
            psl=PSL)

    elif model_type.lower() == "NECNNClass".lower():
            rcn_t.NECNN_MODE = True 
            if   model_size == 1 : necnn_class     = rcn_m.RACNNClass(aux_input_size=3,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double()
            elif model_size == 2 : necnn_class     = rcn_m.RACNNClass(aux_input_size=3,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double()
            else                 : raise KeyError(f"Model Size weird! {model_size}")
            
            optimizer_class = torch.optim.Adam(necnn_class.parameters(), lr=0.001)
            necnn_class, out_dict1 = rcn_t.train_loop_class(
            necnn_class,
            optimizer_class,
            train_loader,
            epochs=epochs,
            val_loader=val_loader,
            clipnorm=clipnorm,
            patience=30,
            save_dir_path=save_path,
            cnn_mode=False,
            psl=PSL)

    elif model_type.lower() == "NECNNClassiFlows".lower():
            rcn_t.NECNN_MODE = True
            if   model_size == 1 : necnnclassiflows        = rcn_m.RACNNClassiFlows(aux_input_size=3,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers=JAMMY_LAYERS,manifold=manifold).double()
            elif model_size == 2 : necnnclassiflows        = rcn_m.RACNNClassiFlows(aux_input_size=3,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers=JAMMY_LAYERS,manifold=manifold).double()
            else                 : raise KeyError(f"Model Size weird! {model_size}")
            optimizer_cf    = torch.optim.Adam(necnnclassiflows.parameters(), lr=0.001)

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
            clipnorm=clipnorm,
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
