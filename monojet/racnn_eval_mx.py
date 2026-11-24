import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

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
import traceback

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_auc_score, roc_curve

import racnn_model_mx as rcn_m
import racnn_trainer_mx as rcn_t

import re
from collections import defaultdict

import flexibledataset as fd

 


# Check for available GPUs
if torch.cuda.is_available():
    # Choose the first available GPU
    device = torch.device("cuda")
    # Optionally, print information about each GPU
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
else:
    # Use the CPU if no GPU is available
    device = torch.device("cpu")
    print("No GPU available. Using CPU.")

# Utils for results class
def group_by_number_diff(strings):
    pattern = re.compile(r'(.*?)(\d+)(.*)')
    groups = defaultdict(list)

    for s in strings:
        match = pattern.match(s)
        if match:
            prefix, num, suffix = match.groups()
            key = f"{prefix}#{suffix}"  # Use a marker to represent the digit
            groups[key].append(s)
        else:
            # If no number is found, group by itself
            groups[s].append(s)

    return dict(groups)

#def z_sig(s,b):
#    return np.sqrt(2*((s+b)*np.log(1+(s/b)) - s))
def z_sig(s, b):
    return torch.where(
        b <= 0,
        torch.tensor(float('inf')),
        torch.sqrt(2*((s+b)*torch.log1p((s/b)) - s))
    )


def plot_evaluation_metrics(e_truth_f1, c_truth_f1,bins=50, range=(-7, 1)):
    """
    Plots:
    1. Histogram of e_truth_f1 split by class
    2. Logit-transformed e_truth_f1
    3. ROC curve with AUC based on the transformed values

    Parameters:
    - e_truth_f1: list or array of predicted scores
    - c_truth_f1: list or array of true binary labels (0 or 1)
    """

    auc, fpr, tpr, thresholds, e_truth_f1_logits = get_evaluation_metrics(e_truth_f1, c_truth_f1)
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # --- Subplot 1: Histogram ---
    axs[0].hist(e_truth_f1, label="truth", alpha=0.5, bins=bins, range=range, histtype='step')
    if auc != 0 : axs[0].hist([e for c, e in zip(c_truth_f1, e_truth_f1) if c == 1], label="1-truth", alpha=0.5, bins=bins, range=range, histtype='step')
    if auc != 0 : axs[0].hist([e for c, e in zip(c_truth_f1, e_truth_f1) if c == 0], label="0-truth", alpha=0.5, bins=bins, range=range, histtype='step')
    axs[0].set_yscale("log")
    axs[0].set_title('e_truth Histogram')
    axs[0].legend()

    # --- Subplot 2: Logit Transformation ---
    if auc != 0 : 
        axs[1].plot(e_truth_f1, e_truth_f1_logits, '.', alpha=0.5)
        axs[1].set_title('Logit Transformation')
        axs[1].set_xlabel('e_truth')
        axs[1].set_ylabel('logits')

        # --- Subplot 3: ROC Curve ---
        axs[2].plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        axs[2].plot([0, 1], [0, 1], linestyle='--', color='gray')
        axs[2].set_title('ROC Curve')
        axs[2].set_xlabel('False Positive Rate')
        axs[2].set_ylabel('True Positive Rate')
        axs[2].legend(loc='lower right')
        axs[2].grid(True)

    plt.tight_layout()
    plt.show()

# def normalize_name(name):
#     # Remove "_noiseless_"
#     name = name.replace("_noiseless_", "_")
#     # Replace number groups like _1_, _2_, ..., _5_ with a placeholder
#     name = re.sub(r'_(\d+)_', '_#_', name)
#     return name

# def group_model_names(names):
#     grouped = defaultdict(list)
#     for name in names:
#         key = normalize_name(name)
#         grouped[key].append(name)
#     return grouped

def normalize_name(name,with_v=True):
    # Remove "_noiseless_"
    name = name.replace("_noiseless_", "_")
    # Replace number groups like _1_, _2_, ..., _5_ with a placeholder
    if with_v: name = re.sub(r'_v(\d+)_', '_#_', name)
    else     : name = re.sub(r'_(\d+)_', '_#_', name)
    return name
    

def group_model_names(names):
    grouped = defaultdict(list)
    for name in names:
        if "_v" in name: with_v=True
        else           : with_v=False
        key = normalize_name(name,with_v)
        grouped[key].append(name)
    return grouped

def check_matches(mchoice_list,inname):
    inname = inname.replace("_gggt","")
    inname_characteristic = inname.split("_")
    ninter_most = 0 
    mbest = None
    same_dat = []
    for mchoice in mchoice_list:
        if inname_characteristic[0] != mchoice.split("_")[0] : continue
        mchoice_characteristic = mchoice.split("_")
        ninter = len(set(inname_characteristic).intersection(mchoice_characteristic))

        if ninter > ninter_most:
            mbest = mchoice
            ninter_most = ninter
            same_dat = []
        elif ninter == ninter_most:
            same_dat.append(mchoice)
    if len(same_dat) > 0:
            print("inname         : ",inname)
            print("mchoice_list   : ",mchoice_list)
            print("mbest          : ",mbest)
            print("same_dat       : ",same_dat)
            print()
            raise NameError("Undifferentiable!")
    return mbest  

Z_SIG = True
print("Z_SIG set to: ",Z_SIG)
# model_dict = {
#     "cnnclass"         :   rcn_m.CNNClass(cond_dim=25,hidden_size=25,hidden_size2=50).double()   , 
#     "cnnflows"         :   rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50).double()   , 
#     "cnnclassiflows"   :   rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50).double()   , 
#     "racnnclass"       :   rcn_m.RACNNClass(aux_input_size=3, cond_dim=25,hidden_size=25,hidden_size2=50).double()   , 
#     "racnnflows"       :   rcn_m.RACNNFlows(aux_input_size=3, cond_dim=25,hidden_size=25,hidden_size2=50).double()   , 
#     "racnnclassiflows" :   rcn_m.RACNNClassiFlows(aux_input_size=3, cond_dim=25,hidden_size=25,hidden_size2=50).double()   , 
# }
aux_input_size = 11
model_dict = {
    "cnnclass_m1"                    : rcn_m.CNNClass(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "cnnclass_m2"                    : rcn_m.CNNClass(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "cnnflows_m1"                    : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "cnnflows_noiseful_m1"           : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "cnnflows_noiseless_m1"          : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "cnnflows_m2"                    : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "cnnflows_noiseful_m2"           : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "cnnflows_noiseless_m2"          : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "cnnclassiflows_m1"              : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "cnnclassiflows_noiseful_m1"     : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "cnnclassiflows_noiseless_m1"    : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "cnnclassiflows_m2"              : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "cnnclassiflows_noiseful_m2"     : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "cnnclassiflows_noiseless_m2"    : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "racnnclass_m1"                  : rcn_m.RACNNClass(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "racnnclass_m2"                  : rcn_m.RACNNClass(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "racnnflows_m1"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "racnnflows_noiseful_m1"         : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "racnnflows_noiseless_m1"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "racnnflows_m2"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "racnnflows_noiseful_m2"         : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "racnnflows_noiseless_m2"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "racnnclassiflows_m1"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "racnnclassiflows_noiseful_m1"   : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "racnnclassiflows_noiseless_m1"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "racnnclassiflows_m2"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "racnnclassiflows_noiseful_m2"   : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "racnnclassiflows_noiseless_m2"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),

    "necnnclass_m1"                  : rcn_m.RACNNClass(aux_input_size=3,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "necnnclass_m2"                  : rcn_m.RACNNClass(aux_input_size=3,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "necnnflows_m1"                  : rcn_m.RACNNFlows(aux_input_size=3,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "necnnflows_noiseful_m1"         : rcn_m.RACNNFlows(aux_input_size=3,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "necnnflows_noiseless_m1"        : rcn_m.RACNNFlows(aux_input_size=3,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "necnnflows_m2"                  : rcn_m.RACNNFlows(aux_input_size=3,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "necnnflows_noiseful_m2"         : rcn_m.RACNNFlows(aux_input_size=3,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "necnnflows_noiseless_m2"        : rcn_m.RACNNFlows(aux_input_size=3,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "necnnclassiflows_m1"            : rcn_m.RACNNClassiFlows(aux_input_size=3,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "necnnclassiflows_noiseful_m1"   : rcn_m.RACNNClassiFlows(aux_input_size=3,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "necnnclassiflows_noiseless_m1"  : rcn_m.RACNNClassiFlows(aux_input_size=3,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30).double(),
    "necnnclassiflows_m2"            : rcn_m.RACNNClassiFlows(aux_input_size=3,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "necnnclassiflows_noiseful_m2"   : rcn_m.RACNNClassiFlows(aux_input_size=3,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),
    "necnnclassiflows_noiseless_m2"  : rcn_m.RACNNClassiFlows(aux_input_size=3,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30).double(),

    "cnnflows_m1_t"                    : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "cnnflows_noiseful_m1_t"          : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "cnnflows_noiseless_m1_t"          : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "cnnflows_m2_t"                    : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "cnnflows_noiseful_m2_t"          : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "cnnflows_noiseless_m2_t"          : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "cnnclassiflows_m1_t"              : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "cnnclassiflows_noiseful_m1_t"    : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "cnnclassiflows_noiseless_m1_t"    : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "cnnclassiflows_m2_t"              : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "cnnclassiflows_noiseful_m2_t"    : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "cnnclassiflows_noiseless_m2_t"    : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="t").double(),
    #
    #
    "racnnflows_m1_t"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "racnnflows_noiseful_m1_t"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "racnnflows_noiseless_m1_t"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "racnnflows_m2_t"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "racnnflows_noiseful_m2_t"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "racnnflows_noiseless_m2_t"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "racnnclassiflows_m1_t"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "racnnclassiflows_noiseful_m1_t"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "racnnclassiflows_noiseless_m1_t"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "racnnclassiflows_m2_t"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "racnnclassiflows_noiseful_m2_t"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="t").double(),
    "racnnclassiflows_noiseless_m2_t"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="t").double(),

    "cnnflows_m1_gt"                    : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "cnnflows_noiseful_m1_gt"          : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "cnnflows_noiseless_m1_gt"          : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "cnnflows_m2_gt"                    : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "cnnflows_noiseful_m2_gt"          : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "cnnflows_noiseless_m2_gt"          : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "cnnclassiflows_m1_gt"              : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "cnnclassiflows_noiseful_m1_gt"    : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "cnnclassiflows_noiseless_m1_gt"    : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "cnnclassiflows_m2_gt"              : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "cnnclassiflows_noiseful_m2_gt"    : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "cnnclassiflows_noiseless_m2_gt"    : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    #
    #
    "racnnflows_m1_gt"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "racnnflows_noiseful_m1_gt"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "racnnflows_noiseless_m1_gt"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "racnnflows_m2_gt"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "racnnflows_noiseful_m2_gt"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "racnnflows_noiseless_m2_gt"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "racnnclassiflows_m1_gt"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "racnnclassiflows_noiseful_m1_gt"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "racnnclassiflows_noiseless_m1_gt"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "racnnclassiflows_m2_gt"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "racnnclassiflows_noiseful_m2_gt"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gt").double(),
    "racnnclassiflows_noiseless_m2_gt"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gt").double(),

    "cnnflows_m1_ggt"                    : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "cnnflows_noiseful_m1_ggt"          : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "cnnflows_noiseless_m1_ggt"          : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "cnnflows_m2_ggt"                    : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "cnnflows_noiseful_m2_ggt"          : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "cnnflows_noiseless_m2_ggt"          : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "cnnclassiflows_m1_ggt"              : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "cnnclassiflows_noiseful_m1_ggt"    : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "cnnclassiflows_noiseless_m1_ggt"    : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "cnnclassiflows_m2_ggt"              : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "cnnclassiflows_noiseful_m2_ggt"    : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "cnnclassiflows_noiseless_m2_ggt"    : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "racnnflows_m1_ggt"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "racnnflows_noiseful_m1_ggt"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "racnnflows_noiseless_m1_ggt"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "racnnflows_m2_ggt"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "racnnflows_noiseful_m2_ggt"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "racnnflows_noiseless_m2_ggt"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "racnnclassiflows_m1_ggt"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "racnnclassiflows_noiseful_m1_ggt"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "racnnclassiflows_noiseless_m1_ggt"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "racnnclassiflows_m2_ggt"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "racnnclassiflows_noiseful_m2_ggt"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="ggt").double(),
    "racnnclassiflows_noiseless_m2_ggt"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="ggt").double(),

    #"cnnflows_m1_gtgt"                    : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"cnnflows_noiseful_m1_gtgt"          : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"cnnflows_noiseless_m1_gtgt"          : rcn_m.CNNFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"cnnflows_m2_gtgt"                    : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"cnnflows_noiseful_m2_gtgt"          : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"cnnflows_noiseless_m2_gtgt"          : rcn_m.CNNFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"cnnclassiflows_m1_gtgt"              : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"cnnclassiflows_noiseful_m1_gtgt"    : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"cnnclassiflows_noiseless_m1_gtgt"    : rcn_m.CNNClassiFlows(cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"cnnclassiflows_m2_gtgt"              : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"cnnclassiflows_noiseful_m2_gtgt"    : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"cnnclassiflows_noiseless_m2_gtgt"    : rcn_m.CNNClassiFlows(cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"racnnflows_m1_gtgt"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"racnnflows_noiseful_m1_gtgt"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"racnnflows_noiseless_m1_gtgt"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"racnnflows_m2_gtgt"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"racnnflows_noiseful_m2_gtgt"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    #"racnnflows_noiseless_m2_gtgt"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    "racnnclassiflows_m1_gtgt"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    "racnnclassiflows_noiseful_m1_gtgt"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    "racnnclassiflows_noiseless_m1_gtgt"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    "racnnclassiflows_m2_gtgt"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    "racnnclassiflows_noiseful_m2_gtgt"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),
    "racnnclassiflows_noiseless_m2_gtgt"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="gtgt").double(),

    # "racnnflows_m1_r"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="r+r",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_m1_rr"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="rr+rr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_m1_rrr"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="rrr+rrr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_noiseful_m1_r"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="r+r",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_noiseful_m1_rr"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="rr+rr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_noiseful_m1_rrr"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="rrr+rrr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_noiseless_m1_r"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="r+r",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_noiseless_m1_rr"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="rr+rr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_noiseless_m1_rrr"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="rrr+rrr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_m2_r"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="r+r",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_m2_rr"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="rr+rr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_m2_rrr"                  : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="rrr+rrr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_noiseful_m2_r"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="r+r",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_noiseful_m2_rr"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="rr+rr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_noiseful_m2_rrr"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="rrr+rrr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_noiseless_m2_r"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="r+r",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_noiseless_m2_rr"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="rr+rr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnflows_noiseless_m2_rrr"        : rcn_m.RACNNFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="rrr+rrr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_m1_r"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="r+r",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_m1_rr"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="rr+rr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_m1_rrr"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="rrr+rrr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_noiseful_m1_r"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="r+r",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_noiseful_m1_rr"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="rr+rr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_noiseful_m1_rrr"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="rrr+rrr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_noiseless_m1_r"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="r+r",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_noiseless_m1_rr"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="rr+rr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_noiseless_m1_rrr"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=25,hidden_size=25,hidden_size2=50,in_channels=3,seq_len=30,jammy_layers="rrr+rrr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_m2_r"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="r+r",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_m2_rr"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="rr+rr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_m2_rrr"            : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="rrr+rrr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_noiseful_m2_r"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="r+r",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_noiseful_m2_rr"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="rr+rr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_noiseful_m2_rrr"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="rrr+rrr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_noiseless_m2_r"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="r+r",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_noiseless_m2_rr"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="rr+rr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
    # "racnnclassiflows_noiseless_m2_rrr"  : rcn_m.RACNNClassiFlows(aux_input_size=aux_input_size,cond_dim=50,hidden_size=100,hidden_size2=200,in_channels=3,seq_len=30,jammy_layers="rrr+rrr",manifold="i1_-0.01_1.02+i1_-0.01_1.02").double(),
}

CNN_MODE_DICT = {   "cnnclass" : True,
                    "cnnflows" : True,
                    "cnnclassiflows" : True,
                    "racnnclass" : False,
                    "racnnflows" : False,
                    "racnnclassiflows"   : False,
                    "necnnclass" : False,
                    "necnnflows" : False,
                    "necnnclassiflows"   : False,
                    }
MODEL_TYPE_DICT = {  "cnnclass"         : "c",
                      "cnnflows"         : "f",
                      "cnnclassiflows"   : "cf",
                      "racnnclass"       : "c",
                      "racnnflows"       : "f",
                      "racnnclassiflows" : "cf",
                      "necnnclass"       : "c",
                      "necnnflows"       : "f",
                      "necnnclassiflows" : "cf"}

class Results():
    """
    Results management class for handling multiple model evaluations.
    
    This class manages the evaluation of multiple RACNN models by:
    1. Loading model data and datasets from specified paths
    2. Organizing models by type and parameters
    3. Running evaluations with error handling and retries
    4. Storing and managing results for analysis
    
    Attributes:
        result_path                : Path to directory containing model results
        model_dict                 : Dictionary mapping model names to model objects
        results                    : Dictionary storing individual model results
        ensemble_results           : Dictionary storing ensemble evaluation results
        results_by_param           : Dictionary storing individual model results    organized by parameter values
        ensemble_results_by_param  : Dictionary storing ensemble evaluation results organized by parameter values
    """
    def __init__(self,result_path, model_dict,results={},ensemble_results={},results_by_param={},ensemble_results_by_param={},test_batch_size=2048,ignore_model=False):
        self.result_path = result_path
        self.model_dict  = model_dict
        
        self.get_data()
        self.get_dataset()
        self.get_outdicts()
        if not ignore_model: self.get_models()

        self.train_loader  = DataLoader(self.datasets['train_dataset.pt'], batch_size=2048, shuffle=True)
        self.val_loader    = DataLoader(self.datasets['val_dataset.pt']  , batch_size=2048, shuffle=True)
        self.test_loader   = DataLoader(self.datasets['test_dataset.pt'] , batch_size=test_batch_size, shuffle=True)
        
        self.results                     = results
        self.ensemble_results            = ensemble_results 

        self.results_by_param            = results_by_param
        self.ensemble_results_by_param   = ensemble_results_by_param 
        
        self.grouped_names               = None 
        
    def get_data(self):
        """Load directory and file information from the result path."""
        
        # find all directories in the result path
        self.dirs = []
        for dir in os.listdir(self.result_path):
            if os.path.isdir(os.path.join(self.result_path, dir)):
                self.dirs.append(dir)

        # find all files in the result path
        self.files = []
        for file in os.listdir(self.result_path):
            if os.path.isfile(os.path.join(self.result_path, file)):
                self.files.append(file)

    def get_dataset(self):
        """
        Load and organize dataset files from the result path.
        
        This method searches for files containing 'dataset' in their filename
        and loads them as PyTorch tensors. The datasets are typically used
        for training, validation, and testing of the models.
        
        Side Effects:
            Updates self.datasets with a dictionary mapping filenames to loaded datasets
        """

        # find all datasets in the result path
        self.datasets = {}
        for file in self.files:
            # if dataset is in the directory name, add it to the list
            if 'dataset' in file: 
                try:
                    self.datasets[file] = torch.load(os.path.join(self.result_path,file),map_location=device,weights_only=False)
                except (EOFError) as e:
                    print("Looking at :",file)
                    print(str(e))
                    continue


    
    def get_outdicts(self):
        """
        Load output dictionaries from subdirectories in the result path.
        
        This method searches through all subdirectories for files containing 'dict'
        in their filename. These typically contain training outputs, hyperparameters,
        or other metadata from model training runs.
        
        Side Effects:
            Updates self.outdicts with a dictionary mapping directory names to loaded dictionaries
        """
        # find all outdicts inside the directoris in result path
        self.outdicts = {}
        for dir in self.dirs:
            for file in os.listdir(os.path.join(self.result_path, dir)):
                # if outdict is in the directory name, add it to the list
                if 'dict' in file: 
                    self.outdicts[dir] = torch.load(os.path.join(os.path.join(self.result_path, dir),file),map_location=device,weights_only=False)

    def get_models(self):
        """
        Load and organize trained models from subdirectories.
        
        This method searches through all subdirectories for files containing 'best_model'
        and '.pt' in their filename. It then:
        1. Determines the model type based on directory name
        2. Loads the model state dictionary
        3. Maps models to their CNN mode and model type
        
        The method handles cases where multiple model types match by prioritizing:
        - Models with 'ra' (RACNN) in the name
        - Models with 'flows' in the name
        - Models with both 'ra' and 'flows'
        
        Side Effects:
            Updates self.models, self.cnn_mode, and self.model_type dictionaries
            
        Raises:
            KeyError: If model type cannot be determined from directory name
        """
        # find all models in the result path
        self.models      = {}
        self.cnn_mode    = {}
        self.model_type  = {}
        
        for dir in self.dirs:
            run_path = os.path.join(self.result_path, dir)
            for file in os.listdir(run_path):
                # If model checkpoint file is found
                if 'best_model' in file and '.pt' in file: 
                    # Determine model type from directory path
                    mchoice = [ k for k in self.model_dict.keys() if k in run_path.split("/")[-1]]
                    mexact = check_matches(mchoice,run_path.split("/")[-1])
                    # print("file: ",file)
                    # print("run_path: ",run_path)
                    # print("mchoice: ",mchoice)
                    # print("mexact: ",mexact)

                    # Create model instance and load weights
                    self.models[dir+"_"+file.split(".pt")[0]] = copy.deepcopy(self.model_dict[mexact]).to(device).double()
                    self.models[dir+"_"+file.split(".pt")[0]].load_state_dict(torch.load(os.path.join(run_path, file),map_location=device, weights_only=False))

                    # Store model configuration
                    self.cnn_mode[dir+"_"+file.split(".pt")[0]] = CNN_MODE_DICT[mexact.split("_")[0]]
                    self.model_type[dir+"_"+file.split(".pt")[0]] = MODEL_TYPE_DICT[mexact.split("_")[0]]

                    # # Handle ambiguous model type matches
                    # if len(mchoice) > 1 : 
                    #     mchoice_byra         = [i for i in mchoice if "ra" in i]
                    #     mchoice_byflows      = [i for i in mchoice if "flows" in i]
                    #     mchoice_byraflows    = [i for i in mchoice if ("flows") in i and ("ra" in i)]
                        
                    #     if len(mchoice_byra) == 1: mchoice =  mchoice_byra[0]
                    #     elif len(mchoice_byflows) == 1: mchoice =  mchoice_byflows[0]
                    #     elif len(mchoice_byraflows) == 1: mchoice =  mchoice_byraflows[0]
                    #     else            : raise KeyError(f"Could not determine unique model type from choices: {mchoice}")

                    #     if "m1" in mchoice        : mchoice_msize = "m1"
                    #     elif  "m2" in mchoice     : mchoice_msize = "m2"
                    #     else : raise KeyError(f"Could not determine unique model type from msize: {mchoice}")


                    # elif len(mchoice) == 1 : mchoice= mchoice[0]
                    # else                : raise KeyError(f"No model type found for directory: {run_path}")

                    # # Create model instance and load weights
                    # self.models[dir+"_"+file.split(".pt")[0]] = copy.deepcopy(self.model_dict[mchoice]).to(device).double()
                    # self.models[dir+"_"+file.split(".pt")[0]].load_state_dict(torch.load(os.path.join(run_path, file),map_location=device))
                    
                    # # Store model configuration
                    # self.cnn_mode[dir+"_"+file.split(".pt")[0]] = CNN_MODE_DICT[mchoice.split("_")[0]]
                    # self.model_type[dir+"_"+file.split(".pt")[0]] = MODEL_TYPE_DICT[mchoice.split("_")[0]]

    def get_grouped_names(self):
        """
        Group model names by common patterns for ensemble evaluation.
        
        This method uses the group_by_number_diff utility function to group
        model names that differ only in numeric identifiers. This is useful
        for identifying models that should be evaluated together as ensembles.
        
        For example, models named "racnn_1_best_model", "racnn_2_best_model"
        would be grouped together under a common pattern.
        
        Side Effects:
            Updates self.grouped_names with a dictionary mapping common patterns
            to lists of model names that match that pattern
        """

        names = sorted(self.models.keys())              # Get all model names
        self.grouped_names = group_model_names(names)   # Group by pattern

# Eval Func for results class
cond_dim=50

def get_evaluation_metrics(e_truth_f1, c_truth_f1):
    if all(c_truth_f1) or not any(c_truth_f1):
        return 0, [], [], [], []
    # Normalize (logit-style) the scores to [0, 1]
    e_truth_f1_max = max(e_truth_f1)
    e_truth_f1_min = min(e_truth_f1)
    e_truth_f1_logits = (e_truth_f1_max - e_truth_f1) / (e_truth_f1_max - e_truth_f1_min)

    # Compute ROC metrics
    auc = roc_auc_score(c_truth_f1, e_truth_f1_logits)
    fpr, tpr, thresholds = roc_curve(c_truth_f1, e_truth_f1_logits)
    return auc, fpr, tpr, thresholds, e_truth_f1_logits

def evaluate_ensemble_entropy_faster(modelf_list,x_cnn, x_aux,n_sample,device,cond_dim=cond_dim,verbose=False,model_type="f",cnn_mode=False):
    n_model       = len(modelf_list)
    model_indices = torch.randint(0, n_model, (n_sample,), device=device)
    counts        = torch.bincount(model_indices, minlength=n_model)
    samples       = torch.tensor([]).to(device)
    if verbose:
        print("n_model:",n_model)
        print("counts:",counts)
        print("x_cnn:",len(x_cnn),x_cnn.shape)
        print("x_aux:",len(x_aux),x_aux.shape)
        print("n_sample:",n_sample)

    cond_dat          =  torch.zeros(n_model,len(x_cnn),cond_dim)
    ts_dat            =  {}
    lp_dat            =  {}
    target_sample_dat =  [] 
    lpdf_dat          =  [] 

    for i in range(n_model):
        modelf_list[i].eval()
        if cnn_mode:
            if model_type == "f":
                cond_dat[i]                = modelf_list[i](x_cnn).double().to(device)
            elif model_type == "cf":
                cond_dat[i], _             = modelf_list[i](x_cnn)
                cond_dat[i] = cond_dat[i].double().to(device)
            else:
                raise KeyError
        else:
            if model_type == "f":
                cond_dat[i]                = modelf_list[i](x_cnn, x_aux).double().to(device)
            elif model_type == "cf":
                cond_dat[i], _             = modelf_list[i](x_cnn, x_aux)
                cond_dat[i] = cond_dat[i].double().to(device)
            else:
                raise KeyError
            
    
        cond_dat_i = cond_dat[i].repeat(counts[i],1,1)
        N, B, D = cond_dat_i.shape
        cond_dat_i_flat = cond_dat_i.view(N * B, D)

        target_sample_flat,_,lpdf_flat,_   = modelf_list[i].pdf.sample(conditional_input=cond_dat_i_flat.double().to(device), samplesize=1)
        target_sample_i                    = target_sample_flat.view(N, B, *target_sample_flat.shape[1:])
        lpdf_i                             = lpdf_flat.view(N, B, *lpdf_flat.shape[1:])

        if verbose:
            print("cond_dat[i]   :",cond_dat[i].shape)
            print("cond_dat[i] r :",cond_dat[i].repeat(counts[i],1,1).shape)
            print("target_sample_flat.shape :",target_sample_flat.shape)
            print("lpdf_flat.shape          :",lpdf_flat.shape)
            print("target_sample_i.shape    :",target_sample_i.shape)
            print("lpdf_i.shape             :",lpdf_i.shape)

        ts_dat[i] = target_sample_i
        lp_dat[i] = lpdf_i
        
    all_lp_dat     = []
    for i_model in range(n_model):
        for i_samples in range(n_model):
           if i_model == i_samples: continue
           cond_dat_i = cond_dat[i_model].repeat(counts[i_samples],1,1)
           Nc, Bc, Dc = cond_dat_i.shape
           cond_dat_i_flat = cond_dat_i.view(Nc * Bc, Dc)
           
           Nt, Bt, Dt = ts_dat[i_samples].shape
           ts_dat_i_flat = ts_dat[i_samples].view(Nt * Bt, Dt)

           lpdf_nm_flat,_,_  = modelf_list[i_model].pdf.forward(ts_dat_i_flat.double().to(device),conditional_input=cond_dat_i_flat.double().to(device))
           lpdf_nm= lpdf_nm_flat.view(Nt, Bt)
           lp_dat[i_model] = torch.vstack([lp_dat[i_model],lpdf_nm])
           if verbose:
               print(f"cond_dat[{i_model}]   :",cond_dat[i_model].shape)
               print("cond_dat_i :",cond_dat[i_model].repeat(counts[i_samples],1,1).shape)
               print(f"ts_dat[{i_samples}]   :",ts_dat[i_samples].shape)
               print(f"ts_dat_i_flat[{i_samples}]   :",ts_dat_i_flat.shape)
               print(f"lpdf_nm_flat   Q{i_model}( X{i_samples} | C{i_model})   :",lpdf_nm_flat.shape)
               print(f"lpdf_nm   Q{i_model}( X{i_samples} | C{i_model})   :",lpdf_nm.shape)

        all_lp_dat.append(lp_dat[i_model])
    all_lp_dat =     torch.stack(all_lp_dat, dim=0).permute(2, 1, 0)
    logsumexp = torch.logsumexp(all_lp_dat, dim=2)
    norm = torch.full(logsumexp.shape, torch.tensor(np.log(n_model))).to(device)
    e_per_sample = logsumexp - norm
    entropy = -e_per_sample.mean(dim=1)
    if verbose:
        print(all_lp_dat.shape)
        print(logsumexp.shape)
        print(norm.shape)
        print(e_per_sample.shape)
        print(entropy.shape)
    return entropy

def evaluate_ensemble_loss(modelf_list,x_cnn, x_aux,y_main,y_neve,device,cond_dim=cond_dim,verbose=False,model_type="f",cnn_mode=False):
    n_model       = len(modelf_list)
    #is_signal  = (y_neve.squeeze() > 0)
    is_signal = (y_neve.squeeze() >= 1).any(dim=1)
    #print(is_signal.sum())

    log_pdf_dat = []
    for i in range(n_model):
        modelf_list[i].eval()
        
        if cnn_mode:
            if model_type == "f":
                cond                = modelf_list[i](x_cnn).double().to(device)
            elif model_type == "cf":
                cond, _             = modelf_list[i](x_cnn)
                cond = cond.double().to(device)
            else:
                raise KeyError
        else:
            if model_type == "f":
                cond                = modelf_list[i](x_cnn, x_aux).double().to(device)
            elif model_type == "cf":
                cond, _             = modelf_list[i](x_cnn, x_aux)
                cond = cond.double().to(device)
            else:
                raise KeyError
        
        
        log_pdf,_,_  = modelf_list[i].pdf(y_main[is_signal], conditional_input=cond[is_signal])
        log_pdf_dat.append(log_pdf)
    log_pdf_dat_s    =     torch.stack(log_pdf_dat, dim=0).T
    logsumexp = torch.logsumexp(log_pdf_dat_s, dim=1)
    norm = torch.full(logsumexp.shape, torch.tensor(np.log(n_model))).to(device)
    return (logsumexp-norm)



## GENERAL UTILS

 
#### get_top_models
def get_top_models(results,all_models_names, eval_type='test_log_loss', n_best=3, n_noiseful_in_mixed=2, n_noiseless_in_mixed=2):
    # Create lists of models with their evaluation scores for ranking
    modelf_list              = [ (i,results.results[i][eval_type]) for i in results.models.keys() if i in all_models_names ]
    modelf_list_noiseful     = [ (i,results.results[i][eval_type]) for i in results.models.keys() if i in all_models_names and "noiseless" not in i]
    modelf_list_noiseless    = [ (i,results.results[i][eval_type]) for i in results.models.keys() if i in all_models_names and "noiseless" in i]

    # Select top-performing models (sorted by evaluation metric, ascending)
    top_model_key            = [ i for i,j in sorted(modelf_list, key=lambda x: x[1])[:n_best]]
    top_model                = [ results.models[i] for i,j in sorted(modelf_list, key=lambda x: x[1])[:n_best]]

    top_model_key_noiseful   = [ i for i,j in sorted(modelf_list_noiseful, key=lambda x: x[1])[:n_best]]
    top_model_noiseful       = [ results.models[i] for i,j in sorted(modelf_list_noiseful, key=lambda x: x[1])[:n_best]]

    top_model_key_noiseless  = [ i for i,j in sorted(modelf_list_noiseless, key=lambda x: x[1])[:n_best]]
    top_model_noiseless      = [ results.models[i] for i,j in sorted(modelf_list_noiseless, key=lambda x: x[1])[:n_best]]

    # Create mixed ensemble combining both noise types
    top_model_key_mixed      = top_model_key_noiseful[:n_noiseful_in_mixed] + top_model_key_noiseless[:n_noiseful_in_mixed]
    top_model_mixed          = top_model_noiseful[:n_noiseful_in_mixed]     + top_model_noiseless[:n_noiseless_in_mixed]

    # Display ensemble composition
    print("All_models_names    : ",len(all_models_names)    , all_models_names)
    print("top_model_noiseful  : ",len(top_model_noiseful)  , top_model_key_noiseful)
    print("top_model_noiseless : ",len(top_model_noiseless) , top_model_key_noiseless)
    print("top_model_mixed     : ",len(top_model_mixed)     , top_model_key_mixed)
    return top_model_key_noiseful, top_model_key_noiseless, top_model_key_mixed, top_model_noiseful, top_model_noiseless, top_model_mixed
 
#### get_ensemble_class
def get_ensemble_class(model_list,x_cnn,x_aux,y_neve,device,model_type="cf",cnn_mode=True):
    n_model       = len(model_list)
    #is_signal  = (y_neve.squeeze() > 0)
    is_signal = (y_neve.squeeze() >= 1).any(dim=1)
    nclass_dat = []
    for i in range(n_model):
        model_list[i].eval()
        if cnn_mode:
            if model_type == "c":
                nclass                     = model_list[i](x_cnn).double().to(device)
            elif model_type == "cf":
                cond, nclass             = model_list[i](x_cnn)
                cond = cond.double().to(device)
            else:
                raise KeyError
        else:
            if model_type == "c":
                nclass                = model_list[i](x_cnn, x_aux).double().to(device)
            elif model_type == "cf":
                cond, nclass             = model_list[i](x_cnn, x_aux)
                cond = cond.double().to(device)
            else:
                raise KeyError
        nclass_dat.append(nclass.squeeze())
    return torch.stack(nclass_dat, dim=0).mean(0), torch.stack(nclass_dat, dim=0)
 
#### get_uniq_param
def get_uniq_param(results):
    uniq_cond = np.unique(((np.stack(list(results.datasets['test_dataset.pt'].flow_outputs.values())))).T,axis=0)
    uniq_cond = uniq_cond[~np.isnan(uniq_cond).all(1)]
    print(len(uniq_cond))

    iloader_dict =  {}
    for m,s in tqdm(uniq_cond):
            indat = results.datasets['test_dataset.pt'].unnormalize_batch('flow_outputs', {"MASS":m,"COUP":s})
            subset_x_hist, subset_x_others, subset_y_flow, subset_y_mlp = results.datasets['test_dataset.pt'].subset_by_multiple_conditions(
                group="flow_outputs",
                conditions={
                    "MASS" : lambda x: np.squeeze(x) == indat["MASS"],
                    "COUP": lambda x: np.squeeze(x) == indat["COUP"],
                }
            )
            idataset = fd.FlexibleMLDataset(
                histograms=subset_x_hist,
                other_inputs=subset_x_others,
                flow_outputs=subset_y_flow,
                mlp_outputs=subset_y_mlp,
                input_norm=results.datasets['test_dataset.pt'].input_norm,
                output_norm=results.datasets['test_dataset.pt'].output_norm,
                input_scalers=results.datasets['test_dataset.pt'].input_scalers,
                output_scalers=results.datasets['test_dataset.pt'].output_scalers,
                normalize_on_init=False
            )
            if len(subset_x_hist) == 0 : raise Error
            iloader = DataLoader(idataset, batch_size=1024)
            iloader_dict[(indat["MASS"],indat["COUP"])] = iloader
    return uniq_cond, iloader_dict

### evaluate_entropy
def evaluate_entropy(model, x_cnn,x_aux,is_signal,samplesize=100,full=True,model_type="cf",cnn_mode=True):
    with torch.no_grad():
        # Forward pass
        if cnn_mode:
            if model_type == "f":
                cond                     = model(x_cnn).double().to(device)
            elif model_type == "cf":
                cond, nclass             = model(x_cnn)
                cond = cond.double().to(device)
            else:
                raise KeyError
        else:
            if model_type == "f":
                cond                = model(x_cnn, x_aux).double().to(device)
            elif model_type == "cf":
                cond, nclass             = model(x_cnn, x_aux)
                cond = cond.double().to(device)
            else:
                raise KeyError

    if full:
        entropy_dict_full  = model.pdf.entropy(conditional_input=cond,samplesize=samplesize)
        return entropy_dict_full['total']
    elif full == False:
        entropy_dict       = model.pdf.entropy(conditional_input=cond[is_signal],samplesize=samplesize)
        return entropy_dict['total']
    else:
        return entropy_dict_full['total'], entropy_dict['total']

### evaluate_loss
def evaluate_loss(model, x_cnn,x_aux,is_signal,samplesize=100,full=True,model_type="cf",cnn_mode=True):
    with torch.no_grad():
        # Forward pass
        if cnn_mode:
            if model_type == "f":
                cond                     = model(x_cnn).double().to(device)
            elif model_type == "cf":
                cond, nclass             = model(x_cnn)
                cond = cond.double().to(device)
            else:
                raise KeyError
        else:
            if model_type == "f":
                cond                = model(x_cnn, x_aux).double().to(device)
            elif model_type == "cf":
                cond, nclass             = model(x_cnn, x_aux)
                cond = cond.double().to(device)
            else:
                raise KeyError
    if full:
        entropy_dict_full  = model.pdf.entropy(conditional_input=cond,samplesize=samplesize)
        return entropy_dict_full['total']
    elif full == False:
        entropy_dict       = model.pdf.entropy(conditional_input=cond[is_signal],samplesize=samplesize)
        return entropy_dict['total']
    else:
        return entropy_dict_full['total'], entropy_dict['total']


## CLASSIFIER

 
#### utils
##### get_classifier_results
def get_classifier_results(model,dataloader,cnn_mode,reruns_max=5,psl=False):
    cdict = {}
    for attempt in range(1, reruns_max + 1):
        try:
            test_class_loss, c_truth, c_pred = rcn_t.test_loop_class(model, dataloader, device,verbose=False,cnn_mode=cnn_mode,psl=psl)
            
            # Convert to tensors and compute standard classification metrics
            c_truth, c_pred      = torch.tensor(c_truth).cpu(), torch.tensor(c_pred).cpu()
            
            if all(c_truth) or not any(c_truth):
                auc         = 0
                fpr         = []
                tpr         = []
                thresholds  = []
            else:
                auc                  = roc_auc_score(c_truth, c_pred)
                fpr, tpr, thresholds = roc_curve(c_truth, c_pred)
            cdict['test_class_loss']  = test_class_loss                              
            cdict['c_truth']          = c_truth                      
            cdict['c_pred']           = c_pred                  
            cdict['auc_c']              = auc                                  
            cdict['fpr_c']              = fpr                  
            cdict['tpr_c']              = tpr                  
            cdict['thresholds_c']       = thresholds                          
            break
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == reruns_max: 
                raise RuntimeError("Maximum retry attempts exceeded")       
    return cdict

##### test_loop_ensemble_class
def test_loop_ensemble_class(dataloader,model_list,cnn_mode,verbose=True,psl=False):
    
    class_dict = {}
    c_pred     = []
    c_truth    = []
    c_pred_i   = []
    e_truth    = []
    e_truth_i  = []
    #n=0
    data_iter = tqdm(dataloader, desc="Validating", leave=False) if verbose else dataloader

    with torch.no_grad():
        for x_cnn, x_aux, y_main, y_neve in data_iter:
    #        n+=1 
    #        if n>3 : break

            # Each of these is a dict of tensors
            x_cnn     = x_cnn.double().to(device)
            x_aux     = x_aux.double().to(device)
            y_main    = y_main.double().to(device)
            y_neve    = y_neve.double().to(device)
            #is_signal = (y_neve.squeeze() > 0).any(dim=1).float().cpu().numpy()
            is_signal = (y_neve.squeeze() >= 1).any(dim=1).float().cpu().numpy()
            if psl:
                min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
                max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
                range_vals  = max_vals - min_vals
                # Avoid division where range is 0
                normalized_x_cnn = torch.where(
                    range_vals > 0,
                    (x_cnn - min_vals) / range_vals,
                    torch.zeros_like(x_cnn)
                )
                x_cnn            = normalized_x_cnn

            
            nc, nc_i = get_ensemble_class(model_list,x_cnn,x_aux,y_neve,device,cnn_mode=cnn_mode, model_type="c")
            c_pred.extend(nc)
            c_pred_i.append(nc_i)
            c_truth.extend(is_signal)
        
        class_dict["c_truth"]           =  torch.tensor(c_truth).float()
        class_dict["c_pred"]            = torch.tensor(c_pred)
        class_dict["c_pred_i"]          = torch.hstack(c_pred_i)
        class_dict["test_class_loss"]   = nn.BCEWithLogitsLoss()(class_dict["c_pred"] ,class_dict["c_truth"])
        class_dict["test_class_loss_i"] = [nn.BCEWithLogitsLoss()(i,class_dict["c_truth"].to(device)).cpu() for i in  class_dict["c_pred_i"]]
        
        if all(c_truth) or not any(c_truth):
            class_dict["auc_c"]         = 0
            class_dict["fpr_c"]         = []
            class_dict["tpr_c"]         = []
            class_dict["thresholds_c"]  = []
        else:
            class_dict["auc_c"]               = roc_auc_score(class_dict["c_truth"], class_dict["c_pred"])
            fpr, tpr, thresholds            = roc_curve(class_dict["c_truth"], class_dict["c_pred"])
            class_dict["fpr_c"]               = fpr
            class_dict["tpr_c"]               = tpr
            class_dict["thresholds_c"]        = thresholds

            class_dict["auc_c_i"]             = [roc_auc_score(class_dict["c_truth"].cpu(), i.cpu()) for i in class_dict["c_pred_i"]]
            eval_dat                   = [roc_curve(class_dict["c_truth"].cpu(), i.cpu()) for i in class_dict["c_pred_i"]]
            fpr_i, tpr_i, thresholds_i = list(zip(*eval_dat))
            class_dict["fpr_c_i"]             = fpr_i
            class_dict["tpr_c_i"]             = tpr_i
            class_dict["thresholds_c_i"]      = thresholds_i
        return class_dict

##### get_ensemble_classifier_results
def get_ensemble_classifier_results(dataloader,model_list,cnn_mode,verbose=True,reruns_max=5,psl=False):
    cdict = {}
    for attempt in range(1, reruns_max + 1):
        try:
            cdict = test_loop_ensemble_class(dataloader,model_list,cnn_mode,verbose,psl=psl)                           
            break
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == reruns_max: 
                raise RuntimeError("Maximum retry attempts exceeded")       
    return cdict
 
### eval_classifier
def eval_classifier(results,first_run=True,reruns_max=5,psl=False):
    for model_name in sorted(results.models.keys()):
        if results.model_type[model_name] != "c": 
            print(f"Not Classifier : {model_name} ({results.model_type[model_name]}). Continue.")
            continue
        
        print(f"Evaluatiing Classifier of {model_name}...\n")
        # Initialize results dictionary if needed
        if model_name not in results.results.keys(): 
             results.results[model_name] = {}

        # If the results dictionary exist, check whether the results exist in them
        elif not 'auc' in results.results[model_name].keys():
             print(f"Not first run : but c_truth of {model_name} are not there. Redo!")
             pass    

        # If the results exist, check whether rerun is required. If its first run, rerun. 
        elif not first_run: 
             print(f"Not first run : {model_name} already done")
             continue
        
        # Evaluate classification-only models
        cdict = get_classifier_results(results.models[model_name].to(device).double(), results.test_loader, cnn_mode=results.cnn_mode[model_name],psl=psl)

        # Store results for classification-only models  
        results.results[model_name].update(cdict)

### eval_ensemble_classifier
def eval_ensemble_classifier(results,first_run=True,eval_type='test_class_loss',n_best=3,n_noiseful_in_mixed=0,n_noiseless_in_mixed=0,batch_size=2048,psl=False):
     if batch_size == 2048 : pass
     else                  : results.test_loader   = DataLoader(results.datasets['test_dataset.pt'] , batch_size=batch_size, shuffle=True)
 
     # Group models by common naming patterns for ensemble creation
     results.get_grouped_names()
     for model_common_name in results.grouped_names: 
          all_models_names    = results.grouped_names[model_common_name]
          cnn_mode            = CNN_MODE_DICT[model_common_name.split("_")[0]]
          model_type          = MODEL_TYPE_DICT[model_common_name.split("_")[0]]
          
          if model_type != 'c' : 
               print(f"Not Classifier : {model_common_name} ({model_type}). Continue.")
               continue
          
          print("Evaluating Ensemble Classifier of :", model_common_name)
          
          # Initialize ensemble results storage if needed
          if model_common_name not in results.ensemble_results.keys(): 
               results.ensemble_results[model_common_name] = {}
          
          # Check if ensemble results already exist
          elif len(results.ensemble_results[model_common_name]) == 0:
               pass
          
          # If ensemble results exist, check if required keys are present
          elif 'c_truth' not in results.ensemble_results[model_common_name].keys():
               print(f"Not first run : c_truth of {model_common_name} in {results.ensemble_results[model_common_name].keys()} are not done. Redo!")
               pass
          # If ensemble results exist and not a rerun. continue
          elif not first_run: 
               print(f"Not first run : {model_common_name} already done")
               continue
          
          top_model_key, _, _, top_model, _, _ =     get_top_models(results,all_models_names,eval_type=eval_type,n_best=n_best,n_noiseful_in_mixed=n_noiseful_in_mixed,n_noiseless_in_mixed=n_noiseless_in_mixed)
          class_dict  = get_ensemble_classifier_results(results.test_loader, top_model   , cnn_mode=cnn_mode,psl=psl)
          
          if len(results.ensemble_results[model_common_name].keys()) == 0 : 
               results.ensemble_results[model_common_name]['top_model_key']  = top_model_key
          results.ensemble_results[model_common_name].update(class_dict)

### eval_classifier_by_point
def eval_classifier_by_point(results,first_run=True,reruns_max=5,psl=False):
    uniq_cond, iloader_dict = get_uniq_param(results)

    for model_name in sorted(results.models.keys()):
        
        if results.model_type[model_name] != "c": 
            print(f"{model_name} ({results.model_type[model_name]}): Not Classifier continue.")
            continue

        print(f"Evaluating Classifier (by_point) of {model_name}...\n")
        if model_name not in results.results_by_param.keys(): 
            results.results_by_param[model_name] = {}

        elif len(results.results_by_param[model_name].keys()) != len(iloader_dict):
            print(f"Missing some param points : Exist {len(iloader_dict)}, have {len(results.results_by_param[model_name].keys())} instead. Rerun")
            pass    

        elif any(['c_truth' not in i.keys() for _, i in results.results_by_param[model_name].items()]):
            print(f"Missing some param points runs. Rerun")
            print(['c_truth' not in i.keys() for _, i in results.results_by_param[model_name].items()])
        
        elif not first_run: 
            print(f"Not first run : {model_name} already done")
            continue 

        for param,iloader in  tqdm(iloader_dict.items()):
            m,s = param
            if (m,s) not in results.results_by_param[model_name].keys():
                results.results_by_param[model_name][(m,s)] = {}      
            
            # Evaluate classification-only models
            cdict = get_classifier_results(results.models[model_name].to(device).double(), iloader, cnn_mode=results.cnn_mode[model_name],psl=psl)
    
            # Store results for classification-only models  
            results.results_by_param[model_name][(m,s)].update(cdict)
 
### eval_ensemble_classifier_by_point
def eval_ensemble_classifier_by_point(results,eval_type="test_class_loss",first_run=True,reruns_max=5,psl=False):
     uniq_cond, iloader_dict = get_uniq_param(results)
    
     # Group models by common naming patterns for ensemble creation
     results.get_grouped_names()
     for model_common_name in results.grouped_names: 
          all_models_names    = results.grouped_names[model_common_name]
          cnn_mode            = CNN_MODE_DICT[model_common_name.split("_")[0]]
          model_type          = MODEL_TYPE_DICT[model_common_name.split("_")[0]]
          
          if model_type != 'c' : 
               print(f"Not Classifier : {model_common_name} ({model_type}). Continue.")
               continue
          
          print("Evaluating Ensemble Classifier (by_point) of :", model_common_name)
          
          # Initialize ensemble results storage if needed
          if model_common_name not in results.ensemble_results_by_param.keys(): 
               results.ensemble_results_by_param[model_common_name] = {}
          
          elif len(results.ensemble_results_by_param[model_common_name].keys()) != len(iloader_dict):
               print(f"Missing some param points : Exist {len(iloader_dict)}, have {len(results.ensemble_results_by_param[model_common_name].keys())} instead. Rerun")
               pass    

          elif any(['c_truth' not in i.keys() for _, i in results.ensemble_results_by_param[model_common_name].items()]):
               print(f"Missing some param points runs. Rerun")
               print(['c_truth' not in i.keys() for _, i in results.ensemble_results_by_param[model_common_name].items()])

          elif not first_run: 
               print(f"Not first run : {model_common_name} already done")
               continue 
          
          top_model_key, _, _, top_model, _, _ =     get_top_models(results,all_models_names,eval_type=eval_type)
          for param,iloader in  tqdm(iloader_dict.items()):
               m,s = param
               if (m,s) not in results.ensemble_results_by_param[model_common_name].keys():
                    results.ensemble_results_by_param[model_common_name][(m,s)] = {}   
                    
               class_dict  = get_ensemble_classifier_results(iloader, top_model   , cnn_mode=cnn_mode , verbose=False,psl=psl)
               results.ensemble_results_by_param[model_common_name][(m,s)]['top_model_key']  = top_model_key
               results.ensemble_results_by_param[model_common_name][(m,s)].update(class_dict)

 

## FLOWS

### utils

 
#### flow_forward
def flow_forward(model,x_cnn, x_aux,y_main,y_neve,cnn_mode):
    #is_signal  = (y_neve.squeeze() > 0)
    is_signal = (y_neve.squeeze() >= 1).any(dim=1)

    # Forward pass
    if cnn_mode:
        cond   = model(x_cnn)  
    else:
        cond   = model(x_cnn, x_aux)  
    
    if is_signal.any():    
        log_pdf,_,_  = model.pdf(y_main[is_signal], conditional_input=cond[is_signal])
        return log_pdf
 
#### get_flows_results
def get_flows_results(model,dataloader,cnn_mode,reruns_max=5,more_eval_info=False,results=None,verbose=True,psl=False,samplesize=100):
    if more_eval_info == True: assert results is not None
    for attempt in range(1, reruns_max + 1):
        try:
            
            fdict = test_loop_flows(model,
                                dataloader,
                                device=device,
                                verbose=verbose,
                                samplesize=samplesize,
                                plot=False,
                                cnn_mode=cnn_mode,
                                more_eval_info=more_eval_info,
                                results=results,
                                psl=psl)
                    
            break
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            traceback.print_exc()

            if attempt == reruns_max: 
                raise RuntimeError("Maximum retry attempts exceeded")       
    return fdict

#### test_loop_flows
def test_loop_flows(modelf,  val_loader, device,verbose=False,samplesize=100,flow_eval=True,cnn_mode=False,plot=True,more_eval_info=False,results=None,psl=False):
    modelf.eval()
    
    fdict = {}
    
    lowest_ent  = np.inf
    lpdf        = torch.tensor([]).to(device)
    ent         = torch.tensor([]).to(device)
    ent_full    = torch.tensor([]).to(device)    
    signif           = {"PT":torch.tensor([]),"MET":torch.tensor([]),"ETA":torch.tensor([])}
    c_truth     = []               

    nsig = 0
    data_iter = tqdm(val_loader, desc="Validating", leave=False) if verbose else val_loader
    with torch.no_grad():
        for x_cnn, x_aux, y_main, y_neve in data_iter:
            # Each of these is a dict of tensors
            x_cnn      = x_cnn.to(device).double()
            x_aux      = x_aux.to(device).double()
            y_main     = y_main.to(device).double()
            y_neve     = y_neve.to(device).double()
            if psl:
                min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
                max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
                range_vals  = max_vals - min_vals
                # Avoid division where range is 0
                normalized_x_cnn = torch.where(
                    range_vals > 0,
                    (x_cnn - min_vals) / range_vals,
                    torch.zeros_like(x_cnn)
                )
                x_cnn            = normalized_x_cnn

            #is_signal = (y_neve.squeeze() > 0).any(dim=1)
            is_signal = (y_neve.squeeze() >= 1).any(dim=1)
            nsig      += sum(is_signal)

            c_truth.extend(is_signal)
            
            # Forward pass
            if cnn_mode:
                cond   = modelf(x_cnn)  
            else:
                cond   = modelf(x_cnn, x_aux)  
           
            if is_signal.any():    
                log_pdf,_,_  = modelf.pdf(y_main[is_signal], conditional_input=cond[is_signal])
                lpdf         = torch.hstack([lpdf,log_pdf])
                
            if flow_eval:    
                entropy_full_dict = modelf.pdf.entropy(conditional_input=cond,samplesize=samplesize)
                ent               = torch.hstack([ent,entropy_full_dict["total"][is_signal]])
                ent_full          = torch.hstack([ent_full,entropy_full_dict["total"]])

                if more_eval_info and is_signal.any():
                    x_aux_dict      =  {i:j for i,j in zip(results.datasets['test_dataset.pt'].other_inputs.keys(),x_aux[is_signal].T.cpu())}
                    y_neve_dict     =  {i:j for i,j in zip(results.datasets['test_dataset.pt'].mlp_outputs.keys(),y_neve[is_signal].T.cpu())}
                    unnormed_x_aux  = results.datasets['test_dataset.pt'].unnormalize_batch('other_inputs',x_aux_dict)
                    unnormed_y_neve = results.datasets['test_dataset.pt'].unnormalize_batch('mlp_outputs',y_neve_dict)

                    nevents_PT       = unnormed_x_aux['PT_NEVENTS']
                    nevents_MET      = unnormed_x_aux['MET_NEVENTS']
                    nevents_ETA      = unnormed_x_aux['ETA_NEVENTS']
                    nevents_PT_sg    = unnormed_y_neve['PT_NEVENTS_SIG']
                    nevents_MET_sg   = unnormed_y_neve['MET_NEVENTS_SIG']
                    nevents_ETA_sg   = unnormed_y_neve['ETA_NEVENTS_SIG']
                    nevents_PT_bg    = nevents_PT - nevents_PT_sg
                    nevents_MET_bg   = nevents_MET - nevents_MET_sg
                    nevents_ETA_bg   = nevents_ETA - nevents_ETA_sg

                    if Z_SIG:
                        significance_PT  = z_sig(nevents_PT_sg, nevents_PT_bg)
                        significance_MET = z_sig(nevents_MET_sg, nevents_MET_bg)
                        significance_ETA = z_sig(nevents_ETA_sg, nevents_ETA_bg)
                    else:
                        significance_PT  = nevents_PT_sg / torch.sqrt(nevents_PT_bg)
                        significance_MET = nevents_MET_sg / torch.sqrt(nevents_MET_bg)
                        significance_ETA = nevents_ETA_sg / torch.sqrt(nevents_ETA_bg)
                    
                    signif['PT']     = torch.hstack([signif['PT'],significance_PT])
                    signif['MET']    = torch.hstack([signif['MET'],significance_MET])
                    signif['ETA']    = torch.hstack([signif['ETA'],significance_ETA])

                    if entropy_full_dict["total"][is_signal].min() < lowest_ent:
                        sorted_indices              = torch.argsort(entropy_full_dict["total"][is_signal])
                        lowest_ent                  = entropy_full_dict["total"][is_signal][sorted_indices][0]
                        fdict['bestsr_x_cnn']       = x_cnn[is_signal][sorted_indices][0].cpu()
                        fdict['bestsr_x_aux']       = x_aux[is_signal][sorted_indices][0].cpu()
                        fdict['bestsr_y_main']      = y_main[is_signal][sorted_indices][0].cpu()
                        fdict['bestsr_y_neve']      = y_neve[is_signal][sorted_indices][0].cpu()
                        fdict['bestsr_signif_PT']   = significance_PT[sorted_indices.cpu()][0]
                        fdict['bestsr_signif_MET']  = significance_MET[sorted_indices.cpu()][0]
                        fdict['bestsr_signif_ETA']  = significance_ETA[sorted_indices.cpu()][0]
                        fdict["bestsr_nlpdf"]       = (-log_pdf)[sorted_indices.cpu()][0]
                        fdict["bestsr_ent"]         = lowest_ent
             

    fdict['test_log_loss']   = ((-lpdf).sum() / nsig).cpu().item()
    fdict['nlpdf']           = -lpdf
    fdict['ent']             = ent
    fdict['ent_full']        = ent_full
    fdict["c_truth"]         = torch.tensor(c_truth).float()

    auc, fpr, tpr, thresholds, e_logits = get_evaluation_metrics(fdict['ent_full'].cpu(), fdict["c_truth"].cpu())
    fdict['auc_e']             = auc
    fdict['fpr_e']             = fpr
    fdict['tpr_e']             = tpr
    fdict['thresholds_e']      = thresholds
    fdict['e_logits_e']        = e_logits
    
    if flow_eval:    
        tail_percentile            = 95
        intensity_percentile       = 95
        ediff                      = (-lpdf-ent).cpu().numpy()
        fdict['ediff_mean']        = ediff.mean()
        fdict['ediff_std']         = ediff.std()
        fdict['tail_thresh']       = np.percentile(ediff, tail_percentile)

        tail_prop                  = np.sum(ediff > fdict['tail_thresh']) / len(ediff)
        absediff                   = np.abs(ediff)
        T                          = np.percentile(absediff, intensity_percentile)
        excess                     = np.sum([x - T for x in absediff if x > T])
        fdict['outlier_intensity'] = excess / np.sum(absediff) if np.sum(absediff) > 0 else 0
        
        if verbose:
            print(f"Validation Log Loss  = {fdict['test_log_loss']    :.4f}")
            print(f"ediff.mean()         = {fdict['ediff_mean']       :.4f}")
            print(f"ediff.std()          = {fdict['ediff_std']        :.4f}")
            print(f"tail_thresh          = {fdict['tail_thresh']      :.4f}")
            print(f"outlier_intensity    = {fdict['outlier_intensity']:.4f}")
        if plot:
            plt.hist2d(-lpdf.cpu(),ent.cpu(),bins=50,norm=LogNorm())
            plt.xlabel("loss")
            plt.ylabel("ent")
            plt.show()
        if more_eval_info:
            fdict['signif']          = signif  
    
    return fdict
 
#### get_ensemble_flow_results
def get_ensemble_flow_results(dataloader,model_list,cnn_mode,model_type,verbose=True,more_eval_info=False,results=None,reruns_max=5,psl=False):
    if more_eval_info == True: assert results is not None
    for attempt in range(1, reruns_max + 1):
        try:
            fdict = test_loop_ensemble_flows(dataloader, 
                                          model_list, 
                                          cnn_mode, 
                                          model_type, 
                                          verbose=verbose, 
                                          more_eval_info=more_eval_info, 
                                          results=results,
                                          psl=psl)
            break
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            traceback.print_exc()
            if attempt == reruns_max: 
                raise RuntimeError("Maximum retry attempts exceeded")       
    return fdict

#### test_loop_ensemble_flows
def test_loop_ensemble_flows(dataloader,model_list,cnn_mode,model_type,verbose=True,more_eval_info=False,results=None,psl=False):
 
    edat_ensemble      = [] # Signal only 
    edat_ensemble_full = [] # Both Sg and Bg
    ldat_ensemble      = [] # Loss [only with signal] 
    edat_i_full        = []              
    edat_i             = []          
    ldat_i             = []          

    c_truth = []
    #signif             = torch.tensor([])
    signif           = {"PT":torch.tensor([]),"MET":torch.tensor([]),"ETA":torch.tensor([])}
    neve_sg           = {"PT":torch.tensor([]),"MET":torch.tensor([]),"ETA":torch.tensor([])}

    lowest_ent    = np.inf
    data_iter = tqdm(dataloader, desc="Validating", leave=True) if verbose else dataloader
    
    fdict = {}
    nsig  = 0 

    with torch.no_grad():
        for i in model_list: i.eval()
        
        for x_cnn, x_aux, y_main, y_neve in data_iter:

            x_cnn      = x_cnn.to(device).double()
            x_aux      = x_aux.to(device).double()
            y_main     = y_main.to(device).double()
            y_neve     = y_neve.to(device).double()
            #is_signal  = (y_neve.squeeze() > 0)
            is_signal = (y_neve.squeeze() >= 1).any(dim=1)
            nsig += sum(is_signal)
            c_truth.extend(is_signal)
            if psl:
                min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
                max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
                range_vals  = max_vals - min_vals
                # Avoid division where range is 0
                normalized_x_cnn = torch.where(
                    range_vals > 0,
                    (x_cnn - min_vals) / range_vals,
                    torch.zeros_like(x_cnn)
                )
                x_cnn            = normalized_x_cnn

            # Ensemble Eval
            try:
                e   = evaluate_ensemble_entropy_faster(model_list,x_cnn,x_aux, 100, device,model_type=model_type,cnn_mode=cnn_mode)  
                l   = evaluate_ensemble_loss(model_list,x_cnn,x_aux,y_main,y_neve,device,model_type=model_type,cnn_mode=cnn_mode)     
                
                e_i_full = torch.stack([evaluate_entropy(model, x_cnn,x_aux,is_signal,samplesize=100, full=True, model_type=model_type,cnn_mode=cnn_mode) for model in model_list])
                e_i      = torch.stack([en[is_signal] for en in e_i_full])
                l_i      = torch.stack([-(flow_forward(model,x_cnn,x_aux,y_main,y_neve,cnn_mode=cnn_mode)) for model in model_list])
             
            except Exception as e :
                print("Fucked (Ensemble) :",str(e))
                traceback.print_exc()

                e = evaluate_ensemble_entropy_faster(model_list,x_cnn,x_aux, 100, device,model_type=model_type,cnn_mode=cnn_mode)  
                l = evaluate_ensemble_loss(model_list,x_cnn,x_aux,y_main,y_neve,device,model_type=model_type,cnn_mode=cnn_mode)       
                           
                e_i_full = torch.stack([evaluate_entropy(model, x_cnn,x_aux,is_signal,samplesize=100, full=True, model_type=model_type,cnn_mode=cnn_mode) for model in model_list])
                e_i      = torch.stack([en[is_signal] for en in e_i_full])
                l_i      = torch.stack([-(flow_forward(model,x_cnn,x_aux,y_main,y_neve,cnn_mode=cnn_mode)) for model in model_list])

            edat_ensemble_full.append(e.cpu())
            edat_i_full.append(e_i_full)

            if len(e[is_signal]) != 0  : 
                edat_ensemble.append(e[is_signal].cpu())
                edat_i.append(e_i)
            if len(l) != 0             : 
                ldat_ensemble.append((-l).cpu())
                ldat_i.append(l_i)

            if more_eval_info:
                x_aux_dict      =  {i:j for i,j in zip(results.datasets['test_dataset.pt'].other_inputs.keys(),x_aux[is_signal].T.cpu())}
                y_neve_dict     =  {i:j for i,j in zip(results.datasets['test_dataset.pt'].mlp_outputs.keys(),y_neve[is_signal].T.cpu())}
                unnormed_x_aux  = results.datasets['test_dataset.pt'].unnormalize_batch('other_inputs',x_aux_dict)
                unnormed_y_neve = results.datasets['test_dataset.pt'].unnormalize_batch('mlp_outputs',y_neve_dict)

                nevents_PT       = unnormed_x_aux['PT_NEVENTS']
                nevents_MET      = unnormed_x_aux['MET_NEVENTS']
                nevents_ETA      = unnormed_x_aux['ETA_NEVENTS']
                nevents_PT_sg    = unnormed_y_neve['PT_NEVENTS_SIG']
                nevents_MET_sg   = unnormed_y_neve['MET_NEVENTS_SIG']
                nevents_ETA_sg   = unnormed_y_neve['ETA_NEVENTS_SIG']
                nevents_PT_bg    = nevents_PT - nevents_PT_sg
                nevents_MET_bg   = nevents_MET - nevents_MET_sg
                nevents_ETA_bg   = nevents_ETA - nevents_ETA_sg
                significance_PT  = nevents_PT_sg / torch.sqrt(nevents_PT_bg)
                significance_MET = nevents_MET_sg / torch.sqrt(nevents_MET_bg)
                significance_ETA = nevents_ETA_sg / torch.sqrt(nevents_ETA_bg)
                signif['PT']     = torch.hstack([signif['PT'],significance_PT])
                signif['MET']    = torch.hstack([signif['MET'],significance_MET])
                signif['ETA']    = torch.hstack([signif['ETA'],significance_ETA])
                neve_sg['PT']     = torch.hstack([neve_sg['PT'],nevents_PT_sg])
                neve_sg['MET']    = torch.hstack([neve_sg['MET'],nevents_MET_sg])
                neve_sg['ETA']    = torch.hstack([neve_sg['ETA'],nevents_ETA_sg])
                

                if e.min() < lowest_ent: 
                    sorted_indices = torch.argsort(e[is_signal])
                    lowest_ent = e.min()
                    fdict["bestsr_x_cnn"]   = x_cnn[is_signal][sorted_indices][0]
                    fdict["bestsr_x_aux"]   = x_aux[is_signal][sorted_indices][0]
                    fdict["bestsr_y_main"]  = y_main[is_signal][sorted_indices][0]
                    fdict["bestsr_y_neve"]  = y_neve[is_signal][sorted_indices][0]
                    #fdict["bestsr_signif"]  = significance[sorted_indices.cpu()][0]
                    fdict['bestsr_signif_PT']   = significance_PT[sorted_indices.cpu()][0]
                    fdict['bestsr_signif_MET']  = significance_MET[sorted_indices.cpu()][0]
                    fdict['bestsr_signif_ETA']  = significance_ETA[sorted_indices.cpu()][0]
                    fdict["bestsr_nlpdf"]   = (-l)[sorted_indices.cpu()][0]
                    fdict["bestsr_ent"]     = lowest_ent



        fdict["ent"]             = torch.cat(edat_ensemble)
        fdict["ent_full"]        = torch.cat(edat_ensemble_full)
        fdict["nlpdf"]           = torch.cat(ldat_ensemble)
        fdict['test_log_loss']   = (fdict["nlpdf"].sum() / nsig).cpu().item()

        fdict["ent_i"]           = torch.hstack(edat_i)
        fdict["ent_i_full"]      = torch.hstack(edat_i_full)
        fdict["nlpdf_i"]         = torch.hstack(ldat_i)
        fdict['test_log_loss_i'] = [(i.sum() / nsig).cpu().item() for i in fdict["nlpdf_i"]]

        fdict['signif']          = signif 
        fdict['neve_sg']         = neve_sg 
        fdict["c_truth"]         =  torch.tensor(c_truth).float()
        auc, fpr, tpr, thresholds, e_logits           = get_evaluation_metrics(fdict['ent_full'].cpu(), fdict["c_truth"].float().cpu())
        eval_dat                                      = [get_evaluation_metrics(e.cpu(), fdict["c_truth"].float().cpu()) for e in fdict["ent_i_full"]]
        auc_i, fpr_i, tpr_i, thresholds_i, e_logits_i = list(zip(*eval_dat))
        
        fdict['auc_e']              = auc               
        fdict['fpr_e']              = fpr               
        fdict['tpr_e']              = tpr               
        fdict['thresholds_e']       = thresholds                   
        fdict['e_logits_e']         = e_logits                   
        
        fdict['auc_e_i']         = auc_i                   
        fdict['fpr_e_i']         = fpr_i                   
        fdict['tpr_e_i']         = tpr_i                   
        fdict['thresholds_e_i']  = thresholds_i                           
        fdict['e_logits_e_i']    = e_logits_i                       
        
    return fdict

# #### test_loop_ensemble_flows
# def test_loop_ensemble_flows(dataloader,model_list,cnn_mode,model_type,verbose=True,more_eval_info=False,results=None,psl=False):
 
#     edat_ensemble      = [] # Signal only 
#     edat_ensemble_full = [] # Both Sg and Bg
#     ldat_ensemble      = [] # Loss [only with signal] 
#     edat_i_full        = []              
#     edat_i             = []          
#     ldat_i             = []          

#     c_truth = []
#     signif             = torch.tensor([])

#     lowest_ent    = np.inf
#     data_iter = tqdm(dataloader, desc="Validating", leave=True) if verbose else dataloader
    
#     fdict = {}
#     nsig  = 0 

#     with torch.no_grad():
#         for i in model_list: i.eval()
        
#         for x_cnn, x_aux, y_main, y_neve in data_iter:

#             x_cnn      = x_cnn.to(device).double()
#             x_aux      = x_aux.to(device).double()
#             y_main     = y_main.to(device).double()
#             y_neve     = y_neve.to(device).double()
#             #is_signal  = (y_neve.squeeze() > 0)
#             is_signal = (y_neve.squeeze() >= 1).any(dim=1)
#             nsig += sum(is_signal)
#             c_truth.extend(is_signal)
#             if psl:
#                 min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
#                 max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
#                 range_vals  = max_vals - min_vals
#                 # Avoid division where range is 0
#                 normalized_x_cnn = torch.where(
#                     range_vals > 0,
#                     (x_cnn - min_vals) / range_vals,
#                     torch.zeros_like(x_cnn)
#                 )
#                 x_cnn            = normalized_x_cnn

#             # Ensemble Eval
#             try:
#                 e   = evaluate_ensemble_entropy_faster(model_list,x_cnn,x_aux, 100, device,model_type=model_type,cnn_mode=cnn_mode)  
#                 l   = evaluate_ensemble_loss(model_list,x_cnn,x_aux,y_main,y_neve,device,model_type=model_type,cnn_mode=cnn_mode)     
                
#                 e_i_full = torch.stack([evaluate_entropy(model, x_cnn,x_aux,is_signal,samplesize=100, full=True, model_type=model_type,cnn_mode=cnn_mode) for model in model_list])
#                 e_i      = torch.stack([en[is_signal] for en in e_i_full])
#                 l_i      = torch.stack([-(flow_forward(model,x_cnn,x_aux,y_main,y_neve,cnn_mode=cnn_mode)) for model in model_list])
             
#             except Exception as e :
#                 print("Fucked (Ensemble) :",str(e))
#                 traceback.print_exc()

#                 e = evaluate_ensemble_entropy_faster(model_list,x_cnn,x_aux, 100, device,model_type=model_type,cnn_mode=cnn_mode)  
#                 l = evaluate_ensemble_loss(model_list,x_cnn,x_aux,y_main,y_neve,device,model_type=model_type,cnn_mode=cnn_mode)       
                           
#                 e_i_full = torch.stack([evaluate_entropy(model, x_cnn,x_aux,is_signal,samplesize=100, full=True, model_type=model_type,cnn_mode=cnn_mode) for model in model_list])
#                 e_i      = torch.stack([en[is_signal] for en in e_i_full])
#                 l_i      = torch.stack([-(flow_forward(model,x_cnn,x_aux,y_main,y_neve,cnn_mode=cnn_mode)) for model in model_list])

#             edat_ensemble_full.append(e.cpu())
#             edat_i_full.append(e_i_full)

#             if len(e[is_signal]) != 0  : 
#                 edat_ensemble.append(e[is_signal].cpu())
#                 edat_i.append(e_i)
#             if len(l) != 0             : 
#                 ldat_ensemble.append((-l).cpu())
#                 ldat_i.append(l_i)

#             if more_eval_info:
#                 x_aux_dict      =  {i:j for i,j in zip(results.datasets['test_dataset.pt'].other_inputs.keys(),x_aux[is_signal].T.cpu())}
#                 y_neve_dict     =  {i:j for i,j in zip(results.datasets['test_dataset.pt'].mlp_outputs.keys(),y_neve[is_signal].T.cpu())}
#                 unnormed_x_aux  = results.datasets['test_dataset.pt'].unnormalize_batch('other_inputs',x_aux_dict)
#                 unnormed_y_neve = results.datasets['test_dataset.pt'].unnormalize_batch('mlp_outputs',y_neve_dict)

#                 nevents         = unnormed_x_aux['nevents']
#                 nevents_sg      = unnormed_y_neve['nevents_sg']
#                 nevents_bg      = nevents - nevents_sg
#                 significance    = nevents_sg / torch.sqrt(nevents_bg)
#                 signif          = torch.hstack([signif,significance])

#                 if e.min() < lowest_ent: 
#                     sorted_indices = torch.argsort(e[is_signal])
#                     lowest_ent = e.min()
#                     fdict["bestsr_x_cnn"]   = x_cnn[is_signal][sorted_indices][0]
#                     fdict["bestsr_x_aux"]   = x_aux[is_signal][sorted_indices][0]
#                     fdict["bestsr_y_main"]  = y_main[is_signal][sorted_indices][0]
#                     fdict["bestsr_y_neve"]  = y_neve[is_signal][sorted_indices][0]
#                     fdict["bestsr_signif"]  = significance[sorted_indices.cpu()][0]
#                     fdict["bestsr_nlpdf"]   = (-l)[sorted_indices.cpu()][0]
#                     fdict["bestsr_ent"]     = lowest_ent



#         fdict["ent"]             = torch.cat(edat_ensemble)
#         fdict["ent_full"]        = torch.cat(edat_ensemble_full)
#         fdict["nlpdf"]           = torch.cat(ldat_ensemble)
#         fdict['test_log_loss']   = (fdict["nlpdf"].sum() / nsig).cpu().item()

#         fdict["ent_i"]           = torch.hstack(edat_i)
#         fdict["ent_i_full"]      = torch.hstack(edat_i_full)
#         fdict["nlpdf_i"]         = torch.hstack(ldat_i)
#         fdict['test_log_loss_i'] = [(i.sum() / nsig).cpu().item() for i in fdict["nlpdf_i"]]

#         fdict["signif"]          = signif    
#         fdict["c_truth"]         =  torch.tensor(c_truth).float()
#         auc, fpr, tpr, thresholds, e_logits           = get_evaluation_metrics(fdict['ent_full'].cpu(), fdict["c_truth"].float().cpu())
#         eval_dat                                      = [get_evaluation_metrics(e.cpu(), fdict["c_truth"].float().cpu()) for e in fdict["ent_i_full"]]
#         auc_i, fpr_i, tpr_i, thresholds_i, e_logits_i = list(zip(*eval_dat))
        
#         fdict['auc_e']              = auc               
#         fdict['fpr_e']              = fpr               
#         fdict['tpr_e']              = tpr               
#         fdict['thresholds_e']       = thresholds                   
#         fdict['e_logits_e']         = e_logits                   
        
#         fdict['auc_e_i']         = auc_i                   
#         fdict['fpr_e_i']         = fpr_i                   
#         fdict['tpr_e_i']         = tpr_i                   
#         fdict['thresholds_e_i']  = thresholds_i                           
#         fdict['e_logits_e_i']    = e_logits_i                       
        
#     return fdict
 
### eval_flows
def eval_flows(results,first_run=True,reruns_max=5,more_eval_info=False,psl=False,skip_noiseless=True):
        
    for model_name in sorted(results.models.keys()):
 
        if results.model_type[model_name] != "f": 
            print(f"{model_name} ({results.model_type[model_name]}): Not Flow continue.")
            continue

        if skip_noiseless and "noiseless" in model_name:
            print(f"{model_name} Skipping noiseless turned on.")
            continue

        print(f"Evaluating Flow of {model_name}...\n")
        # Initialize results dictionary if needed
        if model_name not in results.results.keys(): 
            results.results[model_name] = {}

        # If the results dictionary exist, check whether the results exist in them
        elif 'test_log_loss' not in results.results[model_name].keys():
            print(f"Not first run : test_log_loss of {model_name} are not done. Redo!")
            pass    

        # If the results exist, check whether rerun is required. If its first run, rerun. 
        elif not first_run: 
            print(f"Not first run : {model_name} already done")
            continue
      
        # Evaluate classification-only models
        fdict = get_flows_results(results.models[model_name].to(device).double(), results.test_loader, cnn_mode=results.cnn_mode[model_name],more_eval_info=more_eval_info,results=results,reruns_max=reruns_max,psl=psl)

        # Store results for classification-only models  
        results.results[model_name].update(fdict)
        
### eval_ensemble_flows
def eval_ensemble_flows(results,first_run=True,eval_type='test_log_loss',n_best=3,n_noiseful_in_mixed=2,n_noiseless_in_mixed=2,batch_size=2048,more_eval_info=False,psl=False):
        
    if batch_size == 2048 : 
        pass
    else: 
        results.test_loader   = DataLoader(results.datasets['test_dataset.pt'] , batch_size=batch_size, shuffle=True)

        
    # Group models by common naming patterns for ensemble creation
    results.get_grouped_names()

    for model_common_name in results.grouped_names: 

        # Skip pure classification models (no flow component)
        if MODEL_TYPE_DICT[model_common_name.split("_")[0]] != 'f' : continue

        all_models_names    = results.grouped_names[model_common_name]
        cnn_mode            = CNN_MODE_DICT[model_common_name.split("_")[0]]
        model_type          = MODEL_TYPE_DICT[model_common_name.split("_")[0]]
        
        print("Evaluating Ensemble Flow of :",model_common_name)

        # Initialize ensemble results storage if needed
        if model_common_name not in results.ensemble_results.keys(): 
            results.ensemble_results[model_common_name] = {}

        # Check if ensemble results already exist
        elif len(results.ensemble_results[model_common_name]) == 0:
            pass

        # If ensemble results exist, check if required keys are present
        elif any(['ent' not in i.keys() for _, i in results.ensemble_results[model_common_name].items()]):
            print(f"Not first run : edat_ensemble of {model_common_name} in {[k for k, i in results.ensemble_results[model_common_name].items() if 'edat_ensemble' not in i.keys() ]} are not done. Redo!")
            pass
        
        # If ensemble results exist and not a rerun. continue
        elif not first_run: 
            print(f"Not first run : {model_common_name} already done")
            continue
        
        top_model_key_noiseful, top_model_key_noiseless, top_model_key_mixed, top_model_noiseful, top_model_noiseless, top_model_mixed =  get_top_models(results,all_models_names,eval_type=eval_type,n_best=n_best,n_noiseful_in_mixed=n_noiseful_in_mixed,n_noiseless_in_mixed=n_noiseless_in_mixed)
            
        # Store ensemble results organized by noise configuration
        results.ensemble_results[model_common_name] = {"noiseful":{},"noiseless":{},"mixed":{}}

        # Store noiseful ensemble results
        fdict_noiseful  = get_ensemble_flow_results(results.test_loader, top_model_noiseful,  cnn_mode=cnn_mode, model_type=model_type,more_eval_info=more_eval_info,results=results,psl=psl)
        results.ensemble_results[model_common_name]['noiseful']['top_model_key'] = top_model_key_noiseful
        results.ensemble_results[model_common_name]['noiseful'].update(fdict_noiseful)            

        # # Store noiseless ensemble results
        # fdict_noiseless = get_ensemble_flow_results(results.test_loader, top_model_noiseless, cnn_mode=cnn_mode, model_type=model_type,more_eval_info=more_eval_info,results=results,psl=psl)
        # results.ensemble_results[model_common_name]['noiseless']['top_model_key'] = top_model_key_noiseless
        # results.ensemble_results[model_common_name]['noiseless'].update(fdict_noiseless)            
        
        # # Store mixed ensemble results
        # fdict_mixed     = get_ensemble_flow_results(results.test_loader, top_model_mixed,     cnn_mode=cnn_mode, model_type=model_type,more_eval_info=more_eval_info,results=results,psl=psl)
        # results.ensemble_results[model_common_name]['mixed']['top_model_key']     = top_model_key_mixed
        # results.ensemble_results[model_common_name]['mixed'].update(fdict_mixed)            

### eval_flows_by_point
def eval_flows_by_point(results,first_run=True,reruns_max=5,more_eval_info=True,psl=False):
    uniq_cond, iloader_dict = get_uniq_param(results)

    for model_name in sorted(results.models.keys()):
 
        print(f"Evaluating Flow of {model_name}...\n")
        if results.model_type[model_name] != "f": 
            print(f"{model_name} ({results.model_type[model_name]}): Not Flow continue.")
            continue

        # Initialize results dictionary if needed
        if model_name not in results.results_by_param.keys(): 
            results.results_by_param[model_name] = {}
        
        elif len(results.results_by_param[model_name].keys()) != len(iloader_dict):
            print(f"Missing some param points : Exist {len(iloader_dict)}, have {len(results.results_by_param[model_name].keys())} instead. Rerun")
            pass  

        elif any(['test_log_loss' not in i.keys() for _, i in results.results_by_param[model_name].items()]):
            print(f"Missing some param points runs. Rerun")
            print([k for k, i in results.results_by_param[model_name].items() if 'test_log_loss' not in i.keys()])

        # If the results exist, check whether rerun is required. If its first run, rerun. 
        elif not first_run: 
            print(f"Not first run : {model_name} already done")
            continue

        for param,iloader in  tqdm(iloader_dict.items()):
            m,s = param
            if (m,s) not in results.results_by_param[model_name].keys():
                results.results_by_param[model_name][(m,s)] = {}      

            # Evaluate classification-only models
            fdict = get_flows_results(results.models[model_name].to(device).double(), iloader, verbose=False,cnn_mode=results.cnn_mode[model_name],more_eval_info=more_eval_info,results=results,psl=psl)
        
            # Store results for classification-only models  
            results.results_by_param[model_name][(m,s)].update(fdict)

### eval_ensemble_flows_by_points
def eval_ensemble_flows_by_points(results,first_run=True,eval_type='test_log_loss',n_best=3,n_noiseful_in_mixed=2,n_noiseless_in_mixed=2,batch_size=2048,more_eval_info=True,psl=False):
     uniq_cond, iloader_dict = get_uniq_param(results)
    
     # Group models by common naming patterns for ensemble creation
     results.get_grouped_names()
     for model_common_name in results.grouped_names: 
          if MODEL_TYPE_DICT[model_common_name.split("_")[0]] != 'f' : continue

          all_models_names    = results.grouped_names[model_common_name]
          cnn_mode            = CNN_MODE_DICT[model_common_name.split("_")[0]]
          model_type          = MODEL_TYPE_DICT[model_common_name.split("_")[0]]
          
          print("Evaluating Ensemble Flow (by_point) of :", model_common_name)
          
          # Initialize ensemble results storage if needed
          if model_common_name not in results.ensemble_results_by_param.keys(): 
               results.ensemble_results_by_param[model_common_name] = {}
          
          elif len(results.ensemble_results_by_param[model_common_name].keys()) != len(iloader_dict):
               print(f"Missing some param points : Exist {len(iloader_dict)}, have {len(results.ensemble_results_by_param[model_common_name].keys())} instead. Rerun")
               pass    

          elif any(['test_log_loss' not in i.keys() for _, i in results.ensemble_results_by_param[model_common_name].items()]):
               print(f"Missing some param points runs. Rerun")
               print([k for k, i in results.ensemble_results_by_param[model_common_name].items() if 'test_log_loss' not in i.keys()])

          elif not first_run: 
               print(f"Not first run : {model_common_name} already done")
               continue 
          
          top_model_key_noiseful, top_model_key_noiseless, top_model_key_mixed, top_model_noiseful, top_model_noiseless, top_model_mixed =  get_top_models(results,all_models_names,eval_type=eval_type,n_best=n_best,n_noiseful_in_mixed=n_noiseful_in_mixed,n_noiseless_in_mixed=n_noiseless_in_mixed)
          
          for param,iloader in  tqdm(iloader_dict.items()):
               
          
               m,s = param
               if (m,s) not in results.ensemble_results_by_param[model_common_name].keys():
                    results.ensemble_results_by_param[model_common_name][(m,s)] = {"noiseful":{},"noiseless":{},"mixed":{}}

               # Store noiseful ensemble results
               fdict_noiseful  = get_ensemble_flow_results(iloader, top_model_noiseful,  cnn_mode=cnn_mode, model_type=model_type,more_eval_info=more_eval_info,results=results,verbose=False,psl=psl)
               results.ensemble_results_by_param[model_common_name][(m,s)]['noiseful']['top_model_key'] = top_model_key_noiseful
               results.ensemble_results_by_param[model_common_name][(m,s)]['noiseful'].update(fdict_noiseful)            

            #    # Store noiseless ensemble results
            #    fdict_noiseless = get_ensemble_flow_results(iloader, top_model_noiseless, cnn_mode=cnn_mode, model_type=model_type,more_eval_info=more_eval_info,results=results,verbose=False,psl=psl)
            #    results.ensemble_results_by_param[model_common_name][(m,s)]['noiseless']['top_model_key'] = top_model_key_noiseless
            #    results.ensemble_results_by_param[model_common_name][(m,s)]['noiseless'].update(fdict_noiseless)            
               
            #    # Store mixed ensemble results
            #    fdict_mixed     = get_ensemble_flow_results(iloader, top_model_mixed,     cnn_mode=cnn_mode, model_type=model_type,more_eval_info=more_eval_info,results=results,verbose=False,psl=psl)
            #    results.ensemble_results_by_param[model_common_name][(m,s)]['mixed']['top_model_key']     = top_model_key_mixed
            #    results.ensemble_results_by_param[model_common_name][(m,s)]['mixed'].update(fdict_mixed)            



## CLASSIFLOWS

 
### utils

 
#### classiflow_forward
def classiflow_forward(model,x_cnn, x_aux,y_main,y_neve,cnn_mode):
    #is_signal  = (y_neve.squeeze() > 0)
    is_signal = (y_neve.squeeze() >= 1).any(dim=1)

    # Forward pass
    if cnn_mode  : cond, nclass   = model(x_cnn)  
    else         : cond, nclass   = model(x_cnn, x_aux)  

    if is_signal.any():    
        log_pdf,_,_  = model.pdf(y_main[is_signal], conditional_input=cond[is_signal])
        return log_pdf, nclass

####  get_classiflows_results
def get_classiflows_results(model,dataloader,cnn_mode,reruns_max=5,more_eval_info=False,results=None,verbose=True,psl=False,samplesize=100):
    if more_eval_info == True: assert results is not None
    for attempt in range(1, reruns_max + 1):
        try:
            
            fdict = test_loop_classiflows(model,
                                dataloader,
                                device=device,
                                verbose=verbose,
                                samplesize=samplesize,
                                plot=False,
                                cnn_mode=cnn_mode,
                                more_eval_info=more_eval_info,
                                results=results,
                                psl=psl)
                    
            break
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            traceback.print_exc()

            if attempt == reruns_max: 
                raise RuntimeError("Maximum retry attempts exceeded")       
    return fdict

#### test_loop_classiflows
def test_loop_classiflows(modelcf,  val_loader, device,verbose=False,samplesize=100,flow_eval=True,cnn_mode=False,plot=True,more_eval_info=False,results=None,psl=False):
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    modelcf.eval()
    

    fdict = {}
    
    lowest_ent       = np.inf
    lpdf             = torch.tensor([]).to(device)
    ent              = torch.tensor([]).to(device)
    ent_full         = torch.tensor([]).to(device)    
    signif           = {"PT":torch.tensor([]),"MET":torch.tensor([]),"ETA":torch.tensor([])}
    neve             = {"PT":torch.tensor([]),"MET":torch.tensor([]),"ETA":torch.tensor([])} 
    neve_sg          = {"PT":torch.tensor([]),"MET":torch.tensor([]),"ETA":torch.tensor([])} 
    class_loss_dat   = torch.tensor([]).to(device)
    c_truth          = []               
    c_pred           = []               

    nsig = 0
    data_iter = tqdm(val_loader, desc="Validating", leave=False) if verbose else val_loader
    with torch.no_grad():

        for x_cnn, x_aux, y_main, y_neve in data_iter:

            # Each of these is a dict of tensors
            x_cnn      = x_cnn.to(device).double()
            x_aux      = x_aux.to(device).double()
            y_main     = y_main.to(device).double()
            y_neve     = y_neve.to(device).double()

            #is_signal = (y_neve.squeeze() > 0).any(dim=1)
            is_signal = (y_neve.squeeze() >= 1).any(dim=1)
            nsig      += sum(is_signal)

            c_truth.extend(is_signal)
            if psl:
                min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
                max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
                range_vals  = max_vals - min_vals
                # Avoid division where range is 0
                normalized_x_cnn = torch.where(
                    range_vals > 0,
                    (x_cnn - min_vals) / range_vals,
                    torch.zeros_like(x_cnn)
                )
                x_cnn            = normalized_x_cnn
            
            # Forward pass
            if cnn_mode:
                cond,nclass   = modelcf(x_cnn)  
            else:
                cond,nclass   = modelcf(x_cnn, x_aux)  
            
            c_pred.extend(nclass)
            # flow_weight  = 0
            # log_loss = torch.tensor(0.0, device=device)
            class_loss         = criterion(nclass.squeeze(), is_signal.double())
            class_loss_dat     = torch.hstack([class_loss,class_loss_dat])

            if is_signal.any():    
                log_pdf,_,_  = modelcf.pdf(y_main[is_signal], conditional_input=cond[is_signal])
                lpdf         = torch.hstack([lpdf,log_pdf])
            
            
                
            if flow_eval:    
                entropy_full_dict  = modelcf.pdf.entropy(conditional_input=cond,samplesize=samplesize)
                ent                = torch.hstack([ent,entropy_full_dict["total"][is_signal]])
                ent_full           = torch.hstack([ent_full,entropy_full_dict["total"]])

                if more_eval_info and is_signal.any():
                    x_aux_dict      =  {i:j for i,j in zip(results.datasets['test_dataset.pt'].other_inputs.keys(),x_aux[is_signal].T.cpu())}
                    y_neve_dict     =  {i:j for i,j in zip(results.datasets['test_dataset.pt'].mlp_outputs.keys(),y_neve[is_signal].T.cpu())}
                    unnormed_x_aux  = results.datasets['test_dataset.pt'].unnormalize_batch('other_inputs',x_aux_dict)
                    unnormed_y_neve = results.datasets['test_dataset.pt'].unnormalize_batch('mlp_outputs',y_neve_dict)

                    nevents_PT       = unnormed_x_aux['PT_NEVENTS']
                    nevents_MET      = unnormed_x_aux['MET_NEVENTS']
                    nevents_ETA      = unnormed_x_aux['ETA_NEVENTS']
                    nevents_PT_sg    = unnormed_y_neve['PT_NEVENTS_SIG']
                    nevents_MET_sg   = unnormed_y_neve['MET_NEVENTS_SIG']
                    nevents_ETA_sg   = unnormed_y_neve['ETA_NEVENTS_SIG']
                    nevents_PT_bg    = nevents_PT - nevents_PT_sg
                    nevents_MET_bg   = nevents_MET - nevents_MET_sg
                    nevents_ETA_bg   = nevents_ETA - nevents_ETA_sg
                    significance_PT  = nevents_PT_sg / torch.sqrt(nevents_PT_bg)
                    significance_MET = nevents_MET_sg / torch.sqrt(nevents_MET_bg)
                    significance_ETA = nevents_ETA_sg / torch.sqrt(nevents_ETA_bg)
                    signif['PT']     = torch.hstack([signif['PT'],significance_PT])
                    signif['MET']    = torch.hstack([signif['MET'],significance_MET])
                    signif['ETA']    = torch.hstack([signif['ETA'],significance_ETA])
                    
                    neve['PT']     = torch.hstack([neve['PT'],nevents_PT])
                    neve['MET']    = torch.hstack([neve['MET'],nevents_MET])
                    neve['ETA']    = torch.hstack([neve['ETA'],nevents_ETA])

                    neve_sg['PT']     = torch.hstack([neve_sg['PT'],nevents_PT_sg])
                    neve_sg['MET']    = torch.hstack([neve_sg['MET'],nevents_MET_sg])
                    neve_sg['ETA']    = torch.hstack([neve_sg['ETA'],nevents_ETA_sg])
                    

                    if entropy_full_dict["total"][is_signal].min() < lowest_ent:
                        sorted_indices              = torch.argsort(entropy_full_dict["total"][is_signal])
                        lowest_ent                  = entropy_full_dict["total"][is_signal][sorted_indices][0]
                        fdict['bestsr_x_cnn']       = x_cnn[is_signal][sorted_indices][0].cpu()
                        fdict['bestsr_x_aux']       = x_aux[is_signal][sorted_indices][0].cpu()
                        fdict['bestsr_y_main']      = y_main[is_signal][sorted_indices][0].cpu()
                        fdict['bestsr_y_neve']      = y_neve[is_signal][sorted_indices][0].cpu()
                        fdict['bestsr_signif_PT']   = significance_PT[sorted_indices.cpu()][0]
                        fdict['bestsr_signif_MET']  = significance_MET[sorted_indices.cpu()][0]
                        fdict['bestsr_signif_ETA']  = significance_ETA[sorted_indices.cpu()][0]
                        fdict["bestsr_nlpdf"]       = (-log_pdf)[sorted_indices.cpu()][0]
                        fdict["bestsr_ent"]         = lowest_ent
             

    fdict['test_log_loss']   = ((-lpdf).sum() / nsig).cpu().item()
    fdict['nlpdf']           = -lpdf
    fdict['ent']             = ent
    fdict['ent_full']        = ent_full
    fdict["c_truth"]         = torch.tensor(c_truth).float()
    fdict["c_pred"]          = torch.tensor(c_pred).float()

    if all(c_truth) or not any(c_truth):
                auc         = 0
                fpr         = []
                tpr         = []
                thresholds  = []
    else:
        auc                  = roc_auc_score(fdict["c_truth"], fdict["c_pred"])
        fpr, tpr, thresholds = roc_curve(fdict["c_truth"], fdict["c_pred"])
    fdict['test_class_loss']  = class_loss_dat.mean().cpu().item()
    fdict['class_loss_dat']   = class_loss_dat

    fdict['test_total_loss']  = (fdict['class_loss_dat'].mean()+fdict['nlpdf'].mean()).cpu().item()
    

    fdict['auc_c']              = auc                                  
    fdict['fpr_c']              = fpr                  
    fdict['tpr_c']              = tpr                  
    fdict['thresholds_c']       = thresholds          

    auc, fpr, tpr, thresholds, e_logits = get_evaluation_metrics(fdict['ent_full'].cpu(), fdict["c_truth"].cpu())
    fdict['auc_e']             = auc
    fdict['fpr_e']             = fpr
    fdict['tpr_e']             = tpr
    fdict['thresholds_e']      = thresholds
    fdict['e_logits_e']        = e_logits
    
    if flow_eval:    
        tail_percentile            = 95
        intensity_percentile       = 95
        ediff                      = (-lpdf-ent).cpu().numpy()
        fdict['ediff_mean']        = ediff.mean()
        fdict['ediff_std']         = ediff.std()
        fdict['tail_thresh']       = np.percentile(ediff, tail_percentile)

        tail_prop                  = np.sum(ediff > fdict['tail_thresh']) / len(ediff)
        absediff                   = np.abs(ediff)
        T                          = np.percentile(absediff, intensity_percentile)
        excess                     = np.sum([x - T for x in absediff if x > T])
        fdict['outlier_intensity'] = excess / np.sum(absediff) if np.sum(absediff) > 0 else 0
        
        if verbose:
            print(f"Validation Log Loss  = {fdict['test_log_loss']    :.4f}")
            print(f"ediff.mean()         = {fdict['ediff_mean']       :.4f}")
            print(f"ediff.std()          = {fdict['ediff_std']        :.4f}")
            print(f"tail_thresh          = {fdict['tail_thresh']      :.4f}")
            print(f"outlier_intensity    = {fdict['outlier_intensity']:.4f}")
        if plot:
            plt.hist2d(-lpdf.cpu(),ent.cpu(),bins=50,norm=LogNorm())
            plt.xlabel("loss")
            plt.ylabel("ent")
            plt.show()
        if more_eval_info:
            fdict['signif']          = signif  
    
    return fdict
 
#### get_ensemble_classiflow_results
def get_ensemble_classiflow_results(dataloader,model_list,cnn_mode,model_type,verbose=True,more_eval_info=False,results=None,reruns_max=5,psl=False):
    if more_eval_info == True: assert results is not None
    for attempt in range(1, reruns_max + 1):
        try:
            fdict = test_loop_ensemble_classiflows(dataloader, 
                                          model_list, 
                                          cnn_mode, 
                                          model_type, 
                                          verbose=verbose, 
                                          more_eval_info=more_eval_info, 
                                          results=results,
                                          psl=psl)                    
            break
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            traceback.print_exc()
            if attempt == reruns_max: 
                raise RuntimeError("Maximum retry attempts exceeded")       
    return fdict
 
#### test_loop_ensemble_classiflows
def test_loop_ensemble_classiflows(dataloader,model_list,cnn_mode,model_type,verbose=True,more_eval_info=False,results=None,psl=False):
    
 
    edat_ensemble      = [] # Signal only 
    edat_ensemble_full = [] # Both Sg and Bg
    ldat_ensemble      = [] # Loss [only with signal] 
    edat_i_full        = []              
    edat_i             = []          
    ldat_i             = []          
    
    c_truth = []
    c_pred     = []
    c_pred_i   = []
    #signif             = torch.tensor([])
    signif           = {"PT":torch.tensor([]),"MET":torch.tensor([]),"ETA":torch.tensor([])}
    neve           = {"PT":torch.tensor([]),"MET":torch.tensor([]),"ETA":torch.tensor([])}
    neve_sg           = {"PT":torch.tensor([]),"MET":torch.tensor([]),"ETA":torch.tensor([])}

    
    lowest_ent    = np.inf
     

    data_iter = tqdm(dataloader, desc="Validating", leave=True) if verbose else dataloader
    fdict = {}
    nsig = 0 
    with torch.no_grad():
        for i in model_list: i.eval()

        for x_cnn, x_aux, y_main, y_neve in data_iter:
            
            
            x_cnn      = x_cnn.to(device).double()
            x_aux      = x_aux.to(device).double()
            y_main     = y_main.to(device).double()
            y_neve     = y_neve.to(device).double()
            #is_signal  = (y_neve.squeeze() > 0)
            is_signal = (y_neve.squeeze() >= 1).any(dim=1)
            nsig += sum(is_signal)
            c_truth.extend(is_signal)
            if psl:
                min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
                max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
                range_vals  = max_vals - min_vals
                # Avoid division where range is 0
                normalized_x_cnn = torch.where(
                    range_vals > 0,
                    (x_cnn - min_vals) / range_vals,
                    torch.zeros_like(x_cnn)
                )
                x_cnn            = normalized_x_cnn
            # Ensemble Eval
            try:
                nc, nc_i = get_ensemble_class(model_list,x_cnn,x_aux,y_neve,device,cnn_mode=cnn_mode, model_type=model_type)
                e        = evaluate_ensemble_entropy_faster(model_list,x_cnn,x_aux, 100, device,model_type=model_type,cnn_mode=cnn_mode)  
                l        = evaluate_ensemble_loss(model_list,x_cnn,x_aux,y_main,y_neve,device,model_type=model_type,cnn_mode=cnn_mode)     
                
                e_i_full = torch.stack([evaluate_entropy(model, x_cnn,x_aux,is_signal,samplesize=100, full=True, model_type=model_type,cnn_mode=cnn_mode) for model in model_list])
                e_i      = torch.stack([en[is_signal] for en in e_i_full])
                l_i = []
                for model in model_list:
                    lpdf_i, nclass = classiflow_forward(model,x_cnn,x_aux,y_main,y_neve,cnn_mode=cnn_mode)
                    l_i.append(-lpdf_i)
                l_i = torch.stack(l_i)
             
            except Exception as e :
                print("Fucked (Ensemble) :",str(e))
                traceback.print_exc()
                
                nc, nc_i = get_ensemble_class(model_list,x_cnn,x_aux,y_neve,device,cnn_mode=cnn_mode, model_type=model_type)
                e        = evaluate_ensemble_entropy_faster(model_list,x_cnn,x_aux, 100, device,model_type=model_type,cnn_mode=cnn_mode)  
                l        = evaluate_ensemble_loss(model_list,x_cnn,x_aux,y_main,y_neve,device,model_type=model_type,cnn_mode=cnn_mode)     
                
                e_i_full = torch.stack([evaluate_entropy(model, x_cnn,x_aux,is_signal,samplesize=100, full=True, model_type=model_type,cnn_mode=cnn_mode) for model in model_list])
                e_i      = torch.stack([en[is_signal] for en in e_i_full])
                l_i = []
                for model in model_list:
                    lpdf_i, nclass = classiflow_forward(model,x_cnn,x_aux,y_main,y_neve,cnn_mode=cnn_mode)
                    l_i.append(-lpdf_i)
                l_i = torch.stack(l_i)
            
            c_pred.extend(nc)
            c_pred_i.append(nc_i)
            edat_ensemble_full.append(e.cpu())
            edat_i_full.append(e_i_full)

            if len(e[is_signal]) != 0  : 
                edat_ensemble.append(e[is_signal].cpu())
                edat_i.append(e_i)
            if len(l) != 0             : 
                ldat_ensemble.append((-l).cpu())
                ldat_i.append(l_i)

            if more_eval_info:
                x_aux_dict      =  {i:j for i,j in zip(results.datasets['test_dataset.pt'].other_inputs.keys(),x_aux[is_signal].T.cpu())}
                y_neve_dict     =  {i:j for i,j in zip(results.datasets['test_dataset.pt'].mlp_outputs.keys(),y_neve[is_signal].T.cpu())}
                unnormed_x_aux  = results.datasets['test_dataset.pt'].unnormalize_batch('other_inputs',x_aux_dict)
                unnormed_y_neve = results.datasets['test_dataset.pt'].unnormalize_batch('mlp_outputs',y_neve_dict)

                # nevents         = unnormed_x_aux['nevents']
                # nevents_sg      = unnormed_y_neve['nevents_sg']
                # nevents_bg      = nevents - nevents_sg
                # significance    = nevents_sg / torch.sqrt(nevents_bg)
                # signif          = torch.hstack([signif,significance])
                nevents_PT       = unnormed_x_aux['PT_NEVENTS']
                nevents_MET      = unnormed_x_aux['MET_NEVENTS']
                nevents_ETA      = unnormed_x_aux['ETA_NEVENTS']
                nevents_PT_sg    = unnormed_y_neve['PT_NEVENTS_SIG']
                nevents_MET_sg   = unnormed_y_neve['MET_NEVENTS_SIG']
                nevents_ETA_sg   = unnormed_y_neve['ETA_NEVENTS_SIG']
                nevents_PT_bg    = nevents_PT - nevents_PT_sg
                nevents_MET_bg   = nevents_MET - nevents_MET_sg
                nevents_ETA_bg   = nevents_ETA - nevents_ETA_sg
                if Z_SIG:
                    significance_PT  = z_sig(nevents_PT_sg, nevents_PT_bg)
                    significance_MET = z_sig(nevents_MET_sg, nevents_MET_bg)
                    significance_ETA = z_sig(nevents_ETA_sg, nevents_ETA_bg)
                else:
                    significance_PT  = nevents_PT_sg / torch.sqrt(nevents_PT_bg)
                    significance_MET = nevents_MET_sg / torch.sqrt(nevents_MET_bg)
                    significance_ETA = nevents_ETA_sg / torch.sqrt(nevents_ETA_bg)
                signif['PT']     = torch.hstack([signif['PT'],significance_PT])
                signif['MET']    = torch.hstack([signif['MET'],significance_MET])
                signif['ETA']    = torch.hstack([signif['ETA'],significance_ETA])
                    
                neve['PT']     = torch.hstack([neve['PT'],nevents_PT])
                neve['MET']    = torch.hstack([neve['MET'],nevents_MET])
                neve['ETA']    = torch.hstack([neve['ETA'],nevents_ETA])

                neve_sg['PT']     = torch.hstack([neve_sg['PT'],nevents_PT_sg])
                neve_sg['MET']    = torch.hstack([neve_sg['MET'],nevents_MET_sg])
                neve_sg['ETA']    = torch.hstack([neve_sg['ETA'],nevents_ETA_sg])

                if e.min() < lowest_ent: 
                    sorted_indices = torch.argsort(e[is_signal])
                    lowest_ent = e.min()
                    fdict["bestsr_x_cnn"]   = x_cnn[is_signal][sorted_indices][0]
                    fdict["bestsr_x_aux"]   = x_aux[is_signal][sorted_indices][0]
                    fdict["bestsr_y_main"]  = y_main[is_signal][sorted_indices][0]
                    fdict["bestsr_y_neve"]  = y_neve[is_signal][sorted_indices][0]
                    #fdict["bestsr_signif"]  = significance[sorted_indices.cpu()][0]
                    fdict['bestsr_signif_PT']   = significance_PT[sorted_indices.cpu()][0]
                    fdict['bestsr_signif_MET']  = significance_MET[sorted_indices.cpu()][0]
                    fdict['bestsr_signif_ETA']  = significance_ETA[sorted_indices.cpu()][0]
                    fdict["bestsr_nlpdf"]   = (-l)[sorted_indices.cpu()][0]
                    fdict["bestsr_ent"]     = lowest_ent

        fdict["c_truth"]         =  torch.tensor(c_truth).float()
        fdict["c_pred"]            = torch.tensor(c_pred)
        fdict["c_pred_i"]          = torch.hstack(c_pred_i)
        fdict["test_class_loss"]   = nn.BCEWithLogitsLoss()(fdict["c_pred"] ,fdict["c_truth"])
        fdict["test_class_loss_i"] = [nn.BCEWithLogitsLoss()(i,fdict["c_truth"].to(device)).cpu() for i in  fdict["c_pred_i"]]
        
        if all(c_truth) or not any(c_truth):
            fdict["auc_c"]         = 0
            fdict["fpr_c"]         = []
            fdict["tpr_c"]         = []
            fdict["thresholds_c"]  = []
        else:
            fdict["auc_c"]               = roc_auc_score(fdict["c_truth"], fdict["c_pred"])
            fpr, tpr, thresholds       = roc_curve(fdict["c_truth"], fdict["c_pred"])
            fdict["fpr_c"]               = fpr
            fdict["tpr_c"]               = tpr
            fdict["thresholds_c"]        = thresholds
            
            fdict["auc_c_i"]             = [roc_auc_score(fdict["c_truth"].cpu(), i.cpu()) for i in fdict["c_pred_i"]]
            eval_dat                   = [roc_curve(fdict["c_truth"].cpu(), i.cpu()) for i in fdict["c_pred_i"]]
            fpr_i, tpr_i, thresholds_i = list(zip(*eval_dat))
            fdict["fpr_c_i"]             = fpr_i
            fdict["tpr_c_i"]             = tpr_i
            fdict["thresholds_c_i"]      = thresholds_i
            

        fdict["ent"]             = torch.cat(edat_ensemble)
        fdict["ent_full"]        = torch.cat(edat_ensemble_full)
        fdict["nlpdf"]           = torch.cat(ldat_ensemble)
        fdict['test_log_loss']   = (fdict["nlpdf"].sum() / nsig).cpu().item()

        fdict["ent_i"]           = torch.hstack(edat_i)
        fdict["ent_i_full"]      = torch.hstack(edat_i_full)
        fdict["nlpdf_i"]         = torch.hstack(ldat_i)
        fdict['test_log_loss_i'] = [(i.sum() / nsig).cpu().item() for i in fdict["nlpdf_i"]]

        if more_eval_info:
            fdict['signif']          = signif  
            fdict['neve']          = neve  
            fdict['neve_sg']          = neve_sg  
        
        auc, fpr, tpr, thresholds, e_logits           = get_evaluation_metrics(fdict['ent_full'].cpu(), fdict["c_truth"].float().cpu())
        eval_dat                                      = [get_evaluation_metrics(e.cpu(), fdict["c_truth"].float().cpu()) for e in fdict["ent_i_full"]]
        auc_i, fpr_i, tpr_i, thresholds_i, e_logits_i = list(zip(*eval_dat))
        
        fdict['auc_e']              = auc               
        fdict['fpr_e']              = fpr               
        fdict['tpr_e']              = tpr               
        fdict['thresholds_e']       = thresholds                   
        fdict['e_logits_e']         = e_logits                   
        
        fdict['auc_e_i']         = auc_i                   
        fdict['fpr_e_i']         = fpr_i                   
        fdict['tpr_e_i']         = tpr_i                   
        fdict['thresholds_e_i']  = thresholds_i                           
        fdict['e_logits_e_i']    = e_logits_i                       
        
    return fdict

# #### test_loop_ensemble_classiflows
# def test_loop_ensemble_classiflows(dataloader,model_list,cnn_mode,model_type,verbose=True,more_eval_info=False,results=None,psl=False):
    
 
#     edat_ensemble      = [] # Signal only 
#     edat_ensemble_full = [] # Both Sg and Bg
#     ldat_ensemble      = [] # Loss [only with signal] 
#     edat_i_full        = []              
#     edat_i             = []          
#     ldat_i             = []          
    
#     c_truth = []
#     c_pred     = []
#     c_pred_i   = []
#     signif             = torch.tensor([])

    
#     lowest_ent    = np.inf
     

#     data_iter = tqdm(dataloader, desc="Validating", leave=True) if verbose else dataloader
#     fdict = {}
#     nsig = 0 
#     with torch.no_grad():
#         for i in model_list: i.eval()

#         for x_cnn, x_aux, y_main, y_neve in data_iter:
            
            
#             x_cnn      = x_cnn.to(device).double()
#             x_aux      = x_aux.to(device).double()
#             y_main     = y_main.to(device).double()
#             y_neve     = y_neve.to(device).double()
#             #is_signal  = (y_neve.squeeze() > 0)
#             is_signal = (y_neve.squeeze() >= 1).any(dim=1)
#             nsig += sum(is_signal)
#             c_truth.extend(is_signal)
#             if psl:
#                 min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
#                 max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
#                 range_vals  = max_vals - min_vals
#                 # Avoid division where range is 0
#                 normalized_x_cnn = torch.where(
#                     range_vals > 0,
#                     (x_cnn - min_vals) / range_vals,
#                     torch.zeros_like(x_cnn)
#                 )
#                 x_cnn            = normalized_x_cnn
#             # Ensemble Eval
#             try:
#                 nc, nc_i = get_ensemble_class(model_list,x_cnn,x_aux,y_neve,device,cnn_mode=cnn_mode, model_type=model_type)
#                 e        = evaluate_ensemble_entropy_faster(model_list,x_cnn,x_aux, 100, device,model_type=model_type,cnn_mode=cnn_mode)  
#                 l        = evaluate_ensemble_loss(model_list,x_cnn,x_aux,y_main,y_neve,device,model_type=model_type,cnn_mode=cnn_mode)     
                
#                 e_i_full = torch.stack([evaluate_entropy(model, x_cnn,x_aux,is_signal,samplesize=100, full=True, model_type=model_type,cnn_mode=cnn_mode) for model in model_list])
#                 e_i      = torch.stack([en[is_signal] for en in e_i_full])
#                 l_i = []
#                 for model in model_list:
#                     lpdf_i, nclass = classiflow_forward(model,x_cnn,x_aux,y_main,y_neve,cnn_mode=cnn_mode)
#                     l_i.append(-lpdf_i)
#                 l_i = torch.stack(l_i)
             
#             except Exception as e :
#                 print("Fucked (Ensemble) :",str(e))
#                 traceback.print_exc()
                
#                 nc, nc_i = get_ensemble_class(model_list,x_cnn,x_aux,y_neve,device,cnn_mode=cnn_mode, model_type=model_type)
#                 e        = evaluate_ensemble_entropy_faster(model_list,x_cnn,x_aux, 100, device,model_type=model_type,cnn_mode=cnn_mode)  
#                 l        = evaluate_ensemble_loss(model_list,x_cnn,x_aux,y_main,y_neve,device,model_type=model_type,cnn_mode=cnn_mode)     
                
#                 e_i_full = torch.stack([evaluate_entropy(model, x_cnn,x_aux,is_signal,samplesize=100, full=True, model_type=model_type,cnn_mode=cnn_mode) for model in model_list])
#                 e_i      = torch.stack([en[is_signal] for en in e_i_full])
#                 l_i = []
#                 for model in model_list:
#                     lpdf_i, nclass = classiflow_forward(model,x_cnn,x_aux,y_main,y_neve,cnn_mode=cnn_mode)
#                     l_i.append(-lpdf_i)
#                 l_i = torch.stack(l_i)
            
#             c_pred.extend(nc)
#             c_pred_i.append(nc_i)
#             edat_ensemble_full.append(e.cpu())
#             edat_i_full.append(e_i_full)

#             if len(e[is_signal]) != 0  : 
#                 edat_ensemble.append(e[is_signal].cpu())
#                 edat_i.append(e_i)
#             if len(l) != 0             : 
#                 ldat_ensemble.append((-l).cpu())
#                 ldat_i.append(l_i)

#             if more_eval_info:
#                 x_aux_dict      =  {i:j for i,j in zip(results.datasets['test_dataset.pt'].other_inputs.keys(),x_aux[is_signal].T.cpu())}
#                 y_neve_dict     =  {i:j for i,j in zip(results.datasets['test_dataset.pt'].mlp_outputs.keys(),y_neve[is_signal].T.cpu())}
#                 unnormed_x_aux  = results.datasets['test_dataset.pt'].unnormalize_batch('other_inputs',x_aux_dict)
#                 unnormed_y_neve = results.datasets['test_dataset.pt'].unnormalize_batch('mlp_outputs',y_neve_dict)

#                 nevents         = unnormed_x_aux['nevents']
#                 nevents_sg      = unnormed_y_neve['nevents_sg']
#                 nevents_bg      = nevents - nevents_sg
#                 significance    = nevents_sg / torch.sqrt(nevents_bg)
#                 signif          = torch.hstack([signif,significance])

#                 if e.min() < lowest_ent: 
#                     sorted_indices = torch.argsort(e[is_signal])
#                     lowest_ent = e.min()
#                     fdict["bestsr_x_cnn"]   = x_cnn[is_signal][sorted_indices][0]
#                     fdict["bestsr_x_aux"]   = x_aux[is_signal][sorted_indices][0]
#                     fdict["bestsr_y_main"]  = y_main[is_signal][sorted_indices][0]
#                     fdict["bestsr_y_neve"]  = y_neve[is_signal][sorted_indices][0]
#                     fdict["bestsr_signif"]  = significance[sorted_indices.cpu()][0]
#                     fdict["bestsr_nlpdf"]   = (-l)[sorted_indices.cpu()][0]
#                     fdict["bestsr_ent"]     = lowest_ent

#         fdict["c_truth"]         =  torch.tensor(c_truth).float()
#         fdict["c_pred"]            = torch.tensor(c_pred)
#         fdict["c_pred_i"]          = torch.hstack(c_pred_i)
#         fdict["test_class_loss"]   = nn.BCEWithLogitsLoss()(fdict["c_pred"] ,fdict["c_truth"])
#         fdict["test_class_loss_i"] = [nn.BCEWithLogitsLoss()(i,fdict["c_truth"].to(device)).cpu() for i in  fdict["c_pred_i"]]
        
#         if all(c_truth) or not any(c_truth):
#             fdict["auc_c"]         = 0
#             fdict["fpr_c"]         = []
#             fdict["tpr_c"]         = []
#             fdict["thresholds_c"]  = []
#         else:
#             fdict["auc_c"]               = roc_auc_score(fdict["c_truth"], fdict["c_pred"])
#             fpr, tpr, thresholds       = roc_curve(fdict["c_truth"], fdict["c_pred"])
#             fdict["fpr_c"]               = fpr
#             fdict["tpr_c"]               = tpr
#             fdict["thresholds_c"]        = thresholds
            
#             fdict["auc_c_i"]             = [roc_auc_score(fdict["c_truth"].cpu(), i.cpu()) for i in fdict["c_pred_i"]]
#             eval_dat                   = [roc_curve(fdict["c_truth"].cpu(), i.cpu()) for i in fdict["c_pred_i"]]
#             fpr_i, tpr_i, thresholds_i = list(zip(*eval_dat))
#             fdict["fpr_c_i"]             = fpr_i
#             fdict["tpr_c_i"]             = tpr_i
#             fdict["thresholds_c_i"]      = thresholds_i
            

#         fdict["ent"]             = torch.cat(edat_ensemble)
#         fdict["ent_full"]        = torch.cat(edat_ensemble_full)
#         fdict["nlpdf"]           = torch.cat(ldat_ensemble)
#         fdict['test_log_loss']   = (fdict["nlpdf"].sum() / nsig).cpu().item()

#         fdict["ent_i"]           = torch.hstack(edat_i)
#         fdict["ent_i_full"]      = torch.hstack(edat_i_full)
#         fdict["nlpdf_i"]         = torch.hstack(ldat_i)
#         fdict['test_log_loss_i'] = [(i.sum() / nsig).cpu().item() for i in fdict["nlpdf_i"]]

#         fdict["signif"]                               = signif    
#         auc, fpr, tpr, thresholds, e_logits           = get_evaluation_metrics(fdict['ent_full'].cpu(), fdict["c_truth"].float().cpu())
#         eval_dat                                      = [get_evaluation_metrics(e.cpu(), fdict["c_truth"].float().cpu()) for e in fdict["ent_i_full"]]
#         auc_i, fpr_i, tpr_i, thresholds_i, e_logits_i = list(zip(*eval_dat))
        
#         fdict['auc_e']              = auc               
#         fdict['fpr_e']              = fpr               
#         fdict['tpr_e']              = tpr               
#         fdict['thresholds_e']       = thresholds                   
#         fdict['e_logits_e']         = e_logits                   
        
#         fdict['auc_e_i']         = auc_i                   
#         fdict['fpr_e_i']         = fpr_i                   
#         fdict['tpr_e_i']         = tpr_i                   
#         fdict['thresholds_e_i']  = thresholds_i                           
#         fdict['e_logits_e_i']    = e_logits_i                       
        
#     return fdict

### eval_classiflows
def eval_classiflows(results,first_run=True,reruns_max=5,more_eval_info=False,psl=False,skip_noiseless=True):
        
    for model_name in sorted(results.models.keys()):
        
        print(f"Evaluating ClassiFlow of {model_name}...\n")
        if results.model_type[model_name] != "cf": 
            print(f"{model_name} ({results.model_type[model_name]}): Not ClassiFlow continue.")
            continue

        if skip_noiseless and "noiseless" in model_name:
            print(f"{model_name} Skipping noiseless turned on.")
            continue

        # Initialize results dictionary if needed
        if model_name not in results.results.keys(): 
            results.results[model_name] = {}

        # If the results dictionary exist, check whether the results exist in them
        elif 'test_log_loss' not in results.results[model_name].keys():
            print(f"Not first run : test_log_loss of {model_name} are not done. Redo!")
            pass    

        # If the results exist, check whether rerun is required. If its first run, rerun. 
        elif not first_run: 
            print(f"Not first run : {model_name} already done")
            continue
      
        # Evaluate classification-only models
        fdict = get_classiflows_results(results.models[model_name].to(device).double(), results.test_loader, cnn_mode=results.cnn_mode[model_name],more_eval_info=more_eval_info,results=results,psl=psl)

        # Store results for classification-only models  
        results.results[model_name].update(fdict)

### eval_ensemble_classiflows
def eval_ensemble_classiflows(results,first_run=True,eval_type='test_log_loss',n_best=3,n_noiseful_in_mixed=2,n_noiseless_in_mixed=2,batch_size=2048,more_eval_info=False,psl=False ):
 
        if batch_size == 2048 : 
            pass
        else: 
            results.test_loader   = DataLoader(results.datasets['test_dataset.pt'] , batch_size=batch_size, shuffle=True)

         
        # Group models by common naming patterns for ensemble creation
        results.get_grouped_names()

        for model_common_name in results.grouped_names: 
            
            # Skip pure classification models (no flow component)
            if MODEL_TYPE_DICT[model_common_name.split("_")[0]] != 'cf' : continue

            
            all_models_names    = results.grouped_names[model_common_name]
            cnn_mode            = CNN_MODE_DICT[model_common_name.split("_")[0]]
            model_type          = MODEL_TYPE_DICT[model_common_name.split("_")[0]]

          
            print("Evaluating Ensemble ClassiFlow of :",model_common_name)

            # Initialize ensemble results storage if needed
            if model_common_name not in results.ensemble_results.keys(): 
                results.ensemble_results[model_common_name] = {}

            # Check if ensemble results already exist
            elif len(results.ensemble_results[model_common_name]) == 0:
                pass

            # If ensemble results exist, check if required keys are present
            elif any(['ent' not in i.keys() for _, i in results.ensemble_results[model_common_name].items()]):
                print(f"Not first run : edat_ensemble of {model_common_name} in {[k for k, i in results.ensemble_results[model_common_name].items() if 'edat_ensemble' not in i.keys() ]} are not done. Redo!")
                pass
            
            # If ensemble results exist and not a rerun. continue
            elif not first_run: 
                print(f"Not first run : {model_common_name} already done")
                continue
            
            top_model_key_noiseful, top_model_key_noiseless, top_model_key_mixed, top_model_noiseful, top_model_noiseless, top_model_mixed =  get_top_models(results,all_models_names,eval_type=eval_type,n_best=n_best,n_noiseful_in_mixed=n_noiseful_in_mixed,n_noiseless_in_mixed=n_noiseless_in_mixed)
              
            # Store ensemble results organized by noise configuration
            results.ensemble_results[model_common_name] = {"noiseful":{},"noiseless":{},"mixed":{}}

            # Store noiseful ensemble results
            fdict_noiseful  = get_ensemble_classiflow_results(results.test_loader, top_model_noiseful,  cnn_mode=cnn_mode, model_type=model_type,more_eval_info=more_eval_info,results=results,psl=psl)
            results.ensemble_results[model_common_name]['noiseful']['top_model_key'] = top_model_key_noiseful
            results.ensemble_results[model_common_name]['noiseful'].update(fdict_noiseful)            

            # # Store noiseless ensemble results
            # fdict_noiseless = get_ensemble_classiflow_results(results.test_loader, top_model_noiseless, cnn_mode=cnn_mode, model_type=model_type,more_eval_info=more_eval_info,results=results,psl=psl)
            # results.ensemble_results[model_common_name]['noiseless']['top_model_key'] = top_model_key_noiseless
            # results.ensemble_results[model_common_name]['noiseless'].update(fdict_noiseless)            
            
            # # Store mixed ensemble results
            # fdict_mixed     = get_ensemble_classiflow_results(results.test_loader, top_model_mixed,     cnn_mode=cnn_mode, model_type=model_type,more_eval_info=more_eval_info,results=results,psl=psl)
            # results.ensemble_results[model_common_name]['mixed']['top_model_key']     = top_model_key_mixed
            # results.ensemble_results[model_common_name]['mixed'].update(fdict_mixed)            

### eval_classiflows_by_point
def eval_classiflows_by_point(results,first_run=True,reruns_max=5,more_eval_info=True,psl=False):
    uniq_cond, iloader_dict = get_uniq_param(results)

    for model_name in sorted(results.models.keys()):
        
        if results.model_type[model_name] != "cf": 
            print(f"{model_name} ({results.model_type[model_name]}): Not ClassiFlow continue.")
            continue

        print(f"Evaluating ClassiFlow of {model_name}...\n")
        # Initialize results dictionary if needed
        if model_name not in results.results_by_param.keys(): 
            results.results_by_param[model_name] = {}
        
        elif len(results.results_by_param[model_name].keys()) != len(iloader_dict):
            print(f"Missing some param points : Exist {len(iloader_dict)}, have {len(results.results_by_param[model_name].keys())} instead. Rerun")
            pass  

        elif any(['test_log_loss' not in i.keys() for _, i in results.results_by_param[model_name].items()]):
            print(f"Missing some param points runs. Rerun")
            print([k for k, i in results.results_by_param[model_name].items() if 'test_log_loss' not in i.keys()])

        # If the results exist, check whether rerun is required. If its first run, rerun. 
        elif not first_run: 
            print(f"Not first run : {model_name} already done")
            continue

        for param,iloader in  tqdm(iloader_dict.items()):
            m,s = param
            if (m,s) not in results.results_by_param[model_name].keys():
                results.results_by_param[model_name][(m,s)] = {}      

            # Evaluate classification-only models
            #fdict = get_flows_results(results.models[model_name].to(device).double(), iloader, verbose=False,cnn_mode=results.cnn_mode[model_name],more_eval_info=more_eval_info,results=results)
            fdict = get_classiflows_results(results.models[model_name].to(device).double(), iloader, cnn_mode=results.cnn_mode[model_name],more_eval_info=more_eval_info,results=results,verbose=False,psl=psl)
        
            # Store results for classification-only models  
            results.results_by_param[model_name][(m,s)].update(fdict)

### eval_ensemble_classiflows_by_points
def eval_ensemble_classiflows_by_points(results,first_run=True,eval_type='test_log_loss',n_best=3,n_noiseful_in_mixed=2,n_noiseless_in_mixed=2,batch_size=2048,more_eval_info=True,psl=False):
     uniq_cond, iloader_dict = get_uniq_param(results)
    
     # Group models by common naming patterns for ensemble creation
     results.get_grouped_names()
     for model_common_name in results.grouped_names: 
          if MODEL_TYPE_DICT[model_common_name.split("_")[0]] != 'cf' : continue

          all_models_names    = results.grouped_names[model_common_name]
          cnn_mode            = CNN_MODE_DICT[model_common_name.split("_")[0]]
          model_type          = MODEL_TYPE_DICT[model_common_name.split("_")[0]]
          
          print("Evaluating Ensemble Flow (by_point) of :", model_common_name)
          
          # Initialize ensemble results storage if needed
          if model_common_name not in results.ensemble_results_by_param.keys(): 
               results.ensemble_results_by_param[model_common_name] = {}
          
          elif len(results.ensemble_results_by_param[model_common_name].keys()) != len(iloader_dict):
               print(f"Missing some param points : Exist {len(iloader_dict)}, have {len(results.ensemble_results_by_param[model_common_name].keys())} instead. Rerun")
               pass    

          elif any(['test_log_loss' not in i.keys() for _, i in results.ensemble_results_by_param[model_common_name].items()]):
               print(f"Missing some param points runs. Rerun")
               print([k for k, i in results.ensemble_results_by_param[model_common_name].items() if 'test_log_loss' not in i.keys()])

          elif not first_run: 
               print(f"Not first run : {model_common_name} already done")
               continue 
          
          top_model_key_noiseful, top_model_key_noiseless, top_model_key_mixed, top_model_noiseful, top_model_noiseless, top_model_mixed =  get_top_models(results,all_models_names,eval_type=eval_type,n_best=n_best,n_noiseful_in_mixed=n_noiseful_in_mixed,n_noiseless_in_mixed=n_noiseless_in_mixed)



          for param,iloader in  tqdm(iloader_dict.items()):

               
               m,s = param
               if (m,s) not in results.ensemble_results_by_param[model_common_name].keys():
                    results.ensemble_results_by_param[model_common_name][(m,s)] = {"noiseful":{},"noiseless":{},"mixed":{}}

               # Store noiseful ensemble results
               fdict_noiseful  = get_ensemble_classiflow_results(iloader, top_model_noiseful,  cnn_mode=cnn_mode, model_type=model_type,more_eval_info=more_eval_info,results=results,verbose=False,psl=psl)
               results.ensemble_results_by_param[model_common_name][(m,s)]['noiseful']['top_model_key'] = top_model_key_noiseful
               results.ensemble_results_by_param[model_common_name][(m,s)]['noiseful'].update(fdict_noiseful)            

            #    # Store noiseless ensemble results
            #    fdict_noiseless = get_ensemble_classiflow_results(iloader, top_model_noiseless, cnn_mode=cnn_mode, model_type=model_type,more_eval_info=more_eval_info,results=results,verbose=False,psl=psl)
            #    results.ensemble_results_by_param[model_common_name][(m,s)]['noiseless']['top_model_key'] = top_model_key_noiseless
            #    results.ensemble_results_by_param[model_common_name][(m,s)]['noiseless'].update(fdict_noiseless)            

            #    # Store mixed ensemble results
            #    fdict_mixed     = get_ensemble_classiflow_results(iloader, top_model_mixed,     cnn_mode=cnn_mode, model_type=model_type,more_eval_info=more_eval_info,results=results,verbose=False,psl=psl)
            #    results.ensemble_results_by_param[model_common_name][(m,s)]['mixed']['top_model_key']     = top_model_key_mixed
            #    results.ensemble_results_by_param[model_common_name][(m,s)]['mixed'].update(fdict_mixed)        




## Else

# In case of usage, to be delete if no usage found
def entropy_sb_check_cf(model, val_loader, device,verbose=False,samplesize=100,by_neve=True,cnn_mode =False,psl=False):
    model.eval()
    c_pred     = []             
    c_truth    = []             
    approx_e   = []                 
    e_truth    = []         
    data_iter = tqdm(val_loader, desc="Validating", leave=False) if verbose else val_loader
    with torch.no_grad():
        for x_cnn, x_aux, y_main, y_neve in data_iter:
            # Each of these is a dict of tensors
            x_cnn  = x_cnn.double().to(device)
            x_aux  = x_aux.double().to(device)
            y_main = y_main.double().to(device)
            y_neve = y_neve.double().to(device)
            if psl:
                min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
                max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
                range_vals  = max_vals - min_vals
                # Avoid division where range is 0
                normalized_x_cnn = torch.where(
                    range_vals > 0,
                    (x_cnn - min_vals) / range_vals,
                    torch.zeros_like(x_cnn)
                )
                x_cnn            = normalized_x_cnn
            if cnn_mode:
                cond,nclass = model(x_cnn)  # shape: [batch, cond_dim]
            else:
                cond,nclass = model(x_cnn, x_aux)  # shape: [batch, cond_dim]
            #is_signal = (y_neve.squeeze() > 0).any(dim=1).cpu().numpy()
            is_signal = (y_neve.squeeze() >= 1).any(dim=1).cpu().numpy()
            nclass_class = nn.Sigmoid()(nclass).cpu().squeeze().tolist()
            eval_dict = model.pdf.marginal_moments(conditional_input=cond, samplesize=samplesize,calc_kl_diff_and_entropic_quantities=True,verbose=False) 

            c_pred.extend(nclass_class)
            c_truth.extend(is_signal)
            approx_e.extend(eval_dict['approx_entropy_0'])
            e_truth.extend(eval_dict['entropy_0'])
   

    return c_pred, c_truth, approx_e, e_truth 

def entropy_sb_check2(model_f,model_c, val_loader, device,verbose=False,samplesize=100,by_neve=True,cnn_mode =False,psl=False):
    model_f.eval()  # shape: [batch, cond_dim]
    model_c.eval()  # shape: [batch, cond_dim]
            
    c_pred     = []             
    c_truth    = []             
    approx_e   = []                 
    e_truth    = []         
    data_iter = tqdm(val_loader, desc="Validating", leave=False) if verbose else val_loader
    with torch.no_grad():
        for x_cnn, x_aux, y_main, y_neve in data_iter:
            # Each of these is a dict of tensors
            x_cnn  = x_cnn.double().to(device)
            x_aux  = x_aux.double().to(device)
            y_main = y_main.double().to(device)
            y_neve = y_neve.double().to(device)
            if psl:
                min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
                max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
                range_vals  = max_vals - min_vals
                # Avoid division where range is 0
                normalized_x_cnn = torch.where(
                    range_vals > 0,
                    (x_cnn - min_vals) / range_vals,
                    torch.zeros_like(x_cnn)
                )
                x_cnn            = normalized_x_cnn
            if cnn_mode:
                cond = model_f(x_cnn)  # shape: [batch, cond_dim]
            else:
                cond = model_f(x_cnn, x_aux)  # shape: [batch, cond_dim]
            nclass = model_c(x_cnn, x_aux)  # shape: [batch, cond_dim]
            #is_signal = (y_neve.squeeze() > 0).any(dim=1).cpu().numpy()
            is_signal = (y_neve.squeeze() >= 1).any(dim=1).cpu().numpy()

            
            nclass_class = nn.Sigmoid()(nclass).cpu().squeeze().tolist()
            try:
                eval_dict = model_f.pdf.marginal_moments(conditional_input=cond, samplesize=samplesize,calc_kl_diff_and_entropic_quantities=True,verbose=False) 
                approx_e.extend(eval_dict['approx_entropy_0'])
                e_truth.extend(eval_dict['entropy_0'])
            except Exception as e:
                print("model_f.pdf.marginal_moments crashed!",str(e))
                entropy_dict  = modelf.pdf.entropy(conditional_input=cond[is_signal],samplesize=samplesize)
                approx_e.extend([])
                e_truth.extend(entropy_dict['total'])

            c_pred.extend(nclass_class)
            c_truth.extend(is_signal)
   

    return c_pred, c_truth, approx_e, e_truth 

def entropy_sb_check_f(modelf, val_loader, device,verbose=False,samplesize=100,cnn_mode =False,psl=False):
    modelf.eval()
    c_pred     = []             
    c_truth    = []             
    approx_e   = []                 
    e_truth    = []         
    data_iter = tqdm(val_loader, desc="Validating", leave=False) if verbose else val_loader
    with torch.no_grad():
        for x_cnn, x_aux, y_main, y_neve in data_iter:
            # Each of these is a dict of tensors
            x_cnn  = x_cnn.double().to(device)
            x_aux  = x_aux.double().to(device)
            y_main = y_main.double().to(device)
            y_neve = y_neve.double().to(device)
            if psl:
                min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
                max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
                range_vals  = max_vals - min_vals
                # Avoid division where range is 0
                normalized_x_cnn = torch.where(
                    range_vals > 0,
                    (x_cnn - min_vals) / range_vals,
                    torch.zeros_like(x_cnn)
                )
                x_cnn            = normalized_x_cnn

            if cnn_mode :
                cond = modelf(x_cnn)  # shape: [batch, cond_dim]
            else:
                cond = modelf(x_cnn, x_aux)  # shape: [batch, cond_dim]
            #is_signal = (y_neve.squeeze() > 0).any(dim=1).cpu().numpy()
            is_signal = (y_neve.squeeze() >= 1).any(dim=1).cpu().numpy()
            eval_dict = modelf.pdf.marginal_moments(conditional_input=cond, samplesize=samplesize,calc_kl_diff_and_entropic_quantities=True,verbose=False) 
            c_truth.extend(is_signal)
            approx_e.extend(eval_dict['approx_entropy_0'])
            e_truth.extend(eval_dict['entropy_0'])
   

    return c_truth, approx_e, e_truth 

def entropy_sb_check_ensemble_f(modellist, val_loader, device,verbose=False,samplesize=100,cnn_mode =False,psl=False):
    for model in modellist: model.eval()
    c_pred     = []             
    c_truth    = []             
    approx_e   = []                 
    e_truth    = []         
    data_iter = tqdm(val_loader, desc="Validating", leave=False) if verbose else val_loader
    with torch.no_grad():
        for x_cnn, x_aux, y_main, y_neve in data_iter:
            # Each of these is a dict of tensors
            x_cnn  = x_cnn.double().to(device)
            x_aux  = x_aux.double().to(device)
            y_main = y_main.double().to(device)
            y_neve = y_neve.double().to(device)
            #is_signal = (y_neve.squeeze() > 0).any(dim=1).cpu().numpy()
            is_signal = (y_neve.squeeze() >= 1).any(dim=1).cpu().numpy()
            if psl:
                min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
                max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
                range_vals  = max_vals - min_vals
                # Avoid division where range is 0
                normalized_x_cnn = torch.where(
                    range_vals > 0,
                    (x_cnn - min_vals) / range_vals,
                    torch.zeros_like(x_cnn)
                )
                x_cnn            = normalized_x_cnn
            batch_ents_app = []
            batch_ents_tru = []
            
            for model in modellist:
                if cnn_mode :
                    cond = model(x_cnn)  # shape: [batch, cond_dim]
                else:
                    cond = model(x_cnn, x_aux)  # shape: [batch, cond_dim]
                eval_dict = model.pdf.marginal_moments(conditional_input=cond, samplesize=samplesize,calc_kl_diff_and_entropic_quantities=True,verbose=False) 
                batch_ents_app.append(torch.tensor(eval_dict['approx_entropy_0']))
                batch_ents_tru.append(torch.tensor(eval_dict['entropy_0']))
            
            ent_ensemble_tru     = torch.stack(batch_ents_tru,dim=1)
            ent_mix_tru          = torch.logsumexp(ent_ensemble_tru, dim=1) - np.log(len(modellist))
            
            ent_ensemble_app     = torch.stack(batch_ents_app,dim=1)
            ent_mix_app          = torch.logsumexp(ent_ensemble_app, dim=1) - np.log(len(modellist))
            
            
            c_truth.extend(is_signal)
            approx_e.extend(ent_mix_app)
            e_truth.extend(ent_mix_tru)
            
            
    return c_truth, torch.tensor(approx_e), torch.tensor(e_truth )

def entropy_sb_check_ensemble(modellist, val_loader, device,verbose=False,samplesize=100,cnn_mode =False,model_type="f",psl=False):
    for model in modellist: model.eval()
    c_pred     = []             
    c_pred_ind = []             
    c_truth    = []             
    approx_e   = []                 
    entropy    = []         
    entropy_ind= []
    data_iter  = tqdm(val_loader, desc="Validating", leave=False) if verbose else val_loader
    n=0
    with torch.no_grad():
        for x_cnn, x_aux, y_main, y_neve in data_iter:
            n+=1
            # Each of these is a dict of tensors
            x_cnn     = x_cnn.double().to(device)
            x_aux     = x_aux.double().to(device)
            y_main    = y_main.double().to(device)
            y_neve    = y_neve.double().to(device)
            #is_signal = (y_neve.squeeze() > 0).any(dim=1).cpu().numpy()
            is_signal = (y_neve.squeeze() >= 1).any(dim=1).cpu().numpy()
            if psl:
                min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
                max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
                range_vals  = max_vals - min_vals
                # Avoid division where range is 0
                normalized_x_cnn = torch.where(
                    range_vals > 0,
                    (x_cnn - min_vals) / range_vals,
                    torch.zeros_like(x_cnn)
                )
                x_cnn            = normalized_x_cnn

            ent_dat   = []
            class_dat = []
            for model in modellist:
                if cnn_mode:
                    if model_type == "f":
                        cond                = model(x_cnn).double().to(device)
                    elif model_type == "cf":
                        cond, nclass        = model(x_cnn)
                        cond                = cond.double().to(device)
                        #class_dat.append(nn.Sigmoid()(nclass).cpu().squeeze().tolist())        
                    elif model_type == "c":
                        nclass        = model(x_cnn)
                        #class_dat.append(nn.Sigmoid()(nclass).cpu().squeeze().tolist())        
                    else:
                        raise KeyError
                else:
                    if model_type == "f":
                        cond                = model(x_cnn, x_aux).double().to(device)
                    elif model_type == "cf":
                        cond, nclass        = model(x_cnn, x_aux)
                        cond                = cond.double().to(device)
                        #class_dat.append(nn.Sigmoid()(nclass).cpu().squeeze().tolist())        
                    elif model_type == "c":
                        nclass        = model(x_cnn, x_aux)
                        #class_dat.append(nn.Sigmoid()(nclass).cpu().squeeze().tolist())        
                    else:
                        raise KeyError
                
                if "c" in model_type:
                    class_dat.append(nn.Sigmoid()(nclass).cpu().squeeze().tolist())        
                if "f" in model_type:
                    ent                 = model.pdf.entropy(conditional_input=cond,samplesize=samplesize)['total'].cpu().numpy()
                    ent_dat.append(ent)
            
            c_truth.extend(torch.tensor(is_signal))
            if "f" in model_type:
                ent_ensemble_tru = evaluate_ensemble_entropy_faster(modellist,x_cnn,x_aux, 100, device,model_type=model_type,cnn_mode=cnn_mode)  
                entropy_ind.append(torch.tensor(np.array(ent_dat)))
                entropy.extend(ent_ensemble_tru)    
            
            if "c" in model_type: 
                c_pred_ind.append(torch.tensor(class_dat))
                c_pred.extend(torch.tensor(class_dat).mean(0))
    if len(c_truth) > 0      : c_truth     = torch.stack(c_truth)
    if len(c_pred) > 0       : c_pred      = torch.stack(c_pred)
    if len(c_pred_ind) > 0   : c_pred_ind  = torch.cat(c_pred_ind, dim=1)
    if len(entropy) > 0      : entropy     = torch.stack(entropy)
    if len(entropy_ind) > 0  : entropy_ind = torch.cat(entropy_ind, dim=1)             
            
            
    return c_truth, c_pred, c_pred_ind, entropy, entropy_ind
