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

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial import ConvexHull, Delaunay


UNIFORM_NOISY  = False
SMOOTHED_PRIOR = True
ENTROPY_REG    = False
REG_TYPE1      = False
REG_TYPE2      = False
H_MIN          = -3
LAMBD_HREG1    = 0.2
LAMBD_HREG2    = 0.05
WEIGHTING      = False
#epsilon_x      = 0.065171/2
#epsilon_y      = 0.036447/2
HULL_PRIOR     = True
xscale         = 1
yscale         = 1
NECNN_MODE     = False

print("UNIFORM_NOISY  : " , UNIFORM_NOISY)
print("!"*80)
print("!"*80)
print("SMOOTHED_PRIOR : " , SMOOTHED_PRIOR)
print("!"*80)
print("!"*80)
print("ENTROPY_REG    : " , ENTROPY_REG)
print("REG_TYPE1      : " , REG_TYPE1)
print("REG_TYPE2      : " , REG_TYPE2)
print("H_MIN          : " , H_MIN)
print("LAMBD_HREG1    : " , LAMBD_HREG1)
print("LAMBD_HREG2    : " , LAMBD_HREG2)
print("WEIGHTING      : " , WEIGHTING)
print("xscale         : " , xscale)
print("yscale         : " , yscale)
print("NECNN_MODE         : " , NECNN_MODE)

def bin_width_check(data):
    # Step 0: remove NaNs
    clean_data = np.array(data)[~np.isnan(np.array(data))]
    # print(clean_data)
    # Step 1: sort the data
    sorted_data = np.sort(clean_data)

    # Step 2: compute consecutive differences
    gaps = np.diff(sorted_data)
    # print(gaps)
    # Step 3: minimum gap is the minimum bin width
    min_bin_width = np.max(gaps[gaps > 0])

    print("Minimum bin width to avoid empty bins:", min_bin_width)
    return gaps,min_bin_width

def get_epsilon_smoothing(inloader,xscale=1,yscale=1):
    
    x,y   = torch.tensor([dat for p,dat in inloader.dataset.flow_outputs.items()])
    plotx = x[~torch.isnan(x)]
    ploty = y[~torch.isnan(y)]
    data  = torch.vstack([plotx,ploty]).T 
    
    gaps,mbinwdx =  bin_width_check(x)
    gaps,mbinwdy =  bin_width_check(y)
    
    return torch.tensor([mbinwdx*xscale,mbinwdy*yscale])  # bin width for each column
    
def get_hull_var(inloader,epsilon_rand = 0.1):
    x,y = torch.tensor([dat for p,dat in inloader.dataset.flow_outputs.items()])
    plotx = x[~torch.isnan(x)]
    ploty = y[~torch.isnan(y)]
    data = torch.vstack([plotx,ploty]).T.numpy()
    hull = ConvexHull(data)
    hull_pts = data[hull.vertices]
    centroid = hull_pts.mean(axis=0)
    expanded = centroid + (1 + epsilon_rand) * (hull_pts - centroid)

    # Triangulate the expanded hull polygon
    rp_d_hull = ConvexHull(expanded)
    rp_d_tri = Delaunay(expanded[rp_d_hull.vertices])
    rp_d_mins = expanded.min(axis=0)
    rp_d_maxs = expanded.max(axis=0)
    return rp_d_hull, rp_d_tri, rp_d_mins, rp_d_maxs

def sample_uniform_hull(hull,tri,mins, maxs, n):
    pts = []
    while len(pts) < n:
        cand = np.random.uniform(mins, maxs, size=(n*2,2))
        inside = cand[tri.find_simplex(cand) >= 0]
        pts.extend(inside.tolist())
    return np.array(pts[:n])

def normalize_tensor_log1p(tensor, vmin, vmax, lower=0.5, upper=1.0, eps=1e-6):
    """
    Normalize a tensor using log1p with specified vmin and vmax per feature.

    Args:
        tensor: input tensor of shape [batch, features]
        vmin: tensor of shape [1, features] (min values per feature)
        vmax: tensor of shape [1, features] (max values per feature)
        lower: lower bound of output range
        upper: upper bound of output range
        eps: small epsilon to avoid exact zeros

    Returns:
        Normalized tensor of shape [batch, features] in [lower, upper]
    """
    x = torch.log1p(tensor)  # log1p handles zeros
    denom = torch.clamp(vmax - vmin, min=1e-12)  # avoid division by zero
    y_norm = (x - vmin) / denom
    y_norm = eps + (1 - eps) * y_norm  # enforce non-zero
    y_scaled = lower + (upper - lower) * y_norm
    return y_scaled

def train_loop_nf( modelf, optimizerf, train_loader, 
                   epochs=10, val_loader=None, device=None, scheduler=None, clipnorm=None, patience=10, 
                   save_dir_path="./test_racnnflow", flow_eval_epochs = 10, noisy=True, cnn_mode=False, plot=True, 
                   n_batch_retry=5, n_epoch_retry=5, rollback_epochs=3, psl=False):

    print("UNIFORM_NOISY  : " , UNIFORM_NOISY)
    print("SMOOTHED_PRIOR : " , SMOOTHED_PRIOR)
    print("ENTROPY_REG    : " , ENTROPY_REG)
    print("REG_TYPE1      : " , REG_TYPE1)
    print("REG_TYPE2      : " , REG_TYPE2)
    print("H_MIN          : " , H_MIN)
    print("LAMBD_HREG1    : " , LAMBD_HREG1)
    print("LAMBD_HREG2    : " , LAMBD_HREG2)
    print("WEIGHTING      : " , WEIGHTING)
    print("xscale         : " , xscale)
    print("yscale         : " , yscale)
    print("NECNN_MODE         : " , NECNN_MODE)
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelf.to(device)
    modelf.train()

    os.makedirs(save_dir_path, exist_ok=True)
    save_path_f       = os.path.join(save_dir_path, "best_model.pt")
    ediff_save_path_f = os.path.join(save_dir_path, "best_model_ediff.pt")
    out_dict_path_f   = os.path.join(save_dir_path, "out_dict.pt")
    
    best_val_loss_f = np.inf
    best_ediff_std = np.inf
    best_model_wts_f = copy.deepcopy(modelf.state_dict())

    epochs_without_improvement = 0
    
    avg_logloss_dat        = []
    val_logloss_dat        = []
    ediff_mean_dat         = []
    ediff_std_dat          = []
    tail_thresh_dat        = []
    outlier_intensity_dat  = []
    last_saved_model_f_epoch = 0

    progress_bar = tqdm(range(epochs), leave=False)
    epsilon = get_epsilon_smoothing(train_loader,xscale,xscale).to(device).double()

    #y_main_min, y_main_max = torch.tensor([[np.nanmin(v), np.nanmax(v)*1.05] for k,v in train_loader.dataset.flow_outputs.items()]).T.to(device)
#    y_main_min, y_main_max = torch.tensor([[-0.01, 1.02] for k,v in train_loader.dataset.flow_outputs.items()]).T.to(device).to(torch.float64)
    if HULL_PRIOR  : rp_d_hull, rp_d_tri, rp_d_mins, rp_d_maxs = get_hull_var(train_loader)
    else           : y_main_min, y_main_max = torch.tensor([[-0.01, 1.02] for k,v in train_loader.dataset.flow_outputs.items()]).T.to(device).to(torch.float64)


    # Store model states for epoch-level rollback (keep more states for multi-epoch rollback)
    max_states_to_keep = max(rollback_epochs + n_epoch_retry + 2, 10)  # Keep at least 10 states
    epoch_model_states = {}
    epoch_optimizer_states = {}  # Also store optimizer states for proper rollback
    nval_failed        = 0
    consecutive_failures = 0  # Track consecutive epoch failures
    if WEIGHTING:
        wmax = torch.log1p(torch.tensor([[i.max() for k,i in train_loader.dataset.mlp_outputs.items()]]))
        wmin = torch.log1p(torch.tensor([[i.min() for k,i in train_loader.dataset.mlp_outputs.items()]]))

    for epoch in progress_bar:
        # Save current model and optimizer state at the beginning of each epoch
        epoch_model_states[epoch] = copy.deepcopy(modelf.state_dict())
        epoch_optimizer_states[epoch] = copy.deepcopy(optimizerf.state_dict())

        # Clean up old states (keep only last max_states_to_keep)
        if epoch > max_states_to_keep:
            if epoch - max_states_to_keep - 1 in epoch_model_states:
                del epoch_model_states[epoch - max_states_to_keep - 1]
            if epoch - max_states_to_keep - 1 in epoch_optimizer_states:
                del epoch_optimizer_states[epoch - max_states_to_keep - 1]
        
        if nval_failed > 5: 
            raise RuntimeError(f"More than 5 validation has been skipped... Aborting training")
        
        # Epoch retry mechanism with progressive rollback
        epoch_success     = False
        epoch_retry_count = 0
        max_grad_per_epoch = 0

        
        while not epoch_success and epoch_retry_count <= n_epoch_retry:
            try:
                # Enhanced rollback strategy
                if epoch_retry_count > 0:
                    # Calculate rollback distance based on failure count and consecutive failures
                    base_rollback = min(rollback_epochs, epoch)
                    failure_penalty = min(consecutive_failures, 5)  # Cap at 5 additional epochs
                    total_rollback = min(base_rollback + failure_penalty, epoch)
                    
                    rollback_epoch = epoch - total_rollback
                    rollback_epoch = max(0, rollback_epoch)  # Don't go below epoch 0
                    
                    if rollback_epoch in epoch_model_states and rollback_epoch in epoch_optimizer_states:
                        print(f"Epoch {epoch+1} retry {epoch_retry_count}: Rolling back {total_rollback} epochs to epoch {rollback_epoch+1}")
                        print(f"  - Base rollback: {base_rollback}, Failure penalty: {failure_penalty}")
                        
                        # Load both model and optimizer states
                        modelf.load_state_dict(epoch_model_states[rollback_epoch])
                        optimizerf.load_state_dict(epoch_optimizer_states[rollback_epoch])
                        
                        # Truncate training history to rollback point
                        if len(avg_logloss_dat) > rollback_epoch:
                            avg_logloss_dat = avg_logloss_dat[:rollback_epoch]
                        if len(val_logloss_dat) > rollback_epoch:
                            val_logloss_dat = val_logloss_dat[:rollback_epoch]
                        
                        # Also truncate other tracking lists
                        ediff_mean_dat = [x for x in ediff_mean_dat if x[0] <= rollback_epoch]
                        ediff_std_dat = [x for x in ediff_std_dat if x[0] <= rollback_epoch]
                        tail_thresh_dat = [x for x in tail_thresh_dat if x[0] <= rollback_epoch]
                        outlier_intensity_dat = [x for x in outlier_intensity_dat if x[0] <= rollback_epoch]
                        
                    else:
                        print(f"Epoch {epoch+1} retry {epoch_retry_count}: Desired rollback epoch {rollback_epoch+1} not available")
                        # Fall back to most recent available state
                        available_epochs = sorted([e for e in epoch_model_states.keys() if e < epoch])
                        if available_epochs:
                            fallback_epoch = available_epochs[-1]
                            print(f"  - Falling back to epoch {fallback_epoch+1}")
                            modelf.load_state_dict(epoch_model_states[fallback_epoch])
                            optimizerf.load_state_dict(epoch_optimizer_states[fallback_epoch])
                        else:
                            print(f"  - No rollback states available, using current state")
                
                running_log_loss = 0.0
                modelf.train()
                successful_batches = 0
                
                # Process all batches in this epoch
                for batch_idx, (x_cnn, x_aux, y_main, y_neve) in enumerate(train_loader):
                    
                    #DEBUGGING CODE: COMMENT THIS PART OUT FOR ACTUAL USE
                    # if batch_idx > 5 : break
                    #DEBUGGING CODE: COMMENT THIS PART OUT FOR ACTUAL USE

                    # Batch retry mechanism
                    batch_success = False
                    batch_retry_count = 0
                    
                    x_cnn = x_cnn.to(device).double()
                    x_aux = x_aux.to(device).double()
                    y_main = y_main.to(device).double()
                    y_neve = y_neve.to(device).double()
                    is_signal = (y_neve > 0).any(dim=1)
                    if NECNN_MODE: x_aux = x_aux[:,:3]
                    if SMOOTHED_PRIOR:
                        #epsilon = torch.tensor([epsilon_x,epsilon_y]).to(device).double()  # bin width for each column
                        noise_gaussian = torch.randn_like(y_main) * epsilon.to(device)
                        y_main_noisy_gaussian = y_main + noise_gaussian
                        y_main = y_main_noisy_gaussian
                    
                    w = 1
                    if WEIGHTING: w = normalize_tensor_log1p(y_neve.to(device),wmin.to(device),wmax.to(device)).mean(1)
                    
                    # if UNIFORM_NOISY:
                    #     eps = 1e-2  # tune this, e.g. 1e-3–1e-2
                    #     x_cnn_noisy = x_cnn + torch.empty_like(x_cnn).uniform_(-eps, eps)
                    #     x_cnn_noisy = torch.clamp(x_cnn_noisy, 0.0, 1.0)
                    #     x_cnn = x_cnn_noisy
                    
                    if psl:
                        if UNIFORM_NOISY:
                            eps = 30  # tune this, e.g. 1e-3–1e-2
                            x_cnn_noisy = x_cnn + torch.empty_like(x_cnn).uniform_(-eps, eps)
                            #x_cnn_noisy = torch.clamp(x_cnn_noisy, 0.0, 1.0)
                            x_cnn = x_cnn_noisy

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

                    
                    # Check for NaNs in input data
                    if torch.isnan(x_cnn).any(): print(f"WARNING: NaN values detected in x_cnn at epoch {epoch+1}, batch {batch_idx+1}")
                    if torch.isnan(x_aux).any(): print(f"WARNING: NaN values detected in x_aux at epoch {epoch+1}, batch {batch_idx+1}")
                    
                    while not batch_success and batch_retry_count < n_batch_retry:
                        try:
                            optimizerf.zero_grad()
                            if cnn_mode  : cond = modelf(x_cnn)
                            else         : cond = modelf(x_cnn, x_aux)
                            
                            if noisy:
                                # Find rows where all values are NaN
                                num_nan_rows = (~is_signal).sum()
                                
                                # Generate random values in [x_min, x_max) 
                                #random_vals = y_main_min + (y_main_max - y_main_min) * torch.rand((num_nan_rows.to(device), y_main.to(device).shape[1])).to(device)
                                if HULL_PRIOR : random_vals = torch.tensor(sample_uniform_hull(rp_d_hull,rp_d_tri,rp_d_mins, rp_d_maxs,n=num_nan_rows))
                                else          : random_vals = y_main_min + (y_main_max - y_main_min) * torch.rand((num_nan_rows.to(device), y_main.to(device).shape[1])).to(device)
                                
                                # Replace NaN rows with generated values
                                y_main[~is_signal] = random_vals.to(device)
                                
                                log_pdf, _, _ = modelf.pdf(y_main, conditional_input=cond)
                                log_loss = -(w*log_pdf).mean()

                                #DEBUGGING CODE: COMMENT THIS PART OUT FOR ACTUAL USE
                                # if batch_idx > 3 and epoch == 2: 
                                #     raise ValueError("DEBUGING ERROR")
                                #DEBUGGING CODE: COMMENT THIS PART OUT FOR ACTUAL USE

                            else:
                                if is_signal.any():
                                    log_pdf, _, _ = modelf.pdf(y_main[is_signal], conditional_input=cond[is_signal])
                                    log_loss      = -(w*log_pdf).mean()
                                else:
                                    log_loss      = torch.tensor(0.0, device=device)
                                
                            if ENTROPY_REG:
                                entropy  = modelf.pdf.entropy(conditional_input=cond[is_signal],samplesize=100)['total']
                                entropy_mean = entropy.mean().item()
                                if REG_TYPE1 == True:
                                    penalty = max(0.0, H_MIN - entropy_mean)
                                    log_loss = log_loss+( LAMBD_HREG1*penalty)
                                if REG_TYPE2 == True:
                                    log_loss = log_loss+(-LAMBD_HREG2*entropy_mean)
                                #else:
                                #    raise KeyError(f"ENTROPY_REG can only have REG_TYPE of 1 or 2, got {REG_TYPE}")
                                                    
                            log_loss.backward()
                            
                            if clipnorm is not None: 
                                grad_norm = torch.nn.utils.clip_grad_norm_(modelf.parameters(), clipnorm)
                                if grad_norm > clipnorm and grad_norm > max_grad_per_epoch:
                                    max_grad_per_epoch = grad_norm
                                    
                                    #print(f"Gradient clipped in epoch {epoch+1}, batch {batch_idx+1}: norm {grad_norm:.4f} -> {clipnorm}")
                            optimizerf.step()
                            
                            if scheduler: scheduler.step()
                            
                            running_log_loss   += log_loss.item()
                            batch_success      = True
                            successful_batches += 1
                            
                        except (RuntimeError, ValueError, FloatingPointError, torch.cuda.OutOfMemoryError, Exception) as e:
                            batch_retry_count += 1
                            
                            print(f"\nError in epoch {epoch+1}, batch {batch_idx+1}, attempt {batch_retry_count}/{n_batch_retry}: {type(e).__name__}: {str(e)}")
                            
                            # Clear gradients and reset optimizer state for this batch
                            optimizerf.zero_grad()
                            
                            # Optional: Clear CUDA cache if CUDA OOM error
                            if torch.cuda.is_available() and "out of memory" in str(e).lower():
                                torch.cuda.empty_cache()
                            
                            if batch_retry_count >= n_batch_retry:
                                print(f"Batch failed after {n_batch_retry} attempts. This will trigger epoch retry.")
                                raise e  # Re-raise to trigger epoch retry
                            else:
                                print(f"Retrying batch... {batch_retry_count} time")
                
                # If we get here, all batches in the epoch succeeded
                epoch_success = True
                consecutive_failures = 0  # Reset consecutive failures counter
                
                # Calculate average loss for this epoch
                if successful_batches > 0  : avg_log_loss = running_log_loss / successful_batches
                else                       : avg_log_loss = float('inf')
                    
                avg_logloss_dat.append(avg_log_loss)
                
            except (RuntimeError, ValueError, FloatingPointError, torch.cuda.OutOfMemoryError, Exception) as e:
                epoch_retry_count += 1
                consecutive_failures += 1  # Increment consecutive failures
                
                print(f"\nEpoch {epoch+1} failed (attempt {epoch_retry_count}/{n_epoch_retry + 1}): {type(e).__name__}: {str(e)}")
                print(f"Consecutive failures: {consecutive_failures}")
                
                if epoch_retry_count > n_epoch_retry:
                    print(f"Epoch {epoch+1} failed after {n_epoch_retry + 1} attempts. Aborting training.")
                    raise RuntimeError(f"Training failed: Could not complete epoch {epoch+1} after {n_epoch_retry + 1} attempts")
                else:
                    print(f"Retrying epoch {epoch+1}...")
                    
                # Clear CUDA cache before retry
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        if clipnorm is not None and max_grad_per_epoch > 0:
            print(f"Gradient clipped in epoch {epoch+1}: max_grad_per_epoch {max_grad_per_epoch:.4f} : Using {clipnorm}")

        # Validation phase (only if epoch succeeded)
        if val_loader:
            if ((epoch+1) % flow_eval_epochs) == 0  : flow_eval = True
            else                                    : flow_eval = False
            
            # Add error handling to validation
            try:
                val_log_loss, ediff_mean, ediff_std, tail_thresh, outlier_intensity, _ , _ = validate_loop_nf(modelf, val_loader, device, flow_eval=flow_eval,cnn_mode=cnn_mode,psl=psl)
            except Exception as e:
                print(f"\nValidation error in epoch {epoch+1}: {type(e).__name__}: {str(e)}")
                print("Skipping validation for this epoch...")
                val_log_loss      = np.inf
                ediff_mean        = None
                ediff_std         = None
                tail_thresh       = None
                outlier_intensity = None
                nval_failed      += 1
                
            val_logloss_dat.append(val_log_loss)

            if ediff_mean is not None        : ediff_mean_dat.append([epoch, ediff_mean])
            if ediff_std is not None         : ediff_std_dat.append([epoch, ediff_std])
            if tail_thresh is not None       : tail_thresh_dat.append([epoch, tail_thresh])
            if outlier_intensity is not None : outlier_intensity_dat.append([epoch, outlier_intensity])
            if flow_eval:
                modelf_eval = copy.deepcopy(modelf.state_dict())
                last_ediff_wts = os.path.join(save_dir_path, f"model_ediff_{epoch}.pt")
                torch.save(modelf_eval, last_ediff_wts)
                out_dict = {
                "save_dir_path": save_dir_path,
                "avg_logloss_dat": avg_logloss_dat,
                "val_logloss_dat": val_logloss_dat,
                "ediff_mean_dat": ediff_mean_dat,
                "ediff_std_dat": ediff_std_dat,
                "tail_thresh_dat": tail_thresh_dat,
                "outlier_intensity_dat": outlier_intensity_dat
                }
                torch.save(out_dict, out_dict_path_f)
            
            # Plotting code
            if epoch > 1 and plot:
                if flow_eval:
                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns
                    
                    # Subplot 1
                    axs[0].plot(avg_logloss_dat, label="avg_logloss_dat")
                    axs[0].plot(val_logloss_dat, label="val_logloss_dat")
                    axs[0].legend()
                    axs[0].set_xlabel("epochs")
                    axs[0].set_title("Loss")
                    
                    # Subplot 2
                    axs[1].plot(*np.transpose(ediff_mean_dat), label="ediff_mean_dat")
                    axs[1].plot(*np.transpose(tail_thresh_dat), label="tail_thresh_dat")
                    axs[1].axhline(1, linestyle="--")
                    axs[1].legend(loc="upper right", bbox_to_anchor=(1, 1))
                    axs[1].set_xlabel("epochs")
                    axs[1].set_title("EDiff Mean & Tail Threshold")

                    # Subplot 3
                    axs[2].plot(*np.transpose(ediff_std_dat), label="ediff_std_dat")
                    axs[2].legend(loc="upper right", bbox_to_anchor=(1, 1))
                    axs[2].set_xlabel("epochs")
                    axs[2].set_title("EDiff Std")

                    # Subplot 4
                    axs[3].plot(*np.transpose(outlier_intensity_dat), label="outlier_intensity_dat")
                    axs[3].legend(loc="upper right", bbox_to_anchor=(1, 1))
                    axs[3].set_xlabel("epochs")
                    axs[3].set_title("Outlier Intensity")

                    plt.tight_layout()
                    plt.show()
            
            # Check for improvement
            if val_log_loss < best_val_loss_f:
                last_saved_model_f_epoch = epoch+1
                best_val_loss_f = val_log_loss
                best_model_wts_f = copy.deepcopy(modelf.state_dict())
                torch.save(best_model_wts_f, save_path_f)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if ediff_std is not None and (ediff_std < best_ediff_std) and (epoch > 2*flow_eval_epochs):
                best_ediff_std = ediff_std
                torch.save(copy.deepcopy(modelf.state_dict()), ediff_save_path_f)

            # Early stopping
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

            

        # Progress bar update
        if (len(avg_logloss_dat) > 0) and (len(val_logloss_dat) > 0):
            last_vals_str_logloss = ', '.join(f'{v:.4f}' for v in avg_logloss_dat[-5:])
            val_last_vals_str_logloss = ', '.join(f'{v:.4f}' for v in val_logloss_dat[-5:])
            progress_bar.set_description(
                f"Epoch {epoch+1}/{epochs} | {last_saved_model_f_epoch} ({epochs_without_improvement}/{patience}) | logloss: [{last_vals_str_logloss}] | val_logloss: [{val_last_vals_str_logloss}]"
            )

    # Load best model weights back into the model
    modelf.load_state_dict(best_model_wts_f)
    print("Training complete. Loaded best model weights.")
    
    out_dict = {
        "save_dir_path": save_dir_path,
        "avg_logloss_dat": avg_logloss_dat,
        "val_logloss_dat": val_logloss_dat,
        "ediff_mean_dat": ediff_mean_dat,
        "ediff_std_dat": ediff_std_dat,
        "tail_thresh_dat": tail_thresh_dat,
        "outlier_intensity_dat": outlier_intensity_dat
    }
    torch.save(out_dict, out_dict_path_f)
    return modelf, out_dict

def validate_loop_nf(modelf,  val_loader, device,verbose=False,samplesize=100,flow_eval=True,cnn_mode=False,plot=True,return_best_sr=False,psl=False):
    modelf.eval()
    
    total_log_loss      = 0.0
    ediff_mean          = None      
    ediff_std           = None      
    tail_thresh         = None      
    outlier_intensity   = None 

    lowest_ent    = np.inf
    bestsr_loss   = None
    bestsr_x_cnn  = None
    bestsr_x_aux  = None
    bestsr_y_main = None
    bestsr_y_neve = None
    
    lpdf = torch.tensor([]).to(device)
    ent =  torch.tensor([]).to(device)

    data_iter = tqdm(val_loader, desc="Validating", leave=False) #if verbose else val_loader
    with torch.no_grad():
        for x_cnn, x_aux, y_main, y_neve in data_iter:
            # Each of these is a dict of tensors
            x_cnn      = x_cnn.to(device).double()
            x_aux      = x_aux.to(device).double()
            y_main     = y_main.to(device).double()
            y_neve     = y_neve.to(device).double()
            if NECNN_MODE: x_aux = x_aux[:,:3]
            is_signal = (y_neve > 0).any(dim=1)
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
            
            # raise KeyError
            # Forward pass
            if cnn_mode:
                cond   = modelf(x_cnn)  # shape: [batch, cond_dim]
            else:
                cond   = modelf(x_cnn, x_aux)  # shape: [batch, cond_dim]
           
            if is_signal.any():    
                log_pdf,_,_  = modelf.pdf(y_main[is_signal], conditional_input=cond[is_signal])
                log_loss     = -log_pdf.mean() 
            else:
                log_loss = torch.tensor(0.0, device=device)
                
            if flow_eval:    
                entropy_dict  = modelf.pdf.entropy(conditional_input=cond[is_signal],samplesize=samplesize)
                lpdf = torch.hstack([lpdf,log_pdf])
                ent =  torch.hstack([ent,entropy_dict["total"]])

                if return_best_sr:
                    
                    #print(lowest_ent,entropy_dict["total"][sorted_indices][0])
                    if entropy_dict["total"].min() < lowest_ent:
                        sorted_indices = torch.argsort(entropy_dict["total"])
                        #lowest_ent   = entropy_dict["total"][sorted_indices][0]
                        #bestsr_loss  = -log_pdf[sorted_indices][0]
                        best_x_cnn   = x_cnn[is_signal][sorted_indices][0]
                        best_x_aux   = x_aux[is_signal][sorted_indices][0]
                        best_y_main  = y_main[is_signal][sorted_indices][0]
                        best_y_neve  = y_neve[is_signal][sorted_indices][0]

            total_log_loss    += log_loss.item()

    avg_log_loss    = total_log_loss / len(val_loader)
    tail_percentile = 95
    intensity_percentile = 95
    
    if flow_eval:    

        ediff             = (-lpdf-ent).cpu().numpy()
        ediff_mean        = ediff.mean()
        ediff_std         = ediff.std()
        tail_thresh       = np.percentile(ediff, tail_percentile)
        tail_prop         = np.sum(ediff > tail_thresh) / len(ediff)
        absediff          = np.abs(ediff)
        T                 = np.percentile(absediff, intensity_percentile)
        excess            = np.sum([x - T for x in absediff if x > T])
        outlier_intensity = excess / np.sum(absediff) if np.sum(absediff) > 0 else 0
        if verbose:
            print(f"Validation Log Loss  = {avg_log_loss:.4f}")
            print(f"ediff.mean()         = {ediff.mean():.4f}")
            print(f"ediff.std()          = {ediff.std():.4f}")
            print(f"tail_thresh          = {tail_thresh:.4f}")
            print(f"outlier_intensity    = {outlier_intensity:.4f}")
        if plot:
            plt.hist2d(-lpdf.cpu(),ent.cpu(),bins=50,norm=LogNorm())
            plt.xlabel("loss")
            plt.ylabel("ent")
            plt.show()
    
        if return_best_sr:
            return avg_log_loss,  ediff_mean, ediff_std, tail_thresh, outlier_intensity, -lpdf, ent, best_x_cnn, best_x_aux, best_y_main, best_y_neve
  
    return avg_log_loss,  ediff_mean, ediff_std, tail_thresh, outlier_intensity, -lpdf, ent

def train_loop_class( modelc, optimizerc, train_loader, 
    epochs=10, val_loader=None, device=None, scheduler=None, clipnorm=None, patience=10, save_dir_path="./test_racnnclass", plot_epochs = 10, cnn_mode=False, plot=True,psl=False):
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelc.train()
    modelc.to(device)

    os.makedirs(save_dir_path, exist_ok=True)
    save_path_c = os.path.join(save_dir_path, "best_model_c.pt")
    out_dict_path_c = os.path.join(save_dir_path, "out_dict_c.pt")

    best_val_loss_c = np.inf
    best_model_wts_c = copy.deepcopy(modelc.state_dict())
    
    epochs_without_improvement = 0
    criterion = nn.BCEWithLogitsLoss()
    
    avg_class_loss_dat     = []
    val_class_loss_dat     = []
    last_saved_model_c_epoch = 0

    progress_bar = tqdm(range(epochs), leave=False)

    for epoch in progress_bar:
        running_class_loss   = 0.0 
        modelc.train()
        #progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} ", leave=False)
        for x_cnn, x_aux, _, y_neve in train_loader:
            modelc.train()
            x_cnn      = x_cnn.to(device).double()
            x_aux      = x_aux.to(device).double()
            y_neve     = y_neve.to(device).double()
            is_signal  = (y_neve.squeeze() > 0).any(dim=1)
            if NECNN_MODE: x_aux = x_aux[:,:3]
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
            
            optimizerc.zero_grad()
            
            if cnn_mode:
                nclass       = modelc(x_cnn)
            else:
                nclass       = modelc(x_cnn, x_aux)

            class_loss   = criterion(nclass , is_signal.double().unsqueeze(1))
            class_loss.backward()
            
            optimizerc.step()
            
            if scheduler: scheduler.step()    
            
            running_class_loss += class_loss.item()
            
         
        avg_class_loss   = running_class_loss/ len(train_loader)
        avg_class_loss_dat.append(avg_class_loss)

        if val_loader:
            val_class_loss = validate_loop_class(modelc, val_loader, device,cnn_mode=cnn_mode,psl=psl)
            val_class_loss_dat.append(val_class_loss)
            
            if (epoch+1 > 1) and (((epoch+1) % plot_epochs) == 0 )and plot:
                # Subplot 1
                plt.plot(avg_class_loss_dat, label="avg_class_loss_dat")
                plt.plot(val_class_loss_dat, label="val_class_loss_dat")
                plt.legend()
                plt.xlabel("epochs")
                plt.title("Loss")
                plt.tight_layout()
                plt.show()            
            

            if val_class_loss < best_val_loss_c:
                last_saved_model_c_epoch = epoch+1
                best_val_loss_c = val_class_loss
                best_model_wts_c = copy.deepcopy(modelc.state_dict())
                torch.save(best_model_wts_c, save_path_c)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break
   
        if len(avg_class_loss_dat) > 0:
            last_vals_str_class = ', '.join(f'{v:.4f}' for v in avg_class_loss_dat[-5:])
            val_last_vals_str_class = ', '.join(f'{v:.4f}' for v in val_class_loss_dat[-5:])
            progress_bar.set_description(
                f"Epoch {epoch+1}/{epochs} | {last_saved_model_c_epoch} | classloss: [{last_vals_str_class}] | val_classloss: [{val_last_vals_str_class}]"
            )


    # Load best model weights back into the model
    modelc.load_state_dict(best_model_wts_c)
    print("Training complete. Loaded best model weights.")
    out_dict =  {"save_dir_path"     :save_dir_path,
                 "avg_class_loss_dat":avg_class_loss_dat,
                 "val_class_loss_dat":val_class_loss_dat,
                 }
    torch.save(out_dict,out_dict_path_c)
    return modelc, out_dict

def validate_loop_class(modelc, val_loader, device,verbose=False,cnn_mode=False,psl=False):
    criterion = nn.BCEWithLogitsLoss()
    modelc.eval()
    total_class_loss = 0.0
    data_iter = tqdm(val_loader, desc="Validating", leave=False) #if verbose else val_loader
    with torch.no_grad():
        for x_cnn, x_aux, _, y_neve in data_iter:
            x_cnn      = x_cnn.to(device).double()
            x_aux      = x_aux.to(device).double()
            y_neve     = y_neve.to(device).double()
            if NECNN_MODE: x_aux = x_aux[:,:3]
            is_signal = (y_neve > 0).any(dim=1)
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
                nclass       = modelc(x_cnn)
            else:
                nclass       = modelc(x_cnn, x_aux)
            class_loss         = criterion(nclass.squeeze(), is_signal.double())
            total_class_loss  += class_loss.item()

    avg_class_loss  = total_class_loss / len(val_loader)
    
    return avg_class_loss

def test_loop_class(modelc, val_loader, device,verbose=False,cnn_mode=False,psl=False):
    criterion = nn.BCEWithLogitsLoss()
    modelc.eval()
    total_class_loss = 0.0
    data_iter = tqdm(val_loader, desc="Validating", leave=False) if verbose else val_loader
    tru_dat = []
    pred_dat = []
    with torch.no_grad():
        for x_cnn, x_aux, _, y_neve in data_iter:
            x_cnn      = x_cnn.to(device).double()
            x_aux      = x_aux.to(device).double()
            y_neve     = y_neve.to(device).double()
            if NECNN_MODE: x_aux = x_aux[:,:3]
            is_signal = (y_neve > 0).any(dim=1)
            tru_dat.extend(is_signal)
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
                nclass       = modelc(x_cnn)
            else:
                nclass       = modelc(x_cnn, x_aux)  # shape: [batch, cond_dim]
            class_loss         = criterion(nclass.squeeze(), is_signal.double())
            total_class_loss  += class_loss.item()
            pred_dat.extend(nclass)

    avg_class_loss  = total_class_loss / len(val_loader)
    
    return avg_class_loss, tru_dat, pred_dat

def train_loop_merged_model(model, optimizer, train_loader,
    epochs=10, val_loader=None, device=None, scheduler=None, clipnorm=None, patience=10, 
    save_dir_path="./test_racnnclassiflow", warmup_epochs=10, flow_eval_epochs=10, 
    cnn_mode=False, plot=True, noisy=True, n_batch_retry=5, n_epoch_retry=5, 
    rollback_epochs=3, psl=False):

    print("UNIFORM_NOISY  : " , UNIFORM_NOISY)
    print("SMOOTHED_PRIOR : " , SMOOTHED_PRIOR)
    print("ENTROPY_REG    : " , ENTROPY_REG)
    print("REG_TYPE1      : " , REG_TYPE1)
    print("REG_TYPE2      : " , REG_TYPE2)
    print("H_MIN          : " , H_MIN)
    print("LAMBD_HREG1    : " , LAMBD_HREG1)
    print("LAMBD_HREG2    : " , LAMBD_HREG2)
    print("WEIGHTING      : " , WEIGHTING)
    print("xscale         : " , xscale)
    print("yscale         : " , yscale)
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    os.makedirs(save_dir_path, exist_ok=True)
    save_path = os.path.join(save_dir_path, "best_model.pt")
    save_path_f = os.path.join(save_dir_path, "best_model_f.pt")
    save_path_c = os.path.join(save_dir_path, "best_model_c.pt")
    ediff_save_path = os.path.join(save_dir_path, "best_model_ediff.pt")
    out_dict_path = os.path.join(save_dir_path, "out_dict.pt")
    
    best_val_loss = np.inf
    best_val_loss_f = np.inf
    best_val_loss_c = np.inf
    best_ediff_std = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_wts_f = copy.deepcopy(model.state_dict())
    best_model_wts_c = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    criterion = nn.BCEWithLogitsLoss()
    
    avg_loss_dat = []
    avg_logloss_dat = []
    avg_class_loss_dat = []
    val_loss_dat = []
    val_logloss_dat = []
    val_class_loss_dat = []
    ediff_mean_dat = []
    ediff_std_dat = []
    tail_thresh_dat = []
    outlier_intensity_dat = []
    last_saved_model_epoch = 0
    best_log_loss_epoch = 0
    best_class_loss_epoch = 0

    epsilon = get_epsilon_smoothing(train_loader,xscale,xscale).to(device).double()
    print("epsilon      : " , epsilon)

    
    if WEIGHTING:
        wmax = torch.log1p(torch.tensor([[i.max() for k,i in train_loader.dataset.mlp_outputs.items()]]))
        wmin = torch.log1p(torch.tensor([[i.min() for k,i in train_loader.dataset.mlp_outputs.items()]]))

    progress_bar = tqdm(range(epochs), leave=False)
    #y_main_min, y_main_max = torch.tensor([[np.nanmin(v)*0.95, np.nanmax(v)] for k,v in train_loader.dataset.flow_outputs.items()]).T.to(device)
    #y_main_min, y_main_max = torch.tensor([[-0.01, 1.02] for k,v in train_loader.dataset.flow_outputs.items()]).T.to(device).to(torch.float64)

    if HULL_PRIOR  : rp_d_hull, rp_d_tri, rp_d_mins, rp_d_maxs = get_hull_var(train_loader)
    else           : y_main_min, y_main_max = torch.tensor([[-0.01, 1.02] for k,v in train_loader.dataset.flow_outputs.items()]).T.to(device).to(torch.float64)


    # Store model states for epoch-level rollback (keep more states for multi-epoch rollback)
    max_states_to_keep = max(rollback_epochs + n_epoch_retry + 2, 10)  # Keep at least 10 states
    epoch_model_states = {}
    epoch_optimizer_states = {}  # Also store optimizer states for proper rollback
    nval_failed = 0
    consecutive_failures = 0  # Track consecutive epoch failures
    
    for epoch in progress_bar:
        # Save current model and optimizer state at the beginning of each epoch
        epoch_model_states[epoch] = copy.deepcopy(model.state_dict())
        epoch_optimizer_states[epoch] = copy.deepcopy(optimizer.state_dict())

        # Clean up old states (keep only last max_states_to_keep)
        if epoch > max_states_to_keep:
            if epoch - max_states_to_keep - 1 in epoch_model_states      : del epoch_model_states[epoch - max_states_to_keep - 1]
            if epoch - max_states_to_keep - 1 in epoch_optimizer_states  : del epoch_optimizer_states[epoch - max_states_to_keep - 1]
        
        if nval_failed > 5:
            raise RuntimeError(f"More than 5 validation has been skipped... Aborting training")
        
        # Epoch retry mechanism with progressive rollback
        epoch_success = False
        epoch_retry_count = 0
        max_grad_per_epoch = 0

        while not epoch_success and epoch_retry_count <= n_epoch_retry:
            try:
                # Enhanced rollback strategy
                if epoch_retry_count > 0:
                    # Calculate rollback distance based on failure count and consecutive failures
                    base_rollback   = min(rollback_epochs, epoch)
                    failure_penalty = min(consecutive_failures, 5)  # Cap at 5 additional epochs
                    total_rollback  = min(base_rollback + failure_penalty, epoch)
                    
                    rollback_epoch = epoch - total_rollback
                    rollback_epoch = max(0, rollback_epoch)  # Don't go below epoch 0
                    
                    if rollback_epoch in epoch_model_states and rollback_epoch in epoch_optimizer_states:
                        print(f"Epoch {epoch+1} retry {epoch_retry_count}: Rolling back {total_rollback} epochs to epoch {rollback_epoch+1}")
                        print(f"  - Base rollback: {base_rollback}, Failure penalty: {failure_penalty}")
                        
                        # Load both model and optimizer states
                        model.load_state_dict(epoch_model_states[rollback_epoch])
                        optimizer.load_state_dict(epoch_optimizer_states[rollback_epoch])
                        
                        # Truncate training history to rollback point
                        if len(avg_loss_dat) > rollback_epoch          : avg_loss_dat = avg_loss_dat[:rollback_epoch]
                        if len(avg_logloss_dat) > rollback_epoch       : avg_logloss_dat = avg_logloss_dat[:rollback_epoch]
                        if len(avg_class_loss_dat) > rollback_epoch    : avg_class_loss_dat = avg_class_loss_dat[:rollback_epoch]
                        if len(val_loss_dat) > rollback_epoch          : val_loss_dat = val_loss_dat[:rollback_epoch]
                        if len(val_logloss_dat) > rollback_epoch       : val_logloss_dat = val_logloss_dat[:rollback_epoch]
                        if len(val_class_loss_dat) > rollback_epoch    : val_class_loss_dat = val_class_loss_dat[:rollback_epoch]
                        
                        # Also truncate other tracking lists
                        ediff_mean_dat        = [x for x in ediff_mean_dat if x[0] <= rollback_epoch]
                        ediff_std_dat         = [x for x in ediff_std_dat if x[0] <= rollback_epoch]
                        tail_thresh_dat       = [x for x in tail_thresh_dat if x[0] <= rollback_epoch]
                        outlier_intensity_dat = [x for x in outlier_intensity_dat if x[0] <= rollback_epoch]
                        
                    else:
                        print(f"Epoch {epoch+1} retry {epoch_retry_count}: Desired rollback epoch {rollback_epoch+1} not available")
                        # Fall back to most recent available state
                        available_epochs = sorted([e for e in epoch_model_states.keys() if e < epoch])
                        if available_epochs:
                            fallback_epoch = available_epochs[-1]
                            print(f"  - Falling back to epoch {fallback_epoch+1}")
                            model.load_state_dict(epoch_model_states[fallback_epoch])
                            optimizer.load_state_dict(epoch_optimizer_states[fallback_epoch])
                        else:
                            print(f"  - No rollback states available, using current state")

                model.train()
                running_loss = 0.0 
                running_log_loss = 0.0 
                running_class_loss = 0.0
                successful_batches = 0
                
                # Process all batches in this epoch
                for batch_idx, (x_cnn, x_aux, y_main, y_neve) in enumerate(train_loader):
                    
                    #DEBUGGING CODE: COMMENT THIS PART OUT FOR ACTUAL USE
                    # if batch_idx > 5 : break
                    #DEBUGGING CODE: COMMENT THIS PART OUT FOR ACTUAL USE
                    
                    # Batch retry mechanism
                    batch_success = False
                    batch_retry_count = 0
                    
                    x_cnn = x_cnn.double().to(device)
                    x_aux = x_aux.double().to(device)
                    y_main = y_main.double().to(device)
                    y_neve = y_neve.double().to(device)
                    if NECNN_MODE: x_aux = x_aux[:,:3]
                    is_signal = (y_neve > 0).any(dim=1)

                    # if UNIFORM_NOISY:
                    #     eps = 1e-2  # tune this, e.g. 1e-3–1e-2
                    #     x_cnn_noisy = x_cnn + torch.empty_like(x_cnn).uniform_(-eps, eps)
                    #     x_cnn_noisy = torch.clamp(x_cnn_noisy, 0.0, 1.0)
                    #     x_cnn = x_cnn_noisy

                    if SMOOTHED_PRIOR:
                        #epsilon = torch.tensor([epsilon_x,epsilon_y]).to(device).double()  # bin width for each column
                        noise_gaussian = torch.randn_like(y_main) * epsilon.to(device)
                        y_main_noisy_gaussian = y_main + noise_gaussian
                        y_main = y_main_noisy_gaussian

                    w = 1
                    if WEIGHTING: w = normalize_tensor_log1p(y_neve.to(device),wmin.to(device),wmax.to(device)).mean(1)

                    if psl:
                        if UNIFORM_NOISY:
                            eps = 30  # tune this, e.g. 1e-3–1e-2
                            x_cnn_noisy = x_cnn + torch.empty_like(x_cnn).uniform_(-eps, eps)
                            #x_cnn_noisy = torch.clamp(x_cnn_noisy, 0.0, 1.0)
                            x_cnn = x_cnn_noisy

                        min_vals = x_cnn.min(dim=-1, keepdim=True)[0]
                        max_vals = x_cnn.max(dim=-1, keepdim=True)[0]
                        range_vals = max_vals - min_vals
                        # Avoid division where range is 0
                        normalized_x_cnn = torch.where(
                            range_vals > 0,
                            (x_cnn - min_vals) / range_vals,
                            torch.zeros_like(x_cnn)
                        )
                        x_cnn = normalized_x_cnn

                    # Check for NaNs in input data
                    if torch.isnan(x_cnn).any(): print(f"WARNING: NaN values detected in x_cnn at epoch {epoch+1}, batch {batch_idx+1}")
                    if torch.isnan(x_aux).any(): print(f"WARNING: NaN values detected in x_aux at epoch {epoch+1}, batch {batch_idx+1}")

                    while not batch_success and batch_retry_count < n_batch_retry:
                        try:
                            optimizer.zero_grad()
                            if cnn_mode: cond, nclass = model(x_cnn)
                            else       : cond, nclass = model(x_cnn, x_aux)

                            log_loss    = torch.tensor(0.0, device=device)
                            flow_weight = 0 

                            if epoch >= warmup_epochs : 
                                flow_weight = 1
                                if noisy:
                                    num_nan_rows = (~is_signal).sum()
                                    #random_vals = y_main_min + (y_main_max - y_main_min) * torch.rand((num_nan_rows.to(device), y_main.to(device).shape[1])).to(device)
                                    if HULL_PRIOR : random_vals = torch.tensor(sample_uniform_hull(rp_d_hull,rp_d_tri,rp_d_mins, rp_d_maxs,n=num_nan_rows))
                                    else          : random_vals = y_main_min + (y_main_max - y_main_min) * torch.rand((num_nan_rows.to(device), y_main.to(device).shape[1])).to(device)
                                    y_main[~is_signal] = random_vals.to(device)
                                    
                                    log_pdf, _, _ = model.pdf(y_main, conditional_input=cond)
                                    log_loss = -(w*log_pdf).mean() 

                                    #DEBUGGING CODE: COMMENT THIS PART OUT FOR ACTUAL USE
                                    # if batch_idx > 3 and epoch == 2: 
                                    #     raise ValueError("DEBUGING ERROR")
                                    #DEBUGGING CODE: COMMENT THIS PART OUT FOR ACTUAL USE

                                else:    
                                    if is_signal.any():    
                                        log_pdf, _, _ = model.pdf(y_main[is_signal], conditional_input=cond[is_signal])
                                        log_loss = -(w*log_pdf).mean() 
                                    else:
                                        log_loss = torch.tensor(0.0, device=device)
                                
                                if ENTROPY_REG:
                                    entropy  = model.pdf.entropy(conditional_input=cond[is_signal],samplesize=100)['total']
                                    entropy_mean = entropy.mean().item()
                                    if REG_TYPE1 == True:
                                        penalty = max(0.0, H_MIN - entropy_mean)
                                        log_loss = log_loss+( LAMBD_HREG1*penalty)
                                    if REG_TYPE2 == True:
                                        log_loss = log_loss+(-LAMBD_HREG2*entropy_mean)
                                    #else:
                                    #    raise KeyError(f"ENTROPY_REG can only have REG_TYPE of 1 or 2, got {REG_TYPE}")
                            
                            class_loss = criterion(nclass, is_signal.double().unsqueeze(1))
                            loss = flow_weight * log_loss + class_loss
                            loss.backward()

                            if clipnorm is not None:
                                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
                                if grad_norm > clipnorm and grad_norm > max_grad_per_epoch:
                                    max_grad_per_epoch = grad_norm
                                    
                            optimizer.step()
                            if scheduler: scheduler.step()
                            
                            running_loss += loss.item()
                            running_log_loss += log_loss.item()
                            running_class_loss += class_loss.item()
                            batch_success = True
                            successful_batches += 1
                            
                        except (RuntimeError, ValueError, FloatingPointError, torch.cuda.OutOfMemoryError, Exception) as e:
                            batch_retry_count += 1
                            
                            print(f"\nError in epoch {epoch+1}, batch {batch_idx+1}, attempt {batch_retry_count}/{n_batch_retry}: {type(e).__name__}: {str(e)}")
                            
                            # Clear gradients and reset optimizer state for this batch
                            optimizer.zero_grad()
                            
                            # Optional: Clear CUDA cache if CUDA OOM error
                            if torch.cuda.is_available() and "out of memory" in str(e).lower(): torch.cuda.empty_cache()
                            
                            if batch_retry_count >= n_batch_retry:
                                print(f"Batch failed after {n_batch_retry} attempts. This will trigger epoch retry.")
                                raise e  # Re-raise to trigger epoch retry
                            else:
                                print(f"Retrying batch... {batch_retry_count} time")

                # If we get here, all batches in the epoch succeeded
                epoch_success        = True
                consecutive_failures = 0  # Reset consecutive failures counter
                
                # Calculate average loss for this epoch
                if successful_batches > 0:
                    avg_loss        = running_loss / successful_batches
                    avg_log_loss    = running_log_loss / successful_batches  
                    avg_class_loss  = running_class_loss / successful_batches
                else:
                    avg_loss = float('inf')
                    avg_log_loss = float('inf')
                    avg_class_loss = float('inf')
                    
                avg_loss_dat.append(avg_loss)
                avg_logloss_dat.append(avg_log_loss)
                avg_class_loss_dat.append(avg_class_loss)
                
            except (RuntimeError, ValueError, FloatingPointError, torch.cuda.OutOfMemoryError, Exception) as e:
                epoch_retry_count += 1
                consecutive_failures += 1  # Increment consecutive failures
                
                print(f"\nEpoch {epoch+1} failed (attempt {epoch_retry_count}/{n_epoch_retry + 1}): {type(e).__name__}: {str(e)}")
                print(f"Consecutive failures: {consecutive_failures}")
                
                if epoch_retry_count > n_epoch_retry:
                    print(f"Epoch {epoch+1} failed after {n_epoch_retry + 1} attempts. Aborting training.")
                    raise RuntimeError(f"Training failed: Could not complete epoch {epoch+1} after {n_epoch_retry + 1} attempts")
                else:
                    print(f"Retrying epoch {epoch+1}...")
                    
                # Clear CUDA cache before retry
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        if clipnorm is not None and max_grad_per_epoch > 0:
            print(f"Gradient clipped in epoch {epoch+1}: max_grad_per_epoch {max_grad_per_epoch:.4f} : Using {clipnorm}")

        # Validation phase (only if epoch succeeded)
        if val_loader:
            if (epoch >= warmup_epochs) and (((epoch+1) % flow_eval_epochs) == 0): flow_eval = True
            else                                                                 : flow_eval = False
            
            # Add error handling to validation
            try:
                val_loss, log_loss, class_loss, ediff_mean, ediff_std, tail_thresh, outlier_intensity, _, _ = validate_loop_merged_model(model, val_loader, device, flow_loss_eval=(epoch >= warmup_epochs), 
                                                                                                                                         flow_eval=flow_eval, cnn_mode=cnn_mode, plot=False, psl=psl)
            except Exception as e:
                print(f"\nValidation error in epoch {epoch+1}: {type(e).__name__}: {str(e)}")
                print("Skipping validation for this epoch...")
                val_loss           = np.inf
                log_loss           = np.inf 
                class_loss         = np.inf
                ediff_mean         = None
                ediff_std          = None 
                tail_thresh        = None
                outlier_intensity  = None
                nval_failed        += 1
            
            val_loss_dat.append(val_loss)
            val_logloss_dat.append(log_loss)
            val_class_loss_dat.append(class_loss)

            if ediff_mean is not None          : ediff_mean_dat.append([epoch, ediff_mean])
            if ediff_std is not None           : ediff_std_dat.append([epoch, ediff_std])
            if tail_thresh is not None         : tail_thresh_dat.append([epoch, tail_thresh])
            if outlier_intensity is not None   : outlier_intensity_dat.append([epoch, outlier_intensity])
            
            if flow_eval:
                model_eval = copy.deepcopy(model.state_dict())
                last_model_ediff_wts_path = os.path.join(save_dir_path, f"model_ediff_{epoch}.pt")
                torch.save(model_eval, last_model_ediff_wts_path)
                
                # Save out_dict during flow_eval like in train_loop_nf
                out_dict = {
                    "save_dir_path": save_dir_path,
                    "avg_loss_dat": avg_loss_dat,
                    "avg_logloss_dat": avg_logloss_dat,
                    "avg_class_loss_dat": avg_class_loss_dat,
                    "val_loss_dat": val_loss_dat,
                    "val_logloss_dat": val_logloss_dat,
                    "val_class_loss_dat": val_class_loss_dat,
                    "ediff_mean_dat": ediff_mean_dat,
                    "ediff_std_dat": ediff_std_dat,
                    "tail_thresh_dat": tail_thresh_dat,
                    "outlier_intensity_dat": outlier_intensity_dat
                }
                torch.save(out_dict, out_dict_path)
            
            # Plotting code
            if epoch > 1 and flow_eval and plot:
                fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns
                
                # Subplot 1
                axs[0].plot(avg_loss_dat, label="avg_loss_dat")
                axs[0].plot(avg_logloss_dat, label="avg_logloss_dat")
                axs[0].plot(avg_class_loss_dat, label="avg_class_loss_dat")
                axs[0].plot(val_loss_dat, label="val_loss_dat")
                axs[0].plot(val_logloss_dat, label="val_logloss_dat")
                axs[0].plot(val_class_loss_dat, label="val_class_loss_dat")
                axs[0].legend()
                axs[0].set_xlabel("epochs")
                axs[0].set_title("Loss")

                # Subplot 2
                axs[1].plot(*np.transpose(ediff_mean_dat), label="ediff_mean_dat")
                axs[1].plot(*np.transpose(tail_thresh_dat), label="tail_thresh_dat")
                axs[1].axhline(1, linestyle="--")
                axs[1].legend(loc="upper right", bbox_to_anchor=(1, 1))
                axs[1].set_xlabel("epochs")
                axs[1].set_title("EDiff Mean & Tail Threshold")

                # Subplot 3
                axs[2].plot(*np.transpose(ediff_std_dat), label="ediff_std_dat")
                axs[2].legend(loc="upper right", bbox_to_anchor=(1, 1))
                axs[2].set_xlabel("epochs") 
                axs[2].set_title("EDiff Std")

                # Subplot 4
                axs[3].plot(*np.transpose(outlier_intensity_dat), label="outlier_intensity_dat")
                axs[3].legend(loc="upper right", bbox_to_anchor=(1, 1))
                axs[3].set_xlabel("epochs")
                axs[3].set_title("Outlier Intensity")

                plt.tight_layout()
                plt.show()
            
            # Check for improvement
            if val_loss < best_val_loss:
                last_saved_model_epoch = epoch+1
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, save_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if log_loss < best_val_loss_f: 
                best_val_loss_f = log_loss
                best_log_loss_epoch = epoch
                best_model_wts_f = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts_f, save_path_f)

            if class_loss < best_val_loss_c: 
                best_val_loss_c = class_loss
                best_class_loss_epoch = epoch
                best_model_wts_c = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts_c, save_path_c)

            if ediff_std is not None and (ediff_std < best_ediff_std) and (epoch > 2*flow_eval_epochs):
                best_ediff_std = ediff_std
                torch.save(copy.deepcopy(model.state_dict()), ediff_save_path)

            # Early stopping
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

        # Progress bar update
        if len(avg_logloss_dat) > 0 and len(val_logloss_dat) > 0:
            last_vals_str_logloss = ', '.join(f'{v:.4f}' for v in avg_logloss_dat[-5:])
            last_vals_str_class = ', '.join(f'{v:.4f}' for v in avg_class_loss_dat[-5:])
            val_last_vals_str_logloss = ', '.join(f'{v:.4f}' for v in val_logloss_dat[-5:])
            val_last_vals_str_class = ', '.join(f'{v:.4f}' for v in val_class_loss_dat[-5:])
            progress_bar.set_description(
                f"Epoch {epoch+1}/{epochs} | {last_saved_model_epoch}:({best_log_loss_epoch},{best_class_loss_epoch}) ({epochs_without_improvement}/{patience}) | logloss: [{last_vals_str_logloss}] | classloss: [{last_vals_str_class}] | val_logloss: [{val_last_vals_str_logloss}] | val_classloss: [{val_last_vals_str_class}]"
            )
    
    # Load best model weights back into the model
    model.load_state_dict(best_model_wts)
    print("Training complete. Loaded best model weights.")
    
    out_dict = {
        "save_dir_path": save_dir_path,
        "avg_loss_dat": avg_loss_dat,
        "avg_logloss_dat": avg_logloss_dat,
        "avg_class_loss_dat": avg_class_loss_dat,
        "val_loss_dat": val_loss_dat,
        "val_logloss_dat": val_logloss_dat,
        "val_class_loss_dat": val_class_loss_dat,
        "ediff_mean_dat": ediff_mean_dat,
        "ediff_std_dat": ediff_std_dat,
        "tail_thresh_dat": tail_thresh_dat,
        "outlier_intensity_dat": outlier_intensity_dat
    }
    torch.save(out_dict, out_dict_path)
    return model, out_dict

def validate_loop_merged_model(model, val_loader, device,verbose=False,samplesize=100,flow_loss_eval=True,flow_eval=True,cnn_mode=False,plot=True,return_best_sr=False,psl=False):
    criterion = nn.BCEWithLogitsLoss()

    model.eval()

    total_loss        = 0.0
    total_log_loss    = 0.0
    total_class_loss  = 0.0
    ediff_mean        = None      
    ediff_std         = None      
    tail_thresh       = None      
    outlier_intensity = None 

    lowest_ent    = np.inf
    bestsr_loss   = None
    bestsr_x_cnn  = None
    bestsr_x_aux  = None
    bestsr_y_main = None
    bestsr_y_neve = None
    
    lpdf = torch.tensor([]).to(device)
    ent =  torch.tensor([]).to(device)

    data_iter = tqdm(val_loader, desc="Validating", leave=False) #if verbose else val_loader
    with torch.no_grad():
        for x_cnn, x_aux, y_main, y_neve in data_iter:

            # Each of these is a dict of tensors
            x_cnn  = x_cnn.double().to(device) #    torch.hstack(list(x_cnn.values())).unsqueeze(1).to(device)
            x_aux  = x_aux.double().to(device) #    torch.vstack(list(x_aux.values())).T.to(device)
            y_main = y_main.double().to(device) #   torch.vstack(list(y_main.values())).T.to(device)
            y_neve = y_neve.double().to(device) #   torch.hstack(list(y_neve.values())).unsqueeze(1).to(device)
            if NECNN_MODE: x_aux = x_aux[:,:3]    
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

            is_signal = (y_neve > 0).any(dim=1)

            # Forward pass
            if cnn_mode:
                cond,nclass    = model(x_cnn)
            else:
                cond,nclass    = model(x_cnn, x_aux)
            
            flow_weight  = 0 
            log_loss = torch.tensor(0.0, device=device)
            if flow_loss_eval :
                flow_weight = 1
                if is_signal.any():    
                    log_pdf,_,_  = model.pdf(y_main[is_signal], conditional_input=cond[is_signal])
                    log_loss     = -log_pdf.mean() 
                else:
                    log_loss = torch.tensor(0.0, device=device)

                if flow_eval:
                    entropy_dict  = model.pdf.entropy(conditional_input=cond[is_signal],samplesize=samplesize)
                    lpdf = torch.hstack([lpdf,log_pdf])
                    ent =  torch.hstack([ent,entropy_dict["total"]])

                    if return_best_sr:
                        if entropy_dict["total"].min() < lowest_ent:
                            sorted_indices = torch.argsort(entropy_dict["total"])
                            best_x_cnn   = x_cnn[is_signal][sorted_indices][0]
                            best_x_aux   = x_aux[is_signal][sorted_indices][0]
                            best_y_main  = y_main[is_signal][sorted_indices][0]
                            best_y_neve  = y_neve[is_signal][sorted_indices][0]

            class_loss         = criterion(nclass.squeeze(), is_signal.double())
            total_log_loss    += log_loss.item()
            total_class_loss  += class_loss.item()
            total_loss        += flow_weight*log_loss.item() + class_loss.item()

    avg_val_loss       = total_loss / len(val_loader)
    avg_log_loss       = total_log_loss / len(val_loader)
    avg_class_loss     = total_class_loss / len(val_loader)
    ediff_mean         = None         
    ediff_std          = None         
    tail_thresh        = None         
    outlier_intensity  = None                 

    
    if flow_eval:
        tail_percentile = 95
        intensity_percentile = 95

        ediff             = (-lpdf-ent).cpu().numpy()
        ediff_mean        = ediff.mean()
        ediff_std         = ediff.std()
        tail_thresh       = np.percentile(ediff, tail_percentile)
        tail_prop         = np.sum(ediff > tail_thresh) / len(ediff)
        absediff          = np.abs(ediff)
        T                 = np.percentile(absediff, intensity_percentile)
        excess            = np.sum([x - T for x in absediff if x > T])
        outlier_intensity = excess / np.sum(absediff) if np.sum(absediff) > 0 else 0

        if verbose:
            
            print(f"Validation Loss      = {avg_val_loss:.4f}")
            print(f"Validation Log Loss  = {avg_log_loss:.4f}")
            print(f"Validation Class Loss  = {avg_class_loss:.4f}")
            print(f"ediff.mean()         = {ediff.mean():.4f}")
            print(f"ediff.std()          = {ediff.std():.4f}")
            print(f"tail_thresh          = {tail_thresh:.4f}")
            print(f"outlier_intensity    = {outlier_intensity:.4f}")
        
        if plot:
            print("!")
            plt.hist2d(-lpdf.cpu(),ent.cpu(),bins=50,norm=LogNorm())
            plt.xlabel("loss")
            plt.ylabel("ent")
            plt.show()
    
        if return_best_sr:
            return avg_val_loss,avg_log_loss,avg_class_loss,ediff_mean, ediff_std, tail_thresh, outlier_intensity,-lpdf,ent, best_x_cnn, best_x_aux, best_y_main, best_y_neve

    return avg_val_loss,avg_log_loss,avg_class_loss,ediff_mean, ediff_std, tail_thresh, outlier_intensity,-lpdf,ent

# def train_loop_merged_model(model,optimizer,train_loader,
#     epochs=10,val_loader=None,device=None,scheduler=None,clipnorm=None,patience=10,save_dir_path="./test_racnnclassiflow",warmup_epochs = 10 ,flow_eval_epochs = 10,cnn_mode=False,plot=True,noisy=True,max_retries=3,psl=False):
    
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.train()

#     os.makedirs(save_dir_path, exist_ok=True)
#     save_path    = os.path.join(save_dir_path, "best_model.pt")
#     save_path_f  = os.path.join(save_dir_path, "best_model_f.pt")
#     save_path_c  = os.path.join(save_dir_path, "best_model_c.pt")
#     ediff_save_path = os.path.join(save_dir_path, "best_model_ediff.pt")
#     out_dict_path = os.path.join(save_dir_path, "out_dict.pt")
    
    
#     best_val_loss   = np.inf
#     best_val_loss_f = np.inf
#     best_val_loss_c = np.inf
#     best_model_wts_f = copy.deepcopy(model.state_dict())
#     best_model_wts_c = copy.deepcopy(model.state_dict())
#     epochs_without_improvement = 0
#     criterion = nn.BCEWithLogitsLoss()
    
#     avg_loss_dat           = []
#     avg_logloss_dat        = []
#     avg_class_loss_dat     = []
#     val_loss_dat           = []
#     val_logloss_dat        = []
#     val_class_loss_dat     = []
#     ediff_mean_dat         = []
#     ediff_std_dat          = []
#     tail_thresh_dat        = []
#     outlier_intensity_dat  = []
#     last_saved_model_epoch = 0
#     best_log_loss_epoch = 0
#     best_class_loss_epoch = 0

#     progress_bar           = tqdm(range(epochs), leave=False)
#     y_main_min, y_main_max = torch.tensor([[np.nanmin(v), np.nanmax(v)] for k,v in train_loader.dataset.flow_outputs.items()]).T.to(device)

#     batch_skip = 0
#     loaded_1 = False 
#     loaded_2 = False 
#     mdict = {}
#     nval_failed = 0
#     for epoch in progress_bar:
#         model.train()
#         running_loss       = 0.0 
#         running_log_loss   = 0.0 
#         running_class_loss = 0.0 
#         mdict[epoch] = copy.deepcopy(model.state_dict())
#         if epoch > 5       : del mdict[epoch-5]
#         if nval_failed > 5 : raise RuntimeError(f"More than 5 validation has been skipped... Aborting training")
    
#         for batch_idx, (x_cnn, x_aux, y_main, y_neve) in enumerate(train_loader):
#             if batch_skip == 5 and loaded_1 == False:
#                 print(f"More than 5 batch has been skipped... Reload a saved model (mdict[{epoch-2}])")
#                 model.load_state_dict(mdict[epoch-2])
#                 loaded_1 = True

#             if batch_skip > 10 and loaded_2 == False:
#                 print(f"More than 5 batch has been skipped... Reload a saved model (mdict[{epoch-4}])")
#                 model.load_state_dict(mdict[epoch-4])
#                 loaded_2 = True


#             if batch_skip > 15:
#                 raise RuntimeError(f"More than 15 batch has been skipped... Aborting training")
            
#             # Retry mechanism for this batch
#             batch_success = False
#             retry_count   = 0
#             model.train()
#             x_cnn      = x_cnn.double().to(device)
#             x_aux      = x_aux.double().to(device)
#             y_main     = y_main.double().to(device)
#             y_neve     = y_neve.double().to(device)
#             is_signal  = (y_neve.squeeze() > 0).any(dim=1)
#             if psl:
#                         min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
#                         max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
#                         range_vals  = max_vals - min_vals
#                         # Avoid division where range is 0
#                         normalized_x_cnn = torch.where(
#                             range_vals > 0,
#                             (x_cnn - min_vals) / range_vals,
#                             torch.zeros_like(x_cnn)
#                         )
#                         x_cnn            = normalized_x_cnn

#             while not batch_success and retry_count < max_retries:
                
#                 try:
                    
                    
#                     optimizer.zero_grad()
#                     if cnn_mode:
#                         cond,nclass    = model(x_cnn)
#                     else:
#                         cond,nclass    = model(x_cnn, x_aux)

#                     log_loss     = torch.tensor(0.0, device=device)
#                     flow_weight  = 0 

#                     if epoch >= warmup_epochs : 
#                         flow_weight = 1
#                         if noisy:
#                             num_nan_rows = (~is_signal).sum()
#                             random_vals = y_main_min + (y_main_max - y_main_min) * torch.rand((num_nan_rows.to(device), y_main.to(device).shape[1])).to(device)
#                             y_main[~is_signal] = random_vals.to(device)

#                             # This is where model.pdf might break - wrap in try-catch
#                             log_pdf,_,_  =  model.pdf(y_main, conditional_input=cond)
#                             log_loss     = -log_pdf.mean() 
#                         else:    
#                             if is_signal.any():    
#                                 # This is where model.pdf might break - wrap in try-catch
#                                 log_pdf,_,_  = model.pdf(y_main[is_signal], conditional_input=cond[is_signal])
#                                 log_loss     = -log_pdf.mean() 
#                             else:
#                                 log_loss = torch.tensor(0.0, device=device)
                        
#                     class_loss   = criterion(nclass , is_signal.double().unsqueeze(1))
#                     loss         = flow_weight *log_loss + class_loss
#                     loss.backward()

#                     if clipnorm is not None:torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
                    
#                     optimizer.step()
#                     if scheduler: scheduler.step()    
                    
#                     running_loss       += loss.item()
#                     running_log_loss   += log_loss.item()
#                     running_class_loss += class_loss.item()
                    
#                     batch_success = True  # If we get here, the batch succeeded
                    
#                 except (RuntimeError, ValueError, FloatingPointError, torch.cuda.OutOfMemoryError) as e:
#                     retry_count += 1
#                     print(f"\nError in epoch {epoch+1}, batch {batch_idx+1}, attempt {retry_count}/{max_retries}: {type(e).__name__}: {str(e)}")
                    
#                     # Clear gradients and reset optimizer state for this batch
#                     optimizer.zero_grad()
                    
#                     # Optional: Clear CUDA cache if CUDA OOM error
#                     if torch.cuda.is_available() and "out of memory" in str(e).lower():
#                         torch.cuda.empty_cache()
                    
#                     if retry_count >= max_retries:
#                         print(f"Failed to process batch after {max_retries} attempts. Skipping batch.")
#                         batch_skip+=1
#                         batch_success = True  # Exit the retry loop, but skip this batch
#                         # Set dummy values for this batch to maintain running averages
#                         running_loss       += 0.0  # or use last valid loss
#                         running_log_loss   += 0.0
#                         running_class_loss += 0.0
#                     else:
#                         print(f"Retrying batch...")
#                         #  Optionally add a small delay
#                         # import time
#                         # time.sleep(0.1)

#         if (len(avg_logloss_dat) > 0 )and (len(val_logloss_dat) > 0):
#             progress_bar.set_description(
#             f"Epoch {epoch+1}/{epochs} | {last_saved_model_epoch}:({best_log_loss_epoch},{best_class_loss_epoch}) | logloss: [{last_vals_str_logloss}] | classloss: [{last_vals_str_class}] | val_logloss: [{val_last_vals_str_logloss}] | val_classloss: [{val_last_vals_str_class}]"
#             )
        
#         avg_loss         = running_loss / len(train_loader)
#         avg_log_loss     = running_log_loss/ len(train_loader)
#         avg_class_loss   = running_class_loss/ len(train_loader)

#         avg_loss_dat.append(avg_loss)
#         avg_logloss_dat.append(avg_log_loss)
#         avg_class_loss_dat.append(avg_class_loss)
  
#         if val_loader:
            
#             if (epoch >=warmup_epochs) and (((epoch+1) % flow_eval_epochs) == 0) : flow_eval = True
#             else                                                                 : flow_eval = False
            
#             # Also add error handling to validation
#             try:
#                 val_loss, log_loss,class_loss, ediff_mean, ediff_std, tail_thresh, outlier_intensity, _ , _ = validate_loop_merged_model(model, val_loader, device,flow_loss_eval=(epoch >=warmup_epochs), flow_eval=flow_eval,cnn_mode=cnn_mode,plot=plot,psl=psl)
#             except Exception as e:
#                 print(f"\nValidation error in epoch {epoch+1}: {type(e).__name__}: {str(e)}")
#                 print("Skipping validation for this epoch...")
#                 val_loss = np.inf
#                 log_loss = np.inf 
#                 class_loss = np.inf
#                 ediff_mean = None
#                 ediff_std = None 
#                 tail_thresh = None
#                 outlier_intensity = None
#                 nval_failed+=1
            
#             val_loss_dat.append(val_loss)
#             val_logloss_dat.append(log_loss)
#             val_class_loss_dat.append(class_loss)
            
#             if flow_eval                      :
#                 modelf_eval = copy.deepcopy(model.state_dict())
#                 last_model_ediff_wts_path = os.path.join(save_dir_path, f"model_ediff_{epoch}.pt")
#                 torch.save(modelf_eval, last_model_ediff_wts_path)
                        
#             if epoch > 1 and flow_eval and plot:
#                 fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns
                
#                 # Subplot 1
#                 axs[0].plot(avg_loss_dat,       label="avg_loss_dat")
#                 axs[0].plot(avg_logloss_dat,    label="avg_logloss_dat")
#                 axs[0].plot(avg_class_loss_dat, label="avg_class_loss_dat")
#                 axs[0].plot(val_loss_dat,       label="val_loss_dat")
#                 axs[0].plot(val_logloss_dat,    label="val_logloss_dat")
#                 axs[0].plot(val_class_loss_dat, label="val_class_loss_dat")
#                 #axs[0].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                 axs[0].legend()
#                 axs[0].set_xlabel("epochs")
#                 axs[0].set_title("Loss")
#                 if epoch >=warmup_epochs and flow_eval:
#                     if ediff_mean is not None         : ediff_mean_dat.append([epoch,ediff_mean])
#                     if ediff_std is not None          : ediff_std_dat.append([epoch,ediff_std])
#                     if tail_thresh is not None        : tail_thresh_dat.append([epoch,tail_thresh])
#                     if outlier_intensity is not None  : outlier_intensity_dat.append([epoch,outlier_intensity])

#                     # Subplot 2
#                     axs[1].plot(*np.transpose(ediff_mean_dat), label="ediff_mean_dat")
#                     axs[1].plot(*np.transpose(tail_thresh_dat), label="tail_thresh_dat")
#                     axs[1].axhline(1,linestyle="--")
#                     axs[1].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                     axs[1].set_xlabel("epochs")
#                     axs[1].set_title("EDiff Mean & Tail Threshold")

#                     # Subplot 3
#                     axs[2].plot(*np.transpose(ediff_std_dat), label="ediff_std_dat")
#                     axs[2].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                     axs[2].set_xlabel("epochs") 
#                     axs[2].set_title("EDiff Std")

#                     # Subplot 4
#                     axs[3].plot(*np.transpose(outlier_intensity_dat), label="outlier_intensity_dat")
#                     axs[3].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                     axs[3].set_xlabel("epochs")
#                     axs[3].set_title("Outlier Intensity")

#                 plt.tight_layout()
#                 plt.show()            
            
#             # Check for improvement
#             if val_loss < best_val_loss:
#                 last_saved_model_epoch = epoch+1
#                 best_val_loss = val_loss
#                 best_model_wts = copy.deepcopy(model.state_dict())
#                 torch.save(best_model_wts, save_path)
#                 epochs_without_improvement = 0
#             else:
#                 epochs_without_improvement += 1

#             if log_loss < best_val_loss_f: 
#                 best_log_loss = log_loss
#                 best_log_loss_epoch = epoch
#                 best_model_wts_f = copy.deepcopy(model.state_dict())
#                 torch.save(best_model_wts_f, save_path_f)

#             if class_loss < best_val_loss_c: 
#                 best_class_loss = class_loss
#                 best_class_loss_epoch = epoch
#                 best_model_wts_c = copy.deepcopy(model.state_dict())
#                 torch.save(best_model_wts_c, save_path_c)

#             # Early stopping
#             if epochs_without_improvement >= patience:
#                 print("Early stopping triggered.")
#                 break
            
#             if len(avg_logloss_dat) > 0:
#                 last_vals_str_logloss     = ', '.join(f'{v:.4f}' for v in avg_logloss_dat[-5:])
#                 last_vals_str_class       = ', '.join(f'{v:.4f}' for v in avg_class_loss_dat[-5:])

#             if len(val_logloss_dat) > 0:
#                 val_last_vals_str_logloss = ', '.join(f'{v:.4f}' for v in val_logloss_dat[-5:])
#                 val_last_vals_str_class   = ', '.join(f'{v:.4f}' for v in val_class_loss_dat[-5:])
    
#     # Load best model weights back into the model
#     model.load_state_dict(best_model_wts)
#     print("Training complete. Loaded best model weights.")
#     out_dict =  {"save_dir_path"           : save_dir_path,
#                  "avg_loss_dat":avg_loss_dat,"avg_logloss_dat":avg_logloss_dat,"avg_class_loss_dat":avg_class_loss_dat,
#                  "val_loss_dat":val_loss_dat,"val_logloss_dat":val_logloss_dat,"val_class_loss_dat":val_class_loss_dat,
#                  "ediff_mean_dat":ediff_mean_dat,"ediff_std_dat":ediff_std_dat,"tail_thresh_dat":tail_thresh_dat,"outlier_intensity_dat":outlier_intensity_dat}
#     torch.save(out_dict,out_dict_path)
#     return model, out_dict

# def train_loop_nf( modelf, optimizerf, train_loader, 
#                    epochs=10, val_loader=None, device=None, scheduler=None, clipnorm=None, patience=10, 
#                    save_dir_path="./test_racnnflow", flow_eval_epochs = 10, noisy=True, cnn_mode=False, plot=True, 
#                    n_batch_retry=5, n_epoch_retry=5, psl=False):

#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     modelf.to(device)
#     modelf.train()

#     os.makedirs(save_dir_path, exist_ok=True)
#     save_path_f       = os.path.join(save_dir_path, "best_model.pt")
#     ediff_save_path_f = os.path.join(save_dir_path, "best_model_ediff.pt")
#     out_dict_path_f   = os.path.join(save_dir_path, "out_dict.pt")
    
#     best_val_loss_f = np.inf
#     best_ediff_std = np.inf
#     best_model_wts_f = copy.deepcopy(modelf.state_dict())

#     epochs_without_improvement = 0
    
#     avg_logloss_dat        = []
#     val_logloss_dat        = []
#     ediff_mean_dat         = []
#     ediff_std_dat          = []
#     tail_thresh_dat        = []
#     outlier_intensity_dat  = []
#     last_saved_model_f_epoch = 0

#     progress_bar = tqdm(range(epochs), leave=False)

#     y_main_min, y_main_max = torch.tensor([[np.nanmin(v), np.nanmax(v)] for k,v in train_loader.dataset.flow_outputs.items()]).T.to(device)

#     # Store model states for epoch-level rollback
#     epoch_model_states = {}
#     nval_failed        = 0
    
#     for epoch in progress_bar:
#         # Save current model state at the beginning of each epoch
#         epoch_model_states[epoch] = copy.deepcopy(modelf.state_dict())

#         # Clean up old model states (keep only last n_epoch_retry + 1)
#         if epoch > n_epoch_retry: del epoch_model_states[epoch - n_epoch_retry - 1]
#         if nval_failed > 5: raise RuntimeError(f"More than 5 validation has been skipped... Aborting training")
        
#         # Epoch retry mechanism
#         epoch_success     = False
#         epoch_retry_count = 0
        
#         while not epoch_success and epoch_retry_count <= n_epoch_retry:
#             try:
#                 # If this is a retry, load model from previous epoch
#                 if epoch_retry_count > 0:
#                     rollback_epoch = epoch - epoch_retry_count
#                     if rollback_epoch >= 0 and rollback_epoch in epoch_model_states:
#                         print(f"Epoch {epoch+1} retry {epoch_retry_count}: Rolling back to epoch {rollback_epoch+1} model state")
#                         modelf.load_state_dict(epoch_model_states[rollback_epoch])
#                     else:
#                         print(f"Epoch {epoch+1} retry {epoch_retry_count}: No suitable rollback state available")
#                         modelf.load_state_dict(epoch_model_states[epoch])
                
#                 running_log_loss = 0.0
#                 modelf.train()
#                 successful_batches = 0
                
#                 # Process all batches in this epoch
#                 for batch_idx, (x_cnn, x_aux, y_main, y_neve) in enumerate(train_loader):
                    
#                     #DEBUGGING CODE: COMMENT THIS PART OUT FOR ACTUAL USE
#                     # if batch_idx > 5 : break
#                     #DEBUGGING CODE: COMMENT THIS PART OUT FOR ACTUAL USE


#                     # Batch retry mechanism
#                     batch_success = False
#                     batch_retry_count = 0
                    
#                     x_cnn = x_cnn.to(device).double()
#                     x_aux = x_aux.to(device).double()
#                     y_main = y_main.to(device).double()
#                     y_neve = y_neve.to(device).double()
#                     is_signal = (y_neve.squeeze() > 0).any(dim=1)
                    
#                     if psl:
#                         min_vals    = x_cnn.min(dim=-1, keepdim=True)[0]
#                         max_vals    = x_cnn.max(dim=-1, keepdim=True)[0]
#                         range_vals  = max_vals - min_vals
#                         # Avoid division where range is 0
#                         normalized_x_cnn = torch.where(
#                             range_vals > 0,
#                             (x_cnn - min_vals) / range_vals,
#                             torch.zeros_like(x_cnn)
#                         )
#                         x_cnn            = normalized_x_cnn

                    
#                     # Check for NaNs in input data
#                     if torch.isnan(x_cnn).any(): print(f"WARNING: NaN values detected in x_cnn at epoch {epoch+1}, batch {batch_idx+1}")
#                     if torch.isnan(x_aux).any(): print(f"WARNING: NaN values detected in x_aux at epoch {epoch+1}, batch {batch_idx+1}")
                    
#                     while not batch_success and batch_retry_count < n_batch_retry:
#                         try:
#                             optimizerf.zero_grad()
#                             if cnn_mode  : cond = modelf(x_cnn)
#                             else         : cond = modelf(x_cnn, x_aux)
                            
#                             if noisy:
#                                 # Find rows where all values are NaN
#                                 num_nan_rows = (~is_signal).sum()
                                
#                                 # Generate random values in [x_min, x_max)
#                                 random_vals = y_main_min + (y_main_max - y_main_min) * torch.rand((num_nan_rows.to(device), y_main.to(device).shape[1])).to(device)
                                
#                                 # Replace NaN rows with generated values
#                                 y_main[~is_signal] = random_vals.to(device)
                                
#                                 log_pdf, _, _ = modelf.pdf(y_main, conditional_input=cond)
#                                 log_loss = -log_pdf.mean()

#                                 #DEBUGGING CODE: COMMENT THIS PART OUT FOR ACTUAL USE
#                                 # if batch_idx > 3 and epoch == 2: 
#                                 #     raise ValueError("DEBUGING ERROR")
#                                 #DEBUGGING CODE: COMMENT THIS PART OUT FOR ACTUAL USE


#                             else:
#                                 if is_signal.any():
#                                     log_pdf, _, _ = modelf.pdf(y_main[is_signal], conditional_input=cond[is_signal])
#                                     log_loss      = -log_pdf.mean()
#                                 else:
#                                     log_loss      = torch.tensor(0.0, device=device)
                            
#                             log_loss.backward()
                            
#                             if clipnorm is not None: torch.nn.utils.clip_grad_norm_(modelf.parameters(), clipnorm)
#                             optimizerf.step()
                            
#                             if scheduler: scheduler.step()
                            
#                             running_log_loss   += log_loss.item()
#                             batch_success      = True
#                             successful_batches += 1
                            
#                         except (RuntimeError, ValueError, FloatingPointError, torch.cuda.OutOfMemoryError) as e:
#                             batch_retry_count += 1
                            
#                             print(f"\nError in epoch {epoch+1}, batch {batch_idx+1}, attempt {batch_retry_count}/{n_batch_retry}: {type(e).__name__}: {str(e)}")
                            
#                             # Clear gradients and reset optimizer state for this batch
#                             optimizerf.zero_grad()
                            
#                             # Optional: Clear CUDA cache if CUDA OOM error
#                             if torch.cuda.is_available() and "out of memory" in str(e).lower():
#                                 torch.cuda.empty_cache()
                            
#                             if batch_retry_count >= n_batch_retry:
#                                 print(f"Batch failed after {n_batch_retry} attempts. This will trigger epoch retry.")
#                                 raise e  # Re-raise to trigger epoch retry
#                             else:
#                                 print(f"Retrying batch... {batch_retry_count} time")
                
#                 # If we get here, all batches in the epoch succeeded
#                 epoch_success = True
                
#                 # Calculate average loss for this epoch
#                 if successful_batches > 0  : avg_log_loss = running_log_loss / successful_batches
#                 else                       : avg_log_loss = float('inf')
                    
#                 avg_logloss_dat.append(avg_log_loss)
                
#             except (RuntimeError, ValueError, FloatingPointError, torch.cuda.OutOfMemoryError) as e:
#                 epoch_retry_count += 1
                
#                 print(f"\nEpoch {epoch+1} failed (attempt {epoch_retry_count}/{n_epoch_retry + 1}): {type(e).__name__}: {str(e)}")
                
#                 if epoch_retry_count > n_epoch_retry:
#                     print(f"Epoch {epoch+1} failed after {n_epoch_retry + 1} attempts. Aborting training.")
#                     raise RuntimeError(f"Training failed: Could not complete epoch {epoch+1} after {n_epoch_retry + 1} attempts")
#                 else:
#                     print(f"Retrying epoch {epoch+1}...")
                    
#                 # Clear CUDA cache before retry
#                 if torch.cuda.is_available(): torch.cuda.empty_cache()
        
#         # Validation phase (only if epoch succeeded)
#         if val_loader:
#             if ((epoch+1) % flow_eval_epochs) == 0  : flow_eval = True
#             else                                    : flow_eval = False
            
#             # Add error handling to validation
#             try:
#                 val_log_loss, ediff_mean, ediff_std, tail_thresh, outlier_intensity, _ , _ = validate_loop_nf(modelf, val_loader, device, flow_eval=flow_eval,cnn_mode=cnn_mode,psl=psl)
#             except Exception as e:
#                 print(f"\nValidation error in epoch {epoch+1}: {type(e).__name__}: {str(e)}")
#                 print("Skipping validation for this epoch...")
#                 val_log_loss      = np.inf
#                 ediff_mean        = None
#                 ediff_std         = None
#                 tail_thresh       = None
#                 outlier_intensity = None
#                 nval_failed      += 1
                
#             val_logloss_dat.append(val_log_loss)

#             if ediff_mean is not None        : ediff_mean_dat.append([epoch, ediff_mean])
#             if ediff_std is not None         : ediff_std_dat.append([epoch, ediff_std])
#             if tail_thresh is not None       : tail_thresh_dat.append([epoch, tail_thresh])
#             if outlier_intensity is not None : outlier_intensity_dat.append([epoch, outlier_intensity])
#             if flow_eval:
#                 modelf_eval = copy.deepcopy(modelf.state_dict())
#                 last_ediff_wts = os.path.join(save_dir_path, f"model_ediff_{epoch}.pt")
#                 torch.save(modelf_eval, last_ediff_wts)
            
#             # Plotting code
#             if epoch > 1 and plot:
#                 if flow_eval:
#                     fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns
                    
#                     # Subplot 1
#                     axs[0].plot(avg_logloss_dat, label="avg_logloss_dat")
#                     axs[0].plot(val_logloss_dat, label="val_logloss_dat")
#                     axs[0].legend()
#                     axs[0].set_xlabel("epochs")
#                     axs[0].set_title("Loss")
                    
#                     # Subplot 2
#                     axs[1].plot(*np.transpose(ediff_mean_dat), label="ediff_mean_dat")
#                     axs[1].plot(*np.transpose(tail_thresh_dat), label="tail_thresh_dat")
#                     axs[1].axhline(1, linestyle="--")
#                     axs[1].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                     axs[1].set_xlabel("epochs")
#                     axs[1].set_title("EDiff Mean & Tail Threshold")

#                     # Subplot 3
#                     axs[2].plot(*np.transpose(ediff_std_dat), label="ediff_std_dat")
#                     axs[2].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                     axs[2].set_xlabel("epochs")
#                     axs[2].set_title("EDiff Std")

#                     # Subplot 4
#                     axs[3].plot(*np.transpose(outlier_intensity_dat), label="outlier_intensity_dat")
#                     axs[3].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                     axs[3].set_xlabel("epochs")
#                     axs[3].set_title("Outlier Intensity")

#                     plt.tight_layout()
#                     plt.show()
            
#             # Check for improvement
#             if val_log_loss < best_val_loss_f:
#                 last_saved_model_f_epoch = epoch+1
#                 best_val_loss_f = val_log_loss
#                 best_model_wts_f = copy.deepcopy(modelf.state_dict())
#                 torch.save(best_model_wts_f, save_path_f)
#                 epochs_without_improvement = 0
#             else:
#                 epochs_without_improvement += 1

#             if ediff_std is not None and (ediff_std < best_ediff_std) and (epoch > 2*flow_eval_epochs):
#                 best_ediff_std = ediff_std
#                 torch.save(copy.deepcopy(modelf.state_dict()), ediff_save_path_f)

#             # Early stopping
#             if epochs_without_improvement >= patience:
#                 print("Early stopping triggered.")
#                 break

#         # Progress bar update
#         if (len(avg_logloss_dat) > 0) and (len(val_logloss_dat) > 0):
#             last_vals_str_logloss = ', '.join(f'{v:.4f}' for v in avg_logloss_dat[-5:])
#             val_last_vals_str_logloss = ', '.join(f'{v:.4f}' for v in val_logloss_dat[-5:])
#             progress_bar.set_description(
#                 f"Epoch {epoch+1}/{epochs} | {last_saved_model_f_epoch} ({epochs_without_improvement}/{patience}) | logloss: [{last_vals_str_logloss}] | val_logloss: [{val_last_vals_str_logloss}]"
#             )

#     # Load best model weights back into the model
#     modelf.load_state_dict(best_model_wts_f)
#     print("Training complete. Loaded best model weights.")
    
#     out_dict = {
#         "save_dir_path": save_dir_path,
#         "avg_logloss_dat": avg_logloss_dat,
#         "val_logloss_dat": val_logloss_dat,
#         "ediff_mean_dat": ediff_mean_dat,
#         "ediff_std_dat": ediff_std_dat,
#         "tail_thresh_dat": tail_thresh_dat,
#         "outlier_intensity_dat": outlier_intensity_dat
#     }
#     torch.save(out_dict, out_dict_path_f)
#     return modelf, out_dict

# def train_loop_nf( modelf, optimizerf, train_loader, 
#     epochs=10, val_loader=None, device=None, scheduler=None, clipnorm=None, patience=10, save_dir_path="./test_racnnflow", flow_eval_epochs = 10, noisy=True, cnn_mode=False, plot=True):

#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     modelf.to(device)
#     modelf.train()

#     os.makedirs(save_dir_path, exist_ok=True)
#     save_path_f = os.path.join(save_dir_path, "best_model.pt")
#     ediff_save_path_f = os.path.join(save_dir_path, "best_model_ediff.pt")
#     out_dict_path_f = os.path.join(save_dir_path, "out_dict.pt")
    

    
#     best_val_loss_f = np.inf
#     best_ediff_std = np.inf
#     best_model_wts_f = copy.deepcopy(modelf.state_dict())

#     epochs_without_improvement = 0
    
#     avg_logloss_dat        = []
#     val_logloss_dat        = []
#     ediff_mean_dat         = []
#     ediff_std_dat          = []
#     tail_thresh_dat        = []
#     outlier_intensity_dat  = []
#     last_saved_model_f_epoch = 0

#     progress_bar = tqdm(range(epochs), leave=False)

#     y_main_min, y_main_max = torch.tensor([[np.nanmin(v), np.nanmax(v)] for k,v in train_loader.dataset.flow_outputs.items()]).T.to(device)


#     for epoch in progress_bar:
#         running_log_loss   = 0.0 
#         modelf.train()
#         for x_cnn, x_aux, y_main, y_neve in train_loader:
#             x_cnn      = x_cnn.to(device).double()
#             x_aux      = x_aux.to(device).double()
#             y_main     = y_main.to(device).double()
#             y_neve     = y_neve.to(device).double()
#             is_signal  = (y_neve.squeeze() > 0).any(dim=1)
            
#             optimizerf.zero_grad()
#             if cnn_mode:
#                 cond    = modelf(x_cnn)
#             else:
#                 cond    = modelf(x_cnn, x_aux)
            
#             if noisy:
#                 # Find rows where all values are NaN
#                 #nan_mask     = torch.isnan(y_main).all(dim=1).to(device)
#                 num_nan_rows = (~is_signal).sum()

#                 # Generate random values in [x_min, x_max)
#                 random_vals = y_main_min + (y_main_max - y_main_min) * torch.rand((num_nan_rows.to(device), y_main.to(device).shape[1])).to(device)

#                 # Replace NaN rows with generated values
#                 y_main[~is_signal] = random_vals.to(device)
#                 log_pdf,_,_  = modelf.pdf(y_main, conditional_input=cond)
#                 log_loss     = -log_pdf.mean() 
#             else:
#                 if is_signal.any():    
#                     log_pdf,_,_  = modelf.pdf(y_main[is_signal], conditional_input=cond[is_signal])
#                     log_loss     = -log_pdf.mean() 
#                 else:
#                     log_loss = torch.tensor(0.0, device=device)
            
#             log_loss.backward()
            
#             if clipnorm is not None:torch.nn.utils.clip_grad_norm_(modelf.parameters(), clipnorm)
#             optimizerf.step()

#             if scheduler: scheduler.step()    
            
#             running_log_loss   += log_loss.item()
             

#         avg_log_loss     = running_log_loss/ len(train_loader)
#         avg_logloss_dat.append(avg_log_loss)

#         if val_loader:    
#             if ((epoch+1) % flow_eval_epochs) == 0 : flow_eval = True
#             else                                   : flow_eval = False
            
#             val_log_loss, ediff_mean, ediff_std, tail_thresh, outlier_intensity, _ , _ = validate_loop_nf(modelf, val_loader, device, flow_eval=flow_eval,cnn_mode=cnn_mode)
#             val_logloss_dat.append(val_log_loss)

#             if ediff_mean is not None         : ediff_mean_dat.append([epoch,ediff_mean])
#             if ediff_std is not None          : ediff_std_dat.append([epoch,ediff_std])
#             if tail_thresh is not None        : tail_thresh_dat.append([epoch,tail_thresh])
#             if outlier_intensity is not None  : outlier_intensity_dat.append([epoch,outlier_intensity])
#             if flow_eval                      :
#                 modelf_eval = copy.deepcopy(modelf.state_dict())
#                 torch.save(modelf_eval, os.path.join(save_dir_path, f"model_ediff_{epoch}.pt"))
            
#             if epoch > 1 and plot:
                        
#                 if flow_eval:
#                     fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns
                    
#                     # Subplot 1
#                     axs[0].plot(avg_logloss_dat,    label="avg_logloss_dat")
#                     axs[0].plot(val_logloss_dat,    label="val_logloss_dat")
#                     axs[0].legend()
#                     axs[0].set_xlabel("epochs")
#                     axs[0].set_title("Loss")
#                     # Subplot 2
#                     axs[1].plot(*np.transpose(ediff_mean_dat), label="ediff_mean_dat")
#                     axs[1].plot(*np.transpose(tail_thresh_dat), label="tail_thresh_dat")
#                     axs[1].axhline(1,linestyle="--")
#                     axs[1].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                     axs[1].set_xlabel("epochs")
#                     axs[1].set_title("EDiff Mean & Tail Threshold")
    
#                     # Subplot 3
#                     axs[2].plot(*np.transpose(ediff_std_dat), label="ediff_std_dat")
#                     axs[2].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                     axs[2].set_xlabel("epochs")
#                     axs[2].set_title("EDiff Std")
    
#                     # Subplot 4
#                     axs[3].plot(*np.transpose(outlier_intensity_dat), label="outlier_intensity_dat")
#                     axs[3].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                     axs[3].set_xlabel("epochs")
#                     axs[3].set_title("Outlier Intensity")
    
#                     plt.tight_layout()
#                     plt.show()            
            
#             # Check for improvement            
#             if val_log_loss < best_val_loss_f:
#                 last_saved_model_f_epoch = epoch+1
#                 best_val_loss_f = val_log_loss
#                 best_model_wts_f = copy.deepcopy(modelf.state_dict())
#                 torch.save(best_model_wts_f, save_path_f)
#                 epochs_without_improvement = 0
#             else:
#                 epochs_without_improvement += 1


#             if ediff_std is not None and (ediff_std < best_ediff_std) and (epoch > 2*flow_eval_epochs):
#                 best_ediff_std = ediff_std
#                 torch.save(copy.deepcopy(modelf.state_dict()), ediff_save_path_f)
 
#             # Early stopping
#             if epochs_without_improvement >= patience:
#                 print("Early stopping triggered.")
#                 break

#         if (len(avg_logloss_dat) > 0 ) and (len(val_logloss_dat) > 0):
#             last_vals_str_logloss = ', '.join(f'{v:.4f}' for v in avg_logloss_dat[-5:])
#             val_last_vals_str_logloss = ', '.join(f'{v:.4f}' for v in val_logloss_dat[-5:])
#             progress_bar.set_description(
#             f"Epoch {epoch+1}/{epochs} | {last_saved_model_f_epoch} ({epochs_without_improvement}/{patience}) | logloss: [{last_vals_str_logloss}]  | val_logloss: [{val_last_vals_str_logloss}] "
#             )

#     # Load best model weights back into the model
#     modelf.load_state_dict(best_model_wts_f)
#     print("Training complete. Loaded best model weights.")
#     out_dict =  {"save_dir_path"           : save_dir_path         ,        
#                  "avg_logloss_dat"         : avg_logloss_dat       ,
#                  "val_logloss_dat"         : val_logloss_dat       ,
#                  "ediff_mean_dat"          : ediff_mean_dat        ,
#                  "ediff_std_dat"           : ediff_std_dat         ,
#                  "tail_thresh_dat"         : tail_thresh_dat       ,
#                  "outlier_intensity_dat"   : outlier_intensity_dat }
#     torch.save(out_dict,out_dict_path_f)
#     return modelf,out_dict

# def validate_loop_nf(modelf,  val_loader, device,verbose=False,samplesize=100,flow_eval=True,cnn_mode=False,plot=True):
#     modelf.eval()
    
#     total_log_loss   = 0.0
#     ediff_mean          = None      
#     ediff_std           = None      
#     tail_thresh         = None      
#     outlier_intensity = None 
    
#     lpdf = torch.tensor([]).to(device)
#     ent =  torch.tensor([]).to(device)

#     data_iter = tqdm(val_loader, desc="Validating", leave=False) #if verbose else val_loader
#     with torch.no_grad():
#         for x_cnn, x_aux, y_main, y_neve in data_iter:
#             # Each of these is a dict of tensors
#             x_cnn      = x_cnn.to(device).double()
#             x_aux      = x_aux.to(device).double()
#             y_main     = y_main.to(device).double()
#             y_neve     = y_neve.to(device).double()

#             is_signal = (y_neve.squeeze() > 0)
            
#             # raise KeyError
#             # Forward pass
#             if cnn_mode:
#                 cond   = modelf(x_cnn)  # shape: [batch, cond_dim]
#             else:
#                 cond   = modelf(x_cnn, x_aux)  # shape: [batch, cond_dim]
           
#             if is_signal.any():    
#                 log_pdf,_,_  = modelf.pdf(y_main[is_signal], conditional_input=cond[is_signal])
#                 log_loss     = -log_pdf.mean() 
#             else:
#                 log_loss = torch.tensor(0.0, device=device)
                
#             if flow_eval:    
#                 entropy_dict  = modelf.pdf.entropy(conditional_input=cond[is_signal],samplesize=samplesize)
#                 lpdf = torch.hstack([lpdf,log_pdf])
#                 ent =  torch.hstack([ent,entropy_dict["total"]])
#             total_log_loss    += log_loss.item()

#     avg_log_loss    = total_log_loss / len(val_loader)
#     tail_percentile = 95
#     intensity_percentile = 95
    
#     if flow_eval:    

#         ediff             = (-lpdf-ent).cpu().numpy()
#         ediff_mean        = ediff.mean()
#         ediff_std         = ediff.std()
#         tail_thresh       = np.percentile(ediff, tail_percentile)
#         tail_prop         = np.sum(ediff > tail_thresh) / len(ediff)
#         absediff          = np.abs(ediff)
#         T                 = np.percentile(absediff, intensity_percentile)
#         excess            = np.sum([x - T for x in absediff if x > T])
#         outlier_intensity = excess / np.sum(absediff) if np.sum(absediff) > 0 else 0
#         if verbose:
#             print(f"Validation Log Loss  = {avg_log_loss:.4f}")
#             print(f"ediff.mean()         = {ediff.mean():.4f}")
#             print(f"ediff.std()          = {ediff.std():.4f}")
#             print(f"tail_thresh          = {tail_thresh:.4f}")
#             print(f"outlier_intensity    = {outlier_intensity:.4f}")
#         if plot:
#             plt.hist2d(-lpdf.cpu(),ent.cpu(),bins=50,norm=LogNorm())
#             plt.xlabel("loss")
#             plt.ylabel("ent")
#             plt.show()
  
#     return avg_log_loss,  ediff_mean, ediff_std, tail_thresh, outlier_intensity, -lpdf, ent

# def train_loop_merged_model(model,optimizer,train_loader,
#     epochs=10,val_loader=None,device=None,scheduler=None,clipnorm=None,patience=10,save_dir_path="./test_racnnclassiflow",warmup_epochs = 10 ,flow_eval_epochs = 10,cnn_mode=False,plot=True,noisy=True):
    
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.train()

#     os.makedirs(save_dir_path, exist_ok=True)
#     save_path    = os.path.join(save_dir_path, "best_model.pt")
#     save_path_f  = os.path.join(save_dir_path, "best_model_f.pt")
#     save_path_c  = os.path.join(save_dir_path, "best_model_c.pt")
#     ediff_save_path = os.path.join(save_dir_path, "best_model_ediff.pt")
#     out_dict_path = os.path.join(save_dir_path, "out_dict.pt")
    
    
#     best_val_loss   = np.inf
#     best_val_loss_f = np.inf
#     best_val_loss_c = np.inf
#     best_model_wts_f = copy.deepcopy(model.state_dict())
#     best_model_wts_c = copy.deepcopy(model.state_dict())
#     epochs_without_improvement = 0
#     criterion = nn.BCEWithLogitsLoss()
    
#     avg_loss_dat           = []
#     avg_logloss_dat        = []
#     avg_class_loss_dat     = []
#     val_loss_dat           = []
#     val_logloss_dat        = []
#     val_class_loss_dat     = []
#     ediff_mean_dat         = []
#     ediff_std_dat          = []
#     tail_thresh_dat        = []
#     outlier_intensity_dat  = []
#     last_saved_model_epoch = 0
#     best_log_loss_epoch = 0
#     best_class_loss_epoch = 0

#     progress_bar           = tqdm(range(epochs), leave=False)
#     y_main_min, y_main_max = torch.tensor([[np.nanmin(v), np.nanmax(v)] for k,v in train_loader.dataset.flow_outputs.items()]).T.to(device)

#     for epoch in progress_bar:
#         model.train()
#         running_loss       = 0.0 
#         running_log_loss   = 0.0 
#         running_class_loss = 0.0 

#         for x_cnn, x_aux, y_main, y_neve in train_loader:
#             model.train()
#             #x_cnn      = torch.hstack(list(x_cnn.values())).unsqueeze(1).to(device)
#             #x_aux      = torch.vstack(list(x_aux.values())).T.to(device)
#             #y_main     = torch.vstack(list(y_main.values())).T.to(device)
#             #y_neve     = torch.hstack(list(y_neve.values())).unsqueeze(1).to(device)
#             x_cnn      = x_cnn.double().to(device)
#             x_aux      = x_aux.double().to(device)
#             y_main     = y_main.double().to(device)
#             y_neve     = y_neve.double().to(device)
#             is_signal  = (y_neve.squeeze() > 0).any(dim=1)
            
#             optimizer.zero_grad()
#             if cnn_mode:
#                 cond,nclass    = model(x_cnn)
#             else:
#                 cond,nclass    = model(x_cnn, x_aux)

#             log_loss     = torch.tensor(0.0, device=device)
#             flow_weight  = 0 

#             if epoch >= warmup_epochs : 
#                 flow_weight = 1
#                 if noisy:
#                     num_nan_rows = (~is_signal).sum()
#                     random_vals = y_main_min + (y_main_max - y_main_min) * torch.rand((num_nan_rows.to(device), y_main.to(device).shape[1])).to(device)
#                     y_main[~is_signal] = random_vals.to(device)

#                     log_pdf,_,_  =  model.pdf(y_main, conditional_input=cond)
#                     log_loss     = -log_pdf.mean() 
#                 else:    
#                     if is_signal.any():    
#                         log_pdf,_,_  = model.pdf(y_main[is_signal], conditional_input=cond[is_signal])
#                         log_loss     = -log_pdf.mean() 
#                     else:
#                         log_loss = torch.tensor(0.0, device=device)
                
#             class_loss   = criterion(nclass , is_signal.double().unsqueeze(1))
#             loss         = flow_weight *log_loss + class_loss
#             loss.backward()

#             if clipnorm is not None:torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            
#             optimizer.step()
#             if scheduler: scheduler.step()    
            
#             running_loss       += loss.item()
#             running_log_loss   += log_loss.item()
#             running_class_loss += class_loss.item()

#         if (len(avg_logloss_dat) > 0 )and (len(val_logloss_dat) > 0):
#             progress_bar.set_description(
#             f"Epoch {epoch+1}/{epochs} | {last_saved_model_epoch}:({best_log_loss_epoch},{best_class_loss_epoch}) | logloss: [{last_vals_str_logloss}] | classloss: [{last_vals_str_class}] | val_logloss: [{val_last_vals_str_logloss}] | val_classloss: [{val_last_vals_str_class}]"
#             )
        
#         avg_loss         = running_loss / len(train_loader)
#         avg_log_loss     = running_log_loss/ len(train_loader)
#         avg_class_loss   = running_class_loss/ len(train_loader)

#         avg_loss_dat.append(avg_loss)
#         avg_logloss_dat.append(avg_log_loss)
#         avg_class_loss_dat.append(avg_class_loss)
  
#         if val_loader:
            
#             if (epoch >=warmup_epochs) and (((epoch+1) % flow_eval_epochs) == 0) : flow_eval = True
#             else                                                                 : flow_eval = False
#             val_loss, log_loss,class_loss, ediff_mean, ediff_std, tail_thresh, outlier_intensity, _ , _ = validate_loop_merged_model(model, val_loader, device,flow_loss_eval=(epoch >=warmup_epochs), flow_eval=flow_eval,cnn_mode=cnn_mode,plot=plot)
            
#             val_loss_dat.append(val_loss)
#             val_logloss_dat.append(log_loss)
#             val_class_loss_dat.append(class_loss)
            
#             if flow_eval                      :
#                 modelf_eval = copy.deepcopy(model.state_dict())
#                 torch.save(modelf_eval, os.path.join(save_dir_path, f"model_ediff_{epoch}.pt"))
                        
#             if epoch > 1 and flow_eval and plot:
#                 fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns
                
#                 # Subplot 1
#                 axs[0].plot(avg_loss_dat,       label="avg_loss_dat")
#                 axs[0].plot(avg_logloss_dat,    label="avg_logloss_dat")
#                 axs[0].plot(avg_class_loss_dat, label="avg_class_loss_dat")
#                 axs[0].plot(val_loss_dat,       label="val_loss_dat")
#                 axs[0].plot(val_logloss_dat,    label="val_logloss_dat")
#                 axs[0].plot(val_class_loss_dat, label="val_class_loss_dat")
#                 #axs[0].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                 axs[0].legend()
#                 axs[0].set_xlabel("epochs")
#                 axs[0].set_title("Loss")
#                 if epoch >=warmup_epochs and flow_eval:
#                     if ediff_mean is not None         : ediff_mean_dat.append([epoch,ediff_mean])
#                     if ediff_std is not None          : ediff_std_dat.append([epoch,ediff_std])
#                     if tail_thresh is not None        : tail_thresh_dat.append([epoch,tail_thresh])
#                     if outlier_intensity is not None  : outlier_intensity_dat.append([epoch,outlier_intensity])

#                     # Subplot 2
#                     axs[1].plot(*np.transpose(ediff_mean_dat), label="ediff_mean_dat")
#                     axs[1].plot(*np.transpose(tail_thresh_dat), label="tail_thresh_dat")
#                     axs[1].axhline(1,linestyle="--")
#                     axs[1].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                     axs[1].set_xlabel("epochs")
#                     axs[1].set_title("EDiff Mean & Tail Threshold")

#                     # Subplot 3
#                     axs[2].plot(*np.transpose(ediff_std_dat), label="ediff_std_dat")
#                     axs[2].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                     axs[2].set_xlabel("epochs")
#                     axs[2].set_title("EDiff Std")

#                     # Subplot 4
#                     axs[3].plot(*np.transpose(outlier_intensity_dat), label="outlier_intensity_dat")
#                     axs[3].legend(loc="upper right", bbox_to_anchor=(1, 1))
#                     axs[3].set_xlabel("epochs")
#                     axs[3].set_title("Outlier Intensity")

#                 plt.tight_layout()
#                 plt.show()            
            
#             # Check for improvement
#             if val_loss < best_val_loss:
#                 last_saved_model_epoch = epoch+1
#                 best_val_loss = val_loss
#                 best_model_wts = copy.deepcopy(model.state_dict())
#                 torch.save(best_model_wts, save_path)
#                 epochs_without_improvement = 0
#             else:
#                 epochs_without_improvement += 1

#             if log_loss < best_val_loss_f: 
#                 best_log_loss = log_loss
#                 best_log_loss_epoch = epoch
#                 best_model_wts_f = copy.deepcopy(model.state_dict())
#                 torch.save(best_model_wts_f, save_path_f)

#             if class_loss < best_val_loss_c: 
#                 best_class_loss = class_loss
#                 best_class_loss_epoch = epoch
#                 best_model_wts_c = copy.deepcopy(model.state_dict())
#                 torch.save(best_model_wts_c, save_path_c)

#             # Early stopping
#             if epochs_without_improvement >= patience:
#                 print("Early stopping triggered.")
#                 break
            
#             if len(avg_logloss_dat) > 0:
#                 last_vals_str_logloss     = ', '.join(f'{v:.4f}' for v in avg_logloss_dat[-5:])
#                 last_vals_str_class       = ', '.join(f'{v:.4f}' for v in avg_class_loss_dat[-5:])

#             if len(val_logloss_dat) > 0:
#                 val_last_vals_str_logloss = ', '.join(f'{v:.4f}' for v in val_logloss_dat[-5:])
#                 val_last_vals_str_class   = ', '.join(f'{v:.4f}' for v in val_class_loss_dat[-5:])
    
#     # Load best model weights back into the model
#     model.load_state_dict(best_model_wts)
#     print("Training complete. Loaded best model weights.")
#     out_dict =  {"save_dir_path"           : save_dir_path,
#                  "avg_loss_dat":avg_loss_dat,"avg_logloss_dat":avg_logloss_dat,"avg_class_loss_dat":avg_class_loss_dat,
#                  "val_loss_dat":val_loss_dat,"val_logloss_dat":val_logloss_dat,"val_class_loss_dat":val_class_loss_dat,
#                  "ediff_mean_dat":ediff_mean_dat,"ediff_std_dat":ediff_std_dat,"tail_thresh_dat":tail_thresh_dat,"outlier_intensity_dat":outlier_intensity_dat}
#     torch.save(out_dict,out_dict_path)
#     return model, out_dict
 
# def validate_loop_merged_model(model, val_loader, device,verbose=False,samplesize=100,flow_loss_eval=True,flow_eval=True,cnn_mode=False,plot=True):
#     criterion = nn.BCEWithLogitsLoss()

#     model.eval()

#     total_loss        = 0.0
#     total_log_loss    = 0.0
#     total_class_loss  = 0.0
#     ediff_mean        = None      
#     ediff_std         = None      
#     tail_thresh       = None      
#     outlier_intensity = None 
    
#     lpdf = torch.tensor([]).to(device)
#     ent =  torch.tensor([]).to(device)

#     data_iter = tqdm(val_loader, desc="Validating", leave=False) #if verbose else val_loader
#     with torch.no_grad():
#         for x_cnn, x_aux, y_main, y_neve in data_iter:

#             # Each of these is a dict of tensors
#             x_cnn  = x_cnn.double().to(device) #    torch.hstack(list(x_cnn.values())).unsqueeze(1).to(device)
#             x_aux  = x_aux.double().to(device) #    torch.vstack(list(x_aux.values())).T.to(device)
#             y_main = y_main.double().to(device) #   torch.vstack(list(y_main.values())).T.to(device)
#             y_neve = y_neve.double().to(device) #   torch.hstack(list(y_neve.values())).unsqueeze(1).to(device)

#             is_signal = (y_neve.squeeze() > 0)

#             # Forward pass
#             if cnn_mode:
#                 cond,nclass    = model(x_cnn)
#             else:
#                 cond,nclass    = model(x_cnn, x_aux)
            
#             flow_weight  = 0 
#             log_loss = torch.tensor(0.0, device=device)
#             if flow_loss_eval :
#                 flow_weight = 1
#                 if is_signal.any():    
#                     log_pdf,_,_  = model.pdf(y_main[is_signal], conditional_input=cond[is_signal])
#                     log_loss     = -log_pdf.mean() 
#                 else:
#                     log_loss = torch.tensor(0.0, device=device)

#                 if flow_eval:
#                     entropy_dict  = model.pdf.entropy(conditional_input=cond[is_signal],samplesize=samplesize)
#                     lpdf = torch.hstack([lpdf,log_pdf])
#                     ent =  torch.hstack([ent,entropy_dict["total"]])

#             class_loss         = criterion(nclass.squeeze(), is_signal.double())
#             total_log_loss    += log_loss.item()
#             total_class_loss  += class_loss.item()
#             total_loss        += flow_weight*log_loss.item() + class_loss.item()


#     avg_val_loss = total_loss / len(val_loader)
#     avg_log_loss  = total_log_loss / len(val_loader)
#     avg_class_loss  = total_class_loss / len(val_loader)
#     ediff_mean         = None         
#     ediff_std          = None         
#     tail_thresh        = None         
#     outlier_intensity  = None                 

    
#     if flow_eval:
#         tail_percentile = 95
#         intensity_percentile = 95

#         ediff             = (-lpdf-ent).cpu().numpy()
#         ediff_mean        = ediff.mean()
#         ediff_std         = ediff.std()
#         tail_thresh       = np.percentile(ediff, tail_percentile)
#         tail_prop         = np.sum(ediff > tail_thresh) / len(ediff)
#         absediff          = np.abs(ediff)
#         T                 = np.percentile(absediff, intensity_percentile)
#         excess            = np.sum([x - T for x in absediff if x > T])
#         outlier_intensity = excess / np.sum(absediff) if np.sum(absediff) > 0 else 0

#         if verbose:
            
#             print(f"Validation Loss      = {avg_val_loss:.4f}")
#             print(f"Validation Log Loss  = {avg_log_loss:.4f}")
#             print(f"Validation Class Loss  = {avg_class_loss:.4f}")
#             print(f"ediff.mean()         = {ediff.mean():.4f}")
#             print(f"ediff.std()          = {ediff.std():.4f}")
#             print(f"tail_thresh          = {tail_thresh:.4f}")
#             print(f"outlier_intensity    = {outlier_intensity:.4f}")
#         if plot:
#             plt.hist2d(-lpdf.cpu(),ent.cpu(),bins=50,norm=LogNorm())
#             plt.xlabel("loss")
#             plt.ylabel("ent")
#             plt.show()
#     #model.train()

#     return avg_val_loss,avg_log_loss,avg_class_loss,ediff_mean, ediff_std, tail_thresh, outlier_intensity,-lpdf,ent
