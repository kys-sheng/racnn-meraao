import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
import time
import os
import sys
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
import pickle
import jammy_flows as jf 
from tqdm import tqdm
seed_value = 42
import pandas as pd
tqdm.pandas()
from copy import deepcopy
import tarfile 
import io

def save_pkl(stuff,name):
    with open(name, 'wb')  as file: 
        pickle.dump(stuff  , file)
    return      

def load_pkl(name):
    with open(name, 'rb') as file: 
        return pickle.load(file)       

def load_model(modelobj,name,device):
    model = (modelobj).to(device)
    model.load_state_dict(torch.load(name))  
    return model

def load_pkl_from_targz(tar_gz_path, pickle_filename):
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        with tar.extractfile(pickle_filename) as file:
            return pickle.load(file)


def save_df_to_tarball(df: pd.DataFrame, tarball_path: str, pkl_name: str = "data.pkl"):
    """
    Pickles a DataFrame and saves it inside a .tar.gz archive.

    Parameters:
    - df: DataFrame to save.
    - tarball_path: Output path for the tar.gz archive.
    - pkl_name: Name of the pickle file inside the archive.
    """
    with tarfile.open(tarball_path, "w:gz") as tar:
        buffer = io.BytesIO()
        pickle.dump(df, buffer)
        buffer.seek(0)

        tarinfo = tarfile.TarInfo(name=pkl_name)
        tarinfo.size = len(buffer.getbuffer())
        tar.addfile(tarinfo, buffer)

def load_df_from_tarball(tarball_path: str, pkl_name: str = "data.pkl") -> pd.DataFrame:
    """
    Loads a pickled DataFrame from a .tar.gz archive.

    Parameters:
    - tarball_path: Path to the tar.gz archive.
    - pkl_name: Name of the pickle file inside the archive.

    Returns:
    - A pandas DataFrame.
    """
    with tarfile.open(tarball_path, "r:gz") as tar:
        member = tar.getmember(pkl_name)
        f = tar.extractfile(member)
        if f is not None:
            return pickle.load(f)
        else:
            raise FileNotFoundError(f"{pkl_name} not found in archive.")            
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

def sg_sim(mu,sigma,nevents):
    """
    mu: global theta (param of interest)
    sigma: global theta_nu (global nuisance)
    x: observed data
    """
    return [np.random.normal(mu,sigma) for i in range(nevents)]

def bg_sim(nevents,mu=4):
    """
    mu: global theta (param of interest)
    sigma: global theta_nu (global nuisance)
    x: observed data
    """
    return np.random.default_rng().exponential(scale=mu, size=nevents)

XRANGE           = [0,30]   # range of x or observable range
MIN_WINDOW       = 2.5 # minimum window size or resolution
NSG              = 10000
NBG              = 50000
NBINS            = 100
YSCALE           = "linear"
NOISE_TOLERANCE  = 0.1
MEAN             = np.arange(2,28+0.1,0.5)
#MEAN             = np.arange(2,28+0.1,2)
SIGMA            = np.arange(1,5+0.1,0.5)
#SIGMA            = np.arange(1,5+0.1,2)


def generate_data(mean,sigma,nsg=NSG,nbg=NBG):
    sg=sg_sim(mean,sigma,nsg)
    bg=bg_sim(nbg)
    return sg, bg

def ensemble_data(sg,bg,xrange=XRANGE,nbins=NBINS,yscale=YSCALE,plot=False):
    hist_1, _ = np.histogram(sg, bins=nbins, range=xrange)
    hist_2, _ = np.histogram(bg, bins=nbins, range=xrange)
    hist_both = hist_1+hist_2

    if yscale =="log":
        hist_both,_ = np.histogram(np.append(sg,bg), bins=nbins, range=xrange)
        hist_1 = np.log1p(hist_1)
        hist_2 = np.log1p(hist_2)
        hist_both = np.log1p(hist_both)

    if plot:
            plt.hist([sg,bg],bins=nbins,alpha=0.4,color=["g","b"],label=["sg","bg"],range=xrange,stacked=True)
            plt.yscale(yscale)
            plt.xlim(xrange)
            if yscale =="log":
                plt.plot(np.arange(xrange[0],xrange[1],(xrange[1]-xrange[0])/nbins),np.exp(hist_both),alpha=1)
                plt.plot(np.arange(xrange[0],xrange[1],(xrange[1]-xrange[0])/nbins),np.exp(hist_1),alpha=1)
                plt.plot(np.arange(xrange[0],xrange[1],(xrange[1]-xrange[0])/nbins),np.exp(hist_2),alpha=1)
            else:
                plt.plot(np.arange(xrange[0],xrange[1],(xrange[1]-xrange[0])/nbins),(hist_both),alpha=1)
                plt.plot(np.arange(xrange[0],xrange[1],(xrange[1]-xrange[0])/nbins),(hist_1),alpha=1)
                plt.plot(np.arange(xrange[0],xrange[1],(xrange[1]-xrange[0])/nbins),(hist_2),alpha=1)
            plt.title(str(xrange))
            plt.show()
            
            
    return hist_1, hist_2, hist_both

def generate_sys_cuts(xrange,min_window):
    cuts_list=[]

    # include maximum range
    cuts_list.append(xrange)
    
    # Calcuate how many slices can be made
    k=(xrange[1]-xrange[0])/min_window

    # Generate cuts
    for isize in range(1,int(k)):
        krange_0 = xrange[0]
        while krange_0+isize*min_window <= xrange[1]:
            cuts_list.append([krange_0,krange_0+isize*min_window])
            krange_0 = krange_0+min_window
    return cuts_list

CUTS = generate_sys_cuts(XRANGE,MIN_WINDOW)

def generate_data_systematically():
    print("------Generate dataset systematically------")
    sys_gen_dat_lin = []
    sys_gen_dat_log = []
    for mean,sigma in tqdm(list(itertools.product(MEAN,SIGMA))):
        sg,bg = generate_data(mean,sigma)
        # Systematic augmentation
        for icuts in CUTS:
            hist_sg, hist_bg, hist_both = ensemble_data(sg,bg,yscale="linear",xrange=icuts)
            sys_gen_dat_lin.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"SYS_GEN","SYS_AUG"])

            hist_sg, hist_bg, hist_both =  ensemble_data(sg,bg,yscale="log",xrange=icuts)
            sys_gen_dat_log.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"SYS_GEN","SYS_AUG"])

        # Generate random cuts
        rand_cuts_all= []
        while len(rand_cuts_all ) < len(CUTS):
            rand_cuts = [(np.random.uniform(0,30),np.random.uniform(0,30)) for _ in range(len(CUTS)-len(rand_cuts_all ))]
            rand_cuts = [i for i in rand_cuts if (i[0] < i[1] and i[1]-i[0] > MIN_WINDOW)]    
            rand_cuts_all = rand_cuts_all + rand_cuts
        assert len(CUTS) == len(rand_cuts_all)

        for icuts in rand_cuts_all:
            hist_sg, hist_bg, hist_both = ensemble_data(sg,bg,yscale="linear",xrange=icuts)
            sys_gen_dat_lin.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"SYS_GEN","RAND_AUG"])

            hist_sg, hist_bg, hist_both =  ensemble_data(sg,bg,yscale="log",xrange=icuts)
            sys_gen_dat_log.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"SYS_GEN","RAND_AUG"])
    return sys_gen_dat_lin,sys_gen_dat_log

def generate_data_systematically_mod_wbgonly():
    print("------Generate dataset systematically------")
    sys_gen_dat_lin = []
    sys_gen_dat_log = []
    for mean,sigma in tqdm(list(itertools.product(MEAN,SIGMA))):
        sg,bg = generate_data(mean,sigma)
        
        # Systematic augmentation
        for icuts in CUTS:
            hist_sg, hist_bg, hist_both = ensemble_data(sg,bg,yscale="linear",xrange=icuts)
            sys_gen_dat_lin.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"SYS_GEN","SYS_AUG"])
            
            hist_sg, hist_bg, hist_both = ensemble_data([],bg,yscale="linear",xrange=icuts)
            sys_gen_dat_lin.append([None,None,hist_sg, hist_bg, hist_both,icuts,"SYS_GEN","SYS_AUG"])

            hist_sg, hist_bg, hist_both =  ensemble_data(sg,bg,yscale="log",xrange=icuts)
            sys_gen_dat_log.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"SYS_GEN","SYS_AUG"])

            hist_sg, hist_bg, hist_both =  ensemble_data([],bg,yscale="log",xrange=icuts)
            sys_gen_dat_log.append([None,None,hist_sg, hist_bg, hist_both,icuts,"SYS_GEN","SYS_AUG"])

        # Generate random cuts
        rand_cuts_all= []
        while len(rand_cuts_all ) < len(CUTS):
            rand_cuts = [(np.random.uniform(0,30),np.random.uniform(0,30)) for _ in range(len(CUTS)-len(rand_cuts_all ))]
            rand_cuts = [i for i in rand_cuts if (i[0] < i[1] and i[1]-i[0] > MIN_WINDOW)]    
            rand_cuts_all = rand_cuts_all + rand_cuts
        assert len(CUTS) == len(rand_cuts_all)

        for icuts in rand_cuts_all:
            hist_sg, hist_bg, hist_both = ensemble_data(sg,bg,yscale="linear",xrange=icuts)
            sys_gen_dat_lin.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"SYS_GEN","RAND_AUG"])
            
            hist_sg, hist_bg, hist_both = ensemble_data([],bg,yscale="linear",xrange=icuts)
            sys_gen_dat_lin.append([None,None,hist_sg, hist_bg, hist_both,icuts,"SYS_GEN","RAND_AUG"])

            hist_sg, hist_bg, hist_both =  ensemble_data(sg,bg,yscale="log",xrange=icuts)
            sys_gen_dat_log.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"SYS_GEN","RAND_AUG"])

            hist_sg, hist_bg, hist_both =  ensemble_data([],bg,yscale="log",xrange=icuts)
            sys_gen_dat_log.append([None,None,hist_sg, hist_bg, hist_both,icuts,"SYS_GEN","RAND_AUG"])
    return sys_gen_dat_lin,sys_gen_dat_log

def generate_data_randomly():
    rand_gen_dat_lin = []
    rand_gen_dat_log = []
    rand_dat_mu_sig = [(np.random.uniform(2.0, 28.0),np.random.uniform(1.0, 5.0)) for _ in range(len(list(itertools.product(MEAN,SIGMA))))]
    print("------Generate dataset randomly------")
    for mean,sigma in tqdm(rand_dat_mu_sig):
        sg,bg = generate_data(mean,sigma)
        # Systematic augmentation
        for icuts in CUTS:
            hist_sg, hist_bg, hist_both = ensemble_data(sg,bg,yscale="linear",xrange=icuts)
            rand_gen_dat_lin.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"RAND_GEN","SYS_AUG"])

            hist_sg, hist_bg, hist_both =  ensemble_data(sg,bg,yscale="log",xrange=icuts)
            rand_gen_dat_log.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"RAND_GEN","SYS_AUG"])

        # Generate random cuts
        rand_cuts_all= []
        while len(rand_cuts_all ) < len(CUTS):
            rand_cuts = [(np.random.uniform(0,30),np.random.uniform(0,30)) for _ in range(len(CUTS)-len(rand_cuts_all ))]
            rand_cuts = [i for i in rand_cuts if (i[0] < i[1] and i[1]-i[0] > MIN_WINDOW)]    
            rand_cuts_all = rand_cuts_all + rand_cuts
        assert len(CUTS) == len(rand_cuts_all)

        for icuts in rand_cuts_all:
            hist_sg, hist_bg, hist_both = ensemble_data(sg,bg,yscale="linear",xrange=icuts)
            rand_gen_dat_lin.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"RAND_GEN","RAND_AUG"])

            hist_sg, hist_bg, hist_both =  ensemble_data(sg,bg,yscale="log",xrange=icuts)
            rand_gen_dat_log.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"RAND_GEN","RAND_AUG"])
    return rand_gen_dat_lin,rand_gen_dat_log


def generate_data_randomly_mod_wbgonly():
    rand_gen_dat_lin = []
    rand_gen_dat_log = []
    rand_dat_mu_sig = [(np.random.uniform(2.0, 28.0),np.random.uniform(1.0, 5.0)) for _ in range(len(list(itertools.product(MEAN,SIGMA))))]
    print("------Generate dataset randomly------")
    for mean,sigma in tqdm(rand_dat_mu_sig):
        sg,bg = generate_data(mean,sigma)
        
        # Systematic augmentation
        for icuts in CUTS:
            hist_sg, hist_bg, hist_both = ensemble_data(sg,bg,yscale="linear",xrange=icuts)
            rand_gen_dat_lin.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"RAND_GEN","SYS_AUG"])
            hist_sg, hist_bg, hist_both = ensemble_data([],bg,yscale="linear",xrange=icuts)
            rand_gen_dat_lin.append([None,None,hist_sg, hist_bg, hist_both,icuts,"RAND_GEN","SYS_AUG"])

            hist_sg, hist_bg, hist_both =  ensemble_data(sg,bg,yscale="log",xrange=icuts)
            rand_gen_dat_log.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"RAND_GEN","SYS_AUG"])
            hist_sg, hist_bg, hist_both =  ensemble_data([],bg,yscale="log",xrange=icuts)
            rand_gen_dat_log.append([None,None,hist_sg, hist_bg, hist_both,icuts,"RAND_GEN","SYS_AUG"])

        # Generate random cuts
        rand_cuts_all= []
        while len(rand_cuts_all ) < len(CUTS):
            rand_cuts = [(np.random.uniform(0,30),np.random.uniform(0,30)) for _ in range(len(CUTS)-len(rand_cuts_all ))]
            rand_cuts = [i for i in rand_cuts if (i[0] < i[1] and i[1]-i[0] > MIN_WINDOW)]    
            rand_cuts_all = rand_cuts_all + rand_cuts
        assert len(CUTS) == len(rand_cuts_all)

        for icuts in rand_cuts_all:
            hist_sg, hist_bg, hist_both = ensemble_data(sg,bg,yscale="linear",xrange=icuts)
            rand_gen_dat_lin.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"RAND_GEN","RAND_AUG"])
            hist_sg, hist_bg, hist_both = ensemble_data([],bg,yscale="linear",xrange=icuts)
            rand_gen_dat_lin.append([None,None,hist_sg, hist_bg, hist_both,icuts,"RAND_GEN","RAND_AUG"])

            hist_sg, hist_bg, hist_both =  ensemble_data(sg,bg,yscale="log",xrange=icuts)
            rand_gen_dat_log.append([mean,sigma,hist_sg, hist_bg, hist_both,icuts,"RAND_GEN","RAND_AUG"])
            hist_sg, hist_bg, hist_both =  ensemble_data([],bg,yscale="log",xrange=icuts)
            rand_gen_dat_log.append([None,None,hist_sg, hist_bg, hist_both,icuts,"RAND_GEN","RAND_AUG"])
    return rand_gen_dat_lin,rand_gen_dat_log

def make_df(data,scale):
    df = pd.DataFrame(data,columns=["mean","sigma","hist_sg","hist_bg","hist_both","cuts","gen","aug"])
    df["cuts"]          = df["cuts"].progress_apply(lambda x: list(x))
    df["min"]           = df["hist_both"].progress_apply(lambda x: min(x))
    df["max"]           = df["hist_both"].progress_apply(lambda x: max(x))
    
    if scale == "log"    :  
        df["Significance"]  = df.progress_apply(lambda x: sum([np.exp(i) for i in x["hist_sg"]])/np.sqrt(sum([np.exp(i) for i in x["hist_bg"]])),axis=1)
        #df["nevents"]       = df["hist_both"].progress_apply(lambda x: sum([np.exp(i) for i in x]))
        df["nevents"]       = df["hist_both"].progress_apply(lambda x: sum(x))
        df["nevents_sg"]    = df["hist_sg"].progress_apply(lambda x: sum([np.exp(i) for i in x]))
        df["nevents_bg"]    = df["hist_bg"].progress_apply(lambda x: sum([np.exp(i) for i in x]))
    elif scale == "lin"  :  
        df["Significance"]  = df.progress_apply(lambda x: sum(x["hist_sg"])/np.sqrt(sum(x["hist_bg"])),axis=1)
        df["nevents"]       = df["hist_both"].progress_apply(lambda x: sum(x))
        df["nevents_sg"]    = df["hist_sg"].progress_apply(lambda x: sum(x))
        df["nevents_bg"]    = df["hist_bg"].progress_apply(lambda x: sum(x))
        
    else: raise KeyError
    df["hist_both"]     = df["hist_both"].progress_apply(lambda x: list(x))
    df["SCALE"]         = scale
    df["lower_limit"]   = df["cuts"].progress_apply(lambda x: x[0])
    df["upper_limit"]   = df["cuts"].progress_apply(lambda x: x[1])
    return df.copy()

    
def generate_mr_data(num_per_point):
    print("------Generate dataset systematically------")
    sys_gen_dat_lin = []
    sys_gen_dat_log = []
    rand_gen_dat_lin = []
    rand_gen_dat_log = []
    for mean,sigma in tqdm(list(itertools.product(MEAN,SIGMA))):
        sg,bg = generate_data(mean,sigma)
        # Systematic augmentation
        for _ in range(num_per_point):
            hist_sg, hist_bg, hist_both = ensemble_data(sg,bg,yscale="linear",xrange=XRANGE)
            sys_gen_dat_lin.append([mean,sigma,hist_sg, hist_bg, hist_both,XRANGE,"SYS_GEN","NO_AUG"])

            hist_sg, hist_bg, hist_both =  ensemble_data(sg,bg,yscale="log",xrange=XRANGE)
            sys_gen_dat_log.append([mean,sigma,hist_sg, hist_bg, hist_both,XRANGE,"SYS_GEN","NO_AUG"])
    
    for _ in tqdm(range(len(sys_gen_dat_log))):
        mean,sigma = (np.random.uniform(2.0, 28.0),np.random.uniform(1.0, 5.0)) 
        sg,bg = generate_data(mean,sigma)
        
        hist_sg, hist_bg, hist_both = ensemble_data(sg,bg,yscale="linear",xrange=XRANGE)
        rand_gen_dat_lin.append([mean,sigma,hist_sg, hist_bg, hist_both,XRANGE,"RAND_GEN","NO_AUG"])

        hist_sg, hist_bg, hist_both =  ensemble_data(sg,bg,yscale="log",xrange=XRANGE)
        rand_gen_dat_log.append([mean,sigma,hist_sg, hist_bg, hist_both,XRANGE,"RAND_GEN","NO_AUG"])

    return sys_gen_dat_lin,sys_gen_dat_log,rand_gen_dat_lin,rand_gen_dat_log

def generate_mr_data_mod_wbgonly(num_per_point):
    print("------Generate dataset systematically------")
    sys_gen_dat_lin = []
    sys_gen_dat_log = []
    rand_gen_dat_lin = []
    rand_gen_dat_log = []
    for mean,sigma in tqdm(list(itertools.product(MEAN,SIGMA))):
        sg,bg = generate_data(mean,sigma)
        # Systematic augmentation
        for _ in range(num_per_point):
            hist_sg, hist_bg, hist_both = ensemble_data(sg,bg,yscale="linear",xrange=XRANGE)
            sys_gen_dat_lin.append([mean,sigma,hist_sg, hist_bg, hist_both,XRANGE,"SYS_GEN","NO_AUG"])

            hist_sg, hist_bg, hist_both = ensemble_data([],bg,yscale="linear",xrange=XRANGE)
            sys_gen_dat_lin.append([None,None,hist_sg, hist_bg, hist_both,XRANGE,"SYS_GEN","NO_AUG"])

            hist_sg, hist_bg, hist_both =  ensemble_data(sg,bg,yscale="log",xrange=XRANGE)
            sys_gen_dat_log.append([mean,sigma,hist_sg, hist_bg, hist_both,XRANGE,"SYS_GEN","NO_AUG"])

            hist_sg, hist_bg, hist_both =  ensemble_data([],bg,yscale="log",xrange=XRANGE)
            sys_gen_dat_log.append([None,None,hist_sg, hist_bg, hist_both,XRANGE,"SYS_GEN","NO_AUG"])
    
    for _ in tqdm(range(int(len(sys_gen_dat_log)/2))):
        mean,sigma = (np.random.uniform(2.0, 28.0),np.random.uniform(1.0, 5.0)) 
        sg,bg = generate_data(mean,sigma)
        
        hist_sg, hist_bg, hist_both = ensemble_data(sg,bg,yscale="linear",xrange=XRANGE)
        rand_gen_dat_lin.append([mean,sigma,hist_sg, hist_bg, hist_both,XRANGE,"RAND_GEN","NO_AUG"])

        hist_sg, hist_bg, hist_both = ensemble_data([],bg,yscale="linear",xrange=XRANGE)
        rand_gen_dat_lin.append([None,None,hist_sg, hist_bg, hist_both,XRANGE,"RAND_GEN","NO_AUG"])

        hist_sg, hist_bg, hist_both =  ensemble_data(sg,bg,yscale="log",xrange=XRANGE)
        rand_gen_dat_log.append([mean,sigma,hist_sg, hist_bg, hist_both,XRANGE,"RAND_GEN","NO_AUG"])

        hist_sg, hist_bg, hist_both =  ensemble_data([],bg,yscale="log",xrange=XRANGE)
        rand_gen_dat_log.append([None,None,hist_sg, hist_bg, hist_both,XRANGE,"RAND_GEN","NO_AUG"])

    return sys_gen_dat_lin,sys_gen_dat_log,rand_gen_dat_lin,rand_gen_dat_log

class MyData():
    def __init__(self, data,vsize=0.2,shuffle=True,is_mr=False):
        self.data                         = data.copy()
        self.get_global_norm()
        self.data["hist_both_g_normed"]   = self.data["hist_both"].progress_apply(lambda x: self.normalize_hist_g(x))
        self.data["hist_both_g_normed"]   = self.data["hist_both_g_normed"].progress_apply(lambda x: np.array(x).astype(float))
        self.data["hist_both_l_normed"]   = self.data["hist_both"].progress_apply(lambda x: self.normalize_hist_l(x))
        self.data["hist_both_l_normed"]   = self.data["hist_both_l_normed"].progress_apply(lambda x: np.array(x).astype(float))
        self.data["mean_normed"]          = self.data["mean"].progress_apply(lambda x: self.normalize_mean(x))
        self.data["sigma_normed"]         = self.data["sigma"].progress_apply(lambda x: self.normalize_sigma(x))
        self.data["nevents_normed"]       = self.data["nevents"].progress_apply(lambda x: self.normalize_nevents(x))
        self.data                         = self.data.reset_index(drop=True)
        self.vsize                        = vsize
        self.shuffle                      = shuffle
        self.is_mr                        = is_mr
        
        
        if not is_mr:
            self.data["lower_limit_normed"]                    = self.data["lower_limit"].progress_apply(lambda x: self.normalize_lower_limit(x))
            self.data["upper_limit_normed"]                    = self.data["upper_limit"].progress_apply(lambda x: self.normalize_upper_limit(x))
            self.train_data_sys_aug  , self.val_data_sys_aug   = train_test_split(self.data[self.data["aug"] == "SYS_AUG"], test_size=self.vsize, random_state=seed_value)
            self.train_data_rand_aug , self.val_data_rand_aug  = train_test_split(self.data[self.data["aug"] == "RAND_AUG"], test_size=self.vsize, random_state=seed_value)
            self.train_data_sys_aug                            = self.train_data_sys_aug.reset_index(drop=True)
            self.val_data_sys_aug                              = self.val_data_sys_aug.reset_index(drop=True)
            self.train_data_rand_aug                           = self.train_data_rand_aug.reset_index(drop=True)
            self.val_data_rand_aug                             = self.val_data_rand_aug.reset_index(drop=True)

            self.mr = self.get_mr()

            self.train_data_sys_aug_mr                         = self.train_data_sys_aug[self.train_data_sys_aug["cuts"].progress_apply(lambda x : x == [0,30])]    
            self.val_data_sys_aug_mr                           = self.val_data_sys_aug[self.val_data_sys_aug["cuts"].progress_apply(lambda x : x == [0,30])]    
            self.train_data_rand_aug_mr                        = self.train_data_rand_aug[self.train_data_rand_aug["cuts"].progress_apply(lambda x : x == [0,30])]    
            self.val_data_rand_aug_mr                          = self.val_data_rand_aug[self.val_data_rand_aug["cuts"].progress_apply(lambda x : x == [0,30])]    
            self.train_data_sys_aug_mr                         = self.train_data_sys_aug[self.train_data_sys_aug["cuts"].progress_apply(lambda x : x == [0,30])]    
            self.val_data_sys_aug_mr                           = self.val_data_sys_aug[self.val_data_sys_aug["cuts"].progress_apply(lambda x : x == [0,30])]    
            self.train_data_rand_aug_mr                        = self.train_data_rand_aug[self.train_data_rand_aug["cuts"].progress_apply(lambda x : x == [0,30])]    
            self.val_data_rand_aug_mr                          = self.val_data_rand_aug[self.val_data_rand_aug["cuts"].progress_apply(lambda x : x == [0,30])]    


        self.train_data , self.val_data   = train_test_split(self.data, test_size=self.vsize, random_state=seed_value)
        self.train_data                   = self.train_data.reset_index(drop=True)
        self.val_data                     = self.val_data.reset_index(drop=True)

        self.train_data_mr                = self.train_data[self.train_data["cuts"].progress_apply(lambda x : x == [0,30])]    
        self.val_data_mr                  = self.val_data[self.val_data["cuts"].progress_apply(lambda x : x == [0,30])]    
    
    def get_global_norm(self):
        self.global_min      = self.data["min"].min()
        self.global_max      = self.data["max"].max()
        self.mean_min        = self.data["mean"].min()
        self.mean_max        = self.data["mean"].max()
        self.sigma_min       = self.data["sigma"].min()
        self.sigma_max       = self.data["sigma"].max()        
        self.nevents_min     = self.data["nevents"].min()
        self.nevents_max     = self.data["nevents"].max()
        self.lower_limit_min = self.data["lower_limit"].min()
        self.lower_limit_max = self.data["lower_limit"].max()
        self.upper_limit_min = self.data["upper_limit"].min()
        self.upper_limit_max = self.data["upper_limit"].max()

        return self.global_min,self.global_max,self.mean_min,self.mean_max,self.sigma_min,self.sigma_max,self.nevents_min,self.nevents_max,self.lower_limit_min,self.lower_limit_max,self.upper_limit_min,self.upper_limit_max
    
    def normalize_mean(self,data):
        return (data-self.mean_min)/(self.mean_max-self.mean_min)

    def unnorm_mean(self,data):
        return data*(self.mean_max-self.mean_min)+self.mean_min

    def normalize_sigma(self,data):
        return (data-self.sigma_min)/(self.sigma_max-self.sigma_min)
        
    def unnorm_sigma(self,data):
        return data*(self.sigma_max-self.sigma_min)+self.sigma_min
    
    def normalize_nevents(self,data):
        return (data-self.nevents_min)/(self.nevents_max-self.nevents_min)
    
    def unnorm_nevents(self,data):
        return data*(self.nevents_max-self.nevents_min)+self.nevents_min
    
    def normalize_lower_limit(self,data):
        return (data-self.lower_limit_min)/(self.lower_limit_max-self.lower_limit_min)
    
    def unnorm_lower_limit(self,data):
        return data*(self.lower_limit_max-self.lower_limit_min)+self.lower_limit_min
    
    def normalize_upper_limit(self,data):
        return (data-self.upper_limit_min)/(self.upper_limit_max-self.upper_limit_min)
    
    def unnorm_upper_limit(self,data):
        return data*(self.upper_limit_max-self.upper_limit_min)+self.upper_limit_min

    def normalize_hist_g(self,data):
        try:
            return (data-self.global_min)/(self.global_max-self.global_min)
        except:
            return [(i-self.global_min)/(self.global_max-self.global_min) for i in data]
    
    def unnormalize_hist_g(self,data):
        return data*(self.global_max-self.global_min)+self.global_min
    
    def normalize_hist_l(self,data):
        return (data-min(data))/(max(data)-min(data))

    def get_mr(self):
        self.data_mr = self.data[self.data["cuts"].apply(lambda x : x == [0,30])]
        return MyData(self.data_mr,is_mr=True)


def make_hist_tensor(data):
    return torch.tensor(data.to_list()).reshape(len(data), 1, NBINS)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_dataset_loader(hist,lab,batch_size,aux=None):
    if aux is not None:
        dataset    = TensorDataset(hist,aux,lab)
    else:
        dataset = TensorDataset(hist,lab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader

def prepare_dataset_for_training(data,test_data=None,aux=False,batch_size=32,norm="g"):
    if test_data is not None : 
        test_data = test_data.copy()
    if norm.lower() == "g":    
        train_hist                           = make_hist_tensor(data.train_data["hist_both_g_normed"]).to(device)
        val_hist                             = make_hist_tensor(data.val_data["hist_both_g_normed"]).to(device)
        test_hist                            = None 
        if test_data is not None : 
            test_hist                        = make_hist_tensor(test_data["hist_both"].progress_apply(lambda x: np.array(data.normalize_hist_g(x)).astype(float))).to(device)
    elif norm.lower() == "l":    
        train_hist                           = make_hist_tensor(data.train_data["hist_both_l_normed"]).to(device)
        val_hist                             = make_hist_tensor(data.val_data["hist_both_l_normed"]).to(device)
        test_hist                            = None 
        if test_data is not None : 
            test_hist                        = make_hist_tensor(test_data["hist_both"].progress_apply(lambda x: np.array(data.normalize_hist_l(x)).astype(float))).to(device)

    train_lab                            = torch.tensor(data.train_data[["mean_normed","sigma_normed"]].values).to(device)
    val_lab                              = torch.tensor(data.val_data[  ["mean_normed","sigma_normed"]].values).to(device)
    test_lab                             = None
    if test_data is not None:  
        test_data["mean_normed"]  = test_data["mean"].progress_apply(lambda x: data.normalize_mean(x))
        test_data["sigma_normed"] = test_data["sigma"].progress_apply(lambda x: data.normalize_sigma(x))
        test_lab                  = torch.tensor(test_data[["mean_normed","sigma_normed"]].values).to(device)
                                                
    if aux:
        train_aux                           = torch.tensor(data.train_data[["lower_limit_normed","upper_limit_normed","nevents_normed"]].values).to(device)
        val_aux                             = torch.tensor(data.val_data[["lower_limit_normed","upper_limit_normed","nevents_normed"]].values).to(device)
        test_aux                            = None
        if test_data is not None:  
            test_data["lower_limit_normed"] = test_data["lower_limit"].progress_apply(lambda x: data.normalize_lower_limit(x))
            test_data["upper_limit_normed"] = test_data["upper_limit"].progress_apply(lambda x: data.normalize_upper_limit(x))
            test_data["nevents_normed"]     = test_data["nevents"].progress_apply(lambda x: data.normalize_nevents(x))
            test_aux                        = torch.tensor(test_data[["lower_limit_normed","upper_limit_normed","nevents_normed"]].values).to(device)
        
        _, train_dataloader = make_dataset_loader(train_hist,train_lab,batch_size,train_aux)
        return train_dataloader, val_hist, val_lab, val_aux, test_hist, test_lab, test_aux

    _, train_dataloader = make_dataset_loader(train_hist,train_lab,batch_size)
    return train_dataloader, val_hist, val_lab, test_hist, test_lab