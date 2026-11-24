import math
import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
import pickle
import jammy_flows as jf 

flow_options_overwrite=dict()
flow_options_overwrite["g"]=dict()
flow_options_overwrite["g"]["upper_bound_for_widths"]=1
flow_options_overwrite["g"]["lower_bound_for_widths"]=0.01
flow_options_overwrite["g"]["fit_normalization"]=0
flow_options_overwrite["t"]=dict()
flow_options_overwrite["t"]["cov_type"]="full"
  
class cnnhead(torch.nn.Module):
    def __init__(self,output_dim,seq_len=100,in_channels=1):
        super(cnnhead, self).__init__()
        
        self.relu    = nn.ReLU()
        self.conv1   = torch.nn.Conv1d(in_channels=in_channels, out_channels=20, kernel_size=5)#,padding=2)
        self.conv2   = torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=2)#,padding=2)
        self.conv3   = torch.nn.Conv1d(in_channels=10, out_channels=2, kernel_size=1)#,padding=2)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)
        
        # Compute output length after each layer
        def compute_output_length(L, kernel, stride=1, padding=0, dilation=1):
            return math.floor((L + 2*padding - dilation*(kernel - 1) - 1) / stride + 1)

        l1 = compute_output_length(seq_len, kernel=5)          # conv1
        l1 = compute_output_length(l1, kernel=2, stride=2)     # pool1
        l2 = compute_output_length(l1, kernel=2)               # conv2
        l2 = compute_output_length(l2, kernel=2, stride=2)     # pool2
        l3 = compute_output_length(l2, kernel=1)               # conv3
        flattened_dim = l3 * self.conv3.out_channels  # 2 is the output channels from conv3

        self.linear1 = nn.Linear(flattened_dim, 20)
        self.linear2 = nn.Linear(20, output_dim)

        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)     
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x
    
class auxhead(nn.Module):
    def __init__(self, input_size,output_dim):
        super(auxhead, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 2*input_size)
        self.fc2 = nn.Linear(2*input_size, 2*input_size)
        self.fc3 = nn.Linear(2*input_size, output_dim)  # Output size 2 for the MLP

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class RACNNFlows(nn.Module):
    def __init__(self, aux_input_size,cond_dim=10, hidden_size=10, hidden_size2=10,seq_len=100,in_channels=1,jammy_layers="gggt"):
        super(RACNNFlows, self).__init__()
        self.cnn_head = cnnhead(cond_dim,seq_len=seq_len,in_channels=in_channels)
        self.aux_head = auxhead(aux_input_size,cond_dim)
        
        # Additional hidden layers
        self.merger_layer = nn.Linear(2*cond_dim, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
      

        # Output layer PDF
        self.pdf_hidden_layer = nn.Linear(hidden_size, hidden_size)  # Output size 2 for final predictions
        self.pdf_hidden_layer2 = nn.Linear(hidden_size, hidden_size2)  # Output size 2 for final predictions
        self.pdf_output_layer = nn.Linear(hidden_size2, cond_dim)  # Output size 2 for final predictions
        #self.pdf     = jf.pdf("e3", "t", conditional_input_dim=cond_dim,options_overwrite=flow_options_overwrite.copy())
        self.pdf     = jf.pdf("e2", jammy_layers, conditional_input_dim=cond_dim,options_overwrite=flow_options_overwrite.copy())

    def forward(self, x_cnn, x_mlp):
        cnn_output = self.cnn_head(x_cnn)
        aux_output = self.aux_head(x_mlp)
        # Concatenate the outputs from both heads
        concatenated = torch.cat((cnn_output, aux_output.squeeze(dim=1)), dim=1)
        
        # Pass through additional hidden layers
        merger_output = self.relu(self.merger_layer(concatenated))
        hidden1_output = self.relu(self.hidden1(merger_output))
        hidden2_output = self.relu(self.hidden2(hidden1_output))
        
        # Final predictions
        pdf_output  = self.relu(self.pdf_hidden_layer(hidden2_output))
        pdf_output2 = self.relu(self.pdf_hidden_layer2(pdf_output))
        final_output = self.pdf_output_layer(pdf_output2)

        return final_output
    
class RACNNClass(nn.Module):
    def __init__(self, aux_input_size,cond_dim, hidden_size=10, hidden_size2=10,seq_len=100,in_channels=1):
        super(RACNNClass, self).__init__()
        self.cnn_head = cnnhead(cond_dim,seq_len=seq_len,in_channels=in_channels)
        self.aux_head = auxhead(aux_input_size,cond_dim)
        
        # Additional hidden layers
        self.merger_layer = nn.Linear(2*cond_dim, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

        # Output layer MLP
        self.mlp_hidden_layer  = nn.Linear(hidden_size, hidden_size)  # Output size 2 for final predictions
        self.mlp_hidden_layer2  = nn.Linear(hidden_size, hidden_size2)  # Output size 2 for final predictions
        self.mlp_output_layer = nn.Linear(hidden_size2,1)

    def forward(self, x_cnn, x_mlp):
        cnn_output = self.cnn_head(x_cnn)
        aux_output = self.aux_head(x_mlp)
        # Concatenate the outputs from both heads
        concatenated = torch.cat((cnn_output, aux_output.squeeze(dim=1)), dim=1)
        
        # Pass through additional hidden layers
        merger_output = self.relu(self.merger_layer(concatenated))
        hidden1_output = self.relu(self.hidden1(merger_output))
        hidden2_output = self.relu(self.hidden2(hidden1_output))
        
        mlp_output = self.relu(self.mlp_hidden_layer(hidden2_output))
        mlp_output2 = self.relu(self.mlp_hidden_layer2(mlp_output))
        s_b_class = self.mlp_output_layer(mlp_output2)
        return s_b_class
    
class RACNNClassiFlows(nn.Module):
    def __init__(self, aux_input_size,cond_dim=10, hidden_size=10, hidden_size2=10,seq_len=100,in_channels=1,jammy_layers="gggt"):
        super(RACNNClassiFlows, self).__init__()
        self.cnn_head = cnnhead(cond_dim,seq_len=seq_len,in_channels=in_channels)
        self.aux_head = auxhead(aux_input_size,cond_dim)
        
        # Additional hidden layers
        self.merger_layer = nn.Linear(2*cond_dim, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        

        # Output layer PDF
        self.pdf_hidden_layer = nn.Linear(hidden_size, hidden_size)  # Output size 2 for final predictions
        self.pdf_hidden_layer2 = nn.Linear(hidden_size, hidden_size2)  # Output size 2 for final predictions
        self.pdf_output_layer = nn.Linear(hidden_size2, cond_dim)  # Output size 2 for final predictions
        #self.pdf     = jf.pdf("e3", "t", conditional_input_dim=cond_dim,options_overwrite=flow_options_overwrite.copy())
        self.pdf     = jf.pdf("e2", jammy_layers, conditional_input_dim=cond_dim,options_overwrite=flow_options_overwrite.copy())

        # Output layer MLP
        self.mlp_hidden_layer  = nn.Linear(hidden_size, hidden_size)  # Output size 2 for final predictions
        self.mlp_hidden_layer2  = nn.Linear(hidden_size, hidden_size2)  # Output size 2 for final predictions
        self.mlp_output_layer = nn.Linear(hidden_size2,1)

    def forward(self, x_cnn, x_mlp):
        cnn_output = self.cnn_head(x_cnn)
        aux_output = self.aux_head(x_mlp)
        # Concatenate the outputs from both heads
        concatenated = torch.cat((cnn_output, aux_output.squeeze(dim=1)), dim=1)
        
        # Pass through additional hidden layers
        merger_output = self.relu(self.merger_layer(concatenated))
        hidden1_output = self.relu(self.hidden1(merger_output))
        hidden2_output = self.relu(self.hidden2(hidden1_output))
        
        # Final predictions
        pdf_output  = self.relu(self.pdf_hidden_layer(hidden2_output))
        pdf_output2 = self.relu(self.pdf_hidden_layer2(pdf_output))
        final_output = self.pdf_output_layer(pdf_output2)
 
        mlp_output = self.relu(self.mlp_hidden_layer(hidden2_output))
        mlp_output2 = self.relu(self.mlp_hidden_layer2(mlp_output))
        s_b_class = self.mlp_output_layer(mlp_output2)
        return final_output,s_b_class

class CNNFlows(nn.Module):
    def __init__(self, cond_dim=10, hidden_size=10, hidden_size2=10,seq_len=100,in_channels=1,jammy_layers="gggt"):
        super(CNNFlows, self).__init__()
        self.cnn_head = cnnhead(cond_dim,seq_len=seq_len,in_channels=in_channels)
        
        # Additional hidden layers
        self.merger_layer = nn.Linear(cond_dim, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
      

        # Output layer PDF
        self.pdf_hidden_layer = nn.Linear(hidden_size, hidden_size)  # Output size 2 for final predictions
        self.pdf_hidden_layer2 = nn.Linear(hidden_size, hidden_size2)  # Output size 2 for final predictions
        self.pdf_output_layer = nn.Linear(hidden_size2, cond_dim)  # Output size 2 for final predictions
        #self.pdf     = jf.pdf("e3", "t", conditional_input_dim=cond_dim,options_overwrite=flow_options_overwrite.copy())
        self.pdf     = jf.pdf("e2", jammy_layers, conditional_input_dim=cond_dim,options_overwrite=flow_options_overwrite.copy())

    def forward(self, x_cnn):
        cnn_output = self.cnn_head(x_cnn)
        
        # Pass through additional hidden layers
        merger_output = self.relu(self.merger_layer(cnn_output))
        hidden1_output = self.relu(self.hidden1(merger_output))
        hidden2_output = self.relu(self.hidden2(hidden1_output))
        
        # Final predictions
        pdf_output  = self.relu(self.pdf_hidden_layer(hidden2_output))
        pdf_output2 = self.relu(self.pdf_hidden_layer2(pdf_output))
        final_output = self.pdf_output_layer(pdf_output2)

        return final_output
    
class CNNClass(nn.Module):
    def __init__(self, cond_dim, hidden_size=10, hidden_size2=10,seq_len=100,in_channels=1):
        super(CNNClass, self).__init__()
        self.cnn_head = cnnhead(cond_dim,seq_len=seq_len,in_channels=in_channels)
        
        # Additional hidden layers
        self.merger_layer = nn.Linear(cond_dim, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

        # Output layer MLP
        self.mlp_hidden_layer  = nn.Linear(hidden_size, hidden_size)  # Output size 2 for final predictions
        self.mlp_hidden_layer2  = nn.Linear(hidden_size, hidden_size2)  # Output size 2 for final predictions
        self.mlp_output_layer = nn.Linear(hidden_size2,1)

    def forward(self, x_cnn):
        cnn_output = self.cnn_head(x_cnn)
        # Concatenate the outputs from both heads
        
        # Pass through additional hidden layers
        merger_output = self.relu(self.merger_layer(cnn_output))
        hidden1_output = self.relu(self.hidden1(merger_output))
        hidden2_output = self.relu(self.hidden2(hidden1_output))
        
        mlp_output = self.relu(self.mlp_hidden_layer(hidden2_output))
        mlp_output2 = self.relu(self.mlp_hidden_layer2(mlp_output))
        s_b_class = self.mlp_output_layer(mlp_output2)
        return s_b_class
    
class CNNClassiFlows(nn.Module):
    def __init__(self, cond_dim=10, hidden_size=10, hidden_size2=10,seq_len=100,in_channels=1,jammy_layers="gggt"):
        super(CNNClassiFlows, self).__init__()
        self.cnn_head = cnnhead(cond_dim,seq_len=seq_len,in_channels=in_channels)
        
        # Additional hidden layers
        self.merger_layer = nn.Linear(cond_dim, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

        # Output layer PDF
        self.pdf_hidden_layer = nn.Linear(hidden_size, hidden_size)  # Output size 2 for final predictions
        self.pdf_hidden_layer2 = nn.Linear(hidden_size, hidden_size2)  # Output size 2 for final predictions
        self.pdf_output_layer = nn.Linear(hidden_size2, cond_dim)  # Output size 2 for final predictions
        #self.pdf     = jf.pdf("e3", "t", conditional_input_dim=cond_dim,options_overwrite=flow_options_overwrite.copy())
        self.pdf     = jf.pdf("e2", jammy_layers, conditional_input_dim=cond_dim,options_overwrite=flow_options_overwrite.copy())

        # Output layer MLP
        self.mlp_hidden_layer  = nn.Linear(hidden_size, hidden_size)  # Output size 2 for final predictions
        self.mlp_hidden_layer2  = nn.Linear(hidden_size, hidden_size2)  # Output size 2 for final predictions
        self.mlp_output_layer = nn.Linear(hidden_size2,1)

    def forward(self, x_cnn):
        cnn_output = self.cnn_head(x_cnn)
        # Concatenate the outputs from both heads
        
        # Pass through additional hidden layers
        merger_output = self.relu(self.merger_layer(cnn_output))
        hidden1_output = self.relu(self.hidden1(merger_output))
        hidden2_output = self.relu(self.hidden2(hidden1_output))
        
        # Final predictions
        pdf_output  = self.relu(self.pdf_hidden_layer(hidden2_output))
        pdf_output2 = self.relu(self.pdf_hidden_layer2(pdf_output))
        final_output = self.pdf_output_layer(pdf_output2)
 
        mlp_output = self.relu(self.mlp_hidden_layer(hidden2_output))
        mlp_output2 = self.relu(self.mlp_hidden_layer2(mlp_output))
        s_b_class = self.mlp_output_layer(mlp_output2)
        return final_output,s_b_class

class NECNNFlows(nn.Module):
    def __init__(self, cond_dim=10, hidden_size=10, hidden_size2=10,seq_len=100,in_channels=1,jammy_layers="gggt"):
        super(NECNNFlows, self).__init__()
        self.cnn_head = cnnhead(cond_dim,seq_len=seq_len,in_channels=in_channels)
        self.aux_head = auxhead(1,cond_dim)
        
        # Additional hidden layers
        self.merger_layer = nn.Linear(2*cond_dim, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
      

        # Output layer PDF
        self.pdf_hidden_layer = nn.Linear(hidden_size, hidden_size)  # Output size 2 for final predictions
        self.pdf_hidden_layer2 = nn.Linear(hidden_size, hidden_size2)  # Output size 2 for final predictions
        self.pdf_output_layer = nn.Linear(hidden_size2, cond_dim)  # Output size 2 for final predictions
        #self.pdf     = jf.pdf("e3", "t", conditional_input_dim=cond_dim,options_overwrite=flow_options_overwrite.copy())
        self.pdf     = jf.pdf("e2", jammy_layers, conditional_input_dim=cond_dim,options_overwrite=flow_options_overwrite.copy())

    def forward(self, x_cnn, x_mlp):
        x_mlp = x_mlp[:, -1].unsqueeze(1)
        cnn_output = self.cnn_head(x_cnn)
        aux_output = self.aux_head(x_mlp)
        # Concatenate the outputs from both heads
        concatenated = torch.cat((cnn_output, aux_output.squeeze(dim=1)), dim=1)
        
        # Pass through additional hidden layers
        merger_output = self.relu(self.merger_layer(concatenated))
        hidden1_output = self.relu(self.hidden1(merger_output))
        hidden2_output = self.relu(self.hidden2(hidden1_output))
        
        # Final predictions
        pdf_output  = self.relu(self.pdf_hidden_layer(hidden2_output))
        pdf_output2 = self.relu(self.pdf_hidden_layer2(pdf_output))
        final_output = self.pdf_output_layer(pdf_output2)

        return final_output
    
class NECNNClass(nn.Module):
    def __init__(self, cond_dim, hidden_size=10, hidden_size2=10,seq_len=100,in_channels=1):
        super(NECNNClass, self).__init__()
        self.cnn_head = cnnhead(cond_dim,seq_len=seq_len,in_channels=in_channels)
        self.aux_head = auxhead(1,cond_dim)
        
        # Additional hidden layers
        self.merger_layer = nn.Linear(2*cond_dim, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

        # Output layer MLP
        self.mlp_hidden_layer  = nn.Linear(hidden_size, hidden_size)  # Output size 2 for final predictions
        self.mlp_hidden_layer2  = nn.Linear(hidden_size, hidden_size2)  # Output size 2 for final predictions
        self.mlp_output_layer = nn.Linear(hidden_size2,1)

    def forward(self, x_cnn, x_mlp):
        x_mlp = x_mlp[:, -1].unsqueeze(1)
        cnn_output = self.cnn_head(x_cnn)
        aux_output = self.aux_head(x_mlp)
        # Concatenate the outputs from both heads
        concatenated = torch.cat((cnn_output, aux_output.squeeze(dim=1)), dim=1)
        
        # Pass through additional hidden layers
        merger_output = self.relu(self.merger_layer(concatenated))
        hidden1_output = self.relu(self.hidden1(merger_output))
        hidden2_output = self.relu(self.hidden2(hidden1_output))
        
        mlp_output = self.relu(self.mlp_hidden_layer(hidden2_output))
        mlp_output2 = self.relu(self.mlp_hidden_layer2(mlp_output))
        s_b_class = self.mlp_output_layer(mlp_output2)
        return s_b_class
    
class NECNNClassiFlows(nn.Module):
    def __init__(self, cond_dim=10, hidden_size=10, hidden_size2=10,seq_len=100,in_channels=1,jammy_layers="gggt"):
        super(NECNNClassiFlows, self).__init__()
        self.cnn_head = cnnhead(cond_dim,seq_len=seq_len,in_channels=in_channels)
        self.aux_head = auxhead(1,cond_dim)
        
        # Additional hidden layers
        self.merger_layer = nn.Linear(2*cond_dim, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        

        # Output layer PDF
        self.pdf_hidden_layer = nn.Linear(hidden_size, hidden_size)  # Output size 2 for final predictions
        self.pdf_hidden_layer2 = nn.Linear(hidden_size, hidden_size2)  # Output size 2 for final predictions
        self.pdf_output_layer = nn.Linear(hidden_size2, cond_dim)  # Output size 2 for final predictions
        #self.pdf     = jf.pdf("e3", "t", conditional_input_dim=cond_dim,options_overwrite=flow_options_overwrite.copy())
        self.pdf     = jf.pdf("e2", jammy_layers, conditional_input_dim=cond_dim,options_overwrite=flow_options_overwrite.copy())

        # Output layer MLP
        self.mlp_hidden_layer  = nn.Linear(hidden_size, hidden_size)  # Output size 2 for final predictions
        self.mlp_hidden_layer2  = nn.Linear(hidden_size, hidden_size2)  # Output size 2 for final predictions
        self.mlp_output_layer = nn.Linear(hidden_size2,1)

    def forward(self, x_cnn, x_mlp):
        x_mlp = x_mlp[:, -1].unsqueeze(1)
        
        cnn_output = self.cnn_head(x_cnn)
        aux_output = self.aux_head(x_mlp)
        # Concatenate the outputs from both heads
        concatenated = torch.cat((cnn_output, aux_output.squeeze(dim=1)), dim=1)
        
        # Pass through additional hidden layers
        merger_output = self.relu(self.merger_layer(concatenated))
        hidden1_output = self.relu(self.hidden1(merger_output))
        hidden2_output = self.relu(self.hidden2(hidden1_output))
        
        # Final predictions
        pdf_output  = self.relu(self.pdf_hidden_layer(hidden2_output))
        pdf_output2 = self.relu(self.pdf_hidden_layer2(pdf_output))
        final_output = self.pdf_output_layer(pdf_output2)
 
        mlp_output = self.relu(self.mlp_hidden_layer(hidden2_output))
        mlp_output2 = self.relu(self.mlp_hidden_layer2(mlp_output))
        s_b_class = self.mlp_output_layer(mlp_output2)
        return final_output,s_b_class
