import json
import torch
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from MuMIMOClass import *
from torch.utils.data import Dataset
from torch.nn import functional as F
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig

with open('./param.json', 'r') as f:
    params = json.load(f)


# datasets = np.load('User_{}_RL_datasets.npz'.format(params['NumUE']))
#datasets = np.load('../../../datasets_warehouse(done)/{}Mb/fixed_power/{}_RIS_action_space/User_{}_with_{}_RIS_complete_datasets.npz'.format(params['LocalDataSize'], params['RISActionSpace'], params['NumUE'], params['NumRISEle']))
datasets = np.load('./User_{}_with_{}_RIS_complete_datasets.npz'.format(params['NumUE'], params['NumRISEle']))
# datasets = np.load('User_{}_RL_datasets_recursive.npz'.format(params['NumUE']))

loaded_H_U2B_Ric = datasets['H_U2B_Ric']
loaded_H_R2B_Ric = datasets['H_R2B_Ric']
loaded_H_U2R_Ric = datasets['H_U2R_Ric']
loaded_RefVector = datasets['RefVector']
loaded_power = datasets['power']
loaded_theta = datasets['phase']
loaded_RTG = datasets['RTG']
loaded_load_remaining = datasets['load_remaining']
loaded_time_remaining = datasets['time_remaining']

# print("loaded_H_U2B_Ric.shape: ", loaded_H_U2B_Ric.shape)
# print("loaded_H_R2B_Ric.shape: ", loaded_H_R2B_Ric.shape)
# print("loaded_H_U2R_Ric.shape: ", loaded_H_U2R_Ric.shape)
# print("loaded_RefVector.shape: ", loaded_RefVector.shape)
# print("loaded_power.shape: ", loaded_power.shape)
# print("loaded_theta.shape: ", loaded_theta.shape)
# print("loaded_RTG.shape: ", loaded_RTG.shape)
# print("loaded_load_remaining.shape: ", loaded_load_remaining.shape)
# print("loaded_time_remaining.shape: ", loaded_time_remaining.shape)

# parser = argparse.ArgumentParser()
# args = parser.parse_args()

class CommunicationDataset(Dataset):
    def __init__(self, H_U2B_Ric, H_R2B_Ric, H_U2R_Ric, RefVector, power, theta, RTG, load_remaining, time_remaining, trasmission_length):
        self.H_U2B_Ric = H_U2B_Ric
        self.H_R2B_Ric = H_R2B_Ric
        self.H_U2R_Ric = H_U2R_Ric
        self.RefVector = RefVector
        self.power = power
        self.theta = theta
        self.RTG = RTG
        self.load_remaining = load_remaining
        self.time_remaining = time_remaining
        self.transmission_length = trasmission_length
        

    def __len__(self):
        return len(self.RTG) // self.transmission_length

    def __getitem__(self, idx):
        H_U2B_Ric_per_transmission = self.H_U2B_Ric[idx]           #(transmission_length, NumBSAnt, NumUE)
        H_R2B_Ric_per_transmission = self.H_R2B_Ric[idx]           #(transmission_length, NumBSAnt, NumRISEle)
        H_U2R_Ric_per_transmission = self.H_U2R_Ric[idx]           #(transmission_length, NumRISEle, NumUE)
        RefVector_per_transmission = self.RefVector[idx]            #(transmission_length, NumRISEle)
        power_per_transmission = self.power[idx]                   #(transmission_length, NumUE)
        theta_per_transmission = self.theta[idx]                   #(transmission_length, NumRISEle)
        RTG_per_transmission = self.RTG[idx]                       #(transmission_length, NumUE)
        load_remaining_per_transmission = self.load_remaining[idx] #(transmission_length, NumUE)
        time_remaining_per_transmission = self.time_remaining[idx] #(transmission_length, NumUE)

        #reshape the data (transmission_length, flatten_data)
        H_U2B_Ric_per_transmission = H_U2B_Ric_per_transmission.reshape(self.transmission_length, -1) # (transmission_length, NumBSAnt*NumUE)
        H_R2B_Ric_per_transmission = H_R2B_Ric_per_transmission.reshape(self.transmission_length, -1) # (transmission_length, NumBSAnt*NumRISEle)
        H_U2R_Ric_per_transmission = H_U2R_Ric_per_transmission.reshape(self.transmission_length, -1) # (transmission_length, NumRISEle*NumUE)
        RefVector_per_transmission = RefVector_per_transmission.reshape(self.transmission_length, -1) # (transmission_length, NumRISEle)
        power_per_transmission = power_per_transmission.reshape(self.transmission_length, -1) # (transmission_length, NumUE)
        phase_per_transmission = theta_per_transmission.reshape(self.transmission_length, -1) # (transmission_length, NumRISEle)
        load_remaining_per_transmission = load_remaining_per_transmission.reshape(self.transmission_length, -1) # (transmission_length, NumUE)
        time_remaining_per_transmission = time_remaining_per_transmission.reshape(self.transmission_length, -1) # (transmission_length, NumUE)

        channel_state = np.concatenate((H_U2B_Ric_per_transmission.real, H_U2B_Ric_per_transmission.imag), axis=1)
        channel_state = np.concatenate((channel_state, H_R2B_Ric_per_transmission.real), axis=1)
        channel_state = np.concatenate((channel_state, H_R2B_Ric_per_transmission.imag), axis=1)
        channel_state = np.concatenate((channel_state, H_U2R_Ric_per_transmission.real), axis=1)
        channel_state = np.concatenate((channel_state, H_U2R_Ric_per_transmission.imag), axis=1)
        channel_state = np.concatenate((channel_state, RefVector_per_transmission.real), axis=1)
        channel_state = np.concatenate((channel_state, RefVector_per_transmission.imag), axis=1)
        obss = np.concatenate((channel_state, load_remaining_per_transmission, time_remaining_per_transmission), axis=1)
        obss = torch.tensor(obss, dtype=torch.float32)
        power = torch.tensor(power_per_transmission, dtype=torch.long)
        theta = torch.tensor(phase_per_transmission, dtype=torch.long)
        action = torch.cat((power, theta), dim=1)
        # action = power
        RTG = torch.tensor(RTG_per_transmission, dtype=torch.float32).unsqueeze(-1)
        return obss, action, RTG, torch.tensor([idx], dtype=torch.int64).unsqueeze(1)
        # return obss, power, RTG, torch.tensor([idx], dtype=torch.int64).unsqueeze(1)



# train_dataset = CommunicationDataset(loaded_H_U2B_Ric, loaded_H_R2B_Ric, loaded_H_U2R_Ric, loaded_power, loaded_RTG, loaded_load_remaining, loaded_time_remaining, 20)
train_dataset = CommunicationDataset(loaded_H_U2B_Ric, loaded_H_R2B_Ric, loaded_H_U2R_Ric, loaded_RefVector, loaded_power, loaded_theta, loaded_RTG, loaded_load_remaining, loaded_time_remaining, 20)

model_conf = GPTConfig(params['UserActionSpace'], params['RISActionSpace'], train_dataset.transmission_length*(1+1+params['NumUE']+params['NumRISEle']),
                  n_layer=8, n_head=8, n_embd=128, model_type='reward_conditioned', max_timestep=len(train_dataset) * train_dataset.transmission_length, params = params)
model = GPT(model_conf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer_conf = TrainerConfig(max_epochs=5, batch_size=5, learning_rate=8e-4,
                      lr_decay=False, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*train_dataset.transmission_length*3,
                      num_workers=1, seed=123, model_type='reward_conditioned', max_timestep=len(train_dataset) * train_dataset.transmission_length, params = params)

trainer = Trainer(model, train_dataset, None, trainer_conf)

trainer.train()


