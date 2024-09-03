
## @package calculate_mean_ec
# This module is used to calculate the mean effective capacity of the channel
# The effective capacity is calculated using the formula:
# \f$ EC = \frac{-1}{\theta} \log \left( \frac{1}{N} \sum_{i=1}^{N} \exp(-\theta B W R_i) \right) \f$
# where \f$ R_i \f$ is the achievable rate of the channel
# The mean effective capacity is calculated by taking the mean of the effective capacity over all the steps in the episode
# The mean effective capacity is then used to calculate the recommended data size for the next episode

import json
import numpy as np
from MuMIMOClass import *
import matplotlib.pyplot as plt
import math
import pandas as pd
from math import *
from collections import namedtuple
from tqdm import tqdm


import random
import time


# # set seed
seed = 3
np.random.seed(seed)
random.seed(seed)



with open('./param.json', 'r') as f:
    params = json.load(f)

if __name__ == "__main__":
    # Simulation Parameters
    EPISODES = params['BB']["EPISODES"]
    time_per_episode = 1                                   # update slow fading(pathloss)/UE position every 1s
    time_per_step = 0.05                                # update fast fading(rayleigh/rician fading) every 0.05s
    num_steps = int(time_per_episode/time_per_step)   # each EPISODES has 20 steps
    steps = EPISODES*num_steps
    NumBSAnt = params['NumBsAnt']                   # M (BS antenna number)
    NumRISEle = params['NumRISEle']                 # L (RIS element number)
    NumUE = params['NumUE']                         # K (user number)
    UE_power_selection = 4                          # [0, 3, 10, 200]  (mW)   
    power_space = UE_power_selection**NumUE         # action space of power selection (four power levels for each UE)
    UE_position_variance = 3                        # range of UE distribution
    PhaseShift_bit = 3                              # number of bits of each RIS phase shift
    stages = 2 ** PhaseShift_bit                    # 8 phase shift stages
    LocalDataSize = params['LocalDataSize']         # 900 Mb = 25 * 10**6 * 8 bit
    BW = 100                                        # bandwidth = 100MHz  把後面10^6都省略了 不然在計算EC的log的時候會出問題 哭阿
    noise = 10**(-104/10)                           # noise power = -104dBm = 10**(-104/10) mW
    UE_initial_power = 3                            # UE power(dBm): 0~30dBm  = 0~1000mW
    K_U2B = 10                                      # Rician factor: K-factor, if K = 0, Rician equal to Rayleigh 
    K_R2B = 10
    K_U2R = 10
    D_max = 1

    thetas = [1e-2, 1e-1, 1, 5, 10]
    # thetas = np.linspace(1e-2,10,10)
    # theta = params['QoS_exponent']                  # QoS exponent
    
    
    # 參考通訊環境 paper 的位置
    Pos_BS = np.array([params['Position']['BS']['x'], params['Position']['BS']['y'], params['Position']['BS']['z']])       # Position of BS
    Pos_RIS = np.array([params['Position']['RIS']['x'], params['Position']['RIS']['y'], params['Position']['RIS']['z']])    # Position of RIS
  
    # transmit power of each UE 
    Power_UE = np.ones(NumUE) * UE_initial_power           

    # Environment
    MuMIMO_env = envMuMIMO(NumBSAnt, NumRISEle, NumUE)          
    
    # Others
    ArrayShape_BS = [NumBSAnt, 1, 1]    # BS is equipped with a ULA that is placed along the direction [1, 0, 0] (i.e., x-axis)
    ArrayShape_RIS = [1, NumRISEle, 1]  # IRS is a ULA, which is placed along the direction [0, 1, 0] (i.e., y-axis)
    ArrayShape_UE = [1, 1, 1]           # UE is with 1 antenna 

    # ===================================== START =====================================
    
    # Initialization
    rewards = np.zeros(EPISODES)
    
    # Initialization for phase shifts
    RefVector = np.exp(1j * pi * np.zeros((1, NumRISEle)))       
   


    Pos_UE = np.zeros((NumUE, 3))       # Position of UE
    for i in range(NumUE):
        Pos_UE[i][0] = params['Position']['UE']['x']               # x = 50
        Pos_UE[i][1] = params['Position']['UE']['y']               # y = 50
        Pos_UE[i][2] = params['Position']['UE']['z']                # z = 1     


    UE_power = np.ones(NumUE) * UE_initial_power
    
    capacity_list = []
    for theta in thetas:
        for epi in range(EPISODES):
            # Initialization
            Loss_seq_block = np.zeros(num_steps)
        

            # Path loss for large-scale fading (position dependent)
            pathloss_U2B, pathloss_R2B, pathloss_U2R = MuMIMO_env.H_GenPL(Pos_BS, Pos_RIS, Pos_UE)
            
            # LOS channel for Rician fading (position dependent)
            H_U2B_LoS, H_R2B_LoS, H_U2R_LoS = MuMIMO_env.H_GenLoS(Pos_BS, Pos_RIS, Pos_UE, ArrayShape_BS, ArrayShape_RIS, ArrayShape_UE)    
        
            
            # Each block update NLOS channel for Rician fading
            step = 0
            tqdm.write("Episode------ %d" % (epi))
            load_remaining = LocalDataSize
            for step in tqdm(range(num_steps)):
                # block = 0
                time_step = epi*num_steps + step
                
                # Rician fading NLOS channel 
                H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS = MuMIMO_env.H_GenNLoS()       
                                
                # Overall channel with large-scale fading and small-scale fading (Rician fading)              
                H_U2B_Ric, H_R2B_Ric, H_U2R_Ric = MuMIMO_env.H_RicianOverall(K_U2B, K_R2B, K_U2R, H_U2B_LoS, H_R2B_LoS, H_U2R_LoS, H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS, pathloss_U2B, pathloss_R2B, pathloss_U2R)

                # RefVector = np.exp(1j * np.radians([0, 0, 0, 0, 0, 0, 0, 0]))
                # RefVector = np.exp(1j * np.radians([0, -135, 90, -45]))
                RefVector = np.exp(1j * np.radians([0, 180, 90, -45, 180, 45, -90, 135]))
                # RefVector = np.exp(1j * np.radians( [   0, -135,   90,  -45, -180,   45,  -90,  135,   45, -135,  135,    0, -135,   90,  -45, -180.]))
                # RefVector = np.exp(1j * np.radians( [   0, -180,   90,  -45, -180,   45, -135,  180,   45,  -90,  135,    0, -90,   90,  -45,  180,   45,  -90,  135,    0, -135,  135,  -45, -135, 45,  -90,  180,   45,  -90,  135,    0, -135]))
                
                
                H_Ric_combined = MuMIMO_env.H_Comb(H_U2B_Ric, H_R2B_Ric, H_U2R_Ric, RefVector)          # Use the channel with adjusted phase
                snr, achievable_rate, capacity, _ = MuMIMO_env.SNR_Throughput(H_Ric_combined, BW, noise, UE_power)  
                effective_achievable_rate = (-1 / theta)*np.log(np.mean(np.exp(-1 * theta * achievable_rate)))
                effective_capacity = (-1 / theta)*np.log(np.mean(np.exp(-1 * theta * capacity)))
                # 理論上上面那一條才是對的  但因為有了負號後面的數字會變得很小  造成對0取log的情況  所以我們把負號拔ㄌ
                # effective_capacity = (1 / theta)*np.log(np.mean(np.exp(1 * theta * capacity)))
                capacity_list.append(effective_achievable_rate * time_per_step * BW)
                # capacity_list.append(effective_capacity * time_per_step)

        print("EC lower bound with theta_{}: {}".format( theta, np.mean(capacity_list) / 0.05) )
        # print("Recommeded data size: ", ((np.mean(capacity_list) * (num_steps) // 100) + 1) * 100)
        # print("Recommeded data size: ", ((np.mean(capacity_list) * (num_steps) // 100)) * 100)

