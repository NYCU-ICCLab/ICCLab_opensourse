import time
import json
import math
import random
import numpy as np
from math import *
import pandas as pd
from tqdm import tqdm
from BB_EEE import BB
from MuMIMOClass import *
import matplotlib.pyplot as plt
from collections import namedtuple

with open('./param.json', 'r') as f:
    params = json.load(f)

seed = params["random_seed"]
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    # Simulation Parameters
    EPISODES = params['BB']["EPISODES"]
    time_per_episode = 1                                    # update slow fading(pathloss)/UE position every 1s
    time_per_step = 0.05                                    # update fast fading(rayleigh/rician fading) every 0.05s
    num_steps = params["NUM_STEPS"]                         # each EPISODES has 20 steps
    NumBSAnt = params['NumBsAnt']                           # M (BS antenna number)
    NumRISEle = params['NumRISEle']                         # L (RIS element number)
    NumUE = params['NumUE']                                 # K (user number)
    UE_power_selection = params['UserActionSpace']          # [0, 3, 10, 200]  (mW)   
    LocalDataSize = params['LocalDataSize']         
    BW = params['Bandwidth']                                # bandwidth = 100MHz  把後面10^6都省略了 不然在計算EC的log的時候會出問題 哭阿
    noise = 10**(-104/10)                                   # noise power = -104dBm = 10**(-104/10) mW
    UE_initial_power = params['Initial_Power']                        

    K_U2B = params['K_U2B']                                 # Rician factor: K-factor, if K = 0, Rician equal to Rayleigh 
    K_R2B = params['K_R2B']                                 
    K_U2R = params['K_U2R']                                 
    D_max = 1
    QoS_exponent = params['QoS_exponent']                   # QoS exponent

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
    RefVector = np.exp(1j * pi * np.zeros(NumRISEle)) 
    zeroPhaseShiftVector = np.exp(0j * pi * np.zeros((1, NumRISEle)))       
    print(np.angle(zeroPhaseShiftVector, deg=True))      

    # Each episode is new situation (UE's position & large scale fading & LOS channel for Rician fading)
    initial_power = 3
    initial_theta = 0
    BB_obj_arr = np.zeros(num_steps)
    BB_status_arr = []
    BB_power_arr = []
    BB_timeTotal = 0

    # 先固定用戶位置，因為如果位置不同 path loss就不同 這樣delay的distribution就不同
    Pos_UE = np.zeros((NumUE, 3))       # Position of UE
    UE_mobility_range = 30
    for i in range(NumUE):
        Pos_UE[i][0] = params['Position']['UE']['x'] + np.random.uniform(-UE_mobility_range, UE_mobility_range)            # x = 50
        Pos_UE[i][1] = params['Position']['UE']['y'] + np.random.uniform(-UE_mobility_range, UE_mobility_range)              # y = 50
        Pos_UE[i][2] = params['Position']['UE']['z']                # z = 1     

    # distance between user and RIS + distance between RIS and BS
    distance = np.linalg.norm(Pos_UE - Pos_RIS, axis=1) + np.linalg.norm(Pos_RIS - Pos_BS)

    epi = 0
    #using while and tqdm to show the progress bar
    while(epi < EPISODES):
        # Path loss for large-scale fading (position dependent)
        pathloss_U2B, pathloss_R2B, pathloss_U2R = MuMIMO_env.H_GenPL(Pos_BS, Pos_RIS, Pos_UE)
        
        # LOS channel for Rician fading (position dependent)
        H_U2B_LoS, H_R2B_LoS, H_U2R_LoS = MuMIMO_env.H_GenLoS(Pos_BS, Pos_RIS, Pos_UE, ArrayShape_BS, ArrayShape_RIS, ArrayShape_UE)    
        
        # Each block update NLOS channel for Rician fading
        step = 0
        load_remaining = LocalDataSize*np.ones(NumUE)

        while(step < num_steps):
            # block = 0
            time_step = epi*num_steps + step
            
            # Rician fading NLOS channel 
            H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS = MuMIMO_env.H_GenNLoS()       
            
            DQN_BB_start = time.process_time()
            
            # Overall channel with large-scale fading and small-scale fading (Rician fading)              
            H_U2B_Ric, H_R2B_Ric, H_U2R_Ric = MuMIMO_env.H_RicianOverall(K_U2B, K_R2B, K_U2R, H_U2B_LoS, H_R2B_LoS, H_U2R_LoS, H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS, pathloss_U2B, pathloss_R2B, pathloss_U2R)
            BB_status, BB_obj_arr[step], power_BB, theta_BB, BB_probtime = BB(NumBSAnt, NumRISEle, NumUE, BW, num_steps, num_steps - step, noise, load_remaining, D_max, H_U2B_Ric, H_R2B_Ric, H_U2R_Ric, initial_power, initial_theta)
            BB_timeuse = BB_probtime
            BB_timeTotal += BB_timeuse
            if BB_status == "optimal":
                reflect_phase = np.exp(theta_BB * 2 * pi * 1j / (params['RISActionSpace'] - 1))
                combined_channel = MuMIMO_env.H_Comb(H_U2B_Ric, H_R2B_Ric, H_U2R_Ric, reflect_phase)
                snr, achievable_rate, capacity, _ = MuMIMO_env.SNR_Throughput(combined_channel, BW, noise, power_BB)
                BB_power_arr.append(power_BB)
                EC = BB_obj_arr[step]*np.sum(power_BB) / NumUE # BB計算的是整個系統的EEE，我們要換成單一UE的所以除以NumUE
                if(np.sum(Power_UE[0]) != 0):
                    tqdm.write("episode: {}, step: {}, EC: {:.2f}, power_consumption: {}, RIS phase shift angle: {}".format(epi, step, EC, np.sum(power_BB), np.angle(reflect_phase, deg=True)))      
                load_remaining -= (EC*time_per_step)
                load_remaining[load_remaining < 0] = 0

                step += 1
            else:    # infeasible solution
                EC = BB_obj_arr[step]*np.sum(power_BB) / NumUE # BB計算的是整個系統的EEE，我們要換成單一UE的所以除以NumUE

        rewards[epi] = sum(BB_obj_arr)
        epi += 1
    
    print("RIS elements:{}, avg_reward: {:.2f}".format(NumRISEle, sum(rewards) / len(rewards)) )

    print("Time use {:.2f}s".format((BB_timeTotal) / (EPISODES)))