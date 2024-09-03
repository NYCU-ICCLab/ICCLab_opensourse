import json
import math
from math import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from BB_EEE import BB
from MuMIMOClass import *
import matplotlib.pyplot as plt
from collections import namedtuple


import random
import time



with open('./param.json', 'r') as f:
    params = json.load(f)

seed = params["random_seed"]
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    # Simulation Parameters
    EPISODES = params['BB']["EPISODES"]
    time_per_episode = 1                                   # update slow fading(pathloss)/UE position every 1s
    time_per_step = 0.05                                # update fast fading(rayleigh/rician fading) every 0.05s
    num_steps = int(time_per_episode/time_per_step)   # each EPISODES has 20 steps
    steps = EPISODES*num_steps
    mini_batch_step = 1                             # freq to update policy network            
    NumBSAnt = params['NumBsAnt']                   # M (BS antenna number)
    NumRISEle = params['NumRISEle']                 # L (RIS element number)
    NumUE = params['NumUE']                         # K (user number)
    UE_power_selection = 4                          # [0, 3, 10, 200]  (mW)   
    power_space = UE_power_selection**NumUE         # action space of power selection (four power levels for each UE)
    UE_position_variance = 3                        # range of UE distribution
    PhaseShift_bit = 3                              # number of bits of each RIS phase shift
    stages = 2 ** PhaseShift_bit                    # 8 phase shift stages
    LocalDataSize = params['LocalDataSize']         # 900 Mb = 25 * 10**6 * 8 bit
    BW = 100                                          # bandwidth = 1MHz  把後面10^6都省略了 不然在計算EC的log的時候會出問題 哭阿
    noise = 10**(-104/10)                           # noise power = -104dBm = 10**(-104/10) mW
    UE_initial_power = 3                            # UE power(dBm): 0~30dBm  = 0~1000mW
    # UE_initial_power = 200                            # UE power(dBm): 0~30dBm  = 0~1000mW
    K_U2B = 10                                      # Rician factor: K-factor, if K = 0, Rician equal to Rayleigh 
    K_R2B = 10
    K_U2R = 10
    D_max = 1
    QoS_exponent = 1e-2
    
    
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
    initial_power_offset = 0
    initial_theta = 0
    initial_theta_offset = 0
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
    # print("distance: ", distance)
    
    epi = 0
    # for epi in range(EPISODES):
    #using while and tqdm to show the progress bar
    while(epi < EPISODES):
    # for epi in tqdm(range(EPISODES)):
        # Initialization
        Loss_seq_block = np.zeros(num_steps)
    

        # Path loss for large-scale fading (position dependent)
        pathloss_U2B, pathloss_R2B, pathloss_U2R = MuMIMO_env.H_GenPL(Pos_BS, Pos_RIS, Pos_UE)
        
        # LOS channel for Rician fading (position dependent)
        H_U2B_LoS, H_R2B_LoS, H_U2R_LoS = MuMIMO_env.H_GenLoS(Pos_BS, Pos_RIS, Pos_UE, ArrayShape_BS, ArrayShape_RIS, ArrayShape_UE)    
    
        
        # Each block update NLOS channel for Rician fading
        step = 0
        # tqdm.write("Episode------ %d" % (epi))
        load_remaining = LocalDataSize*np.ones(NumUE)
        # load_remaining = LocalDataSize
        # for step in tqdm(range(num_steps)):
        # for step in range(num_steps):
        while(step < num_steps):
            # block = 0
            time_step = epi*num_steps + step
            
            # Rician fading NLOS channel 
            H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS = MuMIMO_env.H_GenNLoS()       
            
            DQN_BB_start = time.process_time()
            
            # Overall channel with large-scale fading and small-scale fading (Rician fading)              
            H_U2B_Ric, H_R2B_Ric, H_U2R_Ric = MuMIMO_env.H_RicianOverall(K_U2B, K_R2B, K_U2R, H_U2B_LoS, H_R2B_LoS, H_U2R_LoS, H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS, pathloss_U2B, pathloss_R2B, pathloss_U2R)
            BB_status, BB_obj_arr[step], power_BB, theta_BB, BB_probtime = BB(NumBSAnt, NumRISEle, NumUE, BW, num_steps, num_steps - step, noise, load_remaining, D_max, H_U2B_Ric, H_R2B_Ric, H_U2R_Ric, initial_power, initial_theta)
            # BB_status, BB_obj_arr[step], power_BB, theta_BB, BB_probtime, success_rate = BB(NumBSAnt, NumRISEle, NumUE, BW, num_steps, num_steps - step, noise, load_remaining, D_max, H_U2B_Ric, H_R2B_Ric, H_U2R_Ric, initial_power+initial_power_offset, initial_theta + initial_theta_offset,propagation_delay)
            


            BB_timeuse = BB_probtime
            BB_timeTotal += BB_timeuse
            
            if BB_status == "optimal":
                # print("theta: ", theta_BB)
                # reflect_phase = theta_BB * 2 * pi * 1j / NumRISEle
                # reflect_phase = np.exp(theta_BB * 2 * pi * 1j / NumRISEle)
                reflect_phase = np.exp(theta_BB * 2 * pi * 1j / (params['RISActionSpace'] - 1))
                # print("reflect_phase: ", reflect_phase)

                combined_channel = MuMIMO_env.H_Comb(H_U2B_Ric, H_R2B_Ric, H_U2R_Ric, reflect_phase)
                snr, achievable_rate, capacity, _ = MuMIMO_env.SNR_Throughput(combined_channel, BW, noise, power_BB)
                # print("EC: ", -1/QoS_exponent * log(np.sum(exp(-1 * QoS_exponent * capacity[user_id]) for user_id in range(NumUE)) / NumUE))  
                BB_power_arr.append(power_BB)
                EC = BB_obj_arr[step]*np.sum(power_BB) / NumUE # BB計算的是整個系統的EEE，我們要換成單一UE的所以除以NumUE
                if(np.sum(Power_UE[0]) != 0):
                    # print("Capacity: ", capacity*0.05)
                    # print("Effective capacity: ", -1/QoS_exponent * log(np.sum(exp(-1 * QoS_exponent * capacity[user_id]) for user_id in range(NumUE)) / NumUE))
                    tqdm.write("episode: {}, step: {}, EC: {:.2f}, power_consumption: {}, RIS phase shift angle: {}".format(epi, step, EC, np.sum(power_BB), np.angle(reflect_phase, deg=True)))       


                load_remaining -= (EC*time_per_step)
                load_remaining[load_remaining < 0] = 0

                step += 1
                initial_power_offset = 0
            else:    # infeasible solution
                EC = BB_obj_arr[step]*np.sum(power_BB) / NumUE # BB計算的是整個系統的EEE，我們要換成單一UE的所以除以NumUE
                # print("episode: {}, step: {}, EC: {:.2f}, power_consumption: {}".format(epi, step, EC, np.sum(power_BB)))       
                # for i in range(NumRISEle):
                    # print("theta_{}: {:.2f}".format(i, theta_BB[i]))
                # print("*********infeasible***********")
                # pass 

        rewards[epi] = sum(BB_obj_arr)
        epi += 1
        # tqdm.write("Episode {}, Load remaining: {}".format(epi, load_remaining))
    
    # print("%e"%(sum(rewards)/len(rewards)))
    print("RIS elements:{}, avg_reward: {:.2f}".format(NumRISEle, sum(rewards) / len(rewards)) )
    # print(sum(rewards) / len(rewards))             # 1e6: 1M 在計算中bandwidth的單位是MHz
    # print("Time use", (BB_timeTotal))        
    print("Time use {:.2f}s".format((BB_timeTotal) / (EPISODES)))
 

    # np.savez('User_{}_BB_datasets.npz'.format(NumUE),H_R2B_Ric=H_R2B_Ric_history, H_U2B_Ric=H_U2B_Ric_history, H_U2R_Ric=H_U2R_Ric_history, H_overall_Ric=H_overall_Ric_history, power=power_history, theta=theta_history, EEE=EEE_history, load_remaining=load_remaining_history, time_remaining=time_remaining_history)