"""
# CHANNEL FADING 
https://github.com/ken0225/Framework-of-Robust-Transmission-Design-for-IRS-Aided-MISO-Communications/blob/master/TSP2020/Channel/Channel.m

# RIS + RL (AND PARAMETERS)
https://github.com/WeiWang-WYS/IRSconfigurationDRL

# RL TECHNIQUES AND PARAMETERS
https://github.com/le-liang/MARLspectrumSharingV2X
"""

import json
import random
from math import *
import copy
import numpy as np
from tqdm import tqdm
from MuMIMOClass import *

# set seed
seed = 3
np.random.seed(seed)
random.seed(seed)

with open('./param.json', 'r') as f:
    params = json.load(f)


if __name__ == "__main__":

    H_U2B_Ric_History = []
    H_R2B_Ric_History = []
    H_U2R_Ric_History = []
    RefVector_History = []
    H_overall_Ric_History = []
    power_History = []
    phase_History = []
    rtg_History = []
    load_remaining_History = []
    time_remaining_History = []

    training_params = params['training']
    K_U2B = params['K_U2B']                                      # Rician factor: K-factor, if K = 0, Rician equal to Rayleigh 
    K_R2B = params['K_R2B']
    K_U2R = params['K_U2R']

    NumBSAnt = params['NumBsAnt']                   # M (BS antenna number)
    NumRISEle = params['NumRISEle']                 # L (RIS element number)
    NumUE = params['NumUE']                         # K (user number)
    # Position of BS & RIS
    Pos_BS = np.array([0, 0, 10])       
    Pos_RIS = np.array([30, 10, 10])   
    Pos_UE = np.zeros((NumUE, 3))       # Position of UE
    for i in range(NumUE):
        Pos_UE[i][0] = params['Position']['UE']['x']               # x = 50
        Pos_UE[i][1] = params['Position']['UE']['y']               # y = 50
        Pos_UE[i][2] = params['Position']['UE']['z']                # z = 1   

    ArrayShape_BS  = [NumBSAnt, 1, 1]                                    # BS is equipped with a ULA that is placed along the direction [1, 0, 0] (i.e., x-axis)
    ArrayShape_RIS = [1, NumRISEle, 1]                                  # RIS is a ULA, which is placed along the direction [0, 1, 0] (i.e., y-axis)
    ArrayShape_UE  = [1, 1, 1]                                           # UE is with 1 antenna 

    violation_prob = params['Violation_probability']           # violation probability: 0.1
    QoS_exponent = params['QoS_exponent']           # QoS exponent: 
    BW = params['Bandwidth']                                      # bandwidth = 100MHz 省略了10^6 不然在計算EC的log的時候會出問題(數值太小)
    noise = 10**(-104/10)                           # noise variance at UE # AWGN: -104dBm 

    # Environment
    MuMIMO_env = envMuMIMO(NumBSAnt, NumRISEle, NumUE)   

    EPISODES = 1000
    num_steps = params["NUM_STEPS"]        # each EPISODES has 20 steps

    UE_power_selection = params['UserActionSpace']                           # [0, 3, 10, 200]  (mW)   
    RIS_phase_selectoin = params['RISActionSpace']                          # [-135, -45, 0, 45, 135] (5 phase shifts)
    LocalDataSize = params['LocalDataSize']         # 900 Mb = 900 * 10**6 bit 省略了10^6 

    # Training Parameters
    Penalty = LocalDataSize
    max_reward = 0

    shiftable_angle = [-180, -135, -90, -45, 0, 45, 90, 135, 180]  # [0, 180, 90, -45, 180, 45, -90, 135]

    # 創建一個字典來映射角度到 ID
    angle_to_id = {angle: idx for idx, angle in enumerate(shiftable_angle)}

    # 定義一個函數來轉換 RefVector 角度到對應的 ID
    def convert_to_ids(vector, angle_to_id):
        return [angle_to_id[angle] for angle in vector]

    print("shiftable_angle: ", shiftable_angle)
    shiftable_radian = np.radians(shiftable_angle)
    shiftable_complex = np.exp(1j * shiftable_radian)

    # Initialization
    achievable_rate = np.zeros((1, NumUE))
    Reward_DQN_seq_episode = np.zeros(EPISODES)

    num_channel_realization = 40

    count = 0

    H_U2B_Ric_history = np.zeros((num_channel_realization, num_steps, NumBSAnt, NumUE), dtype=complex)
    H_R2B_Ric_history = np.zeros((num_channel_realization, num_steps, NumBSAnt, NumRISEle), dtype=complex)
    H_U2R_Ric_history = np.zeros((num_channel_realization, num_steps, NumRISEle, NumUE), dtype=complex)
    RefVector_history = np.zeros((num_channel_realization, num_steps, NumRISEle), dtype=complex)
    rtg_history = np.zeros((num_channel_realization, num_steps))
    power_history = np.zeros((num_channel_realization,num_steps, NumUE))
    phase_history = np.zeros((num_channel_realization,num_steps, NumRISEle))
    load_remaining_history = np.zeros((num_channel_realization, num_steps, NumUE))
    time_remaining_history = np.zeros((num_channel_realization, num_steps, NumUE))
    total_power_consumption = np.zeros((num_channel_realization, NumUE))

    # Each episode is new situation (UE's position & large scale fading & LOS channel for Rician fading)
    for epi in tqdm(range(EPISODES)):

        UE_load = LocalDataSize * np.ones((num_channel_realization, NumUE))    
        rtg = np.zeros((num_channel_realization))
        time_remaining = np.ones((num_channel_realization, NumUE)) * num_steps
        Power_UE = np.ones((NumUE)) 

        RefVector_angle = np.zeros((num_channel_realization, NumRISEle)) # [0, 180, 90, -45, 180, 45, -90, 135]
        RefVector_radian = np.radians(RefVector_angle)
        RefVector = np.exp(1j * RefVector_radian) 

        # Path loss for large-scale fading (position dependent)
        pathloss_U2B, pathloss_R2B, pathloss_U2R = MuMIMO_env.H_GenPL(Pos_BS, Pos_RIS, Pos_UE)

        # LOS channel for Rician fading (position dependent)
        H_U2B_LoS, H_R2B_LoS, H_U2R_LoS = MuMIMO_env.H_GenLoS(Pos_BS, Pos_RIS, Pos_UE, ArrayShape_BS, ArrayShape_RIS, ArrayShape_UE)
        

        for step in range(num_steps):
            H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS = MuMIMO_env.H_GenNLoS()  
            # Overall channel with large-scale fading and small-scale fading (Rician fading)              
            H_U2B_Ric, H_R2B_Ric, H_U2R_Ric = MuMIMO_env.H_RicianOverall(K_U2B, K_R2B, K_U2R, 
                H_U2B_LoS, H_R2B_LoS, H_U2R_LoS, H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS, pathloss_U2B, pathloss_R2B, pathloss_U2R)
            
            # power_ids = np.random.choice([1,2], (NumUE, num_channel_realization), replace=True)
            # power_ids = np.random.choice([1,2,3], (NumUE, num_channel_realization), replace=True)
            if step / num_steps < 0.8:
                power_ids = np.random.choice([1], (NumUE, num_channel_realization), replace=True)
            else:
                power_ids = np.random.choice([3], (NumUE, num_channel_realization), replace=True)
                # power_ids = np.random.choice([2,3], (NumUE, num_channel_realization), replace=True)
            # power_ids = np.random.choice([1,2,3], (NumUE, num_channel_realization), replace=True)
            phase_shift_ids = np.random.choice([i for i in range(len(shiftable_angle))], (NumRISEle, num_channel_realization), replace=True)
            # power_ids = np.random.choice(range(0, UE_power_selection), num_channel_realization, replace=False)
            # print("power_ids: ", power_ids)

            for realization_id in range(num_channel_realization):

                H_R2B_Ric_history[realization_id][step] = H_R2B_Ric
                H_U2B_Ric_history[realization_id][step] = H_U2B_Ric
                H_U2R_Ric_history[realization_id][step] = H_U2R_Ric
                RefVector_history[realization_id][step] = RefVector[realization_id]

                power_id = power_ids[:,realization_id]
                
                if params['UserActionSpace'] == 4:
                    Power_UE[power_id == 0] = 0
                    Power_UE[power_id == 1] = 3
                    Power_UE[power_id == 2] = 10  
                    Power_UE[power_id == 3] = 200
                elif params['UserActionSpace'] == 5:
                    Power_UE[power_id == 0] = 0
                    Power_UE[power_id == 1] = 3
                    Power_UE[power_id == 2] = 10  
                    Power_UE[power_id == 3] = 150  
                    Power_UE[power_id == 4] = 200
                elif params['UserActionSpace'] == 6:
                    Power_UE[power_id == 0] = 0
                    Power_UE[power_id == 1] = 3
                    Power_UE[power_id == 2] = 5
                    Power_UE[power_id == 3] = 10  
                    Power_UE[power_id == 4] = 150  
                    Power_UE[power_id == 5] = 200
                elif params['UserActionSpace'] == 7:
                    Power_UE[power_id == 0] = 0
                    Power_UE[power_id == 1] = 3
                    Power_UE[power_id == 2] = 5
                    Power_UE[power_id == 3] = 10  
                    Power_UE[power_id == 4] = 20  
                    Power_UE[power_id == 5] = 100  
                    Power_UE[power_id == 6] = 200

                if NumRISEle == 4:
                    RefVector[realization_id]= np.exp(1j * np.radians([0, -135, 90, -45]))
                elif NumRISEle == 8:
                    RefVector[realization_id] = np.exp(1j * np.radians([0, 180, 90, -45, 180, 45, -90, 135]))
                elif NumRISEle == 16:
                    RefVector[realization_id]= np.exp(1j * np.radians( [0, -135, 90, -45, -180, 45,  -90,  135, 45, -135,  135,  0, -135, 90, -45, -180.]))
                else:
                    RefVector[realization_id]= np.exp(1j * np.radians( [0, -180, 90, -45, -180, 45, -135,  180, 45,  -90,  135,  0, -90,  90, -45, 180,   45,  -90,  135,    0, -135,  135,  -45, -135, 45,  -90,  180,   45,  -90,  135,    0, -135]))
                # print("shift ids: ", convert_to_ids(np.angle(RefVector[realization_id], deg=True), angle_to_id))
                phase_shift_ids[:,realization_id] = convert_to_ids(np.angle(RefVector[realization_id], deg=True), angle_to_id)

                H_Ric_overall = MuMIMO_env.H_Comb(H_U2B_Ric, H_R2B_Ric, H_U2R_Ric, RefVector[realization_id])          # Use the channel with adjusted phase

                total_power_consumption[realization_id] += np.sum(Power_UE)

                snr, achievable_rate, capacity, _ = MuMIMO_env.SNR_Throughput(H_Ric_overall, BW, noise, Power_UE)  
                reward = 0
                EC = -1 / QoS_exponent * np.log(np.mean(np.exp(-QoS_exponent * capacity)))
                if EC == -0.0:
                    EC = 0   
                if np.sum(Power_UE[0]) != 0:
                    reward = (EC * NumUE) / np.sum(Power_UE)
                else:
                    reward = 0 
                
                if UE_load[realization_id].any() > 0:
                    UE_load[realization_id] -= EC*0.05
                if step < num_steps - 1:
                    rtg[realization_id] += reward
                else:
                    fail = np.zeros((NumUE))     
                    fail[UE_load[realization_id] > 0] = 1                                                                      # UE fail to transmit model will set to 1
                    if np.sum(fail) > violation_prob*NumUE:
                        reward = -LocalDataSize*np.sum(fail) * np.sum(UE_load[realization_id]) / 0.05
                        rtg[realization_id] = reward
                    else:
                        rtg[realization_id] += reward

                power_history[realization_id][step] = power_ids[:,realization_id]
                phase_history[realization_id][step] = phase_shift_ids[:,realization_id]
                rtg_history[realization_id][step] = rtg[realization_id]
                load_remaining_history[realization_id][step] = UE_load[realization_id]
                time_remaining_history[realization_id][step] = time_remaining[realization_id]
                time_remaining[realization_id] -= 1


        # find the top 3 realizations with respect to the rtg
        winner_realizations = np.argsort(rtg_history[:,-1])[:5]

        for winner in winner_realizations:

            if rtg_history[winner][-1] > 0:

                count += 1
                if np.sum(load_remaining_history[winner][-1]) > 0:
                    print("Survived remaining load: ", load_remaining_history[winner][-1])
                else:
                    H_R2B_Ric_history_winner = copy.deepcopy(H_R2B_Ric_history[winner])
                    H_U2B_Ric_history_winner = copy.deepcopy(H_U2B_Ric_history[winner])
                    H_U2R_Ric_history_winner = copy.deepcopy(H_U2R_Ric_history[winner])
                    RefVector_history_winner = copy.deepcopy(RefVector_history[winner])
                    power_history_winner = copy.deepcopy(power_history[winner])
                    phase_history_winner = copy.deepcopy(phase_history[winner])
                    rtg_history_winner = copy.deepcopy(rtg_history[winner])
                    load_remaining_history_winner = copy.deepcopy(load_remaining_history[winner])
                    time_remaining_history_winner = copy.deepcopy(time_remaining_history[winner])
                    H_R2B_Ric_History.append(H_R2B_Ric_history_winner)
                    H_U2B_Ric_History.append(H_U2B_Ric_history_winner)
                    H_U2R_Ric_History.append(H_U2R_Ric_history_winner)
                    RefVector_History.append(RefVector_history_winner)
                    power_History.append(power_history_winner)
                    phase_History.append(phase_history_winner)
                    rtg_History.append(rtg_history_winner)
                    load_remaining_History.append(load_remaining_history_winner)
                    time_remaining_History.append(time_remaining_history_winner)
    
    print("datasets size: ", len(power_History))

    np.savez('./User_{}_with_{}_RIS_complete_datasets.npz'.format(NumUE, NumRISEle),H_R2B_Ric=H_R2B_Ric_History, \
                                                        H_U2B_Ric=H_U2B_Ric_History, \
                                                        H_U2R_Ric=H_U2R_Ric_History, \
                                                        RefVector=RefVector_History, \
                                                        power=power_History, \
                                                        phase=phase_History, \
                                                        RTG=rtg_History, \
                                                        load_remaining=load_remaining_History, \
                                                        time_remaining=time_remaining_History)

    print("Dataset with {} users are generated".format(NumUE))
