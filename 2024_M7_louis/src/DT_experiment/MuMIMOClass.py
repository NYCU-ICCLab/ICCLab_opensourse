"""
Multi-user MIMO, environment
"""

from __future__ import division
import numpy as np
import math
import cmath

seed = 3
np.random.seed(seed)

class envMuMIMO:
    def __init__(self, NumBSAnt, NumRISEle, NumUE):
        self.NumBSAnt = NumBSAnt    # M
        self.NumRISEle = NumRISEle  # L
        self.NumUE = NumUE          # K

    # JSAC absolute phase shift (DFT vectors)
    def DFT_matrix(self, N_point):  # N_point: 決定有多少種相位，if N_point=4，則從 0*pi/2 ~ 3*pi/2
        n, m = np.meshgrid(np.arange(N_point), np.arange(N_point))
        omega = np.exp(-2 * math.pi * 1j / N_point)
        W = np.power( omega, n * m ) 
        return W
    
    # 用於生成 Steering Vector
    def SubSteeringVec(self, Angle, NumAnt):
        SSV = np.exp(1j * Angle * math.pi * np.arange(0, NumAnt, 1))
        SSV = SSV.reshape(-1, 1)    # 行的元素數量固定為1，並自動生成列 (-1)
        return SSV
    
    # LoS channel response, which is position dependent 
    def ChannelResponse(self, Pos_A, Pos_B, ArrayShape_A, ArrayShape_B):   
        Dis_AB = np.linalg.norm(Pos_A - Pos_B)                                  ## distance between AB (2-norm: 平方合開根號)
        DirVec_AB = (Pos_A - Pos_B) / Dis_AB                                    ## normalized direction vector (每個位置上放的是向量投影到 xyz 軸與向量夾角的 cosine values)
        SteeringVectorA = np.kron(self.SubSteeringVec(DirVec_AB[0], ArrayShape_A[0]), self.SubSteeringVec(DirVec_AB[1], ArrayShape_A[1]))
        SteeringVectorA = np.kron(SteeringVectorA, self.SubSteeringVec(DirVec_AB[2], ArrayShape_A[2]))  # 根據天線所放的軸，計算它的 steering vectoor，其他部分只會是1，所以做 kron 答案不影響
        SteeringVectorB = np.kron(self.SubSteeringVec(DirVec_AB[0], ArrayShape_B[0]), self.SubSteeringVec(DirVec_AB[1], ArrayShape_B[1]))
        SteeringVectorB = np.kron(SteeringVectorB, self.SubSteeringVec(DirVec_AB[2], ArrayShape_B[2]))  # 根據天線所放的軸，計算它的 steering vectoor，其他部分只會是1，所以做 kron 答案不影響
        SteeringVectorB_H = np.matrix.getH(SteeringVectorB)
        H_LoS_matrix = SteeringVectorA @ SteeringVectorB_H                      # size_A x 1 的矩陣 @ 1 x size_B 的矩陣
        return H_LoS_matrix

    # Generate LOS channel for Rician fading
    def H_GenLoS(self, Pos_BS, Pos_RIS, Pos_UE, ArrayShape_BS, ArrayShape_RIS, ArrayShape_UE):   # for multi-user channel (2)
        H_R2B_LoS = self.ChannelResponse(Pos_BS, Pos_RIS, ArrayShape_BS, ArrayShape_RIS)
        H_U2B_LoS = np.zeros((self.NumBSAnt, self.NumUE), dtype = complex)
        H_U2R_LoS = np.zeros((self.NumRISEle, self.NumUE), dtype = complex) 
        # H = [h1, h2, · · · , hK]，一個UE對應一行
        for i in range(self.NumUE):
            h_U2B_LoS = self.ChannelResponse(Pos_BS, Pos_UE[i], ArrayShape_BS, ArrayShape_UE)    # NumBSAnt x 1
            H_U2B_LoS[:, i] = h_U2B_LoS.reshape(-1)                                              # .reshape(-1): 固定一列，自動生成行。這步驟是將向量放入該矩陣的第 i 行
            h_U2R_LoS = self.ChannelResponse(Pos_RIS, Pos_UE[i], ArrayShape_RIS, ArrayShape_UE)  # NumRISEle x 1   
            H_U2R_LoS[:, i] = h_U2R_LoS.reshape(-1)
        return H_U2B_LoS, H_R2B_LoS, H_U2R_LoS
    
    # Generate Rayleigh channel fading or NLOS channel for Rician fading
    def H_GenNLoS(self, ):
        H_U2B_NLoS = 1 / math.sqrt(2) * (np.random.normal(0, 1, size=(self.NumBSAnt, self.NumUE)) + 1j * np.random.normal(0, 1, size=(self.NumBSAnt, self.NumUE)))
        H_R2B_NLoS = 1 / math.sqrt(2) * (np.random.normal(0, 1, size=(self.NumBSAnt, self.NumRISEle)) + 1j * np.random.normal(0, 1, size=(self.NumBSAnt, self.NumRISEle)))
        H_U2R_NLoS = 1 / math.sqrt(2) * (np.random.normal(0, 1, size=(self.NumRISEle, self.NumUE)) + 1j * np.random.normal(0, 1, size=(self.NumRISEle, self.NumUE)))
        return H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS
    
    # Generate Large-scale path loss
    def H_GenPL(self, Pos_BS, Pos_RIS, Pos_UE):   
        # Large-scale pass loss (參數參考 Energy-Efficient Federated Learning With Intelligent Reflecting Surface)
        PL_0 = 10**(-30/10);                                # dB the channel gain at the reference distance
        d_R2B = np.linalg.norm(Pos_RIS - Pos_BS)            # distance from the RIS to BS  
        pathloss_R2B = math.sqrt(PL_0 * (d_R2B)**(-2.2));   # Large-scale pass loss from RIS to BS (α is the path loss exponent, α = 2.2)                                                                
        d_U2B = np.zeros(self.NumUE)  
        d_U2R = np.zeros(self.NumUE) 
        pathloss_U2B = np.zeros(self.NumUE)  
        pathloss_U2R = np.zeros(self.NumUE)
        for k in range(self.NumUE):
            d_U2B[k] = np.linalg.norm(Pos_UE[k] - Pos_BS)      # distance from the user k to the BS  
            d_U2R[k] = np.linalg.norm(Pos_UE[k] - Pos_RIS)      # distance from the user k to the RIS  
            pathloss_U2B[k] = math.sqrt(PL_0 * (d_U2B[k])**(-3.5))   # Large-scale pass loss from user k to BS (α is the path loss exponent, α = 4)
            pathloss_U2R[k] = math.sqrt(PL_0 * (d_U2R[k])**(-2.2))   # Large-scale pass loss from user k to RIS (α is the path loss exponent, α = 2)
        return pathloss_U2B, pathloss_R2B, pathloss_U2R
    
    # The channel include large-scale fading and small-scale fading.
    def H_RicianOverall(self, K_U2B, K_R2B, K_U2R, H_U2B_LoS, H_R2B_LoS, H_U2R_LoS, H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS, pathloss_U2B, pathloss_R2B, pathloss_U2R):
        H_R2B_temp = (math.sqrt(1 / (K_R2B + 1)) * H_R2B_NLoS + math.sqrt(K_R2B / (K_R2B + 1)) * H_R2B_LoS)
        H_U2B_temp = (math.sqrt(1 / (K_U2B + 1)) * H_U2B_NLoS + math.sqrt(K_U2B / (K_U2B + 1)) * H_U2B_LoS) 
        H_U2R_temp = (math.sqrt(1 / (K_U2R + 1)) * H_U2R_NLoS + math.sqrt(K_U2R / (K_U2R + 1)) * H_U2R_LoS)
        H_R2B_Ric = pathloss_R2B * H_R2B_temp
        H_U2B_Ric = np.zeros((self.NumBSAnt, self.NumUE), dtype = complex)
        H_U2R_Ric = np.zeros((self.NumRISEle, self.NumUE), dtype = complex) 
        for i in range(self.NumBSAnt):
            for k in range(self.NumUE):
                H_U2B_Ric[i][k] = pathloss_U2B[k] * H_U2B_temp[i][k]
        for i in range(self.NumRISEle):
            for k in range(self.NumUE):
                H_U2R_Ric[i][k] = pathloss_U2R[k] * H_U2R_temp[i][k]
        return H_U2B_Ric, H_R2B_Ric, H_U2R_Ric
        
    # The channel include large-scale fading and small-scale fading.
    def H_RayleighOverall(self, H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS, pathloss_U2B, pathloss_R2B, pathloss_U2R):
        H_R2B_Ray = pathloss_R2B * H_R2B_NLoS
        H_U2B_Ray = np.zeros((self.NumBSAnt, self.NumUE), dtype = complex)
        H_U2R_Ray = np.zeros((self.NumRISEle, self.NumUE), dtype = complex) 
        for i in range(self.NumBSAnt):
            for k in range(self.NumUE):
                H_U2B_Ray[i][k] = pathloss_U2B[k] * H_U2B_NLoS[i][k]
        for i in range(self.NumRISEle):
            for k in range(self.NumUE):
                H_U2R_Ray[i][k] = pathloss_U2R[k] * H_U2R_NLoS[i][k]
        return H_U2B_Ray, H_R2B_Ray, H_U2R_Ray
    
    # Effective kth-device-BS combined channel
    def H_Comb(self, H_U2B, H_R2B, H_U2R, RefVector): 
        RefPattern_matrix = np.diag(RefVector)  
        H = H_U2B + H_R2B @ RefPattern_matrix @ H_U2R                                           # @: 矩陣乘法   # *:對應元素相乘    
        return H

    # Throughput of each user (SNR)
    def SNR_Throughput(self, H_Ric, BW, noise, Power_UE): 

        H_Ric_gain2 = abs(np.conj(H_Ric.T) @ H_Ric) # 取決對值讓複數變長度
        H_Ric_gain2 = np.diag(H_Ric_gain2)          # 從 KxK 的矩陣中取出 Kx1 的向量，分別代表每個 UE 對應的 Channel gain 的平方
        SigPower = Power_UE * H_Ric_gain2           # Kx1 的向量，分別代表每個 UE 對應的 Channel gain 的平方在乘上所對應的 UE 的發射功率
        # print("H_Ric_gain2: ", H_Ric_gain2)
        SNR = SigPower / (0 + noise)
        Rate = np.log2(1 + SNR)
        Rate = Rate.reshape(1, self.NumUE)      # 回傳後才可以放到STATE
        Throughput = BW * np.log2(1 + SNR)

        # 計算 EE ，不發射功率的 EE 為 0
        # Power_UE = Power_UE.reshape(1, self.NumUE)       # 調整後才可以被下面的拿去用    
        EE = np.ones(self.NumUE)
        EE[Power_UE == 0] = 0                        # 把功率為0的對應位置的EE設為0
        EE[Power_UE != 0] = Throughput[Power_UE != 0] / Power_UE[Power_UE != 0]     # 計算功率不為0的對應位置的EE
        EE = EE.reshape(1, self.NumUE)          # 回傳後才可以放到STATE
        return SNR, Rate, Throughput, EE
    
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Throughput of each user (SINR)
    def SINR_Throughput(self, H_Ric, H_Ray, BW, noise, Power_UE): 
        # 分子 Rician 分母 Rayleigh (分子分母不同通道)
        H_Ric_gain2 = abs(np.conj(H_Ric.T) @ H_Ric) # 取決對值讓複數變長度
        H_Ric_gain2 = np.diag(H_Ric_gain2)          # 從 KxK 的矩陣中取出 Kx1 的向量，分別代表每個 UE 對應的 Channel gain 的平方
        H_Ray_gain2 = abs(np.conj(H_Ray.T) @ H_Ray)
        H_Ray_gain2 = np.diag(H_Ray_gain2)          # 從 KxK 的矩陣中取出 Kx1 的向量，分別代表每個 UE 對應的 Channel gain 的平方
        SigPower = Power_UE * H_Ric_gain2           # Kx1 的向量，分別代表每個 UE 對應的 Channel gain 的平方在乘上所對應的 UE 的發射功率
        Power_Channel_gain = Power_UE * H_Ray_gain2
        Power_Channel_gain_sum = Power_Channel_gain.sum()
        IntfPower = (Power_Channel_gain_sum - Power_Channel_gain)    
        SINR = SigPower / (IntfPower + noise)
        Rate = np.log2(1 + SINR)
        Throughput = BW * np.log2(1 + SINR)
        return SINR, Rate, Throughput
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    




        