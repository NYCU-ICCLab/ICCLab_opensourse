#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:31:26 2023

@author: dddd
"""

import numpy as np
import pandas as pd
import math as ma
import time
from pyomo.environ import *
import json


with open('./param.json', 'r') as f:
    params = json.load(f)

def BB(NumBSAnt, NumRISEle, NumUE, Bandwidth, total_steps, remaining_step, sigma2, load_remaining, D_max, H_U2B_Ric, H_R2B_Ric, H_U2R_Ric, initial_power, initial_theta):
    
    P_max = 200
    
    T_Conbime = np.zeros([NumBSAnt*NumUE, NumRISEle], dtype = 'complex_')    
    flag_row = 0
    for idx_R2B_i in range(0, NumBSAnt*NumUE):
        
        for idx_RISEle in range(0, NumRISEle):
            T_Conbime[idx_R2B_i][idx_RISEle] = H_R2B_Ric[int(idx_R2B_i/NumUE)][idx_RISEle]*H_U2R_Ric[idx_RISEle][(flag_row%NumUE)]
        
        flag_row += 1

    theta = 1e-2
    #----------------------------------------------------Solver model-------------------------------------------------------------#

    def obj_function(model):  
        EC = (-1 / theta) * log( ( sum(  exp( -1 * theta * Bandwidth * ( log( 1 + ((model.power[User_id]*RIS_square(model, User_id))/sigma2) )/log(2) ) ) for User_id in range(NumUE) ) / NumUE ) )
        return (EC*NumUE) / sum(model.power[User_id] for User_id in range(NumUE))
    
    def RIS_received_real(model, idx_risreal):
        return sum((T_Conbime[idx_risreal][idx_ele].real*cos(phase_shift_discrete(model, idx_ele)) - T_Conbime[idx_risreal][idx_ele].imag*sin(phase_shift_discrete(model, idx_ele))) for idx_ele in range(0, NumRISEle))
        
    def RIS_received_imag(model, idx_risimag):
        return sum((T_Conbime[idx_risimag][idx_ele].real*sin(phase_shift_discrete(model, idx_ele)) + T_Conbime[idx_risimag][idx_ele].imag*cos(phase_shift_discrete(model, idx_ele)))  for idx_ele in range(0, NumRISEle))
    
    def RIS_square(model, idx_square_UE):
        return sum((H_U2B_Ric[idx_BS][idx_square_UE].real + RIS_received_real(model, idx_square_UE + idx_BS*NumUE))**2 + (H_U2B_Ric[idx_BS][idx_square_UE].imag +RIS_received_imag(model, idx_square_UE + idx_BS*NumUE))**2 for idx_BS in range(0, NumBSAnt))
    
    def phase_shift_discrete(model, idx_phase):
        return (2*ma.pi*model.theta_discrete[idx_phase])/(params['RISActionSpace'] - 1)     # Fixed 8
        # return (2*ma.pi*model.theta_discrete[idx_phase])/(NumRISEle)     # Fixed 8

    # def EC_constraints(model, user_id):
    def EC_constraints(model):
        EC = (-1 / theta) * \
                log(  \
                    ( sum(  exp( -1 * theta * Bandwidth * ( log( 1 + ((model.power[User_id] * RIS_square(model, User_id)) / sigma2) )/log(2) ) ) for User_id in range(NumUE) ) / NumUE )
                )

        return  (0.05)*EC >= load_remaining[0] / remaining_step 
        ## 可傳輸時間*傳輸速率 >= 一個step分配到的load

    
    model = ConcreteModel(name="BB_EEE")
    

    model.power = Var([i_UE for i_UE in range(NumUE)], bounds=(3,P_max), within=Integers, initialize = initial_power)
    # model.power = Var([i_UE for i_UE in range(NumUE)], bounds=(P_max,P_max), within=Integers, initialize = P_max)
    model.theta_discrete = Var([i for i in range(NumRISEle)], bounds=(-(0.5*params['RISActionSpace'] - 1), (0.5*params['RISActionSpace'] - 1)), within=Integers, initialize = initial_theta)
    # model.theta_discrete = Var([i for i in range(NumRISEle)], bounds=(0, 0), within=Integers, initialize = initial_theta)
    #如果是 within=Integers的情況下
    # bound = (-0.5*RIS_Element,RIS_Element), 代表我們可以使用的相位有RIS_Element這麼多種，在-pi到pi之間
    # Ex: RIS_Element = 8
    # 代表我們可以使用的相位有8+1種，在-pi到pi之間
    # 分別是 -pi, -6/8*pi, -4/8*pi, -2/8*pi, 0, 2/8*pi, 4/8*pi, 6/8*pi, pi
    # 不過我們目前是用Reals，所以不用管這個
    
    model.cons = ConstraintList()
    
    model.cons.add(EC_constraints(model))
    
    model.obj = Objective(expr=obj_function, sense=maximize)  #-----------------setting objective function-------------------#

    solver_path = '/usr/bin/bonmin'

    opt = SolverFactory('bonmin', executable=solver_path)
    opt.options['bonmin.algorithm'] = 'B-BB'
    opt.options['print_level'] = 0
    # opt.options['bonmin.time_limit'] = 0.05
    
    opt.options['mu_strategy'] = 'adaptive'
    
    results = opt.solve(model, tee=False)
    # results = opt.solve(model, tee=True)
    # results.write()
    
    #------------------------------------------------------------------------------------------------------------------------------#
    # 求解模型后
    
    BB_status = results.solver.termination_condition
    BB_objective = model.obj.expr()
    model_theta_BB = np.array(list(model.theta_discrete.get_values().values()))
    model_power_BB = np.array(list(model.power.get_values().values()))
    RIS_matrx_multiply = H_R2B_Ric @ (np.diag(np.exp(1j*(2*ma.pi*model_theta_BB/(NumRISEle))))) @ H_U2R_Ric
    
    Total_link = H_U2B_Ric + RIS_matrx_multiply
    
    Diag_square = np.diag(Total_link.conj().T @ Total_link)    
    BB_time = results.solver.time
    return BB_status, BB_objective, model_power_BB, model_theta_BB, BB_time

