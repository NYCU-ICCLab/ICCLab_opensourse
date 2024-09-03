"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

from datetime import datetime
import math
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from MuMIMOClass import *
import torch.optim as optim
from mingpt.utils import sample
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader



logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = "model.ckpt"
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            # print("k: ", k)
            # print("v: ", v)
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.params = config.params

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, best_return):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)
        torch.save(raw_model.state_dict(), "eval_return_{}.ckpt".format(best_return))

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            # loader = DataLoader(data, shuffle=True, pin_memory=True,
            loader = DataLoader(data, shuffle=False, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, r, t) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    power_logits, phase_logits, loss = model(x, y, y, r, t)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

            plt.plot(losses,label='loss at epoch {}'.format(epoch_num))
            # plt.show()
            plt.savefig('epoch_{}-loss.png'.format(epoch_num))
        # best_loss = float('inf')
        
        best_return = -float('inf')

        self.tokens = 0 # counter used for learning rate decay

        for epoch in range(config.max_epochs):

            run_epoch('train', epoch_num=epoch)

            # -- pass in target returns
            print("Evaluation phase")

            # count the evaluation time in seconds
            import time
            time = datetime.now()
            target_return = self.params['testing']['TARGET_RETURN']
            eval_return, fail_rate = self.get_returns(target_return)
            last_time = datetime.now()
            print("Evaluation time: {:.7f}s".format((last_time - time).seconds / self.params['testing']['EPISODES']))
            if eval_return > best_return and eval_return > target_return * 0.95 and fail_rate < 0.1:
                best_return = eval_return


    def get_returns(self, ret):
        self.model.train(False)
        args=Args()
        env = Env(args, self.params)
        env.eval()

        T_rewards, T_fails = [], []
        done = True
        evaluation_epochs = self.params['testing']['EPISODES']
        for i in range(evaluation_epochs):
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(self.model.module, state, 1, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device), params=self.params)
            j = 0
            all_states = state
            actions = sampled_action

            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,:]
                state, reward, done, fail = env.step(action)
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    T_fails.append(fail)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)


                rtgs += [rtgs[-1] - reward]

                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, sample=True, 
                    actions=actions.unsqueeze(0).unsqueeze(-1), 
                    # actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)),params=self.params)
                actions = torch.cat((actions, sampled_action), dim=0)
                # print("Sampled action.shape: ", sampled_action.shape)
        env.close()
        eval_return = sum(T_rewards)/evaluation_epochs
        print("target return: %d, eval return: %d" % (ret, eval_return))
        print("Fails rate: {:.2f}".format(sum(T_fails)/evaluation_epochs))
        # print("fails rate: %.2f%" % (sum(T_fails)/evaluation_epochs)*100)
        self.model.train(True)
        return eval_return, sum(T_fails)/evaluation_epochs


class Env():
    def __init__(self, args, params):
        self.device = args.device
        self.training = True  # Consistent with model training mode
        self.params = params
        self.num_steps = params['NUM_STEPS']
        self.NumBSAnt = params['NumBsAnt']                   # M (BS antenna number)
        self.NumRISEle = params['NumRISEle']                 # L (RIS element number)
        self.NumUE = params['NumUE']                         # K (user number)
        self.K_U2B = params['K_U2B']                                      # Rician factor: K-factor, if K = 0, Rician equal to Rayleigh 
        self.K_R2B = params['K_R2B']
        self.K_U2R = params['K_U2R']
        self.BW = params['Bandwidth']                                        # bandwidth = 100MHz 省略了10^6 不然在計算EC的log的時候會出問題(數值太小)
        self.noise = 10**(-104/10)                           # noise variance at UE # AWGN: -104dBm 
        self.LocalDataSize = params['LocalDataSize']         # 900 Mb = 900 * 10**6 bit 省略了10^6 
        self.QoS_exponent = params['QoS_exponent']          

        # Position of BS & RIS
        self.Pos_BS = np.array([0, 0, 10])       
        self.Pos_RIS = np.array([30, 10, 10])   
        self.Pos_UE = np.zeros((self.NumUE, 3))       # Position of UE
        for i in range(self.NumUE):
            self.Pos_UE[i][0] = params['Position']['UE']['x']               # x = 50
            self.Pos_UE[i][1] = params['Position']['UE']['y']               # y = 50
            self.Pos_UE[i][2] = params['Position']['UE']['z']                # z = 1   

        # transmit power of each UE 
        self.Power_UE = np.ones(self.NumUE)

        # RIS codebook, 
        # self.shiftable_angle = [-135,-45,0,45,135]    
        # shiftable_angle = [-180, -90, -45, 0, 45, 90, 180]
        self.shiftable_angle = [-180, -135, -90, -45, 0, 45, 90, 135, 180] 
        # self.shiftable_angle = [-135, -120, -105, -90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75 ,90, 105, 120, 135]    

        self.shiftable_radian = np.radians(self.shiftable_angle)
        self.shiftable_complex = np.exp(1j * self.shiftable_radian)

        # Initialization for phase shifts
        self.RefVector_angle = np.zeros(self.NumRISEle)
        self.RefVector_radian = np.radians(self.RefVector_angle)
        self.RefVector = np.exp(1j * self.RefVector_radian) 

        # Others
        self.ArrayShape_BS  = [self.NumBSAnt, 1, 1]                                    # BS is equipped with a ULA that is placed along the direction [1, 0, 0] (i.e., x-axis)
        self.ArrayShape_RIS = [1, self.NumRISEle, 1]                                  # RIS is a ULA, which is placed along the direction [0, 1, 0] (i.e., y-axis)
        self.ArrayShape_UE  = [1, 1, 1]                                              # UE is with 1 antenna 

        self.UE_power_selection = 4                          # [0, 3, 10, 200]  (mW)   
        self.RIS_phase_selectoin = 9                         # [-135, -45, 0, 45, 135] (5 phase shifts)
        self.MuMIMO_env = envMuMIMO(self.NumBSAnt, self.NumRISEle, self.NumUE)          

        self.now_step = 0

    def reset(self):
        # Path loss for large-scale fading (position dependent)
        self.pathloss_U2B, self.pathloss_R2B, self.pathloss_U2R = self.MuMIMO_env.H_GenPL(self.Pos_BS, self.Pos_RIS, self.Pos_UE)
        # LOS channel for Rician fading (position dependent)
        self.H_U2B_LoS, self.H_R2B_LoS, self.H_U2R_LoS = self.MuMIMO_env.H_GenLoS(self.Pos_BS, self.Pos_RIS, self.Pos_UE, self.ArrayShape_BS, self.ArrayShape_RIS, self.ArrayShape_UE)  

        # Rician fading NLOS channel 
        self.H_U2B_NLoS, self.H_R2B_NLoS, self.H_U2R_NLoS = self.MuMIMO_env.H_GenNLoS()    

        # Overall channel with large-scale fading and small-scale fading (Rician fading)              
        self.H_U2B_Ric, self.H_R2B_Ric, self.H_U2R_Ric = self.MuMIMO_env.H_RicianOverall(self.K_U2B, self.K_R2B, self.K_U2R, 
            self.H_U2B_LoS, self.H_R2B_LoS, self.H_U2R_LoS, self.H_U2B_NLoS, self.H_R2B_NLoS, self.H_U2R_NLoS, self.pathloss_U2B, self.pathloss_R2B, self.pathloss_U2R) 

        self.H_Ric_overall = self.MuMIMO_env.H_Comb(self.H_U2B_Ric, self.H_R2B_Ric, self.H_U2R_Ric, self.RefVector)          # Use the channel with adjusted phase

        self.load_remaining = np.ones((1,self.NumUE)) * self.LocalDataSize

        self.time_remaining = np.ones((1,self.NumUE)) * self.num_steps

        self.now_step = 0

        H_U2B_Ric = self.H_U2B_Ric.reshape(1, -1) # (single step, NumBSAnt*NumUE)
        H_R2B_Ric = self.H_R2B_Ric.reshape(1, -1) # (single step, NumBSAnt*NumRISEle)
        H_U2R_Ric = self.H_U2R_Ric.reshape(1, -1) # (single step, NumRISEle*NumUE)


        channel_state = np.concatenate((H_U2B_Ric.real, H_U2B_Ric.imag), axis=1)
        channel_state = np.concatenate((channel_state, H_R2B_Ric.real), axis=1)
        channel_state = np.concatenate((channel_state, H_R2B_Ric.imag), axis=1)
        channel_state = np.concatenate((channel_state, H_U2R_Ric.real), axis=1)
        channel_state = np.concatenate((channel_state, H_U2R_Ric.imag), axis=1)
        channel_state = np.concatenate((channel_state, self.RefVector.reshape(1,-1).real), axis=1)
        channel_state = np.concatenate((channel_state, self.RefVector.reshape(1,-1).imag), axis=1)

        obss = np.concatenate((channel_state, self.load_remaining, self.time_remaining), axis=1)
        obss = torch.tensor(obss, dtype=torch.float32)

        return obss

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames

        reward, done, fail = 0, False, 0
        power_action = action[:self.NumUE]
        phase_action = action[self.NumUE:]

        self.Power_UE[power_action == 0] = 0
        self.Power_UE[power_action == 1] = 3
        self.Power_UE[power_action == 2] = 10
        self.Power_UE[power_action == 3] = 200

        
        self.RefVector = self.shiftable_complex[phase_action]

        
        if np.sum(self.Power_UE) == 0:
            EC = 0
            EEE = 0
        else:

            H_Ric_overall = self.MuMIMO_env.H_Comb(self.H_U2B_Ric, self.H_R2B_Ric, self.H_U2R_Ric, self.RefVector)          # Use the channel with adjusted phase

            snr, achievable_rate, capacity, _ = self.MuMIMO_env.SNR_Throughput(H_Ric_overall, self.BW, self.noise, self.Power_UE)  

            EC = -1 / self.QoS_exponent * np.log(np.mean(np.exp(-self.QoS_exponent * capacity)))

            EEE = EC * self.NumUE / np.sum(self.Power_UE)

        # Rician fading NLOS channel 
        self.H_U2B_NLoS, self.H_R2B_NLoS, self.H_U2R_NLoS = self.MuMIMO_env.H_GenNLoS()    

        # Overall channel with large-scale fading and small-scale fading (Rician fading)              
        self.H_U2B_Ric, self.H_R2B_Ric, self.H_U2R_Ric = self.MuMIMO_env.H_RicianOverall(self.K_U2B, self.K_R2B, self.K_U2R, 
            self.H_U2B_LoS, self.H_R2B_LoS, self.H_U2R_LoS, self.H_U2B_NLoS, self.H_R2B_NLoS, self.H_U2R_NLoS, self.pathloss_U2B, self.pathloss_R2B, self.pathloss_U2R) 

        self.H_Ric_overall = self.MuMIMO_env.H_Comb(self.H_U2B_Ric, self.H_R2B_Ric, self.H_U2R_Ric, self.RefVector)          # Use the channel with adjusted phase

        self.load_remaining -= EC*0.05

        self.time_remaining -= 1
        self.now_step += 1

        if np.sum(self.time_remaining) == 0:
            # print("load_remaining: ",  self.load_remaining)
            # print("\n")
            done = True
            if np.sum(self.load_remaining) > 0:
                fail = 1


        H_U2B_Ric = self.H_U2B_Ric.reshape(1, -1) # (single step, NumBSAnt*NumUE)
        H_R2B_Ric = self.H_R2B_Ric.reshape(1, -1) # (single step, NumBSAnt*NumRISEle)
        H_U2R_Ric = self.H_U2R_Ric.reshape(1, -1) # (single step, NumRISEle*NumUE)


        channel_state = np.concatenate((H_U2B_Ric.real, H_U2B_Ric.imag), axis=1)
        channel_state = np.concatenate((channel_state, H_R2B_Ric.real), axis=1)
        channel_state = np.concatenate((channel_state, H_R2B_Ric.imag), axis=1)
        channel_state = np.concatenate((channel_state, H_U2R_Ric.real), axis=1)
        channel_state = np.concatenate((channel_state, H_U2R_Ric.imag), axis=1)
        channel_state = np.concatenate((channel_state, self.RefVector.reshape(1,-1).real), axis=1)
        channel_state = np.concatenate((channel_state, self.RefVector.reshape(1,-1).imag), axis=1)

        obss = np.concatenate((channel_state, self.load_remaining, self.time_remaining), axis=1)
        obss = torch.tensor(obss, dtype=torch.float32)

        # Return state, reward, done
        return obss, EEE, done, fail

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False
    
    def close(self):
        pass

class Args:
    def __init__(self):
        self.device = torch.device('cuda')
        self.max_episode_length = 108e3
        self.history_length = 4
