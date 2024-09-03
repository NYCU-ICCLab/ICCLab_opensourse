"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, sample=False, actions=None, rtgs=None, timesteps=None, params=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """


    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):

        x_cond = x if x.size(1) <= block_size//3 else x[:, -block_size//3:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:] # crop context if needed
        # print("Sampled actions: ", actions)
        # print("timesteps: ", timesteps)
        power_logits, phase_logits, _= model(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
        # print("power_logits.shape: ", power_logits.shape)
        # print("phase_logits.shape: ", phase_logits.shape)
        power_logits = power_logits[-params['NumUE']:, :] 
        power_logits = power_logits.reshape(params['NumUE'],-1)
        phase_logits = phase_logits[-params['NumRISEle']:, :]
        phase_logits = phase_logits.reshape(params['NumRISEle'],-1)

        # print("Sampled logits.shape: ", logits.shape)
        # optionally crop probabilities to only the top k options

        # apply softmax to convert to probabilities
        power_probs = F.softmax(power_logits, dim=-1)
        phase_probs = F.softmax(phase_logits, dim=-1)
        # print("Sampled probs: ", probs)
        # sample from the distribution or take the most likely
        if sample:
            power_ix = torch.multinomial(power_probs, num_samples=1)
            phase_ix = torch.multinomial(phase_probs, num_samples=1)
        else:
            _, power_ix = torch.topk(power_probs, k=1, dim=-1)
            _, phase_ix = torch.topk(phase_probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        power_x = power_ix
        phase_x = phase_ix
        # print("sampled x: ", x)
        power_x = power_x.reshape(1,-1)
        phase_x = phase_x.reshape(1,-1)
        # print("power_x.shape: ", power_x.shape)
        # print("phase_x.shape: ", phase_x.shape)
        x = torch.cat((power_x, phase_x), dim=1)
    return x
