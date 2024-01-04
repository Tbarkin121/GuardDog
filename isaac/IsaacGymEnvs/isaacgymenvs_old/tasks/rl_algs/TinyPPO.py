# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:57:32 2023

@author: Plutonium
"""

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch import optim
import time
import os

from .buffer import Buffer
from .network import Policy


# torch.set_default_device('cuda')

class PPO_Agent:
    def __init__(self, cfg):
        self.num_envs = cfg['env']['num_envs']
        self.horizon = cfg['env']['horizon']
        self.num_acts = cfg['env']['numActions']
        self.num_obs = cfg['env']['numObservations']
        self.gamma = cfg['env']['gamma']
        self.minibatch_steps = cfg['env']['minibatch_steps']
        self.minibatch_size =  cfg['env']['minibatch_steps']
        self.take_n_actions = 1
        self.entropy_coff_initial = 0.0
        self.clip_range = 0.2
        self.sigma = 0.01
        self.network = Policy(self.num_acts, self.num_obs)
        self.optim = optim.Adam(self.network.parameters(), lr=3e-4)
        self.buffer = Buffer(self.num_envs, self.horizon, self.num_acts, self.num_obs, self.gamma)
                  
        self.mse_loss = torch.nn.MSELoss()
        log_probs = torch.zeros((self.num_envs, self.horizon))
        log_probs_old = torch.zeros((self.num_envs, self.horizon)).detach()
        self.actor_loss_list = []
        self.critic_loss_list = []

    def training_step(self):
        for mb in range(self.minibatch_steps):
            s1, a1, r1, s2, d2, log_probs_old, returns = self.buffer.get_SARS_minibatch(self.minibatch_size) 
            # s1, a1, r1, s2, d2, log_probs_old, returns = self.buffer.get_SARS()

    
            [vals_s1, probs_s1] = self.network(s1)
            # action_pd_s1 = torch.distributions.Normal(probs_s1[:,:,0], probs_s1[:,:,1])
            action_pd_s1 = torch.distributions.Normal(probs_s1, self.sigma*torch.ones_like(probs_s1.detach()))
        

            # td_error = returns - vals_s1.squeeze(-1)
            advantage = returns - vals_s1.squeeze(-1)
        

            # normalize advantage... (Doesn't Seem to work?)
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            log_probs = action_pd_s1.log_prob(a1)
            ratio = torch.exp(log_probs - log_probs_old)

            # entropy_coff = self.entropy_coff_initial * (1-(epoch/self.num_epochs))
            # entropy_loss = -action_pd_s1.entropy().mean() * entropy_coff
            entropy_coff = self.entropy_coff_initial
            entropy_loss = -entropy_coff*torch.mean(-log_probs)

            # policy_loss_1 = advantage.view(self.num_envs, self.horizon, 1).detach() * ratio
            # policy_loss_2 = advantage.view(self.num_envs, self.horizon, 1).detach() * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss_1 = advantage.view(self.minibatch_size, self.horizon, 1).detach() * ratio
            policy_loss_2 = advantage.view(self.minibatch_size, self.horizon, 1).detach() * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean() 
            
            value_loss = self.mse_loss(vals_s1.squeeze(-1), returns)
            
            total_loss = policy_loss + value_loss*0.2 + entropy_loss
            
            self.optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) #Max Grad Norm
            self.optim.step()

            
            self.actor_loss_list.append(policy_loss.detach().cpu().numpy())
            self.critic_loss_list.append(value_loss.detach().cpu().numpy())
            
            

            
            for name, param in self.network.named_parameters():
                if( torch.any(torch.isnan(param)) ):
                    print(name)
                    print(param)
                    BREAKPOINTBULLSHIT

            

        print('Policy Loss Avg: {}. Value Loss Avg: {}. Avg Returns: {}'.format(np.array(self.actor_loss_list).mean(), np.array(self.critic_loss_list).mean(), returns.mean()))

        with torch.no_grad():
            # Useful extra info
            approx_kl1 = ((torch.exp(ratio) - 1) - ratio).mean() #Stable Baselines 3
            approx_kl2 = (log_probs_old - log_probs).mean()    #Open AI Spinup
            # print('kl approx : {} : {} : {}'.formaWDt(approx_kl1, approx_kl2, ratio.mean()))
    
    
        torch.save(self.network.state_dict(), "D2RL_Save.pth")            

            # with torch.no_grad():
            #     for _ in range(self.take_n_actions):    
            #         s1, a1, r1, s2, d2, log_probs_old, returns = self.buffer.get_SARS()
            #         [vals, probs] = self.network(s2)
            #         newest_probs = probs[:,0,0].view(-1,1)
            #         action_pd = torch.distributions.Normal(newest_probs, self.sigma*torch.ones_like(newest_probs))
            #         next_actions = action_pd.sample()
            #         log_probs_sample = action_pd.log_prob(next_actions)

            #         env.step(next_actions, log_probs_sample, Agent)   

    def get_actions(self):
        s1, a1, r1, s2, d2, log_probs_old, returns = self.buffer.get_SARS()
        [vals, probs] = self.network(s2)
        newest_probs = probs[:,0,:].view(self.num_envs, self.num_acts)
        action_pd = torch.distributions.Normal(newest_probs, self.sigma*torch.ones_like(newest_probs))
        next_actions = action_pd.sample()
        next_log_probs = action_pd.log_prob(next_actions)
        return next_actions, next_log_probs
    
    def get_vals(self, state):
        [vals, probs] = self.network(state)
        return vals

    def buffer_store(self, s1, a1, lp, r2, s2, d2, val):
        self.buffer.update1(s1, a1, lp)
        self.buffer.update2(r2, s2, d2, val)

