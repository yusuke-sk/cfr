
# _________________________________ Library _________________________________
from platform import node
from unittest.mock import seal
from importlib_metadata import distribution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import doctest
from collections import deque
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical



import warnings
warnings.filterwarnings('ignore')
doctest.testmod()


# PPO


class CriticNetwork(nn.Module):
    def __init__(self, state_num, hidden_units_num, lr):
        super(CriticNetwork, self).__init__()
        self.state_num = state_num
        self.hidden_units_num = hidden_units_num
        self.lr = lr

        self.critic = nn.Sequential(
            nn.Linear(self.state_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, 1)
            )

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)


    def forward(self, x):
        output = self.critic(x)
        return output


class ActorNetwork(nn.Module):
    def __init__(self, state_num, action_num, hidden_units_num, lr):
        super(ActorNetwork, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.hidden_units_num = hidden_units_num
        self.lr = lr

        self.actor = nn.Sequential(
            nn.Linear(self.state_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, self.action_num),
            nn.Softmax(dim = -1)
            )

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)


    def forward(self, x):
        h1 = self.actor(x)
        dist = Categorical(h1)

        #action distribution
        return dist



class PPO:
  def __init__(self, num_players, hidden_units_num, epochs, eps_clip, policy_lr, value_lr, gamma, lam, policy_clip, wandb_save, entropy_coef):
    self.NUM_PLAYERS = num_players
    self.num_actions = 2
    self.num_states = (self.NUM_PLAYERS + 1) + 2*(self.NUM_PLAYERS *2 - 2)
    self.hidden_units_num = hidden_units_num
    self.epochs = epochs
    self.eps_clip = eps_clip
    self.policy_lr = policy_lr
    self.value_lr = value_lr
    self.gamma = gamma
    self.lam = lam
    self.policy_clip = policy_clip
    self.wandb_save = wandb_save
    self.entropy_coef = entropy_coef


    self.actor = ActorNetwork(self.num_states, self.num_actions, self.hidden_units_num, self.policy_lr)
    self.critic = CriticNetwork(self.num_states, self.num_actions, self.value_lr)


    self.loss_function = nn.MSELoss()

    self.strategy = None


  def select_action(self, state, state_bit):

    dist = self.actor(state_bit)

    value = self.critic(state_bit)

    action = dist.sample()

    #ただの数字を取り出す
    log_prob = torch.squeeze(dist.log_prob(action)).item()

    action = torch.squeeze(action).item()
    value = torch.squeeze(value).item()

    self.strategy[state] = np.array([np.exp(torch.squeeze(dist.log_prob(torch.Tensor([0]))).item()),\
        np.exp(torch.squeeze(dist.log_prob(torch.Tensor([1]))).item())], dtype=float)

    #print(state, action, np.exp(log_prob), value)

    return action, log_prob, value


  def calculate_advantage(self, state_arr, reward_arr, value_arr, done_arr):

      #calulate adavantage (一旦、愚直に前から計算する)
      advantages = np.zeros(len(state_arr), dtype=np.float32)

      for t in range(len(reward_arr)-1):
          discount = 1 # lamda * gamma が 足されていってるもの
          adavantage_t = 0
          for k in range(t, len(reward_arr)-1):
              sigma_k = reward_arr[k] + self.gamma * value_arr[k+1] * (1 - done_arr[k]) - value_arr[k]
              adavantage_t += discount * sigma_k
              discount *= self.gamma * self.lam

          advantages[t] = adavantage_t

      return advantages


  def RL_learn(self, memory):

    avg_total_loss = []
    avg_actor_loss = []
    avg_critic_loss = []
    avg_entropy_loss = []

    for _ in range(self.epochs):

      state_arr, action_arr, old_action_prob_arr, value_arr, reward_arr,  done_arr, batch_index = memory.get_batch()

      advantages = self.calculate_advantage(state_arr, reward_arr, value_arr, done_arr)

      advantages = torch.tensor(advantages)
      values = torch.tensor(value_arr)


      #batch 計算
      for batch in batch_index:
        states = torch.tensor(state_arr[batch], dtype=torch.float)
        actions = torch.tensor(action_arr[batch])
        old_action_probs = torch.tensor(old_action_prob_arr[batch])


        # calculate actor loss
        dist = self.actor(states)
        new_action_probs = dist.log_prob(actions)

        #元々log probだったから * expで 方策に戻る  policy_ratio: 現在とひとつ前の方策を比較
        policy_ratio =  new_action_probs.exp() / old_action_probs.exp()

        #candicate 1
        weighted_probs = policy_ratio * advantages[batch]

        #candicate 2  第一引数inputに処理する配列Tesonr、第二引数minに最小値、第三引数maxに最大値を指定
        weighted_clipped_probs = torch.clamp(policy_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantages[batch]

        actor_loss = - torch.min(weighted_probs, weighted_clipped_probs).mean()


        # calculate critic loss
        critic_value = torch.squeeze(self.critic(states))

        returns = advantages[batch] + values[batch]


        critic_loss = ((returns-critic_value)**2).mean()

        log_prob1 = dist.log_prob(actions)
        log_prob2 = dist.log_prob(1-actions)


        entropy_loss = - ( torch.exp(log_prob1)* log_prob1  + torch.exp(log_prob2)* log_prob2).mean()

        #total loss
        total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss

        avg_total_loss.append(total_loss.detach().item())
        avg_actor_loss.append(actor_loss.detach().item())
        avg_critic_loss.append(critic_loss.detach().item())
        avg_entropy_loss.append(entropy_loss.detach().item())



        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()


    if self.wandb_save:
        wandb.log({'avg_total_loss': np.mean(avg_total_loss), 'avg_actor_loss': np.mean(avg_actor_loss), 'avg_critic_loss': np.mean(avg_critic_loss), 'avg_entropy_loss': np.mean(avg_entropy_loss)})






class PPO_RL_memory:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []


    def store(self, memory):

      self.states.append(memory[0])
      self.actions.append(memory[1])
      self.log_probs.append(memory[2])
      self.values.append(memory[3])
      self.rewards.append(memory[4])
      self.dones.append(memory[5])


    def delete(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []


    def get_size(self):
        return len(self.states)

    def display(self):
      return np.array(self.states), np.array(self.actions), np.array(self.log_probs), \
            np.array(self.values), np.array(self.rewards)


    def get_batch(self):
        #dataをbatch_sizeに切り分ける そのindexを作る
        size = self.get_size()
        batch_start_index = np.arange(0, size, self.batch_size)
        state_index = np.arange(size, dtype=np.int64)
        np.random.shuffle(state_index)
        batch_index = [state_index[i:i+self.batch_size] for i in batch_start_index]



        #npだと array[batch] で 取得可能
        return np.array(self.states), np.array(self.actions), np.array(self.log_probs), \
            np.array(self.values), np.array(self.rewards), np.array(self.dones), batch_index

"""
actor = ActorNetwork(7, 2, 64, 0.01)

ar = torch.Tensor([[0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0]])
#print(actor(ar))
"""
