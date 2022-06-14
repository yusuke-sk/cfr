
# _________________________________ Library _________________________________
from turtle import update
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import itertools
from collections import defaultdict
from tqdm import tqdm
import time
import doctest
import copy
import wandb
from collections import deque


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from NFSP_Leduc_Poker_trainer import LeducTrainer

# _________________________________ SL NN class _________________________________
class SL_Network(nn.Module):
    def __init__(self, state_num, hidden_units_num):
        super(SL_Network, self).__init__()
        self.state_num = state_num
        self.hidden_units_num = hidden_units_num

        self.fc1 = nn.Linear(self.state_num, self.hidden_units_num)
        self.fc2 = nn.Linear(self.hidden_units_num, 3)
        #self.fc3 = nn.Linear(self.state_num, 1)

        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = F.leaky_relu(self.fc1(x))

        #output = self.fc2(h1)
        h2 = self.dropout(h1)

        output = self.fc2(h2)


        return output


# _________________________________ SL class _________________________________
class SupervisedLearning:
  def __init__(self,train_iterations, num_players, hidden_units_num, lr, epochs, sampling_num, leduc_trainer_for_sl, random_seed):

    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.NUM_ACTIONS = 3
    self.STATE_BIT_LEN = 2* ( (self.NUM_PLAYERS + 1) + 3*(self.NUM_PLAYERS *3 - 2) )
    self.hidden_units_num = hidden_units_num
    self.lr = lr
    self.epochs = epochs
    self.sampling_num = sampling_num
    self.random_seed = random_seed

    self.leduc_trainer = leduc_trainer_for_sl

    self.leduc_trainer.random_seed_fix(random_seed = self.random_seed)

    self.card_rank  = self.leduc_trainer.card_rank


    self.sl_network = SL_Network(state_num=self.STATE_BIT_LEN, hidden_units_num=self.hidden_units_num)

    self.optimizer = optim.Adam(self.sl_network.parameters(), lr=self.lr, weight_decay=5*(10**(-4)))
    #self.optimizer = optim.Adam(self.sl_network.parameters(), lr=self.lr)

    self.softmax = nn.Softmax(dim=1)


    self.loss_fn = nn.CrossEntropyLoss()


    self.infoset_action_player_dict = {}
    self.ACTION_DICT = {0:"f", 1:"c", 2:"r"}
    self.ACTION_DICT_verse = {"f":0, "c":1, "r":2}




  def SL_learn(self, memory, target_player, update_strategy, iteration_t):

    #train
    self.sl_network.train()

    if len(memory) < self.sampling_num:
      return

    total_loss = 0


    for _ in range(self.epochs):

      samples =  random.sample(memory, self.sampling_num)


      train_X = np.array([])
      train_y = np.array([])

      for one_s_a_set in samples:
        if one_s_a_set is not None:
          train_i = self.leduc_trainer.from_episode_to_bit([one_s_a_set])
          train_X = np.append(train_X, train_i[0])
          train_y = np.append(train_y, train_i[1])

          #print(one_s_a_set, train_i)

      #print("")
      #print(samples)
      inputs = torch.from_numpy(train_X).float().reshape(-1,self.STATE_BIT_LEN)
      targets = torch.from_numpy(train_y).long().reshape(-1, 1).squeeze_()

      outputs = self.sl_network.forward(inputs)


      #loss = - (targets * outputs).sum(dim=-1).mean()




      loss = self.loss_fn(outputs, targets)

      #print(outputs, targets, loss)


      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      total_loss += loss.item()


    if iteration_t in [int(j) for j in np.logspace(0, len(str(self.train_iterations)), (len(str(self.train_iterations)))*4 , endpoint=False)] :
      if self.leduc_trainer.wandb_save:
        wandb.log({'iteration': iteration_t, 'loss_sl': total_loss/self.epochs})


        #for node_X , _ in update_strategy.items():
          #if (len(node_X)-1) % self.NUM_PLAYERS == target_player :
            #print(node_X, update_strategy[node_X])


    # eval
    self.sl_network.eval()
    with torch.no_grad():
      for node_X , _ in update_strategy.items():
        if self.infoset_action_player_dict[node_X] == target_player :

          inputs_eval = torch.from_numpy(self.leduc_trainer.make_state_bit(node_X)).float().reshape(-1,self.STATE_BIT_LEN)

          y = self.softmax(self.sl_network.forward(inputs_eval)).detach().numpy()[0]




          possible_action_list = self.leduc_trainer.node_possible_action[node_X]
          normalizationSum = 0
          for action_i, yi in enumerate(y):
            if action_i not in possible_action_list:
              y[action_i] = 0
            else:
              normalizationSum += yi

          y /= normalizationSum
          update_strategy[node_X] = y





  def SL_train_AVG(self, memory, target_player, strategy, n_count):
    for one_s_a_set in memory:
      for X, y in [one_s_a_set]:
        if self.infoset_action_player_dict[X] == target_player :
          action_prob_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          action_prob_list[self.ACTION_DICT_verse[y]] = 1.0
          n_count[X] += action_prob_list


    for node_X , action_prob in n_count.items():
        strategy[node_X] = n_count[node_X] / np.sum(action_prob)

    return strategy

doctest.testmod()