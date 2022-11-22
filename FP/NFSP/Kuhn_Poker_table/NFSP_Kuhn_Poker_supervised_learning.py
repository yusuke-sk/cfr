
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

from NFSP_Kuhn_Poker_trainer import KuhnTrainer



# _________________________________ SL NN class _________________________________
class SL_Network(nn.Module):
    def __init__(self, state_num, hidden_units_num):
        super(SL_Network, self).__init__()
        self.state_num = state_num
        self.hidden_units_num = hidden_units_num

        self.fc1 = nn.Linear(self.state_num, self.hidden_units_num)
        self.fc2 = nn.Linear(self.hidden_units_num, 1)
        #self.fc3 = nn.Linear(self.state_num, 1)

        self.dropout = nn.Dropout(0.2)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = F.leaky_relu(self.fc1(x))

        #output = self.fc2(h1)
        h2 = self.dropout(h1)

        output = self.fc2(h2)


        return output


# _________________________________ SL class _________________________________
class SupervisedLearning:
  def __init__(self,train_iterations, num_players, hidden_units_num, lr, epochs, sampling_num, loss_function, kuhn_trainer_for_sl, random_seed, device):

    #self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')



    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.NUM_ACTIONS = 2
    self.STATE_BIT_LEN = (self.NUM_PLAYERS + 1) + 2*(self.NUM_PLAYERS *2 - 2)
    self.hidden_units_num = hidden_units_num
    self.lr = lr
    self.epochs = epochs
    self.sampling_num = sampling_num

    self.kuhn_trainer = kuhn_trainer_for_sl
    self.device = device
    self.save_count = 0


    self.card_rank  = self.kuhn_trainer.card_rank
    self.random_seed = random_seed

    self.kuhn_trainer.random_seed_fix(random_seed = self.random_seed)

    self.sl_network = SL_Network(state_num=self.STATE_BIT_LEN, hidden_units_num=self.hidden_units_num).to(self.device)

    #self.optimizer = optim.Adam(self.sl_network.parameters(), lr=self.lr, weight_decay=5*(10**(-4)))
    self.optimizer = optim.Adam(self.sl_network.parameters(), lr=self.lr)


    self.loss_fn = loss_function
    #self.loss_fn = nn.CrossEntropyLoss()



  def SL_learn(self, memory, update_strategy, iteration_t):
    self.sl_network.train()

    total_loss = []


    for _ in range(self.epochs):
      samples =  random.sample(memory, min(self.sampling_num, len(memory)))

      train_X = [sa_bit[0] for sa_bit in samples]
      train_y = [sa_bit[1] for sa_bit in samples]

      inputs = torch.tensor(train_X).float().reshape(-1,self.STATE_BIT_LEN).to(self.device)
      targets = torch.tensor(train_y).float().reshape(-1, 1).to(self.device)


      outputs = self.sl_network.forward(inputs)
      loss = self.loss_fn(outputs, targets)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      total_loss.append(loss.item())



    if self.kuhn_trainer.wandb_save and self.save_count % 100 == 0:
      wandb.log({'iteration': iteration_t, 'loss_sl':  np.mean(total_loss)})
    self.save_count += 1




    # eval
    """#NNをtableに書き写さない
    self.sl_network.eval()
    with torch.no_grad():
      for node_X , _ in update_strategy.items():

        inputs_eval = torch.tensor(self.kuhn_trainer.make_state_bit(node_X)).float().reshape(-1,self.STATE_BIT_LEN).to(self.device)

        y = torch.sigmoid(self.sl_network.forward(inputs_eval)).to('cpu').detach().numpy()[0]


        update_strategy[node_X] = np.array([1.0-y[0], y[0]])
    """



  def whether_put_memory_i(self, i, data, k):
    if i < k:
      self.new_memory[i] = data
    else:
      r = random.randint(1, i)
      if r < k:
        self.new_memory[r] = data



  def reservoir_sampling(self, memory, k):
    self.new_memory = [None for _ in range(k)]
    for i in range(len(memory)):
      self.whether_put_memory_i(i, memory[i], k)

    return self.new_memory


  def SL_train_AVG(self, memory, strategy, n_count):


    for one_s_a_set in memory:
      for X, y in [one_s_a_set]:
        if y == "p":
          n_count[X] += np.array([1.0, 0.0], dtype=float)
        else:
          n_count[X] += np.array([0.0, 1.0], dtype=float)

    for node_X , action_prob in n_count.items():
        strategy[node_X] = n_count[node_X] / np.sum(action_prob)


    return strategy


doctest.testmod()
