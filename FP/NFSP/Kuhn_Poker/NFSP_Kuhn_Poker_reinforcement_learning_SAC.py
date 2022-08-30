
# _________________________________ Library _________________________________
from platform import node
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

import warnings
warnings.filterwarnings('ignore')




# _________________________________ RL class _________________________________
class ReinforcementLearning:
  def __init__(self, train_iterations, num_players, hidden_units_num, lr, epochs, sampling_num, gamma, tau, update_frequency, loss_function, kuhn_trainer_for_rl, random_seed, device):
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.num_actions = 2
    self.action_id = {"p":0, "b":1}
    self.STATE_BIT_LEN = (self.NUM_PLAYERS + 1) + 2*(self.NUM_PLAYERS *2 - 2)
    self.hidden_units_num = hidden_units_num
    self.lr = lr
    self.epochs = epochs
    self.sampling_num = sampling_num
    self.gamma = gamma
    self.tau = tau
    self.update_frequency = update_frequency
    self.kuhn_trainer = kuhn_trainer_for_rl
    self.card_rank  = self.kuhn_trainer.card_rank
    self.random_seed = random_seed
    self.device = device
    self.save_count = 0

    self.rl_algo = None

    self.kuhn_trainer.random_seed_fix(self.random_seed)

    self.loss_fn = loss_function



  def RL_learn(self, memory, target_player, update_strategy, k):
    pass




doctest.testmod()
