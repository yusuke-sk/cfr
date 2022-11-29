#ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
from collections import defaultdict
import sys
from tqdm import tqdm
import time
import doctest
import copy
from sklearn.neural_network import MLPClassifier
from collections import deque
import math

import FSP_Kuhn_Poker_trainer



class ReinforcementLearning:
  def __init__(self, infoSet_dict_player, num_players, num_actions, random_seed):
    self.gamma = 1
    self.num_players = num_players
    self.num_actions = num_actions
    self.action_id = {"p":0, "b":1}
    self.player_q_state = self.make_each_player_state_idx(infoSet_dict_player)
    #[{'J': 0, 'Jpb': 1, 'Q': 2, 'Qpb': 3, 'K': 4, 'Kpb': 5}, {'Qp': 0, 'Qb': 1, 'Kp': 2, 'Kb': 3, 'Jp': 4, 'Jb': 5}]

    self.kuhn_trainer = FSP_Kuhn_Poker_trainer.KuhnTrainer(num_players=self.num_players)
    self.random_seed = random_seed
    self.kuhn_trainer.random_seed_fix(self.random_seed)



  def make_each_player_state_idx(self, infoSet_dict_player):
    #Q-state for each player
    player_q_state = [{} for _ in range(self.num_players)]
    for player_i, player_i_state in enumerate(infoSet_dict_player):
      for idx, j in enumerate(player_i_state):
        player_q_state[player_i][j] = idx
    return player_q_state



  def Episode_split(self, one_episode):
    """return list
    >>> ReinforcementLearning([],2, 2).Episode_split('QKbp')
    [('Q', 'b', 1, None), ('Kb', 'p', -1, None)]
    >>> ReinforcementLearning([], 2, 2).Episode_split('KJpbb')
    [('K', 'p', 0, 'Kpb'), ('Jp', 'b', -2, None), ('Kpb', 'b', 2, None)]
    """
    one_episode_split = []
    action_history = one_episode[self.num_players:]
    for idx, ai in enumerate(action_history):
      s = one_episode[idx%self.num_players] + action_history[:idx]
      a = ai
      if (idx + self.num_players) <= len(action_history) - 1 :
        s_prime = one_episode[idx%self.num_players] + action_history[:idx+self.num_players]
        r = 0
      else:
        s_prime = None
        r = self.kuhn_trainer.Return_payoff_for_terminal_states(one_episode, idx%self.num_players)

      one_episode_split.append((s, a, r, s_prime))
    return one_episode_split


  def RL_train(self, memory, target_player, update_strategy, q_value, k, rl_algo):
    self.alpha = 0.05/ (1+0.003*(k**0.5))
    self.T = 1 / (1+ 0.02*(k**0.5))
    self.epsilon = 0.06/(k**0.5)
    self.epochs = 1
    self.sample_num = 30


    for _ in range(self.epochs):
      if len(memory) <= self.sample_num:
        return

      replay_sample_list = random.sample(memory, self.sample_num)

      for one_episode in replay_sample_list:
        one_episode_split = self.Episode_split(one_episode)
        for trainsition in one_episode_split:
          s, a, r, s_prime = trainsition[0], trainsition[1], trainsition[2], trainsition[3]
          if (len(s) -1) % self.num_players == target_player:
            s_idx = self.player_q_state[target_player][s]
            a_idx = self.action_id[a]

            if s_prime == None:
              q_value[s_idx][a_idx] = q_value[s_idx][a_idx]  + self.alpha*(r - q_value[s_idx][a_idx])
            else:
              s_prime_idx = self.player_q_state[target_player][s_prime]
              q_value[s_idx][a_idx] = q_value[s_idx][a_idx]  + self.alpha*(r + self.gamma*max(q_value[s_prime_idx]) - q_value[s_idx][a_idx])



    state_space = len(self.player_q_state[target_player])
    action_space = len(self.action_id)

    if rl_algo == "boltzmann":
      q_value_boltzmann = np.zeros((state_space,action_space))
      for si in range(state_space):
        for ai in range(action_space):
          q_value_boltzmann[si][ai] = math.exp(q_value[si][ai]/self.T)

      for state, idx in self.player_q_state[target_player].items():
        update_strategy[state] = q_value_boltzmann[self.player_q_state[target_player][state]] / sum(q_value_boltzmann[self.player_q_state[target_player][state]])


    elif rl_algo == "epsilon-greedy":
      for state, idx in self.player_q_state[target_player].items():
        if np.random.uniform() < self.epsilon:   # 探索(epsilonの確率で)
          action = np.random.randint(action_space)
          if action == 0:
            update_strategy[state] = np.array([1, 0], dtype=float)
          else:
            update_strategy[state] = np.array([0, 1], dtype=float)

        else:
          if q_value[self.player_q_state[target_player][state]][0] > q_value[self.player_q_state[target_player][state]][1]:
            update_strategy[state] = np.array([1, 0], dtype=float)
          else:
            update_strategy[state] = np.array([0, 1], dtype=float)



doctest.testmod()
