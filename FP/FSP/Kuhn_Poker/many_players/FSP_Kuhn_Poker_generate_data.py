#Library
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

import FSP_Kuhn_Poker_trainer


class GenerateData:
  def __init__(self, num_players, num_actions, random_seed):
    self.num_players = num_players
    self.num_actions = num_actions
    self.kuhn_trainer = FSP_Kuhn_Poker_trainer.KuhnTrainer(num_players=self.num_players)
    self.random_seed = random_seed
    self.kuhn_trainer.random_seed_fix(self.random_seed)


  def generate_data0(self, pi_strategy, beta_strategy, n, m, eta):
    sigma_strategy = {}
    for infoset in pi_strategy.keys():
      sigma_strategy[infoset] = (1-eta)*pi_strategy[infoset] + eta*beta_strategy[infoset]


    sigma_strategy_player_list = self.strategy_split_player(sigma_strategy)
    beta_strategy_player_list = self.strategy_split_player(beta_strategy)
    D_history = []

    for ni in range(n):
      ni_episode = self.one_episode("", self.strategy_uion(sigma_strategy_player_list, sigma_strategy_player_list, 0))
      D_history.append(ni_episode)


    D_history_list = [[] for _ in range(self.num_players)]
    for player_i in range(self.num_players):
      for mi in range(m):
        mi_episode = self.one_episode("", self.strategy_uion(beta_strategy_player_list, sigma_strategy_player_list, player_i))
        D_history_list[player_i].append(mi_episode)
      D_history_list[player_i] += D_history
    return D_history_list


  def generate_data1(self, pi_strategy, n, M_rl):
    pi_strategy_player_list = self.strategy_split_player(pi_strategy)
    D_history = []

    for ni in range(n):
      ni_episode = self.one_episode("", self.strategy_uion(pi_strategy_player_list, pi_strategy_player_list, 0))
      D_history.append(ni_episode)

    for player_i in range(self.num_players):
      M_rl[player_i].extend(D_history)



  def generate_data2(self, pi_strategy, beta_strategy, m, M_rl, M_sl):
    pi_strategy_player_list = self.strategy_split_player(pi_strategy)
    beta_strategy_player_list = self.strategy_split_player(beta_strategy)

    for player_i in range(self.num_players):
      for mi in range(m):
        mi_episode = self.one_episode("", self.strategy_uion(beta_strategy_player_list, pi_strategy_player_list, player_i))
        M_rl[player_i].extend([mi_episode])
        M_sl[player_i].extend([mi_episode])



  def strategy_split_player(self, strategy):
    """return string
    >>> GenerateData(2, 2).strategy_split_player({'J':[1,2], 'Jp':[2,3]}) == [{'J':[1,2]}, {'Jp':[2,3]}]
    True
    """
    strategy_player_list = [{} for _ in range(self.num_players)]

    for infoset, avg_strategy in strategy.items():
        player = (len(infoset)-1) % self.num_players
        strategy_player_list[player][infoset] = avg_strategy
    return strategy_player_list


  def strategy_uion(self, strategy_target_player_list, strategy_not_target_player_list, target_player):
    """return string
    >>> GenerateData(2, 2).strategy_uion([{'J':[1,2]}, {'Jp':[2,3]}], [{'J':[11,12]}, {'Jp':[13,14]}], 0) == {'J':[1,2], 'Jp':[13,14]}
    True
    """
    union_strategy = {}
    for i, strategy in enumerate(strategy_target_player_list):
      if i ==target_player:
        for node, strategy_node in strategy.items():
          union_strategy[node] = strategy_node

    for i, strategy in enumerate(strategy_not_target_player_list):
      if i !=target_player:
        for node, strategy_node in strategy.items():
          union_strategy[node] = strategy_node
    return union_strategy


  def one_episode(self, history, strategy):
    plays = len(history)
    player = plays % self.num_players

    if self.kuhn_trainer.whether_terminal_states(history):
      return history

    elif self.kuhn_trainer.whether_chance_node(history):
      cards = self.kuhn_trainer.card_distribution(self.num_players)
      random.shuffle(cards)
      nextHistory = "".join(cards[:self.num_players])
      return self.one_episode(nextHistory, strategy)

    infoSet = history[player] + history[self.num_players:]

    sampling_action = np.random.choice(list(range(self.num_actions)), p=strategy[infoSet])
    nextHistory = history + ("p" if sampling_action == 0 else "b")
    return self.one_episode(nextHistory, strategy)


  def calculate_optimal_gap_best_response_strategy(self, strategy1, strategy2, target_player):
    strategy1_player_list = self.strategy_split_player(strategy1)
    strategy2_player_list = self.strategy_split_player(strategy2)
    return self.calculate_avg_utility_for_strategy("", 0, 0, [1.0 for _ in range(self.num_players)], strategy1_player_list, strategy2_player_list)


  def calculate_avg_utility_for_strategy(self, history, target_player_i, iteration_t, p_list, strategy1_player_list, strategy2_player_list):
    plays = len(history)
    player = plays % self.num_players

    if self.kuhn_trainer.whether_terminal_states(history):
      return self.kuhn_trainer.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.kuhn_trainer.whether_chance_node(history):
      cards = self.kuhn_trainer.card_distribution(self.num_players)
      cards_candicates = [list(cards_candicate) for cards_candicate in itertools.permutations(cards)]
      utility_sum = 0
      for cards_i in cards_candicates:
        nextHistory = "".join(cards_i[:self.num_players])
        utility_sum +=  (1/len(cards_candicates))* self.calculate_avg_utility_for_strategy(nextHistory, target_player_i, iteration_t, p_list, strategy1_player_list, strategy2_player_list)
      return utility_sum

    infoSet = history[player] + history[self.num_players:]

    if player == target_player_i:
      strategy = strategy1_player_list[player][infoSet]
    else:
      strategy = strategy2_player_list[player][infoSet]

    util_list = np.array([0 for _ in range(self.num_actions)], dtype=float)
    nodeUtil = 0

    for ai in range(self.num_actions):
      nextHistory = history + ("p" if ai == 0 else "b")

      p_change = np.array([1 for _ in range(self.num_players)], dtype=float)
      p_change[player] = strategy[ai]

      util_list[ai] = self.calculate_avg_utility_for_strategy(nextHistory, target_player_i, iteration_t, p_list * p_change, strategy1_player_list, strategy2_player_list)

      nodeUtil += strategy[ai] * util_list[ai]

    return nodeUtil


doctest.testmod()
