#Library
from typing import ByteString
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
import wandb

import Online_FSP_Kuhn_Poker_supervised_learning
import Online_FSP_Kuhn_Poker_reinforcement_learning
import Online_FSP_Kuhn_Poker_generate_data


class KuhnTrainer:
  def __init__(self, train_iterations=10**1, num_players =2):
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.NUM_ACTIONS = 2
    self.avg_strategy = {}
    self.card_rank = self.make_rank(self.NUM_PLAYERS)



  def make_rank(self, num_players):
    """return dict
    >>> KuhnTrainer().make_rank(2) == {'J':1, 'Q':2, 'K':3}
    True
    """
    card_rank = {}
    card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    for i in range(num_players+1):
      card_rank[card[11-num_players+i]] =  i+1
    return card_rank


  def card_distribution(self, num_players):
    """return list
    >>> KuhnTrainer().card_distribution(2)
    ['J', 'Q', 'K']
    """
    card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    return card[11-num_players:]


  #return util for terminal state for target_player_i
  def Return_payoff_for_terminal_states(self, history, target_player_i):
      """return list
      >>> KuhnTrainer(num_players=2).Return_payoff_for_terminal_states("JKbb", 0)
      -2
      >>> KuhnTrainer(num_players=2).Return_payoff_for_terminal_states("JKbb", 1)
      2
      >>> KuhnTrainer(num_players=2).Return_payoff_for_terminal_states("JKpp", 0)
      -1
      >>> KuhnTrainer(num_players=2).Return_payoff_for_terminal_states("JKpp", 1)
      1
      >>> KuhnTrainer(num_players=2).Return_payoff_for_terminal_states("JKpbp", 0)
      -1
      >>> KuhnTrainer(num_players=2).Return_payoff_for_terminal_states("JKpbp", 1)
      1
      >>> KuhnTrainer(num_players=3).Return_payoff_for_terminal_states("JKTpbpp", 1)
      2
      """

      pot = self.NUM_PLAYERS * 1 + history.count("b")
      start = -1
      target_player_action = history[self.NUM_PLAYERS+target_player_i::self.NUM_PLAYERS]

      # all players pass
      if ("b" not in history) and (history.count("p") == self.NUM_PLAYERS):
        pass_player_card = {}
        for idx in range(self.NUM_PLAYERS):
          pass_player_card[idx] = [history[idx], self.card_rank[history[idx]]]

        winner_rank = max([idx[1] for idx in pass_player_card.values()])

        target_player_rank = pass_player_card[target_player_i][1]

        if target_player_rank == winner_rank:
          return start + pot
        else:
          return start

      #target plyaer do pass , another player do bet
      elif ("b" not in target_player_action) and ("b" in history):
        return start

      else:
        #bet → +pot or -2
        bet_player_list = [idx%self.NUM_PLAYERS for idx, act in enumerate(history[self.NUM_PLAYERS:]) if act == "b"]
        bet_player_card = {}
        for idx in bet_player_list:
          bet_player_card[idx] = [history[idx], self.card_rank[history[idx]]]


        winner_rank = max([idx[1] for idx in bet_player_card.values()])
        target_player_rank = bet_player_card[target_player_i][1]
        if target_player_rank == winner_rank:
          return start + pot - 1
        else:
          return start - 1


  #whether terminal state
  def whether_terminal_states(self, history):
    #pass only history
    if "b" not in history:
      return history.count("p") == self.NUM_PLAYERS

    plays = len(history)
    first_bet = history.index("b")
    return plays - first_bet -1  == self.NUM_PLAYERS -1


  #whether chance node state
  def whether_chance_node(self, history):
    """return string
    >>> KuhnTrainer().whether_chance_node("")
    True
    >>> KuhnTrainer().whether_chance_node("p")
    False
    """
    if history == "":
      return True
    else:
      return False


  # make node or get node
  def if_nonexistant(self, infoSet):
    if infoSet not in self.avg_strategy:
      self.avg_strategy[infoSet] = np.array([1/self.NUM_ACTIONS for _ in range(self.NUM_ACTIONS)], dtype=float)


  def calc_best_response_value(self, best_response_strategy, best_response_player, history, prob):
      plays = len(history)
      player = plays % self.NUM_PLAYERS

      if self.whether_terminal_states(history):
        return self.Return_payoff_for_terminal_states(history, best_response_player)

      elif self.whether_chance_node(history):
        cards = self.card_distribution(self.NUM_PLAYERS)
        cards_candicates = [list(cards_candicate) for cards_candicate in itertools.permutations(cards)]
        utility_sum = 0
        for cards_i in cards_candicates:
          nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
          utility_sum +=  (1/len(cards_candicates))* self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob)
        return utility_sum

      infoSet = history[player] + history[self.NUM_PLAYERS:]
      self.if_nonexistant(infoSet)

      if player == best_response_player:
        if infoSet not in best_response_strategy:
          action_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          br_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)


          for assume_history, po_ in self.infoSets_dict[infoSet]:
            for ai in range(self.NUM_ACTIONS):
              nextHistory =  assume_history + ("p" if ai == 0 else "b")
              br_value[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, po_)
              action_value[ai] += br_value[ai] * po_

          br_action = 0
          for ai in range(self.NUM_ACTIONS):
            if action_value[ai] > action_value[br_action]:
              br_action = ai
          best_response_strategy[infoSet] = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          best_response_strategy[infoSet][br_action] = 1.0

        node_util = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in range(self.NUM_ACTIONS):
          nextHistory =  history + ("p" if ai == 0 else "b")
          node_util[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob)
        best_response_util = 0
        for ai in range(self.NUM_ACTIONS):
          best_response_util += node_util[ai] * best_response_strategy[infoSet][ai]

        return best_response_util

      else:
        nodeUtil = 0
        action_value_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in range(self.NUM_ACTIONS):
          nextHistory =  history + ("p" if ai == 0 else "b")
          action_value_list[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob* self.avg_strategy[infoSet][ai])
          nodeUtil += self.avg_strategy[infoSet][ai] * action_value_list[ai]
        return nodeUtil


  def create_infoSets(self, history, target_player, po):
    plays = len(history)
    player = plays % self.NUM_PLAYERS

    if self.whether_terminal_states(history):
      return

    elif self.whether_chance_node(history):
      cards = self.card_distribution(self.NUM_PLAYERS)
      cards_candicates = [list(cards_candicate) for cards_candicate in itertools.permutations(cards)]
      for cards_i in cards_candicates:
        nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
        self.create_infoSets(nextHistory, target_player, po)
      return

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    if player == target_player:
      if self.infoSets_dict.get(infoSet) is None:
        self.infoSets_dict[infoSet] = []
        self.infoSets_dict_player[player].append(infoSet)
      self.infoSets_dict[infoSet].append((history, po))


    for ai in range(self.NUM_ACTIONS):
      nextHistory = history + ("p" if ai == 0 else "b")
      if player == target_player:
        self.create_infoSets(nextHistory, target_player, po)
      else:
        self.if_nonexistant(infoSet)
        actionProb =self.avg_strategy[infoSet][ai]
        self.create_infoSets(nextHistory, target_player, po*actionProb)


  def get_exploitability_dfs(self):

    # 各information setを作成 & reach_probabilityを計算
    self.infoSets_dict = {}
    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)

    exploitability = 0
    best_response_strategy = {}
    for best_response_player_i in range(self.NUM_PLAYERS):
        exploitability += self.calc_best_response_value(best_response_strategy, best_response_player_i, "", 1)

    assert exploitability >= 0
    return exploitability


  def eval_vanilla_CFR(self, history, target_player_i, iteration_t, p_list):
    plays = len(history)
    player = plays % self.NUM_PLAYERS

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
      cards = self.card_distribution(self.NUM_PLAYERS)
      cards_candicates = [list(cards_candicate) for cards_candicate in itertools.permutations(cards)]
      utility_sum = 0
      for cards_i in cards_candicates:
        nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
        utility_sum +=  (1/len(cards_candicates))* self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list)
      return utility_sum

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    self.if_nonexistant(infoSet)

    strategy = self.avg_strategy[infoSet]

    util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    nodeUtil = 0

    for ai in range(self.NUM_ACTIONS):
      nextHistory = history + ("p" if ai == 0 else "b")

      p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
      p_change[player] = strategy[ai]

      util_list[ai] = self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list * p_change)

      nodeUtil += strategy[ai] * util_list[ai]

    return nodeUtil



  def train_one_episode(self, history, iteration_t):
    plays = len(history)
    player = plays % self.NUM_PLAYERS

    s = history[player] + history[self.NUM_PLAYERS:]

    if self.sigma_strategy_bit[player] == 0:
      sampling_action = np.random.choice(list(range(self.NUM_ACTIONS)), p=self.best_response_strategy[s])
    elif self.sigma_strategy_bit[player] == 1:
      sampling_action = np.random.choice(list(range(self.NUM_ACTIONS)), p=self.avg_strategy[s])


    a = ("p" if sampling_action == 0 else "b")
    Nexthistory = history + ("p" if sampling_action == 0 else "b")


    next_transition = []
    if self.whether_terminal_states(Nexthistory):
      r = self.Return_payoff_for_terminal_states(Nexthistory, player)
      s_prime = None
      self.M_RL[player].append((s, a, r, s_prime))
      next_transition = [s, a, r, s_prime, Nexthistory]


    else:
      other_s, other_a, other_r, other_s_prime, other_histroy = self.train_one_episode(Nexthistory, iteration_t)

      if self.whether_terminal_states(other_histroy):
        r = self.Return_payoff_for_terminal_states(other_histroy, player)
        s_prime = None
        self.M_RL[player].append((s, a, r, s_prime))
        next_transition = [s, a, r, s_prime, other_histroy]
      else:
        r = 0
        s_prime = other_histroy[player] + other_histroy[self.NUM_PLAYERS:]
        self.M_RL[player].append((s, a, r, s_prime))
        next_transition = [s, a, r, s_prime, other_histroy]


    if self.sigma_strategy_bit[player] == 0:
      self.M_SL[player].append((s, a))
    if len(self.M_SL[player]) != 0:
      if self.sl_algo == "cnt":
        self.SL.SL_train_AVG(self.M_SL[player], player, self.avg_strategy, self.N_count)
        self.M_SL[player] = []
      elif self.sl_algo == "mlp":
        self.SL.SL_train_MLP(self.M_SL[player], player, self.avg_strategy)


    if self.rl_algo == "q_learning":
      self.RL.RL_train(self.M_RL[player], player, self.best_response_strategy, self.Q_value[player], iteration_t)
    elif self.rl_algo == "dfs":
      self.infoSets_dict = {}
      for target_player in range(self.NUM_PLAYERS):
        self.create_infoSets("", target_player, 1.0)
      self.best_response_strategy = {}
      for best_response_player_i in range(self.NUM_PLAYERS):
        self.calc_best_response_value(self.best_response_strategy, best_response_player_i, "", 1)





    return next_transition







  #KuhnTrainer main method
  def train(self, eta, memory_size_rl, memory_size_sl,  wandb_save, rl_algo, sl_algo):
    self.exploitability_list = {}
    self.avg_utility_list = {}
    self.eta = eta
    self.sl_algo = sl_algo
    self.rl_algo = rl_algo

    self.M_SL = [deque([], maxlen=memory_size_sl) for _ in range(self.NUM_PLAYERS)]
    self.M_RL = [deque([], maxlen=memory_size_rl) for _ in range(self.NUM_PLAYERS)]

    self.infoSets_dict_player = [[] for _ in range(self.NUM_PLAYERS)]
    self.infoSets_dict = {}

    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)

    self.best_response_strategy = copy.deepcopy(self.avg_strategy)



    self.N_count = copy.deepcopy(self.avg_strategy)
    for node, cn in self.N_count.items():
      self.N_count[node] = np.array([1.0 for _ in range(self.NUM_ACTIONS)], dtype=float)


    # q_value
    self.Q_value = [np.zeros((len(self.infoSets_dict_player[i]),2)) for i in range(self.NUM_PLAYERS)]


    self.RL = Online_FSP_Kuhn_Poker_reinforcement_learning.ReinforcementLearning(self.infoSets_dict_player, self.NUM_PLAYERS, self.NUM_ACTIONS)
    self.SL = Online_FSP_Kuhn_Poker_supervised_learning.SupervisedLearning(self.NUM_PLAYERS, self.NUM_ACTIONS)
    self.GD = Online_FSP_Kuhn_Poker_generate_data.GenerateData(self.NUM_PLAYERS, self.NUM_ACTIONS)

    for iteration_t in tqdm(range(1, int(self.train_iterations)+1)):

      #0 → epsilon_greedy_q_strategy, 1 → avg_strategy
      self.sigma_strategy_bit = [-1 for _ in range(self.NUM_PLAYERS)]
      for player_i in range(self.NUM_PLAYERS):
        if np.random.uniform() < self.eta:
          self.sigma_strategy_bit[player_i] = 0
        else:
          self.sigma_strategy_bit[player_i] = 1



      cards = self.card_distribution(self.NUM_PLAYERS)
      random.shuffle(cards)
      history = "".join(cards[:self.NUM_PLAYERS])

      self.train_one_episode(history, iteration_t)




      if iteration_t in [int(j) for j in np.logspace(0, len(str(self.train_iterations)), (len(str(self.train_iterations)))*4 , endpoint=False)] :
        self.exploitability_list[iteration_t] = self.get_exploitability_dfs()
        self.avg_utility_list[iteration_t] = self.eval_vanilla_CFR("", 0, 0, [1.0 for _ in range(self.NUM_PLAYERS)])

        self.optimality_gap = 0
        self.infoSets_dict = {}
        for target_player in range(self.NUM_PLAYERS):
          self.create_infoSets("", target_player, 1.0)
        self.best_response_strategy_dfs = {}
        for best_response_player_i in range(self.NUM_PLAYERS):
          self.calc_best_response_value(self.best_response_strategy_dfs, best_response_player_i, "", 1)

        for player_i in range(self.NUM_PLAYERS):
          self.optimality_gap += 1/2 * (self.GD.calculate_optimal_gap_best_response_strategy(self.best_response_strategy_dfs, self.avg_strategy, player_i)
           - self.GD.calculate_optimal_gap_best_response_strategy(self.best_response_strategy, self.avg_strategy, player_i))


        if wandb_save:
          wandb.log({'iteration': iteration_t, 'exploitability': self.exploitability_list[iteration_t], 'avg_utility': self.avg_utility_list[iteration_t], 'optimal_gap':self.optimality_gap})


    #print(self.M_RL)



doctest.testmod()
