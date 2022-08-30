
# _________________________________ Library _________________________________
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
from collections import defaultdict
from tqdm import tqdm
import time
import doctest
import copy
from collections import deque
import wandb

import torch
import torch.nn as nn

from NFSP_PPO_Kuhn_Poker_reinforcement_learning import PPO

from NFSP_PPO_Kuhn_Poker_environment import Environment_Kuhn




# _________________________________ Train class _________________________________
class KuhnTrainer:
  def __init__(self,random_seed=42, train_iterations=10, num_parallel=1, num_players=2, wandb_save=False):
    self.train_iterations = train_iterations
    self.num_parallel = num_parallel
    self.NUM_PLAYERS = num_players
    self.NUM_ACTIONS = 2
    self.STATE_BIT_LEN = (self.NUM_PLAYERS + 1) + 2*(self.NUM_PLAYERS *2 - 2)
    self.wandb_save = wandb_save
    self.card_rank = self.make_rank()
    self.avg_strategy = {}
    self.memory_count_for_sl = 0

    self.random_seed = random_seed

    self.random_seed_fix(self.random_seed)


# _________________________________ Train main method _________________________________
  def train(self,rl_algo, sl_algo, rl_module, sl_module, gd_module, SL_memory, RL_memory):
    self.exploitability_list = {}
    self.avg_utility_list = {}
    self.rl_algo = rl_algo
    self.sl_algo = sl_algo


    self.infoSets_dict_player = [[] for _ in range(self.NUM_PLAYERS)]
    self.infoSets_dict = {}

    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)


    self.best_response_strategy = copy.deepcopy(self.avg_strategy)


    self.RL = rl_module
    self.SL = sl_module
    self.GD = gd_module

    self.RL.strategy = self.best_response_strategy


    self.SL_memory = SL_memory
    self.RL_memory = RL_memory

    if self.sl_algo == "cnt":
      self.N_count = copy.deepcopy(self.avg_strategy)
      for node, cn in self.N_count.items():
        self.N_count[node] = np.array([1.0 for _ in range(self.NUM_ACTIONS)], dtype=float)


    for iteration_t in tqdm(range(1, int(self.train_iterations)+1)):


      for paralell in range(self.num_parallel):

        self.target_player = paralell % self.NUM_PLAYERS


        self.env = Environment_Kuhn(self.NUM_PLAYERS, self.target_player, self.avg_strategy)

        # 1 episode
        history_1 = self.train_one_episode(self.env, self.target_player)

      #方策を学習

      if self.sl_algo == "cnt":
        self.SL.SL_train_AVG(self.SL_memory, self.avg_strategy, self.N_count)
      else:
        self.SL.SL_learn(self.SL_memory, self.avg_strategy, iteration_t)


      self.RL.RL_learn(self.RL_memory)

      #print(self.RL_memory.get_size())


      self.RL_memory.delete()

      if self.sl_algo == "cnt":
        self.SL_memory.delete()


      #print(self.best_response_strategy)

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
          self.optimality_gap_i = 1/2 * (self.GD.calculate_optimal_gap_best_response_strategy(self.best_response_strategy_dfs, self.avg_strategy, player_i)
           - self.GD.calculate_optimal_gap_best_response_strategy(self.best_response_strategy, self.avg_strategy, player_i))


          self.optimality_gap += self.optimality_gap_i

        if self.wandb_save:
          wandb.log({'iteration': iteration_t, 'exploitability': self.exploitability_list[iteration_t], 'avg_utility': self.avg_utility_list[iteration_t], 'optimal_gap':self.optimality_gap})




# _________________________________ Train second main method _________________________________
  def train_one_episode(self, history, target_player):

    done = False
    observation, history = self.env.reset()


    while  not done:

      # target_player
      observation_bit = self.make_state_bit(observation)
      action, log_prob, value = self.RL.select_action(observation, torch.tensor(observation_bit).float())

      observation_next, reward, done, history_next = self.env.step(action)

      self.RL_memory.store((observation_bit, action, log_prob, value, reward, done))


      if self.sl_algo == "cnt":
        self.SL_memory.store((observation, action))
      else:
        self.SL_memory.store((observation_bit, action))


      observation = observation_next
      history = history_next


    return history



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


  # ensure_exsist, create if
  # if → return true or false
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

    # whether → if
    if self.whether_terminal_states(history):
      return

    elif self.whether_chance_node(history):
      cards = self.card_distribution(self.NUM_PLAYERS)
      cards_candicates = [list(cards_candicate) for cards_candicate in itertools.permutations(cards)]
      for cards_i in cards_candicates:
        nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
        self.create_infoSets(nextHistory, target_player, po)
      return

    #history JQKpp → Kpp player2
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



  def make_rank(self):
    """return dict
    >>> KuhnTrainer().make_rank() == {'J':0, 'Q':1, 'K':2}
    True
    """
    card_rank = {}
    card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    for i in range(self.NUM_PLAYERS+1):
      card_rank[card[11-self.NUM_PLAYERS+i]] =  i
    return card_rank



  def from_episode_to_bit(self, one_s_a_set):
    """return list
    >>> KuhnTrainer().from_episode_to_bit([('Q', 'b')])
    ([0, 1, 0, 0, 0, 0, 0], [1])
    """

    for X, y in one_s_a_set:
      y_bit = self.make_action_bit(y)
      X_bit = self.make_state_bit(X)

    return (X_bit,y_bit)


  def make_action_bit(self, y):
    if y == "p":
      y_bit = [0]
    else:
      y_bit = [1]
    return y_bit


  def make_state_bit(self, X):
    """return list
    >>> KuhnTrainer().make_state_bit("J")
    [1, 0, 0, 0, 0, 0, 0]
    >>> KuhnTrainer().make_state_bit("Kb")
    [0, 0, 1, 0, 1, 0, 0]
    """
    X_bit = [0 for _ in range(self.STATE_BIT_LEN)]

    if X != None:

      X_bit[self.card_rank[X[0]]] = 1

      for idx, Xi in enumerate(X[1:]):
        if Xi == "p":
          X_bit[(self.NUM_PLAYERS+1) + 2*idx] = 1
        else:
          X_bit[(self.NUM_PLAYERS+1) + 2*idx +1] = 1

    return X_bit


  def random_seed_fix(self, random_seed):
      random.seed(random_seed)
      np.random.seed(random_seed)
      torch.manual_seed(random_seed)



doctest.testmod()
