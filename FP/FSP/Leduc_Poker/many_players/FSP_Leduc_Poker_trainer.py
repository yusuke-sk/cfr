#Library
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
from sklearn.neural_network import MLPClassifier
from collections import deque
import wandb


import FSP_Leduc_Poker_supervised_learning
import FSP_Leduc_Poker_reinforcement_learning
import FSP_Leduc_Poker_generate_data



class LeducTrainer:
  def __init__(self, train_iterations=10**1, num_players =2, random_seed = 42, save_matplotlib = False):
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.NUM_ACTIONS = 3
    self.ACTION_DICT = {0:"f", 1:"c", 2:"r"}
    self.avg_strategy = {}
    self.node_possible_action = {}
    self.card_rank = self.make_rank()
    self.history_action_player_dict = {}
    self.random_seed = random_seed
    self.save_matplotlib = save_matplotlib

    self.random_seed_fix(self.random_seed)

  def Get_possible_action_by_information_set(self, infoset): #{0:"f", 1:"c", 2:"r"}
    """return int
    >>> LeducTrainer().Get_possible_action_by_information_set("JccKc")
    array([1, 2])
    >>> LeducTrainer().Get_possible_action_by_information_set("Jr")
    array([0, 1, 2])
    >>> LeducTrainer().Get_possible_action_by_information_set("JccJc")
    array([1, 2])
    >>> LeducTrainer().Get_possible_action_by_information_set("J")
    array([1, 2])
    """
    infoset_without_hand_card = infoset[1:]
    if self.card_num_check(infoset_without_hand_card) == 1:
      private_cards, history_before, community_card, history_after = self.Split_history("??" + infoset_without_hand_card)
      infoset_without_hand_card = history_after

    if  len(infoset_without_hand_card) == 0 or infoset_without_hand_card.count("r") == 0:
      return np.array([1,2], dtype=int)
    elif infoset_without_hand_card.count("r") == 1:
      return np.array([0,1,2], dtype=int)
    elif infoset_without_hand_card.count("r") == 2:
      return np.array([0,1], dtype=int)


  def make_rank(self):
    """return dict
    >>> LeducTrainer(num_players=2).make_rank() == {"KK":6, "QQ":5, "JJ":4, "KQ":3, "QK":3, "KJ":2, "JK":2, "QJ":1, "JQ":1}
    True
    """
    card_deck = self.card_distribution()
    card_unique = card_deck[::2]
    card_rank = {}
    count = (len(card_unique)-1)*len(card_unique) //2
    for i in range(len(card_unique)-1,-1, -1):
      for j in range(i-1, -1, -1):
            card_rank[card_unique[i] + card_unique[j]] = count
            card_rank[card_unique[j] + card_unique[i]] = count
            count -= 1

    count = (len(card_unique)-1)*len(card_unique) //2 +1
    for i in range(len(card_unique)):
        card_rank[card_unique[i] + card_unique[i]] = count
        count += 1

    return card_rank


  def Rank(self, my_card, com_card):
    """return int
    >>> LeducTrainer(num_players=2).Rank("J", "Q")
    1
    >>> LeducTrainer(num_players=2).Rank("Q", "J")
    1
    >>> LeducTrainer(num_players=2).Rank("K", "K")
    6
    """
    hand = my_card + com_card
    return self.card_rank[hand]


  def card_distribution(self):
    """return list
    >>> LeducTrainer(num_players=2).card_distribution()
    ['J', 'J', 'Q', 'Q', 'K', 'K']
    >>> LeducTrainer(num_players=3).card_distribution()
    ['T', 'T', 'J', 'J', 'Q', 'Q', 'K', 'K']
    """
    card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    card_deck = []
    for i in range(self.NUM_PLAYERS+1):
      card_deck.append(card[11-self.NUM_PLAYERS+i])
      card_deck.append(card[11-self.NUM_PLAYERS+i])

    return card_deck


  def Split_history(self, history):
    """return history_before, history_after
    >>> LeducTrainer(num_players=3).Split_history("JKQcccKcrcc")
    ('JKQ', 'ccc', 'K', 'crcc')
    >>> LeducTrainer(num_players=2).Split_history("KQrrcQrrc")
    ('KQ', 'rrc', 'Q', 'rrc')
    >>> LeducTrainer(num_players=2).Split_history("QQcrrcKcc")
    ('QQ', 'crrc', 'K', 'cc')
    """
    for ai, history_ai in enumerate(history[self.NUM_PLAYERS:]):
      if history_ai in self.card_distribution():
        idx = ai+self.NUM_PLAYERS
        community_catd = history_ai
    return history[:self.NUM_PLAYERS], history[self.NUM_PLAYERS:idx], community_catd, history[idx+1:]


  def action_history_player(self, history):
    #target_player_iのaction 履歴
    player_action_list = [[] for _ in range(self.NUM_PLAYERS)]
    player_money_list_round1 = [1 for _ in range(self.NUM_PLAYERS)]

    player_money_list_round2 = [0 for _ in range(self.NUM_PLAYERS)]

    f_count, a_count, raise_count = 0, 0, 0

    card = self.card_distribution()
    private_cards, history_before, community_card, history_after = self.Split_history(history)
    for hi in history_before:
      while len(player_action_list[(a_count + f_count)%self.NUM_PLAYERS])>=1 and player_action_list[(a_count + f_count)%self.NUM_PLAYERS][-1] == "f":
        f_count += 1
      player_action_list[(a_count + f_count)%self.NUM_PLAYERS].append(hi)

      if hi == "c":
        player_money_list_round1[(a_count + f_count)%self.NUM_PLAYERS] = max(player_money_list_round1)
      elif hi == "r" and raise_count == 0:
        raise_count += 1
        player_money_list_round1[(a_count + f_count)%self.NUM_PLAYERS] += 2
      elif hi == "r" and raise_count == 1:
        player_money_list_round1[(a_count + f_count)%self.NUM_PLAYERS] += 4

      a_count += 1

    f_count, a_count, raise_count = 0, 0, 0

    for hi in history_after:
        if hi not in card:
          while len(player_action_list[(a_count + f_count)%self.NUM_PLAYERS])>=1 and player_action_list[(a_count + f_count)%self.NUM_PLAYERS][-1] == "f":
            f_count += 1
          player_action_list[(a_count + f_count)%self.NUM_PLAYERS].append(hi)

          if hi == "c":
            player_money_list_round2[(a_count + f_count)%self.NUM_PLAYERS] = max(player_money_list_round2)
          elif hi == "r" and raise_count == 0:
            raise_count += 1
            player_money_list_round2[(a_count + f_count)%self.NUM_PLAYERS] += 4
          elif hi == "r" and raise_count == 1:
            player_money_list_round2[(a_count + f_count)%self.NUM_PLAYERS] += 8

          a_count += 1

    return player_action_list, player_money_list_round1, player_money_list_round2, community_card


  def action_player(self, history):
    """return int
    >>> LeducTrainer().action_player("JJc")
    1
    >>> LeducTrainer().action_player("JQcr")
    0
    >>> LeducTrainer().action_player("JQr")
    1
    >>> LeducTrainer(num_players=3).action_player("JQTrfr")
    0
    """
    if history not in self.history_action_player_dict:
      player_action_list = [[] for _ in range(self.NUM_PLAYERS)]
      a_count = 0
      f_count = 0

      if self.card_num_check(history) == self.NUM_PLAYERS:

        for hi in history[self.NUM_PLAYERS:]:
          while len(player_action_list[(a_count + f_count)%self.NUM_PLAYERS])>=1 and player_action_list[(a_count + f_count)%self.NUM_PLAYERS][-1] == "f":
            f_count += 1
          player_action_list[(a_count + f_count)%self.NUM_PLAYERS].append(hi)
          a_count += 1
      elif self.card_num_check(history) == self.NUM_PLAYERS+1:

        private_cards, history_before, community_card, history_after = self.Split_history(history)
        for hi in history_after:
          while len(player_action_list[(a_count + f_count)%self.NUM_PLAYERS])>=1 and player_action_list[(a_count + f_count)%self.NUM_PLAYERS][-1] == "f":
            f_count += 1
          player_action_list[(a_count + f_count)%self.NUM_PLAYERS].append(hi)
          a_count += 1

      player_i = (a_count + f_count)%self.NUM_PLAYERS
      self.history_action_player_dict[history] = player_i

      return player_i

    else:
      return self.history_action_player_dict[history]


  #6 Return payoff for terminal states #if terminal states  return util
  def Return_payoff_for_terminal_states(self, history, target_player_i):
    """return int
    >>> int(LeducTrainer().Return_payoff_for_terminal_states("KQrf", 0))
    1
    >>> int(LeducTrainer().Return_payoff_for_terminal_states("QKcrf", 0))
    -1
    >>> int(LeducTrainer().Return_payoff_for_terminal_states("QKrrf", 0))
    -3

    >>> int(LeducTrainer().Return_payoff_for_terminal_states("JJccQcc", 0))
    0
    >>> int(LeducTrainer().Return_payoff_for_terminal_states("JKccQcc", 1))
    1
    >>> int(LeducTrainer().Return_payoff_for_terminal_states("JQcrcKcrc", 0))
    -7
    >>> int(LeducTrainer().Return_payoff_for_terminal_states("JQcrcKcrc", 1))
    7
    >>> int(LeducTrainer().Return_payoff_for_terminal_states("QKrrcQrrf", 0))
    -9
    >>> int(LeducTrainer().Return_payoff_for_terminal_states("QKrrcQrrc", 0))
    13
    >>> int(LeducTrainer().Return_payoff_for_terminal_states("QKrrcQcc", 0))
    5
    """
    #round1 finish
    if history.count("f") == self.NUM_PLAYERS -1  and self.card_num_check(history) == self.NUM_PLAYERS:
      player_action_list = [[] for _ in range(self.NUM_PLAYERS)]
      player_money_list_round1 = [1 for _ in range(self.NUM_PLAYERS)]
      player_money_list_round2 = [0 for _ in range(self.NUM_PLAYERS)]

      f_count, a_count, raise_count = 0, 0, 0

      for hi in history[self.NUM_PLAYERS:]:
        while len(player_action_list[(a_count + f_count)%self.NUM_PLAYERS])>=1 and player_action_list[(a_count + f_count)%self.NUM_PLAYERS][-1] == "f":
          f_count += 1
        player_action_list[(a_count + f_count)%self.NUM_PLAYERS].append(hi)

        if hi == "c":
          player_money_list_round1[(a_count + f_count)%self.NUM_PLAYERS] = max(player_money_list_round1)
        elif hi == "r" and raise_count == 0:
          raise_count += 1
          player_money_list_round1[(a_count + f_count)%self.NUM_PLAYERS] += 2
        elif hi == "r" and raise_count == 1:
          player_money_list_round1[(a_count + f_count)%self.NUM_PLAYERS] += 4

        a_count += 1
      if len(player_action_list[target_player_i]) >= 1 and player_action_list[target_player_i][-1] == "f":
        return -player_money_list_round1[target_player_i]
      else:
        return sum(player_money_list_round1) -player_money_list_round1[target_player_i]

    #round2 finish
    #target_player_i action history
    player_action_list, player_money_list_round1, player_money_list_round2, community_card = self.action_history_player(history)

    # target_player_i :fold
    if player_action_list[target_player_i][-1] == "f":
      return -player_money_list_round1[target_player_i] - player_money_list_round2[target_player_i]

    #周りがfold
    last_play =[hi[-1] for idx, hi in enumerate(player_action_list) if idx != target_player_i]
    if last_play.count("f") == self.NUM_PLAYERS - 1:
      return sum(player_money_list_round1) + sum(player_money_list_round2) - player_money_list_round1[target_player_i] - player_money_list_round2[target_player_i]

    #show down
    show_down_player =[idx for idx, hi in enumerate(player_action_list) if hi[-1] != "f"]
    show_down_player_card = {}
    for idx in show_down_player:
      show_down_player_card[idx] = self.Rank(history[idx], community_card)
    max_rank = max(show_down_player_card.values())
    if show_down_player_card[target_player_i] != max_rank:
      return - player_money_list_round1[target_player_i] - player_money_list_round2[target_player_i]
    else:
      win_num = len([idx for idx, card_rank in show_down_player_card.items() if card_rank == max_rank])

      return int((sum(player_money_list_round1) + sum(player_money_list_round2))/win_num) - player_money_list_round1[target_player_i] - player_money_list_round2[target_player_i]



  # whetther terminal_states
  def whether_terminal_states(self, history):
    """return string
    >>> LeducTrainer().whether_terminal_states("JKccKr")
    False
    >>> LeducTrainer().whether_terminal_states("QJccJcc")
    True
    >>> LeducTrainer().whether_terminal_states("QQcr")
    False
    >>> LeducTrainer(num_players=3).whether_terminal_states("QKTrff")
    True
    >>> LeducTrainer(num_players=3).whether_terminal_states("KKTcccQcrcrcc")
    True
    """
    if history.count("f") == self.NUM_PLAYERS -1 :
      return True

    if self.card_num_check(history) == self.NUM_PLAYERS +1 :
      private_cards, history_before, community_card, history_after = self.Split_history(history)
      if history_after.count("r") == 0 and history_after.count("c") == self.NUM_PLAYERS:
        return True

      if history.count("r") >=1 :
        idx = 0
        for i,hi in enumerate(history_after):
          if hi == "r":
            idx = i
        if history_after[idx+1:].count("c") == self.NUM_PLAYERS -1 :
          return True

    return False


  def card_num_check(self, history):
    """return string
    >>> LeducTrainer(num_players=3).card_num_check("JKTccc")
    3
    >>> LeducTrainer(num_players=2).card_num_check("KQcr")
    2
    """
    cards = self.card_distribution()
    count = 0
    for hi in history:
      if hi in cards:
        count += 1
    return count


  def whether_chance_node(self, history):
    """return string
    >>> LeducTrainer().whether_chance_node("JKcc")
    True
    >>> LeducTrainer().whether_chance_node("KQcr")
    False
    >>> LeducTrainer().whether_chance_node("")
    True
    >>> LeducTrainer(num_players=3).whether_chance_node("KQTcc")
    False
    """
    if history == "":
      return True

    if self.card_num_check(history) == self.NUM_PLAYERS :
      if history.count("r") == 0 and history.count("c") == self.NUM_PLAYERS:
        return True

      if history.count("r") >=1 :
        idx = 0
        for i,hi in enumerate(history):
          if hi == "r":
            idx = i
        if history[idx+1:].count("c") == self.NUM_PLAYERS -1 :
          return True

    return False


  # make node or get node
  def if_nonexistant(self, infoSet):
    if infoSet not in self.avg_strategy:
      self.node_possible_action[infoSet] = self.Get_possible_action_by_information_set(infoSet)
      self.avg_strategy[infoSet] = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)


      self.normalizingSum = 0
      for ai in self.node_possible_action[infoSet]:
        self.avg_strategy[infoSet][ai] = 1
        self.normalizingSum += 1

      self.avg_strategy[infoSet] /= self.normalizingSum


  def calc_best_response_value(self, best_response_strategy, best_response_player, history, prob):
      if self.card_num_check(history) == self.NUM_PLAYERS + 1:
        private_cards, history_before, community_card, history_after = self.Split_history(history)

      player = self.action_player(history)

      if self.whether_terminal_states(history):
        return self.Return_payoff_for_terminal_states(history, best_response_player)

      elif self.whether_chance_node(history):
        if len(history) == 0:
          cards = self.card_distribution()
          cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards, self.NUM_PLAYERS)]
          utility_sum = 0
          for cards_i in cards_candicates:
            nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
            utility =  (1/len(cards_candicates))* self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob)
            utility_sum += utility

          return utility_sum

        else:
          com_cards = self.card_distribution()
          com_cards.remove(history[0])
          com_cards.remove(history[1])

          utility_sum_round2 = 0
          for com_cards_i in com_cards:
            nextHistory = history + com_cards_i
            utility_sum_round2 += (1/len(com_cards))*self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob)

          return utility_sum_round2


      infoSet = history[player] + history[self.NUM_PLAYERS:]
      self.if_nonexistant(infoSet)

      if player == best_response_player:
        if infoSet not in best_response_strategy:
          action_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          br_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)

          for assume_history, po_ in self.infoSets_dict[infoSet].items():

            for ai in self.node_possible_action[infoSet]:
              nextHistory = assume_history + self.ACTION_DICT[ai]
              br_value[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, po_)
              action_value[ai] += br_value[ai] * po_

          #br_action = 0  ← action 0 を全てのノードで選択できるわけではないため不適切
          br_action = self.node_possible_action[infoSet][0]
          for ai in self.node_possible_action[infoSet]:
            if action_value[ai] > action_value[br_action]:
              br_action = ai
          best_response_strategy[infoSet] = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          best_response_strategy[infoSet][br_action] = 1.0

        node_util = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in self.node_possible_action[infoSet]:
          nextHistory = history + self.ACTION_DICT[ai]
          node_util[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob)
        best_response_util = 0
        for ai in self.node_possible_action[infoSet]:
          best_response_util += node_util[ai] * best_response_strategy[infoSet][ai]

        return best_response_util

      else:
        nodeUtil = 0
        action_value_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in self.node_possible_action[infoSet]:
          nextHistory = history + self.ACTION_DICT[ai]
          action_value_list[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob*self.avg_strategy[infoSet][ai])
          nodeUtil += self.avg_strategy[infoSet][ai] * action_value_list[ai]
        return nodeUtil


  def create_infoSets(self, history, target_player, po):
    player = self.action_player(history)

    if self.whether_terminal_states(history):
      return

    elif self.whether_chance_node(history):
      #round1
      if len(history) == 0:
        cards = self.card_distribution()
        cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards, self.NUM_PLAYERS)]
        for cards_candicates_i in cards_candicates:
          nextHistory = "".join(cards_candicates_i[:self.NUM_PLAYERS])
          self.create_infoSets(nextHistory, target_player, po*(1/len(cards_candicates)))
        return

      #round2
      else:
        com_cards_candicates = self.card_distribution()
        for player_i in range(self.NUM_PLAYERS):
          com_cards_candicates.remove(history[player_i])

        for com_cards_i in com_cards_candicates:
          nextHistory = history + com_cards_i
          self.create_infoSets(nextHistory, target_player, po*(1/len(com_cards_candicates)))
        return

    infoSet = history[player] + history[self.NUM_PLAYERS:]

    if player == target_player:
      if self.infoSets_dict.get(infoSet) is None:
        self.infoSets_dict[infoSet] = defaultdict(int)
        self.infoSets_dict_player[player].append(infoSet)
        self.infoset_action_player_dict[infoSet] = player

      self.infoSets_dict[infoSet][history]  += po


    self.if_nonexistant(infoSet)

    for ai in self.node_possible_action[infoSet]:
      nextHistory = history + self.ACTION_DICT[ai]
      if player == target_player:
        self.create_infoSets(nextHistory, target_player, po)
      else:

        actionProb = self.avg_strategy[infoSet][ai]
        self.create_infoSets(nextHistory, target_player, po*actionProb)

  def random_seed_fix(self, random_seed):
      random.seed(random_seed)
      np.random.seed(random_seed)


  def get_exploitability_dfs(self):

    # 各information setを作成 & reach_probabilityを計算
    self.infoSets_dict_player = [[] for _ in range(self.NUM_PLAYERS)]
    self.infoSets_dict = {}
    self.infoset_action_player_dict = {}

    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)


    exploitability = 0
    best_response_strategy = {}
    for best_response_player_i in range(self.NUM_PLAYERS):
        exploitability += self.calc_best_response_value(best_response_strategy, best_response_player_i, "", 1)

    assert exploitability >= 0
    return exploitability



  def eval_vanilla_CFR(self, history, target_player_i, iteration_t, p_list):
    if self.card_num_check(history) == self.NUM_PLAYERS + 1:
      private_cards, history_before, community_card, history_after = self.Split_history(history)

    player = self.action_player(history)

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
      if len(history) == 0:
        cards = self.card_distribution()
        cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards, self.NUM_PLAYERS+1)]
        utility_sum = 0
        for cards_i in cards_candicates:
          self.cards_i = cards_i
          nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
          utility_sum += (1/len(cards_candicates))* self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list)
        return  utility_sum
      else:
        nextHistory = history + self.cards_i[self.NUM_PLAYERS]
        return self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list)

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    self.if_nonexistant(infoSet)

    util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    nodeUtil = 0

    for ai in self.node_possible_action[infoSet]:
      nextHistory = history + self.ACTION_DICT[ai]
      p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
      p_change[player] = self.avg_strategy[infoSet][ai]

      util_list[ai] = self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list * p_change)

      nodeUtil += self.avg_strategy[infoSet][ai] * util_list[ai]


    return nodeUtil


  def show_plot(self, method):
    plt.scatter(list(self.exploitability_list.keys()), list(self.exploitability_list.values()), label=method)
    plt.plot(list(self.exploitability_list.keys()), list(self.exploitability_list.values()))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("iterations")
    plt.ylabel("exploitability")
    plt.legend()
    #plt.show()



  #KuhnTrainer main method
  def train(self, n, m, memory_size_rl, memory_size_sl, wandb_save, rl_algo, sl_algo, pseudo_code):
    self.exploitability_list = {}

    #追加 matplotlibで図を書くため
    if self.save_matplotlib:
      self.ex_name = "exploitability_for_{}_FSP".format(self.random_seed)
      self.database_for_plot = {"iteration":[] ,self.ex_name:[]}


    self.M_SL = [deque([], maxlen=memory_size_sl) for _ in range(self.NUM_PLAYERS)]
    self.M_RL = [deque([], maxlen=memory_size_rl) for _ in range(self.NUM_PLAYERS)]

    self.infoSets_dict_player = [[] for _ in range(self.NUM_PLAYERS)]
    self.infoSets_dict = {}
    self.infoset_action_player_dict = {}

    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)


    self.best_response_strategy = copy.deepcopy(self.avg_strategy)

    # n_count
    self.N_count = copy.deepcopy(self.avg_strategy)
    for node, cn in self.N_count.items():
      self.N_count[node] = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
      for key_i, valye_i in enumerate(self.avg_strategy[node]):
        if valye_i != 0:
          self.N_count[node][key_i] = 1.0

    # q_value
    self.Q_value = [np.zeros((len(self.infoSets_dict_player[i]),3)) for i in range(self.NUM_PLAYERS)]


    RL = FSP_Leduc_Poker_reinforcement_learning.ReinforcementLearning(self.random_seed, self.infoSets_dict_player, self.NUM_PLAYERS, self.NUM_ACTIONS, self.node_possible_action, self.infoset_action_player_dict)
    SL = FSP_Leduc_Poker_supervised_learning.SupervisedLearning(self.random_seed, self.NUM_PLAYERS, self.NUM_ACTIONS, self.node_possible_action, self.infoset_action_player_dict)
    GD = FSP_Leduc_Poker_generate_data.GenerateData(self.NUM_PLAYERS, self.NUM_ACTIONS, self.infoset_action_player_dict, self.random_seed)


    for iteration_t in tqdm(range(1, int(self.train_iterations)+1)):
      if pseudo_code == "batch_FSP":
        GD.generate_data1(self.avg_strategy, n, self.M_RL)

        if rl_algo == "dfs":
                self.infoSets_dict = {}
                for target_player in range(self.NUM_PLAYERS):
                  self.create_infoSets("", target_player, 1.0)
                self.best_response_strategy = {}
                for best_response_player_i in range(self.NUM_PLAYERS):
                  self.calc_best_response_value(self.best_response_strategy, best_response_player_i, "", 1)
        else:
          for player_i in range(self.NUM_PLAYERS):
            RL.RL_train(self.M_RL[player_i], player_i, self.best_response_strategy, self.Q_value[player_i], iteration_t, rl_algo)



        GD.generate_data2(self.avg_strategy, self.best_response_strategy, m, self.M_RL, self.M_SL)

        for player_i in range(self.NUM_PLAYERS):
          if sl_algo == "cnt":
            SL.SL_train_AVG(self.M_SL[player_i], player_i, self.avg_strategy, self.N_count)
            self.M_SL[player_i] = []
          elif sl_algo == "mlp":
            SL.SL_train_MLP(self.M_SL[player_i], player_i, self.avg_strategy)


      elif pseudo_code == "general_FSP":
        eta = 1/(iteration_t+1)
        D = GD.generate_data0(self.avg_strategy, self.best_response_strategy, n, m, eta)

        for player_i in range(self.NUM_PLAYERS):
          self.M_SL[player_i].extend(D[player_i])
          self.M_RL[player_i].extend(D[player_i])

          if rl_algo == "dfs":
                self.infoSets_dict = {}
                for target_player in range(self.NUM_PLAYERS):
                  self.create_infoSets("", target_player, 1.0)
                self.best_response_strategy = {}
                for best_response_player_i in range(self.NUM_PLAYERS):
                  self.calc_best_response_value(self.best_response_strategy, best_response_player_i, "", 1)
          else:
            RL.RL_train(self.M_RL[player_i], player_i, self.best_response_strategy, self.Q_value[player_i], iteration_t, rl_algo)


          if sl_algo == "cnt":
            SL.SL_train_AVG(self.M_SL[player_i], player_i, self.avg_strategy, self.N_count)
          elif sl_algo == "mlp":
            SL.SL_train_MLP(self.M_SL[player_i], player_i, self.avg_strategy)


      if iteration_t in [int(j) for j in np.logspace(0, len(str(self.train_iterations)), (len(str(self.train_iterations)))*4 , endpoint=False)] :
        self.exploitability_list[iteration_t] = self.get_exploitability_dfs()

        if wandb_save:
          wandb.log({'iteration': iteration_t, 'exploitability': self.exploitability_list[iteration_t]})

        #追加 matplotlibで図を書くため
        if self.save_matplotlib:
          self.database_for_plot["iteration"].append(iteration_t)
          self.database_for_plot[self.ex_name].append(self.exploitability_list[iteration_t])


    #self.show_plot("FSP")
    if wandb_save:
      wandb.save()


doctest.testmod()
