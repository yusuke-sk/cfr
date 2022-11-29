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

import warnings
warnings.filterwarnings('ignore')




class SupervisedLearning:
  def __init__(self, num_players=2, num_actions=2, random_seed = 42):
    self.num_players = num_players
    self.num_actions = num_actions
    self.max_len_X_bit = (self.num_players + 1) + 2*(self.num_players *2 - 2)

    self.card_order  = self.make_card_order(self.num_players)

    self.random_seed = random_seed
    self.random_seed_fix(random_seed = self.random_seed)


  def make_card_order(self, num_players):
    """return dict
    >>> SupervisedLearning().make_card_order(2) == {'J':0, 'Q':1, 'K':2}
    True
    """
    card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    card_order = {}
    for i in range(num_players+1):
      card_order[card[11-num_players+i]] =  i

    return card_order

  # exploitability: 収束する
  def SL_train_AVG(self, memory, target_player, strategy, n_count):
    for one_episode in memory:
      one_episode_split = self.Episode_split(one_episode)

      for X, y in one_episode_split:
        if (len(X)-1) % self.num_players == target_player :
          if y == "p":
            n_count[X] += np.array([1.0, 0.0], dtype=float)
          else:
            n_count[X] += np.array([0.0, 1.0], dtype=float)

    for node_X , action_prob in n_count.items():
        strategy[node_X] = n_count[node_X] / np.sum(action_prob)


    return strategy


  # exploitability: 収束しない
  def SL_train_MLP(self, memory, target_player, update_strategy):

      train_X = np.array([])
      train_y = np.array([])
      for one_episode in memory:
        train = self.From_episode_to_bit(one_episode, target_player)
        for train_i in train:
          train_X = np.append(train_X, train_i[0])
          train_y = np.append(train_y, train_i[1])

      train_X = train_X.reshape(-1, self.max_len_X_bit)
      train_y = train_y.reshape(-1, 1)
      #print(train_X.shape, train_y.shape)

      #モデル構築 多層パーセプトロン
      clf = MLPClassifier(hidden_layer_sizes=(200,))
      clf.fit(train_X, train_y)

      for node_X , _ in update_strategy.items():
        if (len(node_X)-1) % self.num_players == target_player :
          node_bit_X = self.make_X(node_X).reshape(-1, self.max_len_X_bit)
          y = clf.predict_proba(node_bit_X).ravel()
          update_strategy[node_X] = y



  def From_episode_to_bit(self, one_episode, target_player):
    """return list
    >>> SupervisedLearning(2, 2).From_episode_to_bit('QKbp', 0)
    [(array([0, 1, 0, 0, 0, 0, 0]), array([1]))]
    >>> SupervisedLearning(2, 2).From_episode_to_bit('QKbp', 1)
    [(array([0, 0, 1, 0, 1, 0, 0]), array([0]))]
    """
    one_episode_split = self.Episode_split(one_episode)
    one_episode_bit = []
    for X, y in one_episode_split:
      if (len(X)-1) % self.num_players == target_player :
        y_bit = self.make_y(y)
        X_bit = self.make_X(X)
        one_episode_bit.append((X_bit, y_bit))

    return one_episode_bit

  def make_y(self, y):
    if y == "p":
      y_bit = np.array([0])
    else:
      y_bit = np.array([1])
    return y_bit

  def make_X(self, X):

    X_bit = np.array([0 for _ in range(self.max_len_X_bit)])
    X_bit[self.card_order[X[0]]] = 1

    for idx, Xi in enumerate(X[1:]):
      if Xi == "p":
        X_bit[(self.num_players+1) + 2*idx] = 1
      else:
        X_bit[(self.num_players+1) + 2*idx +1] = 1

    return X_bit


  def first_bit(self, X0, X_bit):
    if X0 == "J":
      X_bit[0] = 1
    elif X0 == "Q":
      X_bit[1] = 1
    elif X0 == "K":
      X_bit[2] = 1
    return X_bit


  def Episode_split(self, one_episode):
    """return list
    >>> SupervisedLearning(2, 2).Episode_split('QKbp')
    [('Q', 'b'), ('Kb', 'p')]
    >>> SupervisedLearning(2, 2).Episode_split('KQpbp')
    [('K', 'p'), ('Qp', 'b'), ('Kpb', 'p')]
    """
    one_episode_split = []
    action_history = one_episode[self.num_players:]
    for idx, ai in enumerate(action_history):
      s = one_episode[idx%self.num_players] + action_history[:idx]
      a = ai
      one_episode_split.append((s,a))

    return one_episode_split


  def random_seed_fix(self, random_seed):
      random.seed(random_seed)
      np.random.seed(random_seed)


doctest.testmod()
