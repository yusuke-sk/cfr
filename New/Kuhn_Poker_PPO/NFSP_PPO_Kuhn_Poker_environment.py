
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random





class Environment_Kuhn:
  def __init__(self, num_players, target_player, strategy):
    self.NUM_PLAYERS = num_players
    self.NUM_ACTIONS = 2
    self.target_player = target_player
    self.strategy = strategy

    self.card_rank = self.make_rank()
    self.history = ""



  def reset(self):
      cards = self.card_distribution(self.NUM_PLAYERS)
      random.shuffle(cards)
      history = "".join(cards[:self.NUM_PLAYERS])


      while len(history)%self.NUM_PLAYERS != self.target_player :
        player = len(history)%self.NUM_PLAYERS
        s = history[player] + history[self.NUM_PLAYERS:]

        sampling_action = np.random.choice(list(range(self.NUM_ACTIONS)), p=self.strategy[s])

        a = ("p" if sampling_action == 0 else "b")
        history +=  a


      self.history = history

      observation = self.history[self.target_player] + self.history[self.NUM_PLAYERS:]

      return observation, self.history



  def step(self, action):


    a = ("p" if action == 0 else "b")
    self.history += a
    done = False
    target_player_action = False

    while not target_player_action:

      if self.whether_terminal_states(self.history):
        reward =  self.Return_payoff_for_terminal_states(self.history, self.target_player)
        done = True
        return None, reward, done, self.history


      elif len(self.history) % self.NUM_PLAYERS == self.target_player :
        reward = 0
        target_player_action = True


      else:
        player = len(self.history) % self.NUM_PLAYERS
        s = self.history[player] + self.history[self.NUM_PLAYERS:]
        sampling_action = np.random.choice(list(range(self.NUM_ACTIONS)), p=self.strategy[s])

        a = ("p" if sampling_action == 0 else "b")
        self.history +=  a


    observation = self.history[self.target_player] + self.history[self.NUM_PLAYERS:]
    return observation, reward, done, self.history


    pass


  def card_distribution(self, num_players):
    """return list
    >>> KuhnTrainer().card_distribution(2)
    ['J', 'Q', 'K']
    """
    card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]

    return card[11-num_players:]

  def whether_terminal_states(self, history):
    #pass only history
    if "b" not in history:
      return history.count("p") == self.NUM_PLAYERS

    plays = len(history)
    first_bet = history.index("b")
    return plays - first_bet -1  == self.NUM_PLAYERS -1


 #return util for terminal state for target_player_i
  def Return_payoff_for_terminal_states(self, history, target_player_i):

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
