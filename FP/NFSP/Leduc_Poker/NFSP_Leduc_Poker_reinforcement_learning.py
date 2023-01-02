
# _________________________________ Library _________________________________


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

# _________________________________ RL NN class _________________________________
class DQN(nn.Module):

    def __init__(self, state_num, action_num, hidden_units_num):
        super(DQN, self).__init__()
        self.hidden_units_num = hidden_units_num
        self.fc1 = nn.Linear(state_num, self.hidden_units_num)
        self.fc2 = nn.Linear(self.hidden_units_num, action_num)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        #h1 = F.leaky_relu(self.fc1(x))
        output = self.fc2(h1)

        return output


# _________________________________ RL class _________________________________
class ReinforcementLearning:
  def __init__(self, train_iterations, num_players, hidden_units_num, lr, epochs, sampling_num, gamma, tau, \
    update_frequency, leduc_trainer_for_rl, random_seed, alpha, rl_strategy, alpha_discrease):
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.num_actions = 3
    self.action_id = {"f":0, "c":1, "r":2}
    self.STATE_BIT_LEN = 2* ( (self.NUM_PLAYERS + 1) + 3*(self.NUM_PLAYERS *3 - 2) ) - 3
    self.hidden_units_num = hidden_units_num
    self.lr = lr
    self.epochs = epochs
    self.sampling_num = sampling_num
    self.gamma = gamma
    self.tau = tau
    self.update_frequency = update_frequency
    self.random_seed = random_seed

    self.leduc_trainer = leduc_trainer_for_rl
    self.card_rank  = self.leduc_trainer.card_rank
    self.infoset_action_player_dict = {}
    self.rl_algo = None

    self.alpha = alpha
    self.rl_strategy = rl_strategy

    self.alpha_discrease = alpha_discrease

    if self.alpha_discrease:
      self.initial_alpha = self.alpha

    self.deep_q_network = DQN(state_num = self.STATE_BIT_LEN, action_num = self.num_actions, hidden_units_num = self.hidden_units_num)
    self.deep_q_network_target = DQN(state_num = self.STATE_BIT_LEN, action_num = self.num_actions, hidden_units_num = self.hidden_units_num)

    self.leduc_trainer.random_seed_fix(self.random_seed)

    self.deep_q_network_target.load_state_dict(self.deep_q_network.state_dict())


    self.optimizer = optim.SGD(self.deep_q_network.parameters(), lr=self.lr, weight_decay=5*(10**(-4)))

    self.update_count =  0
    self.save_count = 0
    self.epsilon = 0


  def RL_learn(self, memory, update_strategy, k):

    self.deep_q_network.train()
    self.deep_q_network_target.eval()
    self.epsilon = 0.06/(k**0.5)

    #new alpha change
    if self.alpha_discrease:
      self.alpha = self.initial_alpha/(k**0.5)


    total_loss = []

    for _ in range(self.epochs):
      samples = random.sample(memory, min(self.sampling_num, len(memory)))
      train_states = [sars[0] for sars in samples]
      train_actions = [sars[1] for sars in samples]
      train_rewards = [sars[2] for sars in samples]
      s_prime_array = [sars[3] for sars in samples]
      train_next_states = [sars[4] for sars in samples]
      train_done = [sars[5] for sars in samples]
      outputs = []


      train_states = torch.tensor(train_states).float().reshape(-1,self.STATE_BIT_LEN)
      train_actions = torch.tensor(train_actions).float().reshape(-1,1)
      train_rewards = torch.tensor(train_rewards).float().reshape(-1,1)
      train_next_states = torch.tensor(train_next_states).float().reshape(-1,self.STATE_BIT_LEN)
      train_done = torch.tensor(train_done).float().reshape(-1,1)


      if  self.rl_algo in ["dqn", "ddqn"] :
        if self.rl_algo == "dqn":
          outputs_all = self.deep_q_network_target(train_next_states).detach()

        #Double DQN
        elif self.rl_algo == "ddqn":
          not_target_nn_max_action_list = []

          for node_X, Q_value in zip(s_prime_array, self.deep_q_network(train_next_states)):
            action_list = self.leduc_trainer.node_possible_action[node_X]
            max_idx = action_list[0]
            if node_X == None:
              not_target_nn_max_action_list.append(max_idx)
            else:
              for ai in action_list:
                if Q_value[ai] >= Q_value[max_idx]:
                  max_idx = ai

              not_target_nn_max_action_list.append(max_idx)


          outputs_all = self.deep_q_network_target(train_next_states).gather(1,not_target_nn_max_action_list.type(torch.int64)).detach()

        for node_X, Q_value in zip(s_prime_array, outputs_all):
          if node_X == None:
            outputs.append(0)
          else:
            action_list = self.leduc_trainer.node_possible_action[node_X]
            max_idx = action_list[0]

            for ai in action_list:
              if Q_value[ai] >= Q_value[max_idx]:
                max_idx = ai


            outputs.append(Q_value[max_idx])

      #SQL
      elif self.rl_algo == "sql":
        outputs_all = self.deep_q_network_target(train_next_states).detach()

        for node_X, Q_value in zip(s_prime_array, outputs_all):
          if node_X == None:
            outputs.append(0)
          else:
            action_list = self.leduc_trainer.node_possible_action[node_X]
            q_values = Q_value[action_list].reshape(1, len(action_list))
            output = self.alpha * torch.logsumexp(q_values/self.alpha, dim=1)
            outputs.append(output.item())


      outputs = torch.tensor(outputs).float().unsqueeze(1)
      q_targets = train_rewards + (1 - train_done) * self.gamma * outputs

      q_now = self.deep_q_network(train_states)
      q_now_value = q_now.gather(1, train_actions.type(torch.int64))

      loss = F.mse_loss(q_targets, q_now_value)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      total_loss.append(loss.item())


      self.update_count += 1


      if self.update_count % self.update_frequency ==  0 :
        self.deep_q_network_target.load_state_dict(self.deep_q_network.state_dict())



    if self.leduc_trainer.wandb_save and self.save_count % 100 == 0:
      wandb.log({'iteration': k, 'loss_rl':  np.mean(total_loss)})
    self.save_count += 1



  def action_step(self, node_X):
    self.deep_q_network.eval()
    with torch.no_grad():
      inputs_eval = torch.Tensor(self.leduc_trainer.make_state_bit(node_X)).float().reshape(-1,self.STATE_BIT_LEN)
      y = self.deep_q_network.forward(inputs_eval).detach().numpy()
      action_list = self.leduc_trainer.node_possible_action[node_X]
      if np.random.uniform() < self.epsilon:   # 探索(epsilonの確率で)
        action = np.random.choice(action_list)
        strategy = np.array([0 for _ in range(self.num_actions)], dtype=float)
        strategy[action] = 1.0

        return strategy

      else:
        max_idx = action_list[0]

        for ai in action_list:
          if y[0][ai] >= y[0][max_idx]:
            max_idx = ai
        strategy = np.array([0 for _ in range(self.num_actions)], dtype=float)
        strategy[max_idx] = 1.0

        return strategy


  def update_strategy_for_table(self, update_strategy):
    #eval
    self.deep_q_network.eval()
    with torch.no_grad():
      for node_X , _ in update_strategy.items():

        inputs_eval = torch.Tensor(self.leduc_trainer.make_state_bit(node_X)).float().reshape(-1,self.STATE_BIT_LEN)
        y = self.deep_q_network.forward(inputs_eval).detach().numpy()

        if np.random.uniform() < self.epsilon:   # 探索(epsilonの確率で)
          action = np.random.choice(self.leduc_trainer.node_possible_action[node_X])
          update_strategy[node_X] = np.array([0 for _ in range(self.num_actions)], dtype=float)
          update_strategy[node_X][action] = 1.0

        else:
          action_list = self.leduc_trainer.node_possible_action[node_X]
          max_idx = action_list[0]

          for ai in action_list:
            if y[0][ai] >= y[0][max_idx]:
              max_idx = ai
          update_strategy[node_X] = np.array([0 for _ in range(self.num_actions)], dtype=float)
          update_strategy[node_X][max_idx] = 1.0



doctest.testmod()
