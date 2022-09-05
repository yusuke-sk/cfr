
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
from torch.distributions import Categorical


#import warnings
#warnings.filterwarnings('ignore')



class ActorNetwork(nn.Module):
    def __init__(self, state_num, action_num, hidden_units_num ):
        super(ActorNetwork, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.hidden_units_num = hidden_units_num


        self.actor = nn.Sequential(
            nn.Linear(self.state_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, self.action_num),
            )


    def forward(self, x):

      action_probs = F.softmax(self.actor(x).reshape(-1, 2), dim=1)

      action_dist = Categorical(action_probs)
      actions = action_dist.sample().view(-1, 1)

      log_action_probs = torch.log(action_probs + 1e-8)

      #print(x, action_probs)

      return actions, action_probs, log_action_probs


class CriticNetwork(nn.Module):
    def __init__(self, state_num, action_num, hidden_units_num):
        super(CriticNetwork, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.hidden_units_num = hidden_units_num

        self.critic = nn.Sequential(
            nn.Linear(self.state_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, self.action_num)
            )


    def forward(self, x):
        output = self.critic(x)
        return output


class Double_Q_network(nn.Module):
    def __init__(self, state_num, action_num, hidden_units_num):
        super(Double_Q_network, self).__init__()
        self.Q1 = CriticNetwork(state_num, action_num, hidden_units_num)
        self.Q2 = CriticNetwork(state_num, action_num, hidden_units_num)

    def forward(self, x):
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


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
    self.update_count = 0

    self.rl_algo = None

    self.kuhn_trainer.random_seed_fix(self.random_seed)

    self.loss_fn = loss_function


    self.critic = Double_Q_network(self.STATE_BIT_LEN, self.num_actions, self.hidden_units_num,).to(self.device)
    self.critic_target = Double_Q_network(self.STATE_BIT_LEN, self.num_actions, self.hidden_units_num).to(self.device)

    self.actor = ActorNetwork(self.STATE_BIT_LEN, self.num_actions, self.hidden_units_num).to(self.device)

    self.critic_1_optim = optim.Adam(self.critic.Q1.parameters(), lr = self.lr)
    self.critic_2_optim = optim.Adam(self.critic.Q2.parameters(), lr = self.lr)
    self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.lr)

    self.critic_target.load_state_dict(self.critic.state_dict())

    self.target_entropy_ratio = 0.98

    self.target_entropy = - np.log(1.0/self.num_actions) * self.target_entropy_ratio

    #optimize log_alpha
    self.log_alpha = torch.zeros(1, requires_grad=True).to(self.device)
    self.alpha = self.log_alpha.exp()
    self.alpha_optim = optim.Adam([self.log_alpha], lr = lr)




  def RL_learn(self, memory, target_player, update_strategy, k):

    samples = random.sample(memory, min(self.sampling_num, len(memory)))

    train_states = [sars[0] for sars in samples]
    train_actions = [sars[1] for sars in samples]
    train_rewards = [sars[2] for sars in samples]
    train_next_states = [sars[3] for sars in samples]
    train_done = [sars[4] for sars in samples]

    train_states = torch.tensor(train_states).float().reshape(-1,self.STATE_BIT_LEN).to(self.device)
    train_actions = torch.tensor(train_actions).float().reshape(-1,1).to(self.device)
    train_rewards = torch.tensor(train_rewards).float().reshape(-1,1).to(self.device)
    train_next_states = torch.tensor(train_next_states).float().reshape(-1,self.STATE_BIT_LEN).to(self.device)
    train_done = torch.tensor(train_done).float().reshape(-1,1).to(self.device)


    #Q関数の更新 J(θ)
    q1_loss, q2_loss = self.calc_critic_loss(train_states, train_actions, train_rewards, train_next_states, train_done)

    self.critic_1_optim.zero_grad()
    q1_loss.backward()
    self.critic_1_optim.step()

    self.critic_2_optim.zero_grad()
    q2_loss.backward()
    self.critic_2_optim.step()


    #方策の更新
    policy_loss, entropies = self.calc_policy_loss(train_states, train_actions, train_rewards, train_next_states, train_done)


    self.actor_optim.zero_grad()
    policy_loss.backward()
    self.actor_optim.step()

    #print(policy_loss)


    #エントロピー係数の更新
    entropy_loss = self.calc_entropy_loss(entropies)

    self.alpha_optim.zero_grad()
    entropy_loss.backward()
    self.alpha_optim.step()

    self.alpha = self.log_alpha.exp()

    self.update_count += 1


    if self.update_count % self.update_frequency ==  0 :
            self.critic_target.load_state_dict(self.critic.state_dict())


  def calc_q_value(self, states, actions):
    """
    入力 state, action → 出力 2つのネットワークの Q(s,a) の値
    """

    current_q1_values, current_q2_values = self.critic(states)
    current_q1 = current_q1_values.gather(1, actions.long())
    current_q2 = current_q2_values.gather(1, actions.long())

    return current_q1, current_q2


  def calc_target_q_value(self, states, actions, rewards, next_states, done):
    # 目標値: next_value

    with torch.no_grad():
      next_actions, next_action_prob, next_action_log_prob = self.actor(next_states)
      next_q1_value, next_q2_value = self.critic_target(next_states)

      #期待値計算 元の型は[62,2] → sum(dim1) → keepdim=Flase→[62] , keepdim=True→[62,1]
      next_V = (next_action_prob * ( torch.min(next_q1_value, next_q2_value)  - self.alpha * next_action_log_prob)).sum(dim=1, keepdim=True)


    next_value = rewards + (1.0 - done) * self.gamma * next_V

    return next_value


  def calc_critic_loss(self, states, actions, rewards, next_states, done):

    current_q1_value , current_q2_value = self.calc_q_value(states, actions)

    # target_q_value
    target_q_value = self.calc_target_q_value(states, actions, rewards, next_states, done)

    q1_loss = F.mse_loss(current_q1_value, target_q_value)
    q2_loss = F.mse_loss(current_q2_value, target_q_value)


    return q1_loss, q2_loss


  def calc_policy_loss(self, states, actions, rewards, next_states, done):

    _, action_prob, action_log_prob = self.actor(states)

    with torch.no_grad():
      q1_value, q2_value = self.critic(states)

    #エントロピーの計算
    entropies = -torch.sum(action_prob * action_log_prob, dim=1, keepdim=True)

    #Q関数の計算
    q_value =  torch.sum(torch.min(q1_value, q2_value) * action_prob, dim=1, keepdim=True)

    # todo self.alpha の前の- + どっち
    policy_loss = -1 * (q_value + self.alpha * entropies).mean()

    return policy_loss, entropies.detach()


  def calc_entropy_loss(self, entropy):

    # todo entropyが0だと log_alpha → 大 これが続く
    entropy_loss = - torch.mean(self.log_alpha * (self.target_entropy - entropy))

    return entropy_loss


  def action_step(self, state):
    with torch.no_grad():
      action, _ , _ = self.actor(state)

      return action.item()


doctest.testmod()
