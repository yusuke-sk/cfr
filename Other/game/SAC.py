# -- library --
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#https://pytorch.org/docs/stable/distributions.html
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt


class SAC:
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.episode = 0
        self.iteration = 1000
        self.number_parallel = 1
        self.hidden_unit_num = 256
        self.batch_size = 256
        self.gamma = 0.99
        self.lr = 0.001
        self.N = 20
        self.history_score = []
        self.hisotry_avg_score = []
        self.update_frequency = 10


        self.agent = Agent(
            state_num= self.env.observation_space.shape[0],
            action_num= self.env.action_space.n,
            hidden_unit_num= self.hidden_unit_num,
            batch_size= self.batch_size,
            gamma= self.gamma,
            lr = self.lr,
            update_frequency = self.update_frequency,
            )




    def train(self):
        n_step = 0
        learn_iters = 0

        for i in tqdm(range(self.iteration)):

            # collect trajectory
            for _ in range(self.number_parallel):
                observation = self.env.reset()
                score = 0
                done = False

                # start 1 episode
                while not done:

                    action  = self.agent.choose_action(observation)
                    next_observation, reward, done, info = self.env.step(action)

                    score += reward
                    n_step += 1

                    self.agent.save_memory(observation, action, reward, next_observation, done)

                    observation = next_observation

                self.history_score.append(score)
                self.avg_score = np.mean(self.history_score[-100:])
                self.hisotry_avg_score.append(self.avg_score)


            # 学習して、良い方策へ
            self.agent.learn()
            learn_iters += 1


            if i % 10 == 0:
                if i == 0:
                    print("i, avg_score, n_step, learn_iters")
                print(i, self.avg_score , n_step, learn_iters)


class Agent:
    def __init__(self, state_num, action_num, hidden_unit_num, batch_size, gamma, lr, update_frequency):
        self.state_num = state_num
        self.action_num = action_num
        self.hidden_unit_num = hidden_unit_num
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_frequency = update_frequency

        self.update_count = 0

        self.memory = SACMemory(self.batch_size)


        self.critic = Double_Q_network(self.state_num, self.action_num, self.hidden_unit_num)
        self.critic_target = Double_Q_network(self.state_num, self.action_num, self.hidden_unit_num)

        self.actor = ActorNetwork(self.state_num, self.action_num, self.hidden_unit_num)

        self.critic_1_optim = optim.Adam(self.critic.Q1.parameters(), lr = self.lr)
        self.critic_2_optim = optim.Adam(self.critic.Q2.parameters(), lr = self.lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.lr)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.target_entropy_ratio = 0.98

        self.target_entropy = - np.log(1.0/self.action_num) * self.target_entropy_ratio

        #optimize log_alpha
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr = lr)


    def choose_action(self, observation):
        state = torch.tensor(np.array(observation),dtype=torch.float)

        with torch.no_grad():
            action, _ , _ = self.actor(state)

        return action.item()


    def save_memory(self, state, action, reward, next_states, done):
        self.memory.store(state, action, reward, next_states, done)


    def learn(self):

        state_arr, action_arr, reward_arr, next_states, done_arr = self.memory.get_batch()


        train_states = torch.tensor(state_arr).float().reshape(-1,self.state_num)
        train_actions = torch.tensor(action_arr).float().reshape(-1,1)
        train_rewards = torch.tensor(reward_arr).float().reshape(-1,1)
        train_next_states = torch.tensor(next_states).float().reshape(-1,self.state_num)
        train_done = torch.tensor(done_arr).float().reshape(-1,1)

        #Q関数の更新 J(θ)
        q1_loss, q2_loss = self.calc_critic_loss(train_states, train_actions, train_rewards, train_next_states, train_done)


        self.critic_1_optim.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        q2_loss.backward(retain_graph=True)
        self.critic_2_optim.step()


        #方策の更新
        policy_loss = self.calc_policy_loss(train_states, train_actions, train_rewards, train_next_states, train_done)


        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        #print(policy_loss)


        #エントロピー係数の更新
        entropy_loss = self.calc_entropy_loss(train_states)

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

        return policy_loss


    def calc_entropy_loss(self, states):
        _, action_prob, action_log_prob = self.actor(states)

        entropy_loss = - torch.mean(self.log_alpha * (self.target_entropy + action_log_prob))

        return entropy_loss


class SACMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def store(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def delete(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []


    def get_size(self):
        return len(self.states)


    def get_batch(self):
        #dataをbatch_sizeに切り分ける そのindexを作る
        size = self.get_size()
        batch_start_index = np.arange(0, size, self.batch_size)
        state_index = np.arange(size, dtype=np.int64)
        np.random.shuffle(state_index)
        batch_index = [state_index[i:i+self.batch_size] for i in batch_start_index]

        batch = batch_index[0]

        #npだと array[batch] で 取得可能
        return np.array(self.states)[batch], np.array(self.actions)[batch] ,np.array(self.rewards)[batch], \
            np.array(self.next_states)[batch], np.array(self.dones)[batch]


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


class Double_Q_network(nn.Module):
    def __init__(self, state_num, action_num, hidden_units_num):
        super(Double_Q_network, self).__init__()
        self.Q1 = CriticNetwork(state_num, action_num, hidden_units_num)
        self.Q2 = CriticNetwork(state_num, action_num, hidden_units_num)

    def forward(self, x):
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2




SAC_trainer = SAC()
SAC_trainer.train()
#print(SAC_trainer.history_score)
#print(SAC_trainer.hisotry_avg_score)

plt.plot(range(len(SAC_trainer.hisotry_avg_score)), SAC_trainer.history_score)
plt.xlabel("iteration")
plt.ylabel("avg score")
plt.show()
