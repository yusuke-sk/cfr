
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
from multiprocessing import Process, Queue




# _________________________________ Train class _________________________________
class LeducTrainer:
  def __init__(self,random_seed=42, train_iterations=10, num_players=2, wandb_save=False, step_per_learning_update=128,batch_episode_num=28, save_matplotlib = False):
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.NUM_ACTIONS = 3
    self.ACTION_DICT = {0:"f", 1:"c", 2:"r"}
    self.ACTION_DICT_verse = {"f":0, "c":1, "r":2}
    self.STATE_BIT_LEN = 2* ( (self.NUM_PLAYERS + 1) + 3*(self.NUM_PLAYERS *3 - 2) ) - 3
    self.cards = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    self.wandb_save = wandb_save
    self.avg_strategy = {}
    self.node_possible_action = {}
    self.history_action_player_dict = {}
    self.random_seed = random_seed
    self.card_rank = self.make_rank()
    self.card_order  = self.make_card_order()
    self.card_set = set(self.card_distribution())

    self.random_seed_fix(self.random_seed)
    self.step_per_learning_update = step_per_learning_update
    self.batch_episode_num = batch_episode_num
    self.save_matplotlib = save_matplotlib

    #可搾取量の合計計算時間
    self.exploitability_time = 0

# _________________________________ Train main method _________________________________
  def train(self, eta, memory_size_rl, memory_size_sl, rl_algo, sl_algo, rl_module, sl_module, gd_module):
    self.exploitability_list = {}
    self.avg_utility_list = {}
    self.eta = eta
    self.rl_algo = rl_algo
    self.sl_algo = sl_algo
    self.memory_size_sl = memory_size_sl
    self.memory_size_rl = memory_size_rl
    self.memory_count_for_sl = 0

    #追加 matplotlibで記録を集計するため
    if self.save_matplotlib:
      self.batch_episode_name = "parallel_batch_episode_time_for_{}_{}".format(self.NUM_PLAYERS, self.random_seed)
      self.ex_name = "exploitability_for_{}_{}".format(self.random_seed, self.rl_algo)
      self.database_for_plot = {"iteration":[] , self.ex_name:[]}
      self.database_for_time = {"iteration":[] , self.batch_episode_name:[]}


    self.M_SL = []
    self.M_RL = deque([], maxlen= self.memory_size_rl)


    self.infoSets_dict_player = [[] for _ in range(self.NUM_PLAYERS)]
    self.infoSets_dict = {}
    self.infoset_action_player_dict = {}

    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)

    self.epsilon_greedy_q_learning_strategy = copy.deepcopy(self.avg_strategy)


    self.RL = rl_module
    self.SL = sl_module
    self.GD = gd_module
    self.GD.infoset_action_player_dict = self.infoset_action_player_dict
    self.SL.infoset_action_player_dict = self.infoset_action_player_dict
    self.RL.infoset_action_player_dict = self.infoset_action_player_dict


    # n_count
    self.N_count = copy.deepcopy(self.avg_strategy)
    for node, cn in self.N_count.items():
      self.N_count[node] = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
      for key_i, valye_i in enumerate(self.avg_strategy[node]):
        if valye_i != 0:
          self.N_count[node][key_i] = 1.0


    #プロセス立ち上げ
    q_in1, q_out_sl1, q_out_rl1, q_finish1 = Queue(), Queue(), Queue(), Queue()
    q_in2, q_out_sl2, q_out_rl2, q_finish2 = Queue(), Queue(), Queue(), Queue()
    q_in3, q_out_sl3, q_out_rl3, q_finish3 = Queue(), Queue(), Queue(), Queue()
    q_in4, q_out_sl4, q_out_rl4, q_finish4 = Queue(), Queue(), Queue(), Queue()

    process1 = Process(target=self.wait_and_make_episode_loop, args=(q_in1, q_out_sl1, q_out_rl1, q_finish1, self.SL, self.RL))
    process2 = Process(target=self.wait_and_make_episode_loop, args=(q_in2, q_out_sl2, q_out_rl2, q_finish2, self.SL, self.RL))
    process3 = Process(target=self.wait_and_make_episode_loop, args=(q_in3, q_out_sl3, q_out_rl3, q_finish3, self.SL, self.RL))
    process4 = Process(target=self.wait_and_make_episode_loop, args=(q_in4, q_out_sl4, q_out_rl4, q_finish4, self.SL, self.RL))


    process1.start()
    process2.start()
    process3.start()
    process4.start()



    self.calculate_evalation_values(0)
    for iteration_t in tqdm(range(1, int(self.train_iterations//self.batch_episode_num)+1)):


      #1 iteraion = 1episode を守る
      iteration_t *= self.batch_episode_num

      #エピソード作成
      start_time = time.time()
      q_in1.put(self.batch_episode_num//4)
      q_in2.put(self.batch_episode_num//4)
      q_in3.put(self.batch_episode_num//4)
      q_in4.put(self.batch_episode_num//4)

      #エピソード作成し終わるまで待機
      q_finish1.get()
      q_finish2.get()
      q_finish3.get()
      q_finish4.get()


      #queueに溜まってるデータがあれば、取り出す
      while not q_out_sl1.empty():
        for data_SL in q_out_sl1.get():
          self.reservior_add(self.M_SL,data_SL)
      while not q_out_rl1.empty():
        for data_RL in q_out_rl1.get():
          self.M_RL.append(data_RL)

      while not q_out_sl2.empty():
        for data_SL in q_out_sl2.get():
          self.reservior_add(self.M_SL,data_SL)
      while not q_out_rl2.empty():
        for data_RL in q_out_rl2.get():
          self.M_RL.append(data_RL)

      while not q_out_sl3.empty():
        for data_SL in q_out_sl3.get():
          self.reservior_add(self.M_SL,data_SL)
      while not q_out_rl3.empty():
        for data_RL in q_out_rl3.get():
          self.M_RL.append(data_RL)

      while not q_out_sl4.empty():
        for data_SL in q_out_sl4.get():
          self.reservior_add(self.M_SL,data_SL)
      while not q_out_rl4.empty():
        for data_RL in q_out_rl4.get():
          self.M_RL.append(data_RL)


      if self.save_matplotlib :
        end_time = time.time()
        make_episode_time = end_time - start_time
        #print(make_episode_time, time_1-start_time, time_2-time_1, end_time-time_2)
        self.database_for_time["iteration"].append(iteration_t)
        self.database_for_time[self.batch_episode_name].append(make_episode_time)


      if self.wandb_save:
        end_time = time.time()
        make_episode_time = end_time - start_time
        wandb.log({'iteration': iteration_t,'batch_episode_time': make_episode_time})


      #学習
      self.SL_and_RL_learn(iteration_t)


      #batch_sizeに比例した値でないとif文クリアせず、従来とあわなくなるので調整
      exploitability_check_t = [int(j)//self.batch_episode_num * self.batch_episode_num for j in np.logspace(0, len(str(self.train_iterations)), (len(str(self.train_iterations)))*10 , endpoint=False)]

      if iteration_t in exploitability_check_t :
        self.calculate_evalation_values(iteration_t)

    #process終了
    q_in1.put(-1)
    q_in2.put(-1)
    q_in3.put(-1)
    q_in4.put(-1)

    process1.join()
    process2.join()
    process3.join()
    process4.join()



  def calculate_evalation_values(self, iteration_t):
      self.optimal_gap, self.dfs_exploitability , self.current_br_exploitability = self.get_exploitability_and_optimal_gap()
      self.exploitability_list[iteration_t] = self.dfs_exploitability
      self.avg_utility_list[iteration_t] = self.eval_vanilla_CFR("", 0, 0, [1.0 for _ in range(self.NUM_PLAYERS)])

      if self.wandb_save:
              wandb.log({'iteration': iteration_t, 'exploitability': self.exploitability_list[iteration_t], 'avg_utility': self.avg_utility_list[iteration_t]})


      #追加 matplotlibで図を書くため
      if self.save_matplotlib:
        self.database_for_plot["iteration"].append(iteration_t)
        self.database_for_plot[self.ex_name].append(self.exploitability_list[iteration_t])


  def get_exploitability_and_optimal_gap(self):
      #最適反応戦略と平均戦略のテーブルを更新: change
      self.RL.update_strategy_for_table(self.epsilon_greedy_q_learning_strategy)
      self.SL.update_strategy_for_table(self.avg_strategy)


      optimality_gap = 0
      self.infoSets_dict = {}
      for target_player in range(self.NUM_PLAYERS):
        self.create_infoSets("", target_player, 1.0)
      self.best_response_strategy_dfs = {}
      for best_response_player_i in range(self.NUM_PLAYERS):
        self.calc_best_response_value(self.best_response_strategy_dfs, best_response_player_i, "", 1)

      dfs_exploitability = 0
      current_br_exploitability = 0
      for player_i in range(self.NUM_PLAYERS):
          dfs_exploitability +=  self.GD.calculate_optimal_gap_best_response_strategy(self.best_response_strategy_dfs, self.avg_strategy, player_i)
          current_br_exploitability += self.GD.calculate_optimal_gap_best_response_strategy(self.epsilon_greedy_q_learning_strategy, self.avg_strategy, player_i)

      optimality_gap = 1/self.NUM_PLAYERS * (dfs_exploitability - current_br_exploitability)
      assert optimality_gap >= 0
      return optimality_gap , dfs_exploitability, current_br_exploitability


  def wait_and_make_episode_loop(self, q_in, q_out_sl, q_out_rl, q_finish, sl, rl):
        """
        合図が来たら、make_episodesを実行し、結果をqueueに渡す。

        Parameters
        ----------
        q_in : queue
            合図を送るqueue, 一つのqueueの中身は、[episode_num, module]
        q_out : queue
            得られたデータを送るqueue
        """
        while True:
          episode_num  = q_in.get()

          #プロセス終了の場合
          if episode_num < 0:
              break

          sl_memory, rl_memory = self.make_episodes(episode_num, sl, rl)
          q_out_sl.put(sl_memory)
          q_out_rl.put(rl_memory)

          #エピソード作成終了の合図
          q_finish.put("finish")



  def SL_and_RL_learn(self, iteration_t):
    start_time = time.time()
    if self.sl_algo == "mlp":
      self.SL.SL_learn(self.M_SL, self.avg_strategy, iteration_t)
    elif self.sl_algo == "cnt":
      self.SL.SL_train_AVG(self.M_SL, self.avg_strategy, self.N_count)
      self.M_SL = []

    #強化学習
    if self.rl_algo != "dfs":
      self.RL.rl_algo = self.rl_algo
      self.RL.RL_learn(self.M_RL, self.epsilon_greedy_q_learning_strategy, iteration_t)

    elif self.rl_algo == "dfs":
      self.infoSets_dict = {}
      for target_player in range(self.NUM_PLAYERS):
        self.create_infoSets("", target_player, 1.0)
      self.epsilon_greedy_q_learning_strategy = {}
      for best_response_player_i in range(self.NUM_PLAYERS):
        self.calc_best_response_value(self.epsilon_greedy_q_learning_strategy, best_response_player_i, "", 1)
    end_time = time.time()



  def make_episodes(self, episode_num, sl, rl):
    list_SL = []
    list_RL = []

    for ii in range(episode_num):

      #data 収集part
      #0 → epsilon_greedy_q_strategy, 1 → avg_strategy
      self.sigma_strategy_bit = [-1 for _ in range(self.NUM_PLAYERS)]
      for player_i in range(self.NUM_PLAYERS):
        if np.random.uniform() < self.eta:
          self.sigma_strategy_bit[player_i] = 0
        else:
          self.sigma_strategy_bit[player_i] = 1

      self.cards = self.card_distribution()
      random.shuffle(self.cards)
      history = "".join(self.cards[:self.NUM_PLAYERS])
      self.player_sars_list = [{"s":None, "a":None, "r":None, "s_prime":None} for _ in range(self.NUM_PLAYERS)]

      self.train_one_episode(history, list_SL, list_RL, sl, rl)


    return list_SL, list_RL




# _________________________________ Train second main method _________________________________
  def train_one_episode(self, history, list_SL, list_RL, sl, rl):
  # one episode
    while  not self.whether_terminal_states(history):
      if self.whether_chance_node(history):
        history += self.cards[self.NUM_PLAYERS]

      else:

        player = self.action_player(history)


        s = history[player] + history[self.NUM_PLAYERS:]


        if self.player_sars_list[player]["s"] is not None:
          self.player_sars_list[player]["s_prime"] = s

          sars_list = self.make_sars_list(self.player_sars_list[player])
          list_RL.append(sars_list)

          self.player_sars_list[player] = {"s":None, "a":None, "r":None, "s_prime":None}


        if self.sigma_strategy_bit[player] == 0:
          sampling_action = np.random.choice(list(range(self.NUM_ACTIONS)), p=self.RL.action_step(s))

        elif self.sigma_strategy_bit[player] == 1:
          sampling_action = np.random.choice(list(range(self.NUM_ACTIONS)), p=self.SL.action_step(s))


        a = self.ACTION_DICT[sampling_action]
        history  += a
        r = 0

        self.player_sars_list[player]["s"] = s
        self.player_sars_list[player]["a"] = a
        self.player_sars_list[player]["r"] = r


        if self.sigma_strategy_bit[player] == 0:
          if self.sl_algo == "mlp":
            sa_bit = self.from_episode_to_bit([(s, a)])
            list_SL.append(sa_bit)
          else:
            list_SL.append(sa_bit)

    # terminal state
    if self.whether_terminal_states(history):
      for target_player_i in range(self.NUM_PLAYERS):
        r = self.Return_payoff_for_terminal_states(history, target_player_i)
        self.player_sars_list[target_player_i]["r"] = r

        sars_list = self.make_sars_list(self.player_sars_list[target_player_i])
        list_RL.append(sars_list)

        self.player_sars_list[target_player_i] = {"s":None, "a":None, "r":None, "s_prime":None}




  def make_sars_list(self, sars_memory):
    #s_bit, a, r, s_prime, s_prime_bit, done
    sars_list = []
    for idx, x in enumerate(sars_memory.values()):
      if idx == 0:
        sars_list.append(self.make_state_bit(x))
      elif idx == 1:
        sars_list.append(self.make_action_bit(x))
      elif idx == 2:
        sars_list.append(x)
      elif idx == 3:
        sars_list.append(x)
        sars_list.append(self.make_state_bit(x))
        if x == None:
          sars_list.append(1)
        else:
          sars_list.append(0)
    return sars_list



  def random_seed_fix(self, random_seed):
      random.seed(random_seed)
      np.random.seed(random_seed)



  def reservior_add(self, memory, data):
    if len(memory) < self.memory_size_sl:
        memory.append(data)
    else:
        r = random.randint(0, self.memory_count_for_sl)
        if r < self.memory_size_sl:
            memory[r] = data

    self.memory_count_for_sl += 1



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



  def from_episode_to_bit(self, one_s_a_set):
    """return list
    >>> LeducTrainer().from_episode_to_bit([('Q', 'r')])
    ([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2])
    """
    for X, y in one_s_a_set:
      y_bit = self.make_action_bit(y)
      X_bit = self.make_state_bit(X)

    return (X_bit, y_bit)



  def make_action_bit(self, y):
    """return array
    >>> LeducTrainer().make_action_bit("f")
    [0]
    """

    if y == "f":
      y_bit = [0]
    elif y == "c":
      y_bit = [1]
    elif y == "r":
      y_bit = [2]
    return y_bit



  def make_state_bit(self, X):
    """return array
    >>> LeducTrainer().make_state_bit("Q")
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> LeducTrainer().make_state_bit("KrrcKrr")
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    >>> LeducTrainer().make_state_bit("QrrccJrrc")
    [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
    """
    X_bit = [0 for _ in range(self.STATE_BIT_LEN)]

    if X == None:
      return X_bit

    X_bit[self.card_order[X[0]]] = 1
    com_bit = -1

    idx_now = 0
    for Xi in (X[1:]):
      if Xi not in self.cards:
        if com_bit == -1:
          X_bit[(self.NUM_PLAYERS+1) + 3*idx_now + self.ACTION_DICT_verse[Xi]] = 1
          idx_now += 1
        else:
          X_bit[ - ((self.NUM_PLAYERS+1) + 3*idx_now + self.ACTION_DICT_verse[Xi] + 1)] = 1
          idx_now += 1

      else:
        com_idx = self.card_order[Xi] + 1
        X_bit[-com_idx] = 1
        com_bit = 1
        idx_now = 0

    return X_bit


  def make_card_order(self):
    """return dict
    >>> LeducTrainer(num_players=2).make_card_order() == {'J':0, 'Q':1, 'K':2}
    True
    >>> LeducTrainer(num_players=3).make_card_order() == {'T':0, 'J':1, 'Q':2, 'K':3}
    True
    """
    card_order = {}
    for i in range(self.NUM_PLAYERS+1):
      card_order[self.cards[11-self.NUM_PLAYERS+i]] =  i

    return card_order


doctest.testmod()
