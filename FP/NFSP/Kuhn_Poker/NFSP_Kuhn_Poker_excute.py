
# _________________________________ Library _________________________________

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
import time
import doctest
import copy
import wandb
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from collections import defaultdict
from tqdm import tqdm
from collections import deque

import NFSP_Kuhn_Poker_trainer
import NFSP_Kuhn_Poker_supervised_learning
import NFSP_Kuhn_Poker_reinforcement_learning_DQN
import NFSP_Kuhn_Poker_reinforcement_learning_SAC
import NFSP_Kuhn_Poker_generate_data


# _________________________________ config _________________________________

config = dict(
  random_seed = [42, 1000, 10000][0],
  iterations = 10**4,
  num_players = 2,
  wandb_save = [True, False][1],

  #rl
  rl_algo = ["dfs", "dqn", "ddqn", "sac", "sql"][4]

)

if  config["rl_algo"] in ["dqn" , "dfs" , "ddqn", "sql"] :
  config_plus = dict(
  eta = 0.1,
  memory_size_rl = 2*(10**4),
  memory_size_sl = 2*(10**5),
  step_per_learning_update = 128,

  #sl
  sl_hidden_units_num= 64,
  sl_lr = 0.001,
  sl_epochs = 2,
  sl_sampling_num = 64,
  sl_loss_function = [nn.BCEWithLogitsLoss()][0],
  sl_algo = ["cnt", "mlp"][1],

  #dqn 用
  rl_lr = 0.1,
  rl_hidden_units_num= 64,
  rl_epochs = 2,
  rl_sampling_num = 64,
  rl_gamma = 1.0,
  rl_tau = 0.1,
  rl_update_frequency = 30,
  rl_loss_function = [F.mse_loss, nn.HuberLoss()][0],
  # device
  #device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
  device = torch.device('cpu'),
  #sql
  rl_alpha = 0.005,
  )

  config.update(config_plus)

elif config["rl_algo"] in ["sac"] :
  config_plus = dict(
  eta = 0.1,
  memory_size_rl = 2*(10**4),
  memory_size_sl = 2*(10**5),
  step_per_learning_update = 128,

  sl_hidden_units_num= 64,
  sl_lr = 0.001,
  sl_epochs = 2,
  sl_sampling_num = 64,
  sl_loss_function = [nn.BCEWithLogitsLoss()][0],
  sl_algo = ["cnt", "mlp"][1],

  rl_hidden_units_num= 64,
  rl_epochs = 2,
  rl_sampling_num = 64,
  rl_gamma = 1.0,
  rl_tau = 0.1,
  rl_update_frequency = 30,
  rl_entropy_lr = 0.0001,
  rl_policy_lr =  0.0001,
  rl_critic_lr =  0.1,
  rl_loss_function = [F.mse_loss, nn.HuberLoss()][0],
  rl_value_of_alpha_change = False,
  rl_alpha = 0.1,
  # device
  #device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
  device = torch.device('cpu')
  )

  config.update(config_plus)


if config["wandb_save"]:
  #wandb.init(project="Kuhn_Poker_n_players", name="{}_players_NFSP".format(config["num_players"]))
  if config["rl_algo"] == "sac":
    wandb.init(project="Kuhn_Poker_{}players_SAC".format(config["num_players"]), name="{}_{}_NFSP".format(config["rl_algo"], config["sl_algo"]))
  else:
    wandb.init(project="Kuhn_Poker_{}players".format(config["num_players"]), name="{}_{}_NFSP".format(config["rl_algo"], config["sl_algo"]))
  wandb.config.update(config)
  wandb.define_metric("exploitability", summary="last")
  wandb.define_metric("avg_utility", summary="last")



# _________________________________ train _________________________________

kuhn_trainer = NFSP_Kuhn_Poker_trainer.KuhnTrainer(
  random_seed = config["random_seed"],
  train_iterations = config["iterations"],
  num_players= config["num_players"],
  wandb_save = config["wandb_save"],
  step_per_learning_update = config["step_per_learning_update"],
  )


if config["rl_algo"] in ["dqn" , "dfs" , "ddqn", "sql"]:
  kuhn_RL = NFSP_Kuhn_Poker_reinforcement_learning_DQN.ReinforcementLearning(
    random_seed = config["random_seed"],
    train_iterations = config["iterations"],
    num_players= config["num_players"],
    hidden_units_num = config["rl_hidden_units_num"],
    lr = config["rl_lr"],
    epochs = config["rl_epochs"],
    sampling_num = config["rl_sampling_num"],
    gamma = config["rl_gamma"],
    tau = config["rl_tau"],
    update_frequency = config["rl_update_frequency"],
    loss_function = config["rl_loss_function"],
    kuhn_trainer_for_rl = kuhn_trainer,
    device = config["device"],
    alpha = config["rl_alpha"]
    )


elif config["rl_algo"] == "sac":
  kuhn_RL = NFSP_Kuhn_Poker_reinforcement_learning_SAC.ReinforcementLearning(
    random_seed = config["random_seed"],
    train_iterations = config["iterations"],
    num_players= config["num_players"],
    hidden_units_num = config["rl_hidden_units_num"],
    entropy_lr = config["rl_entropy_lr"],
    policy_lr = config["rl_policy_lr"],
    critic_lr = config["rl_critic_lr"],
    epochs = config["rl_epochs"],
    sampling_num = config["rl_sampling_num"],
    gamma = config["rl_gamma"],
    tau = config["rl_tau"],
    update_frequency = config["rl_update_frequency"],
    loss_function = config["rl_loss_function"],
    kuhn_trainer_for_rl = kuhn_trainer,
    device = config["device"],
    value_of_alpha_change = config["rl_value_of_alpha_change"],
    alpha= config["rl_alpha"]
    )


kuhn_SL = NFSP_Kuhn_Poker_supervised_learning.SupervisedLearning(
  random_seed = config["random_seed"],
  train_iterations = config["iterations"],
  num_players= config["num_players"],
  hidden_units_num= config["sl_hidden_units_num"],
  lr = config["sl_lr"],
  epochs = config["sl_epochs"],
  sampling_num = config["sl_sampling_num"],
  loss_function = config["sl_loss_function"],
  kuhn_trainer_for_sl = kuhn_trainer,
  device = config["device"]
  )


kuhn_GD = NFSP_Kuhn_Poker_generate_data.GenerateData(
  random_seed = config["random_seed"],
  num_players= config["num_players"],
  kuhn_trainer_for_gd= kuhn_trainer
  )


kuhn_trainer.train(
  eta = config["eta"],
  memory_size_rl = config["memory_size_rl"],
  memory_size_sl = config["memory_size_sl"],
  rl_algo = config["rl_algo"],
  sl_algo = config["sl_algo"],
  rl_module= kuhn_RL,
  sl_module= kuhn_SL,
  gd_module= kuhn_GD
  )


# _________________________________ result _________________________________

if not config["wandb_save"]:
  print("avg_utility", list(kuhn_trainer.avg_utility_list.items())[-1])
  print("final_exploitability", list(kuhn_trainer.exploitability_list.items())[-1])


if config["rl_algo"] in ["dqn" , "dfs" , "ddqn", "sql"]:
  result_dict_avg = {}
  for key, value in sorted(kuhn_trainer.avg_strategy.items()):
    result_dict_avg[key] = value
    eval_s_bit = torch.Tensor(kuhn_trainer.make_state_bit(key))
    Q_value = kuhn_trainer.RL.deep_q_network.forward(eval_s_bit)
    print(key, Q_value)
  df = pd.DataFrame(result_dict_avg.values(), index=result_dict_avg.keys(), columns=['Pass_avg', "Bet_avg"])
  df.index.name = "Node"

  result_dict_br = {}
  for key, value in sorted(kuhn_trainer.epsilon_greedy_q_learning_strategy.items()):
    result_dict_br[key] = value
  df1 = pd.DataFrame(result_dict_br.values(), index=result_dict_br.keys(), columns=['Pass_br', "Bet_br"])
  df1.index.name = "Node"

  df2 = pd.concat([df, df1], axis=1)

elif config["rl_algo"] == "sac":
  result_dict_avg = {}
  for key, value in sorted(kuhn_trainer.avg_strategy.items()):
    result_dict_avg[key] = value
  df = pd.DataFrame(result_dict_avg.values(), index=result_dict_avg.keys(), columns=['Pass_avg', "Bet_avg"])
  df.index.name = "Node"

  result_dict_br = {}
  for key, value in sorted(kuhn_trainer.epsilon_greedy_q_learning_strategy.items()):
    eval_s_bit = torch.Tensor(kuhn_trainer.make_state_bit(key))
    action_prob = kuhn_trainer.RL.action_step_prob(eval_s_bit)
    current_q1_values, current_q2_values = kuhn_trainer.RL.critic(eval_s_bit)
    print(key, current_q1_values, current_q2_values)

    result_dict_br[key] = action_prob
  df1 = pd.DataFrame(result_dict_br.values(), index=result_dict_br.keys(), columns=['Pass_br', "Bet_br"])
  df1.index.name = "Node"

  df2 = pd.concat([df, df1], axis=1)


if config["wandb_save"]:
  tbl = wandb.Table(data=df2)
  tbl.add_column("Node", [i for i in df2.index])
  wandb.log({"table:":tbl})
  wandb.save()
else:
  print(df2)

#追加 matplotlibで図を書くため

#df = pd.DataFrame(kuhn_trainer.database_for_plot)
#df = df.set_index('iteration')
#df.to_csv('../../../Make_png/output/database_for_plot_{}_{}.csv'.format(config["num_players"],config["random_seed"]))


doctest.testmod()
