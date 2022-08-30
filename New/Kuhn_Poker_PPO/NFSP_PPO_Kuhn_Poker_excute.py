
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

import NFSP_PPO_Kuhn_Poker_trainer
import NFSP_PPO_Kuhn_Poker_supervised_learning
import NFSP_PPO_Kuhn_Poker_reinforcement_learning
import NFSP_PPO_Kuhn_Poker_generate_data


# _________________________________ config _________________________________

config = dict(
  random_seed = 42,
  iterations = 10000,
  num_players = 2,
  num_parallel = 100,
  wandb_save = [True, False][0],


  #sl
  sl_hidden_units_num= 64,
  sl_lr = 0.001,
  sl_epochs = 2,
  sl_sampling_num = 128,
  sl_batch_size = 128,
  sl_loss_function = [nn.BCEWithLogitsLoss()][0],
  memory_size_sl = 2*(10**5),


  #rl
  rl_hidden_units_num= 64,
  rl_epochs = 2,
  rl_batch_size = 128,
  sl_algo = ["cnt", "mlp"][0],
  rl_algo = ["ppo"][0],
  rl_eps_clip = 0.2,
  rl_policy_lr = 0.0003,
  rl_value_lr = 0.001,
  rl_gamma = 0.99,
  rl_lam = 0.95,
  rl_policy_clip = 0.2,
  rl_entropy = 0.01,

  # device
  #device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
  device = torch.device('cpu')

)



if config["wandb_save"]:
  wandb.init(project="Kuhn_Poker_{}players_PPO".format(config["num_players"]), name="{}_{}_NFSP".format(config["rl_algo"], config["sl_algo"]))
  wandb.config.update(config)
  wandb.define_metric("exploitability", summary="last")
  wandb.define_metric("avg_utility", summary="last")


# _________________________________ train _________________________________

kuhn_trainer = NFSP_PPO_Kuhn_Poker_trainer.KuhnTrainer(
  random_seed = config["random_seed"],
  train_iterations = config["iterations"],
  num_parallel= config["num_parallel"],
  num_players= config["num_players"],
  wandb_save = config["wandb_save"]
  )

kuhn_RL = NFSP_PPO_Kuhn_Poker_reinforcement_learning.PPO(
  num_players= config["num_players"],
  hidden_units_num = config["rl_hidden_units_num"],
  epochs = config["rl_epochs"],
  eps_clip = config["rl_eps_clip"],
  policy_lr = config["rl_policy_lr"],
  value_lr = config["rl_value_lr"],
  gamma = config["rl_gamma"],
  lam = config["rl_lam"],
  policy_clip = config["rl_policy_clip"],
  wandb_save = config["wandb_save"],
  entropy_coef = config["rl_entropy"]
  )


kuhn_SL = NFSP_PPO_Kuhn_Poker_supervised_learning.SupervisedLearning(
  random_seed = config["random_seed"],
  train_iterations = config["iterations"],
  num_players= config["num_players"],
  hidden_units_num= config["sl_hidden_units_num"],
  lr = config["sl_lr"],
  epochs = config["sl_epochs"],
  loss_function = config["sl_loss_function"],
  kuhn_trainer_for_sl = kuhn_trainer,
  device = config["device"]
  )

kuhn_SL_memory = NFSP_PPO_Kuhn_Poker_supervised_learning.PPO_SL_memory(
  batch_size= config["sl_batch_size"],
  memory_size = config["memory_size_sl"]
)

kuhn_RL_memory = NFSP_PPO_Kuhn_Poker_reinforcement_learning.PPO_RL_memory(
  batch_size= config["rl_batch_size"]
)


kuhn_GD = NFSP_PPO_Kuhn_Poker_generate_data.GenerateData(
  random_seed = config["random_seed"],
  num_players= config["num_players"],
  kuhn_trainer_for_gd= kuhn_trainer
  )



kuhn_trainer.train(
  rl_algo = config["rl_algo"],
  sl_algo = config["sl_algo"],
  rl_module= kuhn_RL,
  sl_module= kuhn_SL,
  gd_module= kuhn_GD,
  SL_memory = kuhn_SL_memory,
  RL_memory = kuhn_RL_memory
  )



# _________________________________ result _________________________________

if not config["wandb_save"]:
  print("avg_utility", list(kuhn_trainer.avg_utility_list.items())[-1])
  print("final_exploitability", list(kuhn_trainer.exploitability_list.items())[-1])


result_dict_avg = {}
for key, value in sorted(kuhn_trainer.avg_strategy.items()):
  result_dict_avg[key] = value
df = pd.DataFrame(result_dict_avg.values(), index=result_dict_avg.keys(), columns=['Pass_avg', "Bet_avg"])
df.index.name = "Node"

result_dict_br = {}
for key, value in sorted(kuhn_trainer.best_response_strategy.items()):
  result_dict_br[key] = value
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



doctest.testmod()
