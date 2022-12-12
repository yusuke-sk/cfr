
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

from collections import defaultdict
from tqdm import tqdm
from collections import deque

import NFSP_Leduc_Poker_trainer
import NFSP_Leduc_Poker_supervised_learning
import NFSP_Leduc_Poker_reinforcement_learning
import NFSP_Leduc_Poker_generate_data


# _________________________________ config _________________________________
start_time =time.time()

config = dict(
  random_seed = [42, 1000, 10000][0],
  iterations = 10**5,
  num_players = 2,
  wandb_save = [True, False][0],
  save_matplotlib = [True, False][1],

  #train
  eta = 0.1,
  memory_size_rl = 2*(10**5),
  memory_size_sl = 2*(10**6),

  #sl
  sl_hidden_units_num= 64,
  sl_lr = 0.001,
  sl_epochs = 2,
  sl_sampling_num =128,

  #rl
  rl_hidden_units_num= 64,
  rl_lr = 0.1,
  rl_epochs = 2,
  rl_sampling_num = 128,
  rl_gamma = 1.0,
  rl_tau = 0.1,
  rl_update_frequency = 300,
  sl_algo = ["cnt", "mlp"][1],
  rl_algo = ["dfs", "dqn", "ddqn", "sql"][1],
  #sql
  rl_alpha = 5e+1,
  rl_strategy = ["ε-greedy", "proportional_Q"][0],
  alpha_discrease = [True, False][0],
)



if config["wandb_save"]:
  if config["rl_algo"] == "sql":
    #wandb.init(project="Leduc_Poker_{}players_SQL".format(config["num_players"]), name="{}_{}_{}_NFSP".format(config["rl_algo"], config["rl_alpha"], config["alpha_discrease"]))
    wandb.init(project="Leduc_Poker_{}players".format(config["num_players"]), name="{}_{}_NFSP".format(config["rl_algo"], config["sl_algo"]))
  else:
    wandb.init(project="Leduc_Poker_{}players".format(config["num_players"]), name="{}_{}_NFSP".format(config["rl_algo"], config["sl_algo"]))
  wandb.config.update(config)
  wandb.define_metric("exploitability", summary="last")
  wandb.define_metric("avg_utility", summary="last")


# _________________________________ train _________________________________

leduc_trainer = NFSP_Leduc_Poker_trainer.LeducTrainer(
  random_seed = config["random_seed"],
  train_iterations = config["iterations"],
  num_players= config["num_players"],
  wandb_save = config["wandb_save"],
  save_matplotlib = config["save_matplotlib"],
  )


leduc_RL = NFSP_Leduc_Poker_reinforcement_learning.ReinforcementLearning(
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
  leduc_trainer_for_rl = leduc_trainer,
  alpha = config["rl_alpha"],
  rl_strategy = config["rl_strategy"],
  alpha_discrease = config["alpha_discrease"]
  )




leduc_SL = NFSP_Leduc_Poker_supervised_learning.SupervisedLearning(
  random_seed = config["random_seed"],
  train_iterations = config["iterations"],
  num_players= config["num_players"],
  hidden_units_num= config["sl_hidden_units_num"],
  lr = config["sl_lr"],
  epochs = config["sl_epochs"],
  sampling_num = config["sl_sampling_num"],
  leduc_trainer_for_sl = leduc_trainer
  )




leduc_GD = NFSP_Leduc_Poker_generate_data.GenerateData(
  random_seed = config["random_seed"],
  num_players= config["num_players"],
  leduc_trainer_for_gd= leduc_trainer
  )


leduc_trainer.train(
  eta = config["eta"],
  memory_size_rl = config["memory_size_rl"],
  memory_size_sl = config["memory_size_sl"],
  rl_algo = config["rl_algo"],
  sl_algo = config["sl_algo"],
  rl_module= leduc_RL,
  sl_module= leduc_SL,
  gd_module= leduc_GD
  )


# _________________________________ result _________________________________

if not config["wandb_save"]:
  print("avg_utility", list(leduc_trainer.avg_utility_list.items())[-1])
  print("final_exploitability", list(leduc_trainer.exploitability_list.items())[-1])


result_dict_avg = {}
for key, value in sorted(leduc_trainer.avg_strategy.items()):
  result_dict_avg[key] = value
df = pd.DataFrame(result_dict_avg.values(), index=result_dict_avg.keys(), columns=["Fold_avg", "Call_avg", "Raise_avg"])
df.index.name = "Node"

result_dict_br = {}
for key, value in sorted(leduc_trainer.epsilon_greedy_q_learning_strategy.items()):
  result_dict_br[key] = value
df1 = pd.DataFrame(result_dict_br.values(), index=result_dict_br.keys(), columns=["Fold_br", "Call_br", "Raise_br"])
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
  if config["save_matplotlib"]:
    df = pd.DataFrame(leduc_trainer.database_for_plot)
    df = df.set_index('iteration')
    if config["rl_algo"] == "sql":
      df.to_csv('../../../Other/Make_png/output/Leduc_Poker/{}players/DB_for_{}_NFSP_{}_{}_{}.csv'.format(config["num_players"],config["random_seed"], config["rl_algo"], config["rl_alpha"], config["alpha_discrease"]))
    else:
      df.to_csv('../../../Other/Make_png/output/Leduc_Poker/{}players/DB_for_{}_NFSP_{}.csv'.format(config["num_players"],config["random_seed"], config["rl_algo"]))

    #実験時間の計測
    end_time = time.time()
    total_time = end_time - start_time
    if config["rl_algo"] == "sql":
      path = '../../../Other/Make_png/output/Leduc_Poker/Time/time_{}players_NFSP_{}_{}_{}_{}.txt'.format(config["num_players"], config["rl_algo"], config["rl_alpha"], config["alpha_discrease"], config["random_seed"])

    else:
      path = '../../../Other/Make_png/output/Leduc_Poker/Time/time_{}players_NFSP_{}_{}.txt'.format(config["num_players"], config["rl_algo"], config["random_seed"])

    f = open(path, 'w')
    f.write(str(total_time))
    f.close()

doctest.testmod()
