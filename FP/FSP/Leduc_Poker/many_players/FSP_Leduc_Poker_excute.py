#Library
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
import wandb

import FSP_Leduc_Poker_trainer


start_time = time.time()
#config
config = dict(
  random_seed = [42][0],
  iterations = 10**6,
  num_players = 2,
  n= 2,
  m= 1,
  memory_size_rl= 1000,
  memory_size_sl= 1000,
  rl_algo = ["epsilon-greedy", "boltzmann", "dfs"][0],
  sl_algo = ["cnt", "mlp"][0],
  pseudo_code = ["general_FSP", "batch_FSP"][1],
  wandb_save = [True, False][1],
  save_matplotlib = [True, False][0],
)



if config["wandb_save"]:
  wandb.init(project="Leduc_Poker_{}players".format(config["num_players"]), name="fsp_{}_{}_{}".format(config["rl_algo"], config["sl_algo"], config["pseudo_code"]))
  wandb.config.update(config)


#train

leduc_trainer = FSP_Leduc_Poker_trainer.LeducTrainer(
  train_iterations = config["iterations"],
  num_players= config["num_players"],
  random_seed = config["random_seed"],
  save_matplotlib = config["save_matplotlib"],
  )


leduc_trainer.train(
  n = config["n"],
  m = config["m"],
  memory_size_rl = config["memory_size_rl"],
  memory_size_sl = config["memory_size_sl"],
  rl_algo = config["rl_algo"],
  sl_algo = config["sl_algo"],
  pseudo_code = config["pseudo_code"],
  wandb_save = config["wandb_save"],
  )


#result

print("")
print("avg_utility", leduc_trainer.eval_vanilla_CFR("", 0, 0, [1.0 for _ in range(config["num_players"])]))
print("final_exploitability", list(leduc_trainer.exploitability_list.items())[-1])
print("")


pd.set_option('display.max_rows', None)
result_dict_avg = {}
for key, value in sorted(leduc_trainer.avg_strategy.items()):
  result_dict_avg[key] = value
df = pd.DataFrame(result_dict_avg.values(), index=result_dict_avg.keys(), columns=["Fold_avg", "Call_avg", "Raise_avg"])
df.index.name = "Node"

result_dict_br = {}
for key, value in sorted(leduc_trainer.best_response_strategy.items()):
  result_dict_br[key] = value
df1 = pd.DataFrame(result_dict_br.values(), index=result_dict_br.keys(), columns=["Fold_br", "Call_br", "Raise_br"])
df1.index.name = "Node"

print(pd.concat([df, df1], axis=1))


#追加 matplotlibで図を書くため
if config["save_matplotlib"]:
  df = pd.DataFrame(leduc_trainer.database_for_plot)
  df = df.set_index('iteration')
  df.to_csv('../../../../Other/Make_png/output/Leduc_Poker/{}players/DB_for_{}_FSP.csv'.format(config["num_players"], config["random_seed"]))

  #実験時間の計測
  end_time = time.time()
  total_time = end_time - start_time
  path = '../../../../Other/Make_png/output/Leduc_Poker/Time/time_{}players_FSP_{}.txt'.format(config["num_players"], config["random_seed"])

  f = open(path, 'w')
  f.write(str(total_time))
  f.close()


#for i in range(2,3):
#  leduc_poker_agent = FSP_Leduc_Poker_trainer.LeducTrainer(train_iterations=0, num_players=i)
#  print("{}player game:".format(i), leduc_poker_agent.get_exploitability_dfs())


doctest.testmod()
