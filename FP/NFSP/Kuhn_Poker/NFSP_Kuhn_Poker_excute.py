
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
import Episodic_NFSP_Kuhn_Poker_trainer
import Paralleled_DC_NFSP_Kuhn_Poker_trainer
import Paralleled_SU_NFSP_Kuhn_Poker_trainer
import NFSP_Kuhn_Poker_supervised_learning
import NFSP_Kuhn_Poker_reinforcement_learning_DQN
import NFSP_Kuhn_Poker_generate_data


if __name__ == '__main__':
# _________________________________ config _________________________________

  config = dict(
    random_seed = [42, 1000, 10000][0],
    iterations = 10**6,
    num_players = 2,
    #parallelized
    batch_episode_num = [40, 30, 20, 20, 15][2-2],
    wandb_save = [True, False][1],
    parallelized = ["DataCollect","StrategyUpdate", False][2],
    collect_step_or_episode = ["step", "episode"][0],
    whether_accurate_exploitability =[True, False, "Dont_calculate"][0],
    #rl
    rl_algo = ["dfs", "dqn", "ddqn", "sql"][3],
    #result matplotlib
    save_matplotlib = [True, False][0],
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
    rl_alpha = 1e-1,
    rl_strategy = ["ε-greedy", "proportional_Q"][0],
    alpha_discrease = [True, False][1],
    )

    config.update(config_plus)


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
    rl_critic_lr =  0.01,
    rl_loss_function = [F.mse_loss, nn.HuberLoss()][0],
    # device
    #device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    )

    config.update(config_plus)

  if config["wandb_save"]:
    #並列化の実験用
    if config["parallelized"] ==  "DataCollect" or   config["parallelized"] ==  "StrategyUpdate":
      wandb.init(project="Kuhn_Poker_trained_Parallel_{}players".format(config["num_players"])
      , name="{}_NFSP".format(config["parallelized"]))

    elif config["rl_algo"] == "sql":
      #wandb.init(project="Kuhn_Poker_{}players_SQL".format(config["num_players"]), name="{}_{}_A_{}_P_{}_NFSP".
      #format(config["rl_algo"], config["rl_alpha"], config["alpha_discrease"], config["parallelized"]))
      if config["alpha_discrease"]:
        alpha_type = "alpha:" + str(config["rl_alpha"]) + "_decrease"
      else:
        alpha_type = "alpha:" + str(config["rl_alpha"]) + "_const"
      wandb.init(project="Kuhn_Poker_{}players".format(config["num_players"]), name="{}_{}_NFSP".format(config["rl_algo"],alpha_type))

    else:
      #wandb.init(project="Kuhn_Poker_{}players".format(config["num_players"]), name="{}_{}_{}_NFSP".format(config["rl_algo"], config["sl_algo"],
      #config["collect_step_or_episode"]))
      wandb.init(project="Kuhn_Poker_{}players".format(config["num_players"]), name="{}_{}_NFSP".format(config["rl_algo"], config["sl_algo"]))

    wandb.config.update(config)
    wandb.define_metric("exploitability", summary="last")
    wandb.define_metric("avg_utility", summary="last")



  # _________________________________ train _________________________________
  if config["parallelized"] == "DataCollect":
    kuhn_trainer = Paralleled_DC_NFSP_Kuhn_Poker_trainer.KuhnTrainer(
      random_seed = config["random_seed"],
      train_iterations = config["iterations"],
      num_players= config["num_players"],
      wandb_save = config["wandb_save"],
      step_per_learning_update = config["step_per_learning_update"],
      batch_episode_num = config["batch_episode_num"],
      whether_accurate_exploitability = config["whether_accurate_exploitability"],
      save_matplotlib = config["save_matplotlib"],
      )

  elif config["parallelized"] == "StrategyUpdate":
    kuhn_trainer = Paralleled_SU_NFSP_Kuhn_Poker_trainer.KuhnTrainer(
      random_seed = config["random_seed"],
      train_iterations = config["iterations"],
      num_players= config["num_players"],
      wandb_save = config["wandb_save"],
      step_per_learning_update = config["step_per_learning_update"],
      batch_episode_num = config["batch_episode_num"],
      whether_accurate_exploitability = config["whether_accurate_exploitability"],
      save_matplotlib = config["save_matplotlib"],
      )

  elif config["parallelized"] == False and config["collect_step_or_episode"] == "step":
    kuhn_trainer = NFSP_Kuhn_Poker_trainer.KuhnTrainer(
      random_seed = config["random_seed"],
      train_iterations = config["iterations"],
      num_players= config["num_players"],
      wandb_save = config["wandb_save"],
      step_per_learning_update = config["step_per_learning_update"],
      whether_accurate_exploitability = config["whether_accurate_exploitability"],
      save_matplotlib = config["save_matplotlib"],
      )

  elif config["parallelized"] == False and config["collect_step_or_episode"] == "episode":
    kuhn_trainer = Episodic_NFSP_Kuhn_Poker_trainer.KuhnTrainer(
      random_seed = config["random_seed"],
      train_iterations = config["iterations"],
      num_players= config["num_players"],
      wandb_save = config["wandb_save"],
      step_per_learning_update = config["step_per_learning_update"],
      batch_episode_num = config["batch_episode_num"],
      whether_accurate_exploitability = config["whether_accurate_exploitability"],
      save_matplotlib = config["save_matplotlib"],
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
      alpha = config["rl_alpha"],
      rl_strategy = config["rl_strategy"],
      alpha_discrease = config["alpha_discrease"]
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

  if not config["wandb_save"] and config["whether_accurate_exploitability"]!= "Dont_calculate":
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


  if config["wandb_save"]:
    tbl = wandb.Table(data=df2)
    tbl.add_column("Node", [i for i in df2.index])
    wandb.log({"table:":tbl})
    wandb.save()
  else:
    print(df2)

  #追加 matplotlibで図を書くため
  if config["save_matplotlib"]:
    df = pd.DataFrame(kuhn_trainer.database_for_plot)
    df = df.set_index('iteration')

    #並列化
    #df.to_csv('../../../Other/Make_png/output/Kuhn_Poker/Parallel/DB_for_NFSP_{}_{}.csv'.format(config["num_players"], config["parallelized"]))

    #提案手法SQL
    if config["rl_algo"] == "sql":
      df.to_csv('../../../Other/Make_png/output/Kuhn_Poker/{}players/DB_for_NFSP_{}_{}_{}.csv'.format(config["num_players"], config["rl_algo"], config["rl_alpha"], config["alpha_discrease"]))
    else:
      df.to_csv('../../../Other/Make_png/output/Kuhn_Poker/{}players/DB_for_NFSP_{}.csv'.format(config["num_players"], config["rl_algo"]))


  doctest.testmod()
