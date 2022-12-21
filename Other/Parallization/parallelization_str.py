from multiprocessing import Process, Queue
import time
import math
from collections import deque
import random

#この5つが時間かかる原因 1s → 0.11 (0.9s の削減にはなっている)
import torch.nn as nn  #そもそも使ってない
import wandb #計算時間はかるだけだとなくていい
import matplotlib.pyplot as plt # そもそも使っていない
import torch #使っているけど、使わないように変更することは可能
import pandas as pd #そもそも使っていない


#下のライブラリで(0.05sかかっている)
import numpy as np
import itertools
from collections import defaultdict
from tqdm import tqdm
import doctest
import copy
from collections import deque



class Toy1:
    def __init__(self):
        pass

    def make_str_episode(self, iter, queue):
        str_list = ["a", "b", "c", "d", "e","f", "g", "h", "i", "j"]
        for ii in range(iter):
            result = ""
            not_j = True
            while not_j:
                ri = random.randint(0,9)
                st = str_list[ri]
                result += st
                if st == "j":
                    not_j = False
                time.sleep(0.01)
            queue.put(result)


    def make_episodes_paralleled(self, episode_num):

        queue = Queue()
        process1 = Process(target=self.make_str_episode, args=(episode_num, queue))

        start_time = time.time()

        process1.start()
        process1.join()

        end_time = time.time()
        print("ver1 time:", end_time - start_time)




class Toy:
    def __init__(self, episode_num):
        self.episode_num = episode_num
        self.queue = Queue()
        self.process1 = Process(target=self.make_str_episode, args=(self.episode_num,))
        self.counter = 0


    def make_str_episode(self, iter):
        str_list = ["a", "b", "c", "d", "e","f", "g", "h", "i", "j"]
        for ii in range(iter):
            result = ""
            not_j = True
            while not_j:
                ri = random.randint(0,9)
                st = str_list[ri]
                result += st
                if st == "j":
                    not_j = False
                time.sleep(0.01)
            self.queue.put(result)


    def make_episodes_paralleled(self):

        start_time = time.time()
        #process　重み共有する shared memory
        #job_queue
        #joib_queue (episode作成して) push して pop (process targetの関数 の中で popする) 、process　エピソード作成する
        #data_queue
        #dataがあれば、受け取る
        self.process1.start()
        self.process1.join()
        end_time = time.time()

        print("ver1 time:", end_time - start_time)



    def count_and_delte(self):
        while not self.queue.empty():
            self.queue.get()
            self.counter += 1
        print(self.counter)



if __name__ == '__main__':
    #toy_trainer = Toy1()
    #toy_trainer.make_episodes_paralleled(episode_num=0)


    iteration = 10
    toy_trainer = Toy(episode_num = 10)
    for _ in range(iteration):
        toy_trainer.make_episodes_paralleled()
        toy_trainer.count_and_delte()







"""
if __name__ == '__main__':
    # --  ver1 --
    start_time = time.time()

    queue = Queue()
    process1 = Process(target=make_str_episode, args=(100, queue))
    process2 = Process(target=make_str_episode, args=(100, queue))
    process3 = Process(target=make_str_episode, args=(100, queue))


    process1.start()
    process2.start()
    process3.start()

    process1.join()
    process2.join()
    process3.join()

    end_time = time.time()
    print(queue.get())
    print("ver1 time:", end_time - start_time)


    # --  ver2 --
    start_time = time.time()

    queue = Queue()
    process4 = Process(target=make_str_episode, args=(300, queue))
    process4.start()
    process4.join()

    end_time = time.time()
    print(queue.get())
    print("ver2 time:", end_time - start_time)
"""
