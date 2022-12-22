import multiprocessing as mp
# https://docs.python.org/3/library/multiprocessing.html
import random
import time
from collections import deque

def make_episodes(episode_num, rl_module):
    """
    複数のエピソードを作成し、データを取得

    Parameters
    ----------
    episode_num : int
        生成するエピソードの個数
    Returns
    -------
    output_list : list
        エピソードのデータ
    """
    output_list = []
    for _ in range(episode_num):
        str_sequence = ""
        num = 0
        while random.choice([True, False]):
            str_sequence += str(num)
            num +=1
        str_sequence += "_{}".format(rl_module.weight)
        output_list.append(str_sequence)
    return output_list


def wait_and_make_episode_loop(q_in, q_out):
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
        episode_num , rl_module = q_in.get()

        #プロセス終了
        if episode_num < 0:
            break
        #仕事start
        episode_memory = make_episodes(episode_num, rl_module)
        q_out.put(episode_memory)


def strategy_update(memory, rl_module):
    #重みが変更されるイメージ
    rl_module.weight += 1
    print(len(memory), [memory[0] if len(memory)>0 else "Nothing"])



class RL:
    def __init__(self):
        #NNの重みをイメージ
        self.weight = 0



if __name__ == '__main__':
    #q_in → 仕事startの合図 (エピソード数、モジュール), q_out → エピソードデータ貯める
    q_out, q_in = mp.Queue(), mp.Queue()
    episode_num = 10

    #強化学習module
    rl_module = RL()

    p = mp.Process(target=wait_and_make_episode_loop, args=(q_in, q_out))
    p.start()


    #エピソード作成と戦略更新
    iteration = 1000

    Memory = deque([], maxlen=100)

    for _ in range(iteration):

        #エピソード作成
        q_in.put([episode_num, rl_module])

        #queueに溜まってるデータがあれば、取り出す
        while not q_out.empty():
            for data_RL in q_out.get():
                Memory.append(data_RL)


        #戦略更新
        strategy_update(Memory, rl_module)


    #Process終了の合図
    x = -1
    q_in.put([x, None])
    p.join()
