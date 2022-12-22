import multiprocessing as mp
# https://docs.python.org/3/library/multiprocessing.html
import random
import time

def make_episodes(episode_num, info):
    """
    複数のエピソードを作成し、データを取得

    Parameters
    ----------
    episode_num : int
        生成するエピソードの個数
    info : dict
        重要な情報 ex NNの重みなど

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
        output_list.append(str_sequence)
    return output_list


def run_and_make_whether_need_episodes(q_in, q_out, episode_num, info):
    """
    合図が来たら、make_episodesを実行し、結果をqueueに渡す。

    Parameters
    ----------
    q_in : queue
        合図を送るqueue
    q_out : queue
        得られたデータを送るqueue
    episode_num : int
        生成するエピソードの個数
    info : dict
        重要な情報 ex NNの重みなど
    """

    while True:
        whether_need_episode = q_in.get()
        #プロセス終了
        if whether_need_episode < 0:
            break
        #仕事start
        episode_memory = make_episodes(episode_num, info)
        q_out.put(episode_memory)


def strategy_update(memory):
    count = 0
    for list_i in memory:
        count += len(list_i)
    print(count)



if __name__ == '__main__':
    #q_in → 仕事startの合図, q_out → エピソードデータ貯める
    q_out, q_in = mp.Queue(), mp.Queue()
    episode_num = 10

    #info → processにうけわたす必要のあるもの、NNの重みなど？
    info = ""
    p = mp.Process(target=run_and_make_whether_need_episodes, args=(q_in, q_out, episode_num, info))
    p.start()

    #エピソード作成と戦略更新
    iteration = 10
    Memory = []
    for _ in range(iteration):

        #エピソード作成
        x = 1
        q_in.put(x)

        Memory.append(q_out.get())

        #戦略更新
        strategy_update(Memory)


    #Process終了の合図
    x = -1
    q_in.put(x)
    p.join()
