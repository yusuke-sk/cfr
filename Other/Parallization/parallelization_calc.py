from multiprocessing import Process, Queue, Pool, Manager
import time
import math
from collections import deque

def memo(memory, strategy):
    for food, money in memory:
        strategy[food] = money



if __name__ == '__main__':
    # --  ver1 --
    with Manager() as manager:
        d = manager.dict()
        M = [["apple", 100], ["banana", 200] ,["orange", 300]]
        process = Process(target = memo, args=(M, d))
        process.start()
        process.join()

        print(d)
