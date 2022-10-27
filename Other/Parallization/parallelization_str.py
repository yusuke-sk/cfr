from multiprocessing import Process, Queue
import time
import math
from collections import deque
import random

def make_str_episode(iter, queue):
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
