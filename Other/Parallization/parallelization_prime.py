from multiprocessing import Process, Queue
import time
import math
from collections import deque

def count_prime_number(num1, num2, queue):
    #count_p = 0
    for i in range(max(num1,2), num2+1):
        for j in range(2,int(math.sqrt(i))+1):
            if i % j == 0:
                break
        else:
            #count_p += 1
            queue.put(i)


if __name__ == '__main__':
    # --  ver1 --
    start_time = time.time()

    queue = Queue()
    process1 = Process(target=count_prime_number, args=(1,100, queue))
    process2 = Process(target=count_prime_number, args=(101,200, queue))

    process1.start()
    process2.start()

    process1.join()
    process2.join()

    end_time = time.time()
    print(queue)
    print(queue.get())
    print(queue.get())
    print("time:", end_time - start_time)

    # --  ver2 --
    start_time = time.time()
    queue = Queue()
    process1 = Process(target=count_prime_number, args=(1,2000000, queue))
    process1.start()
    process1.join()

    end_time = time.time()
    print(queue.get())
    print("time:", end_time - start_time)
