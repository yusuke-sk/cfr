import time
import ray
import random


class DC_Trainer:
    def __init__(self, num_agents, batch_episode_num):
        self.num_agents = num_agents
        self.batch_episode_num = batch_episode_num

    @ray.remote
    def make_episodes(self):
        memory = []
        #episode_num分のデータ作成
        for i in range(self.batch_episode_num):
            print(i)
            one_episode = self.make_one_episode()
            memory.append(one_episode)
            print(one_episode)

        return memory


    def make_one_episode(self):
        str_list = ["a", "b", "c", "d", "e","f", "g", "h", "i", "j"]
        result = ""
        not_j = True
        while not_j:
            ri = random.randint(0,9)
            st = str_list[ri]
            result += st
            if st == "j":
                not_j = False
            time.sleep(0.01)
        return result


    def learn(self):
        all_data = []
        data_rep_object_id = [self.make_episodes.remote() for i in range(self.num_agents)]
        print(data_rep_object_id)
        exit()
        data_list = ray.get(data_rep_object_id)
        all_data += [c for b in data_list for c in b]


        print("data amount:", len(all_data))


if __name__ == '__main__':
    ray.init()
    start_time = time.time()
    agent = DC_Trainer(num_agents=1, batch_episode_num=300)
    agent.learn()
    end_time = time.time()

    print("time:", end_time-start_time)
