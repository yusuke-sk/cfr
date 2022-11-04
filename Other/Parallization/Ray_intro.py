import time
import ray
import random


@ray.remote
class DC_Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.memory = []


    def collect_memory(self):
        memory = self.memory
        self.memory = []
        return memory


    def make_episodes(self, episodes_num):
        #episode_num分のデータ作成
        for _ in range(episodes_num):
            one_episode = self.make_one_episode()
            self.memory.append(one_episode)

        #メモリー出力して中身は空っぽにする
        return self.collect_memory()



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




def learn(num_agents, batch_episode_num):
    all_data = []
    ray.init()

    dc_agents = [DC_Agent.remote(agent_id=i) for i in range(num_agents)]

    data_rep_object_id = [dc_agents[i].make_episodes.remote(batch_episode_num) for i in range(num_agents)]
    data_list = ray.get(data_rep_object_id)
    all_data += [c for b in data_list for c in b]

    print("data amount:", len(all_data))


if __name__ == '__main__':
    start_time = time.time()

    learn(num_agents=1, batch_episode_num=300)
    end_time = time.time()

    print("time:", end_time-start_time)
