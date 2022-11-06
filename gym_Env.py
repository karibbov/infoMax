import gym, ray
import numpy as np
import networkx as nx
from gym.spaces import Box, MultiBinary, Dict, MultiDiscrete
from scratch import SocialGraph, Node, Message


class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.n_nodes = 100
        self.action_space = MultiDiscrete([self.n_nodes, self.n_nodes])
            # Box(low=0, high=self.n_nodes-1, shape=(1,), dtype=int)
        self.observation_space = Dict({"connected_nodes": MultiBinary(self.n_nodes),
                                       "original_messages": MultiBinary(self.n_nodes),
                                       "original_sources": Box(low=0, high=np.inf, shape=(self.n_nodes,))})
        self.graph = SocialGraph()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_graph(seed)
        for _ in range(300):
            self.graph = self.graph.step()
        print(len(list(nx.isolates(self.graph))))
        SocialGraph.total_message_count = 1
        SocialGraph.unique_message_count = 1
        self._connected_nodes = np.zeros(self.n_nodes)
        self._original_messages = np.zeros(self.n_nodes)
        self._original_sources = np.zeros(self.n_nodes)
        self.messages = set()
        self.unique_messages = 0
        self.total_messages = 0
        self.steps = 0
        return self._get_obs()

    def step(self, action):
        self._original_messages = np.zeros(self.n_nodes)
        self._take_action(action)
        self.steps += 1
        reward = self._compute_reward()
        info = {}
        done = (self.steps == 400)
        print(f"step: {self.steps}, reward: {reward}, uniques: {self.unique_messages}, our ratio: {self.unique_messages / self.total_messages} "
              f"connected: {sum(self._connected_nodes)}, total: {self.total_messages} avegare ratio: {self.graph.unique_message_count / self.graph.total_message_count}")
        return self._get_obs(), reward, done, info

    def _compute_reward(self):
        if sum(self._connected_nodes) == 0:
            return -1
        return (2*sum(self._original_messages) - sum(self._connected_nodes)) / self.graph.step_messages



    def _take_action(self, action):
        def get_messages():
            connected_nodes = np.argwhere(self._connected_nodes == 1).flatten()
            for i in connected_nodes:
                if self.graph.out_edges(self.nodes[i], data=True):
                    _, _, attr = list(self.graph.out_edges(self.nodes[i], data=True))[0]
                    message = attr['obj']
                    if message.signature not in self.messages:
                        self.messages.add(message.signature)
                        self.unique_messages += 1
                        self._original_messages[i] = 1
                        self._original_sources[message.origin.id] += 1
                    self.total_messages += 1

        add_id = action[0]
        remove_id = action[1]
        self._connected_nodes[add_id] = 1
        self._connected_nodes[remove_id] = 0
        self.graph = self.graph.step(dynamic=False)
        get_messages()



    def _get_obs(self):
        return {"connected_nodes": self._connected_nodes,
                "original_messages": self._original_messages,
                "original_sources": self._original_sources}


    def _initialize_graph(self, seed):
        def create_nodes(n_nodes):
            originality = get_probs(n_nodes, scale=10)
            follow_prob = get_probs(n_nodes)
            unfollow_prob = get_probs(n_nodes, scale=32)
            nodes = [Node(id=id, originality_prob=originality[id],
                          follow_prob=follow_prob[id],
                          unfollow_prob=unfollow_prob[id]) for id in range(n_nodes)]
            return nodes

        def get_probs(n: int, offset: float = 0.0, scale: float = 12.0, negative: bool = False):
            sample = np.zeros(n)
            for i in range(n):
                for _ in range(100):
                    a = (self.np_random.gamma(shape=3, scale=2, size=1) / scale) + offset
                    if 1 > a > 0:
                        break
                else:
                    a = 1 if a >= 0.5 else 0
                if negative:
                    a = 1 - a
                sample[i] = a
            return sample

        n_nodes = self.n_nodes
        n_edges = 400
        self.nodes = create_nodes(n_nodes)
        edge_ids = [np.random.choice(np.arange(self.n_nodes), 2, replace=False) for _ in range(n_edges)]
        edges = [(self.nodes[a], self.nodes[b], {"obj": self.graph.create_message(self.nodes[a])}) for (a, b) in edge_ids]
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(edges)


# env = MyEnv(None)
# env.reset()
# actions = Box(low=0, high=100, shape=(1,), dtype=int)
#
# env.step(actions.sample())
from ray.rllib.algorithms import ppo

ray.init()
algo = ppo.PPO(env=MyEnv, config={
    "env_config": {},  # config to pass to env class
    "framework": "torch",
})

while True:
    print(algo.train())