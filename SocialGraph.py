import networkx as nx
from dataclasses import dataclass
from numpy import random as rn


@dataclass(frozen=True)
class Node:
    id: int
    originality_prob: float
    follow_prob: float
    unfollow_prob: float
    random_follow_prob: float = 0.05
    # rng = None

    def is_original(self):
        return self.originality_prob > rn.uniform(low=0, high=1)

    def is_follow(self):
        return self.follow_prob > rn.uniform(low=0, high=1)

    def is_unfollow(self):
        return self.unfollow_prob > rn.uniform(low=0, high=1)

    def is_random_follow(self):
        return self.random_follow_prob > rn.uniform(low=0, high=1)

    def __repr__(self):
        return f"{self.id}"

@dataclass()
class Message:
    signature: str
    origin: int
    share_prob: float

    def is_share(self):
        return self.share_prob > rn.uniform(low=0, high=1)


# nodes = [Node(0, 0.7, 0.2, 0.1), Node(1, 0.5, 0.2, 0.1)]
#
# G = nx.MultiDiGraph()
# # G.add_node(nodes[0])
# # G.add_node(nodes[1])
# # print(nodes[0])
# G.add_nodes_from(nodes)
# G.add_edge(nodes[0], nodes[1], obj=Message("sadfsadfsadf", 0, 0.2))
# # G.add_edge(nodes[1], nodes[1], obj=Message("sadfsadfsadf", 1, 0.2))
# print(list(G.successors(nodes[0])))
# print(list(G.predecessors(nodes[1])))
# # G.remove_edge(nodes[0], nodes[1])
# print(G.nodes())
# # print(nodes[0].out_degree())
# print(G.out_degree)
# print(G.out_edges(nodes[0]))
# print(G.out_edges(nodes[1]))
# t = [x for x in G.nodes() if G.out_degree(x) == 0]
# print(t)
# print(G.edges)
# print(G.in_edges(nodes[1], data=True))

import string
from itertools import combinations_with_replacement
import random


# print(len(alphabet))


# def create_message(node):
#     signature = "".join(next(strings))
#     origin = node
#     share_prob = rn.uniform(low=0, high=1)
#     return Message(signature, origin, share_prob)

class SocialGraph(nx.DiGraph):

    alphabet = string.ascii_letters + string.digits + string.punctuation
    strings = combinations_with_replacement(alphabet, 64)
    total_message_count: int = 0
    unique_message_count: int = 0
    step_messages: int = 0

    def __init__(self):
        super(SocialGraph, self).__init__()

    def step(self, dynamic: bool = True):
        """
        Traverse all the nodes and edges, update all edge data add and remove edges where necessary
        :return:
        """
        G = SocialGraph()
        tep_messages = 0
        # Loop for adding edges to the new graph
        for node in self.nodes:
            G.add_node(node)
            # Send messages/create edges
            if node.is_original():
                message = self.create_message(node)
            elif self.in_edges(node, data=True):
                # print(self.in_edges(node, data=True))
                _, _, attr = random.choice(list(self.in_edges(node, data=True)))
                message = attr['obj']
            elif self.out_edges(node, data=True):
                _, _, attr = list(self.out_edges(node, data=True))[0]
                message = attr['obj']
            else:
                message = None

            message_shared = False
            for successor in self.successors(node):
                G.add_edge(node, successor, obj=message)
                message_shared = True

            step_messages = 1 + step_messages if message_shared else step_messages

            if dynamic:
                # Follow someone
                if node.is_follow() and message and message.origin.id != node.id:
                    G.add_edge(message.origin, node, obj=message)

                if node.is_random_follow():
                    end_nodes = [x for x in self.nodes() if self.out_degree(x) == 0 and x.id != node.id]
                    if end_nodes:
                        follow_node = random.choice(end_nodes)
                        message = self.create_message(follow_node)
                    else:
                        follow_node = random.choice([x for x in self.nodes() if x.id != node.id])
                        _, _, attr = list(self.out_edges(follow_node, data=True))[0]
                        message = attr['obj']
                    G.add_edge(follow_node, node, obj=message)


        if dynamic:
            # Loop for removing nodes
            for node in self.nodes:
                if node.is_unfollow() and self.in_edges(node, data=True):
                    u, v, attr = random.choice(list(self.in_edges(node, data=True)))
                    G.remove_edge(u, v)

        # Total count of nodes sharing messages all across the network
        SocialGraph.total_message_count += step_messages
        SocialGraph.step_messages = step_messages

        return G.copy()

    @staticmethod
    def create_message(node):
        signature = "".join(next(SocialGraph.strings))
        origin = node
        share_prob = rn.uniform(low=0, high=1)
        SocialGraph.unique_message_count += 1
        return Message(signature, origin, share_prob)

# SG = SocialGraph()
# nodes = [Node(0, 0.7, 0.4, 0.05), Node(1, 0.5, 0.6, 0.1), Node(2, 0.5, 0.4, 0.05), Node(3, 0.5, 0.3, 0.05)]
# SG.add_edge(nodes[0], nodes[1], obj=SG.create_message(nodes[0]))
# SG.add_edge(nodes[1], nodes[2], obj=SG.create_message(nodes[1]))
# SG.add_edge(nodes[2], nodes[1], obj=SG.create_message(nodes[2]))
# SG.add_edge(nodes[2], nodes[0], obj=SG.create_message(nodes[2]))
# SG.add_edge(nodes[1], nodes[3], obj=SG.create_message(nodes[1]))
# SG.add_edge(nodes[3], nodes[2], obj=SG.create_message(nodes[3]))
# print(SG.edges)
# SG = SG.step()
# # print(SG.edges)
#
# for i in range(3000):
#     SG = SG.step(dynamic=False)
#     print(SG.edges)
# print(SG.unique_message_count)
# print(SG.total_message_count)
