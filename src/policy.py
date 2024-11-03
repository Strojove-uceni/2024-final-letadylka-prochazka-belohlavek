import numpy as np
import torch
from gymnasium.spaces import Discrete
from routing import Routing


"""
    We proly don't need other policies right now.
"""

class ShortestPath:
    """
    :statis_shortest_paths: REMOVED
    """

    def __init__(self, env, model, action_space, args) -> None:
        self.env = env
        assert isinstance(env.get(), Routing)

        self.n_agents = env.get_num_agents()
        self.model = model
        self.action_space = action_space
        self.args = args
        self.network = None

    

    def reset_episode(self):
        self.network = None

    def __call__(self, obs, adj):
        act = np.zeros(self.env.n_planes, dtype=np.int32)

        network = self.env.network

        # Choose next node for each plane
        for i in range(self.env.n_planes):
            plane = self.env.planes[i]
            current_node = plane.now
            target_node = plane.target

            if current_node == target_node:
                #act[i] = 0 # This cannot be here since our planes cannot stop in the air
                continue

            # Get the next waypoint on the flight path
            next_node = network.shortest_paths[current_node][target_node][1]

            # Find the edge that leads to the next_node
            for index, j in enumerate(network.nodes[current_node].edges):
                current_edge = network.edges[j]

                # Now we choose an edge
                if current_edge.get_other_node(plane.now) == next_node:
                    act[i] = index + 1 # Should the + 1 be here? I dunno now
                    break
        return act
    