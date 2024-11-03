import textwrap
import numpy as np
from collections import defaultdict

from environment import EnvironmentVariant, NetworkEnv
from gymnasium.spaces import Discrete

from network import Network
from util import one_hot_list

class Plane:
    """
    A plane that travels through the waypoints.

    This class is a changed version of the Data class from routing.py from the original implementation.
    The changes are:
        - Removed TTL(Time To Live) as we do not discard planes
        - Removed size as we do not really need it now
        - Added speed since that is pretty important in plane travel, we will see if we are able to incorporate it
    """


    def __init__(self, id) -> None:
        self.id = id
        self.now = None
        self.target = None
        self.size = None    # May remove also
        self.start = None
        self.time = None
        self.edge = -1
        self.neigh = None
        self.shortest_path_weight = None
        self.visited_nodes = None
        self.speed = None       # Added

    def reset(self, start, target, shortest_path_weight, speed):
        self.now = start
        self.target = target
        self.start = start
        self.size = 0.5     # May be removed
        self.time = 0
        self.edge = -1
        self.neigh = [self.id]
        self.speed = speed
        self.shortest_path_weight = shortest_path_weight
        self.visited_nodes = set([start])


class Routing(NetworkEnv):
    """
        Changes are:
            - data is renamed to planes
            - n_data is renamed to n_planes
            - added two additional params: adj_mat and dist_mat
    """


    def __init__(self, network: Network, n_planes, env_var: EnvironmentVariant,  adj_mat, dist_mat , k =3, enable_action_mask=True) -> None:
        super(Routing, self).__init__()

        self.network = network
        self.n_planes = n_planes
        self.planes = []

        self.env_var = EnvironmentVariant(env_var)

        self.adj_mat = adj_mat
        self.amount_of_neighbors = len(np.nonzero(adj_mat[1,:]))
        self.dist_mat = dist_mat

        # k neighbors
        self.k = k

        # log information
        self.agent_steps = np.zeros(self.n_planes)

        # Commented for now
         # whether to use random targets or target == 0 for all packets
        # self.num_random_targets = self.network.n_nodes
        # assert self.num_random_targets >= 0

        # map shortest path to actual agents steps
        self.distance_map = defaultdict(list)
        self.sum_packets_per_node = None
        self.sum_packets_per_edge = None

        self.enable_action_mask = enable_action_mask
        self.action_mask = np.zeros((n_planes, self.amount_of_neighbors), dtype=bool)


        """
        It is set like this, because the original had Discrete(4, start = 0). That is because the original could have had packets waiting within a router. Unfortunately, we cannot have a plane waiting in a waypoint. 
            Maybe we could argue that is flying in a circle, but we will try it like this now.
        """
        # The number of choices should be equal to the maximum number of neighbors a node can have
        # in our case it is quite simple as every node as a static number of neighbors, at least for now
        self.action_space = Discrete(self.amount_of_neighbors-1, start=0) # {0,1,2} using gym action space

        self.eval_info_enables = False


    def set_eval_info(self, val):
            """
            Whether the step function should return additional info for evaluation.

            :param val: the step function returns additional info if true
            """
            self.eval_info_enabled = val


    def reset_plane(self, plane: Plane):
        """
        Resets the given plane using the settings of this environment.

        :param plane: the plane will be reset in place.
        """

        # Freeing resources on used edge
        if plane.edge != -1:
            self.network.edges[plane.edge].load -= plane.size


        # TODO: proper set up of start and target + generate speed from a list for the plane
        # Reset plane in place
        speed = 100
        start = np.random.randint(self.network.n_nodes)
        target = np.random.randint(self.network.n_nodes)
        plane.reset(start=start, target=target, shortest_path_weight=self.network.shortest_paths_weights[start][target] , speed=speed)
        
    ## Skipping the __str__ function for now

    def reset(self):
        """
        ...
        Adding a 'load' attribute to edges.
        """

        self.agent_steps = np.zeros(self.n_planes)
        self.network.reset(self.adj_mat, self.dist_mat)     # This maybe shouldn't be here. -> will result in removing the reset function from the Network class and moving the code from there into the init
        
        for edge in self.network.edges:
            # add new "load" attribude to edges 
            edge.load = 0       # Set the attribute to 0
        
        # Generate planes
        self.planes = []
        for i in range(self.n_planes):
            new_plane = Plane(i)
            self.reset_plane(new_plane)
            self.planes.append(new_plane)

        return self.get_observation(), self.get_plane_adjacency()
    
    def render(self):
        """
        Renders the airspace
        """
        pass

    def get_plane_adjacency(self):
        return self.network.adj_mat
    
    def get_node_observation(self) -> np.ndarray:
        """
        Get the node observation for each node in the airspace.

        :return: node observations of shape (num_waypoints, node_observtaion_size)
        """

        observations = []
        for j in range(self.network.n_nodes):
            observation = []

            observation += one_hot_list(j, self.network.n_nodes) 

            # Info of a waypoint
            num_planes = 0
            total_load = 0
            for i in range(self.n_planes):
                if self.planes[i].now == j and self.planes[i].edge == -1:
                    num_planes += 1
                    total_load += self.planes[i].load

            observation.append(num_planes)
            observation.append(total_load)


            # Edge info
            for k in self.network.nodes[j].edges:
                other_node = self.network.edges[k].get_other_node(j)

                observation += one_hot_list(other_node, self.network.n_nodes)
                observation.append(self.network.edges[k].length)
                observation.append(self.network.edges[k].load)


            observations.append(observation)

        return np.array(observations, dtype=np.float32)


    def get_node_aux(self):
        """
        Auxilary targets for each node in the network.

        :return: Auxilary targets of shape (num_nodes, node_aux_target_size)
        """
        aux = []
        for j in range(self.network.n_nodes):
            aux_j = []

            # For routing, it is essential for a node to estimate the distance to other nodes
            # -> auxilary target is length of shortest paths to all nodes
            for k in range(self.network.n_nodes):
                aux_j.append(self.network.shortest_paths_weights[j][k])

            aux.append(aux_j)

        return np.array(aux, dtype=np.float32)

    def get_node_agent_matrix(self):
        """
        Gets a matrix that indicates where agents are located,
        matrix[n, a] = 1 if agent a is on node n and 0 otherwise.

        :return: the node agent matrix of shape (n_nodes, n_agents)
        """
        node_agent = np.zeros((self.network.n_nodes, self.n_planes), dtype=np.int8)
        for a in range(self.n_planes):
            node_agent[self.planes[a].now, a] = 1

        return node_agent

    def get_observation(self):
        """
        Pass
        """
        observations = [] 

        # Leaving out global for now
        #...
        # -

        for i in range(self.n_planes):  # Get for each plane
            observation = []

            if self.env_var == EnvironmentVariant.GLOBAL:
                # for the global observation
                nodes_adjacency = self.get_nodes_adjacency().flatten()
                node_observation = self.get_node_observation().flatten()
                global_obs = np.concatenate((nodes_adjacency, node_observation))


            # Plane observation
            observation += one_hot_list(self.planes[i].now, self.network.n_nodes)       # Places 1 on the index of the node where the plane is
            observation += one_hot_list(self.planes[i].target, self.network.n_nodes)    # Places 1 on the index of the target node

            # Planes should know where they are coming from when traveling on an edge
            observation.append(int(self.planes[i].edge != -1))  # Check if the plane is on an edge
            if self.planes[i].edge != -1:   # Is it is on an edge
                other_node = self.network.edges[self.planes[i].edge].get_other_node(self.planes[i].now)     # Get to where it is going 
            else:
                other_node = -1     # In a waypoint
            observation += one_hot_list(other_node, self.network.n_nodes)   # Again, encode the position in a vector - if it is in a waypoint, it will create a 0 vector

            observation.append(self.planes[i].time)
            observation.append(self.planes[i].size) #TODO: WE WOULD LIKE TO ADD SPEED HERE!!!
            observation.append(self.planes[i].id)



            # Edge information
            for j in self.network.nodes[self.planes[i].now].edges:
                other_node = self.network.edges[j].get_other_node(self.planes[i].now)

                observation += one_hot_list(other_node, self.network.n_nodes)
                observation.append(self.network.edges[j].length)
                observation.append(self.network.edges[j].load)

                # Here is missing a big chunk of commmented code that involed appending the shortest path weights

            # Other data
            count = 0
            self.planes[i].neigh = []
            self.planes[i].neigh.append(i)
            for j in range(self.n_planes):
                if j == i:
                    continue

                if (self.planes[j].now in self.network.nodes[self.planes[i].now].neighbors) | (self.planes[j].now == self.planes[i].now):   # If some other plane is in some neighboring waypoint or they are at the same waypoint

                    self.planes[i].neigh.append(j)  # Add the plane as a neighbor to the i-th plane

                    # Add neighbor information in observation (until k neighbors)
                    if self.env_var == EnvironmentVariant.WITH_K_NEIGHBORS and count < self.k:
                        count += 1
                        observation.append(self.planes[j].now)
                        observation.append(self.planes[j].target)
                        observation.append(self.planes[j].edge)
                        observation.append(self.planes[j].size)
                        observation.append(self.planes[i].id)

            if self.env_var == EnvironmentVariant.WITH_K_NEIGHBORS:
                for j in range(self.k - count):
                    for _ in range(5):
                        observation.append(-1) # invalid placeholder

            observation_numpy = np.array(observation)

            # Add global information
            if self.env_var == EnvironmentVariant.GLOBAL:
                observation_numpy = np.concatenate((observation_numpy, global_obs))

            observations.append(observation_numpy)

        return np.array(observations, dtype=np.float32)


    def step(self, act):

        reward = np.zeros(self.n_planes, dtype=np.float32)
        looped = np.zeros(self.n_planes, dtype=np.float32)
        done = np.zeros(self.n_planes, dtype=bool)
        # ... excluding drop packet 
        success = np.zeros(self.n_planes, dtype=bool)
        blocked = 0 # ???

        delays = []
        delays_arrived = []
        spr = []
        self.agent_steps += 1

        # Could shuffle planes so planes with lower id number are not preffered

        # Handle actions
        for i in range(self.n_planes):
            # agent i control plane i
            plane = self.planes[i]

            # eval)info_enables leaving out for now


            # Select outgoing edge 
            if plane.edge == -1:
                t = self.network.nodes[plane.now].edges[act[i]-1]   # TODO: is this correct?

                # Note that planes that are handled earlier in this loop are prioritized

                if self.network.edges[t].load + plane.size > 1: # Not possible to take this edge
                    reward[i] -= 0.2
                    blocked += 1
                else:
                    # Take this edge
                    plane.edge = t      # Begin traversal of this edge
                    plane.time = self.network.edges[t].length       # TODO: adjust it based on the speed of the plane: plane.time = self.network.edges[t].length/ plane.speed

                    # Assign load to the selected edge
                    self.network.edges[t].load += plane.size

                    # Already set the next position
                    plane.now = self.network.edges[t].get_other_node(plane.now)
                    if plane.now in plane.visited_nodes:
                        looped[i] = 1
                    else:
                        plane.visited_nodes.add(plane.now)
            else:
                pass # Plane already in transit, don't do anything (for now)

        
        #  if self.eval_info_enabled:
        #     total_edge_load = 0
        #     occupied_edges = 0
        #     packets_on_edges = 0
        #     total_packet_size = 0
        #     packet_sizes = []

        #     for edge in self.network.edges:
        #         total_edge_load += edge.load
        #         if edge.load > 0:
        #             occupied_edges += 1

        #     for i in range(self.n_data):
        #         packet = self.data[i]
        #         if packet.edge != -1:
        #             self.sum_packets_per_edge[packet.edge] += 1

        #         total_packet_size += packet.size
        #         packet_sizes.append(self.data[i].size)
        #         if packet.edge != -1:
        #             packets_on_edges += 1

        #     packet_distances = list(
        #         map(
        #             lambda p: self.network.shortest_paths_weights[p.now][p.target],
        #             self.data,
        #         )
        #     )

        # Then simulate in-flight planes (=> effect of actions)
        for i in range(self.n_planes):
            plane = self.planes[i]

            if plane.edge != -1: # Plane on an edge
                plane.time -=1
                # Plane finished traversing the edge
                if plane.time <= 0:
                    self.network.edges[plane.edge].load -= plane.size   # Remove the plane from the edge
                    plane.edge = -1     # Plane not on an edge
            

            if self.enable_action_mask:     # If action mask is enabled, we will mask the actions of planes
                if plane.edge != -1:    # If on an edge
                    self.action_mask[i] = 0     # No actions are valid
                else:
                    """
                    This is the old implementation.
                    """
                    # self.action_mask[i, 0] = 1    # The plane is no longer allowed to stay in place
                    # for edge_i, e in enumerate(self.network.nodes[plane.now].edges):
                    #     self.action_mask[i, edge_i] = self.network.edges[e].get_other_node(plane.now) in plane.visited_nodes

                    avail_edges = self.network.nodes[plane.now].edges
                    numb_avail_edges = len(avail_edges)
                    self.action_mask[i] = 0
                    self.action_mask[i][:numb_avail_edges] = 1

                    # We do not want to revisit waypoints
                    for edge, edge_idx in enumerate(avail_edges):
                        next_node = self.network.edges[edge_idx].get_other_node(plane.now)
                        if next_node in plane.visited_nodes:
                            self.action_mask[i][edge] = 0   # Mask the visited node - the plane cannot go back there



                    # If no valid actions are possible, we gotta do something:          # TODO: is this the correct way to hand this?
                    if not self.action_mask[i].any():
                        reward[i] -= 5.0
                        done[i] = True
                        # Optional reset of the plane
                        self.reset_plane(plane)
                        continue


            # The plane has reached the target
            has_reached_target = plane.edge == -1 and plane.now == plane.target     # If not on an edge and at the target waypoints 
            if has_reached_target:
                reward[i] += 10 
                done[i] = True
                success[i] = True

                # We need at least 1 step if we spawn at the target
                opt_distance = max(plane.shortest_path_weight, 1)

                
                # insert delays before resetting pakcets
                delays_arrived.append(self.agent_steps[i])
                spr.append(self.agent_steps[i]/opt_distance)

                
                delays.append(self.agent_steps[i])
                self.agent_steps[i] = 0
                self.reset_plane(plane)
                continue

            # I think this should be here
            if not done[i]:
                self.agent_steps[i] += 1
        



        observations = self.get_observation()
        adj = self.get_data_adjacency()

        # Info
        info = {
            "delays": delays,
            "delays_arrived": delays_arrived,
            # shortest path ratio in [1, inf) where 1 is optimal
            "spr": spr,
            "looped": looped.sum(),
            "throughput": success.sum(),
           # "dropped": (done & ~success).sum(),
            #"blocked": blocked,
        }

        return observations, adj, reward, done, info 
                        
        
        

    def get_data_adjacency(self):
        """
        Get an adjacency matrix for planes of shape (n_planes, n_planes)
        where the second dimension contains the neighbors of the agents in the first dimension -> (plane, neighbors)
        """

        adj = np.eye(self.n_planes, self.n_planes, dtype=np.int8)
        for i in range(self.n_planes):
            for n in self.planes[i].neigh:
                if n != -1:
                    adj[i, n] = 1
        return adj

    def get_num_agents(self):
        return self.n_planes

    def get_num_nodes(self):
        return self.network.n_nodes
