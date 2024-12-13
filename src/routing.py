import textwrap
import numpy as np
from collections import defaultdict
import networkx as nx
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from environment import EnvironmentVariant, NetworkEnv
from gymnasium.spaces import Discrete

from network import Network
from util import one_hot_list

from matplotlib import animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

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
        self.size = None   
        self.start = None
        self.time = None
        self.edge = -1
        self.neigh = None
        self.shortest_path_weight = None
        self.visited_nodes = None
        self.speed = None       # Added
        self.on_edge = None
        self.run = -1
        self.paths = []
        self.trajectory = []
        self.last_node = None
        self.targets = []


    def reset(self, start, target, shortest_path_weight, speed):
        self.now = start
        self.target = target
        self.start = start
        self.size = 0.5     
        self.time = 0
        self.edge = -1
        self.neigh = [self.id]
        self.speed = speed
        self.shortest_path_weight = shortest_path_weight
        self.visited_nodes = set([start])
        self.last_node = -1
        self.targets.append(target)
    
class Routing(NetworkEnv):
    """
        Changes are:
            - data is renamed to planes
            - n_data is renamed to n_planes
            - added two additional params: adj_mat and dist_mat
    """


    def __init__(self, network: Network, n_planes, env_var: EnvironmentVariant,  adj_mat, dist_mat , k = 3, enable_action_mask=False, spp=5) -> None:
        super(Routing, self).__init__()

        self.network = network  # This is the network environment that has the graph with nodes and edges
        self.n_planes = n_planes    #   Number of planes
        self.planes = []    # List of planes

        self.old_node_plane_ids = []
        self.new_node_plane_ids = []

        self.spp = spp  # This paramter contorls the shortest path that can occur between start and target of a plane

        self.special_nodes = [114, 115, 116, 91, 23, 79, 18, 12, 4, 0, 93, 104, 117, 100, 106, 88, 27]  # Airports

        """
        Choose how will the info gathered
            - 1 just from an agent
            - 2 from neighbors
            - 3 from the whole network

        Advice:
            Do not select thrird option for large graphs(50+). 
            Single vector of observation for a single agent will be of length 200k+ for 118 nodes.
        """
        self.env_var = EnvironmentVariant(env_var) 


        # Basics
        self.adj_mat = adj_mat
        self.dist_mat = dist_mat

        # k neighbors to aggregate information from
        self.k = k

        # Log information
        self.agent_steps = np.zeros(self.n_planes)

        # Evaluation metrics
        self.distance_map = defaultdict(list)
        self.sum_planes_per_node = None
        self.sum_planes_per_edge = None

        self.enable_action_mask = enable_action_mask
        self.action_mask = None
        self.action_space = None
        self.eval_info_enabled = False


    def set_action_mask(self):
        """
            Activates the action mask. Agent will have the visited nodes masked out. 
        """
        self.enable_action_mask = True

    def set_eval_info(self, val):
        """
            Whether the step function should return additional info for evaluation.

            :param val: the step function returns additional info if true
        """
        self.eval_info_enabled = val

    def set_valid_target(self, valid_range, start):
        """
            Selects a valid target for the plane. It is different from start and it is at least self.spp hops away from start
        """
        while True:
            target = np.random.choice(valid_range, size=1, replace=False)[0]
            path_to_target = len(self.network.shortest_paths[start][target])
            if self.spp < path_to_target:
                break
        return target
    
    def set_valid_start(self, valid_range):
        """
            Selects a valid start for the plane. Ensures that maximum of 2 planes can spawn at the same airport.
        """
        while True:
            start = np.random.choice(valid_range, size=1, replace=False)[0]
            start_occupied = 0
            for plane in self.planes:
                if plane.now == start:
                    start_occupied += 1
            if start_occupied < 3:
                break
        return start

    def reset_plane(self, plane: Plane):
        """
            Resets the given plane using the settings of this environment.

            :param plane: the plane will be reset in place.
        """
        # Freeing resources on used edge    
        if plane.edge != -1:
            self.network.edges[plane.edge].load -= plane.size

        # Reset plane in place
        speed = 5
        range_values = np.array(self.special_nodes)
        

        # Reset plane
        if plane.start is None and plane.target is None:    # First init
            start = self.set_valid_start(range_values)
            target = self.set_valid_target(range_values, start)
        else:
            start = plane.target  # New start is the target from the last path
            target = self.set_valid_target(range_values, start)
            
        plane.reset(start=start, target=target, shortest_path_weight=self.network.shortest_paths_weights[start][target], speed=speed)
        if self.eval_info_enabled:
            plane.paths.append([])
            plane.run += 1

        if self.enable_action_mask:            
            # Allow all links
            self.action_mask[plane.id] = 0
        
    def reset(self):
        """
        This resets the environment together with agents.
        Adding a 'load' attribute to edges.
        """

        self.agent_steps = np.zeros(self.n_planes)
        self.network.reset(self.adj_mat, self.dist_mat)     # Initialize the underlying graph

        self.action_mask = np.zeros((self.n_planes, self.network.max_neighbors), dtype=bool)
        self.action_space = Discrete(self.network.max_neighbors, start=1)
        
        for edge in self.network.edges:
            # Adds new "load" attribude to edges 
            edge.load = 0       # Set the attribute to 0

        if self.eval_info_enabled:
            self.sum_planes_per_node = np.zeros(self.network.n_nodes)
            self.sum_planes_per_edge = np.zeros(len(self.network.edges))
        
        # Generate planes
        self.planes = []
        for i in range(self.n_planes):
            new_plane = Plane(i)    #  Begin with id = 0
            self.reset_plane(new_plane)     # Reset plane - set source, target, speed
            
            # TODO: Check spawning points
            self.planes.append(new_plane)   # Add plane to the list of planes 

        return self.get_observation(), self.get_plane_adjacency()
    
    def get_node_observation(self):
        """
            Get the node observation for each node in the airspace.

            :return: node observations of shape (num_waypoints, node_observation_size)
        """
        observations = []   # All observataions
        for j in range(self.network.n_nodes):   # Go through all the waypoints
            observation = []    # Observation of a single node

            """
                - one_hot_list creates a list that contains 1 at the positions j
                    [0,...,1,0,..,n_nodes-1]
            """
            observation += one_hot_list(j, self.network.n_nodes)    # This is equivalent to .extend()

            # Info of a waypoint
            num_planes = 0
            total_load = 0  # Total load of planes on the waypoint
            for i in range(self.n_planes):  # Go through every plane
                if self.planes[i].now == j and self.planes[i].edge == -1:
                    num_planes += 1
                    total_load += self.planes[i].size

            observation.append(num_planes)  # Add the info into the observation
            observation.append(total_load)  # Add the info into the observation

            # Edge info
            for k in self.network.nodes[j].edges:   # Go through every edge that is connected to the waypoint
                other_node = self.network.edges[k].get_other_node(j)    # Get conncted node
                """
                This gathering i do not really understand, but i will trust the original authors.
                """
                observation += one_hot_list(other_node, self.network.n_nodes)   # Again create a list of zeros with 1 on the index of the other_node
                observation.append(self.network.edges[k].length)    # Add the info into the observation
                observation.append(self.network.edges[k].load)      # Add the info into the observation


            # Mask nodes with less than maximum neighbors 
            num_edges = len(self.network.nodes[j].edges)
            if num_edges != self.network.max_neighbors:
                for _ in range(self.network.max_neighbors-num_edges):
                    observation += one_hot_list(-1, self.network.n_nodes)
                    observation.append(-1)  # Invalid placeholder
                    observation.append(-1)  # Invalid placeholder


            observations.append(observation)    # Append the waypoint observation into all observations

        # The array will be of size (num_nodes, total_observation_size), total_observation_size is something like 1463
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
                aux_j.append(self.network.shortest_paths_weights[j][k])     # Append the shortest path from j to k

            aux.append(aux_j)

        
        # The output is of shape: (n_waypoints, n_waypoints) with the addition that diagonal is 0
        
        return np.array(aux, dtype=np.float32)

    def get_node_agent_matrix(self):
        """
            Gets a matrix that indicates where agents are located,
            matrix[n, a] = 1 if agent a is on node n and 0 otherwise.

            :return: the node agent matrix of shape (n_nodes, n_agents)
        """
        node_agent = np.zeros((self.network.n_nodes, self.n_planes), dtype=np.int8)
        for i in range(self.n_planes):
            node_agent[self.planes[i].now, i] = 1
        return node_agent

    def get_observation(self):
        """
            This function gathers observations for each PLANE.
        """
        observations = [] 

        if self.env_var == EnvironmentVariant.GLOBAL:
            # for the global observation
            nodes_adjacency = self.get_nodes_adjacency().flatten()
            node_observation = self.get_node_observation().flatten()
            global_obs = np.concatenate((nodes_adjacency, node_observation))

        for i in range(self.n_planes):  # Get for each plane
            observation = []    # Init observation

            # Plane observation
            observation += one_hot_list(self.planes[i].now, self.network.n_nodes)  
            observation += one_hot_list(self.planes[i].target, self.network.n_nodes)

            # Planes should know where they are coming from when traveling on an edge
            observation.append(int(self.planes[i].edge != -1))  # Check if the plane is on an edge
            if self.planes[i].edge != -1:   # If it is on an edge
                other_node = self.network.edges[self.planes[i].edge].get_other_node(self.planes[i].now)
            else:
                other_node = -1     # In a waypoint

            observation += one_hot_list(other_node, self.network.n_nodes) 
            observation.append(self.planes[i].time)
            observation.append(self.planes[i].size) 
            observation.append(self.planes[i].id)

            
            # Edge information
            for j in self.network.nodes[self.planes[i].now].edges:  # Go through all edges of i-th node
                other_node = self.network.edges[j].get_other_node(self.planes[i].now)
                observation += one_hot_list(other_node, self.network.n_nodes)
                observation.append(self.network.edges[j].length)
                observation.append(self.network.edges[j].load)

            # Mask nodes with less than maximum neighbors
            num_edges = len(self.network.nodes[self.planes[i].now].edges)
            if num_edges != self.network.max_neighbors:
                for i in range(self.network.max_neighbors-num_edges):
                    observation += one_hot_list(-1, self.network.n_nodes)
                    observation.append(-1)  # Invalid placeholder
                    observation.append(-1)



            # Other data
            count = 0
            self.planes[i].neigh = []
            self.planes[i].neigh.append(i)
            for j in range(self.n_planes):  # For each plane, go through every other plane
                if j == i:
                    continue
                # If some other plane is in some neighboring waypoint or they are at the same waypoint
                if (self.planes[j].now in self.network.nodes[self.planes[i].now].neighbors) | (self.planes[j].now == self.planes[i].now):
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
        """
        What is act? 
            - act is a vector of shape (n_planes)
            - each number is the index of the edge they have chosen for traversal
        """
        reward = np.zeros(self.n_planes, dtype=np.float32)  # Define reward for each plane
        looped = np.zeros(self.n_planes, dtype=np.float32)
        done = np.zeros(self.n_planes, dtype=bool)          # Done planes
        success = np.zeros(self.n_planes, dtype=bool)       # Succesful planes
        drop_plane = np.zeros(self.n_planes, dtype=bool)
        shortest_edges = np.zeros(self.n_planes, dtype=bool)
        blocked = 0 # Blocked planes

        delays = [] # Dunno about this
        delays_arrived = []
        spr = []
        self.agent_steps += 1

        # Update old_node_plane_ids for q_values masking
        self.old_node_plane_ids = [plane.now for plane in self.planes]
        
        # Collect positions of planes
        if self.eval_info_enabled:
            for i in range(self.n_planes):
                plane = self.planes[i]

        # Planes are shuffled to not prefer planes with lower ids
        random_plane_order = np.arange(self.n_planes)
        np.random.shuffle(random_plane_order)
        
        # Handle actions
        for i in random_plane_order:
            # agent i controls plane i
            plane = self.planes[i]
            
            # Eval info
            if self.eval_info_enabled:
                plane.trajectory.append(plane.now)  # Remember trajectory of a plane
                if plane.edge == -1:
                    self.sum_planes_per_node[plane.now] += 1

            # Select outgoing edge 
            if plane.edge == -1: #and act[i] != 0:    # If at a waypoint

                t = self.network.nodes[plane.now].edges[act[i]] #-1   # Select an outgoing edge based on policy

                if self.network.edges[t].load + plane.size > 1.5: 
                    reward[i] -= 10.0
                    blocked += 1
                    shortest_edges[i] = False
                    # print("I am BLOCKED")
                else:
                    # Take this edge
                    plane.edge = t      # Begin traversal of this edge
                    plane.time = self.network.edges[t].length/plane.speed   

                    # Assign load to the selected edge
                    self.network.edges[t].load += plane.size

                    # Take next node 
                    next_node = self.network.edges[t].get_other_node(plane.now)
                    
                    # Remember last node
                    plane.last_node = plane.now

                    dist_to_target = self.compute_distance(plane.now, plane.target)
                    dist_after_step = self.compute_distance(next_node, plane.target)

                    # Subrewards
                    if next_node == self.network.shortest_paths[plane.now][plane.target][1]:
                        shortest_edges[i] = True
                        reward[i] += 3.0    # 5.0
                    else:
                        shortest_edges[i] = False
                        progres = dist_to_target - dist_after_step
                        if progres > 0:
                            reward[i] += 1.5 * np.exp(0.002 * progres) 
                    
                    # Time penalty
                    reward[i] -= 0.01

                    if dist_to_target < 250:
                        for edge in self.network.nodes[plane.now].edges:
                            if self.network.edges[edge].get_other_node(plane.now) == plane.target and next_node != plane.target:
                                reward[i] -= 1.0
                       
                    # Already set the next position
                    plane.now = next_node

                    if plane.now in plane.visited_nodes:
                        looped[i] = 1 
                        reward[i] -= 2.0
                    else:
                        plane.visited_nodes.add(plane.now)
                        reward[i] += 2.5

        # INFO
        if self.eval_info_enabled:
            """
                This is gathers basic information about the current state of the Routing network
                    - this data is gathered only during evaluation phase
            """
            total_edge_load = 0
            occupied_edges = 0
            planes_on_edges = 0
            total_plane_size = 0
            plane_sizes = []

            # Gather information about edges 
            for edge in self.network.edges:
                if edge.load > 0:
                    total_edge_load += edge.load
                    occupied_edges += 1


            # Gather information from planes
            for i in range(self.n_planes):
                plane = self.planes[i]
                if plane.edge != -1:
                    self.sum_planes_per_edge[plane.edge] += 1
                    planes_on_edges += 1

                total_plane_size += plane.size
                plane_sizes.append(self.planes[i].size)
                    

            plane_distances = list(
                map(
                    lambda p: self.network.shortest_paths_weights[p.now][p.target],
                    self.planes,
                )
            )

        # In-flight planes (=> effect of actions)
        for i in range(self.n_planes):
            plane = self.planes[i]

            if plane.edge != -1: # Plane on an edge
                plane.time -= 0.5 # This number should be adjusted based on the calcuted time from actions
                if shortest_edges[i]:
                    reward[i] += 0.2
                else:
                    reward[i] -= 0.1

                # Plane finished traversing the edge
                if plane.time <= 0:
                    self.network.edges[plane.edge].load -= plane.size   # Remove the plane from the edge
                    plane.edge = -1     # Plane not on an edge
            
            drop_plane[i] = drop_plane[i]
            if self.enable_action_mask:     # If action mask is enabled, we will mask the actions of planes
                if plane.edge != -1:    # If on an edge
                    self.action_mask[i] = 0
                else:
                    for edge_i, e in enumerate(self.network.nodes[plane.now].edges):
                         self.action_mask[i, edge_i] = self.network.edges[e].get_other_node(plane.now) in plane.visited_nodes   # If it is, place 1
                    # Check if plane can fly anywhere
                    boundary = len(self.network.nodes[plane.now].edges)
                    if self.action_mask[i].sum() == boundary:
                        drop_plane[i] = True
                        print(f"EMERGENCY LANDING for plane {plane.id}")

            # The plane has reached the target
            has_reached_target = plane.edge == -1 and plane.now == plane.target     # If not on an edge and at the target waypoint            
            if has_reached_target or drop_plane[i]:
                if has_reached_target == False:
                    reward[i] += -20
                done[i] = True
                success[i] = has_reached_target

                # We need at least 1 step if we spawn at the target
                opt_distance = max(plane.shortest_path_weight, 1)
                
                # Insert delays before resetting planes
                if success[i]:
                    plane.trajectory.append(plane.now)
                    delays_arrived.append(self.agent_steps[i])
                    spr_ratio = self.agent_steps[i]/opt_distance*plane.speed
                    spr.append(spr_ratio)
                    reward[i] += 10.0 * (1+ 1/spr_ratio)    # Incraesed reward for shortest path

                    if self.eval_info_enabled:
                        self.distance_map[opt_distance].append(self.agent_steps[i])
                        print("Plane ", plane.id, " reached the target.")

                # Reset plane and metrics after finishing route
                delays.append(self.agent_steps[i])
                self.agent_steps[i] = 0
                self.reset_plane(plane)
                
        observations = self.get_observation()   # Get plane observations
        adj = self.get_plane_adjacency() 


        # Info
        info = {
            "delays": delays,
            "delays_arrived": delays_arrived,
            # shortest path ratio in [1, inf) where 1 is optimal
            "spr": spr,
            "looped": looped.sum(),
            "throughput": success.sum(),
            "dropped": (done & ~success).sum(),
            "blocked": blocked,
        }


        if self.eval_info_enabled:
            info.update(
                {
                    "total_edge_load": total_edge_load,
                    "occupied_edges": occupied_edges,
                    "planes_on_edges": planes_on_edges,
                    "total_plane_size": total_plane_size,
                    "plane_sizes": plane_sizes,
                    "plane_distances": plane_distances,
                }
            )

        # Update new_node_plane_ids for q_values masking
        self.new_node_plane_ids = [plane.now for plane in self.planes]

        return observations, adj, reward, done, info
                        
    def get_nodes_adjacency(self):
        return self.network.adj_mat
    
    def get_plane_adjacency(self):
        """
        Get an adjacency matrix for planes of shape (n_planes, n_planes),
            where the second dimension contains the neighbors of the agents in the first dimension -> (plane, neighbors)
        """
        adj = np.eye(self.n_planes, self.n_planes, dtype=np.int8)
        for i in range(self.n_planes):
            for j in self.planes[i].neigh:
                if j != -1:
                    adj[i, j] = 1
        return adj

    def get_num_agents(self):
        return self.n_planes

    def get_num_nodes(self):
        return self.network.n_nodes
    
    def compute_distance(self, idx1, idx2):
        """
            Compute airdistance between two nodes.
        """
        node_1 = self.network.nodes[idx1]
        node_2 = self.network.nodes[idx2]
        return ((node_1.x-node_2.x)**2 + (node_1.y -node_2.y)**2)**0.5

    def get_final_info(self, info: dict):
        agent_steps = self.agent_steps
        for agent_step in agent_steps:
            if agent_step != 0:
                info["delays"].append(agent_step)
        return info

    def plot_trajectory(self):
        """
            Plots the trajectory of each plane.
        """
        
        special_nodes = [114, 115, 116, 91, 23, 79, 18, 12, 4, 0, 93, 104, 117, 100, 106, 88, 27]
        special_labels = {
            12: "Berlin",
            117: "Copenhagen",
            0: "Vienna",
            27: "Warsaw",
            4: "Prague",
            104: "Budapest",
            91: "Amsterdam",
            18: "Munich",
            23: "Zurich",
            88: "Bucharest",
            93: "Zagreb",
            115: "Milan",
            79: "Frankfurt",
            116: "Brussels",
            100: "Belgrade",
            114: "Rome",
            106: "Sarajevo"
        }

        planes_colors = {plane: f"C{i}" for i, plane in enumerate(self.planes)}
        
        pos = {node: coord for node, coord in zip(self.network.G.nodes, self.network.coordinates)}
        
        fig, ax = plt.subplots(figsize=(10, 8))

        img = mpimg.imread('data/map.png')
        ax.imshow(img, extent=[min(x[0] for x in pos.values()), max(x[0] for x in pos.values()), 
                               min(x[1] for x in pos.values()), max(x[1] for x in pos.values())], alpha=0.05)
        
        planes_trajectories = {plane: [] for plane in self.planes}
        
        node_colors = ["#DAA520" if node in special_nodes else "grey" for node in self.network.G.nodes]
        node_size = [100 if node in special_nodes else 100 for node in self.network.G.nodes]
        node_alpha = [1 if node in special_nodes else 0.3 for node in self.network.G.nodes]   
        
        nx.draw_networkx_nodes(self.network.G, pos, node_color=node_colors,ax=ax, alpha=node_alpha, node_size=node_size)
        nx.draw_networkx_labels(self.network.G, pos, labels=special_labels, font_size=7, ax=ax)
        nx.draw_networkx_edges(
            self.network.G,
            pos,
            edge_color="black",
            ax=ax,
            alpha=0.3,  # Uniform alpha for edges
            width=1
        )
        
        
        legend_lines = [ax.plot([], [], color=color, label=f'Plane: {plane.id}', linewidth=1)[0] for plane, color in planes_colors.items()]
        
        ax.legend()

        # Create a list to hold the annotation boxes for planes 
        plane_texts = {plane: None for plane in self.planes}
        targets_reached = [0 for _ in range(self.n_planes)]
        
        temp_texts = []

        def update_trajectory_render(frame):
            ax.clear()
            ax.imshow(img, extent=[min(x[0] for x in pos.values()) - 200 , max(x[0] for x in pos.values())-200, 
                               min(x[1] for x in pos.values())-50, max(x[1] for x in pos.values())-50], alpha=0.05)
            nx.draw_networkx_nodes(self.network.G, pos, node_color=node_colors, ax=ax,  alpha=node_alpha, node_size=node_size)
            nx.draw_networkx_labels(self.network.G, pos, labels=special_labels, font_size=7, ax=ax)
                # Draw edges
            nx.draw_networkx_edges(
                self.network.G,
                pos,
                edge_color="black",
                ax=ax,
                alpha=0.3,  # Uniform alpha for edges
                width=1
            )
                
            legend_lines = [ax.plot([], [], color=color, label=f'Plane: {plane.id}', linewidth=1)[0] for plane, color in planes_colors.items()]
            ax.legend()
            
            for i, plane in enumerate(self.planes):
                color = planes_colors[plane]
            
            for text in temp_texts:
                text.remove()
            temp_texts.clear()
            
            for i, plane in enumerate(self.planes):
                color = planes_colors[plane]
                
                if 0 < frame < len(plane.trajectory):
                    current_edge = plane.trajectory[frame - 1], plane.trajectory[frame]
                    planes_trajectories[plane].append(current_edge)

                    end_pos =  self.network.nodes[plane.trajectory[frame]].x, self.network.nodes[plane.trajectory[frame]].y

                    # Add new text with Unicode airplane symbol
                    plane_texts[plane] = ax.text(
                    end_pos[0],
                    end_pos[1],
                    '\u2708',  # Unicode airplane symbol ✈
                    fontsize=40,  # Adjust as needed
                    color=color,
                    ha='center',
                    va='center',
                    )
                    if plane.trajectory[frame] == plane.targets[targets_reached[i]]:
                        targets_reached[i] += 1
                        text = ax.text(
                            end_pos[0],
                            end_pos[1] - 75,
                            f"Plane {plane.id} landed.",
                            fontsize=15,
                            color='black',
                            ha='center',
                            va='bottom'
                        )
                        temp_texts.append(text)


                
                for i, edge in enumerate(planes_trajectories[plane]):
                    alpha = max(0.1, 1 - (frame - i) / 50)  # Fading factor (adjust "5" for longer trails)
                    line_width = max(1, 5 - (frame - i) / 2)  # Adjust "3" for initial width and "2" for fading speed
                    nx.draw_networkx_edges(self.network.G, pos, edgelist=[edge], edge_color=color, alpha=alpha, width=line_width, ax=ax)
            
                if frame < len(plane.trajectory):
                    current_node = plane.trajectory[frame]
                    nx.draw_networkx_nodes(self.network.G, pos, nodelist=[current_node], node_color=color, ax=ax,node_size=1)
                
            return list(plane_texts.values())
        
        num_frames = max(len(plane.trajectory) for plane in self.planes)
        self.animation = FuncAnimation(fig, update_trajectory_render, frames=num_frames, interval=500, repeat=False)
        
        # self.animation.save("planes_trajectory_animation.mp4", writer="ffmpeg") ## possibly save the animation
        
        plt.title("Planes Trajectory Animation")
        plt.show()

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""\
            Routing environment with parameters
            > Network: {self.network.n_nodes} nodes
            > Number of planes: {self.n_planes}
            > Environment variant: {self.env_var.name}
            > Number of considered neighbors (k): {self.k if self.env_var == EnvironmentVariant.WITH_K_NEIGHBORS else "disabled"}
            > Action mask: {self.enable_action_mask}\
            """
        )
