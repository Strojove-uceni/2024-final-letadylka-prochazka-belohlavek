import textwrap
import numpy as np
from collections import defaultdict
import networkx as nx
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import json


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
        #self.checked_looped = None
        # TODO ADD RESET if stuck


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
        #self.check_looped = (start,0) # (ID NODE, #Steps of looped)



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        elif isinstance(obj, np.generic):
            return obj.item()  # Convert NumPy scalars to native Python types
        return super().default(obj)
    


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
        self.planes = []    # It will be a list of planes

        self.old_node_plane_ids = []
        self.new_node_plane_ids = []

        self.spp = spp


        """
        Choose how will the info gathered
            - 1 just from an agent
            - 2 from neighbors
            - 3 from the whole network
        """
        self.env_var = EnvironmentVariant(env_var) 


        # Basics
        self.adj_mat = adj_mat
        #self.amount_of_neighbors = 10 
        self.dist_mat = dist_mat

        # k neighbors
        self.k = k

        # Log information
        self.agent_steps = np.zeros(self.n_planes)

        # Map shortest path to actual agents steps
        self.distance_map = defaultdict(list)
        self.sum_planes_per_node = None
        self.sum_planes_per_edge = None

        self.enable_action_mask = enable_action_mask
        self.action_mask = None # np.zeros((n_planes, self.amount_of_neighbors), dtype=bool)


        """
        It is set like this, because the original had Discrete(4, start = 0). That is because the original could have had packets waiting within a router. Unfortunately, we cannot have a plane waiting in a waypoint. 
            Maybe we could argue that is flying in a circle, but we will try it like this now.
        """
        # The number of choices should be equal to the maximum number of neighbors a node can have
        # in our case it is quite simple as every node as a static number of neighbors, at least for now
        self.action_space = None #Discrete(self.amount_of_neighbors, start=1) # {0,1,2, ..., 10} using gym action space

        self.eval_info_enabled = False


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

    def reset_plane(self, plane: Plane):
        """
        Resets the given plane using the settings of this environment.

        :param plane: the plane will be reset in place.
        """

        # Freeing resources on used edge    
        if plane.edge != -1:
            self.network.edges[plane.edge].load -= plane.size

        # Reset plane in place
        speed = 3
        range_values = np.arange(self.network.n_nodes- 17, self.network.n_nodes, dtype=int)
        

        # Reset plane
        if plane.start is None and plane.target is None:    # First init
            start = np.random.choice(range_values, size=1, replace=False)[0]
            target = self.set_valid_target(range_values, start)
        else:
            start = plane.target  # New start is the target from the last path
            target = self.set_valid_target(range_values, start)
            
        plane.reset(start=start, target=target, shortest_path_weight=self.network.shortest_paths_weights[start][target], speed=speed)
        if self.eval_info_enabled:
            plane.paths.append([])
            plane.run +=1

        if self.enable_action_mask:            
            # Allow all links
            self.action_mask[plane.id] = 0
            #self.action_mask[plane.id, 0] = plane.now != plane.target
        
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
    
    def render(self):
        """
        Renders the airspace
        """
        self.network.render(self.planes)

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
            # observation += self.network.embeddings[j]   # Get dense representation instead of a sparse onehot

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
                # observation += self.network.embeddings[other_node]
                observation.append(self.network.edges[k].length)    # Add the info into the observation
                observation.append(self.network.edges[k].load)      # Add the info into the observation


            # Mask nodes with less than maximum neighbors 
            num_edges = len(self.network.nodes[j].edges)
            if num_edges != self.network.max_neighbors:
                for _ in range(self.network.max_neighbors-num_edges):
                    observation += one_hot_list(-1, self.network.n_nodes)
                    # observation += one_hot_list(-1, 32)
                    observation.append(-1)
                    observation.append(-1)


            observations.append(observation)    # Append the waypoint observation into all observations
        # The array will be of size (num_nodes, total_observation_size), total_observation_size is something like 1463
        return np.array(observations, dtype=np.float32) # Transform this into a numpy array
    
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

        """
        The output is of shape: (n_waypoints, n_waypoints) with the addition that diagonal is 0
        """
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
            #print(one_hot_list(self.planes[i].now, self.network.n_nodes))
            observation += one_hot_list(self.planes[i].now, self.network.n_nodes)       # Places 1 on the index of the node where the plane is
            observation += one_hot_list(self.planes[i].target, self.network.n_nodes)    # Places 1 on the index of the target node
            # observation += self.network.embeddings[self.planes[i].now]
            # observation += self.network.embeddings[self.planes[i].target]

            # Planes should know where they are coming from when traveling on an edge
            observation.append(int(self.planes[i].edge != -1))  # Check if the plane is on an edge
            if self.planes[i].edge != -1:   # If it is on an edge
                other_node = self.network.edges[self.planes[i].edge].get_other_node(self.planes[i].now)     # Get to where it is going 
                # observation += self.network.embeddings[other_node]
            else:
                other_node = -1     # In a waypoint
                # observation += one_hot_list(other_node, 32)

            observation += one_hot_list(other_node, self.network.n_nodes)   # Again, encode the position in a vector - if it is in a waypoint, it will create a 0 vector
            # observation += self.network.embeddings[other_node]

            observation.append(self.planes[i].time)
            observation.append(self.planes[i].size) # TODO: WE WOULD LIKE TO ADD SPEED HERE!!!
            observation.append(self.planes[i].id)

            
            # Edge information
            for j in self.network.nodes[self.planes[i].now].edges:  # Go through all edges of i-th node
                # print(len(self.network.nodes[self.planes[i].now].edges))
                other_node = self.network.edges[j].get_other_node(self.planes[i].now)

                observation += one_hot_list(other_node, self.network.n_nodes)
                # observation += self.network.embeddings[other_node]
                observation.append(self.network.edges[j].length)
                observation.append(self.network.edges[j].load)

                # Here is missing a big chunk of commmented code that involed appending the shortest path weights, authors had it commented


            # Mask nodes with less than maximum neighbors
            num_edges = len(self.network.nodes[self.planes[i].now].edges)
            if num_edges != self.network.max_neighbors:
                for i in range(self.network.max_neighbors-num_edges):
                    observation += one_hot_list(-1, self.network.n_nodes)
                    # observation += one_hot_list(-1, 32)
                    observation.append(-1)
                    observation.append(-1)



            # Other data
            count = 0
            self.planes[i].neigh = []
            self.planes[i].neigh.append(i)
            for j in range(self.n_planes):  # For each plane, go through every other plane
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
        """
        What is act? 
            - act is a vector of shape (n_planes)
            - each number is the index of the edge they have chosen for traversal
        """

        reward = np.zeros(self.n_planes, dtype=np.float32)  # Define reward for each plane
        looped = np.zeros(self.n_planes, dtype=np.float32)
        done = np.zeros(self.n_planes, dtype=bool)          # Done planes
        success = np.zeros(self.n_planes, dtype=bool)       # Succesful planes

        shortest_edges = np.zeros(self.n_planes, dtype=bool)
        blocked = 0 # 

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
                if plane.edge != -1:
                    plane.paths[plane.run].append(-1*plane.edge)    # Add node to plane path
                else:
                    plane.paths[plane.run].append(plane.now)               # Add node to plane path

        random_plane_order = np.arange(self.n_planes)
        np.random.shuffle(random_plane_order)
        
        # Handle actions
        for i in range(self.n_planes):
            # agent i controls plane i
            plane = self.planes[i]

            # Eval info
            if self.eval_info_enabled:
                if plane.edge == -1:
                    self.sum_planes_per_node[plane.now] += 1

            # Select outgoing edge 
            if plane.edge == -1: #and act[i] != 0:    # If at a waypoint

                t = self.network.nodes[plane.now].edges[act[i]] #-1   # Select an outgoing edge based on policy

                # Note that planes that are handled earlier in this loop are prioritized
                if self.network.edges[t].load + plane.size > 1.5: 
                    reward[i] -= 20.0   # we really don't want this to happen
                    blocked += 1
                    shortest_edges[i] = False
                    print("I am BLOCKED")
                else:
                    # Take this edge
                    plane.edge = t      # Begin traversal of this edge
                    #print(t)
                    plane.time = self.network.edges[t].length/plane.speed   

                    # Assign load to the selected edge
                    self.network.edges[t].load += plane.size

                    # Take next node 
                    next_node = self.network.edges[t].get_other_node(plane.now)


                    dist_to_target = self.compute_distance(next_node, plane.target)
                    dist_after_step = self.compute_distance(plane.now, plane.target)

                    # Subrewards
                    if next_node == self.network.shortest_paths[plane.now][plane.target][1]:
                        shortest_edges[i] = True
                        reward[i] += 5.0
                    else:
                        shortest_edges[i] = False
                        reward[i] += 0.5
                        progres = dist_to_target - dist_after_step
                        if progres > 0:
                            reward[i] += 2.0 * np.exp(0.005 * progres) 
                        else:
                            reward[i] -= 3 + min(abs(0.0002 * progres), 0.5)

                    
                    # Time penalty
                    reward[i] -= 0.01

                    if dist_to_target < 200: #and next_node != plane.target:
                        for edge in self.network.nodes[plane.now].edges:
                            if self.network.edges[edge].get_other_node(plane.now) == plane.target and next_node != plane.target:
                                #print("Route exits, didn't take it!")
                                reward[i] -= 3.5
                       

                    # Already set the next position
                    plane.now = next_node

                    if plane.now in plane.visited_nodes:
                        looped[i] = 1 
                        reward[i] -= 1.0
                    else:
                        plane.visited_nodes.add(plane.now)
                        reward[i] += 3.0

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
                    reward[i] += 0.5 

                # Plane finished traversing the edge
                if plane.time <= 0:
                    self.network.edges[plane.edge].load -= plane.size   # Remove the plane from the edge
                    plane.edge = -1     # Plane not on an edge
            

            if self.enable_action_mask:     # If action mask is enabled, we will mask the actions of planes
                if plane.edge != -1:    # If on an edge
                    self.action_mask[i] = 0
                else:
                    """
                    This is the old implementation.
                    """
                    self.action_mask[i, 0] = 1    # The plane is allowed to stay in place
                    for edge_i, e in enumerate(self.network.nodes[plane.now].edges):
                         self.action_mask[i, edge_i+1] = self.network.edges[e].get_other_node(plane.now) in plane.visited_nodes   # If it is, place 1

                    if sum(self.action_mask[i, :]) == self.action_space.n:
                        pass



            # The plane has reached the target
            has_reached_target = plane.edge == -1 and plane.now == plane.target     # If not on an edge and at the target waypoint            
            if has_reached_target:
                
                reward[i] += 20.0 #* (1 - (len(plane.visited_nodes) /opt_hop_dist ))
                done[i] = True
                success[i] = True

                # We need at least 1 step if we spawn at the target
                opt_distance = max(plane.shortest_path_weight, 1)
                
                # Insert delays before resetting planes
                delays_arrived.append(self.agent_steps[i])
                spr.append(self.agent_steps[i]/opt_distance*plane.speed) # Nasobeni    # Normalize OPT DISTANCE => LENGTH OF EDGES
                if self.eval_info_enabled:
                    self.distance_map[opt_distance].append(self.agent_steps[i])

                # Not sure about delays
                delays.append(self.agent_steps[i])
                self.agent_steps[i] = 0
                self.reset_plane(plane)
                print("Plane ", plane.id, " reached the target.")
    

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
        node_1 = self.network.nodes[idx1]
        node_2 = self.network.nodes[idx2]
        return ((node_1.x-node_2.x)**2 + (node_1.y -node_2.y)**2)**0.5

    def get_final_info(self, info: dict):
        agent_steps = self.agent_steps
        for agent_step in agent_steps:
            if agent_step != 0:
                info["delays"].append(agent_step)

        return info
    
    def save_paths(self):
        for i in range(self.n_planes):
            main_dict = {}
            plane = self.planes[i]
            main_dict[plane.id] = {i: np.array(path,dtype=np.int64) for i, path in enumerate(plane.paths)}
            #print("Plane")
            with open(f'planes/data{plane.id}.json', 'w') as f:
                json.dump(main_dict, f, cls=NumpyEncoder)
 
    def plot_trajectory(self):
            
            """ Plots the trajectory of each plane """
            
            planes_colors = {plane: f"C{i}" for i, plane in enumerate(self.planes)}
            
            pos = {node: coord for node, coord in zip(self.network.G.nodes, self.network.coordinates)}
            
            fig, ax = plt.subplots(figsize=(6, 8))
            
            planes_trajectories = {plane: [] for plane in self.planes}
            
            nx.draw_networkx(self.network.G, pos, with_labels=True, node_color="pink", edge_color="grey", ax=ax)
            
            legend_lines = [ax.plot([], [], color=color, label=f'Plane: {plane.id}', linewidth=1)[0] for plane, color in planes_colors.items()]
            
            ax.legend()
            
            def update_trajectory_render(frame):
                for plane in self.planes:
                    color = planes_colors[plane]
                    
                    if frame > 0 and frame < len(plane.visited_nodes):
                        current_edge = (list(plane.visited_nodes)[frame - 1], list(plane.visited_nodes)[frame])
                        planes_trajectories[plane].append(current_edge)
                    
                    nx.draw_networkx_edges(self.network.G, pos, edgelist=planes_trajectories[plane], edge_color=color, ax=ax)
                    
                    if frame < len(plane.visited_nodes):
                        nodelist =[sorted(list(plane.visited_nodes))[frame]]
                        nx.draw_networkx_nodes(self.network.G, pos, nodelist=nodelist, node_color=color, ax=ax)
            
            num_frames = max(len(list(plane.visited_nodes)) for plane in self.planes)
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


    def animation(self):
        plane_colors = {plane.id: f"C{i}" for i, plane in enumerate(self.planes)}

        def gather_flight_paths():
            flight_paths = []
            for plane in self.planes:
                flight_paths.append(plane.paths[0])
            return np.array(flight_paths)

        path_matrix = np.transpose(gather_flight_paths())   # Transpose for faster retrivial

        pos = {i: np.array([node.x, node.y]) for i, node in enumerate(self.network.nodes)}

        nodes = np.array([pos[v] for v in self.network.G])

        edges = np.array([(pos[u], pos[v]) for u, v in self.network.G.edges()])

        num_edges = len(self.network.G.edges())


        def get_colors(index):
            node_colors = ["grey" for i in range(self.network.n_nodes)]
            edge_colors = ["black" for i in range(num_edges)]
            edge_widths = [1 for i in range(num_edges)]
            sizes = [100 for i in range(self.network.n_nodes)]
            for j in range(path_matrix.shape[1]):
                i = path_matrix[index,j]
                if i >= 0:
                    node_colors[i] = plane_colors[j]
                    sizes[i] = 350
                else:
                    edge_colors[i] = plane_colors[j]
                    edge_widths[i] = 3

            return node_colors, edge_colors, edge_widths,sizes

        def init():
            ax.scatter(*nodes.T, alpha=1, s =100, color="blue")
            for edge in edges:
                ax.plot(*edge.T, color="black", linewidth=1)
            ax.grid(False)
            ax.set_axis_off()
            plt.tight_layout()
            return
        
        def _frame_update(index):
            ax.clear()
            node_colors, edge_colors, edge_widths, sizes = get_colors(index)
            ax.scatter(*nodes.T, alpha=1, s =sizes, color=node_colors)
            for i, vizedge in enumerate(edges):
                ax.plot(*vizedge.T, color=edge_colors[i], linewidth=edge_widths[i])


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(False)
        ax.set_axis_off()
        plt.tight_layout()
        ani = animation.FuncAnimation(
            fig,
            _frame_update,
            interval=1500,
            cache_frame_data=False,
            frames=path_matrix.shape[0],
        )
        plt.show()

    def animation_2(self):


        plane_colors = {plane.id: f"C{i}" for i, plane in enumerate(self.planes)}

        def gather_flight_paths():
            flight_paths = []
            for plane in self.planes:
                flight_paths.append(plane.paths[0])
            return np.array(flight_paths)
        
        num_planes = self.n_planes

        path_matrix = np.transpose(gather_flight_paths())   # Transpose for faster retrivial

        pos = {i: np.array([node.x, node.y]) for i, node in enumerate(self.network.nodes)}

        nodes = np.array([pos[v] for v in self.network.G])

        edges = list(self.network.G.edges())

        #edge_coords = np.array([(pos[u], pos[v]) for u, v in edges])

        num_nodes = len(nodes)

        num_edges = len(self.network.G.edges())

        #edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}

        plane_img = mpimg.imread('data/plane_pic.png')

        #
        def get_offset_img(img, zoom=1):
            return OffsetImage(img, zoom=zoom)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_axis_off()
        plt.tight_layout()

        # Draw nodes
        scatter_nodes = ax.scatter(*nodes.T, alpha=1, s=100, color="grey")

        # Draw edges
        edge_lines = []
        for edge in edges:
            line, = ax.plot([pos[edge[0]][0], pos[edge[1]][0]], [pos[edge[0]][1], pos[edge[1]][1]], color="black", linewidth=1, alpha=0.5)
            edge_lines.append(line)

        trail_length=2
        plane_artists = {}
        trail_artists = {}
        for plane in self.planes:
            imagebox = AnnotationBbox(get_offset_img(plane_img, zoom=1), (0,0), frameon=False, zorder=5)
            ax.add_artist(imagebox)
            plane_artists[plane.id] = imagebox

            trail_scatter = []
            for t in range(trail_length):
                trail_imagebox = AnnotationBbox(get_offset_img(plane_img, zoom=1), (0, 0), frameon=False, zorder=4, alpha=0.5)
                ax.add_artist(trail_imagebox)
                trail_scatter.append(trail_imagebox)
            trail_artists[plane.id] = trail_scatter
        # ax.legend(loc='upper right')

         # Function to initialize the animation
        def init_anim():
            scatter_nodes.set_facecolor("grey")
            scatter_nodes.set_sizes([100]*len(nodes))
            for line in edge_lines:
                line.set_color("black")
                line.set_linewidth(1)
                line.set_alpha(0.5)
            for imagebox in plane_artists.values():
                imagebox.xy = (-100, -100)  # Place off the plot initially
            for trail_list in trail_artists.values():
                for trail in trail_list:
                    trail.xy = (-100, -100)  # Place off the plot initially
            return [scatter_nodes] + edge_lines + list(plane_artists.values()) + [trail for trails in trail_artists.values() for trail in trails]
        
        def _frame_update(index):
            # Reset edges
            for line in edge_lines:
                line.set_color("black")
                line.set_linewidth(0.5)
                line.set_alpha(0.1)
            
            # Init counts for nodes and edges
            node_counts = np.zeros(num_nodes)
            edge_counts = np.zeros(num_edges)
            plane_nodes = np.zeros(num_nodes)
            plane_edges = np.zeros(num_edges)

            for j in range(num_planes):
                i = path_matrix[index,j]
                if i >= 0:
                    node_counts[i] += 1
                    plane_nodes[i] = self.planes[j].id 
                else:
                    edge_idx = abs(i)
                    edge_counts[edge_idx] += 1
                    plane_edges[edge_idx] = self.planes[j].id


            # Update nodes
            node_colors = []
            node_sizes = []
            for i in range(num_nodes):
                count = node_counts[i]
                if count > 0:
                    node_colors.append(plane_colors[plane_nodes[i]])
                    node_sizes.append(100 + 50*count)
                else:
                    node_colors.append("grey")
                    node_sizes.append(100)
                    
            scatter_nodes.set_facecolor(node_colors)
            scatter_nodes.set_sizes(node_sizes)


            # Update edges
            for i, line in enumerate(edge_lines):
                if edge_counts[i] > 0:
                    line.set_color(plane_colors[plane_edges[i]])
                    line.set_linewidth(3)
                    line.set_alpha(0.9)
            
            # Update plane positions + trails
            for j, plane in enumerate(self.planes):
                id = plane.id
                i = path_matrix[index,j]
                if i >= 0:
                    coord = pos[i]
                    plane_artists[id].xy = (coord[0], coord[1])
                else:
                    edge_idx = abs(i)
                    edge = edges[edge_idx]
                    coord = (pos[edge[0]] + pos[edge[1]])/2
                    plane_artists[id].xy = (coord[0], coord[1])

                for t in range(trail_length):
                    past_index = index -(t+1)
                    if past_index >= 0:
                        past_i = path_matrix[past_index,j]
                        if past_i >= 0:
                            past_coord = pos[past_i]
                            trail_artists[id][t].xy = (past_coord[0], past_coord[1])
                        else:
                            past_edge_idx = abs(past_i)
                            past_edge = edges[past_edge_idx]
                            # print((pos[past_edge[0]] + pos[past_edge[1]])/2)
                            past_coord = (pos[past_edge[0]] + pos[past_edge[1]])/2
                            trail_artists[id][t].xy = (past_coord[0], past_coord[1])
                            
                    else:
                        trail_artists[id][t].xy = (-100, -100)
                  
            return [scatter_nodes] + edge_lines + list(plane_artists.values()) + [trail for trails in trail_artists.values() for trail in trails]

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            _frame_update,
            init_func=init_anim,
            frames=path_matrix.shape[0],
            interval=1500,
            blit=True,
            repeat=False
        )
        plt.show()


