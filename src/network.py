from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.classes.function import path_weight
from collections import defaultdict
from node2vec import Node2Vec


class Node:
    """
    A waypoint in the airspace.
    """

    def __init__(self, x ,y) -> None:
        # Maybe add index: self.index = index
        self.x = x
        self.y = y
        self.neighbors = []
        self.edges = []

class Edge:
    """
    A connection between two waypoints.
    """

    def __init__(self, start, end, length) -> None:
        self.start = start
        self.end = end
        self.length = length

    # Return the node index start/end
    def get_other_node(self, node): 
        if self.start == node:
            return self.end
        elif self.end == node:
            return self.start
        else:
            raise ValueError("Node not in edges.")
# class Edge:
#     """
#     A connection between two waypoints.
#     """

#     def __init__(self, start, end, length, weather_condition) -> None:
#         self.start = start
#         self.end = end
#         self.base_length = length
#         self.weather_condition = weather_condition if weather_condition is not None else np.zeros((4,))
#         self.length = self.get_weight()

#     # Return the node index start/end
#     def get_other_node(self, node): 
#         if self.start == node:
#             return self.end
#         elif self.end == node:
#             return self.start
#         else:
#             raise ValueError("Node not in edges.")
        

#     def update_weather_condition(self, weather_condition: float) -> None:
#         """
#             Update the weather condition and recalculate the weight.
#         """
#         self.weather_condition = weather_condition
#         self.length = self.get_weight()

#     def get_weight(self) -> float:
#         """
#         Weather coefficient mark how much does the weather influence the travel time -> prolonging the distance if unfavourable
#             - they contain (wind_speed, temperature, pressure, storm_strength)
#         """
#         weather_coefficients = np.array([0.06, 0.01, 0.008, 0.1])
#         return self.base_length * (1 + weather_coefficients.T @ self.weather_condition)
    

class Network:
    """
    Network that manages the creation of the network
    """

    def __init__(self, adjacency_matrix, distance_matrix, coordinates) -> None:

        self.n_nodes = np.size(adjacency_matrix, 0)
        self.nodes = []
        self.edges = []

        self.G = nx.Graph()
        self.G_weight_key = "weight"
        self.shortest_paths = None
        self.shortest_paths_weights = None

        self.adj_mat = adjacency_matrix
        self.dist_mat = distance_matrix

        self.coordinates = coordinates  # Waypoint coordinates, oftype list of tuples
        self.node_mask = None
        self.embeddings = None
        self.max_neighbors = 0

    # def update_adj(self):
    #     self.adj_mat[89, 52] = 1
    #     self.adj_mat[52, 89] = 1

    #     self.adj_mat[33, 62] = 1
    #     self.adj_mat[62, 33] = 1

    #     self.adj_mat[104, 62] = 1
    #     self.adj_mat[62, 104] = 1

    #     self.adj_mat[110, 55] = 1
    #     self.adj_mat[55, 110] = 1

    #     self.adj_mat[30, 44] = 1
    #     self.adj_mat[44, 30] = 1

    #     self.adj_mat[27, 25] = 1
    #     self.adj_mat[25, 27] = 1

    #     self.adj_mat[45, 41] = 1
    #     self.adj_mat[41, 45] =1

    #     self.adj_mat[96,115] = 1
    #     self.adj_mat[115, 96] = 1

    #     self.adj_mat[33,54] = 1
    #     self.adj_mat[54,33] = 1


    def build_network(self, adjacency_matrix, distance_matrix):
        """
        Builds the network from the given adjacency matrix and weights it with the distance matrix
        """
        self.G = nx.Graph()     # Define a graph
        self.nodes = []     # 
        self.edges = []     #
        self.adj_mat = adjacency_matrix
        self.dist_mat = distance_matrix

        t_edge = 0
        # Create waypoints and add them to the graph
        for i in range(self.n_nodes):
            xy = self.coordinates[i]    # Get coordinates
            new_waypoint = Node(xy[0], xy[1])   # Create a waypoint with calculated x,y coordinates
            self.nodes.append(new_waypoint)
            self.G.add_node(i, pos = (new_waypoint.x, new_waypoint.y))      # Add a node to the graph



        # Create edges and neighbors for each node
        for i in range(self.n_nodes):

            non_zero_positions = np.nonzero(adjacency_matrix[i,:])[0]         # Returns indicies, where adj_matrix[i, :] .!= 0
            local_neighbors = [int(k) for k in non_zero_positions]  # Create a list for the neighbors of that node - indicies of columns in the adjacency matrix
          
            # Sorting neighbors based on distance in ascending order 
            local_neighbors.sort(key=lambda x: distance_matrix[i,x], reverse=False) # Sort it based on the distance

            for k in local_neighbors:   # Go through indexes of neighbors

                if i not in self.nodes[k].neighbors:

                    # Adding neighbors
                    self.nodes[k].neighbors.append(i)   # Add i to js neighbors
                    self.nodes[i].neighbors.append(k)   # Add j to is neighbors

                    # Sorted by the index - create an edge between them
                    edge_distance = distance_matrix[i,k]

                    # Check
                    if edge_distance == 0:
                        raise Exception("Cannot have a zero edge distance")
                    
                    if i < k:
                        new_edge = Edge(i, k, edge_distance)    # Edge from i to j
                    else:
                        new_edge = Edge(k, i, edge_distance)

                    self.edges.append(new_edge)             # Add edge into the list of all edges
                    self.nodes[k].edges.append(t_edge)      # Number of the edge
                    self.nodes[i].edges.append(t_edge)      # Same number of the original
                    self.G.add_edge(new_edge.start, new_edge.end, weight=new_edge.length)       # Add the edge into the graph
                    t_edge +=1      # Increment the edge index  



        # print(self.nodes[0].neighbors)
        # print(self.nodes[10].edges)

        # raise ValueError("a")
        # Find max amount of neighbors
        max_neighbors = 0
        for node in self.nodes:
            if len(node.neighbors) > max_neighbors:
                max_neighbors = len(node.neighbors)
        self.max_neighbors = max_neighbors

        # Order waypoint edges by neighbor node id to remove symetries
        """
        We sort the list of edges connected to each node in ascending order based on the IDs 
            of nerighboring nodes they connect to. Why? This helps with consisten processing 
            of the edges and removes any symetries. 

        :for: iterates over each node in the network
        :sorted: it sorts the list seslf.nodes[i].edges, used with a special key to sort by
            - edge_index represents an index of an edge connected to node i
            - self.edges[edge_index].end represents the node that edge_index connects to
            - get_other_node(i) returns the index of the node on the other end of an edge, 
                esseantially returning the neighbor node connected to node i by that edge
    
        This may be overexplained, but it took me a while to understand
        """
     
        for i in range(self.n_nodes):
            self.nodes[i].edges = sorted(
                self.nodes[i].edges, 
                key=lambda edge_index: self.edges[edge_index].get_other_node(i))
            
        # self.create_node_embeddings()
            

    def update_shortest_paths(self):
        """
        Calculate shortest paths and store them in self.shortest.paths. 
        The corresponding weights are stored in self.shortest_path_weights.

            - nx.shortest_paths() automatically computes the shortest paths between 
                each node in the graph based on the weights
        """
        self.shortest_paths = dict(nx.shortest_path(self.G, weight = "weight"))   # Calculates the shortest paths -> dict

        #self.shortest_paths_weights = dict(nx.shortest_path_length(self.G, weight="weight"))  


        """
        The code below is the same thing as the one-liner above. Dunno why they didn't use that. Even if the dicts are created and compared, they are the same. :D
        """
        self.shortest_paths_weights = defaultdict(dict) 
        for start in self.shortest_paths:
            for end in self.shortest_paths[start]:
                a = path_weight(self.G, self.shortest_paths[start][end], "weight")
                self.shortest_paths_weights[start][end] = a     # Calculate the weight between each shortest path

    def reset(self, adjacency_matrix, distance_matrix):
        """
        Basically initialize the network.
        """

        self.build_network(adjacency_matrix, distance_matrix)   # Build the network
        self.update_shortest_paths()    # Update the paths
        self.update_nodes_adjacency()   # Update node adjacency
        self.create_node_mask()         # Create node mask for policy

        # for i in range(118):
        #     print(len(self.nodes[i].neighbors) == len(self.node_mask[i].nonzero()[0]))



    def get_nodes_adjacency(self):
        """
        Returns the adjacency matrix of waypoints in the airspace.
        """
        return self.adj_mat

    def update_nodes_adjacency(self):
        self.adj_mat = np.eye(self.n_nodes, self.n_nodes, dtype = np.int8)
        for i in range(self.n_nodes):
            for neighbor in self.nodes[i].neighbors:    # Utilize that the neighbors are saved as indiced from the adjacency matrix
                self.adj_mat[i, neighbor] = 1

    def render(self):
        # pass
        fig, ax = plt.subplots(figsize=(8, 6))

        last = self.nodes[self.n_nodes-17:self.n_nodes]
        node_colors = ['pink' if node not in last else 'lightblue' for node in self.nodes]


        positions = {node: coordinates for node, coordinates in zip(self.G.nodes, self.coordinates)}
        #nx.draw_networkx(self.G, positions, with_labels=True, node_color = "pink")
        nx.draw_networkx_nodes(self.G, positions, node_color =node_colors)
        nx.draw_networkx_edges(self.G, positions, edge_color='gray')
        nx.draw_networkx_labels(self.G, positions, font_size=5, font_color='black', ax=ax)
        
        # for plane in planes:
        #     x, y = self.nodes[plane.now].x, self.nodes[plane.now].y
        #     ax.plot(x,y, marker=(3,0,0), markersize = 15, markerfacecolor = 'red', markeredgecolor='k', label=f'Plane {plane.id}')

        plt.show()
        fig.canvas.draw()
        # plt.pause(5)
        # plt.ioff()
        # plt.close(fig)
        
    def create_node_mask(self):
        """
        Creates a mask for node edges. Policy decisions are masked with it. 
        """
        full_mask = []
        for node in self.nodes:
            num_edges = len(node.edges)
            node_mask = [1] * num_edges # + [1]
            if num_edges < self.max_neighbors:  # Less than 10 neighbors
                node_mask += [0] * (self.max_neighbors-num_edges)
            full_mask.append(node_mask)
        self.node_mask = np.array(full_mask, dtype=np.bool_)

    def create_node_embeddings(self):
        nodetovec = Node2Vec(self.G, dimensions=32, walk_length=80, workers=4, p=1, q=1, weight_key='weight')
        mod = nodetovec.fit(window=10, min_count=1, batch_words=4)
        embeddings = {node:  mod.wv[node].tolist() for node in self.G.nodes()}
        self.embeddings = embeddings



