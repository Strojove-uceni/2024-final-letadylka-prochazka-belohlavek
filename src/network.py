from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.classes.function import path_weight
from collections import defaultdict


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

        self.adj_mat = adjacency_matrix
        self.dist_mat = distance_matrix
        self.n_neighbors = None

        self.shortest_paths = None
        self.shortest_paths_weights = None

        self.coordinates = coordinates  # Waypoint coordinates, oftype list of tuples


    def build_network(self, adjacency_matrix, distance_matrix):
        """
        Builds the network from the given adjacency matrix and weights it with the distance matrix
        """
        self.G = nx.Graph()     # Define a graph
        self.adj_mat = adjacency_matrix     # Assign adj
        self.dist_mat = distance_matrix     # Assign dist
        self.nodes = []     # 
        self.edges = []     # 
        self.n_neighbors = np.count_nonzero(adjacency_matrix[0])     # Number of neighbors - every node has the same amount of neighbors 

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

                if i not in self.nodes[k].neighbors:# and len(self.nodes[k].neighbors) < self.n_neighbors:

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
                # print(a)
                self.shortest_paths_weights[start][end] = a# Calculate the weight between each shortest path



    def reset(self, adjacency_matrix, distance_matrix):
        """
        Basically initialize the network.
        """
        self.build_network(adjacency_matrix, distance_matrix)   # Build the network
        self.update_shortest_paths()    # Update the paths
        self.update_nodes_adjacency()   # Update node adjacency


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



## HERE YOU CAN CHECK WHAT IS HAPPENING IN THE UPDATE SHORTEST PATH FUNCTION
# # Create a graph with nodes and edges
# G = nx.Graph()
# G.add_nodes_from(["A", "B", "C", "D", "E", "F", "G", "H"])
# G.add_edge("A", "B", weight=4)
# G.add_edge("A", "H", weight=8)
# G.add_edge("B", "C", weight=8)
# G.add_edge("B", "H", weight=11)
# G.add_edge("C", "D", weight=7)
# G.add_edge("C", "F", weight=4)
# G.add_edge("C", "I", weight=2)
# G.add_edge("D", "E", weight=9)
# G.add_edge("D", "F", weight=14)
# G.add_edge("E", "F", weight=10)
# G.add_edge("F", "G", weight=2)
# G.add_edge("G", "H", weight=1)
# G.add_edge("G", "I", weight=6)
# G.add_edge("H", "I", weight=7)

# # # Find the shortest path from node A to node E
# path = dict(nx.shortest_path(G))
# shortest_paths_weights = defaultdict(dict)
# for start in path:
#     for end in path[start]:
#         shortest_paths_weights[start][end] = path_weight(G, path[start][end], "weight")
#         print(start, " ", end, " ",  path[start][end])

# # b = dict(nx.shortest_path_length(G, weight='weight'))
# # for start in b:
# #     for end in b[start]:
# #         #print(start, " ", end, " ",  b[start][end])
# #         pass

# # print(b == shortest_paths_weights)

# a = [1,2]
# print(int(1 in a))


b = [[1,2,1], [3,4,5]]
print(np.array(b))