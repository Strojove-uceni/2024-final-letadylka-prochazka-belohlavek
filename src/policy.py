import numpy as np
import torch
from gymnasium.spaces import Discrete
from routing import Routing




# Model based policy
class EpsilonGreedy:
    def __init__(self, env, model, action_space, 
                 epsilon = None,
                 step_before_train = None,
                 epsilon_update_freq = None,
                 epsilon_decay = None
                 ) -> None:
        
        self.env = env
        self.enable_action_mask = (hasattr(self.env, "enable_action_mask") and self.env.enable_action_mask)

        self.model = model  # NN 
        self.step_before_train = step_before_train
        self.epsilon_update_freq = epsilon_update_freq
        self.epsilon_decay = epsilon_decay
        self.action_space = action_space
        self.epsilon = epsilon
        self.step = 0
        self.epsilon_tmp = None

    def __call__(self, obs, adj):
        #print("obs is: ", obs.shape)
        #print("Shape is: ", adj.shape)
        self.step += 1

        # Actions (length is == #of agents)
        actions = np.zeros(obs.shape[0], dtype=np.int32)
        
        with torch.no_grad():
            
            device = next(self.model.parameters()).device # where the model is

            obs = (torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True))

            adj = (torch.tensor(adj, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True))

            node_mask = torch.tensor(self.extract_node_mask(), dtype=torch.float32).to(device, non_blocking=True)   # Mask for invalid q_values
 
            # Forward pass of the model
            q_values = self.model(obs, adj)

            # Squeezes first dimensino: 'batch_size' == 1
            q_values = q_values.cpu().squeeze(0).detach().numpy()
            q_values[node_mask==0] = float('-inf')  # Set invalid edges to 0
    
            if self.enable_action_mask:
                q_values[self.env.action_mask.nonzero()] = float("-inf")

        # Epsilon greedy action selections
        random_actions = self.get_random_actions(node_mask)     # We need to mask of random choices of edges that are 'non-existing'
        random_filter = np.random.rand(actions.shape[0]) < self.epsilon
        
        actions = (np.argmax(q_values, axis=-1)) * ~random_filter + random_filter*random_actions
        

        if (self.epsilon > 0 and self.step > self.step_before_train and self.step % self.epsilon_update_freq == 0):
            self.epsilon *= self.epsilon_decay
            if self.epsilon < 0.01:
                self.epsilon = 0.01

        return actions
                

    def eval(self):
            # Remember old epsilon and then switch to greedy policy
            self.eps_tmp = self.epsilon
            self.epsilon = 0

    def reset(self, agents_to_reset):
        """
        Resets agents.

        :param agents_to_reset: is of shape (batch_size, n_agents). A value of 1 indicates reset!
        
        """
        if hasattr(self.model, "state") and self.model.state is not None:
            self.model.state = self.model.state * ~torch.tensor(agents_to_reset, dtype=bool, device= self.model.state.divce).unsqueeze(-1)

    def train(self):
        # Switch back to old eps
        if self.epsilon_tmp is not None:
            self.epsilon_tmp = None
            self.epsilon = self.epsilon_tmp

    def extract_node_mask(self):
        """
        Extracts possible edges from the node_mask matrix from Network for plane points.
        """
        ids = [plane.now for plane in self.env.planes]
        mask = self.env.network.node_mask[ids,:]    # Take rows from ids with all columns
        return mask

    def get_random_actions(self, node_mask):
        """
        Generate random VALID actions for exploration.
        """
        actions = []
        for i in range(node_mask.shape[0]):
            valid_acts = len(node_mask[i].nonzero()[0]) # Get #of non-zero elements
            actions.append(np.random.randint(valid_acts))
        return np.array(actions)
        




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
                act[i] = 0 # The plane is finished -> it stays here for one step
                continue

            # Get the next waypoint on the flight path
            next_node = network.shortest_paths[current_node][target_node][1]
 
            # Find the edge that leads to the next_node
            for index, j in enumerate(network.nodes[current_node].edges):
                current_edge = network.edges[j]

                # Now we choose an edge
                if current_edge.get_other_node(plane.now) == next_node:
                    act[i] = index + 1 # + 1 is here because index of a node can be 0
                    break

        return act