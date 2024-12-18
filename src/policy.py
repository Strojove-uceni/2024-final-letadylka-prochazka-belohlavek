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

        self.model = model
        self.step_before_train = step_before_train
        self.epsilon_update_freq = epsilon_update_freq
        self.epsilon_decay = epsilon_decay
        self.action_space = action_space
        self.epsilon = epsilon
        self.step = 0
        self.epsilon_tmp = None

    def __call__(self, obs, adj):
        self.step += 1

        # Actions (length is == #of agents)
        actions = np.zeros(obs.shape[0], dtype=np.int32)

        with torch.no_grad():
            
            device = next(self.model.parameters()).device # where the model is

            obs = (torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True))

            adj = (torch.tensor(adj, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True))

            node_mask = torch.tensor(self.extract_node_mask(), dtype=torch.bool).to(device, non_blocking=True)   # Mask for invalid q_values

            last_mask = torch.tensor(self.last_node_mask(), dtype=torch.bool).to(device, non_blocking=True)   # Mask for invalid q_values
 
            # Forward pass of the model
            q_values = self.model(obs, adj)

            q_vals = q_values.masked_fill(node_mask==0, -1e9)
            q_vals = q_vals.masked_fill(last_mask==0, -1e9)
            
            # Squeezes first dimension: 'batch_size' == 1
            q_vals = q_vals.cpu().squeeze(0).detach().numpy()
            for i in range(q_vals.shape[0]):
                if all(q_vals[i,:]==0):
                    raise ValueError("All actions are bad")
                    
            if self.enable_action_mask:
                # Remove visited nodes
                q_vals[self.env.action_mask.nonzero()] = float("-inf")

        # Epsilon greedy action selections
        random_actions = self.get_random_actions(node_mask)     # We need to mask of random choices of edges that are 'non-existing'
        random_filter = np.random.rand(actions.shape[0]) < self.epsilon
        actions = ((np.argmax(q_vals, axis=-1))) * ~random_filter + random_filter*random_actions
        

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
            self.model.state = self.model.state * ~torch.tensor(agents_to_reset, dtype=bool, device= self.model.state.device).unsqueeze(-1)

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
        for i in range(self.env.network.n_nodes):
             if not len(self.env.network.nodes[i].neighbors) == len(self.env.network.node_mask[i].nonzero()[0]):
                 raise ValueError("")

        actions = []
        for i in range(node_mask.shape[0]):
            valid_acts = len(node_mask[i].nonzero(as_tuple=True)[0]) # Get #of non-zero elements
            gen_num = np.random.randint(0,valid_acts)
            actions.append(gen_num)
        return np.array(actions)
        
    def last_node_mask(self):
        """
        Find the last node where the agent came from and mask it.
        """
        full_mask = []
        for plane in self.env.planes:
            plane_mask = [1 for i in range(self.env.network.max_neighbors)]
            for i, edge in enumerate(self.env.network.nodes[plane.now].edges):
                if self.env.network.edges[edge].get_other_node(plane.now) == plane.last_node:
                    plane_mask[i] = 0
            full_mask.append(plane_mask)
        return np.array(full_mask, dtype= np.bool_)