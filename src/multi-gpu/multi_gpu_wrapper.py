import os
import torch
import numpy as np
from multi_gpu_netmon import GNetMon


class GNetMonWrapper:
    """
    Wraps a given environment with a netmon instance. Creates new observations by
    concatenating the agent's observations and the respective graph observations.
    """

    def __init__(self, env, netmon, startup_iterations) -> None:
        self.env = env
        self.netmon = netmon    # This is the netmon class from model.py
        self.device = next(netmon.parameters()).device  

        self.node_obs = None
        self.node_adj = None
        self.node_agent_matrix = None
        self.last_netmon_state = None
        self.current_netmon_state = None
        assert startup_iterations >= 1, "Number of startup iterations must be >= 1" # This is basically to warm up the LSTMs
        self.startup_iterations = startup_iterations
        self.frozen = False #   Freeze training 

    def __getattr__(self, name):
        # Allow to access attributes of the environment
        return getattr(self.env, name)  # If not found in the Netmon, look into the environment

    def __str__(self) -> str:
        return (
            self.env.__str__()
            + os.linesep
            + "▲ environment is wrapped with NetMon (graph obs)"
        )

    def reset(self):    
        self.frozen = False
        self.current_netmon_state = None
        self.last_netmon_state = None
        obs, adj = self.env.reset()     # Call reset on the environment -> on Routing

        # Init of external state
        external_state = None

        for _ in range(self.startup_iterations):    # "Warm up" the environment
            network_obs, external_state = self._netmon_step(external_state)       # Perform a step in the environment
        
        self.current_netmon_state = external_state
        return np.concatenate((obs, network_obs), axis=-1), adj

    def step(self, actions):
        next_obs, next_adj, reward, done, info = self.env.step(actions)     # Perform the step within an environment

        next_network_obs, next_external_state = self._netmon_step(self.current_netmon_state)      # Perform a step within the netmon class from model.py
        self.last_netmon_state = self.current_netmon_state
        self.current_netmon_state = next_external_state

        next_joint_obs = np.concatenate((next_obs, next_network_obs), axis=-1)      # Concatenate the results
        return next_joint_obs, next_adj, reward, done, info

    def freeze(self):
        """
        Disable message-passing for the rest of this episode. Agents still receive
        graph observations depending on their position in the graph.
        """
        self.frozen = True

    def get_netmon_info(self):
        return (self.node_obs, self.node_adj, self.node_agent_matrix)   # Return basic information 

    def get(self):
        return self.env     # Get environment

    def _netmon_step(self, external_state):
        """
        This function performs the step within the environment and with the agents while handling observations.
        """

        # If we froze Message Passing
        if self.frozen:
            self.node_agent_matrix = self.env.get_node_agent_matrix()
            node_agent_matrix_in = torch.tensor(
                self.node_agent_matrix, dtype=torch.float32
            ).unsqueeze(0) # Add a dimension to the first position

            network_obs = torch.bmm(    # Perform Batch Matrix Multiplication
                self.netmon_out.transpose(1, 2).cpu().detach(), node_agent_matrix_in
            ).transpose(1, 2)
            return network_obs.squeeze(0).numpy(), external_state

        self.node_obs = self.env.get_node_observation()     # Gather observations
        self.node_adj = self.env.get_nodes_adjacency()      # Gather adjacency matrix
        self.node_agent_matrix = self.env.get_node_agent_matrix()   # Gather agent positions

        with torch.no_grad():
            # Prepare inputs
            node_obs_in = (
                torch.tensor(self.node_obs, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device, non_blocking=True)
            )
            node_adj_in = (
                torch.tensor(self.node_adj, dtype=torch.float32)
                .unsqueeze(0)   # Adds dimension to the 0th position
                .to(self.device, non_blocking=True)     # Moves the tensor to device
            )
            node_agent_matrix_in = (
                torch.tensor(self.node_agent_matrix, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device, non_blocking=True)
            )

            # Perform netmon step with correct state
            self.netmon_out, updated_state = self.netmon(
                node_obs_in, node_adj_in, node_agent_matrix_in, state=external_state, no_agent_mapping=True
            )

            network_obs = GNetMon.output_to_network_obs(
                self.netmon_out, node_agent_matrix_in
            )
            return network_obs.squeeze(0).cpu().detach().numpy(), updated_state


