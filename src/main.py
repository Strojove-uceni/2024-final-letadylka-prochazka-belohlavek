from network import Network
from environment import reset_and_get_sizes

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from collections import defaultdict
#from tdqm import tdqm

from replay_buffer import ReplayBuffer
from model import DGN, DQNR, DQN, MLP, NetMon
from routing import Routing

from wrapper import NetMonWrapper
from policy import ShortestPath
from buffer import Buffer
from util import (
    dim_str_to_list,
    filter_dict,
    get_state_dict,
    interpolate_model,
    load_state_dict,
    set_attributes,
    set_seed,
)

# Loading matricies and points
import json

# Define the file path
file_path = "data/sparse_points.json"

# Load sparse_points from the JSON file
with open(file_path, 'r') as f:
    sparse_points = [tuple(point) for point in json.load(f)]  # Convert lists back to tuples


# Some start parameters
adj_mat = np.load("data/ad_mat.npy")
dist_mat = np.load("data/dist_mat.npy")
n_planes = 20
n_neighbors = np.count_nonzero(adj_mat[0,:])


# ENV var
env_var = 1 # We choose k-neighbors

n_waypoints = np.shape(adj_mat)[0]

# Define network environment
network = Network(adj_mat, dist_mat, sparse_points)

# Define type of environment
env = Routing(network, n_planes, env_var, adj_mat, dist_mat, k=n_neighbors)



# Define activation function
activ_f = 'gelu'
activation_function = getattr(F, activ_f)

# Dynamically resets the environment
n_agents, agent_obs_size, n_nodes, node_obs_size = reset_and_get_sizes(env) 

netmon_dim = 128
hidden_dim = [512,256]
netmon_enc_dim = [512,256]
netmon_iterations = 3
netmon_rnn_type = "lstm"
netmon_rnn_carryover = True
netmon_agg_type = "sum"
netmon_neighbor = True
netmon_global = True
device = 'cpu' # For now

# Use NetMon - init is rather long :)
netmon = NetMon(node_obs_size, netmon_dim, netmon_enc_dim , iterations=netmon_iterations, activation_fn=activation_function,
                rnn_type= netmon_rnn_type, rnn_carryover=netmon_rnn_carryover, agg_type=netmon_agg_type,
                output_neighbor_hidden=True, output_global_hidden=netmon_global
                ).to(device)    # Move to device



# Get observations from the environment
summary_node_obs = torch.tensor(env.get_node_observation(), dtype=torch.float32, device=device).unsqueeze(0)
summary_node_adj = torch.tensor(env.get_nodes_adjacency(), dtype=torch.float32, device=device).unsqueeze(0)
summary_node_agent = torch.tensor(env.get_node_agent_matrix(), dtype=torch.float32, device=device).unsqueeze(0)


# Summarizes our current model - just to have it somewhere
netmon_summary = netmon.summarize(summary_node_obs, summary_node_adj, summary_node_agent)



# Will explain later
node_state_size = netmon.get_state_size()

node_aux_size = 0 if env.get_node_aux() is None else len(env.get_node_aux()[0])

# Now we wrap the whole netmon class with a Wrapper - agents will use observations from netmon
netmon_startup_iterations = 1
env = NetMonWrapper(env, netmon, netmon_startup_iterations)
_, agent_obs_size, _, _ = reset_and_get_sizes(env)  # Observation length




# Below we select the model that we want to use 
# We can choose from 'dgn', 'dqnr', 'commnet', 'dqn'
chosen_model = "dgn"
num_heads = 8
num_attention_layers = 2
if chosen_model == "dgn":
    # In_features are 'agent_obs_size'
    # 'env.action_space.n' is equal to the number of neighbors - choices, 'num_actions' in DGN definition
    # 
    model = DGN(agent_obs_size, hidden_dim, env.action_space.n, num_heads, num_attention_layers, activation_function).to(device)
elif chosen_model == "dqnr":
    pass
elif chosen_model == "dqn":
    pass
else:
    raise ValueError(f"Unknown model type {chosen_model}")


model_tar = copy.deepcopy(model).to(device)     # Create a deep copy of the current model == DGN
model = model.to(device)
model_has_state = hasattr(model, "state")
aux_model = None

args = 0
# Choosing the policy for decisions, 'env.action_space.n' is equal to the number of choice the agent can perform
policy = ShortestPath(env, None, env.action_space.n, args)


# Some evaluation, not right now
#...
#



### Training ###
learning_rate = 0.001
aux_loss_coeff = 0.0 # Dunno what this exactly means
parameters = list(model.parameters()) + list(netmon.parameters())
if node_aux_size > 0 and aux_loss_coeff > 0:
    aux_model = MLP(node_state_size, (node_state_size, node_aux_size), activation_function, activation_on_output=False)
    aux_model = aux_model.to(device)
    parameters = parameters + list(aux_model.parameters())

# Init optimizer for training
optimizer = optim.AdamW(parameters, lr = learning_rate) # Adam with weight decay

state_len = model.get_state_len() if model_has_state else 0


# Init buffer
seed = 1
capacity = 2e5
replay_half_precision = True 
buff = ReplayBuffer(seed, capacity, n_agents, agent_obs_size, state_len, n_nodes, 
                    node_obs_size, node_state_size, node_aux_size, half_precision=replay_half_precision)

print("Succesfull")
raise Exception("Terminating the program.")

# Temp log variables
log_buffer_size = 1000
log_reward = Buffer(log_buffer_size, (n_planes,), np.float32) # Init of buffer
buffer_plot_last_n = 5000

log_info = defaultdict(lambda: Buffer(log_buffer_size, (1,), np.float32))

#  Just some summary comments
#...
#


# Define some reward, but i think this is for the overall graph training #TODO: Check why it is here
best_mean_reward = -float("inf")

# TODO: Check this
exception_training = None
exception_evaluation = None


try:
    # Some summary here
    #...

    # Variables for the transition buffer
    netmon_info = next_netmon_info = (0,0,0)
    buffer_state = buffer_node_state = 0
    buffer_node_aux = 0
    # Maybe render environment here
    # env.render()

    episode_step = 0
    episode_steps = 300
    current_episode = 0
    episode_done = False
    training_iteration = 0

    total_steps = 1e6 # TODO: Where is it defined in the original implementation??

    # tdqm is used just so everything looks nice
    for step in tdqm(range(1, int(total_steps)+ 1), miniters= 100, dynamic_ncols = True, disable=False): # Disable disables progress bar
        
        model.eval()    # Set the model into evaluation state 

        netmon.eval()   # Set the model into evaluation state

        if episode_step is None or episode_done:
            # Reset episode values
            episode_step = 0
            
            obs, adj = env.reset()  # Basically inits the environment
            current_episode += 1

            # Reset states
            last_state = None

        # Set current state
        if model_has_state:
            model.state = last_state
        
        # Using netmon
        buffer_node_state = (env.last_netmon_state.cpu().detach().numpy() if env.last_netmon_state is not None else 0)  # Move to the cpu, .detach() - removes the tensor from the computational graph, conversion to numpy array

        netmon_info = env.get_netmon_info()
        if hasattr(env, "get_node_aux"):
            buffer_node_aux = env.get_node_aux()

        # Get actions and execute step in environment (This changes model states)
        joint_actions = policy(obs, adj)

        next_obs, next_adj, reward, done, info = env.step(joint_actions)    # Perform step within the environment

        # Remember state for the buffer, update state afterwards
        if model_has_state:
            buffer_state = last_state.cpu().numpy() if last_state is not None else 0
            done_mask = ~torch.tensor(done, dtype=torch.bool, device=device).view(1,-1,1)
            last_state = model.state * done_mask

        # Use netmon
        if netmon is not None:
            next_netmon_info = env.get_netmon_info()
        
        episode_step += 1
        episode_done = episode_step >= episode_steps

        # Add info to the replay buffer
        buff.add(obs, joint_actions, reward, next_obs, adj, next_adj, done, episode_done, buffer_state, 
                 buffer_node_state, buffer_node_aux, *netmon_info, *next_netmon_info)

        # 'Move' observations and adjacency
        obs = next_obs
        adj = next_adj

        ## Training stats

        """
            Won't go into this right now
        """
        # # log number of steps for all agents after each episode
        # log_reward.insert(reward.mean())
        # # get all delays
        # if episode_done:
        #     info = env.get_final_info(info)
        # for k, v in info.items():
        #     log_info[k].insert(v)

        # mean_output = {}

        # if step % log_buffer_size == 0:
        #     base_path = Path(writer.get_logdir())
        #     if args.debug_plots:
        #         if netmon is not None:
        #             buff.save_node_state_diff_plot(
        #                 base_path / f"z_img_node_diff_{step}.png",
        #                 buffer_plot_last_n,
        #             )
        #             buff.save_node_state_std_plot(
        #                 base_path / f"z_img_node_std_{step}.png", buffer_plot_last_n
        #             )
        #         if (netmon is not None and isinstance(env.env, Routing)) or isinstance(
        #             env, Routing
        #         ):
        #             if env.record_distance_map:
        #                 env.save_distance_map_plot(
        #                     base_path / f"z_img_spawn_distance_{step}.png"
        #                 )
        #                 env.distance_map.clear()
        #             if env.random_topology or step == log_buffer_size:
        #                 import networkx as nx
        #                 import matplotlib.pyplot as plt

        #                 nx.draw_networkx(
        #                     env.G,
        #                     pos=nx.get_node_attributes(env.G, "pos"),
        #                     with_labels=True,
        #                     node_color="pink",
        #                 )
        #                 plt.savefig(base_path / f"z_img_topology_{step}.png")
        #                 plt.clf()

        #     mean_reward = log_reward.mean()
        #     log_reward.clear()
        #     for k, v in log_info.items():
        #         if log_info[k]._count > 0:
        #             mean_output[k] = v.mean()
        #             v.clear()

        #     eps_str = (
        #         f"  eps: {policy._epsilon:.2f}" if hasattr(policy, "_epsilon") else ""
        #     )
        #     tqdm.write(
        #         f"Episode: {current_episode}"  # print current episode
        #         f"  step: {step/1000:.0f}k"
        #         f"  reward: {mean_reward:.2f}"
        #         f"{''.join(f'  {k}: {v:.2f}' for k, v in mean_output.items())}"
        #         f"{eps_str}"
        #         f"{' | BEST' if mean_reward > best_mean_reward else ''}"
        #     )

        #     if writer is not None:
        #         writer.add_scalar("Iteration", training_iteration, step)
        #         writer.add_scalar("Train/Reward", mean_reward, step)
        #         for k, v in mean_output.items():
        #             writer.add_scalar("Train/" + k.capitalize(), v, step)
        #         if hasattr(policy, "_epsilon"):
        #             writer.add_scalar("Train/Epsilon", policy._epsilon, step)
        #         writer.add_scalar("Train/Episode", current_episode, step)
        #         writer.flush()

        #         if mean_reward > best_mean_reward:
        #             torch.save(
        #                 get_state_dict(model, netmon, args.__dict__),
        #                 Path(writer.get_logdir()) / "model_best.pt",
        #             )
        #             best_mean_reward = mean_reward

        # if (    # These are really not defined anywhere
        #     step < step_before_train or buff.count < mini_batch_size
        #     or step % step_between_train != 0
        # ):
        #     continue

        # Now to the actual training part

        training_iteration += 1

        loss_q = torch.zeros(1, device=device)
        loss_aux = torch.zeros(1, device=device)
        loss_att = torch.zeros(1, device=device)

        model.train()   # Set the model into the training mode
        netmon.train()

        # Some params, should be somewhere else
        mini_batch_size = 1
        sequence_length = 1
        gamma = 0.98
        att_regularization_coeff = 0.03

        for t, batch in enumerate(buff.get_batch(mini_batch_size, device=device, sequence_length=sequence_length)):
            if model_has_state and t == 0:
                # Load the state from the begining
                model.state = batch.agent_state


            if t == 0:
                # Get state from batch
                netmon.state = batch.node_state
            else:
                # Reset netmon state if env episode was done in this step
                netmon.state = last_netmon_state * (~last_batch_episode_done).view(-1,1,1)

            # Run netmon step
            network_obs = netmon(batch.next_obs, batch.node_adj, batch.node_agent_matrix)

            # Netmon aux loss, just a calculation
            if aux_model is not None:
                aux_prediction = aux_model(netmon.state)
                loss_aux = (loss_aux + torch.mean((aux_prediction - batch.node_aux) **2)/sequence_length)

            # Replace observation in place
            net_obs_dim = network_obs.shape[-1]     # Take last 
            batch.obs[:, :, -net_obs_dim:] = network_obs

            # Some code for t > 0:
            # ...model_checkpoint_steps
            # Let's now remember the correct netmon state for gradient calculation
            last_netmon_state = netmon.state

            # Done episodes, this is here, otherwise masking would be incorrect
            last_batch_episode_done = batch.episode_done
            last_batch_idx = batch.idx


            q_values = model(batch.obs, batch.adj)

            # Run target module
            with torch.no_grad():
                if model_has_state:
                    # This is here to avoid storing state and enxt state of the current model
                    model_tar.state = model.state.detach()

                # Get network observation with netmon
                next_network_obs = netmon(batch.next_node_obs, batch.next_node_adj, batch.next_node_agent_matrix)
                next_net_obs_dim = next_network_obs.shape[-1]#model_checkpoint_steps
                next_q = model_tar(batch.next_obs, batch.next_adj)
                next_q_max = next_q.max(dim=2)[0]


            if model_has_state:
                # If the next step belongs to a new episode we mask agent state
                state_mask = ~batch.done * (~batch.episode_done).view(-1,1)
                model.state = model.state * state_mask.unsqueeze(-1)


            # DQN target with 1 step bootstrapping
            chosen_action_target_q = (batch.reward + (~batch.done) * gamma * next_q_max)


            # Original DGN loss on all actions - even unused ones. This can be adjusted so that it is DQN loss just on selected actions
            q_target = q_values.detach()
            q_target = torch.scatter(q_target, -1, batch.action.unsqueeze(-1), chosen_action_target_q.unsqueeze(-1))

            # Combined value loss for each sample in the batch
            td_error = q_values - q_target

            # Update of q-value loss
            loss_q = loss_q + torch.mean(td_error.pow(2)) / sequence_length

            # The following code is for attention. We have it, but i am going to leave it within the if hassattr statement
            if hasattr(model, 'att_weights') and att_regularization_coeff > 0:

                # First KL_div argument is given in log-probability
                attention = F.log_softmax(torch.stack(model.att_weights), dim = -1) # Note: .stack just concatenates a sequence of tensors along a new dimension

                # Target model contains the attention weights for the next step
                target_attention = F.softmax(torch.stack(model_tar.att_weights), dim=-1)

                # Remember the original shape
                old_att_shape = attention.shape

                # Now we need to join first dimensions to apply KL divergence
                attention = attention.view(-1, n_agents)


                # Now calculate pointwise KL divergence
                kl_div = F.kl_div(attention, target_attention, reduction="none")

                # We then bring the tensor back to the old shape 
                #       (num_layers, batch_size, heads, n_agents_source, n_agents_dest)
                kl_div = kl_div.view(old_att_shape)

                # Then we transpose and reduce last three dimensions with sum
                """
                    - first transpose results in (n_agents_source, batch_size, heads, num_layers, n_agents_dest)
                    - second one results in (batch_size, n_agents_source, heads, num_layers, n_agents_dest)
                    - then we reduce with sum to (batch_size, n_agents_source, ...)

                """
                kl_div = kl_div.transpose(0, -2).transpose(0,1).sum(dim=(-1,-2,-3))

                # Now we mask out done agents (their next step in already in a new episode)
                # Reduction loss like 'batchmean', we clamp the number of not done agents with min 1 -> avoid division by zero
                kl_div = (kl_div * (~batch.done)).sum() /torch.clamp((~batch.done).sum(), min =1)

                loss_att = loss_att + kl_div / sequence_length

        loss = (loss_q + att_regularization_coeff * loss_att + aux_loss_coeff * loss_aux)
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients to faciliate learning
        torch.nn.utils.clip_grad_value_(parameters, 0.5)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)

        if aux_model is not None:
            log_info["loss_aux"].insert(loss_aux.detach().mean().item())

        log_info["q_values"].insert(q_values.detach().mean().item())
        log_info["q_target"].insert(q_target.detach().mean().item())

        log_info["loss"].insert(loss.item())
        # only log q and attention loss if necessary
        if hasattr(model, "att_weights") and args.att_regularization_coeff > 0:
            log_info["loss_q"].insert(loss_q.item())
            log_info["loss_att"].insert(loss_att.item())

        if args.target_update_steps <= 0:
            # smooth target update as in DGN
            interpolate_model(model, model_tar, args.tau, model_tar)

        elif training_iteration % args.target_update_steps == 0:
            # regular target update
            model_tar.load_state_dict(model.state_dict())
            tqdm.write(f"Update network, train iteration {training_iteration}")

        # save checkpoints
        # if step % args.model_checkpoint_steps == 0 and writer is not None:
        #     torch.save(
        #         get_state_dict(model, netmon, args.__dict__),
        #         Path(writer.get_logdir()) / f"model_{int(step):_d}.pt",
    
        


except Exception as e:
    exception_training = e

finally:
    print("Clean exit")
    del buff














