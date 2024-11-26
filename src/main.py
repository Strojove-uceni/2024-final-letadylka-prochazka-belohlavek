from network import Network
from environment import reset_and_get_sizes

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import traceback
import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


from replay_buffer import ReplayBuffer
from model import DGN, DQNR, DQN, MLP, NetMon
from routing import Routing

from wrapper import NetMonWrapper
from policy import ShortestPath, EpsilonGreedy
from buffer import Buffer
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from util import (
    dim_str_to_list,
    filter_dict,
    get_state_dict,
    interpolate_model,
    load_state_dict,
    set_attributes,
    set_seed,
)
from eval import evaluate



# File path
file_path = "data/sparse_points.json"

# Load sparse_points from the JSON file
with open(file_path, 'r') as f:
    sparse_points = [tuple(point) for point in json.load(f)]  # Convert lists back to tuples

# Some start parameters
adj_mat = np.load("data/ad_mat.npy")
dist_mat = np.load("data/dist_mat.npy")
n_planes = 10
n_neighbors = 5 # 3 by default # np.count_nonzero(adj_mat[0,:])


new_min = 0.1
new_max = 1.0
min = 135.19907434848324
max = 4324.933341605337
# dist_mat = ((dist_mat-min)/(max-min))*(new_max-new_min) + new_min
# b = []
# for i in range(131):
#     non_zero_positions = np.nonzero(adj_mat[i,:])[0]         # Returns indicies, where adj_matrix[i, :] .!= 0
#     g = []
#     for j in range(131):
#         if j in non_zero_positions:
#             g.append(1)
#         else:
#             g.append(0)
#     b.append(g)
        
# b = np.array(b)
# dist_mat[b==0] = 0
# mi = float('inf')
# for k in range(131):
#     for i in range(131):
#         if dist_mat[k,i] < mi and dist_mat[k,i] > 0:
#             mi = dist_mat[k,i]

# print(mi)
# print(dist_mat.max())

# ENV var
env_var = 1 # We choose k-neighbors

n_waypoints = np.shape(adj_mat)[0]

# Define network environment
network = Network(adj_mat, dist_mat, sparse_points)

# Define type of environment
env = Routing(network, n_planes, env_var, adj_mat, dist_mat, k=n_neighbors, enable_action_mask=False)

# Define activation function
activ_f = 'leaky_relu'
activation_function = getattr(F, activ_f)

# Dynamically resets the environment
n_agents, agent_obs_size, n_nodes, node_obs_size = reset_and_get_sizes(env) 

# env.render()

print("Agent observation size: ", agent_obs_size)
print("Node observation size: ", node_obs_size)


# from node2vec import Node2Vec


# node2vec = Node2Vec(network.G, dimensions=64, walk_length=50, workers=4, p=1, q=1, weight_key='weight')

# mod = node2vec.fit(window=10, min_count=1, batch_words=4)

# embeddings = {node: mod.wv[node] for node in network.G.nodes()}
# print(embeddings[1])



# raise ValueError("a")
netmon_dim = 128
hidden_dim = [1024,1024,512]
netmon_enc_dim = [512,256] #[512,256]
netmon_iterations = 3
netmon_rnn_type = "lstm"
netmon_rnn_carryover = True
netmon_agg_type = "gcn" # "gcn", "graphsage"
netmon_neighbor = True
netmon_global = True
device = 'cpu' # For now
target_update_steps = 0     # Number of steps between target model updates (0 for smooth updates)
tau = 0.01  # Interpolatin factor for smooth target model updates
eval_episodes = 3 # Default 1000
eval_episode_steps = 40 # Default = 300  - maximum number of steps per eval episode
eval_output_detailed = True # Default
output_node_state_aux = False # Default
att_regularization_coeff = 0.03

# Args to define
debug_plots = False # Default to false
step_before_train = 100 # Default 2000
step_between_train = 1 # Default from authors 
args = None
epsilon = 0.6
epsilon_decay = 0.996
epsilon_update_freq = 100

# Training params
learning_rate = 0.001
aux_loss_coeff = 0.00 # Default: 0.03 
sequence_length = 5
gamma = 0.98

# Buffer settings
seed = 1
capacity = 2000 # Default: 200000 but that is too much for my pc
replay_half_precision = False 

# Use NetMon - init is rather long :)
netmon = NetMon(node_obs_size,  # 'in_features' in init
                netmon_dim,     # 'hidden_features' in init
                netmon_enc_dim , # 'encoder_units' in init
                iterations=netmon_iterations, 
                activation_fn=activation_function,
                rnn_type= netmon_rnn_type, rnn_carryover=netmon_rnn_carryover, agg_type=netmon_agg_type,
                output_neighbor_hidden=True, output_global_hidden=netmon_global
                ).to(device)    # Move to device


# Get observations from the environment
summary_node_obs = torch.tensor(env.get_node_observation(), dtype=torch.float32, device=device).unsqueeze(0)
summary_node_adj = torch.tensor(env.get_nodes_adjacency(), dtype=torch.float32, device=device).unsqueeze(0)
summary_node_agent = torch.tensor(env.get_node_agent_matrix(), dtype=torch.float32, device=device).unsqueeze(0)


# Summarizes our current model - just to have it somewhere
netmon_summary = netmon.summarize(summary_node_obs, summary_node_adj, summary_node_agent)


node_state_size = netmon.get_state_size()


node_aux_size = 0 if env.get_node_aux() is None else len(env.get_node_aux()[0]) # = n_waypoints

# Now we wrap the whole netmon class with a Wrapper - agents will use observations from netmon
netmon_startup_iterations = 1   # Number of MP interations after the env reset - before first step
env = NetMonWrapper(env, netmon, netmon_startup_iterations)
_, agent_obs_size, _, _ = reset_and_get_sizes(env)  # Observation length

print(f"Node state size: {node_state_size}")        # 256
print(f"Agent observation size: {agent_obs_size}")  # 3263
print(f"Node auxiliary size: {node_aux_size}")      # 0


# Below we select the model that we want to use 
# We can choose from 'dgn', 'dqnr', 'commnet', 'dqn'
chosen_model = "dgn"
num_heads = 8
num_attention_layers = 3
if chosen_model == "dgn":
    # In_features are 'agent_obs_size'
    # 'env.action_space.n' is equal to the number of neighbors - choices, 'num_actions' in DGN definition
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


# Choosing the policy for decisions, 'env.action_space.n' is equal to the number of choice the agent can perform
# It must be train for our purposes
policy_type = "trained"
if policy_type == "trained":
    policy = EpsilonGreedy(env, model, env.action_space.n, epsilon=epsilon, step_before_train=step_before_train, epsilon_update_freq= epsilon_update_freq, epsilon_decay=epsilon_decay)
else:
    policy = ShortestPath(env, None, env.action_space.n, args)



### Training ###
parameters = list(model.parameters()) + list(netmon.parameters())
if node_aux_size > 0 and aux_loss_coeff > 0:
    aux_model = MLP(node_state_size, (node_state_size, node_aux_size), activation_function, activation_on_output=False)
    aux_model = aux_model.to(device)
    parameters = parameters + list(aux_model.parameters())

# Init optimizer for training
optimizer = optim.AdamW(parameters, lr = learning_rate) # Adam with weight decay

state_len = model.get_state_len() if model_has_state else 0 # 0 for DGN


# Init buffer
buff = ReplayBuffer(seed, capacity, n_agents, agent_obs_size, state_len, n_nodes, 
                    node_obs_size, node_state_size, node_aux_size, half_precision=replay_half_precision)


# Temp log variables
log_buffer_size = 100   # Default: 1000
log_reward = Buffer(log_buffer_size, (n_planes,), np.float32) # Init of buffer
buffer_plot_last_n = 50   # Default: 5000


# This is for evaluation and logging 
log_info = defaultdict(lambda: Buffer(log_buffer_size, (1,), np.float32))
comment = "_"
if hasattr(env, "env_var"):
    comment += f"R{env.env_var.value}"
if netmon is not None:
    comment += "_netmon"
writer = SummaryWriter(comment=comment)


# Define reward
best_mean_reward = -float("inf")
exception_training = None
exception_evaluation = None


try:
    # Variables for the transition buffer
    netmon_info = next_netmon_info = (0,0,0)
    buffer_state = buffer_node_state = 0
    buffer_node_aux = 0

    episode_step = None
    episode_steps = 50 # Default
    current_episode = 0
    episode_done = False
    training_iteration = 0
    disable_prog = False
    total_steps = 200 # Default: 1e6
    mini_batch_size = 10     # The number of individual experiences that are drawn from the replay buffer


    # tdqm is used just so everything looks nice
    for step in tqdm(range(1, int(total_steps)+ 1), miniters= 100, dynamic_ncols = True, disable=disable_prog): # Disable disables progress bar
        
        model.eval()    # Set the model into evaluation state 

        netmon.eval()   # Set the model into evaluation state

        if episode_step is None or episode_done:
            # Reset episode values
            episode_step = 0
            
            obs, adj = env.reset()  # Inits the environment
            current_episode += 1

            # Reset states
            last_state = None

        # Set current state
        if model_has_state:
            model.state = last_state

        # Using netmon
        buffer_node_state = (env.last_netmon_state.cpu().detach().numpy() if env.last_netmon_state is not None else 0)  # Move to the cpu, .detach() - removes the tensor from the computational graph, conversion to numpy array
        
        netmon_info = env.get_netmon_info()     # Returns (node_obs, node_adj, node_agent_mat)
        if hasattr(env, "get_node_aux"):
            buffer_node_aux = env.get_node_aux()
       
        # Get actions based on policy and execute step in environment (This changes model states)
        joint_actions = policy(obs, adj)
        next_obs, next_adj, reward, done, info = env.step(joint_actions)    # Perform step within the environment
    

        # Remember state for the buffer, update state afterwards
        if model_has_state:
            buffer_state = last_state.cpu().numpy() if last_state is not None else 0
            done_mask = ~torch.tensor(done, dtype=torch.bool, device=device).view(1,-1,1)
            last_state = model.state * done_mask

        # Gather info from NetMon
        next_netmon_info = env.get_netmon_info()
        
        episode_step += 1
        episode_done = episode_step >= episode_steps

        # Add collected information to the replay buffer
        buff.add(obs, joint_actions, reward, next_obs, adj, next_adj, done, episode_done, buffer_state, 
                    buffer_node_state, buffer_node_aux, *netmon_info, *next_netmon_info)

        
        # 'Move' observations and adjacency
        obs = next_obs
        adj = next_adj

        ## Training stats

        """
            The code below gathers information and saves it for evaluation
        """
        # log number of steps for all agents after each episode
        log_reward.insert(reward.mean())

        # get all delays
        if episode_done:
            info = env.get_final_info(info)
        for k, v in info.items():
            log_info[k].insert(v)

        mean_output = {}

        if step % log_buffer_size == 0:
            base_path = Path(writer.get_logdir())

            if debug_plots:
                if netmon is not None:
                    buff.save_node_state_diff_plot(
                        base_path / f"z_img_node_diff_{step}.png",
                        buffer_plot_last_n,
                    )
                    buff.save_node_state_std_plot(
                        base_path / f"z_img_node_std_{step}.png", buffer_plot_last_n
                    )
                if (netmon is not None and isinstance(env.env, Routing)) or isinstance(
                    env, Routing
                ):
                    if env.record_distance_map:
                        env.save_distance_map_plot(
                            base_path / f"z_img_spawn_distance_{step}.png"
                        )
                        env.distance_map.clear()
                    if step == log_buffer_size:

                        nx.draw_networkx(
                            env.G,
                            pos=nx.get_node_attributes(env.G, "pos"),
                            with_labels=True,
                            node_color="pink",
                        )
                        plt.savefig(base_path / f"z_img_topology_{step}.png")
                        plt.clf()

            mean_reward = log_reward.mean()
            log_reward.clear()
            for k, v in log_info.items():
                if log_info[k]._count > 0:
                    mean_output[k] = v.mean()
                    v.clear()

            tqdm.write(
                f"Episode: {current_episode}"  # print current episode
                f"  step: {step/1000:.0f}k"
                f"  reward: {mean_reward:.2f}"
                f"{''.join(f'  {k}: {v:.2f}' for k, v in mean_output.items())}"
                f"{' | BEST' if mean_reward > best_mean_reward else ''}"
            )

            writer.add_scalar("Iteration", training_iteration, step)
            writer.add_scalar("Train/Reward", mean_reward, step)
            for k, v in mean_output.items():
                writer.add_scalar("Train/" + k.capitalize(), v, step)
            writer.add_scalar("Train/Episode", current_episode, step)
            writer.flush()

            if mean_reward > best_mean_reward:
                # torch.save(
                #     get_state_dict(model, netmon, args.__dict__),
                #     Path(writer.get_logdir()) / "model_best.pt",
                # )
                best_mean_reward = mean_reward

        # For full code, this has to be running
        if (step < step_before_train or buff.count < mini_batch_size or step % step_between_train != 0):    
            # step_before_train ensures that we make at least 2000 iterations to collect initial experience from the environment
            #   before we actually start the training - this servers as a warm-up phase -> we populate the replay buffer with experiences
            print("Skipped for step:", step)
            continue



        # Now to the actual training part
        training_iteration += 1

        loss_q = torch.zeros(1, device=device)
        loss_aux = torch.zeros(1, device=device)
        loss_att = torch.zeros(1, device=device)

        model.train()   # Set the model into the training mode
        netmon.train()  # Set the model into the training mode

  
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
            network_obs = netmon(batch.node_obs, batch.node_adj, batch.node_agent_matrix)   # Forward pass

            if aux_model is not None:
                # Netmon aux loss
                aux_prediction = aux_model(netmon.state)    # Netmon.state is (1,131,256) -> prediction is of shape (131,131)
                #rint(torch.mean((aux_prediction - batch.node_aux)** 2)/sequence_length)
                loss_aux = (loss_aux + torch.mean((aux_prediction - batch.node_aux)** 2)/sequence_length)
                #print("Auxilary loss it: ", loss_aux)

            # Replace observation in place
            net_obs_dim = network_obs.shape[-1]     # Take last 
            batch.obs[:, :, -net_obs_dim:] = network_obs

            # Let's now remember the correct netmon state for gradient calculation
            last_netmon_state = netmon.state

            # Done episodes, this is here, otherwise masking would be incorrect
            last_batch_episode_done = batch.episode_done
            last_batch_idx = batch.idx

            # Get Node masks
            old_node_mask = torch.tensor(env.network.node_mask[env.old_node_plane_ids,:], dtype = torch.float32, device = device)
            new_node_mask = torch.tensor(env.network.node_mask[env.new_node_plane_ids,:], dtype = torch.float32, device = device)

            q_values = model(batch.obs, batch.adj)  # Estimate the expected reward for each possible action
            q_values = q_values * old_node_mask # Multiply by old node_mask
            
            # Run target module
            with torch.no_grad():
                if model_has_state:
                    # This is here to avoid storing state and next state of the current model
                    model_tar.state = model.state.detach()

                # Get network observation with netmon
                next_network_obs = netmon(batch.next_node_obs, batch.next_node_adj, batch.next_node_agent_matrix)
                next_net_obs_dim = next_network_obs.shape[-1]   #   model_checkpoint_steps
                next_q = model_tar(batch.next_obs, batch.next_adj)
                next_q = next_q * new_node_mask
                next_q_max = next_q.max(dim=2)[0]


            if model_has_state:
                # If the next step belongs to a new episode we mask agent state
                state_mask = ~batch.done * (~batch.episode_done).view(-1,1)
                model.state = model.state * state_mask.unsqueeze(-1)


            # DQN target with 1 step bootstrapping
            # Computes the target Q-values for the actions taken
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
                """
                    This moves it from 
                            -   (softmax_dim, mini_batch_size, num_head, n_agents, n_agents) to     [2,1,8,20,20]
                                    (softmax_dim * mini_batch_size * num_head * n_agents, n_agents) [320,20]
                """
                attention = attention.view(-1, n_agents)    
                target_attention = target_attention.view(-1, n_agents)

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

        
        log_info["loss_aux"].insert(loss_aux.detach().mean().item())
        log_info["q_values"].insert(q_values.detach().mean().item())
        log_info["q_target"].insert(q_target.detach().mean().item())

        log_info["loss"].insert(loss.item())
        # only log q and attention loss if necessary
        if hasattr(model, "att_weights") and att_regularization_coeff > 0:
            log_info["loss_q"].insert(loss_q.item())
            log_info["loss_att"].insert(loss_att.item())

        if target_update_steps <= 0:
            # smooth target update as in DGN
            interpolate_model(model, model_tar, tau, model_tar)

        elif training_iteration % target_update_steps == 0:
            # regular target update
            model_tar.load_state_dict(model.state_dict())
            tqdm.write(f"Update network, train iteration {training_iteration}")

        
        # save checkpoints
        # if step % args.model_checkpoint_steps == 0 and writer is not None:
        #     torch.save(
        #         get_state_dict(model, netmon, args.__dict__),
        #         Path(writer.get_logdir()) / f"model_{int(step):_d}.pt",

        # print("Succesfull")
        # raise Exception("Terminating the program.")   
    


except Exception as e:
    exception_training = e
    print(exception_training)

finally:
    print("Clean exit")
    del buff

    if writer is not None:
        try:
            # Reset states
            if hasattr(model, "state"):
                model.state = None
            if netmon is not None:
                netmon.state = None

            # Evaluate 
            print("Performing evaluation:")
            metrics = evaluate(
                env, 
                policy,
                eval_episodes,
                eval_episode_steps,
                disable_prog,
                Path(writer.get_logdir()) /"eval",
                eval_output_detailed,
                output_node_state_aux

            )
            paths_to_save = env.save_paths()
            np.save("list_lists", paths_to_save)
            # with open('list_of_lists.json', 'w') as file:
            #     json.dump(paths_to_save, file, indent =2)
            # print(json.dumps(metrics, indent = 4, sort_keys=True, default=str))
            # print(env.planes[0].paths)
            for plane in env.planes:
                print(plane.paths)

            for key in metrics:
                print(key, ": ", metrics[key])

        except Exception as e:
            traceback.print_exc()
            exception_evaluation = e
        finally:

            torch.save(get_state_dict(model, netmon, "args"),
                Path(writer.get_logdir()) / "model_last.pt")
            writer.flush()
            writer.close()
            


if exception_training is not None or exception_evaluation is not None:
    if exception_training is not None and exception_evaluation is not None:
        str_ex = "training and evaluation"
    elif exception_training is not None:
        str_ex = "training"
    elif exception_evaluation is not None:
        str_ex = "evaluation"

    
    raise SystemExit(f"An exception was raised during {str_ex} (see above).")














