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
from model import DGN, MLP, NetMon
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
import yaml


# File path
file_path = "data/sparse_points.json"

# Load sparse_points from the JSON file
with open(file_path, 'r') as f:
    sparse_points = [tuple(point) for point in json.load(f)]  # Convert lists back to tuples

# sparse_points = sparse_points[51:]

# Some start parameters
adj_mat = np.load("data/ad_mat.npy")
dist_mat = np.load("data/dist_mat.npy")

#TEST CODE 
# dist_mat[adj_mat==0] = 0
# min = 1000
# for i in range(131):
#     for j in range(131):
#         if 0< dist_mat[i][j] < min:
#             min = dist_mat[i][j]
# max = np.max(dist_mat)


# Normalize distance matrix
min = 7.617381026011701
max = 484.4574151657363
new_min = 1
new_max = 10
dist_mat = ((dist_mat-min)/(max-min))*(new_max-new_min) + new_min

# raise ValueError("a")

# Load config parameters from a file
with open('data/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Base classes of parameters
cbase = config['base']
cnetmon = config['netmon']
cdgn = config['dgn']
device = config['device']
ctar_update = config['target_update']
ceval = config['evaluation']
ctraining = config['training']
cbuffer = config['buffer']
clog_buffer = config['log_buffer']
ceps = config['epsilon_greedy']



# Define network environment
network = Network(adj_mat, dist_mat, sparse_points)


# Define type of environment
env = Routing(network, cbase['n_planes'], cbase['env_var'], adj_mat, dist_mat, k=cbase['n_neighbors'], enable_action_mask=False)

# Define activation function
activation_function = getattr(F, cbase['activ_f'])

# Dynamically resets the environment
n_agents, agent_obs_size, n_nodes, node_obs_size = reset_and_get_sizes(env) 


print("Sizes before netmon:")
print("Agent observation size: ", agent_obs_size)
print("Node observation size: ", node_obs_size)


# Use NetMon - init is rather long :)
netmon = NetMon(node_obs_size,  # 'in_features' in init
                cnetmon['dim'],     # 'hidden_features' in init
                cnetmon['enc_dim'] , # 'encoder_units' in init
                iterations=cnetmon['iterations'], 
                activation_fn=activation_function,
                rnn_type= cnetmon['rnn_type'], rnn_carryover=cnetmon['rnn_carryover'], agg_type=cnetmon['agg_type'],
                output_neighbor_hidden=cnetmon['neighbor'], output_global_hidden=cnetmon['global']
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
env = NetMonWrapper(env, netmon, cnetmon['start_up_iters'])
_, agent_obs_size, _, _ = reset_and_get_sizes(env)  # Observation length

print("Sizes after netmon:")
print(f"Node state size: {node_state_size}")        # 256
print(f"Agent observation size with netmon: {agent_obs_size}")  # 3263
print(f"Node auxiliary size: {node_aux_size}")      # 0


# Below we select the model that we want to use 
# We can choose from 'dgn', 'dqnr', 'commnet', 'dqn'

# In_features are 'agent_obs_size'
# 'env.action_space.n' is equal to the number of neighbors - choices, 'num_actions' in DGN definition
model = DGN(agent_obs_size, cdgn['hidden_dim'], env.action_space.n, cdgn['heads'], cdgn['att_layers'], activation_function).to(device)


model_tar = copy.deepcopy(model).to(device)     # Create a deep copy of the current model == DGN
model = model.to(device)
model_has_state = False #hasattr(model, "state")
aux_model = None
# print(model_has_state)


# Choosing the policy for decisions, 'env.action_space.n' is equal to the number of choice the agent can perform
# It must be train for our purposes
policy = EpsilonGreedy(env, model, env.action_space.n, epsilon=ceps['epsilon'], step_before_train=ctraining['step_before_train'], epsilon_update_freq=ceps['epsilon_update_freq'], epsilon_decay=ceps['epsilon_decay'])


### Training ###
parameters = list(model.parameters()) + list(netmon.parameters())
if node_aux_size > 0 and ctraining['aux_loss_coeff'] > 0:
    aux_model = MLP(node_state_size, (node_state_size, node_aux_size), activation_function, activation_on_output=False)
    aux_model = aux_model.to(device)
    parameters = parameters + list(aux_model.parameters())

# Init optimizer for training
optimizer = optim.AdamW(parameters, lr = ctraining['learning_rate']) # Adam with weight decay

state_len = model.get_state_len() if model_has_state else 0 # 0 for DGN

mask_size = env.network.max_neighbors   # Should be 10
# Init buffer
buff = ReplayBuffer(cbuffer['seed'], cbuffer['capacity'], n_agents, agent_obs_size, state_len, n_nodes, mask_size,
                    node_obs_size, node_state_size, node_aux_size, half_precision=cbuffer['replay_half_precision'])

# Temp log variables
log_reward = Buffer(clog_buffer['size'], (cbase['n_planes'],), np.float32) # Init of buffer


# This is for evaluation and logging 
log_info = defaultdict(lambda: Buffer(clog_buffer['size'], (1,), np.float32))
comment = "_"
if hasattr(env, "env_var"):
    comment += f"R{env.env_var.value}"
comment += "_netmon"
writer = SummaryWriter(comment=comment)


# Define reward
best_mean_reward = -float("inf")
exception_training = None
exception_evaluation = None


# Wrap netmon with DataParallel if multiple GPUs are available
# if device == "cuda":
#     print(f"Using {torch.cuda.device_count()} GPUs for NetMon DataParallel.")
#     netmon = torch.nn.DataParallel(netmon)
#     print(f"Using {torch.cuda.device_count()} GPUs for model DataParallel.")
#     model = torch.nn.DataParallel(model)
#     print(f"Using {torch.cuda.device_count()} GPUs for target model DataParallel.")
    # model_tar = torch.nn.DataParallel(model_tar)

try:
    print("Training with parameters")
    print(netmon_summary)
    print(env)
    # Variables for the transition buffer
    netmon_info = next_netmon_info = (0,0,0)
    buffer_state = buffer_node_state = 0
    buffer_node_aux = 0
    episode_step = None
    current_episode = 0
    episode_done = False
    training_iteration = 0
    disable_prog = True

    #
    log_buffer_size = clog_buffer['size']
    debug_plots = ctraining['debug_plots']
    buffer_plot_last_n = clog_buffer['plot_last_n']
    step_before_train = ctraining['step_before_train']
    step_between_train = ctraining['step_between_train']
    mini_batch_size = ctraining['mini_batch_size']
    sequence_length = ctraining['sequence_length']
    gamma = ctraining['gamma']
    att_regularization_coeff = ctraining['att_regularization_coeff']
    aux_loss_coeff = ctraining['aux_loss_coeff']
    target_update_steps = ctar_update['steps']
    episode_steps = ceval['episode_steps']  # Default 

    # tdqm is used just so everything   looks nice
    for step in tqdm(range(1, ctraining['total_steps']+ 1), miniters= 100, dynamic_ncols = True, disable=disable_prog): # Disable disables progress bar
        
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

        mask = env.network.node_mask[env.old_node_plane_ids,:]
        after_mask = env.network.node_mask[env.new_node_plane_ids,:]


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
        buff.add(obs, 
                    joint_actions,
                    reward,
                    next_obs,
                    adj,
                    next_adj,
                    done,
                    episode_done,
                    buffer_state, 
                    buffer_node_state, 
                    buffer_node_aux, 
                    *netmon_info, 
                    *next_netmon_info,
                    mask,
                    after_mask
                    )

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
                buff.save_node_state_diff_plot(
                    base_path / f"z_img_node_diff_{step}.png",
                    buffer_plot_last_n,
                )
                buff.save_node_state_std_plot(
                    base_path / f"z_img_node_std_{step}.png", buffer_plot_last_n
                )
            
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

        if (step < step_before_train or buff.count < mini_batch_size or step % step_between_train != 0):    
            # step_before_train ensures that we make at least 2000 iterations to collect initial experience from the environment
            #   before we actually start the training - this servers as a warm-up phase -> we populate the replay buffer with experiences
            #print("Skipped for step:", step)
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
                loss_aux = (loss_aux + torch.mean((aux_prediction - batch.node_aux)** 2)/sequence_length)

            # Replace observation in place
            net_obs_dim = network_obs.shape[-1]     # Take last 
            batch.obs[:, :, -net_obs_dim:] = network_obs

            # Let's now remember the correct netmon state for gradient calculation
            last_netmon_state = netmon.state

            # Done episodes, this is here, otherwise masking would be incorrect
            last_batch_episode_done = batch.episode_done
            last_batch_idx = batch.idx

            # Get Node masks
            q_values = model(batch.obs, batch.adj)  # Estimate the expected reward for each possible action
            
            # q_values = q_values * batch.mask #old_node_mask # Multiply by old node_mask
            q_values[batch.mask==0] = -1e9
            

            # Run target module
            with torch.no_grad():
                if model_has_state:
                    # This is here to avoid storing state and next state of the current model
                    model_tar.state = model.state.detach()

                # Get network observation with netmon
                next_network_obs = netmon(batch.next_node_obs, batch.next_node_adj, batch.next_node_agent_matrix)
                next_net_obs_dim = next_network_obs.shape[-1]   #   model_checkpoint_steps
                batch.next_obs[:, :, -next_net_obs_dim:] = next_network_obs

                next_q = model_tar(batch.next_obs, batch.next_adj)
                next_q[batch.after_mask==0] = -1e9 
                next_q_max = next_q.max(dim=2)[0]
            
            if model_has_state:
                # If the next step belongs to a new episode we mask agent state
                state_mask = ~batch.done * (~batch.episode_done).view(-1,1)
                model.state = model.state * state_mask.unsqueeze(-1)


            # DQN target with 1 step bootstrapping
            # Computes the target Q-values for the actions taken
            chosen_action_target_q = (batch.reward + (~batch.done) * gamma * next_q_max)

            
            # Original DGN loss on all actions - even unused ones. This can be adjusted so that it is DQN loss just on selected actions
            # q_target = q_values.detach()
            # q_target = torch.scatter(q_target, -1, batch.action.unsqueeze(-1), chosen_action_target_q.unsqueeze(-1))

            q_target = chosen_action_target_q
            # print("Q_values are of shape: ", q_values.shape)
            # print("Batch is of shape: ", batch.action.unsqueeze(-1).shape)
            # for line in q_values:
            #     print(line)
            # print(batch.action)
            # print(torch.gather(
            #         q_values, -1, batch.action.unsqueeze(-1)#.cuda()
            #     ))
            q_values = torch.gather(
                    q_values, -1, batch.action.unsqueeze(-1)#.cuda()
                ).squeeze(-1)
            # print(q_values)
            
            # Combined value loss for each sample in the batch
            td_error = q_values - q_target
            

            # Update of q-value loss
            loss_q = loss_q + torch.mean(td_error.pow(2)) / sequence_length
            #print(loss_q)


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

        # print(loss_att)
        loss = (loss_q + att_regularization_coeff * loss_att + aux_loss_coeff * loss_aux)
        optimizer.zero_grad()
        loss.backward()


        # Clip gradients to faciliate learning
        torch.nn.utils.clip_grad_value_(parameters, 0.5)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        


        # Logs
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
            interpolate_model(model, model_tar, ctar_update['tau'], model_tar)

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
            # raise ValueError("a")

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
                ceval['episodes'],
                ceval['episode_steps'],
                disable_prog,
                Path(writer.get_logdir()) /"eval",
                ceval['output_detailed'],
                ceval['output_node_state_aux']

            )
            
            paths_to_save = env.save_paths()
          
            # print(json.dumps(metrics, indent = 4, sort_keys=True, default=str))
            # print(env.planes[0].paths)
            for plane in env.planes:
                print(plane.paths)

            for key in metrics:
                print(key, ": ", metrics[key])
            
            # env.plot_trajectory()
            #env.animation()
            # env.animation_2()

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













