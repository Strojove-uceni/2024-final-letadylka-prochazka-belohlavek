import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, AntiSymmetricConv,GraphSAGE
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn.summary import summary
from layernormlstm import LayerNormLSTMCell
#




# First definining classical MLP
class MLP(nn.Module):
    """
    
    """

    def __init__(self, in_features, mlp_units, activation_fn, activation_on_output = True):
        super(MLP, self).__init__()

        #print("Shape of in_features: ", in_features)
        #print("MLP units are: ", mlp_units) 

        self.activation = activation_fn


        self.linear_layers = nn.ModuleList() # Storage for L layers
        previous_units = in_features
        
        # Transform units into a list
        if isinstance(mlp_units, int):
            mlp_units = [mlp_units]

        # Create a chain of layers
        for units in mlp_units:
            self.linear_layers.append(nn.Linear(previous_units, units))
            previous_units = units


        self.out_features = previous_units
        self.activation_on_ouput = activation_on_output

    # Define forward pass for the MLP 
    def forward(self, x):
        
        # print("Shape of x is: ", x.shape)


        # Inter layers
        for module in self.linear_layers[:-1]:
            if self.activation is not None:
                x = self.activation(module(x))
            else:
                x = module(x)
        
        # Pass through the last layer
        x = self.linear_layers[-1](x)
        if self.activation_on_ouput:
            x = self.activation(x)
        
        return x 
    


# Defining the attention model that will be used in GCN for RL
class AttModel(nn.Module):
    """
    
    """

    def __init__(self, in_features, k_features, v_features, out_features, num_heads, activation_fn, vkq_activation_fn):
        super(AttModel, self).__init__()


        self.k_features = k_features
        self.v_features = v_features
        self.num_heads = num_heads      # Number of attention heads
 
        self.fc_v = nn.Linear(in_features, v_features * num_heads)  # Transforming input features into Values for attention
        self.fc_k = nn.Linear(in_features, k_features * num_heads)  # Transforming input features into Keys for attention
        self.fc_q = nn.Linear(in_features, k_features * num_heads)  # Transforming input values into Queries for attention

        self.fc_out = nn.Linear(v_features * num_heads, out_features)   # Transforms the outputs from all attention heads into output dimension

        self.activation = activation_fn  
        self.vkq_activation = vkq_activation_fn     # Activation function that can be applied into Values, Keys, Queries

        
        """
        Defining the scaling factor for attention as 1/ sqrt(d_k), this is the same as the publishing paper "Attention is All You Need".
        This is done for the purpose of reducing the gradient so it does not become too large. Later you will see that without it, the dot product 
        would grow too large without the scaling
        """
        self.attention_scale = 1 / (k_features **0.5)




    def forward(self, x, mask):
        batch_size, num_agents = x.shape[0], x.shape[1]

        """
        The code below does the following:
            - a linear mapping is applied on the inputs to obtain Values, Keys, Queries
            - the Values, Keys, Queries are then reshaped to separate the different attention heads of the model
            :reshape: will result in (batch_size, num_agents, num_heads, features_per_head)

        Visual representation:
            Input x
               |
            [Linear Layers] -> V, Q, K
               |
            [Optional Activation] (vkq_activation_fn)
               |
            [Reshape for Multi-Head]
               |
            [Transpose for Heads]
               |
            [Compute Attention Weights (Dot Product, Scale, Mask, Softmax)]
               |
            [Apply Attention to Values]
               |
            [Skip Connection]
               |
            [Transpose and Concatenate Heads]
               |
            [Final Linear Layer and Activation]
               |
            Output

        """
        v = self.fc_v(x).view(batch_size, num_agents, self.num_heads, self.v_features)
        q = self.fc_q(x).view(batch_size, num_agents, self.num_heads, self.k_features)
        k = self.fc_k(x).view(batch_size, num_agents, self.num_heads, self.k_features)

        if self.vkq_activation is not None:
            v = self.vkq_activation(v)
            q = self.vkq_activation(q)
            k = self.vkq_activation(k)
    
        # We rearrange the tensors to shape (batch_size, num_heads, num_agents, features_per_head)
        # This is done so we can perform batch multiplication over the batch size and heads
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        # Add head axis (we are keeping the same mask for all attention heads)
        mask = mask.unsqueeze(1)    # (batch_size, 1, num_agents, num_agents) (1,1,20,20)


        # Now we calculate attention
        """
        The attention is calculated as a dot product of all queries with all keys, 
            while scaling it with the attention scale so it does not explode.
            - q is of shape             (batch_size, num_heads, num_agents, features_per_head)
            - k transposed is of shape  (batch_size, num_heads, features_per_head, num_agents)
            - the multiplication result is of shape (batch_size, num_heads, num_agents, num_agents)

        :masked_fill sets positions where mask == 0 to a large negative value - removes them from the attention computation practically
        """
        att_weights = torch.matmul(q, k.transpose(2, 3)) * self.attention_scale
        att = att_weights.masked_fill(mask==0, -1e9)
        att = F.softmax(att, dim=-1)    # Softmax is applied along the last dimension to obtain normalized attention probabilities

        # Now we combine the Values with respect to the attention we just computed
        """
            - att is of shape (batch_size, num_heads, num_agents, num_agents)
            - v is of shape (batch_size, num_heads, num_agents, v_features)
            - the multiplication result is of shape (batch_size, num_heads, num_agents, v_features)
        """
        out = torch.matmul(att, v) 

        # We add a skip connection 
        out  = torch.add(out, v)    # This additionally promotes gradient flow and mitigates vanishing gradient

        # Now "remove" the transpose and concatenate all heads together
        """
            - out is of shape (batch_size, num_heads, num_agents, v_features)
            - out after transpose is of shape (batch_size, num_agents, num_heads, v_features)
            - contiguous() ensures that the tensor is stored in a contiguous chunk of memory so that the reshape for view can happen
            - view is used to reshape the tensor to (batch_size, num_agents, v_features), therefore, we flatten the last two dimensions
                into a single one (num_heads * v_features)
            - final out is of shape  (batch_size, num_agents, num_heads * v_features)
        """
        out = out.transpose(1,2).contiguous().view(batch_size, num_agents, -1) 

        out = self.activation(self.fc_out(out)) # Linear map into a desired feature dimension

        return out, att_weights


class Q_Net(nn.Module):
    """
    This servers as the Q-function  approximator in RL. It estimates Q-values for each possible action given a particular state.
    What are Q-values? Rewards.
    So, given a particular state, this estimates the expected future rewards(Q-values) for each possible action our plane(agent) can take.
    """
    def __init__(self, in_features, actions):
        super(Q_Net, self).__init__()
        
        print("In features are: ", in_features)
        self.fc = MLP(in_features, (256,actions), None, False)#nn.Linear(in_features, actions)

    def forward(self, x):
        return self.fc(x)
    
class SimpleAggregation(nn.Module):
    def __init__(self, agg: str, mask_eye: bool) -> None:
        super().__init__()
        self.agg = agg
        assert self.agg == "mean" or self.agg == "sum"
        self.mask_eye = mask_eye

    def forward(self, node_features, node_adjacency):
        if self.mask_eye:
            node_adjacency = node_adjacency * ~(
                torch.eye(
                    node_adjacency.shape[1],
                    node_adjacency.shape[1],
                    device=node_adjacency.device,
                )
                .repeat(node_adjacency.shape[0], 1, 1)
                .bool()
            )
        feature_sum = torch.bmm(node_adjacency, node_features)
        if self.agg == "sum":
            return feature_sum
        if self.agg == "mean":
            num_neighbors = torch.clamp(node_adjacency.sum(dim=-1), min=1).unsqueeze(-1)
            return feature_sum / num_neighbors


class DGN(nn.Module):
    """
    
    """

    def __init__(self, in_features, mlp_units, num_actions, num_heads, num_attention_layers, activation_fn, kv_values):
        super(DGN, self).__init__()
        
        self.encoder = MLP(in_features, mlp_units, activation_fn)
        self.att_layers = nn.ModuleList()
        hidden_features = self.encoder.out_features

        #print("In features of DGN: ", in_features)
        #print("MLP units are: ", mlp_units)

        for _ in range(num_attention_layers):
            self.att_layers.append(
                AttModel(hidden_features, kv_values, kv_values, hidden_features, num_heads, activation_fn, activation_fn) 
                                   )
        
        self.q_net = Q_Net(hidden_features * (num_attention_layers + 1), num_actions)

        self.att_weights = []

    def forward(self, x, mask):
        """
        Additional comment to the function:
            - each attention layer refines the representation h by focusing on relevant parts of the input
            - by concatenating the representations the feature set for the Q-network is enhanced, consequently making more informed decisions
        
        """

        h = self.encoder(x)     # Encodes the input featuers, has a shape of (batch_size, num_agents, hidden_features)

        q_input = h     # Initialize the q_input with encoded features

        self.att_weights.clear()    # Ensuring that attention weights from previous forward passes do not accumulate

        for attention_layer in self.att_layers:
            h, att_weight = attention_layer(h, mask)
            self.att_weights.append(att_weight)

            # Concatenation of outputs 
            q_input = torch.cat((q_input, h), dim=-1)

        # Final q_input is of shape (batch_size, num_agents, hidden_features * (num_attention_layers +1))
        q = self.q_net(q_input)


        # print("Predicted Q_values:")
        # for i in range(5):
        #     tensor = q[0,i,:].numpy()
        #     formatted = ", ".join(f"{x:.2f}" for x in tensor)
        #     print(formatted)
        #     print("")
    
    
        return q    # is of shape (batch_size, num_agents, num_action)
        

class DQN(nn.Module):
    """
    Introduces Deep Feed Forward Neural Network( = MLP) as the encoder.
    """

    def __init__(self, in_features, mlp_units, num_actions, activation_fn):
        super(DQN, self).__init__()

        self.encoder = MLP(in_features, mlp_units, activation_fn)   # Encodes incoming features
        self.q_net = Q_Net(self.encoder.out_features, num_actions)  # Outputs Q-values
        self.activation = activation_fn

    def forward(self, x, mask):
        batch, agent, features = x.shape    # This is here in the original implementation so i will leave it here for now
        h = self.encoder(x)
        q = self.q_net(h)
        return q


class NetMon(nn.Module):
    """
    Why does this even exist?
        - processing observations from nodes in the graph
        - performs message passing
        - aggregation of information from neighboring nodes
        - updating node states with RNN
        - produces embeddings or features for nodes or agents in the graph
    """
    def __init__(self, in_features, hidden_features: int, encoder_units, iterations, activation_fn, 
                rnn_type="lstm",
                rnn_carryover=True, agg_type="sum", 
                output_neighbor_hidden = False, output_global_hidden = False
    ) -> None:
        super().__init__()

        assert isinstance(hidden_features, int)
        
        print("In-features to Netmon encoder:", in_features)
        self.encode = MLP(in_features, (*encoder_units, hidden_features), activation_fn)    # Define simple MLP as the endocer function
        self.state = None
        self.output_neighbor_hidden = output_neighbor_hidden
        self.output_global_hidden = output_global_hidden
        self.rnn_carryover = rnn_carryover
        self.iterations = iterations

        # 0 = dense input - dense matricies
        # 1 = sparse input - sparse matricies
        # 2 = gconvlstm (sparse input) - uses sparse inputs + own RNN integration
        # 3 = GraphSAGE (sparse input, directly outputs neighbor info)
        self.aggregation_def_type = None

        # Agreggation
        self.agg_type_str = agg_type    # GCN
        """
        Here they resolve different aggregation functions for different types of networks like GraphSAGE etc. 
        """
     

        self.jk = None
        if "jk-cat" in agg_type:
            self.jk_out = nn.Linear(hidden_features * iterations, hidden_features)      # Linear layer for final representation after concatenation
            self.jk_neighbors = nn.Linear(hidden_features * (iterations - 1), hidden_features)      # Linear layer for the neighbor representations, but without the last iteration
            
            def jk_cat(xs):
                # xs is a list of hidden states from different iterations
                return (self.jk_out(torch.cat(xs, dim=-1)),   # Those are the concatenated -> transformed node representations
                        self.jk_neighbors(torch.cat(xs[:-1], dim=-1)))      # Those are the concatenated and transformed neighbor representations without the last iteration

            self.jk = jk_cat

        elif "jk-max" in agg_type:

            def jk_max(xs):
                # Uses element-wise maximum across the outputs from different layers - stacks the hidden states and computes the maximum along the iteration dimension
                return (torch.max(torch.stack(xs), dim=0)[0],  # The maximum node representations across iterations
                        torch.max(torch.stack(xs[:-1]), dim=0)[0])  # The maximum neighbor representations without the last iteration
            
            self.jk = jk_max

        elif agg_type in ["graphsage", "adgn"]:

            def jk(xs):
                return (xs[-1], xs[-2])  # No explicit jumping knowledge method is applied - we simply return the last two hidden states
            
            self.jk = jk


        # Now we will resolve the actual aggregation with the individual networks
        if agg_type == "sum" or agg_type == "mean":
            self.aggregate = SimpleAggregation(agg=agg_type, mask_eye=False)
            self.aggregation_def_type = 0
        elif agg_type == "gcn":
            self.aggregate = GCNConv(hidden_features, hidden_features, improved=True)
            self.aggregation_def_type = 1
        elif agg_type == "sage":
            self.aggregate = SAGEConv(hidden_features, hidden_features)
            self.aggregation_def_type = 1
        elif "graphsage" in agg_type:
            self.aggregate = GraphSAGE(hidden_features, hidden_features, num_layers=iterations)
            self.agg_type_str = agg_type + f" ({iterations} layer)"
            assert self.jk is not None
            self.aggregate.jk = self.jk
            self.aggregate_jk_mode = "custom"
            self.aggregation_def_type = 3
            self.iterations = 1
            if rnn_type != "none":
                print(f"WARNING: Overwritten given rnn type {rnn_type} with 'none'.")
                rnn_type = "none"
        elif "adgn" in agg_type:
            self.aggregate = JumpingKnowledgeADGN(hidden_features, num_iters=iterations, jk = self.jk)
            self.agg_type_str = agg_type + f" ({iterations} layer)"
            self.aggregation_def_type = 3
            self.iterations = 1


            if rnn_type != "none":
                print(f"WARNING: Overwritten given rnn type {rnn_type} with 'none'.")
                rnn_type = "none"
        elif agg_type == "antisymgcn":
            # Use single iteration so that we still get the last hidden node states
            self.aggregate = AntiSymmetricConv(hidden_features, num_iters=1)
            self.aggregation_def_type = 1
        elif agg_type == "gconvlstm":
            pass
            # Filter size 1 => only from neighbors
            # from torch_geometric_temporal.nn.recurrent.gconv_lstm improt GCNConvLSTM

            # self.agg_type_str = agg_type + f" (filter size {iterations - 1})"
            # self.aggregate = GCNConvLSTM(hidden_features, hidden_features, K=(self.iterations + 1))
            if rnn_type != "gconvlstm":
                print(f"WARNING: Overwritten given rnn type {rnn_type} with 'gconvlstm'")
                rnn_type = "gconvlstm"
        else:
            raise ValueError(f"Unknown aggregation type {agg_type}")
        


        # Update and observation encoding
        self.rnn_type = rnn_type    # lstm

        if self.rnn_type == "lstm":
            self.rnn_obs = nn.LSTMCell(hidden_features, hidden_features)
            self.rnn_update = nn.LSTMCell(hidden_features, hidden_features)
            self.num_states = 2 if rnn_carryover else 4 # 2
        elif self.rnn_type == "lnlstm":
            self.rnn_obs = LayerNormLSTMCell(hidden_features, hidden_features)
            self.rnn_update = LayerNormLSTMCell(hidden_features, hidden_features)
            self.num_states = 2 if rnn_carryover else 4
        elif self.rnn_type == "gru":
            self.rnn_obs = nn.GRUCell(hidden_features, hidden_features)
            self.rnn_update = nn.GRUCell(hidden_features, hidden_features)
            self.num_states = 1 if rnn_carryover else 2
        elif self.rnn_type == "gconvlstm":
            # rnn is part of aggregate function
            self.num_states = 2
        elif self.rnn_type == "none":
            # empty state / stateless => simply store h for debugging
            self.num_states = 1
        else:
            raise ValueError(f"Unknown rnn type {self.rnn_type}")

        self.hidden_features = hidden_features
        self.state_size = hidden_features * self.num_states     # 256



    def forward(self, x, mask, node_agent_matrix, max_degree=None, no_agent_mapping = False):

        # print("Shape of x is: ", x.shape)
        # print("Shape of mask is: ", mask.shape)
        
        # This function contains steps (1), (2) and (3)
        h, last_neighbor_h = self.update_node_states(x, mask)

        # Step (4), Check what type of node states to aggregate. Either global or neighbor 
        if self.output_neighbor_hidden or self.output_global_hidden:
            extended_h = [h]

            # Aggregate neighbors
            if self.output_neighbor_hidden:
                extended_h.append(
                        self.get_neighbor_h(last_neighbor_h, mask, max_degree)
                    )

            # Aggregate global
            if self.output_global_hidden:
                extended_h.append(self.get_global_h(h))

            h = torch.cat(extended_h, dim=-1)   # Concatenates all features along the last dimension


        if no_agent_mapping: 
            return h

        return NetMon.output_to_network_obs(h, node_agent_matrix)


    def get_state_size(self):
        return self.state_size

    def get_global_h(self, h):
        """
        Computes a global summary of the nodes states and appends it to each Node's representation.
        """
        _, n_nodes, _ = h.shape     # (batch_size, n_nodes, hidden_size)

        """
            - mean(dim=1) computes the mean along all nodes for each batch
                -> (batch_size, hidden_size)
            - repeat repeats the global hidden state n_node times along a new dimension -> (n_nodes, batch_size, hidden_size) 
            - transpose resutls in (batch_size, n_nodes, hidden_size)
        """
        global_h = h.mean(dim=1).repeat((n_nodes,1,1)).transpose(0,1)
        return global_h

    def get_neighbor_h(self, neighbor_h, mask, max_degree):
        """
        Computes a summary based on Nodes neighbors and appends it to each Node's representatino.
        """
        batch_size, n_nodes , _ = neighbor_h.shape

        # Get max node id for dense observation tensor (excludes self)
        if max_degree is None:  # The maximum number of neigbors for each node -> if it is none -> compute from adjacency matrix
            max_degree = torch.sum(mask, dim=1).max().long().item() - 1
        
        # Pre-allocate a placeholder for each neighbor
        h_neighbors = torch.zeros((batch_size, n_nodes, max_degree, neighbor_h.shape[-1]), device = neighbor_h.device)

        # Get mask without self (pure neighbors)
        neighbor_mask = mask * ~(                               # ~ is negation -> creates a matrix of ones where diagonal is 0
            torch.eye(n_nodes, n_nodes, device=mask.device)
            .unsqueeze(0)   # Add dimension to the 0th positions
            .repeat(mask.shape[0], 1, 1) # Repeat the tensor mask.shape[0] times along the first dimension and once along the second and the third
            .bool()
        )

        # Now we want to collect features from neighbors 
        h_index = neighbor_mask.nonzero()

        # Get the relative neighbor ID for the insertion into h_neighbors
        cumulative_neighbor_index = neighbor_mask.cumsum(dim=-1).long() - 1     # Cumulatively sums the neighbor mask along the last dimension to assign a unique index to each neighbor per node
        h_neighbors_index = cumulative_neighbor_index[h_index[:,0], h_index[:, 1], h_index[:, 2]] # TODO: This maybe relates to the fact that they assume 3 neighbors

        # Copy the last hidden state of all neighbors into the hidden state tensor
        h_neighbors[h_index[:,0], h_index[:, 1], h_neighbors_index] = neighbor_h[h_index[:,0], h_index[:, 2]]   # For each neighbor connection, copies neighbor's hidden state into h_neighbors to corresponding position

        # Concatenate info for each node
        return h_neighbors.reshape(batch_size, n_nodes, -1)     # Reshape from (batch_size, n_nodes, max_degree, hidden_size) to (batch_size, n_nodes, max_degree*hidden_size)
                                                                # |
                                                                #  -> each node has a concatenated vector of its neighbors' hidden states


    def update_node_states(self, x, mask):
        """
        This function performs message passing and state updates over a specified number of iterations.
        It integrates both node features and graph structure.

        :mask: it is the adjacency matrix of the graph -> (131,131)
        """
        batch_size, n_nodes, feature_dim = x.shape # (1, 131, 1463)
        
        x = x.reshape(batch_size * n_nodes, -1)  # New shape is (batch_size * n_nodes, feature_dim)
        # x is of shape (131,1463)

        if self.state == None: # For storing hidden states
            # Init
            self.state = torch.zeros((batch_size, n_nodes, self.state_size), device = x.device) 
        
        # Reshape the state before further processing
        self.state_reshape_in(batch_size, n_nodes)  


        # step (1): encode observation to get h^0_v and combine with state
        h = self.encode(x)  # Producing initial hidden representations

        # Choose what we are using. Either LSTM or GRU
        if self.rnn_type in ["lstm", "lnlstm"]:
            h0, cx0 = self.rnn_obs(h, (self.state[0], self.state[1])) # rnn.obs processes the encoded features h along with the previous states
            h, cx = h0, cx0

        # Message passing iterations
        if self.iterations <= 0 and self.output_neighbor_hidden:
            last_neighbor_h = torch.zeros_like(h, device=h.device)  # Returns a tensor filled with 0s in the shape of h
        else:
            last_neighbor_h = None

        if self.aggregation_def_type != 0:
            mask_sparse, mask_weights = dense_to_sparse(mask)   # Conversion to a sparse representation for the aggreagtion function

        if self.aggregation_def_type == 2:
            H, C = self.state[0], self.state[1] # Init of additional aggregation types 

        # Iteration
        for it in range(self.iterations):
            if self.output_neighbor_hidden and it == self.iterations-1:
               
                if self.aggregation_def_type == 2:
                    # we know that the aggregation step will exchange the hidden states
                    # (and much more..) so we can just use them for the skip connection
                    # instead of the other nodes' input.
                    # This is only relevant for a single iteration per step.
                    last_neighbor_h = H
                else:
                    # use the last received hidden state
                    last_neighbor_h = h


            # step (2): aggregate - computes the aggregated messages M for each node.
            if self.aggregation_def_type == 0:  # Simple aggregation
                M = self.aggregate(h.view(batch_size, n_nodes, -1), mask).view(
                    batch_size * n_nodes, -1
                )
            elif self.aggregation_def_type == 1:    # Aggregation through conv. layers with sparse mask
                M = self.aggregate(h, mask_sparse)

            elif self.aggregation_def_type == 2:    # Specialized aggregation with additional states
                H, C = self.aggregate(h, mask_sparse, H=H, C=C)
                M = H
            elif self.aggregation_def_type == 3:    # Uses models like GraphSAGE etc. proly won't be useful to us
                # overwrite last_neighbor_h with jumping knowledge output
                M, last_neighbor_h = self.aggregate(h, mask_sparse)


            # step (3): update - it is performed using RNN cell with the aggregated messages M
            """
            What is the carryover mechanism?
                Carryover mechanism controls whether to carry over states between iterations or reset them
            """
            if self.rnn_type in ["lstm", "lnlstm"]:
                if not self.rnn_carryover and it == 0:
                    rnn_input = (self.state[2], self.state[3])
                else:
                    rnn_input = (h, cx)

                h1, cx1 = self.rnn_update(M, rnn_input)
                h, cx = h1, cx1
            elif self.rnn_type == "gru":
                if not self.rnn_carryover and it == 0:
                    rnn_input = self.state[1]
                else:
                    rnn_input = h

                h1 = self.rnn_update(M, rnn_input)
                h = h1
            else:
                h = M


        # Reshaping
        if last_neighbor_h is not None:
            last_neighbor_h = last_neighbor_h.reshape(batch_size, n_nodes, -1)  # Reshaping to original dimensions for output
        h = h.reshape(batch_size, n_nodes, -1)  # Reshaping to original dimensions for output


        # Updating of the internal state
        if self.rnn_type in ["lstm", "lnlstm"]:
            if self.rnn_carryover:
                self.state = torch.stack((h1, cx1))     # Concatenating tensors along a new dimension
            else:
                self.state = torch.stack((h0, cx0, h1, cx1))

        elif self.rnn_type == "gru":
            if self.rnn_carryover:
                self.state = h1.unsqueeze(0)
            else:
                self.state = torch.stack((h0.unsqueeze(0), h1.unsqueeze(0)))

        elif self.rnn_type == "gconvlstm":
            self.state = torch.stack((H, C))

        elif self.rnn_type == "none":
            # store last node state for debugging and aux loss
            self.state = h.unsqueeze(0)

        self.state_reshape_out(batch_size, n_nodes)

        return h, last_neighbor_h

    def state_reshape_in(self, batch_size, n_agents):
        """
        Reshapes the state of shape (batch_size, n_agents, self.get_state_len())
            to shape
                (2, batch_size * n_agents, hidden_size)
        """

        if self.state.numel() == 0:
            return

        self.state = self.state.reshape(batch_size * n_agents, self.num_states, -1).transpose(0,1)

    def state_reshape_out(self, batch_size, n_agents):
        """
        Reshapes the state of shape
            (2, batch_size * n_agents, hidden_size)
        to shape
            (batch_size, n_agents, self.get_state_len()).

        :param batch_size: the batch size
        :param n_agents: the number of agents
        """
        if self.state.numel() == 0:
            # print("TRUE")
            return

        self.state = self.state.transpose(0, 1).reshape(batch_size, n_agents, -1)

    
    @staticmethod
    def output_to_network_obs(netmon_out, node_agent_matrix):
        """
        Netmon_out is called within the forward function. Why? It performs the mapping of the node information to agents.
            It multiplies the node outputs with node_agent_matrix to aggregate/map node outputs to agent-specific outputs.
        """
        # print(netmon_out.shape)
        # print(node_agent_matrix.shape)
        # bmm performs a batch matrix-matrix product of matricies stored in netmon_out.transpose(1,2) and node_agent_matrix
        return torch.bmm(netmon_out.transpose(1, 2), node_agent_matrix).transpose(1, 2)
    
    def summarize(self, *args):
        """
        Additional function to just summarize the current model.
        """
        str_out = []
        str_out.append("NetMon Module")
        str_out.append(summary(self, *args, max_depth=10))
        self.state = None
        str_out.append(f"> Aggregation Type: {self.agg_type_str}")
        str_out.append(f"> RNN Type: {self.rnn_type}")
        str_out.append(f"> Carryover: {self.rnn_carryover}")
        str_out.append(f"> Iterations: {self.iterations}")
        readout_str = "> Readout: local"
        if self.output_neighbor_hidden:
            readout_str += " + last neighbors"
        if self.output_global_hidden:
            readout_str += " + global agg"
        str_out.append(readout_str)
        import os

        return os.linesep.join(str_out)
    
    def get_out_features(self):
        """
        Just a helper function. It is used within the sl.py file.
        """
        out_features = self.hidden_features

        if self.output_neighbor_hidden:
            out_features += self.hidden_features * 3

        if self.output_global_hidden:
            out_features += self.hidden_features

        return out_features


class DQNR(nn.Module):
    """
    Recurrent DQN with an lstm cell.
    """

    def __init__(self, in_features, mlp_units, num_actions, activation_fn):
        super(DQNR, self).__init__()
        self.encoder = MLP(in_features, mlp_units, activation_fn)
        self.lstm = nn.LSTMCell(
            input_size=self.encoder.out_features, hidden_size=self.encoder.out_features
        )
        self.state = None
        self.q_net = Q_Net(self.encoder.out_features, num_actions)

    def get_state_len(self):
        return 2 * self.lstm.hidden_size

    def _state_reshape_in(self, batch_size, n_agents):
        """
        Reshapes the state of shape
            (batch_size, n_agents, self.get_state_len())
        to shape
            (2, batch_size * n_agents, hidden_size).

        :param batch_size: the batch size
        :param n_agents: the number of agents
        """
        self.state = (
            self.state.reshape(
                batch_size * n_agents,
                2,
                self.lstm.hidden_size,
            )
            .transpose(0, 1)
            .contiguous()
        )

    def _state_reshape_out(self, batch_size, n_agents):
        """
        Reshapes the state of shape
            (2, batch_size * n_agents, hidden_size)
        to shape
            (batch_size, n_agents, self.get_state_len()).

        :param batch_size: the batch size
        :param n_agents: the number of agents
        """
        self.state = self.state.transpose(0, 1).reshape(batch_size, n_agents, -1)

    def _lstm_forward(self, x, reshape_state=True):
        """
        A single lstm forward pass

        :param x: Cell input
        :param reshape_state: reshape the state to and from (batch_size, n_agents, -1)
        """
        batch_size, n_agents, feature_dim = x.shape
        # combine agent and batch dimension
        x = x.view(batch_size * n_agents, -1)

        if self.state is None:
            lstm_hidden_state, lstm_cell_state = self.lstm(x)
        else:
            if reshape_state:
                self._state_reshape_in(batch_size, n_agents)
            lstm_hidden_state, lstm_cell_state = self.lstm(
                x, (self.state[0], self.state[1])
            )

        self.state = torch.stack((lstm_hidden_state, lstm_cell_state))
        x = lstm_hidden_state

        # undo combine
        x = x.view(batch_size, n_agents, -1)
        if reshape_state:
            self._state_reshape_out(batch_size, n_agents)

        return x

    def forward(self, x, mask):
        h = self.encoder(x)
        h = self._lstm_forward(h)
        return self.q_net(h)


class CommNet(DQNR):
    """
    Implementation of CommNet https://arxiv.org/abs/1605.07736 with masked communication
    between agents.

    While the hidden state is aggregated over the neighbors during communication, the
    individual cell states stay the same. This is how IC3Net implemented CommNet
    https://github.com/IC3Net/IC3Net. The CommNet paper does not elaborate on if and how
    the cell states are combined.
    """

    def __init__(
        self,
        in_features,
        mlp_units,
        num_actions,
        comm_rounds,
        activation_fn,
    ):
        super().__init__(in_features, mlp_units, num_actions, activation_fn)
        assert comm_rounds >= 0
        self.comm_rounds = comm_rounds

    def forward(self, x, mask):
        batch_size, n_agents, feature_dim = x.shape
        h = self.encoder(x)

        # manually reshape state
        if self.state is not None:
            self._state_reshape_in(batch_size, n_agents)

        h = self._lstm_forward(h, reshape_state=False)

        # explicitly exclude self-communication from mask
        mask = mask * ~torch.eye(n_agents, dtype=bool, device=x.device).unsqueeze(0)

        for _ in range(self.comm_rounds):
            # combine hidden state h according to mask
            # first add up hidden states according to mask
            #    h has dimensions (batch, agents, features)
            #    and mask has dimensions (batch, agents, neighbors)
            #    => we have to transpose the mask to aggregate over all neighbors
            c = torch.bmm(h.transpose(1, 2), mask.transpose(1, 2)).transpose(1, 2)
            # then normalize according to number of neighbors per agent
            c = c / torch.clamp(mask.sum(dim=-1).unsqueeze(-1), min=1)

            # skip connection for hidden state and communication
            h = h + c
            # use new hidden state
            self.state[0] = h.view(batch_size * n_agents, -1)

            # pass through forward module
            h = self._lstm_forward(h, reshape_state=False)

        # manually reshape state in the end
        self._state_reshape_out(batch_size, n_agents)
        return self.q_net(h)


