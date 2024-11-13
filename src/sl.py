import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm


# Our modules
from model import NetMon
from util import dim_str_to_list, set_seed

# Our training model
class NetMonSL(nn.Module):


    def __init__(self, node_obs, dim, nb_classes, nb_nodes, config) -> None:
        super().__init__()

        # Init netmon with params
        self.netmon = NetMon(node_obs, config['netmon_dim'], dim_str_to_list(config['netmon_encoder_dim']), iterations=config['netmon_iterations'],
                            activation_fn=F.leaky_relu, rnn_type=config['netmon_rnn_type'], rnn_carryover=config['netmon_rnn_carryover'],
                            agg_type=config['netmon_agg_type'], output_neighbor_hidden=config['netmon_last_neighbors'],
                            output_global_hidden=config['netmon_global']                              )
        

        self.linear = nn.Linear(self.netmon.get_out_features(), nb_classes)
        self.lienar_reg = nn.Linear(self.netmon.get_out_features(), 1)
        self.linear_reg_all = nn.Linear(self.netmon.get_out_features(), nb_nodes)
        self.class_logits = None


    def forward(self, node_obs, node_adj):
        batches, nodes, features = node_obs.shape
        eye = torch.eye(nodes).repeat(batches, 1,1)

        node_features = self.netmon(node_obs, node_adj, eye)
        class_logits = self.linear(node_features)
        pred = self.lienar_reg(node_features)
        pred_all = self.linear_reg_all(node_features)

        self.class_logits = class_logits.detach()
        return class_logits,  pred, pred_all



    def get_class_probabilities(self):
        return torch.softmax(self.class_logits, dim=-1)

    def get_prediction(self):
        return torch.argmax(self.get_class_probabilities(), axis=-1)
