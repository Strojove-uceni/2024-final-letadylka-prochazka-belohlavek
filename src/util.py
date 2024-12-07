import os
import random
import torch.nn as nn
import numpy as np
import torch


def interpolate_model(a: nn.Module, b: nn.Module, a_weight: float, target: nn.Module):
    """
    Interpolates model parameters from a and b, saves a_weight * a + (1-a_weight) * b in target.

    :param a: First input module
    :param b: Second input module
    :param a_weight: Weight of the first input module
    :param target: The output module
    """
    a_dict = a.state_dict()
    b_dict = b.state_dict()
    for key in a_dict:
        # store interpolation results in a_dict
        a_dict[key] = a_weight * a_dict[key] + (1 - a_weight) * b_dict[key]

    # target.load_state_dict(a_dict)

    # Load into target
    if isinstance(target, nn.DataParallel):
        target.module.load_state_dict(a_dict)
    else:
        target.load_state_dict(a_dict)


def get_state_dict(model, netmon, args):
    state_dict = {
        "type": type(model).__name__,
        "state_dict": model.state_dict(),
        "args": args,
    }
    if netmon is not None:
        state_dict["netmon_state_dict"] = netmon.state_dict()

    return state_dict


def load_state_dict(state_dict, model, netmon):
    if state_dict["type"] != type(model).__name__:
        print(
            f"Warning: Loader expected {type(model).__name__} "
            f"but found {state_dict['type']}"
        )
    if "netmon_state_dict" in state_dict:
        if netmon is None:
            raise ValueError("Model uses NetMon which has not been initialized.")
        else:
            #print(state_dict['netmon_state_dict'].keys())
            if isinstance(netmon, nn.DataParallel):
                netmon.module.load_state_dict(state_dict["netmon_state_dict"])
            else:
                netmon.load_state_dict(state_dict["netmon_state_dict"])
    elif netmon is not None:
        raise ValueError("NetMon state could not be found.")

    
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict["state_dict"])
    else:
        model.load_state_dict(state_dict["state_dict"])


def set_attributes(obj, key_value_dict, verbose=False):
    changes_str = ""
    for key in key_value_dict:
        if verbose:
            if hasattr(obj, key) and getattr(obj, key) != key_value_dict[key]:
                changes_str += f"> Updated: {key} = {key_value_dict[key]}" + os.linesep
            if not hasattr(obj, key):
                changes_str += f"> Added: {key} = {key_value_dict[key]}" + os.linesep

        setattr(obj, key, key_value_dict[key])

    if verbose:
        print(changes_str, end="")


def filter_dict(dict, keys):
    # return {key: dict[key] for key in keys}
    to_ret = {}
    to_ret["model"] = dict['base']['model_type']
    to_ret["hidden_dim"] = dict['dgn']['hidden_dim']
    to_ret["dim"] = dict['netmon']['dim']
    to_ret["encoder_dim"] = dict['netmon']['enc_dim']
    to_ret["iterations"] = dict['netmon']['iterations']
    to_ret["rnn_type"] = dict['netmon']['rnn_type']
    to_ret["agg_type"] = dict['netmon']['agg_type']
    to_ret["global"] = dict['netmon']['global']
    to_ret["activation_function"] = dict['base']['activ_f']
    to_ret["num_heads"] = dict['dgn']['heads']
    to_ret["num_attention_layers"] = dict['dgn']['att_layers']  
    return to_ret


def one_hot_list(i, max_indices):
    a = [0] * max_indices
    if i >= 0:
        a[i] = 1
    return a

def set_seed(seed):
    """
    Sets seeds for better reproducibility.

    Disclaimer:
    Note that this method alone does NOT guarantee deterministic
    execution. When using CUDA, there are multiple potential sources of randomness,
    including the execution of RNNs.
    Also see https://pytorch.org/docs/2.0/notes/randomness.html
    and https://pytorch.org/docs/2.0/generated/torch.nn.LSTM.html#torch.nn.LSTM.

    We tested our implementation with
        torch.backends.cudnn.deterministic = True
    and the environment variables
        export CUBLAS_WORKSPACE_CONFIG=:4096:2
        export CUDA_LAUNCH_BLOCKING=1
    but found that some models (e.g. NetMon with GConvLSTM) still show nondeterministic
    behavior, at the cost of almost doubled training time. Because of this, we chose not
    include these settings here.

    :param seed: the seed
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def dim_str_to_list(dims: str):
    if len(dims) == 0:
        return []
    return [int(item) for item in dims.split(",")]