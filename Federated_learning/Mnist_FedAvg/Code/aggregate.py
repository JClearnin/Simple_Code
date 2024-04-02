import torch
import numpy as np

def server_aggregate(client_sum, local_weights):
    """
    Aggregates the weights using JS_weights as the weights for aggregation.
    """
    client_weight = [1 / client_sum] * client_sum
    aggregated_weights = dict()
    for key in local_weights[0].keys():
        aggregated_weights[key] = torch.zeros_like(local_weights[0][key])
        for i in range(len(local_weights)):
            weight_tensor = torch.from_numpy(np.array(client_weight[i]))
            aggregated_weights[key] += weight_tensor.to(local_weights[i][key].dtype) * local_weights[i][key]
    return aggregated_weights
