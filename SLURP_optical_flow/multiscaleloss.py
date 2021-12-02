import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size

def NLL(input_var, output_flow, target_flow, sparse=False, mean=True):
    batch_size = target_flow.size(0)
    squared_difference = (target_flow - output_flow)**2
    input_var_exp = torch.exp(input_var) + 1e-6
    const = torch.sum(torch.log(input_var_exp), dim=1, keepdim=True)
    likelihood_lap = torch.sqrt(
        torch.sum(squared_difference / input_var_exp, dim=1, keepdim=True))
    loss_map = const + likelihood_lap
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        loss_map = loss_map[~mask]

    if mean:
        return loss_map.mean()
    else:
        return loss_map.sum()/batch_size

def BCE(input_var, output_flow, target_flow, sparse=False, mean=True):
    batch_size = target_flow.size(0)
    abs_difference = torch.tanh(abs(target_flow - output_flow))
    loss_map = nn.BCEWithLogitsLoss(reduction = 'none')(input_var, abs_difference)
    loss_map = torch.sum(loss_map, dim=1, keepdim=True)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        loss_map = loss_map[~mask]

    if mean:
        return loss_map.mean()
    else:
        return loss_map.sum()/batch_size

def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.

    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):

        b, _, h, w = output.size()

        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
        else:
            target_scaled = F.interpolate(target, (h, w), mode='area')
        return EPE(output, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)
    return loss

def multiscaleNLL_detach(network_output, output_flows, target_flow, weights=None, sparse=False):
    def one_scale(output, output_flow, target, sparse):

        b, _, h, w = output.size()

        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
            flow_scaled = sparse_max_pool(output_flow, (h, w))
        else:
            target_scaled = F.interpolate(target, (h, w), mode='area')
            flow_scaled = F.interpolate(output_flow, (h, w), mode='area')
        return NLL(output, flow_scaled, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, output_flows.detach(), target_flow, sparse)
    return loss

def multiscaleBCE_detach(network_output, output_flows, target_flow, weights=None, sparse=False):
    def one_scale(output, output_flow, target, sparse):

        b, _, h, w = output.size()

        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
            flow_scaled = sparse_max_pool(output_flow, (h, w))
        else:
            target_scaled = F.interpolate(target, (h, w), mode='area')
            flow_scaled = F.interpolate(output_flow, (h, w), mode='area')
        return BCE(output, flow_scaled, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, output_flows.detach(), target_flow, sparse)
    return loss

def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, mean=True)

def realNLL(output, output_flow, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_flow = F.interpolate(output_flow, (h,w), mode='bilinear', align_corners=False)
    upsampled_var = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return NLL(upsampled_var, upsampled_flow, target, sparse, mean=True)

def realBCE(output, output_flow, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_flow = F.interpolate(output_flow, (h,w), mode='bilinear', align_corners=False)
    upsampled_var = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return BCE(upsampled_var, upsampled_flow, target, sparse, mean=True)