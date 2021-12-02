## SLURP on optical flow task
It is implemented on FlowNet: Learning Optical Flow with Convolutional Networks. Links for BTS paper, Supp. and codes are as follows:

Check [arXiv](https://arxiv.org/abs/1504.06852).

Check [here](https://github.com/ClementPinard/FlowNetPytorch) for its pytorch implementation by Clément Pinard.

For training SLURP, we have to add several extra python files to it, such as `train_slurp.py`, `uncertainty.py`, `flownet_test_uncer.py`, `sparsification.py`. but it doesn't need to change anything with respect to the original version. Moreover, `uncertainty.py` and `sparsification.py` could be re-utilised in other tasks without making changes, such as in monocular depth task.

## Requirements:
Same as the ones in the FlowNet pytorch implementation.

## Steps:
To train SLURP, you can use the following command:

1/ A trained FlowNetS checkpoint on FlyingChairs-official-split.

SLURP sequential training requires the existing main task checkpoint (which is FlowNetS here)

2/ Training a side learner as uncertainty estimator:
    
    python train_slurp.py arguments_train_slurp_flow.py

Note: The dataset paths, main task model path, auxiliary file paths and other relative settings (channel number of output uncertainty, etc.) have to be modified. The principle settings are the same as the original, such as batch size, input size, etc.

This example aims to show that if the main task model's encoder is hard to use or the main task model is a black box, we can train an other encoder for image from beginning.