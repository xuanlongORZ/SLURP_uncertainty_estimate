## SLURP on monocular depth task
It is implemented based on BTS: From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation. Links for BTS paper, Supp. and codes are as follows:

Check [arXiv](https://arxiv.org/abs/1907.10326) and [Supplementary material](https://arxiv.org/src/1907.10326v4/anc/bts_sm.pdf).

Check [here](https://github.com/cogaplex-bts/bts/tree/master/pytorch) for its original pytorch implementation.

For training SLURP, we have to add several extra python files to it, such as `train_slurp.py`, `uncertainty.py`, `bts_test_uncer.py`, `sparsification.py`. but it doesn't need to change anything with respect to the original version. Moreover, `uncertainty.py` and `uncertainty_eval.py` could be re-utilised in other tasks without making changes, such as in optical flow task.

## Requirements:
Same as the ones in the BTS original pytorch implementation.

## Steps:
To train SLURP, you can use the following command:

1/ A trained BTS checkpoint on KITTI-eigen-spilt.

SLURP sequential training requires the existing main task checkpoint (which is BTS here)

2/ Training a side learner as uncertainty estimator:

    cd pytorch
    python train_slurp.py arguments_train_slurp_eigen.py

Note: The dataset paths, main task model paths, auxiliary file paths and other relative settings have to be modified. The principle settings are the same as the original, such as batch size, input size, etc. 

You can also train SLURP in a joint-training way like the one in toy example, but in this case you might need a bigger video memory. For training on Cityscapes, you can just modify `bts_dataloader.py`, because Cityscapes provides only disparity maps, you have to transfer it to depth map using the camera parameters. I also provide a .txt file similar to the one for KITTI for reading images from Cityscapes.