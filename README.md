## SLURP_uncertainty_estimation

The implementations for SLURP: Side Learning Uncertainty for Regression Problems (BMVC 2021)

Check [arXiv](https://arxiv.org/abs/2110.11182).

The three folders correspond to three different tasks. See the `README.md` in the folders for more details.

Here are some visualizations for the results of our side learner.

On FlyingChairs dataset:
![plot](./others/slurp_chairs_vis.gif)

On Sintel dataset (without fine-tuning):
![plot](./others/slurp_sintel_clean_vis.gif)

On KITTI dataset:
![plot](./others/slurp_kitti_vis.gif)

## Citation
If you find this work useful for your research, please consider citing our paper:

    @inproceedings{yu21bmvc,
    author    = {Xuanlong Yu and
                Gianni Franchi and
                Emanuel Aldea},
    title     = {SLURP: Side Learning Uncertainty for Regression Problems},
    booktitle = {32nd British Machine Vision Conference, {BMVC} 2021,
                Virtual Event /  November 22-25, 2021},
    year      = {2021}
    }