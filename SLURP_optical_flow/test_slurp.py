import os
import sys
import argparse
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import datasets 
import flow_transforms
from FlowNetS import flownets
from uncertainty_eval import sparsification_error
from uncertainty import *

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

# settings
parser = argparse.ArgumentParser(description='Settings', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

# Training setting
parser.add_argument('--test_batch_size', type=int, default=1, help='Testing batch size, default: 1')
# Slurp
parser.add_argument('--encoder_U', type=str, default='densenet161', help='Resnet or densenet or others, default: densenet161')
parser.add_argument('--uncer_ch', type=int, default=1, help='Uncertainty output channel number. For optical flow, it is possible to set as 2, default: 1')
parser.add_argument('--with_en_rgb', action='store_true', help='Train a new rgb encoder?')
parser.add_argument('--with_pre', action='store_true', help='Train a pre-encoder to transfer the input channel to three?')
parser.add_argument('--mix_gt_gau', action='store_true', help='Add GT and Gaussian noise to the input of SLURP?') 

# Loading dataset and models
parser.add_argument('--dataset_path', type=str, default=None, help='FlyingChairs_release/data/, default: None', required=True)
parser.add_argument('--dataset_name', choices = ['flying_chairs','KITTI_occ','KITTI_noc','mpi_sintel_clean','mpi_sintel_final','mpi_sintel_both'], default='flying_chairs', help='flying_chairs or kitti or sintel or flying_chairs_inverse, default: flying_chairs')
parser.add_argument('--split_file', default=None, type=str, help='test-val split file, there is one for FlyingChairs dataset, default: None')
parser.add_argument('--split_value', default=0.1, type=float, help='test-val split ratio, default: 0.1')
parser.add_argument('--checkpoint_path_slurp', type=str, default = None, help='Path of pretrained uncertainty model, default: None')
parser.add_argument('--checkpoint_path', type=str, default = None, help='Path of pretrained FlowNetS, default: None', required=True)

# Others
parser.add_argument('--val_freq', default=500, type=int, help='Validation frequency, default: 500')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use, default: 123')
parser.add_argument('--threads', type=int, default=4, help='Number of threads for data loader to use, default: 4')
parser.add_argument('--cuda', action='store_true', help='Use cuda?')


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
    print(args)
else:
    args = parser.parse_args()

# transforms.Normalize(mean=[0.45,0.432,0.411], std=[1,1,1])
inv_normalize = transforms.Normalize(
    mean=[-0.45, -0.432, -0.41],
    std=[1, 1, 1]
)

def normalize_result(value, vmin=None, vmax=None):
    value = value[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)

def main():
        if args.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        cudnn.benchmark = True

        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        # Data loading
        print('===> Loading datasets')
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
            transforms.Normalize(mean=[0.45,0.432,0.411], std=[1,1,1])
        ])

        target_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0,0], std=[20,20])
        ])

        print("=> fetching img pairs in '{}'".format(args.dataset_path))
        train_set, test_set = datasets.__dict__[args.dataset_name](
                args.dataset_path,
                transform=input_transform,
                target_transform=target_transform,
                co_transform=None,
                split=args.split_file if args.split_file else args.split_value)
        
        # len(train_set) = 0
        print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                                len(train_set),
                                                                                len(test_set)))
        # train_loader = torch.utils.data.DataLoader(
        #         train_set, batch_size=args.batch_size,
        #         num_workers=args.threads, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
                test_set, batch_size=args.test_batch_size,
                num_workers=args.threads, pin_memory=True, shuffle=False)
        device = torch.device("cuda:0" if args.cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        print(device)

        print('===> Building models')

        # load FlowNetS
        if not os.path.isfile(args.checkpoint_path):
            print('Please train the main task model first and then load it.')
            exit()
        else:
            flownet = flownets(torch.load(args.checkpoint_path)).cuda()
            flownet = torch.nn.DataParallel(flownet).cuda()
            flownet.eval()

        # load sider learner
        slurp = UncerModel(args).cuda()
        slurp = torch.nn.DataParallel(slurp).cuda()
        if args.checkpoint_path_slurp != None:
            slurp_checkpoint = torch.load(args.checkpoint_path_slurp)
            slurp.load_state_dict(slurp_checkpoint)
        slurp.eval()

        loss_criterion = logitbce_loss()

        # validate
        avg_error = 0
        avg_spar_error = 0

        with torch.no_grad():
            for input_imgpair, gt_flow in tqdm(val_loader):
                input_imgpair = torch.cat(input_imgpair, 1).to(device)
                gt_flow = gt_flow.to(device)

                output_flow = flownet(input_imgpair)
                output_flow = F.interpolate(output_flow, size=gt_flow.size()[-2:], mode='bilinear', align_corners=False)
                        
                output_uncertainty = slurp(output_flow, input_imgpair[:,:3,:,:], gt_flow.size()[2:], args)
                        
                output_uncertainty_final = nn.Sigmoid()(output_uncertainty)
                output_uncertainty_final[output_uncertainty_final==1] = output_uncertainty_final[output_uncertainty_final==1] - 1e-6
                output_uncertainty_final = (torch.atanh(output_uncertainty_final) * 20)  # restore the predicted uncertainty

                # calculate EPE and Oracle
                target = 400 * (gt_flow - output_flow)**2 # EPE/Oracle
                target = torch.sqrt(target[:,0,:,:] + target[:,1,:,:])
                target = torch.unsqueeze(target, 1)

                # reproduce the used loss during training
                if args.uncer_ch == 1:
                    target_for_bce = (gt_flow - output_flow)**2
                    target_for_bce = (torch.tanh((target_for_bce[:,0,:,:] + target_for_bce[:,1,:,:]).sqrt())).unsqueeze(1)
                else:
                    target_for_bce = torch.tanh(abs(gt_flow - output_flow))
                        
                bce_error = loss_criterion(output_uncertainty, target_for_bce)
                avg_error = avg_error + bce_error.cpu().numpy()

                # perparing for visualizations
                output_uncertainty_final = (output_uncertainty_final[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.float32)

                target_for_spar = abs(gt_flow - output_flow) * 20
                target_for_spar = (target_for_spar[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.float32)

                # calcualte sparsification error
                if args.uncer_ch == 1:
                    target_for_spar = target_for_spar ** 2
                    target_for_spar = np.sqrt(target_for_spar[:,:,0] + target_for_spar[:,:,1])
                    target_for_spar = np.expand_dims(target_for_spar, axis=2)
                    spar_error = sparsification_error(output_uncertainty_final[:,:,0].reshape(gt_flow.size()[-2]*gt_flow.size()[-1]), target_for_spar[:,:,0].reshape(gt_flow.size()[-2]*gt_flow.size()[-1]), is_epe=True)
                    avg_spar_error = avg_spar_error + spar_error
                else:
                    spar_error1 = sparsification_error(output_uncertainty_final[:,:,0].reshape(gt_flow.size()[-2]*gt_flow.size()[-1]), target_for_spar[:,:,0].reshape(gt_flow.size()[-2]*gt_flow.size()[-1]), is_epe=False)
                    spar_error2 = sparsification_error(output_uncertainty_final[:,:,1].reshape(gt_flow.size()[-2]*gt_flow.size()[-1]), target_for_spar[:,:,1].reshape(gt_flow.size()[-2]*gt_flow.size()[-1]), is_epe=False)
                    avg_spar_error = avg_spar_error + (spar_error1 + spar_error2)/2
                            
            avg_error = avg_error /len(val_loader)
            avg_spar_error = avg_spar_error/len(val_loader)
            print("===> Avg. error: {:.4f}; Avg. Spar: {:.4f}".format(avg_error, avg_spar_error))
                    

if __name__ == '__main__':
    main()