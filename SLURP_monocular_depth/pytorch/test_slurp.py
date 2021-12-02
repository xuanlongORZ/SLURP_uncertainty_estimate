import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from bts_dataloader import *

import errno
from tqdm import tqdm

from bts_dataloader import *
from bts import BtsModel_SLRUP
from uncertainty import *

from uncertainty_eval import sparsification_error, AUROC

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args
parser.add_argument('--save_name', type=str, help='save name', default='')
parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--gt_path', type=str, help='path to the data')
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path_slurp', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--checkpoint_path_pred', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
# slurp 
parser.add_argument('--uncer_ch', type=int, default=1, help='Uncertainty output channel number. For optical flow, it is possible to set as 2.')
parser.add_argument('--with_en_rgb', action='store_true', help='Train a new rgb encoder?')
parser.add_argument('--with_pre', action='store_true', help='Train a pre-encoder to transfer the input channel to three?')
parser.add_argument('--encoder_U', type=str, help='Keep the same as the args.encoder. Yype of encoder, desenet121_bts, densenet161_bts, '
    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts', default='densenet161_bts')


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test():
    """Test function."""
    args.mode = 'test'
    args.encoder_U = args.encoder

    dataloader = BtsDataLoader(args, 'test')
    
    slurp = UncerModel(args)
    model_pred = BtsModel_SLRUP(params=args)
    slurp = torch.nn.DataParallel(slurp)
    model_pred = torch.nn.DataParallel(model_pred)

    checkpoint = torch.load(args.checkpoint_path_slurp)
    slurp.load_state_dict(checkpoint['model'])
    slurp.eval()
    slurp.cuda()

    checkpoint = torch.load(args.checkpoint_path_pred)
    model_pred.load_state_dict(checkpoint['model'])
    model_pred.eval()
    model_pred.cuda()

    num_params = sum([np.prod(p.size()) for p in slurp.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path_slurp))
    print('Saving result pngs..')

    save_name = 'result_' +args.save_name+ '_'+args.model_name
    if not os.path.exists(os.path.dirname(save_name)):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            os.mkdir(save_name + '/error')
            os.mkdir(save_name + '/var')
            os.mkdir(save_name + '/gt')
            os.mkdir(save_name + '/curves')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    hist_pred_rmses = 0
    hist_oracle_rmses = 0
    hist_pred_absrels = 0
    hist_oracle_absrels = 0
    valid_auroc = 0
    roc_aucs = 0
    sparsification_errors_pred_rmses = 0
    sparsification_errors_pred_absrels = 0
    start_time = time.time()
    valid_count = 0
    with torch.no_grad():
        for s, sample in enumerate(tqdm(dataloader.data)):
            if lines[s].split()[1] == 'None': continue
            else: valid_count += 1

            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())
            size_gt = image.size()[2:]

            # Predict
            skip_feat, depth_pred = model_pred(image, focal)
            depth_pred = depth_pred[-1]
            b, _, h, w = depth_pred.size()
            depth_pred_3 = depth_pred.expand(b, 3, h, w)/args.max_depth
            uncer = slurp(depth_pred_3, skip_feat, size_gt, args)
            uncer = nn.Sigmoid()(uncer)
            uncer[uncer == 1] = uncer[uncer == 1] - 1e-6
            uncer = torch.atanh(uncer)*args.max_depth

            depth_pred = depth_pred.cpu().numpy().squeeze()
            uncer = uncer.cpu().numpy().squeeze()

            gt_depth_path = os.path.join(args.gt_path, lines[s].split()[1])
            gt_depth = cv2.imread(gt_depth_path, -1)/256.0
            gt_depth = gt_depth.astype(np.float32)

            if args.do_kb_crop is True:
                height = gt_depth.shape[0]
                width = gt_depth.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                gt_depth = gt_depth[top_margin:top_margin + 352, left_margin:left_margin + 1216]

            valid_mask = np.logical_and(gt_depth > 1e-3, gt_depth <= 80)

            if args.garg_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)
                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                valid_mask = np.logical_and(valid_mask, eval_mask)

            error = abs(depth_pred - gt_depth)
            hist_pred_rmse, hist_oracle_rmse, nb_remain, _ = sparsification_error(uncer[valid_mask], error[valid_mask], 20, True)
            hist_pred_absrel, hist_oracle_absrel, nb_remain, _ = sparsification_error(uncer[valid_mask], error[valid_mask], 20, True, gt_depth[valid_mask], False)

            hist_pred_rmses += hist_pred_rmse
            hist_oracle_rmses += hist_oracle_rmse
            hist_pred_absrels += hist_pred_absrel
            hist_oracle_absrels += hist_oracle_absrel

            roc_auc, valid = AUROC(gt_depth[valid_mask], depth_pred[valid_mask], uncer[valid_mask])
            roc_aucs += roc_auc
            valid_auroc += valid

    roc_aucs /= valid_auroc
    hist_pred_rmses /= valid_count
    hist_oracle_rmses /= valid_count
    hist_pred_absrels /= valid_count
    hist_oracle_absrels /= valid_count

    sparsification_errors_pred_rmses = abs(hist_pred_rmses - hist_oracle_rmses).sum()
    sparsification_errors_pred_absrels = abs(hist_pred_absrels - hist_oracle_absrels).sum()

    print('AUSE_RMSE: {0}; AUSE_Absrel: {1}; AUROC: {2}'.format(sparsification_errors_pred_rmses, sparsification_errors_pred_absrels, roc_aucs))

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')
    
    np.save(os.path.join(save_name + '/curves/','brox_var_rmse.npy'),hist_pred_rmses)
    np.save(os.path.join(save_name + '/curves/','brox_error_rmse.npy'),hist_oracle_rmses)

    np.save(os.path.join(save_name + '/curves/','brox_var_absrel.npy'),hist_pred_rmses)
    np.save(os.path.join(save_name + '/curves/','brox_error_absrel.npy'),hist_oracle_rmses)

    return

if __name__ == '__main__':
    test()
