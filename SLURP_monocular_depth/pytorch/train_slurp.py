# Moidified according to https://github.com/cleinc/bts

import time
import argparse
import sys
import os

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

from tqdm import tqdm

from bts import BtsModel_SLRUP
from bts_dataloader import *
import torch.nn as nn
from uncertainty import *
from uncertainty_eval import sparsification_error

import numpy as np
import cv2

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

# Some of them are useless for side learning training 
parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen_v2')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                               default='densenet161_bts')
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--checkpoint_path_slurp',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=5000)

# Training
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
parser.add_argument('--weight_decay_en',              type=float, help='weight decay factor for optimization encoder', default=1e-2)
parser.add_argument('--weight_decay_de',              type=float, help='weight decay factor for optimization decoder', default=1e-2)
parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Slurp 
parser.add_argument('--uncer_ch',                   type=int, default=1,    help='Uncertainty output channel number. For optical flow, it is possible to set as 2.')
parser.add_argument('--with_en_rgb',                  action='store_true',    help='Train a new rgb encoder?')
parser.add_argument('--with_pre',                     action='store_true',    help='Train a pre-encoder to transfer the input channel to three?')
parser.add_argument('--encoder_U',                  type=str,   help='Keep the same as the args.encoder. Yype of encoder, desenet121_bts, densenet161_bts, '
                                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts', default='densenet161_bts')
parser.add_argument('--lambda_loss',                   type=float, default=0.0125,    help='lambda value in BCE loss to rescale the error. In monocular depth, it is 1/max_depth')
parser.add_argument('--mix_gt_gau',                 action='store_true', help='Add GT and Gaussian noise to the input of SLURP?') 

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=1000)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')
if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

best_eval_bce = 1e10
best_eval_brox = 1e10

def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)

def main_worker(gpu, ngpus_per_node, args):
    global best_eval_bce
    global best_eval_brox
    args.gpu = gpu
    args.encoder_U = args.encoder
    args.lambda_loss = 1/args.max_depth

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # Create model
    slurp = UncerModel(args)
    model_pred = BtsModel_SLRUP(args)

    slurp.train()
    model_pred.eval()

    num_params = sum([np.prod(p.size()) for p in slurp.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in slurp.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            slurp.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            slurp = torch.nn.parallel.DistributedDataParallel(slurp, device_ids=[args.gpu], find_unused_parameters=True)
            
            model_pred = torch.nn.DataParallel(model_pred, device_ids=[args.gpu], find_unused_parameters=True)
            model_pred.cuda(args.gpu)
        else:
            slurp.cuda()
            slurp = torch.nn.parallel.DistributedDataParallel(slurp, find_unused_parameters=True)
            model_pred = torch.nn.DataParallel(model_pred,  find_unused_parameters=True)
            model_pred.cuda()
    else:
        slurp = torch.nn.DataParallel(slurp)
        slurp.cuda()
        model_pred = torch.nn.DataParallel(model_pred)
        model_pred.cuda()

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")

    global_step = 0

    # Training parameters
    optimization_group = [
        {'params': filter(lambda p: p.requires_grad, slurp.module.encoder_pred.parameters()), 'weight_decay': args.weight_decay_en},
        {'params': filter(lambda p: p.requires_grad, slurp.module.decoder.parameters()), 'weight_decay': args.weight_decay_de}]
    if args.with_pre:
        optimization_group.append({'params': filter(lambda p: p.requires_grad, slurp.module.pre_encoder.parameters()), 'weight_decay': 4e-4})
    if args.with_en_rgb:
        optimization_group.append({'params': filter(lambda p: p.requires_grad, slurp.module.encoder_rgb.parameters()), 'weight_decay': 1e-4})

    optimizer = torch.optim.AdamW(optimization_group, lr=args.learning_rate, eps=args.adam_eps)

    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            model_pred.load_state_dict(checkpoint['model'])
            print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True
    if args.checkpoint_path_slurp != '':
        if os.path.isfile(args.checkpoint_path_slurp):
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path_slurp, map_location=loc)
                slurp.load_state_dict(checkpoint['model'])
                print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path_slurp, checkpoint['global_step']))


    if args.retrain:
        global_step = 0

    cudnn.benchmark = True

    dataloader = BtsDataLoader(args, 'train')
    dataloader_eval = BtsDataLoader(args, 'online_eval')

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    loss_criterion = logitbce_loss()

    start_time = time.time()
    duration = 0
    print_img = 0

    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    var_sum = [var.detach().cpu().numpy().sum() for var in slurp.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch
    start_lr = args.learning_rate 

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            focal = torch.autograd.Variable(sample_batched['focal'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))

            size_gt = depth_gt.size()[2:]
            with torch.no_grad():
                skip_feat_rgb, depth_est_pred = model_pred(image, focal)
                depth_est_pred = depth_est_pred[-1]
                b, _, h, w = depth_est_pred.size()

            # pred mix GT as input
            # additional random Gaussian noise to the input
            # in original paper, we didn't consider it
            if args.mix_gt_gau:
                gau_mix_list = np.random.choice([0, 1], size=(depth_est_pred.size()[0],), p=[1./3, 2./3])
                mix_list = np.random.choice([0, 1], size=(args.batch_size,), p=[1./3, 2./3])
                for mix_index, (mix_item, gau_item) in enumerate(zip(mix_list, gau_mix_list)):
                    if mix_item == 0:
                        depth_est_pred[mix_index] = depth_gt[mix_index]
                    if gau_item == 0:
                        # Gaussian noise
                        gaussian = 0.1 * torch.from_numpy(np.random.random((1, depth_est_pred[mix_index].shape[1], depth_est_pred[mix_index].shape[2])).astype(np.float32)).cuda()
                        depth_est_pred[mix_index] = depth_est_pred[mix_index] + gaussian
            
            depth_est_pred_3 = depth_est_pred.expand(b, 3, h, w)/args.max_depth
            uncer = slurp(depth_est_pred_3, skip_feat_rgb, size_gt, args)

            if args.dataset == 'nyu':
                mask = depth_gt > 0.1
            else:
                mask = depth_gt > 1.0
            mask = mask.to(torch.bool)

            target = (abs(depth_est_pred - depth_gt))[mask]
            target = torch.tanh(args.lambda_loss * target)
            loss = loss_criterion.forward(uncer[mask], target)
            loss = loss/args.batch_size
            loss.backward()

            for param_group in optimizer.param_groups:
                current_lr = (start_lr - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.8 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.8}, loss: {:.5f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.detach().cpu().numpy().sum() for var in slurp.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} var cnt: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_cnt, var_sum.item()/var_cnt, time_sofar, training_time_left))
                
                uncer = nn.Sigmoid()(uncer) * (1/args.lambda_loss)
                depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                
                writer.add_scalar('total loss', loss, global_step)
                writer.add_scalar('learning_rate', current_lr, global_step)
                writer.add_scalar('var average', var_sum.item()/var_cnt, global_step)

                writer.add_image('depth_gt/image/{}'.format(0), normalize_result(1/depth_gt[0, :, :, :].data), global_step)
                writer.add_image('depth_est/image/{}'.format(0), normalize_result(1/depth_est_pred[0, :, :, :].data), global_step)
                writer.add_image('depth_uncer/image/{}'.format(0), normalize_result(uncer[0, :, :, :].data), global_step)
                writer.add_image('image/image/{}'.format(0), inv_normalize(image[0, :, :, :]).data, global_step)

                writer.flush()

            if not args.do_online_eval and global_step and global_step % args.save_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    checkpoint = {'global_step': global_step,
                                  'model': slurp.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    torch.save(checkpoint, args.log_directory + '/' + args.model_name + '/model-{}'.format(global_step))

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                slurp.eval()
                bce_loss = 0
                spar_losses = 0
                count_eval = 0
                for eval_sample_batched in tqdm(dataloader_eval.data):
                    with torch.no_grad():
                        image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
                        focal = torch.autograd.Variable(eval_sample_batched['focal'].cuda(gpu, non_blocking=True))
                        gt_depth = eval_sample_batched['depth']
                        has_valid_depth = eval_sample_batched['has_valid_depth']
                        if not has_valid_depth: continue
                        else: count_eval += 1
                        size_gt = image.size()[2:]
                        skip_feat_rgb, depth_est_pred = model_pred(image, focal)
                        depth_est_pred = depth_est_pred[-1]
                        b, _, h, w = depth_est_pred.size()
                        depth_est_pred_3 = depth_est_pred.expand(b, 3, h, w)/args.max_depth
                        uncer_org = slurp(depth_est_pred_3, skip_feat_rgb, size_gt, args)
                        
                        gt_depth_cuda = gt_depth.view(gt_depth.size()[0],gt_depth.size()[3],gt_depth.size()[1],gt_depth.size()[2])
                        depth_est_pred_cuda = depth_est_pred.clone()
                        
                        uncer = nn.Sigmoid()(uncer_org)

                        uncer = uncer.cpu().numpy().squeeze()
                        depth_est_pred = depth_est_pred.cpu().numpy().squeeze()
                        gt_depth = gt_depth.cpu().numpy().squeeze()

                        # uncomment it to save the results during validation
                        # if epoch == 0 and count_eval == 10 and print_img == 0:
                        #     if not os.path.isdir(args.log_directory + '/' + args.model_name + '/output_imgs/'):
                        #         os.mkdir(args.log_directory + '/' + args.model_name + '/output_imgs/')
                        #     pred_save_name = '/pred_uncrop.png'
                        #     pred_save_name = args.log_directory + args.model_name +  pred_save_name
                        #     image_save_name = '/img_uncrop.png'
                        #     image_save_name = args.log_directory + '/' + args.model_name + image_save_name
                        #     cv2.imwrite(pred_save_name, (depth_est_pred * 256.0).astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        #     cv2.imwrite(image_save_name, (image.cpu().numpy().squeeze()).transpose(1, 2, 0).astype(np.uint8))

                        # if count_eval == 10:
                        #     var_save_name = '/uncer-{}_uncrop.png'.format(global_step)
                        #     var_save_name = args.log_directory + '/' + args.model_name + '/output_imgs/' + var_save_name
                        #     cv2.imwrite(var_save_name, ((np.arctanh(uncer) * (1/args.lambda_loss))**2 * 256.0).astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    
                    if args.do_kb_crop:
                        height, width = gt_depth.shape
                        top_margin = int(height - 352)
                        left_margin = int((width - 1216) / 2)
                        uncer_uncropped = np.zeros((height, width), dtype=np.float32)
                        uncer_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = uncer
                        uncer = uncer_uncropped
                        pred_est_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                        pred_est_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = depth_est_pred
                        depth_est_pred = pred_est_depth_uncropped
                    
                    valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

                    if args.garg_crop or args.eigen_crop:
                        gt_height, gt_width = gt_depth.shape
                        eval_mask = np.zeros(valid_mask.shape)
                        if args.garg_crop:
                            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                        elif args.eigen_crop:
                            if args.dataset == 'kitti':
                                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                            else:
                                eval_mask[45:471, 41:601] = 1
                        valid_mask = np.logical_and(valid_mask, eval_mask)

                    # if epoch == 0 and count_eval == 10 and print_img == 0:
                    #     pred_save_name = '/pred_crop.png'
                    #     pred_save_name = args.log_directory + '/' + args.model_name + pred_save_name
                    #     image_save_name = '/img_crop.png'
                    #     image_save_name = args.log_directory + '/' + args.model_name + image_save_name
                    #     cv2.imwrite(pred_save_name, (depth_est_pred * 256.0 * valid_mask).astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    #     cv2.imwrite(image_save_name, (image.cpu().numpy().squeeze() * valid_mask).transpose(1, 2, 0).astype(np.uint8))
                    #     print_img = 1
                 
                    # if count_eval == 10:
                    #         var_save_name = '/uncer-{}_crop.png'.format(global_step)
                    #         var_save_name = args.log_directory + '/' + args.model_name + '/output_imgs/' + var_save_name
                    #         cv2.imwrite(var_save_name, ((uncer*args.max_depth)**2 * 256.0 * valid_mask).astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])

                    mask = torch.Tensor(valid_mask).unsqueeze(0).unsqueeze(0).cuda().to(torch.bool)
                    target = (abs(depth_est_pred_cuda.cuda() - gt_depth_cuda.cuda()))[mask]
                    target = torch.tanh(args.lambda_loss * target)
                    bce_loss = bce_loss + loss_criterion.forward(uncer_org[mask], target)

                    uncer[uncer == 1] = uncer[uncer == 1] - 1e-6
                    uncer = np.arctanh(uncer) * (1/args.lambda_loss)
                    square_diff = abs(depth_est_pred - gt_depth)
                    
                    spar_losses = spar_losses + sparsification_error(uncer[valid_mask], square_diff[valid_mask])

                spar_losses = spar_losses/count_eval
                bce_loss = bce_loss/count_eval

                print('sparsification_losses: {0:.4f}; bce_losses: {1:.4f}'.format(spar_losses, bce_loss))
                eval_summary_writer.add_scalar('sparsification_losses', spar_losses, int(global_step))
                eval_summary_writer.add_scalar('bce_losses', bce_loss, int(global_step))
                if best_eval_bce > bce_loss or best_eval_brox > spar_losses:
                    checkpoint = {'global_step': global_step,
                        'model': slurp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_eval_steps': global_step
                    }
                    content = 'Epoch: {0}; Iter: {1}; Loss: {2}; Spar_error: {3} Status: '.format(epoch, global_step, bce_loss, spar_losses)
                    if best_eval_bce > bce_loss:
                        content = content + 'Best Loss. '
                        best_eval_bce = bce_loss
                        model_save_name = '/model-best_loss'
                        torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    if best_eval_brox > spar_losses:
                        content = content + 'Best Spar. '
                        best_eval_brox = spar_losses
                        model_save_name = '/model-best_spar'
                        torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    content = content + '\n'
                    filename = open(os.path.join(args.log_directory + '/' + args.model_name, 'training_record.txt'), "a")
                    filename.write(content)
                    filename.close()
                else:
                    filename = open(os.path.join(args.log_directory + '/' + args.model_name, 'training_record.txt'), "a")
                    filename.write('Epoch: {0}; Iter: {1}; Loss: {2}; Spar_error: {3}\n'.format(epoch, global_step, bce_loss, spar_losses))
                    filename.close()
        
                model_save_name = '/model-current'
                torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                slurp.train()
                
            eval_summary_writer.flush()
            model_just_loaded = False
            global_step += 1

        epoch += 1

def main():
    if args.mode != 'train':
        print('bts_main.py is only for training. Use bts_test.py instead.')
        return -1

    model_filename = args.model_name + '.py'

    if not os.path.isfile(args.checkpoint_path):
        print('Please train the main task model first and then load it.')
        exit()

    if not os.path.isdir(args.log_directory):
        command = 'mkdir ' + args.log_directory
        os.system(command)

    command = 'mkdir ' + args.log_directory + '/' + args.model_name
    os.system(command)

    args_out_path = args.log_directory + '/' + args.model_name + '/' + sys.argv[1]
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    command =  'cp uncertainty.py ' + args.log_directory + '/' + args.model_name + '/'
    os.system(command)

    command =  'cp train_slurp.py ' + args.log_directory + '/' + args.model_name + '/'
    os.system(command)
    
    if args.checkpoint_path == '':
        aux_out_path = args.log_directory + '/' + args.model_name + '/.'
        command = 'cp bts_main.py ' + aux_out_path
        os.system(command)
        command = 'cp bts_dataloader.py ' + aux_out_path
        os.system(command)
        command = 'cp bts.py ' + aux_out_path
        os.system(command)
    else:
        loaded_model_dir = os.path.dirname(args.checkpoint_path)
        loaded_model_name = os.path.basename(loaded_model_dir)
        loaded_model_filename = loaded_model_name + '.py'

        command = 'cp ' + loaded_model_dir + '/' + loaded_model_filename
        os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

if __name__ == '__main__':
    main()
