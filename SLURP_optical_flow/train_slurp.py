import os
import sys
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter

import datasets 
import flow_transforms
from FlowNetS import flownets
from utils import flow_to_image
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
parser.add_argument('--batch_size', type=int, default=8, help='Training batch size, default: 8')
parser.add_argument('--test_batch_size', type=int, default=1, help='Testing batch size, default: 1')
parser.add_argument('--epoch', type=int, default=20, help='Total epoch count, default: 20')
parser.add_argument('--start_lr', type=float, default=0.0001, help='Initial learning rate for adam, default: 1e-4')
parser.add_argument('--end_lr', type=float, default=0.00001, help='Ending learning rate for adam, default: 1e-5')
parser.add_argument('--lr_policy', type=str, default=None, help='Learning rate policy: lambda|step|plateau|cosine, default: None')
parser.add_argument('--lr_decay_iters', type=int, default=100, help='Multiply by a gamma every lr_decay_iters iterations, default: 100')
parser.add_argument('--milestones', nargs="+", default=[150, 200, 250], help='Milestones for lr schedule if milestone is chosen, default: [150, 200, 250]')
parser.add_argument('--weight_decay_en', type=float, default=1e-2, help='Weight decay of encoders, default: 1e-4')
parser.add_argument('--weight_decay', type=float, default=4e-4, help='Weight decay of decoder and pre-encoder, default: 4e-4')
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
parser.add_argument('--save_name', type=str, default=None, help='Name of the save path where the models and examples are saved, default: None', required=True)
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
    sparse = False

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

    if 'KITTI' in args.dataset_name:
        sparse = True
    if sparse:
        co_transform = flow_transforms.Compose([
            flow_transforms.RandomCrop((320,448)),
            flow_transforms.RandomVerticalFlip(),
            flow_transforms.RandomHorizontalFlip()
        ])
    else:
        co_transform = flow_transforms.Compose([
            flow_transforms.RandomTranslate(10),
            flow_transforms.RandomRotate(10,5),
            flow_transforms.RandomCrop((320,448)),
            flow_transforms.RandomVerticalFlip(),
            flow_transforms.RandomHorizontalFlip()
        ])

    print("=> fetching img pairs in '{}'".format(args.dataset_path))
    train_set, test_set = datasets.__dict__[args.dataset_name](
            args.dataset_path,
            transform=input_transform,
            target_transform=target_transform,
            co_transform=co_transform,
            split=args.split_file if args.split_file else args.split_value)

    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                            len(train_set),
                                                                            len(test_set)))
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size,
            num_workers=args.threads, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.test_batch_size,
            num_workers=args.threads, pin_memory=True, shuffle=False)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print(device)

    eval_summary_path = os.path.join('checkpoints', args.save_name, 'eval')
    eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)
    
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


    # Training parameters
    optimization_group = [{'params': filter(lambda p: p.requires_grad, slurp.module.encoder_pred.parameters()), 'weight_decay': args.weight_decay_en},
                        {'params': filter(lambda p: p.requires_grad, slurp.module.decoder.parameters()), 'weight_decay': args.weight_decay}]
    if args.with_pre:
        optimization_group.append({'params': filter(lambda p: p.requires_grad, slurp.module.pre_encoder.parameters()), 'weight_decay': args.weight_decay})
    if args.with_en_rgb:
        optimization_group.append({'params': filter(lambda p: p.requires_grad, slurp.module.encoder_rgb.parameters()), 'weight_decay': args.weight_decay_en})

    optimizer = torch.optim.AdamW(optimization_group, lr = args.start_lr, eps = 1e-3)
    net_uncertainty_scheduler = get_scheduler(optimizer, args)

    # create save folders and do copies
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(os.path.join('checkpoints', args.save_name)):
        os.mkdir(os.path.join('checkpoints', args.save_name))
    if not os.path.exists(os.path.join('checkpoints', args.save_name, 'examples')):
        os.mkdir(os.path.join('checkpoints', args.save_name, 'examples'))

    command = 'cp uncertainty.py ' + os.path.join('checkpoints', args.save_name, 'examples/')
    os.system(command)
    command = 'cp train_slurp.py ' + os.path.join('checkpoints', args.save_name, 'examples/')
    os.system(command)
    command = 'cp ' + sys.argv[1]  + '  ' + os.path.join('checkpoints', args.save_name, 'examples/')
    os.system(command)

    BEST_ERROR = 1e10
    BEST_ERROR_SPAR = 1e10
    TOTAL_ITER = 0
    loss_criterion = logitbce_loss()
    num_total_steps = len(train_loader) * args.epoch
    print('num_total_steps: ', num_total_steps)
    for epoch in range(args.epoch):
        # train
        epoch_loss_final = 0
        slurp.train()
        for iteration, (input_imgpair, gt_flow) in enumerate(train_loader):
            input_imgpair = torch.cat(input_imgpair,1).to(device)
            gt_flow = gt_flow.to(device)

            # make flow
            with torch.no_grad():
                output_flow = flownet(input_imgpair)
            output_flow = F.interpolate(output_flow, size=gt_flow.size()[-2:], mode='bilinear', align_corners=False)

            # pred mix GT as input
            # additional random Gaussian noise to the input
            # in original paper, we didn't consider it
            if args.mix_gt_gau:
                gau_mix_list = np.random.choice([0, 1], size=(gt_flow.size()[0],), p=[1./3, 2./3])
                mix_list = np.random.choice([0, 1], size=(args.batch_size,), p=[1./3, 2./3])
                for mix_index, (mix_item, gau_item) in enumerate(zip(mix_list, gau_mix_list)):
                    if mix_item == 0:
                        output_flow[mix_index] = gt_flow[mix_index]
                    if gau_item == 0:
                        # Gaussian noise
                        gaussian1 = 0.1 * torch.from_numpy(np.random.random((1, output_flow[mix_index].shape[1], output_flow[mix_index].shape[2])).astype(np.float32)).cuda()
                        gaussian2 = 0.1 * torch.from_numpy(np.random.random((1, output_flow[mix_index].shape[1], output_flow[mix_index].shape[2])).astype(np.float32)).cuda()
                        gaussian = torch.cat((gaussian1, gaussian2), dim = 0)
                        output_flow[mix_index] = output_flow[mix_index] + gaussian

            # make uncertainty
            output_uncertainty = slurp(output_flow, input_imgpair[:,:3,:,:], gt_flow.size()[2:], args)

            # make epe
            # no need to put lambda here because the training target is by default devided by 20
            # which indicates that lambda = 0.05 here
            if args.uncer_ch == 1:
                target = (gt_flow - output_flow)**2
                target = (torch.tanh((target[:,0,:,:] + target[:,1,:,:]).sqrt())).unsqueeze(1)
            else:
                target = torch.tanh(abs(gt_flow - output_flow))
            loss = loss_criterion(output_uncertainty, target)/args.batch_size
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_lr = update_learning_rate(scheduler=net_uncertainty_scheduler, optimizer=optimizer, global_step = TOTAL_ITER, num_total_steps=num_total_steps, arg = args)

            epoch_loss_final += loss.item()

            print("===> Epoch[{}]({}/{}/{}): lr: {:.8f} Final output loss: {:.4f}[{:.4f}]".format(
                epoch, iteration, len(train_loader), TOTAL_ITER, current_lr, loss.item(), epoch_loss_final/(iteration+1)))

            if TOTAL_ITER and (TOTAL_ITER % args.val_freq == 0):
                # validate
                avg_error = 0
                avg_spar_error = 0
                slurp.eval()

                with torch.no_grad():
                    for iteration_val, (input_imgpair, gt_flow) in tqdm(enumerate(val_loader)):
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
                        save_target = (target[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.float64)

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
                        # output_uncertainty_final = output_uncertainty_final**2 

                        target_for_spar = abs(gt_flow - output_flow)
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
                            
                            output_uncertainty_final = output_uncertainty_final ** 2
                            output_uncertainty_final = np.sqrt(output_uncertainty_final[:,:,0] + output_uncertainty_final[:,:,1])
                            output_uncertainty_final = np.expand_dims(output_uncertainty_final, axis=2)

                        # save visualizations
                        if iteration_val == (args.val_freq//200):
                            # cv2.imwrite(os.path.join('checkpoints/'+ args.save_name +'/examples/','val_uncertainty_img_' + str(TOTAL_ITER) + '_'+str(save_output_uncertainty_final_img_max)+'.png'), save_output_uncertainty_final_img)
                            # cv2.imwrite(os.path.join('checkpoints/'+ args.save_name +'/examples/','gt_uncertainty_' + str(TOTAL_ITER) + '_'+str(gt_uncertainty_final.max())+'.png'), gt_uncertainty_final/gt_uncertainty_final.max() * 255)        
                            # # print input images, groundtruth flow and predicted flow for just one time
                            # if TOTAL_ITER == args.val_freq:
                            #     save_target_img_max = save_target.max()
                            #     save_target_img = save_target/save_target_img_max * 255
                            #     output_flow = (output_flow[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.float32)
                            #     output_flow = flow_to_image(output_flow)
                            #     gt_flow = (gt_flow[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.float32)
                            #     gt_flow = flow_to_image(gt_flow)
                            #     save_input_imgpair1 = (input_imgpair[0, :3, :, :].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.float32) * 255
                            #     save_input_imgpair2 = (input_imgpair[0, 3:, :, :].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.float32) * 255
                            #     cv2.imwrite(os.path.join('checkpoints/'+ args.save_name +'/examples/','input_target_'+str(save_target_img_max)+'.png'), save_target_img)
                            #     cv2.imwrite(os.path.join('checkpoints/'+ args.save_name +'/examples/','input_imgpair1.png'), save_input_imgpair1)
                            #     cv2.imwrite(os.path.join('checkpoints/'+ args.save_name +'/examples/','input_imgpair2.png'), save_input_imgpair2)
                            #     cv2.imwrite(os.path.join('checkpoints/'+ args.save_name +'/examples/','predicted_flow.png'), output_flow)
                            #     cv2.imwrite(os.path.join('checkpoints/'+ args.save_name +'/examples/','groundtruth_flow.png'), gt_flow)
                            gt_flow = (gt_flow[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.float32)
                            gt_flow = flow_to_image(gt_flow)
                            gt_flow = gt_flow.transpose(2, 0, 1)
                            eval_summary_writer.add_image('gt_flow/image/{}'.format(0), gt_flow, TOTAL_ITER)

                            output_flow = (output_flow[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.float32)
                            output_flow = flow_to_image(output_flow)
                            output_flow = output_flow.transpose(2, 0, 1)
                            eval_summary_writer.add_image('output_flow/image/{}'.format(0), output_flow, TOTAL_ITER)

                            output_uncertainty_final = output_uncertainty_final.transpose(2, 0, 1)
                            eval_summary_writer.add_image('depth_uncer/image/{}'.format(0), normalize_result(output_uncertainty_final), TOTAL_ITER)
                            save_target = save_target.transpose(2, 0, 1)
                            eval_summary_writer.add_image('gt_uncer/image/{}'.format(0), normalize_result(save_target), TOTAL_ITER)
                            
                            eval_summary_writer.add_image('image1/image/{}'.format(0), inv_normalize(input_imgpair[0, :3, :, :]).data, TOTAL_ITER)
                            eval_summary_writer.add_image('image2/image/{}'.format(0), inv_normalize(input_imgpair[0, 3:, :, :]).data, TOTAL_ITER)

                    avg_error = avg_error /len(val_loader)
                    avg_spar_error = avg_spar_error/len(val_loader)
                    print("===> Avg. error: {:.4f}; Avg. Spar: {:.4f}".format(avg_error, avg_spar_error))
                    
                    eval_summary_writer.add_scalar('total loss', avg_error, TOTAL_ITER)
                    eval_summary_writer.add_scalar('sparsification error', avg_spar_error, TOTAL_ITER)
                    eval_summary_writer.add_scalar('learning_rate', current_lr, TOTAL_ITER)

                    eval_summary_writer.flush()

                    # save models
                    content = 'Epoch: {0}; Batch: {1}; Loss: {2}; Spar_error: {3} Status: '.format(epoch, TOTAL_ITER, avg_error, avg_spar_error)
                    if BEST_ERROR > avg_error or BEST_ERROR_SPAR > avg_spar_error:
                        if BEST_ERROR > avg_error:
                            content = content + 'Best Loss. '
                            BEST_ERROR = avg_error
                            net_uncertainty_best_model_out_path = os.path.join('checkpoints', args.save_name, "model_best_loss.pth")
                            torch.save(slurp.state_dict(), net_uncertainty_best_model_out_path)
                        if BEST_ERROR_SPAR > avg_spar_error:
                            content = content + 'Best Spar. '
                            BEST_ERROR_SPAR = avg_spar_error
                            # in case of early-stop, save all best checkpoints
                            net_uncertainty_best_model_out_path = os.path.join('checkpoints', args.save_name, "model_best_spar_"+ str(TOTAL_ITER) +".pth")
                            torch.save(slurp.state_dict(), net_uncertainty_best_model_out_path)
                        content = content + '\n'
                        filename  = open(os.path.join('checkpoints', args.save_name, 'training_record.txt'), "a")
                        filename.write(content)
                        filename.close()
                    else:
                        filename  = open(os.path.join('checkpoints', args.save_name, 'training_record.txt'), "a")
                        filename.write('Epoch: {0}; Batch: {1}; Loss: {2}; Spar_error: {3}\n'.format(epoch, TOTAL_ITER, avg_error, avg_spar_error))
                        filename.close()

                    # latest checkpoint saved in case of some sudden issues
                    torch.save(slurp.state_dict(), os.path.join('checkpoints', args.save_name) + "/model_latest.pth")
                    slurp.train()

            TOTAL_ITER += 1

        print("===> Epoch[{}]: Final output loss for this epoch: {:.4f}".format(epoch, epoch_loss_final/iteration))


if __name__ == '__main__':
    main()