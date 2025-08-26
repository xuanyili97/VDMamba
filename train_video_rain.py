#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, copy, pickle
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = "1"

from datetime import datetime
import numpy as np
import multiprocessing as mp

### torch lib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

### custom lib
import networks
import datasets_multiple
import utils
from utils import *
import torch.nn.init as init
from torch.nn.init import *
import torchvision
from loss import *

import torch.nn.functional as F

from networks.SoftMedian import softMedian, softMin, softMax
from networks.vdmamba import Model

import time
import os
os.environ['MASTER_PORT'] = '23456'
os.environ['MASTER_ADDR'] = 'localhost'

import faulthandler
def EPE_loss(inp, gt):
    tmp = torch.norm(gt - inp, 2, 1)
    return tmp.mean()
def rgb_2_Y(im):
    b, c, h, w = im.shape
    weight = torch.Tensor([0.299, 0.587, 0.114]).view(1, -1, 1, 1).to(im)
    return (weight * im).mean(1, keepdim=True).repeat(1, c, 1, 1)

def seed_torch(seed=3407):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def check_nan_inf(x):
    ifnan = torch.isnan(x).sum() == 0
    ifinf = torch.isinf(x).sum() == 0
    return ifnan* ifinf

faulthandler.enable()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fast Blind Video Temporal Consistency")
    parser.add_argument('-Net',   type=str,     default="mymodel",     help='Multi-frame models for hanlde videos')
    parser.add_argument('-model_name',      type=str,     default='none',           help='path to save model')

    parser.add_argument('-dataset_task',  type=str,     default='nturain', choices=['nturain', 'rainvidss', 'rainsynlight25', 'rainsyncomplex25', 'rainsynall100'],    help='dataset-task pairs list')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to lists folder')
    parser.add_argument('-crop_size',       type=int,     default=32,               help='patch size')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',    help='path to checkpoint folder')
    parser.add_argument('-geometry_aug',    type=int,     default=1,                help='geometry augmentation (rotation, scaling, flipping)')
    parser.add_argument('-order_aug',       type=int,     default=1,                help='temporal ordering augmentation')
    parser.add_argument('-scale_min',       type=float,   default=0.4,              help='min scaling factor')
    parser.add_argument('-scale_max',       type=float,   default=2.0,              help='max scaling factor')
    parser.add_argument('-sample_frames',   type=int,     default=9,                help='#sampled frames')
    parser.add_argument('-train_frames',   type=int,     default=7,                help='#frames for training')    
    parser.add_argument('-alpha',           type=float,   default=50.0,             help='alpha for computing visibility mask')
    parser.add_argument('-loss',            type=str,     default="L2",             help="optimizer [Options: SGD, ADAM]")
    parser.add_argument('-solver',          type=str,     default="ADAM",           choices=["SGD", "ADAIM"],   help="optimizer")
    parser.add_argument('-momentum',        type=float,   default=0.9,              help='momentum for SGD')
    parser.add_argument('-beta1',           type=float,   default=0.9,              help='beta1 for ADAM')
    parser.add_argument('-beta2',           type=float,   default=0.999,            help='beta2 for ADAM')
    parser.add_argument('-weight_decay',    type=float,   default=1e-2,                help='weight decay')

    parser.add_argument('-lr_init',         type=float,   default=1e-4,             help='initial learning Rate')
    parser.add_argument('-lr_offset',       type=int,     default=20,               help='epoch to start learning rate drop [-1 = no drop]')
    parser.add_argument('-lr_step',         type=int,     default=25,               help='step size (epoch) to drop learning rate')
    parser.add_argument('-lr_drop',         type=float,   default=0.5,              help='learning rate drop ratio')
    parser.add_argument('-lr_min_m',        type=float,   default=0.01,             help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')
    
    parser.add_argument('-seed',            type=int,     default=9487,             help='random seed to use')
    parser.add_argument('-threads',         type=int,     default=16,               help='number of threads for data loader to use')
    parser.add_argument('-batch_size',      type=int,     default=16,               help='size of batch for data loader to use')
    parser.add_argument('-suffix',          type=str,     default='.jpg',               help='name suffix')
    parser.add_argument('-gpu',             type=int,     default=0,                help='gpu device id')
    parser.add_argument('-cpu',             action='store_true',                    help='use cpu?')
    #parser.add_argument('--local_rank',     type=int,     default=os.getenv('LOCAL_RANK',-1),                    help='use cpu?')

    parser.add_argument('-list_filename',   type=str,      help='use cpu?')
    parser.add_argument('-test_list_filename',   type=str,      help='use cpu?')

    parser.add_argument('-ifAlign',             type=int,     default=1,                help='if align or not')
    parser.add_argument('-ifAggregate',             type=int,     default=1,                help='if aggregate or not')

    opts = parser.parse_args()
    if opts.dataset_task == 'nturain':
#        from options.option_nturain_train import *
        from options.option_nturain import *
    elif opts.dataset_task == 'rainvidss':
        from options.option_rainvidss import * 
    elif opts.dataset_task == 'rainsynlight25':
        from options.option_rainsynlight25 import *  
    elif opts.dataset_task == 'rainsyncomplex25':
        from options.option_rainsyncomplex25 import *
    elif opts.dataset_task == 'rainsynall100':
        from options.option_rainsynall100 import *

    opts.checkpoint_dir = opt.checkpoint_dir
    opts.data_dir = opt.data_dir
    opts.datatest_dir = opt.datatest_dir

    opts.list_filename = opt.list_filename
    opts.test_list_filename = opt.test_list_filename
    opts.suffix = opt.suffix

    opts.train_epoch_size = opt.train_epoch_size
    opts.valid_epoch_size = opt.valid_epoch_size
    opts.epoch_max = opt.epoch_max
    opts.threads = opt.threads
    opts.cuda = (opts.cpu != True)
    opts.lr_min = opts.lr_init * opts.lr_min_m   

    opts.size_multiplier = 2 ** 4
    print(opts)

    seed_torch(opts.seed)

    opts.model_dir = os.path.join(opts.checkpoint_dir, opts.model_name)
    print("========================================================")
    print("===> Save model to %s" %opts.model_dir)
    print("========================================================")
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)

    print('===> Initializing model from %s...' %opts.Net)
    Net = Model(opts)

    Net.ifAggregate = torch.nn.DataParallel(Net.ifAggregate)
    Net.ifAlign = torch.nn.DataParallel(Net.ifAlign)
    Net.flow_estimator = torch.nn.DataParallel(Net.flow_estimator)
    Net.frame_restorer = torch.nn.DataParallel(Net.frame_restorer)
    Net.FlowNet = torch.nn.DataParallel(Net.FlowNet)
    Net.epoch = 0 
    
    opts.rgb_max = 1.0
    opts.fp16 = False
    
    model_filename = os.path.join("./pretrained_models", "network-sintel.pytorch")
    print("===> Load %s" %model_filename)
    checkpoint = torch.load(model_filename)
    state_dict_w_module = {}
    for key in checkpoint:
        key_w_module = key.split('.')
        state_dict_w_module['.'.join(['module', key_w_module[0]] + key_w_module[1:])] = checkpoint[key]
    Net.FlowNet.load_state_dict(state_dict_w_module)
    assert opts.lr_init == 5e-5
    optimizer = optim.AdamW(Net.frame_restorer.parameters(), lr=opts.lr_init, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2), eps=1e-8)

    optimizer_flow = optim.AdamW(Net.flow_estimator.parameters(), lr=opts.lr_init, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2), eps=1e-8)

    lr_scheduler_flow = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_flow, mode="min", factor=0.5, patience=6, threshold=1e-6)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6, threshold=1e-6)

    name_list = glob.glob(os.path.join(opts.model_dir, "model_epoch_*.pth"))
    epoch_st = -1

    if len(name_list) > 0:
        epoch_list = []
        for name in name_list:
            s = re.findall(r'\d+', os.path.basename(name))[0]
            epoch_list.append(int(s))

        epoch_list.sort()
        epoch_st = epoch_list[-1]

    if epoch_st >= 0:
        print('=====================================================================')
        print('===> Resuming model from epoch %d' %epoch_st)
        print('=====================================================================')
        Net, optimizer, optimizer_flow, lr_scheduler, lr_scheduler_flow = utils.load_model(Net, optimizer, optimizer_flow, lr_scheduler, lr_scheduler_flow, opts, epoch_st)

    print(Net)

    num_params = utils.count_network_parameters(Net)
    print('\n=====================================================================')
    print("===> Model has %d parameters" %num_params)
    print('=====================================================================')

    num_params = utils.count_network_parameters(Net.FlowNet)
    print('\n=====================================================================')
    print("===> Model FlowNet has %d parameters" %num_params)
    print('=====================================================================')

    num_params = utils.count_network_parameters(Net.frame_restorer)
    print('\n=====================================================================')
    print("===> Model frame_restorer has %d parameters" %num_params)
    print('=====================================================================')

    num_params = utils.count_network_parameters(Net.frame_restorer.module.frame_restorer_spa)
    print('\n=====================================================================')
    print("===> Model frame_restorer_spa has %d parameters" %num_params)
    print('=====================================================================')

    num_params = utils.count_network_parameters(Net.flow_estimator)
    print('\n=====================================================================')
    print("===> Model flow_estimator has %d parameters" %num_params)
    print('=====================================================================')

    loss_dir = os.path.join(opts.model_dir, 'loss')
    loss_writer = SummaryWriter(loss_dir)

    device = torch.device(0 if opts.cuda else "cpu")

    Net.frame_restorer = Net.frame_restorer.cuda()
    Net.flow_estimator = Net.flow_estimator.cuda()

    Net.FlowNet = Net.FlowNet.cuda()

    Net.frame_restorer.train()
    Net.flow_estimator.train()
    Net.FlowNet.eval()

    train_dataset = datasets_multiple.MultiFramesDataset(opts, "paired_train")

    if opts.test_list_filename:
        val_dataset = datasets_multiple.MultiFramesDataset(opts, "paired_test")
        val_loader = utils.create_data_loader(val_dataset, opts, "paired_test")
        valid_dir = os.path.join(opts.model_dir, 'validation')
        os.makedirs(os.path.join(valid_dir), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, 'gt'), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, 'pred'), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, 'pred_spa'), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, 'input'), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, 'diff'), exist_ok=True)


    loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
    loss_fn2 = torch.nn.MSELoss(reduce=True, size_average=True)

    while Net.epoch < opts.epoch_max:
        torch.cuda.empty_cache()
        Net.epoch += 1

        data_loader = utils.create_data_loader(train_dataset, opts, "paired_train")
        
        prev_epoch_distill_loss = 1e10

        distill_loss_list = []
        overall_loss_list = []

        ts = datetime.now()
        
        for iteration, data_batch in enumerate(data_loader, 1):
#            torch.cuda.empty_cache()
            batch_batch, gt_batch, paired_index_batch = data_batch
            Net.set_new_video()
            for index_inner in range(opts.sample_frames - opts.train_frames + 1):
                i1, i2 = index_inner, index_inner + opts.train_frames
                batch, gt, paired_index = batch_batch[i1:i2], gt_batch[i1:i2], paired_index_batch[i1:i2]
                print("paired_index", paired_index)
                unpaired_index = np.where(np.array(paired_index[0])==0)[0]
                paired_index = np.where(np.array(paired_index[0])==1)[0]

                total_iter = (Net.epoch - 1) * opts.train_epoch_size + iteration
                cross_num = 1

                frame_i = []

                for t in range(opts.train_frames):
                    frame_i.append(batch[t * cross_num].cuda())

                data_time = datetime.now() - ts
                ts = datetime.now()

                optimizer.zero_grad()
                optimizer_flow.zero_grad()


                [b, c, h, w] = frame_i[0].shape
                sq = len(frame_i)
                frame_i3 = frame_i[sq//2]
            
                flow_warping = networks.LiteFlowNet.backwarp
                seq_around = [frame_i[i] for i in range(sq) if i != sq//2]
                num_around = len(seq_around)           
            
                ###### Train Derain ######

                seq_input = torch.cat(frame_i, 1).view(b, sq, c, h, w)

                gt_tensor = torch.cat(gt, 1).view(b, sq, c, h, w).cuda()

                frame_target = gt_tensor[:,sq//2]

                distill_loss = torch.Tensor([0.0]).float().cuda()
                flow_inward = [Net.FlowNet(frame_i3, frame) for i, frame in enumerate(seq_around)] # teacher on lq
                flow_inward_tensor = torch.cat(flow_inward, 1).detach()
                
                if opts.ifAlign and len(paired_index) != 0:
                    flow_list = Net.predict_flow(seq_input[paired_index], Net.scales) # student on lq
                    flow_inward_gt = [Net.FlowNet(gt_tensor[paired_index,sq//2], gt_tensor[paired_index,i]) for i in range(sq) if i != sq//2] # teacher on gt
                    flow_inward_gt_tensor = torch.cat(flow_inward_gt,1).detach()

                    for j in range(len(flow_list)):
                        flow_tea_gt = F.interpolate(flow_inward_gt_tensor, scale_factor=1./Net.scales[j], mode="bilinear", align_corners=False)
                        flow = flow_list[j]

                        loss0 = EPE_loss(flow, flow_tea_gt)
                        distill_loss = distill_loss + loss0 #if check_nan_inf(loss0) else distill_loss

                    distill_loss = distill_loss / len(flow_list) #* 3# * num_around * 1000

                    distill_loss.backward()            
                    torch.nn.utils.clip_grad_norm_(Net.flow_estimator.module.parameters(), max_norm=20, norm_type=2)
                    optimizer_flow.step()

                ################ Reconstruction Loss ################
                frames_warp = [flow_warping(frame, flow_inward_tensor[:,2*i:2*i+2]).unsqueeze(0) for i, frame in enumerate(seq_around)] + [frame_i3.unsqueeze(0), ]
                center_median = torch.cat((frames_warp), 0).median(0)[0].detach()
                overall_loss = torch.Tensor([0.0]).float().cuda()
                if True: # Net.epoch > 1 and prev_epoch_distill_loss <= 20:
                    frame_pred, frame_pred_spa, _, seq_input_warp = Net(seq_input, ifInferece=False, )

                    print(paired_index, unpaired_index, center_median.shape, frame_pred.shape, seq_input_warp.shape)
                    loss1, loss2, loss3 = torch.Tensor([0.0]).float().cuda(), torch.Tensor([0.0]).float().cuda(), torch.Tensor([0.0]).float().cuda()
                    if len(paired_index) != 0:
                        loss1 += loss_fn(frame_pred[paired_index], frame_target[paired_index])
                    if len(unpaired_index) != 0:
                        loss2 += loss_center_median(frame_pred[unpaired_index], center_median[unpaired_index], frame_i3[unpaired_index])
                    
                    flow_outward = [Net.FlowNet(frame, frame_pred) for i, frame in enumerate(seq_around)] # teacher on lq
                    frames_out_warp = [flow_warping(frame_pred, flow) for i, flow in enumerate(flow_outward)]

                    for frame1, frame2 in zip(seq_around, frames_out_warp):
                        loss3 += loss_fn(frame2, frame1)

                    loss4 = loss_fn(frame_pred_spa[paired_index], frame_target[paired_index]) 
#                    loss4 = loss_fn(frame_pred_spa[paired_index], frame_pred[paired_index].detach())

                    print(loss1, loss2, loss3)

                    loss_writer.add_scalar('Loss1', loss1.item(), total_iter)
                    loss_writer.add_scalar('Loss2', loss2.item(), total_iter)
                    loss_writer.add_scalar('Loss3', loss3.item(), total_iter)
                    loss_writer.add_scalar('Loss4', loss4.item(), total_iter)
                    overall_loss = overall_loss + loss1 #if check_nan_inf(loss1) else overall_loss
                    overall_loss = overall_loss + 0.01 * loss2 #if check_nan_inf(loss2) else overall_loss
                    overall_loss = overall_loss + 0.001 * loss3 / 6 #if check_nan_inf(loss3) else overall_loss
                    overall_loss = overall_loss + 0.1 * loss4


                    total_loss = overall_loss #+ distill_loss # + overall_loss + optical_loss
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(Net.frame_restorer.module.parameters(), max_norm=20, norm_type=2)
                    optimizer.step()

                with torch.no_grad():
                    save_img = torchvision.utils.make_grid(torch.cat([gt_tensor.contiguous().view(-1, c, h, w), seq_input.contiguous().view(-1, c, h, w)], 0), nrow=7)
                    torchvision.utils.save_image(save_img, os.path.join(valid_dir, "target.jpg"))
                    torchvision.utils.save_image(torch.cat([frame_pred, seq_input_warp.contiguous().view(-1, c, h, w), frame_target],0), os.path.join(valid_dir, "seq_input.jpg"))
                    torchvision.utils.save_image(center_median, os.path.join(valid_dir, "center_median.jpg"))

                overall_loss_list.append(overall_loss.item())
                if distill_loss != 0:
                    distill_loss_list.append(distill_loss.item())

                network_time = datetime.now() - ts

                info = "[GPU %d]: " %(opts.gpu)
                info += "Epoch %d; Batch %d / %d; " %(Net.epoch, iteration, len(data_loader))

                batch_freq = opts.batch_size / (data_time.total_seconds() + network_time.total_seconds())
                info += "data loading = %.3f sec, network = %.3f sec, batch = %.3f Hz\n" %(data_time.total_seconds(), network_time.total_seconds(), batch_freq)
                info += "\tmodel = %s\n" %opts.model_name

                loss_writer.add_scalar('Rect Loss', overall_loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("Rect Loss", overall_loss.item())

                if not isinstance(distill_loss, int):
                    distill_loss = distill_loss.item()
                loss_writer.add_scalar('Distill Loss', distill_loss, total_iter)
                info += "\t\t%25s = %f\n" %("Distill Loss", distill_loss)

                print(info)
                ts = datetime.now()

        utils.save_model(Net, optimizer, optimizer_flow, lr_scheduler, lr_scheduler_flow, opts)        
        lr_scheduler.step(np.mean(overall_loss_list))
        prev_epoch_distill_loss = np.mean(distill_loss_list)
        lr_scheduler_flow.step(prev_epoch_distill_loss)

        ################################# test #################################
        if opts.test_list_filename:
            PSNRs = []
            PSNRs_spa = []
            times = []
            prev_file_name = None
            for iteration, (batch, batch_gt, file_name) in enumerate(val_loader, 1):
                if iteration >= 310:
                    break
                if file_name != prev_file_name:
                    prev_file_name = file_name
                    Net.set_new_video()
                print(file_name[0])
                file_name = file_name[0].replace("../", "")
                os.makedirs(os.path.join(valid_dir, 'gt', file_name), exist_ok=True)
                os.makedirs(os.path.join(valid_dir, 'input', file_name), exist_ok=True)
                os.makedirs(os.path.join(valid_dir, 'pred', file_name), exist_ok=True)
                os.makedirs(os.path.join(valid_dir, 'pred_spa', file_name), exist_ok=True)
                os.makedirs(os.path.join(valid_dir, 'diff', file_name), exist_ok=True)

                frame_gt = batch_gt[3]
                with torch.no_grad():
                    batch_ = []
                    for frame in batch:
                        frame = frame.cuda()
                        frame, f_h_pad, f_w_pad = utils.align_to_f(frame, opts.size_multiplier)
                        batch_.append(frame)

                    frame_i0, \
                    frame_i1, \
                    frame_i2, \
                    frame_i3, \
                    frame_i4, \
                    frame_i5, \
                    frame_i6 = batch_

                    [b, c, h, w] = frame_i0.shape

                    seq_around = (frame_i0,
                                  frame_i1,
                                  frame_i2,
                                  frame_i4,
                                  frame_i5,
                                  frame_i6)

                    num_around = len(seq_around)
                    seq_input = torch.cat(batch_, 1).view(b, -1, c, h, w)
                    # seq_input[:,:,3] = softMedian(torch.cat([frame.unsqueeze(2) for frame in warp_inward]+[frame_i3.unsqueeze(2)], 2), 2) 
                    time1 = time.time()
                    frame_pred, frame_pred_spa, _, _ = Net(seq_input)
                    time2 = time.time()
                    used_time = time2-time1
                    times.append(used_time)
                    frame_pred = frame_pred[:, :, 0:h-f_h_pad, 0:w-f_w_pad]
                    frame_pred_spa = frame_pred_spa[:, :, 0:h-f_h_pad, 0:w-f_w_pad]
                print("Time:", used_time)
                fusion_frame_pred = utils.tensor2img(frame_pred)
                fusion_frame_pred_spa = utils.tensor2img(frame_pred_spa)

                output_filename = os.path.join(valid_dir, 'pred', file_name, "%05d.jpg" % iteration)
                utils.save_img(fusion_frame_pred, output_filename)

                output_filename_spa = os.path.join(valid_dir, 'pred_spa', file_name, "%05d.jpg" % iteration)
                utils.save_img(fusion_frame_pred_spa, output_filename_spa)

                frame_input = utils.tensor2img(frame_i3.view(b, c, h, w))
                output_filename = os.path.join(valid_dir, 'input', file_name, "%05d.jpg" % iteration)
                utils.save_img(frame_input, output_filename)

                frame_gt = utils.tensor2img(frame_gt.view(b, c, h, w))
                output_filename = os.path.join(valid_dir, 'gt', file_name, "%05d.jpg" % iteration)
                utils.save_img(frame_gt, output_filename)

                frame_diff = utils.tensor2img((frame_i3).view(b, c, h, w)) - frame_gt
                output_filename = os.path.join(valid_dir, 'diff', file_name, "%05d.jpg" % iteration)
                utils.save_img(frame_diff, output_filename)

                psnr = utils.compute_psnr(frame_gt, fusion_frame_pred, MAX=1.)
                psnr_spa = utils.compute_psnr(frame_gt, fusion_frame_pred_spa, MAX=1.)
                print(psnr, psnr_spa)
                PSNRs.append(psnr)
                PSNRs_spa.append(psnr_spa)
            loss_writer.add_scalar('Epoch PSNR', np.mean(PSNRs), Net.epoch)
            loss_writer.add_scalar('Epoch PSNR_spa', np.mean(PSNRs_spa), Net.epoch)
            lr_cur = optimizer.state_dict()['param_groups'][0]['lr']
            loss_writer.add_scalar('Epoch LR', lr_cur, Net.epoch)
            lr_flow_cur = optimizer_flow.state_dict()['param_groups'][0]['lr']
            loss_writer.add_scalar('Epoch Flow LR', lr_flow_cur, Net.epoch)

            loss_writer.add_scalar('Epoch overall loss', np.mean(np.nan_to_num(overall_loss_list)), Net.epoch)
            loss_writer.add_scalar('Epoch distill loss', np.mean(np.nan_to_num(distill_loss_list)), Net.epoch)
            print("Used Time:", np.mean(times))
            with open(os.path.join(valid_dir, "psnr_val.txt"), "a+") as f_:
                for iteration, psnr in enumerate(PSNRs):
                    f_.write("{}: {}\n".format(iteration, psnr))
                f_.write("Average: {}\n".format(np.mean(PSNRs)))
             
