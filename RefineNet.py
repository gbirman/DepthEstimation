import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torchsummary import summary

from models import modules, net, resnet, densenet, resnet4
import net_mask
import loaddata4
import util
import numpy as np
import sobel
import net_mask

import time
import os
import copy
import shutil
import glob
import logging

import sklearn

import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.image
matplotlib.rcParams['image.cmap'] = 'viridis'

from itertools import *

import pdb

parser = argparse.ArgumentParser(description='single depth estimation')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')


# threshold: 50% => 0.5% 

def define_model(encoder='resnet'):
    if encoder is 'resnet':
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if encoder is 'densenet':
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if encoder is 'senet':
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if encoder is 'resnet4':
        original_model = resnet4.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
        
    return model
   

def main():
    global args
    args = parser.parse_args()

#     model_selection = 'resnet4'
#     model = define_model(encoder = model_selection)

#     original_model2 = net_mask.drn_d_22(pretrained=True)
#     model2 = net_mask.AutoED(original_model2)  
  
    model_selection = 'resnet'
    model = define_model('resnet')
    rmodel = define_model('resnet')
    
    if torch.cuda.device_count() == 8:
        print('8 devices')
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        rmodel = torch.nn.DataParallel(rmodel, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
    elif torch.cuda.device_count() == 4:
        print('4 devices')
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        rmodel = torch.nn.DataParallel(rmodel, device_ids=[0, 1, 2, 3]).cuda()
        batch_size = 32
    else:
        print('1 device')
        model = torch.nn.DataParallel(model).cuda()
        rmodel = torch.nn.DataParallel(rmodel).cuda()
        batch_size = 8
    
    model.load_state_dict(torch.load('./pretrained_model/model_' + model_selection))
    
    rmodel.load_state_dict(torch.load('./pretrained_model/model_' + model_selection))
    rmodel.module.E.conv1.stride = (1,1) # ensures that input size == output size 
    
    
#     # reinitialize weights 
#     def init_weights(m):
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#             m.reset_parameters()
#     rmodel.apply(init_weights)
        
    # don't backpropagate through base module 
    for param in model.parameters():
        param.requires_grad = False  
        
        
#     model.module.E.conv1.stride = (1,1)
#     model.module.E.conv1.weight.requires_grad = True
        
#     #add 4th channel to conv weights 
#     conv1 = model.module.E.conv1.weight 
#     mask = torch.empty(64,1,7,7).cuda()
#     nn.init.xavier_normal_(mask)
#     points = torch.empty(64,1,7,7).cuda()
#     nn.init.xavier_normal_(points)
# #     nn.init.zeros_(mask)
# #     nn.init.zeros_(points)
#     conv1_new = torch.cat((conv1,mask,points),dim=1)
    conv1_new = torch.empty(64,1,7,7).cuda()
#     nn.init.zeros_(conv1_new)
    rmodel.module.E.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
    rmodel.module.E.conv1.weight = nn.Parameter(conv1_new, requires_grad=True)
    
    cudnn.benchmark = True
#     list(fc1.parameters()) + list(fc2.parameters()) adding two sets of parameters
    
    start_epoch = args.start_epoch
    epochs = args.epochs
    
    output_dir = './cp/rel_selmax_ml_adjnone/'
    
#     PATH = output_dir + 'checkpoint-'+str(start_epoch-1)+'.pth.tar'
    PATH = './cp/identity/' + 'checkpoint-101.pth.tar'
#     PATH = output_dir + 'checkpoint-0.pth.tar'
    checkpoint = torch.load(PATH)   
    rmodel.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#     conv1_og = model.module.E.conv1.weight
    conv1 = rmodel.module.E.conv1.weight
    mask = torch.empty(64,1,7,7).cuda()
#     nn.init.xavier_normal_(mask)
    nn.init.normal_(mask, mean=0.0, std=1e-4)
    points = torch.empty(64,1,7,7).cuda()
#     nn.init.xavier_normal_(points)
    nn.init.normal_(points, mean=0.0, std=1e-4)
    conv1_new = torch.cat((conv1,mask,points),dim=1)
    rmodel.module.E.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
    rmodel.module.E.conv1.weight = nn.Parameter(conv1_new, requires_grad=True)
    
    optimizer = torch.optim.Adam(rmodel.parameters(), args.lr, weight_decay=args.weight_decay)
    
#     PATH = output_dir + 'checkpoint-10.pth.tar'
#     PATH = output_dir + 'checkpoint-0.pth.tar'
#     checkpoint = torch.load(PATH)   
#     rmodel.load_state_dict(checkpoint['model_state_dict'])
    
# #     train_loader = loaddata4.getTrainingData(batch_size)
    dataloader = loaddata4.getTrainValData(batch_size)
    train_model(dataloader, rmodel, model, optimizer, start_epoch, epochs, output_dir)
        
#     model.load_state_dict(torch.load('./1percentnomasklradjustdepthgradnormal_traincheckpoints/checkpoint-49.pth.tar'))
    test_loader = loaddata4.getTestingData(1)
    test_model(test_loader, rmodel, model, output_dir)

#     torch.save(model.state_dict(), './pretrained_model/model_'+model_selection)        



def train_model(dataloader, rmodel, model, optimizer, start_epoch, epochs, output_dir): 
    
#     # initialize grid for point selection 
#     step = 5 
#     grid = torch.zeros([114,152],dtype=torch.uint8)
#     x = np.arange(0,114,step)
#     y = np.arange(0,152,step)
#     X,Y = np.meshgrid(x,y)
#     grid[X.ravel(),Y.ravel()]=1
#     grid = grid.view(-1).cuda()
                
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    best_acc = np.inf
    
    valfile = open(output_dir + 'validation.log', 'w') 
    trainfile = open(output_dir + 'training.log', 'w')
    resultsfile = open(output_dir + 'results.log', 'w')
    
#     d1 = 114
#     d2 = 152
#     xx, yy = torch.meshgrid([torch.arange(d1), torch.arange(d2)])
#     xx = xx.cuda()
#     yy = yy.cuda()
            
    for epoch in range(start_epoch, epochs):
        
#         adjust_learning_rate(optimizer, epoch)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                rmodel.train()  # Set model to training mode
                model.eval()  # Set model to training mode
            else:
                continue
                rmodel.eval()   # Set model to evaluate mode
                model.eval()   # Set model to evaluate mode
        
            since = time.time()

            threshold_percent = AverageMeter()
            batch_time = AverageMeter()
            losses = AverageMeter()
            totalNumber = 0
            acc = AverageMeter()
            errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                        'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            cos = nn.CosineSimilarity(dim=1, eps=0)
            get_gradient = sobel.Sobel().cuda()
            mu = 1
            lamda = 1
            alpha = 0.5
    
            # Iterate over data.
            end = time.time()
            for i, sample_batched in enumerate(dataloader[phase]):

                image, depth, mask, imagedepth, imagemask = \
                    sample_batched['image'].cuda(), sample_batched['depth'].cuda(), sample_batched['mask'].cuda(), sample_batched['imagedepth'].cuda(), sample_batched['imagemask'].cuda()
                
                bs = image.size(0)
                
                depth_mean = 0 #1.8698
                depth_std = 10 #1.6716
                
#                 mi, _ = depth.view(bs,-1).min(1)
#                 mi = mi.view(-1,1,1,1)
#                 ma, _ = depth.view(bs,-1).max(1)
#                 ma = ma.view(-1,1,1,1)
#                 depth = (depth - mi)/(ma-mi)

                depth = (depth-depth_mean)/depth_std
                
                # zero the parameter gradients
                optimizer.zero_grad()

                zero_points = torch.zeros_like(imagemask)
                base_input = image #torch.cat((image,imagemask, zero_points),dim=1)
                
                with torch.set_grad_enabled(False):
                    output = model(base_input)
#                     output2x = nn.functional.interpolate(output, size=None, scale_factor=2)
#                     output = output.squeeze().view(114,152).data.cpu().float().numpy()
#                     matplotlib.image.imsave(output_dir+'base'+'.png', output)
                    
    
#                     # normalize
#                     mi, _ = output.view(bs,-1).min(1)
#                     mi = mi.view(-1,1,1,1)
#                     ma, _ = output.view(bs,-1).max(1)
#                     ma = ma.view(-1,1,1,1)
#                     output = (output - mi)/(ma-mi)

                    output = (output-depth_mean)/depth_std

# #                     imagemask = image[:,3,:,:].unsqueeze(1)
                    P = torch.zeros_like(output)
                          
#                     diff_map = (output/depth - 1)*mask # < 0 if output point is relatively farther  
#                     adj_diff_map = torch.abs(diff_map)>0.5
#                     # 3 levels of ordinal relatinos
# #                     abs_diff_map = torch.abs(diff_map)
# #                     K = 3
                        
                    diff_map = (output/depth - 1)*mask 
                    diff_map_reverse = (depth/output - 1)*mask
                    adj_diff_map = torch.max(torch.abs(diff_map),torch.abs(diff_map_reverse))>0.25

                    threshold_percent.update(adj_diff_map.sum()/mask.sum(), 1) 
                    
# #                     diff_map = output - depth
# #                     adj_diff_map = torch.zeros_like(diff_map).byte()
# #                     for j in range(bs):
# #                         subdiff = diff_map[j,:,:,:]
# #                         adj_diff_map[j,:,:,:] = torch.abs(subdiff)>0.1*(subdiff.max())

#                     selection_percent = 0.25
#                     r = (torch.rand_like(mask) <= selection_percent) * adj_diff_map
#                     P = torch.sign(diff_map) * mask * r.float()
            
#                     m = r.view(bs,-1).data.nonzero()
#                     sig = 5
#                     d1 = r.size(2)                
#                     d2 = r.size(3)
# #                     xx, yy = torch.meshgrid([torch.arange(d1), torch.arange(d2)])
# #                     xx = xx.cuda()
# #                     yy = yy.cuda()
#                     F = torch.zeros_like(r).float()
#                     g = F[0,0,:,:]
#                     old_idx = 0
#                     for idx in m:        
#                         mask_idx = idx[0]
#                         if (mask_idx != old_idx):
#                             F[mask_idx-1,0,:,:] = g/g.max()
#                             g = F[mask_idx,0,:,:]
#                         one_idx = idx[1]
#                         x0 = one_idx // d2
#                         y0 = one_idx % d2
#                         t = -0.5 * ((xx-x0)**2 + (yy-y0)**2).float() / sig**2
#                         kernel = torch.exp(t)
#                         g += kernel
#                     F[mask_idx,0,:,:] = g/g.max()
#                     r = F

                    D = diff_map.view(bs, -1)
                    m = torch.max(torch.abs(D),torch.abs(diff_map_reverse).view(bs,-1)).argmax(1).view(-1,1)

                    # P is -1 at point where 
                    updates = torch.sign(D.gather(1,m))
                    P.view(bs,-1).scatter_(1,m,updates)      
                                        
#                     new_input = torch.cat((image, P),dim=1)
                    
#                     d = torch.abs(diff_map[0,:,:,:])
#                     depth = d.squeeze().view(228,304).data.cpu().float().numpy()
#                     implot = plt.imshow(depth)
#                     d2 = diff_map.size(3)
                    
#                     indices = torch.cat((m // d2, m % d2), dim=1).data.cpu()

#                     plt.scatter(indices[0,1],indices[0,0],c='r')
#                     plt.savefig(output_dir+'point'+'.png')
                     
#                     mask = mask[0,:,:,:]
#                     mask = mask.squeeze().view(114,152).data.cpu().float().numpy()
#                     matplotlib.image.imsave(output_dir+str(i)+'mask'+'.png', mask)
                          
                
#                     for j in range(bs): # batch size 
# #                         submask = mask[j,:,:,:].view(-1).byte()
#                         submask = mask[j,:,:,:].view(-1).byte()
#                         z = output[j,:,:,:].view(-1)
#                         gt_depth = depth[j,:,:,:].view(-1)
#                         subP = P[j,:,:,:].view(-1)
                        
#                         num_sampled = 1 # number of randomly sample points pairs in mask 
#                         NZ = submask.nonzero()
#                         sample_idx = torch.multinomial(torch.ones(submask.nonzero().size(0)),num_sampled*2) 
#                         randomly_sampled = NZ[sample_idx,:]
#                         J = randomly_sampled.view(num_sampled,2)
# #                         # choose 
# #                         for ik in randomly_sampled:
# #                             diff = gt_depth[ik]-z[i]
                        
#                         # select point-pair with greatest discrepancy 
#                         best_pair = None
#                         max_diff = 0
#                         for (ik,jk) in J: #combinations(randomly_sampled, 2):
#                             gt_diff = gt_depth[ik]/gt_depth[jk]
#                             z_diff = z[ik]/z[jk]
#                             diff = gt_diff - z_diff
#                             if torch.abs(diff) > max_diff:
#                                 best_pair = (ik,jk)
#                                 max_diff = torch.abs(diff)
                                
#                         ik = best_pair[0]
#                         jk = best_pair[1]
#                         if max_diff<0: # predicted P1 should be relatively closer, P2 relatively further 
#                             subP[ik] = 1
#                             subP[jk] = -1 
#                         else: 
#                             subP[ik] = -1
#                             subP[jk] = 1 
                    
                    new_input = torch.cat((output, mask, P),dim=1)
                    
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
#                     output = model(image).detach()
#                     P = torch.zeros_like(output).cuda()
                    
                    refined_output = rmodel(new_input)
                
#                     P.zero_()

#                     diff_map = (refined_output.detach()/depth - 1)*mask # < 0 if output point is relatively farther  

#                     D = diff_map.view(bs, -1)
#                     m = torch.abs(D).argmax(1).view(-1,1)

#                     updates = torch.sign(D.gather(1,m))
#                     P.view(bs,-1).scatter_(1,m,updates)
                    
#                     new_input = torch.cat((refined_output, mask, P),dim=1)
                    
#                     refined_output = rmodel(new_input)
                    
#                     P.zero_()

#                     diff_map = (refined_output.detach()/depth - 1)*mask # < 0 if output point is relatively farther  

#                     D = diff_map.view(bs, -1)
#                     m = torch.abs(D).argmax(1).view(-1,1)

#                     updates = torch.sign(D.gather(1,m))
#                     P.view(bs,-1).scatter_(1,m,updates)
                    
#                     new_input = torch.cat((refined_output, mask, P),dim=1)
                    
#                     refined_output = rmodel(new_input)
                    
#                     routput_im = refined_output.squeeze().view(114,152).data.cpu().float().numpy()
#                     matplotlib.image.imsave(output_dir+'refined'+'.png', routput_im)
                    
#                     output_im = refined_output.squeeze().view(114,152).data.cpu().float().numpy()
#                     matplotlib.image.imsave(output_dir+'output'+'.png', output_im)
                    
                    byte_mask = mask.byte()
                    masked_depth = depth[byte_mask]
                    masked_output = output[byte_mask]
                    masked_refined_output = refined_output[byte_mask]
        
                    depth_ = depth
                    output_ = refined_output

                    ones = torch.ones(depth_.size(0), 1, depth_.size(2),depth_.size(3)).float().cuda()
                    depth_grad = get_gradient(depth_)
                    output_grad = get_gradient(output_)
                    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth_)
                    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth_)
                    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth_)
                    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth_)

                    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
                    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

                    loss_depth = torch.log(torch.abs(output_ - depth_) + alpha)
                    loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + alpha)
                    loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + alpha)
                    loss_grad = loss_dx+loss_dy
                    loss_normal = torch.abs(1 - cos(output_normal, depth_normal))
                    
                    loss_normal.unsqueeze_(1)
                    loss_depth_masked = loss_depth[byte_mask].mean()
                    loss_grad_masked = loss_grad[byte_mask].mean()
                    loss_normal_masked = loss_normal[byte_mask].mean()
                    
                    loss = loss_depth_masked + mu*loss_normal_masked + lamda*loss_grad_masked
                    
#                     loss_point_depth_masked = loss_depth.view(bs,-1).gather(1,m).mean() 
#                     loss_point_grad_masked = loss_grad.view(bs,-1).gather(1,m).mean() 
#                     loss_point_normal_masked = loss_normal.view(bs,-1).gather(1,m).mean() 
                    
#                     loss_point_depth_masked = (loss_depth)[byte_mask*r].mean() 
#                     loss_point_grad_masked = (loss_grad)[byte_mask* r].mean() 
#                     loss_point_normal_masked = (loss_normal)[byte_mask* r].mean() 
                    
#                     point_loss = loss_point_depth_masked + mu*loss_point_grad_masked + lamda*loss_point_normal_masked
            
#                     loss += 0.5*point_loss
                    
# #                     #berHu loss (Deeper Depth Prediction with Fully Convolutional Residual Networks)
#                     diff = torch.abs(masked_refined_output - masked_depth)
#                     c = torch.max(diff)/5
#                     bh = diff.where(diff<=c, (diff**2+c**2)/(2*c))
#                     bh_loss = bh.sum()/mask.sum()
                
#                     # point loss 
#                     diff_map = torch.abs((refined_output/depth - 1)*mask)
#                     D = diff_map.view(bs, -1)
#                     point_loss = torch.sum(D.gather(1,m))/bs
                    
#                     loss = bh_loss + 3*point_loss
                    
#                     loss = diff[diff<=c] + ((diff[diff>c])**2)/(2*c)
                    
#                     # loss function 
#                     rankingloss = 0
#                     for j in range(bs): # batch size 
#                         submask = mask[j,:,:,:].view(-1).byte()
#                         z = refined_output[j,:,:,:].view(-1)
#                         gt_depth = depth[j,:,:,:].view(-1)
# #                         selection_points = torch.mul(submask,grid)
                        
# #                         NZ = selection_points.nonzero()
#                         NZ = submask.nonzero()
                        
#                         M = 10 # number of pairs of points selected k = 1..M-1
#                         sample_idx = torch.multinomial(torch.ones(NZ.size(0)),2*M)
                
# #                         if NZ.size(0) < 2*M:
# #                             sample_idx = torch.multinomial(torch.ones(submask.nonzero().size(0)),2*M)                             
# # #                             M = NZ.size(0)//2
# #                         else:
# #                             sample_idx = torch.multinomial(torch.ones(NZ.size(0)),2*M)
#                         J = NZ[sample_idx,:]
#                         J = J.view(M,2)
                        
# #                         r = torch.zeros(M,).cuda()
# #                         k = 0
#                         tau = 0.02
#                         rloss = 0
#                         for ik, jk in J: # M*(M-1)/2 loop iterations (so keep M small!)  
#                             if (gt_depth[ik]/gt_depth[jk] > 1 + tau):
#                                 rloss += torch.log(1+torch.exp(-z[ik]+z[jk])) # jk closer than ik 
#     #                             r[k] = 1 
#                             elif (gt_depth[jk]/gt_depth[ik] > 1 + tau): # ik closer than jk 
#                                 rloss += torch.log(1+torch.exp(z[ik]-z[jk]))
#     #                             r[k] = -1
#                             else: # equal 
#                                 rloss += (z[ik]-z[jk])**2
#     #                             r[k] = 0
#     #                         k = k + 1
                        
#                         rankingloss += rloss/M
#                     rl = rankingloss/bs
                
#                     loss += 2*rl
                    
                    losses.update(loss.item(), bs)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
#                 masked_depth = depth[mask.byte()]
#                 masked_output = output[mask.byte()]
#                 if phase == 'val':
#                     depth_val = masked_depth.data.cpu().numpy()
#                     output_val = masked_output.data.cpu().numpy()
#                     indices = np.argsort(depth_val,kind='stable')
#                     idx = np.argsort(output_val[indices],kind='stable')
#                     n = idx.size
#                     num_swaps = countSwaps(idx, n)
#                     acc.update(num_swaps/n)
                        
                # statistics
                batch_time.update(time.time() - end)
                end = time.time()

                errors = util.evaluateError(masked_refined_output,masked_output)
                errorSum = util.addErrors(errorSum, errors, bs)
                totalNumber = totalNumber + bs
                averageError = util.averageErrors(errorSum, totalNumber)

                if i%100==0:
                    out_str = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'
                      .format(epoch, i, len(dataloader[phase]), batch_time=batch_time, loss=losses))
                    print(out_str)
                    trainfile.write(out_str + '\n')
        
            # get accuracy as RMSE 
#             epoch_acc = np.sqrt(averageError['MSE'])
#             epoch_acc = acc.avg
            
            # deep copy the model
            if phase == 'val':
                epoch_acc = np.sqrt(averageError['MSE'])
                valfile.write('epoch: ' + str(epoch) + ', rmse: ' + str(epoch_acc) + '\n')
                if epoch_acc < best_acc:
                    best_acc = epoch_acc
                    is_best = True
                else:
                    is_best = False
            save_checkpoint(rmodel, optimizer, loss, False, epoch, output_dir)

            time_elapsed = time.time() - since
            s1 = 'Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
            s2 = 'rmse:' + str(np.sqrt(averageError['MSE']))
            s3 = 'abs_rel:' + str(averageError['ABS_REL'])
            s4 = 'mae:'+ str(averageError['MAE']) 
            print(s1)
            print(s2)
            print(s3)
            print(s4)
            resultsfile.write(phase + '\n')
            resultsfile.write(s1 + '\n')
            resultsfile.write(s2 + '\n')
            resultsfile.write(s3 + '\n')
            resultsfile.write(s4 + '\n')
            
            print(threshold_percent.avg)
              
            valfile.write('\n')
            trainfile.write('\n')
            resultsfile.write('\n')
            
#             print('avg. num swaps:',epoch_acc)
            
        print()

    print('errors: ', averageError)

def test_model(test_loader, rmodel, model, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
            
#     logging.basicConfig(filename=output_dir + 'testing.log',filemode='w',format='%(message)s',level=logging.INFO)
    testfile = open(output_dir + 'testing.log', 'w')
    
    since = time.time()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    base_losses = AverageMeter()
    totalNumber = 0
    base_totalNumber = 0
    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
    base_errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    rmodel.eval()   # Set model to evaluate mode
    model.eval()
    
    # Iterate over data.
    end = time.time()
    for i, sample_batched in enumerate(test_loader['val']):

        image, depth, mask, imagedepth, imagemask = \
                    sample_batched['image'].cuda(), sample_batched['depth'].cuda(), sample_batched['mask'].cuda(), sample_batched['imagedepth'].cuda(), sample_batched['imagemask'].cuda()          

        bs = image.size(0)
        
        depth_mean = 0 #1.8698
        depth_std = 10 # 1.6716

#                 mi, _ = depth.view(bs,-1).min(1)
#                 mi = mi.view(-1,1,1,1)
#                 ma, _ = depth.view(bs,-1).max(1)
#                 ma = ma.view(-1,1,1,1)
#                 depth = (depth - mi)/(ma-mi)

        depth = (depth-depth_mean)/depth_std
                
        zero_points = torch.zeros_like(imagemask)
        base_input = image #torch.cat((image,imagemask, zero_points),dim=1)
        
        with torch.set_grad_enabled(False):
            output = model(base_input)
#             output2x = nn.functional.interpolate(output, size=None, scale_factor=2)
#                     output = output.squeeze().view(114,152).data.cpu().float().numpy()
#                     matplotlib.image.imsave(output_dir+'base'+'.png', output)

#             # normalize
#             mi, _ = output.view(bs,-1).min(1)
#             mi = mi.view(-1,1,1,1)
#             ma, _ = output.view(bs,-1).max(1)
#             ma = ma.view(-1,1,1,1)
#             output = (output - mi)/(ma-mi)
            
            output = (output-depth_mean)/depth_std
    
#             imagemask = image[:,3,:,:].unsqueeze(1)
            P = torch.zeros_like(output)

            diff_map = (output/depth - 1)*mask 
            diff_map_reverse = (depth/output - 1)*mask
            adj_diff_map = torch.max(torch.abs(diff_map),torch.abs(diff_map_reverse))>0.25
            
# #             diff_map = output - depth
# #             adj_diff_map = torch.zeros_like(diff_map).byte()
# #             for j in range(bs):
# #                 subdiff = diff_map[j,:,:,:]
# #                 adj_diff_map[j,:,:,:] = torch.abs(subdiff)>0.1*(subdiff.max())
        
#             selection_percent = 0.25
#             r = (torch.rand_like(mask) <= selection_percent) * adj_diff_map
#             P = torch.sign(diff_map) * mask * r.float()
                            
            D = diff_map.view(bs, -1)
            m = torch.max(torch.abs(D),torch.abs(diff_map_reverse).view(bs,-1)).argmax(1).view(-1,1)

            # P is -1 at point where 
            updates = torch.sign(D.gather(1,m))
            P.view(bs,-1).scatter_(1,m,updates)   
                    
            new_input = torch.cat((output, mask, P),dim=1)#torch.cat((image, P),dim=1)
            
            refined_output = rmodel(new_input)

            masked_depth = depth[mask.byte()]
            masked_output = output[mask.byte()]
            masked_refined_output = refined_output[mask.byte()]

            # statistics
            batch_time.update(time.time() - end)
            end = time.time()

#             batchSize = depth.size(0)

            errors = util.evaluateError(depth_std*masked_refined_output+depth_mean, depth_std*masked_depth+depth_mean)
            errorSum = util.addErrors(errorSum, errors, bs)
            totalNumber = totalNumber + bs
            averageError = util.averageErrors(errorSum, totalNumber)
            
        base_totalNumber = base_totalNumber + bs
        # base error
        base_errors = util.evaluateError(depth_std*masked_output+depth_mean, depth_std*masked_depth+depth_mean)
        base_errorSum = util.addErrors(base_errorSum, base_errors, bs)
        base_averageError = util.averageErrors(base_errorSum, base_totalNumber)  
#         # refined error
#         errors = util.evaluateError(routput_, depth_)
#         errorSum = util.addErrors(errorSum, errors, batchSize)
#         averageError = util.averageErrors(errorSum, totalNumber)        
        
        # statistics
        batch_time.update(time.time() - end)
        end = time.time()

        if i%100==0:
            out_str = ('Time {batch_time.val:.2f} ({batch_time.sum:.1f})\t'
#                   'N={n} '
#                   'RL {loss.val:.1f} ({loss.avg:.1f})\t'
                  'RMSE {rmse:.3f} ({rmse_avg:.3f})'
                  'BASE RMSE {base_rmse:.3f} ({base_rmse_avg:.3f})'
                  .format(batch_time=batch_time, loss=losses, rmse=np.sqrt(errors['MSE']), rmse_avg=np.sqrt(averageError['MSE']),base_rmse=np.sqrt(base_errors['MSE']), base_rmse_avg=np.sqrt(base_averageError['MSE'])))
                       
                       #,n=n, base_rmse=np.sqrt(base_errors['MSE']), base_rmse_avg=np.sqrt(base_averageError['MSE'])))     
            print(out_str)      
            testfile.write(out_str + '\n')
#             logging.info(out_str)
        
        # plot approx. 10 images
#         files = glob.glob(dir + '*')
#         for f in files:
#             os.remove(f)
        if i%(len(test_loader['val'])//8)==0: 
            depth = depth.squeeze().view(114,152).data.cpu().float().numpy()
            matplotlib.image.imsave(output_dir+str(i)+'groundtruth'+'.png', depth)

            output = output.squeeze().view(114,152).data.cpu().float().numpy()
            matplotlib.image.imsave(output_dir+str(i)+'base'+'.png', output)
            
            refined_output = refined_output.squeeze().view(114,152).data.cpu().float().numpy()
            matplotlib.image.imsave(output_dir+str(i)+'refine'+'.png', refined_output)
            
            mask = mask.squeeze().view(114,152).data.cpu().float().numpy()
            matplotlib.image.imsave(output_dir+str(i)+'mask'+'.png', mask)

            image = image.squeeze().data.cpu().numpy().transpose(1,2,0)
            image = image[:,:,[0,1,2]]
#             image = image.view(228,304,3).data.cpu().float().numpy()
            matplotlib.image.imsave(output_dir+str(i)+'image'+'.png', image)
    
    time_elapsed = time.time() - since
    s1 = 'Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    s2 = 'rmse:' + str(np.sqrt(averageError['MSE']))
    s3 = 'abs_rel:' + str(averageError['ABS_REL'])
    s4 = 'mae:'+ str(averageError['MAE']) 
    s5 = 'lg10:' + str(averageError['LG10'])
    s6 = 'delta1:' + str(averageError['DELTA1'])
    s7 = 'delta2:' + str(averageError['DELTA2'])
    s8 = 'delta3:' + str(averageError['DELTA3'])
    bs2 = 'base_rmse:' + str(np.sqrt(base_averageError['MSE']))
    bs3 = 'base_abs_rel:' + str(base_averageError['ABS_REL'])
    bs4 = 'base_mae:'+ str(base_averageError['MAE'])
    bs5 = 'lg10:' + str(base_averageError['LG10'])
    bs6 = 'delta1:' + str(base_averageError['DELTA1'])
    bs7 = 'delta2:' + str(base_averageError['DELTA2'])
    bs8 = 'delta3:' + str(base_averageError['DELTA3'])
    
    print(s1)
    print(s2)
    print(s3)
    print(s4)
    print(s5)
    print(s6)
    print(s7)
    print(s8)
    print(bs2)
    print(bs3)
    print(bs4)
    print(bs5)
    print(bs6)
    print(bs7)
    print(bs8)
    testfile.write(s1 + '\n')
    testfile.write(s2 + '\n')
    testfile.write(s3 + '\n')
    testfile.write(s4 + '\n')
    testfile.write(s5 + '\n')
    testfile.write(s6 + '\n')
    testfile.write(s7 + '\n')
    testfile.write(s8 + '\n')
    testfile.write(bs2 + '\n')
    testfile.write(bs3 + '\n')
    testfile.write(bs4 + '\n')
    testfile.write(bs5 + '\n')
    testfile.write(bs6 + '\n')
    testfile.write(bs7 + '\n')
    testfile.write(bs8 + '\n')
                
#     print('base_rl:',base_losses.avg)
#     print('base_rmse:',np.sqrt(base_averageError['MSE'])) 
#     print('base_abs_rel:',base_averageError['ABS_REL']) 
#     print('base_mae:',base_averageError['MAE']) 
#     logging.info('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     logging.info(('rl:',losses.avg))
#     logging.info(('rmse:',np.sqrt(averageError['MSE'])))
#     logging.info(('abs_rel:',averageError['ABS_REL']))
#     logging.info(('mae:',averageError['MAE']))
#     logging.info(('base_rl:',base_losses.avg))
#     logging.info(('base_rmse:',np.sqrt(base_averageError['MSE'])))
#     logging.info(('base_abs_rel:',base_averageError['ABS_REL']))
#     logging.info(('base_mae:',base_averageError['MAE']))
    
# def validate_model(train_loader, validate_loader, model):
    
    
    
    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 5)) # 5 originally

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def save_checkpoint(model, optimizer, loss, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
#     torch.save(state, checkpoint_filename)
        
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_filename)
    
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)

if __name__ == '__main__':
    main()


