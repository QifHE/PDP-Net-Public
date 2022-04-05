import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append("//path//to//the//repo//")
from dataset.ShapeNetDataset import *
from extensions.chamfer_dist import ChamferDistance
from models.PDP-Net import PDP_Net_Model
from utils.metrics import AverageValueMeter
from utils.utils import save_model, save_paths, weights_init

##### Input options #####
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--model', type=str, default = 'PDP-Net',  help='optional reload model')
parser.add_argument('--workers', type=int, default=16, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=80, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default = 2048,  help='number of points')
parser.add_argument('--gpu_n', type=int, default = 0,  help='cuda gpu device number')
parser.add_argument('--lrate', type=float, default = 0.0001,  help='learning rate')
parser.add_argument('--lrate_decay', type=float, default=0.7, help='lr decay rate [default: 0.7]')

opt = parser.parse_args()

##### Initialize Variables #####
dir_name, logname  = save_paths(opt.model, "train_PDP-Net", "ShapeNetDataset", "PDP-Net")

loss_train = AverageValueMeter()
loss_test = AverageValueMeter()

loss_train_up = AverageValueMeter()
rec_loss_train_up2 = AverageValueMeter()
rec_loss_train_down1 = AverageValueMeter()
loss_train_down = AverageValueMeter()

loss_test_up = AverageValueMeter()
rec_loss_test_up2 = AverageValueMeter()
rec_loss_test_down1 = AverageValueMeter()
loss_test_down = AverageValueMeter()

loss_CD = AverageValueMeter()

best_loss = 200000

CD_compute = ChamferDistance()

tensorBoardPath = os.path.join('log',opt.model,'tensorBoardLog')
tensorBoardWriter = SummaryWriter(tensorBoardPath)

##### Shapenet-Part Dataloader #####
class_choice = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Guitar': 6, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Skateboard': 14, 'Table': 15}
dataset_dir = './data/shapenet_part'

dataset_train = ShapeNetDataset(root_dir=dataset_dir, class_choice=class_choice, npoints=int(opt.num_points), split='train')
dataloader_train = DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
dataset_test = ShapeNetDataset(root_dir=dataset_dir, class_choice=class_choice, npoints=int(opt.num_points), split='test')
dataloader_test = DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

##### CUDA Initialization ##### 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:" + str(opt.gpu_n) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("====== CUDA ACTIVATED ======")
    torch.cuda.set_device(device)

##### Model Initialization ##### 
network = PDP_Net_Model(device)
network.apply(weights_init)
network.cuda()

#####  Optimizer Initialization ##### 
network_optimizer = optim.Adam(network.parameters(),
        lr=opt.lrate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0)

LEARNING_RATE_CLIP = 0.01 * opt.lrate

#####  Load the pretrained model under log directory ##### 
if opt.model != '' and os.path.isfile("log/" + opt.model + "/model.pth"):
    
    model_checkpoint = torch.load("log/" + opt.model + "/model.pth")
    network.load_state_dict(model_checkpoint['state_dict'])
    network_optimizer.load_state_dict(model_checkpoint['optimizer'])

##### Logging #####
with open(logname, 'a') as f:
        f.write(str(network) + '\n')

torch.backends.cudnn.enabled = False

##### TRAINING #####
for epoch in range(opt.nepoch):
    network.train()

    lr = max(opt.lrate * (opt.lrate_decay ** (epoch // 20)), LEARNING_RATE_CLIP)
    for param_group in network_optimizer.param_groups:
        param_group['lr'] = lr
    
    # Train
    for i, data in enumerate(dataloader_train, 0):

        network_optimizer.zero_grad()

        name, in_partial, in_hole, in_complete = data
        
        # Data to GPU
        in_partial = in_partial.contiguous().float().to(device)
        in_hole = in_hole.contiguous().float().to(device)
        in_complete = in_complete.contiguous().float().to(device)
        
        in_partial = in_partial.transpose(2,1).contiguous()
        in_hole = in_hole.transpose(2,1).contiguous()
        in_complete = in_complete.transpose(2,1).contiguous()

        # Data to Network
        out_complete, loss_up, loss_down, loss_expansion = network(in_partial, in_hole, in_complete)

        # EMD Loss and Backward Propagation
        loss = loss_up + loss_down + 0.01 * loss_expansion
        loss.backward()
        
        # Optimizer
        network_optimizer.step()

        # CD Loss for reference
        CD_Loss = CD_compute(out_complete, in_complete.transpose(2,1).contiguous())
        # values to plot and save
        loss_train.update(loss.item())
        loss_train_up.update(loss_up.item())
        loss_train_down.update(loss_down.item())
        loss_CD.update(CD_Loss.item())
        
        # Print per batch
        print("Train -> Epoch: {} Batch: ({}/{}) Train Loss: {} CD Loss {} lr: {}".format(epoch, i, len(dataloader_train), loss_train_down.avg, loss_CD.avg, lr))
        
    # Validation per five epoches
    if epoch % 5 == 0:
        network.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader_test, 0):
                name, in_partial, in_hole, in_complete = data
        
                in_partial = in_partial.contiguous().float().to(device)
                in_hole = in_hole.contiguous().float().to(device)
                in_complete = in_complete.contiguous().float().to(device)
                
                in_partial = in_partial.transpose(2,1).contiguous()
                in_hole = in_hole.transpose(2,1).contiguous()
                in_complete = in_complete.transpose(2,1).contiguous()

                out_complete, loss_up, loss_down, loss_expansion = network(in_partial, in_hole, in_complete)
                
                loss = loss_up + loss_down + 0.01 * loss_expansion
                
                loss_test.update(loss.item())
                loss_test_up.update(loss_up.item())
                loss_test_down.update(loss_down.item())
                                
                # Print per batch
                print("Test -> Epoch: {} Batch: ({}/{}) Loss: {} lr: {}".format(epoch, i, len(dataloader_train), loss_test_down.avg, lr))
    
    # Log per epoch
    tensorBoardWriter.add_scalar('Train Loss/Total EMD Loss', loss_train.avg, epoch)
    tensorBoardWriter.add_scalar('Train Loss/loss_up', loss_train_up.avg, epoch)
    tensorBoardWriter.add_scalar('Train Loss/loss_down', loss_train_down.avg, epoch)
    tensorBoardWriter.add_scalar('Eval Loss/Total EMD Loss', loss_test.avg, epoch)
    tensorBoardWriter.add_scalar('Eval Loss/loss_up1', loss_test_up.avg, epoch)
    tensorBoardWriter.add_scalar('Eval Loss/loss_down2', loss_test_down.avg, epoch)
    tensorBoardWriter.add_scalar('Learning Rate', lr, epoch)

    loss_train.end_epoch()
    loss_test.end_epoch()
    loss_train_up.end_epoch()
    loss_train_down.end_epoch()
    loss_test_up.end_epoch()
    loss_test_down.end_epoch()
    loss_CD.end_epoch() 
    
    # Save model per five Epoch
    if (epoch % 5 == 0) or (epoch == opt.nepoch - 1):
        print("Loss reduced from %8.5f to %8.5f" % (best_loss, loss_test.avg))
        best_loss = loss_test.avg
        save_model(network.state_dict(), network_optimizer.state_dict(), logname, dir_name, loss_train, loss_test, epoch, opt.lrate, loss_train.avg, loss_test.avg)

##### END Training #####
save_model(network.state_dict(), network_optimizer.state_dict(), logname, dir_name, loss_train, loss_test, epoch, opt.lrate, loss_train.avg, loss_test.avg, net_name= "model")
