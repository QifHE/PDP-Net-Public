import argparse
import csv
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append("//path//to//the//repo//")
from dataset.ShapeNetDataset import *
from extensions.chamfer_dist import ChamferDistance
from models.PDP-Net import PDP_Net_Model
from utils.metrics import AverageValueMeter
from utils.utils import save_model, save_paths, weights_init
from utils.pcutils import mean_min_square_distance, save_point_cloud

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--model', type=str, default = 'PDP-Net',  help='optional reload model path')
parser.add_argument('--workers', type=int,default=16, help='number of data loading workers')
parser.add_argument('--num_points', type=int, default = 2048,  help='number of points')
parser.add_argument('--outputFolder', type=str, default='', help='Folder output')
parser.add_argument('--holeSize', type=int, default=35, help='hole size')


opt = parser.parse_args()

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

##### Model Loading #####
if os.path.isfile("log/" + opt.model + "/model.pth"):
    print(opt.model, os.path.isfile("log/" + opt.model + "/model.pth"))
else:
    print("No Model Found")
    sys.exit()

if opt.model != '' and os.path.isfile("log/" + opt.model + "/model.pth"):
    model_checkpoint = torch.load("log/" + opt.model + "/model.pth",map_location='cuda:0')
    
    print("Model network weights loaded ")
    network.load_state_dict(model_checkpoint['state_dict'])

##### CSV Logging #####
logpath = os.path.join('log',opt.model,'csvLog')
f = open(logpath + "_evaluation.csv",'w',newline='',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(["Category","CD-L2 Loss"])
f.close()

# Shapenet-Part Dataset
class_choice = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Guitar': 6, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Skateboard': 14, 'Table': 15}
categories = class_choice.keys()

R = []

chamfer_dist = ChamferDistance()

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)

for category in categories:

    pred_error = AverageValueMeter()
    gt_error = AverageValueMeter()
    chamfer_error = AverageValueMeter()


    dataset_dir = './data/shapenet_part'

    dataset_test = ShapeNetDataset(root_dir=dataset_dir, class_choice={category}, npoints=2048, split='test', hole_size=opt.holeSize/100)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    
    network.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader_test, 0)):
            
            name, in_partial, in_hole, in_complete = data
        
            in_partial = in_partial.contiguous().float().to(device)
            in_hole = in_hole.contiguous().float().to(device)
            in_complete = in_complete.contiguous().float().to(device)
            
            in_partial = in_partial.transpose(2,1).contiguous()
            in_hole = in_hole.transpose(2,1).contiguous()
            in_complete = in_complete.transpose(2,1).contiguous()

            out_complete, loss_up, loss_down, loss_expansion = network(in_partial, in_hole, in_complete)
            
            in_complete = in_complete.transpose(2,1).contiguous()
            dist = chamfer_dist(out_complete, in_complete)
            chamfer_error.update(dist.item()*10000)

            pred = out_complete.cpu().numpy()[0]
            gt = in_complete.cpu().numpy()[0]
            partial = in_partial.transpose(2,1).cpu().numpy()[0]


            # Save Point Cloud
            
            opt.outputFolder = os.path.join("Result_" + opt.model, category)
            if not os.path.exists(opt.outputFolder):
                    os.makedirs(opt.outputFolder)
            
            save_point_cloud(os.path.join(opt.outputFolder, name[0]+'_gt.xyz'), gt)
            save_point_cloud(os.path.join(opt.outputFolder, name[0]+'_partial.xyz'), partial)
            save_point_cloud(os.path.join(opt.outputFolder, name[0]+'_pred.xyz'), pred)

        gt_error.end_epoch() 
        pred_error.end_epoch()
        chamfer_error.end_epoch()
    
    
    
    R.append({'cat': category, 'chamfer': chamfer_error.avg, 'pred': pred_error.avg, 'gt':gt_error.avg})
    
    f = open(logpath + "_evaluation.csv",'w',newline='',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow([category,chamfer_error.avg])
    f.close()  

print('category:', end='\t')
print('Chamfer:', end='\t')
print('Pred->GT:', end='\t')
print('GT->Pred:', end='\t')
print()


for dc in R:
    print(dc['cat'], end='\t')    
    print(dc['chamfer'], end='\t')
    print(dc['pred'], end='\t')
    print(dc['gt'], end='\t')
    print()
