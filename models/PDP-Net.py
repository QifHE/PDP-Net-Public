import itertools
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.neighbors import KDTree

sys.path.append("/home/heqifeng/PDP-NET")
from extensions.emd import emd_module as emd
from extensions.chamfer_dist import ChamferDistance, ChamferDistanceSingle
from extensions.expansion_penalty import expansion_penalty_module as expansion
from extensions.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation


###### Sample and merge functions ######

def fps_subsample(pcd, n_points = 2048):
    """
    FPS used in PointNet++ (CUDA extension needed)
    Args
        pcd: (b, N, 3)
    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd

def fps_subsample_with_oneHot(pcd, n_points = 2048):
    """
    FPS used in PointNet++ with one extented channel for one-hot labeling (CUDA extension needed)
    Args
        pcd: (b, N, 4)
    returns
        new_pcd: (b, n_points, 4)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd[:,:,0:3].contiguous(), n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd

def merge_pointClouds_with_oneHot(pc1, pc2):
    """
    Merge two pcds with one-hot labeling
    Args
        pcd: (b, C, N1), (b, C, N2)
    return
        pcd:(b, C + 1, N1 + N2)
    """

    id1 = torch.zeros(pc1.shape[0], 1, pc1.shape[2]).cuda().contiguous()
    pc1 = torch.cat( (pc1, id1), 1)
    
    id2 = torch.ones(pc2.shape[0], 1, pc2.shape[2]).cuda().contiguous()
    pc2 = torch.cat( (pc2, id2), 1)

    MPC = torch.cat([pc1, pc2], 2).contiguous()
    return MPC


###### KNN and EdgeConv imlementations ######

def knn(x, k):
    """
    K-Nearest Neighbors algorithm implementation
    Args
        x: (b, C, N)
        k: int
    return
        idx:(b, N, k)
    """

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k = 20, minus_center = True):
    """
    EdgeConv operation from DGCNN (CUDA needed)
    Args
        x: (b, C, N)
        k: int
        minus_center: bool
    return
        feature: (b, 2 * C, N, k)
    """

    idx = knn(x, k = k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda:0')

    idx_base = torch.arange(0, batch_size, device = device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if minus_center:
        feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)
    else:
        feature = torch.cat((x, feature), dim=3).permute(0, 3, 1, 2)
    return feature


###### PDP-Net unique modules ######

def pc_density_thresh(pc):
    """
    Compute pcd density defined by average euclidean distance between the nearest two points (sklearn package needed)
    Args
        pc: (b, N, 3)
    returns
        thresh: float
    """

    batch, npoints, _ = pc.shape

    thresh = np.zeros((batch, 1))
    for i in range(batch):
        pc_numpy = pc[i].detach().cpu().numpy()
        tree = KDTree(pc_numpy, metric='euclidean')
        distance, _ = tree.query(pc_numpy, 2)
        mean = np.mean(distance[:,1])
        std = np.std(distance[:,1], ddof=1)
        thresh[i] = mean + 3 * std
    thresh = torch.from_numpy(thresh)

    return thresh

def identifier(in_partial, pred_hole, device):
    """
    Seperate outlier points regarding to the partial pcd, and re-merge/resample (CUDA extension needed)
    Args
        in_partial: (b, N, 3)
        pred_hole: (b, N', 3)
    returns
        pc_merge_batch: (b, N, 3)
    """

    batch, npoints, _ = in_partial.shape
    chamfer_single = ChamferDistanceSingle()
    in_partial = in_partial.to(device)
    _, z = chamfer_single(in_partial, pred_hole)
    mask = torch.ones_like(z, dtype=torch.bool)
    thresh = pc_density_thresh(in_partial).to(device)
    
    thresh = thresh.repeat(1,pred_hole.shape[1])
    mask = torch.ge(z, thresh)
    
    pc_merge_batch = torch.zeros(batch, npoints, 3).cuda()
    for j in range(batch):
        pc_extract = pred_hole[j][mask[j]]
        pc_merge = torch.cat((in_partial[j], pc_extract), 0).to(device)
        pc_merge = fps_subsample(pc_merge.unsqueeze(0), npoints)   
        pc_merge_batch[j] = pc_merge.squeeze(0)
    
    return pc_merge_batch


###### Encoders ######

class PCN_Encoder(nn.Module):
    """
    Modification based on PCN (3DV 2018)
    GitHub: https://github.com/wentaoyuan/pcn

    forward
        x: (B, C, N)
    returns
        v: (B, 1024)
    """
    def __init__(self):
        super(PCN_Encoder, self).__init__()
   
        # First MLP
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        # Second MLP
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)

    
    def forward(self, x):
        n = x.size()[2]

        # First MLP
        x = F.relu(self.bn1(self.conv1(x)))           
        f = self.bn2(self.conv2(x))                  
        
        # Point-wise maxpool
        g = torch.max(f, dim=2, keepdim=True)[0]     
        
        # Expand and concat
        x = torch.cat([g.repeat(1, 1, n), f], dim=1)  

        # Second MLP
        x = F.relu(self.bn3(self.conv3(x)))           
        x = self.bn4(self.conv4(x))      
        
        # Point-wise maxpool
        v = torch.max(x, dim=-1)[0]                  
      
        return v

class DGCNN_Encoder(nn.Module):
    """
    Modification based on DGCNN (TOG 2019) Pytorch implementation
    GitHub: https://github.com/AnTao97/dgcnn.pytorch

    forward
        x: (B, C, N)
    returns
        v: (B, 1024)
    """
    def __init__(self):
        super(DGCNN_Encoder, self).__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):

        batch_size = x.size()[0]
        x = get_graph_feature(x, k = 20)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k = 20)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k = 20)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k = 20)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        feature = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        return feature


###### Decoders ######

class FC_Decoder(nn.Module):
    """
    Fully-Connected Layers

    forward
        x: (B, C * N)
    returns
        v: (B, 1024, 3)

    """
    def __init__(self):
        super(FC_Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 3 * 1024)

        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, 3, 1024)
        x = x.transpose(1,2).contiguous()

        return x
    
class FC_Decoder_2(nn.Module):
    """
    Fully-Connected Layers

    forward
        x: (B, C * N)
    returns
        v: (B, 1024, 3)
    """
    def __init__(self):
        super(FC_Decoder_2, self).__init__()
        self.fc1 = torch.nn.Linear(1027, 1027)
        self.fc2 = torch.nn.Linear(1027, 1024)
        self.fc3 = torch.nn.Linear(1024, 3 * 1024)

        self.bn1 = torch.nn.BatchNorm1d(1027)
        self.bn2 = torch.nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, 3, 1024)  # (B, 3, 1024)
        x = x.transpose(1,2).contiguous()

        return x

class EF_expansion(nn.Module):
    """
    Modification based on ECG
    GitHub: https://github.com/paul007pl/ECG

    Args
        input_size: C
        output_size: int
        step_ratio = 1
        k: int
    forward
        x: (B, C, N)
    returns
        v: (B, output_size, N)
    """

    def __init__(self, input_size = 3, output_size = 64, step_ratio = 1, k = 4):
        super(EF_expansion, self).__init__()
        self.step_ratio = step_ratio
        self.k = k
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(input_size * 2, output_size, 1)
        self.conv2 = nn.Conv2d(input_size * 2 + output_size, output_size * step_ratio, 1)
        self.conv3 = nn.Conv2d(output_size, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()

        input_edge_feature = get_graph_feature(x, self.k, minus_center=False).permute(0, 1, 3, 2).contiguous()
        edge_feature = self.conv1(input_edge_feature)
        edge_feature = F.relu(torch.cat((edge_feature, input_edge_feature), 1))

        edge_feature = F.relu(self.conv2(edge_feature)) 
        edge_feature = edge_feature.permute(0, 2, 3, 1).contiguous().view(batch_size, self.k, num_points * self.step_ratio, self.output_size).permute(0, 3, 1, 2)

        edge_feature = self.conv3(edge_feature)
        edge_feature, _ = torch.max(edge_feature, 2)

        return edge_feature

class Folding_Decoder(nn.Module):
    """
    Folding operations for independent patches

    Args
        Initialized by Patch_Generator()

    forward
        x: (B, 1024 + 3)
        sk_point: (B, 3)
    returns
        fine: (B, 1024)
    """
    
    def __init__(self, num_points = 1024, bottleneck_size = 1024, n_primitives = 16):
        super(Folding_Decoder, self).__init__()
        self.grid_size = 1
        self.grid_scale = 0.05
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.device = "cuda:0"
        self.patchpoint = self.num_points // self.n_primitives
        self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
                         [-self.grid_scale, self.grid_scale, self.grid_size]]

        self.folding1 = nn.Sequential(
            nn.Conv1d(bottleneck_size + 3, bottleneck_size // 2, 1),
            nn.BatchNorm1d(bottleneck_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck_size // 2, bottleneck_size // 4, 1),
            nn.BatchNorm1d(bottleneck_size//4),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck_size // 4, 3, 1))

        self.folding2 = nn.Sequential(
            nn.Conv1d(bottleneck_size + 3 + 2 + 3, bottleneck_size // 2, 1),
            nn.BatchNorm1d(bottleneck_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck_size // 2, bottleneck_size // 4, 1),
            nn.BatchNorm1d(bottleneck_size // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck_size // 4, 3, 1))

        self.fclayers = FC_Decoder_2()
        self.expand = EF_expansion(input_size = 3, output_size = 64, step_ratio = 1, k = 4)
        
    def build_grid(self, batch_size):
        x, y = np.linspace(*self.meshgrid[0]), np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)

        return torch.tensor(points).float().to(self.device)

    def tile(self, tensor, multiples):
        def tile_single_axis(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*repeat_idx)
            order_index = torch.Tensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).long()
            return torch.index_select(a, dim, order_index.to(self.device))

        for dim, n_tile in enumerate(multiples):
            if n_tile == 1:
                continue
            tensor = tile_single_axis(tensor, dim, n_tile)
        return tensor

    @staticmethod
    def expand_dims(tensor, dim):
        return tensor.unsqueeze(-1).transpose(-1, dim)
    
    def forward(self, x, sk_point):

        x_coarse = self.fclayers(x)
        x_expanded = self.expand(x_coarse.transpose(2,1).contiguous())
        
        sk_point = sk_point.unsqueeze(1).repeat(1, self.patchpoint, 1)
        x = torch.cat((sk_point, x_expanded), 2)

        coarse = self.folding1(x.transpose(2,1).contiguous())
        coarse = coarse.transpose(2,1)

        grid = self.build_grid(x.size()[0])
        grid_feat = grid.repeat(1, self.patchpoint, 1) 

        feat = torch.cat([grid_feat, coarse, x], dim=2)

        fine = self.folding2(feat.transpose(2, 1)).transpose(2, 1) + coarse

        return fine

class Patch_Generator(nn.Module):
    """
    Patch_Generator Module

    Args
        num_points: int, total point number
        bottleneck_size: int
        n_primitives: int, total patch number
    forward
        x: (B, 1024)
    returns
        outs: (B, 1024, 3)
        loss_mst: float
    """
    def __init__(self, num_points = 1024, bottleneck_size = 1024, n_primitives = 16):
        super(Patch_Generator, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.decoder = nn.ModuleList([Folding_Decoder(self.num_points, self.bottleneck_size, self.n_primitives) for i in range(0,self.n_primitives)])
        self.expansion = expansion.expansionPenaltyModule()
    
    def forward(self, x, skeleton):
        outs = []

        for i in range(0,self.n_primitives):
            sk_point = skeleton[:,i,:].contiguous()
            y = torch.cat((sk_point, x), 1).contiguous()
            outs.append(self.decoder[i](y, sk_point))

        outs = torch.cat(outs,1).contiguous() 
    
        dist, _, mean_mst_dis = self.expansion(outs, self.num_points // self.n_primitives, 1.5)
        loss_mst = torch.mean(dist)
        
        return outs, loss_mst

###### FULL MODEL ######

class PDP_Net_Model(nn.Module):
    """
    PDP-Net Model

    Args
        Refer to Patch_Generator()
    forward
        x: input partial pcd
        in_hole: GT for the missing region
        in_complete: GT complete pcd
    returns
        out_complete: predicted complete pcd
        loss_up: EMD loss 1
        loss_down: EMD loss 2
        loss_expansion: Expansion penalty
    """
    def __init__(self, device):
        super(PDP_Net_Model, self).__init__()
        self.encoder1 = PCN_Encoder()
        self.encoder2 = DGCNN_Encoder()
        self.decoder1 = FC_Decoder()
        self.n_primitives = 16
        self.decoder2 = Patch_Generator(num_points = 1024, bottleneck_size = 1024, n_primitives = self.n_primitives)
        self.loss1 = emd.emdModule()
        self.loss2 = emd.emdModule()
        self.device = device
        
    def forward(self, x, in_hole, in_complete):
        # up encoding
        y1 = self.encoder1(x)

        # down encoding   
        y2 = self.encoder2(x)   

        # up
        coarse_up_hole = self.decoder1(y1)
        
        # skeleton for down
        skeleton = fps_subsample(coarse_up_hole, self.n_primitives)
        skeleton = skeleton.contiguous()

        # down
        coarse_down_hole, loss_expansion = self.decoder2(y2, skeleton)
        coarse_down = torch.cat((x, coarse_down_hole.transpose(2,1).contiguous()), 2)
        coarse_down_refine = fps_subsample(coarse_down.transpose(2,1).contiguous(), in_complete.shape[2])
        # identifier module
        out_complete = identifier(x.transpose(2,1).contiguous(), coarse_down_refine, self.device)
        
        # EMD loss
        in_hole = in_hole.transpose(2,1).contiguous()
        in_complete = in_complete.transpose(2,1).contiguous()

        dist, _ = self.loss1(coarse_up_hole, in_hole, 0.005, 50)
        emd1 = torch.sqrt(dist).mean(1)
        loss_up = emd1.mean()

        dist, _ = self.loss2(coarse_down_refine, in_complete, 0.005, 50)
        emd2 = torch.sqrt(dist).mean(1)
        loss_down = emd2.mean()

        return out_complete, loss_up, loss_down, loss_expansion


###### For Testing and Debugging ######

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("CUDA activated")
    torch.cuda.set_device(device)

    model = PDP_Net_Model(device).cuda()

    print(model)

    batch = 2
    input_pc = torch.rand(batch, 3, 2048).cuda()
    input_pc2 = torch.rand(batch, 3, 1024).cuda()
    input_pc3 = torch.rand(batch, 3, 2048).cuda()

    out_complete, loss_up, loss_down, loss_expansion = model(input_pc, input_pc2, input_pc3)

    print(out_complete.shape)

