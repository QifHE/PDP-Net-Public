import os
import sys

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("/home/heqifeng/PDP-NET")
from extensions.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import (
    ball_query, furthest_point_sample, gather_operation, grouping_operation,
    three_interpolate, three_nn)
from utils.pcutils import make_holes_pcd_2, normalize, write_ply


def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)
    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

def rotate_pcd_shapeNet(pcd, posA = 1, posB = 2):
    n = pcd.shape[0]
    for i in range(n):
        temp = pcd[i][posA]
        pcd[i][posA] = pcd[i][posB]
        pcd[i][posB] = temp
    return pcd

def walk_file(file):
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            print(os.path.join(root, f))

        # 遍历所有的文件夹
        for d in dirs:
            print(os.path.join(root, d))

PCN_dir = "/home/heqifeng/ShapeNetCompletion"
category_file = "/home/heqifeng/ShapeNetCompletion/synsetoffset2category.txt"

class PCNDataset(data.Dataset):
    def __init__(self, root_dir = PCN_dir, npoints = 2048, category_choice = None, split = 'train'):
        self.npoints = npoints      # 原始点云点数量
        self.root = root_dir       # 数据集最上级文件夹
        self.classification = False
        self.normalize = False
        self.category = category_choice   # 数据集分类

        self.split = split        # 训练还是测试

        self.file_list_partial = os.path.join(root_dir, split, "partial/")
        self.file_list_gt = os.path.join(root_dir, split, "complete/")

        self.category_file = category_file
        self.category_id = {} # 储存选择的类别id
        
        # 选择类别下的数据
        with open(self.category_file, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.category_id[ls[1]] = ls[0]
        if not category_choice is None:
            self.category_id = {v: k for k, v in self.category_id.items() if k in category_choice}
        
        print(self.category_id)
        
        self.pc_list_partial = []

        for cat in self.category_id:
            self.category_file_list_partial = self.file_list_partial + cat
            
            for root, dirs, files in os.walk(self.category_file_list_partial):
                dirs.sort()
                files.sort()
                for f in files:
                    self.pc_list_partial.append([os.path.join(root, f)])        
    
    def __getitem__(self, index):

        pc_partial = self.pc_list_partial[index][0]
        pc_gt = pc_partial[:-7].replace("partial","complete") + ".pcd"

        pcd_partial = o3d.io.read_point_cloud(pc_partial)
        pcd_gt = o3d.io.read_point_cloud(pc_gt)

        point_set_partial = np.asarray(pcd_partial.points, dtype = np.float32)
        point_set_gt = np.asarray(pcd_gt.points, dtype = np.float32)
        
        point_set_partial = resample_pcd(point_set_partial, self.npoints)
        #point_set_gt = resample_pcd(point_set_gt, self.npoints)

        filename = pc_partial

        if self.normalize:
            point_set_partial = normalize(point_set_partial, unit_ball = True)
            point_set_gt = normalize(point_set_gt, unit_ball = True)
        
        return filename, point_set_partial, point_set_gt
            
    def __len__(self):
        return len(self.pc_list_partial)

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("CUDA activated")
    torch.cuda.set_device(device)
        

    dataset_dir = "/home/heqifeng/ShapeNetCompletion"
    category_choice = ["airplane", "cabinet", "car", "chair", "lamp", "sofa", "table", "watercraft"]
    
    for category in category_choice:
        dataset_train = PCNDataset(root_dir=dataset_dir, category_choice=category, npoints=2048, split='val')
        dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=64)

        print("Dataset Length: {}".format(len(dataset_train)))

        for i, data in enumerate(tqdm(dataloader_train, 0)):
            filename, in_partial, in_complete = data

            in_partial = in_partial.contiguous().float().to(device)
            in_complete = in_complete.contiguous().float().to(device)

            print("partial shape {}".format(in_partial.shape))
            print("complete shape {}".format(in_complete.shape))
