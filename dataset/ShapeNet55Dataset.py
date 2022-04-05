import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("/home/heqifeng/PDP-NET")
from extensions.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import (
    ball_query, furthest_point_sample, gather_operation, grouping_operation,
    three_interpolate, three_nn)


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

ShapeNet55_dir = "/home/heqifeng/ShapeNet55/shapenet_pc"
split_file_dir = "/home/heqifeng/PoinTr/data/ShapeNet55-34/ShapeNet-55"

class ShapeNet55(data.Dataset):
    def __init__(self, root_dir = ShapeNet55_dir, npoints = 2048, class_choice = None, split = 'train'):
        self.data_root = root_dir
        self.subset = split
        self.npoints = npoints
        self.data_list_file = os.path.join(split_file_dir, f'{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            if taxonomy_id in class_choice:
                model_id = line.split('-')[1].split('.')[0]
                self.file_list.append({
                    'taxonomy_id': taxonomy_id,
                    'model_id': model_id,
                    'file_path': line
                })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def seprate_point_cloud(self, xyz, num_points, crop, fixed_points = None, padding_zeros = False):
        '''
        seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
        '''
        xyz = xyz.unsqueeze(0)
        _,n,c = xyz.shape

        assert n == num_points
        assert c == 3
        if crop == num_points:
            return xyz, None
            
        INPUT = []
        CROP = []
        for points in xyz:
            if isinstance(crop,list):
                num_crop = random.randint(crop[0],crop[1])
            else:
                num_crop = crop

            points = points.unsqueeze(0)

            if fixed_points is None:       
                center = F.normalize(torch.randn(1,1,3),p=2,dim=-1)
            else:
                if isinstance(fixed_points,list):
                    fixed_point = random.sample(fixed_points,1)[0]
                else:
                    fixed_point = fixed_points
                center = fixed_point.reshape(1,1,3)

            distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

            idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

            if padding_zeros:
                input_data = points.clone()
                input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

            else:
                input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

            crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

            if isinstance(crop,list):
                INPUT.append(fps_subsample(input_data,2048))
                CROP.append(fps_subsample(crop_data,2048))
            else:
                INPUT.append(input_data)
                CROP.append(crop_data)

        input_data = torch.cat(INPUT,dim=0)# B N 3
        crop_data = torch.cat(CROP,dim=0)# B M 3

        return input_data.squeeze(0).contiguous(), crop_data.squeeze(0).contiguous()

        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        in_complete = np.load(os.path.join(self.data_root, sample['file_path'])).astype(np.float32)
        in_complete = self.pc_norm(in_complete)
        in_complete = torch.from_numpy(in_complete).float()

        in_partial, in_hole = self.seprate_point_cloud(in_complete, in_complete.shape[0], self.npoints)

        return sample['taxonomy_id'], sample['model_id'], in_partial, in_hole, in_complete

    def __len__(self):
        return len(self.file_list)

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("CUDA activated")
    torch.cuda.set_device(device)
        
    category_choice = {"02691156": "airplane", "02747177": "trash bin", "02773838": "bag", "02801938": "basket", "02808440": "bathtub", "02818832": "bed", "02828884": "bench", "02843684": "birdhouse", "02871439": "bookshelf", "02876657": "bottle", "02880940": "bowl", "02924116": "bus", "02933112": "cabinet", "02942699": "camera", "02946921": "can", "02954340": "cap", "02958343": "car", "02992529": "cellphone", "03001627": "chair", "03046257": "clock", "03085013": "keyboard", "03207941": "dishwasher", "03211117": "display", "03261776": "earphone", "03325088": "faucet", "03337140": "file cabinet", "03467517": "guitar", "03513137": "helmet", "03593526": "jar", "03624134": "knife", "03636649": "lamp", "03642806": "laptop", "03691459": "loudspeaker", "03710193": "mailbox", "03759954": "microphone", "03761084": "microwaves", "03790512": "motorbike", "03797390": "mug", "03928116": "piano", "03938244": "pillow", "03948459": "pistol", "03991062": "flowerpot", "04004475": "printer", "04074963": "remote", "04090263": "rifle", "04099429": "rocket", "04225987": "skateboard", "04256520": "sofa", "04330267": "stove", "04379243": "table", "04401088": "telephone", "04460130": "tower", "04468005": "train", "04530566": "watercraft", "04554684": "washer"}
    
    for category in category_choice.keys():
        dataset_train = ShapeNet55(root_dir = ShapeNet55_dir, npoints = 2048, class_choice = category, split = 'train')
        dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0)

        print("Dataset Length: {}".format(len(dataset_train)))

        for i, data in enumerate(tqdm(dataloader_train, 0)):
            categor, name, in_partial, in_hole, in_complete = data

            '''
            in_partial = in_partial.contiguous().float().to(device)
            in_complete = in_complete.contiguous().float().to(device)

            dir1 = os.path.join("VisualData/", name[0] + "part.ply")
            dir2 = os.path.join("VisualData/", name[0] + "comp.ply")
            write_ply(in_partial.cpu().numpy()[0],dir1)
            write_ply(in_complete.cpu().numpy()[0],dir2)
            '''

