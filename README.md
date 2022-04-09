# PDP-Net: Patch-based Dual-Path Network for Point Cloud Completion

**This Repository is still under construction.**

Created by Qifeng He, Ning Xie, Zhenjiang Du, Ren Ke, Jiale Dou.

This is the official public repository of PyTorch implementation for PDP-Net: Patch-based Dual-Path Network for Point Cloud Completion (ICME 2022).
PDP-Net aims for the optimization of point cloud completion performance.

[Other materials will be provided later]

## Environment Setup

The codes have been tested under following settings.

### 2080Ti / TITAN
 - Python 3.7
 - Pytorch 1.4.0 / 1.5.0
 - CUDA 10.0 / 10.2

Use the below to install Pytorch 1.4.0 (CUDA 10.0).
```
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```
Use the below to install Pytorch 1.5.0 (CUDA 10.2).
```
pip install torch==1.5.0 torchvision==0.6.0
```
### 3090
 - Python 3.7
 - Pytorch 1.7.0
 - CUDA 11.0

 Use the below to install Pytorch.

```
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
For the possible failure of CUDA extension complilation, add the below to your .bashrc file.
```
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
``` 
Then
```
source .bashrc
```
Note that other Pytorch and CUDA versions might be supported as well, you can find the installation here:
[INSTALLING PREVIOUS VERSIONS OF PYTORCH](https://pytorch.org/get-started/previous-versions/)

### Other Dependency
```
pip install -r requirements.txt
```
#### CUDA Extensions
```
extensions
├── chamfer_dist
├── pointnet2_ops_lib
├── emd
├── expansion_penalty
```
For each directory, run the command below.
```python
python setup.py install
```
If your CUDA or torch versions have been changed, the built files should be deleted.
```python
python setup.py clean
```
#### Path Redirection
Make sure that your pathes is properly set before running any script. The pathes that you have to modify are shown below.

Modify the path in the following files:
1. train_PDP-Net.py
2. Evaluation_PDP-Net.py
3. ./dataset/ShapeNet55Dataset.py
4. ./dataset/PCNDataset.py
```
sys.path.append("//path//to//the//repo//")
```
In addition, also make sure that the dataset path is properly set in train_PDP-Net.py and Evaluation_PDP-Net.py.

## Dataset
### ShapeNet-Part Dataset
Go to ./data directory, and run the command.
```
sh get_dataset.sh
```
### PCN Dataset and ShapeNet-55 Dataset
Please refer to PoinTr repository for obtaining the datasets: [PoinTr Datasets](https://github.com/yuxumin/PoinTr/blob/master/DATASET.md)

### Dataset Loading
In ./dataset directory, I have provided codes for loading the above three datasets. Please modify the pathes to the data before use them.

## Trained Models
Coming soon.

## Training
```
python train_PDP-Net.py
```

## Evaluation
```
python Evaluation_PDP-Net.py
```

## Visualization

Please refer to my another repository: [3D-Point-Cloud-Rendering-with-Mitsuba](https://github.com/QifHE/3D-Point-Cloud-Rendering-with-Mitsuba)

## Acknowledgement

Some codes and CUDA extensions are based on following repositories. I thank them for their excellent work sincerely.

- [PoinTr](https://github.com/yuxumin/PoinTr)
- [ECG](https://github.com/paul007pl/ECG)
- [MSN](https://github.com/Colin97/MSN-Point-Cloud-Completion)
- [PF-Net](https://github.com/zztianzz/PF-Net-Point-Fractal-Network)
- [DGCNN](https://github.com/AnTao97/dgcnn.pytorch)
- [PCN](https://github.com/wentaoyuan/pcn)
- [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)