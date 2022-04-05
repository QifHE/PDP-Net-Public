# PDP-Net: Patch-based Dual-Path Network for Point Cloud Completion

## Environment Setup

The codes have been tested under following settings.

### 2080Ti / TITAN
 - Python 3.7
 - Pytorch 1.4.0
 - CUDA 10.0

Use the below to install Pytorch.
```
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3090Ti
 - Python 3.8
 - Pytorch 1.7.0
 - CUDA 11.0

 Use the below to install Pytorch.

```
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
### Other Dependency
```
pip install -r requirements.txt
```
### CUDA Extensions
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

## Dataset
Go to ./data directory, and run the command.
```
sh get_dataset.sh
```

## Training
Modify the path in train_PDP-Net.py
```
sys.path.append("//path//to//the//repo//")
```

Run the training file.

```
python train_PDP-Net.py
```

## Evaluation
Modify the path in Evaluation_PDP-Net.py
```
sys.path.append("//path//to//the//repo//")
```

Run the evaluation file.

```
python Evaluation_PDP-Net.py
```

## Visualization

Please refer to my aother repository:
https://github.com/QifHE/3D-Point-Cloud-Rendering-with-Mitsuba