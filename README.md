# Multi-fingered bimanual grasp dataset: Toward Dexterity-Aware Dual-Arm Grasping

This project is inspired on https://sites.google.com/view/da2dataset and https://github.com/mkiatos/geometric-object-grasper.git.

The current generated dataset is in the folder final_data_set. To visualize the generated grasps, please follow the installation instructions.

## Installation
### Basic installation
```
conda create -n DA2 python=3.8
conda activate DA2
git clone https://github.com/ymxlzgy/DA2.git
cd path/to/DA2
mkdir grasp test_tmp
pip install -r requirements.txt
```
### Meshpy installation
```
cd path/to/DA2/meshpy
python setup.py develop
```
### Pytorch installation
Please refer to [pytorch](https://pytorch.org/) official website to find the best version in your case, e.g.,:
```
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
### Mayavi installation
```
conda install mayavi -c conda-forge
```

### Geometric object grasper installation
```
cd geometric_object_grasper/
pip install -e .
```

### Bimanual grasp visualization
To visualize the bimanual grasps, please run the grasp_visualization.py, considering:

Object name: long string that correspond to the mesh name, but without the .obj
Grasp id: See the file contacts_finger, which contain the indexes of the grasps for each object


