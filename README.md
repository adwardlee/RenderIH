# RenderIH
Official PyTorch implementation of "RenderIH: A large-scale synthetic dataset for 3D interacting hand pose estimation", ICCV 2023

## Prequeries
download and unzip [misc.tar].
Register and download [MANO](https://mano.is.tue.mpg.de/)  data. Put `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` in `misc/mano`
After collecting the above necessary files, the directory structure of `./misc` is expected as follows:

```
./misc
├── mano
│   └── MANO_LEFT.pkl
│   └── MANO_RIGHT.pkl
├── model
│   └── config.yaml
│   └── pretrain.pth
├── graph_left.pkl
├── graph_right.pkl
├── upsample.pkl
├── v_color.pkl

```

## Requirements
- Tested with python3.8.8 on Ubuntu 18.04, CUDA 11.3.

torch1.12.1: `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`

pytorch3d: `pip install fvcore iopath; pip install git+https://github.com/facebookresearch/pytorch3d.git@stable`

opencv4.7:`pip install opencv_python==4.7.0.72`

[manopth](https://github.com/hassony2/manopth) `pip install git+https://github.com/hassony2/chumpy.git`,`pip install git+https://github.com/hassony2/manopth.git`

"[sdf](https://github.com/JiangWenPL/multiperson/tree/master/sdf)" change **AT_CHECK** in `multiperson/sdf/csrc/sdf_cuda.cpp` to **TORCH_CHECK** 

mmcv:`pip install -U openmim`,`mim install mmcv`
numpy,tqdm,yacs,tensorboardX,scipy,imageio,matplotlib,scikit-image,manopth,timm,imgaug,fvcore,iopath


## DATASET
1.INTERHAND2.6M: 1) Download [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) dataset and unzip it. (Noted: we used the v1.0_5fps version and H+M subset for training and evaluating)

Process the dataset by :

```python utils/dataset_gen/interhand.py --data_path PATH_OF_INTERHAND2.6M --save_path ./interhand2.6m/ --gen_anno 1```
```python utils/dataset_gen/interhand.py --data_path PATH_OF_INTERHAND2.6M --save_path ./interhand2.6m/ --gen_anno 0```
Replace PATH_OF_INTERHAND2.6M with your own store path of InterHand2.6M dataset.

2.[RenderIH](./rendering_code): Download from [imgs](https://drive.google.com/file/d/1nl5VZvnKN3SIJnBOis4rfsuG_DT0smLl/view?usp=drive_link), [annotations](https://drive.google.com/file/d/1wOuZTgWODhyelLXJr7Kv9tuEiFxcWIif/view?usp=drive_link). Untar the compressed files, and run step7.

## Training
`python apps/train.py --gpu 0,1,2,3`
change `INTERHAND_PATH` in `utils/default.yaml` to the dataset path

## Evaluation
`python apps/eval_interhand.py --model MODEL_PATH --data_path INTERHAND2.6M_PATH`
change `MODEL_PATH` to the pretrained model path, and `INTERHAND2.6M_PATH` to dataset path.

data_type=0, dataset/interhand.py syn=True, 使用renderih与real一起训练

data_type=1, loader_ori 使用synthetic+real

data_type=2, loader.py 使用interhand_withother.py, 训练ego3dhand , h2o3d，或者renderih

data_type=3, loader.py, 使用interhand_orisyn.py ，使用pose和interhand相同合成数据

data_type=4, loader.py, 使用interhand_subset.py ，使用poseaug, subset合成数据与full real interhand data