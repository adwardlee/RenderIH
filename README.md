# RenderIH
Official PyTorch implementation of "RenderIH: A large-scale synthetic dataset for 3D interacting hand pose estimation", ICCV 2023

## Our dataset
[RenderIH](./rendering_code): Download from Google Drive: [imgs](https://drive.google.com/file/d/1nl5VZvnKN3SIJnBOis4rfsuG_DT0smLl/view?usp=drive_link), [annotations](https://drive.google.com/file/d/1wOuZTgWODhyelLXJr7Kv9tuEiFxcWIif/view?usp=drive_link), [materials](https://drive.google.com/file/d/1NQJvLTuY2hKYfhMBqG-OADrosDGMuPzr/view?usp=drive_link); or BaiduPan: [imgs](https://pan.baidu.com/s/1M0vxWRbBu1lH_fV9FPBHbg?pwd=mo5n) [annotations](https://pan.baidu.com/s/1XFIbU_QHT1Smi2WL_LmCJw?pwd=ajbf). Untar the compressed files of **imgs** and **annotations**, then run [step7](https://github.com/adwardlee/RenderIH/blob/main/rendering_code/step7_gen_annotations.py) in rendering_code. **Materials** is used for generation process in previous steps in rendering.

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
numpy,tqdm,yacs==0.1.8,tensorboardX,scipy,imageio,matplotlib,scikit-image,manopth,timm,imgaug,fvcore,iopath


## DATASET
### INTERHAND2.6M
1) Download [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) dataset and unzip it. (Noted: we used the v1.0_5fps version and H+M subset for training and evaluating)

2) Process the dataset by :

```python utils/dataset_gen/interhand.py --data_path PATH_OF_INTERHAND2.6M --save_path ./interhand2.6m/ --gen_anno 1```
```python utils/dataset_gen/interhand.py --data_path ./interhand2.6m/ --gen_anno 0```
Replace PATH_OF_INTERHAND2.6M with your own store path of InterHand2.6M dataset.

### Tzionas Dataset
1) Download Hand-Hand Interaction from the [website](https://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/), categories from Walking to Hugging (01.zip~07.zip). Moreover, download the mano annotations from [MANO_fits](http://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/Dataset/MANO_compatible/IJCV16___Results_MANO___parms_for___joints21.zip).

2) Process the dataset by:
```python utils/dataset_gen/tzionas_generation.py --mano_annot xxx --detection_path xxx --rgb_path xxx --output_path xxx```

## Pretrained model
[model without syntheic data](https://drive.google.com/file/d/192abd-pdyHl89Td0or7fll38KCsBC7bv/view?usp=drive_link)
[model with synthetic data](https://drive.google.com/file/d/13zsI-8PQn2UFqOjwObZrw9KHIpdFnHrg/view?usp=drive_link)

## Training
`python apps/train.py --gpu 0,1,2,3`
change `INTERHAND_PATH` in `utils/default.yaml` to the dataset path

`utils/default.yaml` has some argments that can be tuned

## Evaluation

### INTERHAND2.6M
`python apps/eval_interhand.py --model MODEL_PATH --data_path INTERHAND2.6M_PATH`
change `MODEL_PATH` to the pretrained model path, and `INTERHAND2.6M_PATH` to dataset path.


## Miscellaneous
data_type=0, dataset/interhand.py syn=True, use renderih together with Interhand2.6M

data_type=1, loader_ori using synthetic+real

data_type=2, loader.py using interhand_withother.py, training ego3dhand , h2o3d，or renderih

data_type=3, loader.py, using interhand_orisyn.py ，using the synthetic data

data_type=4, loader.py, using interhand_subset.py ，poseaug, subset synthetic and full real interhand data

`utils/compute_maskiou.py`. Calculate the iou distribution for hand data.
## Citation

```bibtex
@article{li2023renderih,
  title={RenderIH: A Large-scale Synthetic Dataset for 3D Interacting Hand Pose Estimation},
  author={Li, Lijun and Tian, Linrui and Zhang, Xindi and Wang, Qi and Zhang, Bang and Liu, Mengyuan and Chen, Chen},
  journal={arXiv preprint arXiv:2309.09301},
  year={2023}
}
```
