Manopth
=======

**Deprecation notice**  
This package is deprecated and kept for archival purposes only.
New version has been moved to [maontorch](https://github.com/lixiny/manotorch).


**change log**
* Additionally return transf_global at the 3rd position of ManoLayer
* Add verts, joints, axes in `pyrender`.
* Return joints, verts in `meter` instead of `mm`



[MANO](http://mano.is.tue.mpg.de) layer for [PyTorch](https://pytorch.org/) (tested with v0.4 and v1.x)

ManoLayer is a differentiable PyTorch layer that deterministically maps from pose and shape parameters to hand joints and vertices.
It can be integrated into any architecture as a differentiable layer to predict hand meshes.

![image](assets/mano_layer.png)

ManoLayer takes **batched** hand pose and shape vectors and outputs corresponding hand joints and vertices.

The code is mostly a PyTorch port of the original [MANO](http://mano.is.tue.mpg.de) model from [chumpy](https://github.com/mattloper/chumpy) to [PyTorch](https://pytorch.org/).
It therefore builds directly upon the work of Javier Romero, Dimitrios Tzionas and Michael J. Black.

This layer was developped and used for the paper *Learning joint reconstruction of hands and manipulated objects* for CVPR19.
See [project page](https://github.com/hassony2/obman) and [demo+training code](https://github.com/hassony2/obman_train).


It [reuses](https://github.com/hassony2/manopth/blob/master/manopth/rodrigues_layer.py) [part of the great code](https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py) from the  [Pytorch layer for the SMPL body model](https://github.com/MandyMo/pytorch_HMR/blob/master/README.md) by Zhang Xiong ([MandyMo](https://github.com/MandyMo)) to compute the rotation utilities !

It also includes in `mano/webuser` partial content of files from the original [MANO](http://mano.is.tue.mpg.de) code ([posemapper.py](mano/webuser/posemapper.py), [serialization.py](mano/webuser/serialization.py), [lbs.py](mano/webuser/lbs.py), [verts.py](mano/webuser/verts.py), [smpl_handpca_wrapper_HAND_only.py](mano/webuser/smpl_handpca_wrapper_HAND_only.py)).

If you find this code useful for your research, consider citing:

- the original [MANO](http://mano.is.tue.mpg.de) publication:

```
@article{MANO:SIGGRAPHASIA:2017,
  title = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
  author = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
  publisher = {ACM},
  month = nov,
  year = {2017},
  url = {http://doi.acm.org/10.1145/3130800.3130883},
  month_numeric = {11}
}
```

- the publication this PyTorch port was developped for:

```
@INPROCEEDINGS{hasson19_obman,
  title     = {Learning joint reconstruction of hands and manipulated objects},
  author    = {Hasson, Yana and Varol, G{\"u}l and Tzionas, Dimitris and Kalevatykh, Igor and Black, Michael J. and Laptev, Ivan and Schmid, Cordelia},
  booktitle = {CVPR},
  year      = {2019}
}
```

The training code associated with this paper, compatible with manopth can be found [here](https://github.com/hassony2/obman_train). The release includes a model trained on a variety of hand datasets.

# Installation

## Get code and dependencies

- `git clone https://github.com/lixiny/manopth`
- `cd manopth`
- Install the dependencies listed in [environment.yml](environment.yml)
  - In an existing conda environment, `conda env update -f environment.yml`
  - In a new environment, `conda env create -f environment.yml`, will create a conda environment named `manopth`

## Download MANO pickle data-structures

- Go to [MANO website](http://mano.is.tue.mpg.de/)
- Create an account by clicking *Sign Up* and provide your information
- Download Models and Code (the downloaded file should have the format `mano_v*_*.zip`). Note that all code and data from this download falls under the [MANO license](http://mano.is.tue.mpg.de/license).
- unzip and copy the `models` folder into the `manopth/mano` folder
- Your folder structure should look like this:
```
manopth/
  mano/
    models/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
      ...
  manopth/
    __init__.py
    ...
```

To check that everything is going well, run `python examples/manopth_mindemo.py`, which should generate from a random hand using the MANO layer !

## Install `manopth` package

To be able to import and use `ManoLayer` in another project, go to your `manopth` folder and run `pip install .`


`cd /path/to/other/project`

You can now use `from manopth import ManoLayer` in this other project!

# Usage

## Minimal usage script

See [examples/manopth_mindemo.py](examples/manopth_mindemo.py)

Simple forward pass with random pose and shape parameters through MANO layer

```python
import torch
from manopth.manolayer import ManoLayer
from manopth import demo

batch_size = 10
# Select number of principal components for pose space
ncomps = 6

# Initialize MANO layer
mano_layer = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=ncomps)

# Generate random shape parameters
random_shape = torch.rand(batch_size, 10)
# Generate random pose parameters, including 3 values for global axis-angle rotation
random_pose = torch.rand(batch_size, ncomps + 3)

# Forward pass through MANO layer
hand_verts, hand_joints = mano_layer(random_pose, random_shape)
demo.display_hand({'verts': hand_verts, 'joints': hand_joints}, mano_faces=mano_layer.th_faces)
```

Visualize :

<img src="assets/random_hand.png" height="240"> <img src="assets/render_hand.png" height="240">


## Demo
You can run it locally with:

`python examples/app.py`

