# SemiMultiPose

## Installation Instructions

conda create --name directpose -y

conda activate directpose

conda install ipython pip

conda install python=3.6.10

conda install -c pytorch pytorch-nightly

conda install cudatoolkit==10.0.130

pip install torchvision==0.2.2

pip install ninja yacs cython matplotlib tqdm opencv-python numpy

conda install -c psi4 gcc-5

export INSTALL_DIR=$PWD

cd $INSTALL_DIR

cd cocoapi/PythonAPI

python setup.py build_ext install

cd $INSTALL_DIR

cd cityscapesScripts/

python setup.py build_ext install

cd $INSTALL_DIR

cd apex

python setup.py install --cuda_ext --cpp_ext

cd $INSTALL_DIR

python setup.py build develop


## Instructions to train model
Call ./train_model.sh in direstpose/tools  with args

-o: output directory

-d: datatype (usually the name of the animal)

-a: alpha (optional, default is 0.01)

-b: beta (optional, default is 0.1)

-n: number of iterations (should be a multiple of 1k; optional, default is 40000)

## Instructions to add new datatype

Define the keypoints for the animal you want to model in directpose/structures

Add the paths to your annotation files directpose/config/paths_catalog.py.

Add new datatype info to directpose/modeling/heatmap.py and directpose/modeling/rpn/fcos/fcos.py
