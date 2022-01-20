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
Call train_model.py in direstpose/tools  with args

arg1: output directory

arg2: datatype (usually the name of the animal)

arg3: alpha

arg4: beta

arg5: number of iterations

Make sure to define the keypoints for the animal you want to model in directpose/structures and to add the paths to your annotation files directpose/config/paths_catalog.py.
