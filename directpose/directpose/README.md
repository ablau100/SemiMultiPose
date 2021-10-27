# DirectPose: Direct End-to-End Multi-Person Pose Estimation

The full paper is available at: [https://arxiv.org/abs/1911.07451](https://arxiv.org/abs/1911.07451). 

## Installation

This DirectPose implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Therefore the installation is the same as original maskrcnn-benchmark.

Please check [INSTALL.md](INSTALL.md) for installation instructions.
You may also want to see the original [README.md](MASKRCNN_README.md) of maskrcnn-benchmark.

## Inference
The inference command line on coco minival split:

    NCCL_P2P_DISABLE=1 python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/test_net.py \
        --config-file configs/fcos/fcos_kps_ms_training_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/fcos_kps_ms_training_R_50_FPN_1x \
        MODEL.WEIGHT fcos_kps_ms_training_R_50_FPN_1x.pth
        
1) Our trained model `fcos_kps_ms_training_R_50_FPN_1x.pth` can be downloaded [here](https://cloudstor.aarnet.edu.au/plus/s/ZRtK4mNmroHN5hs/download).

## Training

The following command line will train `fcos_kps_ms_training_R_50_FPN_1x` on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    NCCL_P2P_DISABLE=1 python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/fcos/fcos_kps_ms_training_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/fcos_kps_ms_training_R_50_FPN_1x \
        SOLVER.KPS_GRAD_MULT 10.0 \
        SOLVER.MAX_GRAD_NORM 5.0 \
        SOLVER.POWER 1.0 \
        INPUT.CROP_SIZE 800 \
        INPUT.MIN_SIZE_RANGE_TRAIN "(480, 1600)" \
        INPUT.MAX_SIZE_TRAIN 2666 \
        MODEL.WEIGHT ../../.torch/models/resnet50_lpf3.pth \
        SOLVER.IMS_PER_BATCH 32 \
        SOLVER.MAX_ITER 180000 \
        MODEL.HEATMAPS_LOSS_WEIGHT 4.0
        
Note that:
1) If you want to use fewer GPUs, please reduce `--nproc_per_node`. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size, please change `SOLVER.IMS_PER_BATCH`.
2) The models will be saved into `OUTPUT_DIR`.
3) Please download the ImageNet pretrained model `resnet50_lpf3.pth` from [https://github.com/adobe/antialiased-cnns](https://github.com/adobe/antialiased-cnns).
4) Sometimes you may encounter a deadlock with 100% GPUs' usage, which might be a problem of NCCL. Please try `export NCCL_P2P_DISABLE=1` before running the training command line.

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 

