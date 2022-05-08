# AB3DMOT

This part has AB3MOT. 


<img align="center" width="100%" src="https://github.com/xinshuoweng/AB3DMOT/blob/master/main1.gif">
<img align="center" width="100%" src="https://github.com/xinshuoweng/AB3DMOT/blob/master/main2.gif">

## Introduction

3D multi-object tracking (MOT) is an essential component technology for many real-time applications such as autonomous driving or assistive robotics. However, recent works for 3D MOT tend to focus more on developing accurate systems giving less regard to computational cost and system complexity. In contrast, this work proposes a simple yet accurate real-time baseline 3D MOT system. We use an off-the-shelf 3D object detector to obtain oriented 3D bounding boxes from the LiDAR point cloud. Then, a combination of 3D Kalman filter and Hungarian algorithm is used for state estimation and data association.

## Installation

Please follow carefully our provided [installation instructions](docs/INSTALL.md), to avoid errors when running the code.

## Quick Demo on KITTI

To quickly get a sense of our method's performance on the KITTI dataset, one can run the following command after installation of the code. This step does not require you to download any dataset (a small set of data is already included in this code repository).

```
python3 main.py --dataset KITTI --split val --det_name pointrcnn
python3 scripts/post_processing/trk_conf_threshold.py --dataset KITTI --result_sha pointrcnn_val_H1
python3 scripts/post_processing/visualization.py --result_sha pointrcnn_val_H1_thres --split val
```

## Benchmarking

We provide instructions (inference, evaluation and visualization) for reproducing our method's performance on various supported datasets ([KITTI](docs/KITTI.md), [nuScenes](docs/nuScenes.md)) for benchmarking purposes. 

### Acknowledgement

The idea of this method is inspired by "[SORT](https://github.com/abewley/sort)"
