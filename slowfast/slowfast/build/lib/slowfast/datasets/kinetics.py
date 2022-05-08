#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from torchvision import transforms
import numpy as np
import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment

import torchvision.transforms as tt

import glob
import struct

logger = logging.get_logger(__name__)
P = np.array([7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01, 0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]) 
P = np.reshape(P, [3,4])
print("Calibration Matrix: ", P)
def cart2hom(pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom

def project_rect_to_image(pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def box2corners3d_camcoord(bbox):
        ''' Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and 
            convert it to the 8 corners of the 3D box, the box is in the camera coordinate
            with right x, down y, front z
            
            Returns:
                corners_3d: (8,3) array in in rect camera coord

            box corner order is like follows
                    1 -------- 0         top is bottom because y direction is negative
                   /|         /|
                  2 -------- 3 .
                  | |        | |
                  . 5 -------- 4
                  |/         |/
                  6 -------- 7    
            
            rect/ref camera coord:
            right x, down y, front z

            x -> w, z -> l, y -> h
        '''

        # if already computed before, then skip it
        #bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry
        R = roty(bbox[6])   

        # 3d bounding box dimensions
        l, w, h = bbox[2], bbox[1], bbox[0]

        # 3d bounding box corners
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
        y_corners = [0,0,0,0,-h,-h,-h,-h];
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0,:] = corners_3d[0,:] + bbox[3]
        corners_3d[1,:] = corners_3d[1,:] + bbox[4]
        corners_3d[2,:] = corners_3d[2,:] + bbox[5]
        corners_3d = np.transpose(corners_3d)
        
        return corners_3d

def draw_box3d_image(image, qs, img_size=(900, 1600), color=(255,255,255), thickness=4):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''

    def check_outside_image(x, y, height, width):
        if x < 0 or x >= width: return True
        if y < 0 or y >= height: return True

    # if 6 points of the box are outside the image, then do not draw
    pts_outside = 0
    for index in range(8):
        check = check_outside_image(qs[index, 0], qs[index, 1], img_size[0], img_size[1])
        if check: pts_outside += 1
    if pts_outside >= 6: return image, False

    # actually draw
    if qs is not None:
        qs = qs.astype(np.int32)
        for k in range(0,4):
            i,j=k,(k+1)%4
            image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA) # use LINE_AA for opencv3

            i,j=k+4,(k+1)%4 + 4
            image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k,k+4
            image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

    return image, True


@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        # self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every video.

        # GAO: NUM_SPATIAL_CROPS is 1 because the framse prepared by Celso is 224 x 224.

        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
#         print("******************************************PATH********************************************************************", path_to_file)
        assert pathmgr.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._labels = []
        self._num_frames_of_folder = [] # GAO: add num_frames of each frame folder
        self._spatial_temporal_idx = []
        print("************************************", self.cfg.DATA.PATH_LABEL_SEPARATOR)
        with pathmgr.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                #print(len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)))
                assert len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)) == 2
                path, num_frame = path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR) ; label = "0"
                #print("******************************************PATH and frame ********************************************************************")
#                 if isinstance(path,str) and isinstance(num_frame,str): continue
#                 print(path)
#                 print(num_frame)
#                 print(self.cfg.DATA.PATH_PREFIX)
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._num_frames_of_folder.append(int(num_frame))
                    self._spatial_temporal_idx.append(idx)
                    # self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    # GAO: 
    def _get_seq_frames(self, index, temporal_sample_index = -1, spatial_sample_index = -1):
        """
        Given the video index, return the list of indexs of sampled frames.
        Args:
            index (int): the video index.
            temporal_sample_index: -1 for random select
            spatial_sample_index: -1 for random select
        Returns:
            seq (list): the indexes of sampled frames from the video.
        """
        num_frames = self.cfg.DATA.NUM_FRAMES
        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        
        if self.mode in ["train"]:
            
            path = os.getcwd()+ '/' + 'data/' +  self._path_to_videos[index].split('/')[-3] + '/' + self._path_to_videos[index].split('/')[-2] + '/' + self._path_to_videos[index].split('/')[-1] + '/'
            txt_name = self._path_to_videos[index].split('/')[-1]
            #print("******************************************", path)
            self._path_to_videos[index] = path
            num_frames_of_folder = self._num_frames_of_folder[index]

            delta = max(num_frames_of_folder - num_frames * sampling_rate, 0)

            if temporal_sample_index == -1:
                start = int(random.uniform(0, delta))

                seq = list(range(start, start + num_frames * sampling_rate))
                # seq_sampled = seq[::2] GAO, 20220111
                seq_sampled = seq #[::sampling_rate]
                frame_samples = seq[::sampling_rate]
                frames_seq = [self._path_to_videos[index] + str(i).zfill(6) + ".png" for i in frame_samples] 
                #print("******************************************seq_sampled", seq_sampled)
                # GAO, 20211202, change to i+1 because stylized synthetic is start from 00001
                # GAO, 20211206, change back to i 
                frame_count = len(glob.glob(os.path.join(self._path_to_videos[index], "*[0-9].png")))
                #print(frame_count, num_frames_of_folder)

                assert (frame_count == num_frames_of_folder 
                ), "{} num of png dose not equal to json".format(self._path_to_videos[index])
                textfile = os.path.join(self._path_to_videos[index], txt_name + ".txt")
                dets = []
                velo = []
                centers = []
                with open(textfile, 'r') as f:
                    for line in f:
                        det = eval(line.split('\t')[1])
                        dets.append(det)
                        velo.append(eval(line.split('\t')[2]))
                        center = []
                        for box in det:
                            
                            obj_8corner = box2corners3d_camcoord(box)
                            obj_pts_2d = project_rect_to_image(obj_8corner)
#                             print(np.mean(obj_pts_2d, axis = 0))
                            x, y = np.mean(obj_pts_2d, axis = 0)
                            x = int(x) if x > 0 else 0; y = int(y) if y > 0 else 0
                            center.append([x,y])
                        #print(center)
                        centers.append(center)
#                 print("******************************************lenght ****************************")
#                 print(len(dets), len(velo))
                
                new_velos = []
                confid_scs = []
                new_dets = []
                #new_centers = []
                for i,s in enumerate(seq_sampled):
                    if i%2 != 0:
                        continue
                    new_det = []
                    l = len(dets[s])
                    new_velo = []
                    confid_sc = []
                    #new_center = []
                    for i in range(10):
                        if i >= l:
                            confid_sc.append(0)
                            new_velo.append(0)
                            new_det.append(0)
                            #new_center.append([-1,-1])
                        else:
                            confid_sc.append(1)
                            new_velo.append(np.linalg.norm(velo[s][i]))
                            new_det.append(dets[s][i][3]) #bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry
                            #new_center.append([centers[s][i][0],centers[s][i][1]])
                    new_velos.append(new_velo)
                    new_dets.append(new_det)
                    confid_scs.append(confid_sc)
                    #new_centers.append(new_center)
#                     print(new_det)
#                     print(new_velo)
#                     print(confid_sc)
                new_velos =  torch.Tensor(new_velos)
                new_dets = torch.Tensor(new_dets)
                confid_scs = torch.Tensor(confid_scs)
#                 print(new_centers)
#                 print(new_velos.shape)
#                 print(new_dets.shape)
#                 print(confid_scs.shape)
                frame_labels = torch.cat((confid_scs,new_dets,new_velos ), dim = 0).transpose(1,0)
#                 print(frame_labels.shape)
                    
                     
                    
                

#                 fmt_str = "<{}i".format(frame_count)
# #                 labels = list(
# #                     struct.unpack(fmt_str, open(os.path.join(self._path_to_videos[index], "label.bin"), "rb").read())
# #                 )
               
#                 frame_labels = torch.as_tensor(list(labels[i] for i in seq[::8]))
#                 #print("GFLAFALFALKJSFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLFFAKSF")
#                 #print(frame_labels)
#                 ratio = 0
#                 label = 0
#                 while(ratio < 0.8):
#                     start = int(random.uniform(0, delta))

#                     seq = list(range(start, start + num_frames * sampling_rate))
#                     # seq_sampled = seq[::2] GAO, 20220111
#                     seq_sampled = seq[::sampling_rate]
#                     frames_seq = [self._path_to_videos[index] + str(i).zfill(6) + ".jpg" for i in seq_sampled]
      
#                     frame_count = len(glob.glob(os.path.join(self._path_to_videos[index], "*[0-9].jpg")))
#                     #print(self._path_to_videos[index])
#                     #print(ratio)

#                     assert (frame_count == num_frames_of_folder
#                     ), "{} num of png dose not equal to json".format(self._path_to_videos[index])
    
#                     fmt_str = "<{}i".format(frame_count)
# #                     labels = list(
# #                         struct.unpack(fmt_str, open(os.path.join(self._path_to_videos[index], txt_name + ".txt"), "rb").read())
# #                     )
#                     textfile = os.path.join(self._path_to_videos[index], txt_name + ".txt")
#                     det = []
#                     velo = []
#                     with open(textfile, 'r') as f:
#                         for line in f:
#                             det.append(eval(line.split('\t')[1]))
#                             velo.append(eval(line.split('\t')[2]))
#                     print(len(det), len(velo))
#                     assert len(det) == len(velo)
#                     seql = list(labels[i] for i in seq)
#                     #print(seql)
#                     ratiol = seql.count(max(seql))/len(seql)
#                     ratioz = seql.count(min(seql))/len(seql)
#                     if ratioz > ratiol:
#                         ratio = ratioz
#                         label = 0
#                     else:
#                         ratio = ratiol
#                         label = max(seql)

#                 #print("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
#                 #print(seql, ratio, label)
                    
#                 frame_labels = label
                    
                    




            
        if self.mode in ["val", "test"]:
            # GAO: for eval, not num of frames, but start frame number
            path = os.getcwd()+ '/' + 'data/' +  self._path_to_videos[index].split('/')[-3] + '/' + self._path_to_videos[index].split('/')[-2] + '/' + self._path_to_videos[index].split('/')[-1] + '/'
            txt_name = self._path_to_videos[index].split('/')[-1]
            self._path_to_videos[index] = path
            start = self._num_frames_of_folder[index]
            #print("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
            #print(self._path_to_videos[index],start)
            self._path_to_videos[index] = path
            num_frames_of_folder = self._num_frames_of_folder[index]

            delta = max(num_frames_of_folder - num_frames * sampling_rate, 0)

            start = int(random.uniform(0, delta))
            seq = list(range(start, start + num_frames * sampling_rate))

            frame_samples = seq[::sampling_rate]
            # seq_sampled = seq[::2] GAO, 20220111
            seq_sampled = seq #[::sampling_rate]
            frames_seq = [self._path_to_videos[index] + str(i).zfill(6) + ".png" for i in frame_samples]
            
            frame_count = len(glob.glob(os.path.join(self._path_to_videos[index], "*[0-9].png")))
            fmt_str = "<{}i".format(frame_count)
#             labels = list(
#                 struct.unpack(fmt_str, open(os.path.join(self._path_to_videos[index], "label.bin"), "rb").read())
#             )
                
            #frame_labels = torch.as_tensor(list(labels[i] for i in seq[::8]))
            #self.ignore = 1
            #ratio = seq.count(max(seq))/len(seq)
            #label = max(seq)
            textfile = os.path.join(self._path_to_videos[index], txt_name + ".txt")
            dets = []
            velo = []
            centers = []
            with open(textfile, 'r') as f:
                for line in f:
                    det = eval(line.split('\t')[1])
                    dets.append(det)
                    velo.append(eval(line.split('\t')[2]))
                    center = []
                    for box in det:

                        obj_8corner = box2corners3d_camcoord(box)
                        obj_pts_2d = project_rect_to_image(obj_8corner)
#                             print(np.mean(obj_pts_2d, axis = 0))
                        x, y = np.mean(obj_pts_2d, axis = 0)
                        x = int(x) if x > 0 else 0; y = int(y) if y > 0 else 0
                        center.append([x,y])
                    #print(center)
                    centers.append(center)
#                 print("******************************************lenght ****************************")
#                 print(len(dets), len(velo))

            new_velos = []
            confid_scs = []
            new_dets = []
            for i,s in enumerate(seq_sampled):
                if i%2 != 0:
                    continue
                new_det = []
                l = len(dets[s])
                new_velo = []
                confid_sc = []
                for i in range(10):
                    if i >= l:
                        confid_sc.append(0)
                        new_velo.append(0)
                        new_det.append(0)
                    else:
                        confid_sc.append(1)
                        new_velo.append(np.linalg.norm(velo[s][i]))
                        new_det.append(dets[s][i][3]) #bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry

                new_velos.append(new_velo)
                new_dets.append(new_det)
                confid_scs.append(confid_sc)

#                     print(new_det)
#                     print(new_velo)
#                     print(confid_sc)
            new_velos =  torch.Tensor(new_velos)
            new_dets = torch.Tensor(new_dets)
            confid_scs = torch.Tensor(confid_scs)
#                 print(new_velos.shape)
#                 print(new_dets.shape)
#                 print(confid_scs.shape)
            frame_labels = torch.cat((confid_scs,new_dets,new_velos ), dim = 0).transpose(1,0)
#             seql = list(labels[i] for i in seq)
#             #print(seql)
#             ratiol = seql.count(max(seql))/len(seql)
#             ratioz = seql.count(min(seql))/len(seql)
#             if ratioz > ratiol:
#                 ratio = ratioz
#                 label = 0
#             else:
#                 ratio = ratiol
#                 label = max(seql)

#             frame_labels = label
#             print(seql, ratio, label)



        frames = torch.as_tensor(
            utils.retry_load_images(frames_seq, self._num_retries))

#         if self.mode in ["train"]:
#             if self.cfg.DATA.COLOR_JITTER:
#                 if random.choice([True, False]):
#                     #print("CJ")
#                     frames = frames.permute(0,3,1,2)
#                     # jitter = tt.functional.adjust_brightness(frames)
#                     brightness_factor = random.random()*2
#                     hue_factor = random.random()-0.5
#                     contrast_factor = random.random()*2
#                     saturation_factor = random.random()*2
#                     for i in range(len(frames)):
#                         transformed = tt.functional.adjust_brightness(frames[i], brightness_factor)
#                         transformed = tt.functional.adjust_hue(transformed, hue_factor)
#                         transformed = tt.functional.adjust_contrast(transformed, contrast_factor)
#                         transformed = tt.functional.adjust_saturation(transformed, saturation_factor)
#                         frames[i] = transformed
#                     frames = frames.permute(0,2,3,1)

#             if self.cfg.DATA.GAUSSIAN_BLUR:
#                 if random.choice([True, False]):
#                     #print("GB")
#                     frames = frames.permute(0,3,1,2)
#                     for i in range(len(frames)):
#                         transformed = tt.functional.gaussian_blur(frames[i], 3)
#                         frames[i] = transformed
#                     frames = frames.permute(0,2,3,1)

#             if self.cfg.DATA.ADJUST_SHARPNESS:
#                 if random.choice([True, False]):
#                     #print("AS")
#                     frames = frames.permute(0,3,1,2)
#                     factor = random.choice([0,1,2,3])
#                     for i in range(len(frames)):
#                         transformed = tt.functional.adjust_sharpness(frames[i], factor)
#                         frames[i] = transformed
#                     frames = frames.permute(0,2,3,1)


        return frames, frame_labels



    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S)
                )
        elif self.mode in ["val", "test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS)
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )


        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):

            frames, frame_labels = self._get_seq_frames(index, temporal_sample_index, spatial_sample_index)
            #print(frames.shape, frame_labels.shape)

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            if self.aug:
                if self.cfg.AUG.NUM_SAMPLE > 1:

                    frame_list = []
                    label_list = []
                    index_list = []
                    for _ in range(self.cfg.AUG.NUM_SAMPLE):
                        new_frames = self._aug_frame(
                            frames,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )
                        label = frame_labels
                        new_frames = utils.pack_pathway_output(self.cfg, new_frames)
                        frame_list.append(new_frames)
                        label_list.append(label)
                        index_list.append(index)
                    #print(len(label_list), label_list)
                    return frame_list, label_list, index_list, {}

                else:
                    frames = self._aug_frame(
                        frames,
                        spatial_sample_index,
                        min_scale,
                        max_scale,
                        crop_size,
                    )

            else:
                frames = utils.tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                # Perform data augmentation.

                #GAO  20220129 add spatial sampling
                frames = utils.spatial_sampling(
                     frames,
                     spatial_idx=spatial_sample_index,
                     min_scale=min_scale,
                     max_scale=max_scale,
                     crop_size=crop_size,
                     random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                     inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                 )

            label = frame_labels
            #print(label.shape)
            #print(frames.shape)
            #print("GET ITEMMMMMMMM")
            frames = utils.pack_pathway_output(self.cfg, frames)
            return frames, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(self._num_retries)
            )

    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = None if (self.mode not in ["train"] or len(scl) == 0) else scl
        relative_aspect = None if (self.mode not in ["train"] or len(asp) == 0) else asp
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

    def _frame_to_list_img(self, frames):
        img_list = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

if __name__ == '__main__':
    
    k = Kinetics(cfg=0)
    
