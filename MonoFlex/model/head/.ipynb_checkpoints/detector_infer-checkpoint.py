import torch
import pdb
import math

from torch import nn
from shapely.geometry import Polygon
from torch.nn import functional as F

from model.layers.iou_loss import *


from model.anno_encoder import Anno_Encoder
from model.layers.utils import (
	nms_hm,
	select_topk,
	select_point_of_interest,
)

from model.layers.utils import Converter_key2channel
from engine.visualize_infer import box_iou, box_iou_3d, box3d_to_corners

def get_rotys_from_locs(locs):

	th_k1k2 = torch.atan(  (locs[:,0,2] - locs[:,1,2])  /  (locs[:,0,0] - locs[:,1,0])   )
	th_k4k3 = torch.atan(  (locs[:,3,2] - locs[:,2,2])  /  (locs[:,3,0] - locs[:,2,0])   )
	th_k5k6 = torch.atan(  (locs[:,4,2] - locs[:,5,2])  /  (locs[:,4,0] - locs[:,5,0])   )
	th_k8k7 = torch.atan(  (locs[:,7,2] - locs[:,6,2])  /  (locs[:,7,0] - locs[:,6,0])   )

	avg = (th_k1k2 + th_k4k3 + th_k5k6 + th_k8k7) / 4

	return avg

def get_dims_from_locs(locs):

    # locs (B,8,3)
	h_k5k1 = torch.sqrt( (locs[:,4,0] - locs[:,0,0])**2 + (locs[:,4,1] - locs[:,0,1])**2 + (locs[:,4,2] - locs[:,0,2])**2  )
	h_k8k4 = torch.sqrt( (locs[:,7,0] - locs[:,3,0])**2 + (locs[:,7,1] - locs[:,3,1])**2 + (locs[:,7,2] - locs[:,3,2])**2  )
	h_k6k2 = torch.sqrt( (locs[:,5,0] - locs[:,1,0])**2 + (locs[:,5,1] - locs[:,1,1])**2 + (locs[:,5,2] - locs[:,1,2])**2  )
	h_k7k3 = torch.sqrt( (locs[:,6,0] - locs[:,2,0])**2 + (locs[:,6,1] - locs[:,2,1])**2 + (locs[:,6,2] - locs[:,2,2])**2 )

	l_k2k1 = torch.sqrt( (locs[:,1,0] - locs[:,0,0])**2 + (locs[:,1,1] - locs[:,0,1])**2 + (locs[:,1,2] - locs[:,0,2] **2) )
	l_k3k4 = torch.sqrt( (locs[:,2,0] - locs[:,3,0])**2 + (locs[:,2,1] - locs[:,3,1])**2  + (locs[:,2,2] - locs[:,3,2])**2  )
	l_k6k5 = torch.sqrt( (locs[:,5,0] - locs[:,4,0])**2 + (locs[:,5,1] - locs[:,4,1])**2 + (locs[:,5,2] - locs[:,4,2])**2 )
	l_k7k8 = torch.sqrt( (locs[:,6,0] - locs[:,7,0])**2 + (locs[:,6,1] - locs[:,7,1])**2 + (locs[:,6,2] - locs[:,7,2])**2 )

	w_k1k4 = torch.sqrt( (locs[:,0,0] - locs[:,3,0])**2  + (locs[:,0,1] - locs[:,3,1])**2  + (locs[:,0,2] - locs[:,3,2])**2  )
	w_k2k3 = torch.sqrt( (locs[:,1,0] - locs[:,2,0])**2  + (locs[:,1,1] - locs[:,2,1])**2  + (locs[:,1,2] - locs[:,2,2])**2 )
	w_k6k7 = torch.sqrt( (locs[:,5,0] - locs[:,6,0])**2  + (locs[:,5,1] - locs[:,6,1])**2  + (locs[:,5,2] - locs[:,6,2])**2 )
	w_k5k8 = torch.sqrt( (locs[:,4,0] - locs[:,7,0])**2  + (locs[:,4,1] - locs[:,7,1])**2  + (locs[:,4,2] - locs[:,7,2])**2 )
 



	avg_h = (h_k5k1 + h_k8k4 + h_k6k2 + h_k7k3) / 4
	avg_l = (l_k2k1 + l_k3k4 + l_k6k5 + l_k7k8) / 4
	avg_w = (w_k1k4 + w_k2k3 + w_k6k7 + w_k5k8) / 4
	result = torch.stack((avg_h, avg_w, avg_l), dim=1)

	return result


def make_post_processor(cfg):
	anno_encoder = Anno_Encoder(cfg)
	key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS, channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
	postprocessor = PostProcessor(cfg=cfg, anno_encoder=anno_encoder, key2channel=key2channel)
	
	return postprocessor

class PostProcessor(nn.Module):
	def __init__(self, cfg, anno_encoder, key2channel):
		
		super(PostProcessor, self).__init__()

		self.anno_encoder = anno_encoder
		self.key2channel = key2channel

		self.det_threshold = cfg.TEST.DETECTIONS_THRESHOLD
		self.max_detection = cfg.TEST.DETECTIONS_PER_IMG		
		self.eval_dis_iou = cfg.TEST.EVAL_DIS_IOUS
		self.eval_depth = cfg.TEST.EVAL_DEPTH
		
		self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
		self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
		self.output_depth = cfg.MODEL.HEAD.OUTPUT_DEPTH
		self.pred_2d = cfg.TEST.PRED_2D

		self.pred_direct_depth = 'depth' in self.key2channel.keys
		self.depth_with_uncertainty = 'depth_uncertainty' in self.key2channel.keys
		self.regress_keypoints = 'corner_offset' in self.key2channel.keys
		self.keypoint_depth_with_uncertainty = 'corner_uncertainty' in self.key2channel.keys

		# use uncertainty to guide the confidence
		self.uncertainty_as_conf = cfg.TEST.UNCERTAINTY_AS_CONFIDENCE

	def prepare_targets(self, targets, test):
		pad_size = torch.stack([t.get_field("pad_size") for t in targets])
		calibs = [t.get_field("calib") for t in targets]
		size = torch.stack([torch.tensor(t.size) for t in targets])

		if test: return dict(calib=calibs, size=size, pad_size=pad_size)

		cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
		# regression locations (in pixels)
		target_centers = torch.stack([t.get_field("target_centers") for t in targets])
		# 3D infos
		dimensions = torch.stack([t.get_field("dimensions") for t in targets])
		rotys = torch.stack([t.get_field("rotys") for t in targets])
		locations = torch.stack([t.get_field("locations") for t in targets])
		# offset_2D = torch.stack([t.get_field("offset_2D") for t in targets])
		offset_3D = torch.stack([t.get_field("offset_3D") for t in targets])
		# reg mask
		reg_mask = torch.stack([t.get_field("reg_mask") for t in targets])

		target_varibales = dict(pad_size=pad_size, calib=calibs, size=size, cls_ids=cls_ids, target_centers=target_centers,
							dimensions=dimensions, rotys=rotys, locations=locations, offset_3D=offset_3D, reg_mask=reg_mask)

		return target_varibales

	def forward(self, predictions, targets, features=None, test=False, refine_module=None):
		pred_heatmap, pred_regression = predictions['cls'], predictions['reg']
		batch = pred_heatmap.shape[0]

		target_varibales = self.prepare_targets(targets, test=test)
		calib, pad_size = target_varibales['calib'], target_varibales['pad_size']
		img_size = target_varibales['size']

		# evaluate the disentangling IoU for each components in (location, dimension, orientation)
		self.eval_dis_iou = True
		dis_ious = self.evaluate_3D_detection(target_varibales, pred_regression) if self.eval_dis_iou else None


		# evaluate the accuracy of predicted depths
		depth_errors = self.evaluate_3D_depths(target_varibales, pred_regression) if self.eval_depth else None

		# max-pooling as nms for heat-map
		heatmap = nms_hm(pred_heatmap)
		visualize_preds = {'heat_map': pred_heatmap.clone()}

		# select top-k of the predicted heatmap
		scores, indexs, clses, ys, xs = select_topk(heatmap, K=self.max_detection)

		pred_bbox_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1)
		pred_regression_pois = select_point_of_interest(batch, indexs, pred_regression).view(-1, pred_regression.shape[1])

		# thresholding with score
		scores = scores.view(-1)
		valid_mask = scores >= self.det_threshold

		# no valid predictions
		if valid_mask.sum() == 0:
			result = scores.new_zeros(0, 14)
			visualize_preds['keypoints'] = scores.new_zeros(0, 20)
			visualize_preds['proj_center'] = scores.new_zeros(0, 2)
			eval_utils = {'dis_ious': dis_ious, 'depth_errors': depth_errors, 'vis_scores': scores.new_zeros(0),
					'uncertainty_conf': scores.new_zeros(0), 'estimated_depth_error': scores.new_zeros(0)}
			
			return result, eval_utils, visualize_preds

		scores = scores[valid_mask]
		
		clses = clses.view(-1)[valid_mask]
		pred_bbox_points = pred_bbox_points[valid_mask]
		pred_regression_pois = pred_regression_pois[valid_mask]

		pred_2d_reg = F.relu(pred_regression_pois[:, self.key2channel('2d_dim')])
		pred_offset_3D = pred_regression_pois[:, self.key2channel('3d_offset')]
		pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
		pred_orientation = torch.cat((pred_regression_pois[:, self.key2channel('ori_cls')], pred_regression_pois[:, self.key2channel('ori_offset')]), dim=1)
		visualize_preds['proj_center'] = pred_bbox_points + pred_offset_3D

		#pred_box2d = self.anno_encoder.decode_box2d_fcos(pred_bbox_points, pred_2d_reg, pad_size, img_size)
		pred_box2d = 0
		pred_dimensions = self.anno_encoder.decode_dimension(clses, pred_dimensions_offsets)

		if self.pred_direct_depth:
			pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')].squeeze(-1)
			pred_direct_depths = self.anno_encoder.decode_depth(pred_depths_offset)

		if self.depth_with_uncertainty:
			pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
			visualize_preds['depth_uncertainty'] = pred_regression[:, self.key2channel('depth_uncertainty'), ...].squeeze(1)

		if self.regress_keypoints:
			pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]
			pred_keypoint_offset = pred_keypoint_offset.view(-1, 8, 3)
			#print(iou_3d)
			#print(calib)
			pred_box2d = calib[0].project_rect_to_image_tensor(pred_keypoint_offset.reshape(-1, 3))[0].view(-1, 8, 2)
			pred_box2d_f = pred_box2d[:, 0:4, 0].clone()
			for ind in range(pred_box2d.shape[0]):
				pred_box2d_f[ind, 0] = torch.min(pred_box2d[ind, :, 0])
				pred_box2d_f[ind, 1] = torch.min(pred_box2d[ind, :, 1])
				pred_box2d_f[ind, 2] = torch.max(pred_box2d[ind, :, 0])
				pred_box2d_f[ind, 3] = torch.max(pred_box2d[ind, :, 1])

			pred_box2d = pred_box2d_f
			#print(pred_box2d)            
                
                
			#print(pred_box2d.shape)           
			# solve depth from estimated key-points
			#pred_keypoints_depths = self.anno_encoder.decode_depth_from_keypoints_batch(pred_keypoint_offset, pred_dimensions, calib)
			pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]
			pred_keypoints_3D = pred_keypoint_offset
			pred_keypoints_3D = pred_keypoints_3D.view(pred_keypoints_3D.shape[0], -1, 3)
			pred_keypoints_depths = torch.squeeze(pred_keypoints_3D[:,0,0:2].clone())
			for i in range(pred_keypoints_3D.shape[0]):
				pred_keypoints_depths[i,0] = ((pred_keypoints_3D[i,0,2] + pred_keypoints_3D[i,4,2])/2 + (pred_keypoints_3D[i,2,2] + pred_keypoints_3D[i,6,2])/2)/2
				pred_keypoints_depths[i,1] = ((pred_keypoints_3D[i,3,2] + pred_keypoints_3D[i,7,2])/2 + (pred_keypoints_3D[i,1,2] + pred_keypoints_3D[i,5,2])/2)/2        
			visualize_preds['keypoints'] = pred_keypoint_offset

		if self.keypoint_depth_with_uncertainty:
			pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()

		estimated_depth_error = None

		if self.output_depth == 'direct':
			pred_depths = pred_direct_depths

			if self.depth_with_uncertainty: estimated_depth_error = pred_direct_uncertainty.squeeze(dim=1)
		
		elif self.output_depth.find('keypoints') >= 0:
			if self.output_depth == 'keypoints_avg':
				pred_depths = pred_keypoints_depths.mean(dim=1)
				if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty.mean(dim=1)

			#elif self.output_depth == 'keypoints_center':
			#	pred_depths = pred_keypoints_depths[:, 0]
			#	if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 0]

			elif self.output_depth == 'keypoints_02':
				pred_depths = pred_keypoints_depths[:, 0]
				if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 0]

			elif self.output_depth == 'keypoints_13':
				pred_depths = pred_keypoints_depths[:, 1]
				if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 1]

			else:
				raise ValueError

		# hard ensemble, soft ensemble and simple average
		elif self.output_depth in ['hard', 'soft', 'mean', 'oracle']:
			if self.pred_direct_depth and self.depth_with_uncertainty:
				pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)
				pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), dim=1)
			else:
				pred_combined_depths = pred_keypoints_depths.clone()
				pred_combined_uncertainty = pred_keypoint_uncertainty.clone()
			
			depth_weights = 1 / pred_combined_uncertainty
			visualize_preds['min_uncertainty'] = depth_weights.argmax(dim=1)

			if self.output_depth == 'hard':
				pred_depths = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]), depth_weights.argmax(dim=1)]

				# the uncertainty after combination				
				estimated_depth_error = pred_combined_uncertainty.min(dim=1).values

			elif self.output_depth == 'soft':
				depth_weights = depth_weights / depth_weights.sum(dim=1, keepdim=True)
				#print(depth_weights.shape, pred_combined_depths.shape)
				pred_depths = torch.sum(pred_combined_depths * depth_weights, dim=1)

				# the uncertainty after combination
				estimated_depth_error = torch.sum(depth_weights * pred_combined_uncertainty, dim=1)
				
			elif self.output_depth == 'mean':
				pred_depths = pred_combined_depths.mean(dim=1)

				# the uncertainty after combination
				estimated_depth_error = pred_combined_uncertainty.mean(dim=1)

			# the best estimator is always selected
			elif self.output_depth == 'oracle':
				pred_depths, estimated_depth_error = self.get_oracle_depths(pred_box2d, clses, pred_combined_depths, 
																pred_combined_uncertainty, targets[0])

		batch_idxs = pred_depths.new_zeros(pred_depths.shape[0]).long()
		pred_locations = self.anno_encoder.decode_location_flatten(pred_bbox_points, pred_offset_3D, pred_depths, calib, pad_size, batch_idxs)
		pred_rotys, pred_alphas = self.anno_encoder.decode_axes_orientation(pred_orientation, pred_locations)

		pred_locations[:, 1] += pred_dimensions[:, 1] / 2
		clses = clses.view(-1, 1)
		pred_alphas = pred_alphas.view(-1, 1)
		pred_rotys = pred_rotys.view(-1, 1)
		scores = scores.view(-1, 1)
		# change dimension back to h,w,l
		pred_dimensions = pred_dimensions.roll(shifts=-1, dims=1)

		# the uncertainty of depth estimation can reflect the confidence for 3D object detection
		vis_scores = scores.clone()
		if self.uncertainty_as_conf and estimated_depth_error is not None:
			uncertainty_conf = 1 - torch.clamp(estimated_depth_error, min=0.01, max=1)	
			scores = scores * uncertainty_conf.view(-1, 1)
		else:
			uncertainty_conf, estimated_depth_error = None, None

		# kitti output format pred_keypoints_3D
		#print(pred_box2d)        
		result = torch.cat([clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores], dim=1)
		
		eval_utils = {'dis_ious': dis_ious, 'depth_errors': depth_errors, 'uncertainty_conf': uncertainty_conf,
					'estimated_depth_error': estimated_depth_error, 'vis_scores': vis_scores}
		
		return result, eval_utils, visualize_preds

	def get_oracle_depths(self, pred_bboxes, pred_clses, pred_combined_depths, pred_combined_uncertainty, target):
		calib = target.get_field('calib')
		pad_size = target.get_field('pad_size')
		pad_w, pad_h = pad_size

		valid_mask = target.get_field('reg_mask').bool()
		num_gt = valid_mask.sum()
		gt_clses = target.get_field('cls_ids')[valid_mask]
		gt_boxes = target.get_field('gt_bboxes')[valid_mask]
		gt_locs = target.get_field('locations')[valid_mask]

		gt_depths = gt_locs[:, -1]
		gt_boxes_center = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2

		iou_thresh = 0.5

		# initialize with the average values
		oracle_depth = pred_combined_depths.mean(dim=1)
		estimated_depth_error = pred_combined_uncertainty.mean(dim=1)

		for i in range(pred_bboxes.shape[0]):
			# find the corresponding object bounding boxes
			box2d = pred_bboxes[i]
			box2d_center = (box2d[:2] + box2d[2:]) / 2
			img_dis = torch.sum((box2d_center.reshape(1, 2) - gt_boxes_center) ** 2, dim=1)
			same_cls_mask = gt_clses == pred_clses[i]
			img_dis[~same_cls_mask] = 9999
			near_idx = torch.argmin(img_dis)
			# iou 2d
			iou_2d = box_iou(box2d.detach().cpu().numpy(), gt_boxes[near_idx].detach().cpu().numpy())
			
			if iou_2d < iou_thresh:
				# match failed, simply choose the default average
				continue
			else:
				estimator_index = torch.argmin(torch.abs(pred_combined_depths[i] - gt_depths[near_idx]))
				oracle_depth[i] = pred_combined_depths[i,estimator_index]
				estimated_depth_error[i] = pred_combined_uncertainty[i, estimator_index]

		return oracle_depth, estimated_depth_error

	def evaluate_3D_depths(self, targets, pred_regression):
		# computing disentangling 3D IoU for offset, depth, dimension, orientation
		batch, channel = pred_regression.shape[:2]

		# 1. extract prediction in points of interest
		target_points = targets['target_centers'].float()
		pred_regression_pois = select_point_of_interest(
			batch, target_points, pred_regression
		)

		pred_regression_pois = pred_regression_pois.view(-1, channel)
		reg_mask = targets['reg_mask'].view(-1).bool()
		pred_regression_pois = pred_regression_pois[reg_mask]
		target_points = target_points[0][reg_mask]

		# depth predictions
		pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')]
		pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]

		pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
		pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()

		# dimension predictions
		target_clses = targets['cls_ids'].view(-1)[reg_mask]
		pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
		pred_dimensions = self.anno_encoder.decode_dimension(
			target_clses,
			pred_dimensions_offsets,
		)
		# direct
		pred_direct_depths = self.anno_encoder.decode_depth(pred_depths_offset.squeeze(-1))
		# three depths from keypoints
		#pred_keypoints_depths = self.anno_encoder.decode_depth_from_keypoints_batch(pred_keypoint_offset.view(-1, 10, 2), pred_dimensions, targets['calib'])
		preds_keypoints_3D = pred_keypoint_offset.view(-1, 8, 3)
		pred_keypoints_depths = torch.squeeze(pred_keypoints_3D[:,0,0:2].clone())
		for i in range(pred_keypoints_3D.shape[0]):
			pred_keypoints_depths[i,0] = ((pred_keypoints_3D[i,0,2] + pred_keypoints_3D[i,4,2])/2 + (pred_keypoints_3D[i,2,2] + pred_keypoints_3D[i,6,2])/2)/2
			pred_keypoints_depths[i,1] = ((pred_keypoints_3D[i,3,2] + pred_keypoints_3D[i,7,2])/2 + (pred_keypoints_3D[i,1,2] + pred_keypoints_3D[i,5,2])/2)/2        
        
		# combined uncertainty
		pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), dim=1)
		pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)

		# min-uncertainty
		pred_uncertainty_min_depth = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]), pred_combined_uncertainty.argmin(dim=1)]
		# inv-uncertainty weighted
		pred_uncertainty_weights = 1 / pred_combined_uncertainty
		pred_uncertainty_weights = pred_uncertainty_weights / pred_uncertainty_weights.sum(dim=1, keepdim=True)
		
		pred_uncertainty_softmax_depth = torch.sum(pred_combined_depths * pred_uncertainty_weights, dim=1)

		# 3. get ground-truth
		target_locations = targets['locations'].view(-1, 3)[reg_mask][:,1:3]
		target_depths = target_locations[:, -1]

		# abs error
		pred_combined_error = (pred_combined_depths - target_depths[:, None]).abs()
		pred_uncertainty_min_error = (pred_uncertainty_min_depth - target_depths).abs()
		pred_uncertainty_softmax_error = (pred_uncertainty_softmax_depth - target_depths).abs()
		pred_direct_error = pred_combined_error[:, 0]
		pred_keypoints_error = pred_combined_error[:, 1:]

		pred_mean_depth = pred_combined_depths.mean(dim=1)
		pred_mean_error = (pred_mean_depth - target_depths).abs()
		# upper-bound
		pred_min_error = pred_combined_error.min(dim=1)[0]

		pred_errors = {
						'direct': pred_direct_error,
						'direct_sigma': pred_direct_uncertainty[:, 0],
						
						'keypoint_center': pred_keypoints_error[:, 0],
						'keypoint_02': pred_keypoints_error[:, 1],
						'keypoint_13': pred_keypoints_error[:, 2],
						'keypoint_center_sigma': pred_keypoint_uncertainty[:, 0],
						'keypoint_02_sigma': pred_keypoint_uncertainty[:, 1],
						'keypoint_13_sigma': pred_keypoint_uncertainty[:, 2],

						'sigma_min': pred_uncertainty_min_error,
						'sigma_weighted': pred_uncertainty_softmax_error,
						'mean': pred_mean_error,
						'min': pred_min_error,
						'target': target_depths, 
					}

		return pred_errors

	def evaluate_3D_detection(self, targets, pred_regression):
		# computing disentangling 3D IoU for offset, depth, dimension, orientation
		batch, channel = pred_regression.shape[:2]

		# 1. extract prediction in points of interest
		target_points = targets['target_centers'].float()
		pred_regression_pois = select_point_of_interest(
			batch, target_points, pred_regression
		)

		# 2. get needed predictions
		pred_regression_pois = pred_regression_pois.view(-1, channel)
		reg_mask = targets['reg_mask'].view(-1).bool()
		pred_regression_pois = pred_regression_pois[reg_mask]
		target_points = target_points[0][reg_mask]

		pred_offset_3D = pred_regression_pois[:, self.key2channel('3d_offset')]
		pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
		pred_orientation = torch.cat((pred_regression_pois[:, self.key2channel('ori_cls')], pred_regression_pois[:, self.key2channel('ori_offset')]), dim=1)
		pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')].view(-1, 8, 3)


		# 3. get ground-truth
		target_clses = targets['cls_ids'].view(-1)[reg_mask]
		target_offset_3D = targets['offset_3D'].view(-1, 2)[reg_mask]
		target_locations = targets['locations'].view(-1, 3)[reg_mask]
		target_dimensions = targets['dimensions'].view(-1, 3)[reg_mask]
		target_rotys = targets['rotys'].view(-1)[reg_mask]
        
		inds = target_clses == 0
		target_corners_3D = self.anno_encoder.encode_box3d(target_rotys, target_dimensions, target_locations)
		#print(pred_orientation.shape, pred_keypoint_offset.shape, target_corners_3D.shape)
		if target_corners_3D[inds] is not None:
			iou_3d = get_iou_3d(pred_keypoint_offset[inds], target_corners_3D[inds]).mean(); print(iou_3d)

		target_depths = target_locations[:, -1]

		# 4. decode prediction
		pred_dimensions = self.anno_encoder.decode_dimension(
			target_clses,
			pred_dimensions_offsets,
		)
		self.output_depth = 'direct'

		pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')].squeeze(-1)
		if self.output_depth == 'direct':
			pred_depths = self.anno_encoder.decode_depth(pred_depths_offset)

		elif self.output_depth == 'keypoints':
			pred_depths = self.anno_encoder.decode_depth_from_keypoints(pred_offset_3D, pred_keypoint_offset, pred_dimensions, targets['calib'])
			pred_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()
			pred_depths = pred_depths[torch.arange(pred_depths.shape[0]), pred_uncertainty.argmin(dim=1)]

		elif self.output_depth == 'combine':
			pred_direct_depths = self.anno_encoder.decode_depth(pred_depths_offset)
			pred_keypoints_depths = self.anno_encoder.decode_depth_from_keypoints(pred_offset_3D, pred_keypoint_offset, pred_dimensions, targets['calib'])
			pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)

			pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
			pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()
			pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), dim=1)
			pred_depths = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]), pred_combined_uncertainty.argmin(dim=1)]

		batch_idxs = pred_depths.new_zeros(pred_depths.shape[0]).long()
		pred_locations_offset = self.anno_encoder.decode_location_flatten(target_points, pred_offset_3D, target_depths, 
								targets['calib'], targets["pad_size"], batch_idxs)

		pred_locations_depth = self.anno_encoder.decode_location_flatten(target_points, target_offset_3D, pred_depths, 
								targets['calib'], targets["pad_size"], batch_idxs)

		pred_locations = self.anno_encoder.decode_location_flatten(target_points, pred_offset_3D, pred_depths, 
								targets['calib'], targets["pad_size"], batch_idxs)

		pred_rotys, _ = self.anno_encoder.decode_axes_orientation(
			pred_orientation,
			target_locations,
		)

		fully_pred_rotys, _ = self.anno_encoder.decode_axes_orientation(pred_orientation, pred_locations)
        
		pred_locations = torch.squeeze(torch.mean(pred_keypoint_offset, dim = 1))
		pred_dimensions = get_dims_from_locs(pred_keypoint_offset)
		fully_pred_rotys = get_rotys_from_locs(pred_keypoint_offset)

		# fully predicted
		pred_bboxes_3d = torch.cat((pred_locations, pred_dimensions, fully_pred_rotys[:, None]), dim=1)
		# ground-truth
		target_bboxes_3d = torch.cat((target_locations, target_dimensions, target_rotys[:, None]), dim=1)
		# disentangling
		offset_bboxes_3d = torch.cat((pred_locations_offset, target_dimensions, target_rotys[:, None]), dim=1)
		depth_bboxes_3d = torch.cat((pred_locations_depth, target_dimensions, target_rotys[:, None]), dim=1)
		dims_bboxes_3d = torch.cat((target_locations, pred_dimensions, target_rotys[:, None]), dim=1)
		orien_bboxes_3d = torch.cat((target_locations, target_dimensions, pred_rotys[:, None]), dim=1)

		# 6. compute 3D IoU
		pred_IoU = get_iou3d(pred_bboxes_3d, target_bboxes_3d)
		offset_IoU = get_iou3d(offset_bboxes_3d, target_bboxes_3d)
		depth_IoU = get_iou3d(depth_bboxes_3d, target_bboxes_3d)
		dims_IoU = get_iou3d(dims_bboxes_3d, target_bboxes_3d)
		orien_IoU = get_iou3d(orien_bboxes_3d, target_bboxes_3d)
		output = dict(pred_IoU=pred_IoU, offset_IoU=offset_IoU, depth_IoU=depth_IoU, dims_IoU=dims_IoU, orien_IoU=orien_IoU)

		return output

def get_iou3d(pred_bboxes, target_bboxes):
	num_query = target_bboxes.shape[0]

	# compute overlap along y axis
	min_h_a = - (pred_bboxes[:, 1] + pred_bboxes[:, 4] / 2)
	max_h_a = - (pred_bboxes[:, 1] - pred_bboxes[:, 4] / 2)
	min_h_b = - (target_bboxes[:, 1] + target_bboxes[:, 4] / 2)
	max_h_b = - (target_bboxes[:, 1] - target_bboxes[:, 4] / 2)

	# overlap in height
	h_max_of_min = torch.max(min_h_a, min_h_b)
	h_min_of_max = torch.min(max_h_a, max_h_b)
	h_overlap = (h_min_of_max - h_max_of_min).clamp_(min=0)

	# volumes of bboxes
	pred_volumes = pred_bboxes[:, 3] * pred_bboxes[:, 4] * pred_bboxes[:, 5]
	target_volumes = target_bboxes[:, 3] * target_bboxes[:, 4] * target_bboxes[:, 5]

	# derive x y l w alpha
	pred_bboxes = pred_bboxes[:, [0, 2, 3, 5, 6]]
	target_bboxes = target_bboxes[:, [0, 2, 3, 5, 6]]

	# convert bboxes to corners
	pred_corners = get_corners(pred_bboxes)
	target_corners = get_corners(target_bboxes)
	iou_3d = pred_bboxes.new_zeros(num_query)

	for i in range(num_query):
		ref_polygon = Polygon(pred_corners[i])
		target_polygon = Polygon(target_corners[i])
		overlap = ref_polygon.intersection(target_polygon).area
		# multiply bottom overlap and height overlap
		# for 3D IoU
		overlap3d = overlap * h_overlap[i]
		union3d = ref_polygon.area * (max_h_a[0] - min_h_a[0]) + target_polygon.area * (max_h_b[i] - min_h_b[i]) - overlap3d
		iou_3d[i] = overlap3d / union3d

	return iou_3d
