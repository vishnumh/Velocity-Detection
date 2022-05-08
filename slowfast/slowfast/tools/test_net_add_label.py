#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch

import csv
import json



import slowfast.utils.metrics as metrics
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter, ValMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    val_epoch_err = [] # GAO
    num_val = []


    all_out = []
  

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            preds = model(inputs)

            _, out = torch.max(preds, 1)

            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

            
            top1_err, top5_err = [
                (1.0 - x / (preds.size(0) * preds.size(1))) * 100.0 for x in num_topks_correct
            ]
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])
                # GAO: combine the preds and labels across the GPUs, this is originally forgotten!
                preds, labels = du.all_gather([preds, labels])


            top1_err, top5_err = top1_err.item(), top5_err.item()

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                top1_err,
                top5_err,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            test_meter.log_iter_stats(0, cur_iter) # epoch, iter
            
            # GAO: epoch average error
            val_epoch_err.append(top1_err)
            num_val.append(len(preds))
            out = out.clone().detach().cpu()
            all_out.append(out)
           

            test_meter.update_predictions(preds, labels)
            test_meter.iter_tic()

           

    val_ave_top1_err = np.sum((val_epoch_err[i] * num_val[i] for i in range(len(val_epoch_err)))) / np.sum((num_val))
    if writer is not None:
        writer.add_scalars({"Val/Ave_top1_err": val_ave_top1_err}, global_step = 0)

  
    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        if writer is not None:
            all_preds = [pred.clone().detach() for pred in test_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in test_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=0
            )

        if cfg.TEST.SAVE_RESULTS_PATH == "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)
            
            

            f1 = open("/home/zhiweng/slowfast/log/experiment/experiment/all.csv", "w+") 
            csv_writer1 = csv.writer(f1)
            for i in range(len(all_labels)):
                a = (all_labels[i]).view(-1).numpy()
                b = (all_out[i]).view(-1).numpy()
                for j in range(len(a)):
                    csv_writer1.writerow([a[j], b[j]])
                    csv_writer1.writerow([a[j], b[j]])
                    csv_writer1.writerow([a[j], b[j]])
                    csv_writer1.writerow([a[j], b[j]])
                    csv_writer1.writerow([a[j], b[j]])
                    csv_writer1.writerow([a[j], b[j]])
                    csv_writer1.writerow([a[j], b[j]])
                    csv_writer1.writerow([a[j], b[j]])
            
            f1.close()

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "val")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        # Create meters for multi-view testing.
        test_meter = ValMeter(len(test_loader), cfg)

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()



