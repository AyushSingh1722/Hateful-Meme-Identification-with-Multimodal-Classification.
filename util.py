"""
Utility classes and methods for Hateful Memes Classification
"""


import cv2
from copy import deepcopy
import logging
import os
import queue
import re
import shutil
import string
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import ujson as json

import matplotlib.pyplot as plt

from collections import Counter

import fasttext as ft
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg


class HatefulMemes(data.Dataset):
    """
    preprocess image and text data to multimodal tensors
    """
    def __init__(
        self,
        json_path,
        img_folder_dir,
        text_model_path,
        balance=False,
    ):
        self.data = pd.read_json(json_path, lines = True)
        if balance:
            neg = self.data[self.data.label.eq(0)]
            pos = self.data[self.data.label.eq(1)]
            self.data = pd.concat([neg.sample(pos.shape[0]), pos])
        self.data = self.data.reset_index(drop = True)
        self.data['img'] = img_folder_dir +  self.data['img']



    def __getitem__(self, index):

        # get id
        img_id = self.data.loc[index, "id"]
        
        # get image
        image = Image.open(self.data.loc[index, "img"]).convert("RGB")
        
        std_image = Compose(
            [
                Resize(
                    size=(224, 224)
                ),        
                ToTensor(),
                Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        image = std_image(image)
        
        add_feat = 0

        # text
        text = self.data.loc[index, 'text']
        
        # label (test set has labels for our project since challenge closed)
        label = self.data.loc[index, "label"]

        example = (
            img_id,
            image,
            text,
            label,
            add_feat
        )

        return example


    def __len__(self):

        return len(self.data)

class HatefulMemesRawImages(data.Dataset):
    """
    preprocess image and text data to multimodal tensors
    """
    def __init__(
        self,
        json_path,
        img_folder_dir,
        text_model_path,
        balance=False
    ):
        self.data = pd.read_json(json_path, lines = True)
        
        # balance the dataset
        if balance:
            neg = self.data[self.data.label.eq(0)]
            pos = self.data[self.data.label.eq(1)]
            self.data = pd.concat([neg.sample(pos.shape[0]), pos])
        self.data = self.data.reset_index(drop = True)

        self.data['img'] = img_folder_dir +  self.data['img']



    def __getitem__(self, index):

        # get id
        img_id = self.data.loc[index, "id"]
        
        # get image
        
        image = plt.imread(self.data.loc[index, "img"])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image=  cv2.resize(image, (224, 224))

        # text
        text = self.data.loc[index, 'text']
        
        # label (test set has labels for our project since challenge closed)
        label = self.data.loc[index, "label"]

        add_feat = 0

        example = (
            img_id,
            image,
            text,
            label,
            add_feat
        )

        return example


    def __len__(self):

        return len(self.data)


class HatefulMemesRawImagesAdditionalFeat(data.Dataset):
    """
    preprocess image and text data to multimodal tensors
    """
    def __init__(
        self,
        json_path,
        img_folder_dir,
        text_model_path,
        balance=False
    ):
        self.data = pd.read_json(json_path, lines = True)
        
        # balance the dataset
        if balance:
            neg = self.data[self.data.label.eq(0)]
            pos = self.data[self.data.label.eq(1)]
            self.data = pd.concat([neg.sample(pos.shape[0]), pos])
        self.data = self.data.reset_index(drop = True)

        self.data['img'] = img_folder_dir +  self.data['img']



    def __getitem__(self, index):

        # get id
        img_id = self.data.loc[index, "id"]
        
        # get image
        
        image = plt.imread(self.data.loc[index, "img"])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image=  cv2.resize(image, (224, 224))

        # text
        text = self.data.loc[index, 'text']
        
        # label (test set has labels for our project since challenge closed)
        label = self.data.loc[index, "label"]

        add_feat = np.zeros((2))
        add_feat[0] = self.data.loc[index, "minority_flag"]
        add_feat[1] = self.data.loc[index, "gender_flag"]
        
        example = (
            img_id,
            image,
            text,
            label,
            add_feat
        )

        return example


    def __len__(self):

        return len(self.data)

class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count

class CheckpointSaver:
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, device):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir,
                                       f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass

class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]

class RPN:
    """
    RPN for VisualBert
    Adapted from the HuggingFace VisualBert tutorial
    """
    def __init__(self, batch_size, device, cfg_path =  "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"):
        self.device = device
        self.cfg_path = cfg_path
        self.batch_size = batch_size

        # get config
        self.cfg = self.load_config_and_model_weights(self.cfg_path)

        # get model
        self.model = self.get_model(self.cfg)

    def get_embeds(self, images):
        with torch.no_grad():
            N = images.shape[0]
            images, batched_inputs = self.prepare_image_inputs(self.cfg, images, self.model)

            # use resnet to get features
            fpn_features = self.model.backbone(images.tensor)

            # generate proposal regions
            proposals, _ = self.model.proposal_generator(images, fpn_features)

            # get the features and logits
            box_features, features_list = self.get_box_features(self.model, fpn_features, proposals, N)
            pred_class_logits, pred_proposal_deltas = self.get_prediction_logits(self.model, features_list, proposals)
            
            # get boxes
            boxes, scores, image_shapes = self.get_box_scores(self.cfg, pred_class_logits, pred_proposal_deltas, proposals)
            output_boxes = [self.get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]
            
            temp = [self.select_boxes(self.cfg, output_boxes[i], scores[i]) for i in range(len(scores))]
            keep_boxes, max_conf = [],[]
            for keep_box, mx_conf in temp:
                keep_boxes.append(keep_box)
                max_conf.append(mx_conf)     

            keep_boxes = [self.filter_boxes(keep_box, mx_conf) for keep_box, mx_conf in zip(keep_boxes, max_conf)]


        visual_embeds = [box_feature[keep_box.detach()] for box_feature, keep_box in zip(box_features, keep_boxes)]
        for box in visual_embeds:
            box.requires_grad = True

        return visual_embeds

    def load_config_and_model_weights(self, cfg_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)
        return cfg

    def prepare_image_inputs(self, cfg, img_list, model):
        # # Resizing the image according to the configuration
        # transform_gen = T.ResizeShortestEdge(
        #             [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        #         )

        # img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

        # # Convert to C,H,W format
        # convert_to_tensor = lambda x: torch.Tensor(x.astype("float32"))
        batched_inputs = []
        for i in range(img_list.shape[0]):
            img = img_list[i]
            batch_input = {"image": (img).transpose(2, 0), "height": img.shape[0], "width": img.shape[1]}
            batched_inputs.append(batch_input)

        # Normalizing the image
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1).to(self.device)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1).to(self.device)
        normalizer = lambda x: (x - pixel_mean) / pixel_std
        images = [normalizer(x["image"]) for x in batched_inputs]

        # Convert to ImageList
        images =  ImageList.from_tensors(images, model.backbone.size_divisibility)
        
        return images, batched_inputs

    def get_model(self, cfg):
        model = build_model(cfg)

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        # eval mode
        model.eval()
        return model

    def get_box_features(self, model, features, proposals, batch_size):
        features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
        box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head.flatten(box_features)
        box_features = model.roi_heads.box_head.fc1(box_features)
        box_features = model.roi_heads.box_head.fc_relu1(box_features)
        box_features = model.roi_heads.box_head.fc2(box_features)

        box_features = box_features.reshape(batch_size, 1000, 1024) # depends on your config and batch size
        return box_features, features_list

    def get_prediction_logits(self, model, features_list, proposals):
        cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        cls_features = model.roi_heads.box_head(cls_features)
        pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
        return pred_class_logits, pred_proposal_deltas    
    
    def get_box_scores(self, cfg, pred_class_logits, pred_proposal_deltas, proposals):
        box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

        outputs = FastRCNNOutputs(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
        )

        boxes = outputs.predict_boxes()
        scores = outputs.predict_probs()
        image_shapes = outputs.image_shapes

        return boxes, scores, image_shapes    
    
    def get_output_boxes(self, boxes, batched_inputs, image_size):
        proposal_boxes = boxes.reshape(-1, 4).clone()
        scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
        output_boxes = Boxes(proposal_boxes)

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(image_size)

        return output_boxes

    # function to select boxes based on NMS threshold and score threshold
    def select_boxes(self, cfg, output_boxes, scores):
        test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        cls_prob = scores.detach()
        cls_boxes = output_boxes.tensor.detach().reshape(1000,80,4)
        max_conf = torch.zeros((cls_boxes.shape[0])).to(self.device)
        for cls_ind in range(0, cls_prob.shape[1]-1):
            cls_scores = cls_prob[:, cls_ind+1].to(self.device)
            det_boxes = cls_boxes[:,cls_ind,:]
            keep = torch.Tensor(np.array(nms(det_boxes, cls_scores, test_nms_thresh).cpu())).to(torch.long).to(self.device)
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
        keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
        return keep_boxes, max_conf

    def filter_boxes(self, keep_boxes, max_conf, 
                    min_boxes= 10, max_boxes = 100):
        if len(keep_boxes) < min_boxes:
            keep_boxes = torch.argsort(max_conf, descending=True)[:min_boxes]
        elif len(keep_boxes) > max_boxes:
            keep_boxes = torch.argsort(max_conf, descending=True)[:max_boxes]
        return keep_boxes

def make_update_dict(img_ids, preds, scores, labels):
    pred_dict = {}

    for img_id, pred, score, label in zip(img_ids, preds, scores, labels):
        pred_dict[img_id] = (score, pred, label)

    return pred_dict


def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')

def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:1')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    device = "cuda:1"
    gpu_ids = [1]
    return device, gpu_ids


def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.

    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    # device = f"cuda:{gpu_ids[0]}" if gpu_ids else 'cpu'
    device = f"cuda:1" if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model

def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return y_pred_tag, correct_results_sum, acc

# def collate_fn(batch):
#     r"""Puts each data field into a tensor with outer dimension batch size"""

#     error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
#     elem_type = type(batch[0])
#     if isinstance(batch[0], torch.Tensor):
#         out = None
#         if _use_shared_memory:
#             # If we're in a background process, concatenate directly into a
#             # shared memory tensor to avoid an extra copy
#             numel = sum([x.numel() for x in batch])
#             storage = batch[0].storage()._new_shared(numel)
#             out = batch[0].new(storage)
#         return torch.stack(batch, 0, out=out)
#     elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
#             and elem_type.__name__ != 'string_':
#         elem = batch[0]
#         if elem_type.__name__ == 'ndarray':
#             # array of string classes and object
#             if re.search('[SaUO]', elem.dtype.str) is not None:
#                 raise TypeError(error_msg.format(elem.dtype))

#             return torch.stack([torch.from_numpy(b) for b in batch], 0)
#         if elem.shape == ():  # scalars
#             py_type = float if elem.dtype.name.startswith('float') else int
#             return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
#     elif isinstance(batch[0], int_classes):
#         return torch.LongTensor(batch)
#     elif isinstance(batch[0], float):
#         return torch.DoubleTensor(batch)
#     elif isinstance(batch[0], string_classes):
#         return batch
#     elif isinstance(batch[0], container_abcs.Mapping):
#         return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
#     elif isinstance(batch[0], container_abcs.Sequence):
#         transposed = zip(*batch)
#         return [default_collate(samples) for samples in transposed]

#     raise TypeError((error_msg.format(type(batch[0]))))
