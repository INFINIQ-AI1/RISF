import os
import cv2
import json
import torch
import logging
import detectron2
import numpy as np
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler
from risf.dataloader import build_detection_test_loader
import torch.nn
from detectron2.data import MetadataCatalog
from .archs import clip
import copy

logger = logging.getLogger(__name__)

class CMCLIP:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.alpha = self.cfg.TEST.CMCLIP_ALPHA

        self.imagenet_model, self.preprocess = clip.load("ViT-L/14@336px")
        self.imagenet_model.cuda().eval()
        self.input_resolution = self.imagenet_model.visual.input_resolution
        self.context_length = self.imagenet_model.context_length
        self.vocab_size = self.imagenet_model.vocab_size
        self.dataloader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TRAIN[0])
        self.roi_pooler = ROIPooler(output_size=(336, 336), scales=(1,), sampling_ratio=(0), pooler_type="ROIAlignV2")
        self.exclude_cls = self.clsid_filter()
        self.class_vector = self.text_encode()

        

    def text_encode(self):
        dsname = self.cfg.DATASETS.TEST[0]
        
        self.classes = copy.deepcopy(MetadataCatalog.get(dsname).thing_classes)
        novel_id = copy.deepcopy(MetadataCatalog.get(dsname).get("novel_dataset_id_to_contiguous_id"))
        if novel_id != None:
            thing_id = copy.deepcopy(MetadataCatalog.get(dsname).thing_dataset_id_to_contiguous_id)
            self.class_mapper = {thing_id[k]:idx for idx,k in enumerate(novel_id.keys())}
        elif 'voc' in dsname:
            self.class_mapper = {k:idx for idx,k in enumerate(range(15,20))}
        else:
            print('implement class mapper!')
            raise NotImplementedError
        
        prompts = []
        self.classes.append('background')
        for idx, _class in enumerate(self.classes):
            if idx in self.exclude_cls:
                pass
            else:
                prompt = f"a photo of {_class}"
                prompts.append(prompt)
        text_tokens = clip.tokenize(prompts).cuda()
        text_features = self.imagenet_model.encode_text(text_tokens)
        return text_features / text_features.norm(dim = -1, keepdim = True)


    def extract_roi_features(self, img, boxes):
        """
        :param img:
        :param boxes:
        :return:
        """
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device)

        images = [img/255.]
        images = ImageList.from_tensors(images, 0)

        box_features = self.roi_pooler([images.tensor], boxes)

        if len(box_features) == 0:
            return torch.zeros(1,768, dtype=torch.float16).to(self.device), []

        conv_feature = self.imagenet_model.encode_image(box_features)
        return conv_feature, box_features

    def execute_calibration(self, inputs, dts):

        img = cv2.imread(inputs[0]['file_name'])
        img_id = inputs[0]['image_id']

        ileft = (dts[0]['instances'].scores > self.cfg.TEST.CMCLIP_UPPER).sum().detach().cpu().numpy()
        iright = (dts[0]['instances'].scores > self.cfg.TEST.CMCLIP_LOWER).sum().detach().cpu().numpy()
        assert ileft <= iright

        idx = []
        pred_class_list = []
        for i in range(ileft, iright):
            pred_class = int(dts[0]['instances'].pred_classes[i])
            if pred_class in self.exclude_cls:
               pass
            else:
                pred_class_list.append(self.class_mapper[pred_class])
                idx.append(i)
        
        idx = np.array(idx)
        pred_class_list = np.array(pred_class_list)
        boxes = [dts[0]['instances'].pred_boxes[idx]]
        features, box_features = self.extract_roi_features(img, boxes)

        features /=features.norm(dim = -1, keepdim= True)
        simmilarity =  features @self.class_vector.T
        
        score = torch.exp(simmilarity.float()*100)/torch.sum(torch.exp(simmilarity.float()*100), axis = 1).unsqueeze(1)
        dts[0]['instances'].scores[idx] = dts[0]['instances'].scores[idx] * self.alpha + score[range(len(idx)), pred_class_list] * (1 - self.alpha) # range(len(idx)) : already score is sorted in "boxes = [dts[0]['instances'].pred_boxes[idx]]"

        return dts
    
    def clsid_filter(self):
        dsname = self.cfg.DATASETS.TEST[0]
        exclude_ids = []
        if 'test_all' in dsname:
            if 'coco' in dsname:
                exclude_ids = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                               46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                               66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
            elif 'voc' in dsname:
                exclude_ids = list(range(0, 15))
            else:
                raise NotImplementedError
        return exclude_ids


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
