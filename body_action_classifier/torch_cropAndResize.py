import os
import sys
import json
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import  models
from torchvision import transforms, utils
from torch.autograd import Variable
sys.path.append(os.path.join(os.path.dirname(__file__), '../CropAndResize.pytorch'))
from roi_align.crop_and_resize import CropAndResizeFunction
import time


def to_varabile(tensor, requires_grad=False, is_cuda=True):
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


class TorchCrop(object):
    def __init__(self):
        self.model = self.initialize_model('mobilenet_v2', 4, use_pretrained=False)
        self.checkpoint = torch.load(os.path.join(os.path.dirname(__file__),'XMC2-Cls_body_action_classifier.pth.tar'),map_location='cuda:0')
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.softmax = nn.Softmax(dim=1).cuda()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.model.cuda()
        self.model.eval()

    def initialize_model(self, model_name, num_classes, use_pretrained=False):
        """
        Initialize these variables which will be set in this if statement. Each of these
        variables is model specific.
        """
        model_ft = None
        if model_name == "resnet18":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)

        elif model_name == "mobilenet_v2":
            """ Mobilenetv2
            """
            model_ft = models.mobilenet_v2(pretrained=use_pretrained)
            num_ftrs = model_ft.last_channel
            model_ft.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, num_classes),
            )
        return model_ft
  

    def predict_behavior(self, image, bboxes, crop_height = 112, crop_width=112):
        width, height = image.size
        boxes_data = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            boxes_data.append([y1/height,x1/width,y2/height,x2/width])

        image = np.array(image, dtype = np.float32)
        image = torch.from_numpy(image.transpose((2,0,1)))
        image_data = image.unsqueeze(0)
        image_data = torch.FloatTensor(image_data)

        boxes_data = torch.FloatTensor(boxes_data)
        boxes_index_data = torch.IntTensor([0 for _ in range(len(bboxes))])

        image_torch = to_varabile(image_data) # N * C * H * W
        boxes = to_varabile(boxes_data)
        box_index = to_varabile(boxes_index_data)

        crops_torch = CropAndResizeFunction(crop_height, crop_width, 0)(image_torch, boxes, box_index)
        crops_torch = crops_torch.sub_(self.mean).div_(self.std)

        outputs = self.model(crops_torch)
        score = self.softmax(outputs)

        pred = score.cpu().data.numpy()
        return pred


