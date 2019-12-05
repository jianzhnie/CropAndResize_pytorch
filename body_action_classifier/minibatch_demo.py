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
import Mydataset


def load_json(json_path):
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    return json_file

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


    def predict_behavior(self, image_torch,  boxes, box_index, crop_height = 112, crop_width=112):
        """
        predict batch images
        """
        crops_torch = CropAndResizeFunction(crop_height, crop_width, 0)(image_torch, boxes, box_index)
        crops_torch = crops_torch.sub_(self.mean).div_(self.std)
        
        outputs = self.model(crops_torch)
        score = self.softmax(outputs)
        pred = score.cpu().data.numpy()
        return pred


def main(batch_size=8):
    "Predicted the whole image bbox"
    print("===> load model")
    estimator = TorchCrop()
    # image_bboxes_info is a dict
    image_bboxes_info = load_json('data/bodybbox.json')
    workdir = os.getcwd()
    imgdir = os.path.join(workdir, 'data/images')
    classes = ['listen','read','hand','stand']
    student_behavior = {}
    ## dataloader
    print ('===> load data')
    datasets = Mydataset.XMCData(imgdir)
    dataloaders = torch.utils.data.DataLoader(datasets, 
                                            batch_size=batch_size, 
                                            shuffle=False,
                                            num_workers=4, 
                                            pin_memory=True)

    with torch.no_grad():
        for batch, (images,imgnames) in enumerate(dataloaders):         
            print("Batch", batch)
            boxes_lst = []
            boxes_index_lst = []
            height = 1080
            width = 1920
            for index, name in enumerate(imgnames):
                bbox_per_img = image_bboxes_info[name]
                boxes_lst_per_img = []
                for bbox in bbox_per_img:
                    x1, y1, x2, y2 = bbox[:4]
                    boxes_lst_per_img.append([y1/height,x1/width,y2/height,x2/width])
                boxes_index_lst_per_img = [index for _ in range(len(bbox_per_img))]
                boxes_lst.extend(boxes_lst_per_img)
                boxes_index_lst.extend(boxes_index_lst_per_img)


            image_data = torch.FloatTensor(images)
            boxes_data = torch.FloatTensor(boxes_lst)
            boxes_index_data = torch.IntTensor(boxes_index_lst)

            image_torch = to_varabile(image_data) # N * C * H * W
            boxes = to_varabile(boxes_data)
            box_index = to_varabile(boxes_index_data)

            preds = estimator.predict_behavior(image_torch,  boxes, box_index)
            indexes = np.argmax(preds, axis=1)
            scores = preds.max(axis=1)
            
            # write the result in json file
            k = 0
            for idx, name in enumerate(imgnames):
                bbox_per_img = image_bboxes_info[name]
                l = len(bbox_per_img)
                index_per_img = indexes[k:(k+l)]
                score_per_img = scores[k:(k+l)]
                k = k+l
                behavior_per_image = []
                for i, bbox in enumerate(bbox_per_img):
                    index_i = int(index_per_img[i])
                    cls_i = classes[index_i]
                    score_i = float(score_per_img[i])
                    behavior_per_image.append([bbox[0], bbox[1], bbox[2],bbox[3], cls_i, index_i, score_i])
                student_behavior[name] = behavior_per_image

        with open('student_behavior_batch.json', 'w',encoding='utf-8') as f:
            json.dump(student_behavior,f, indent=4, ensure_ascii=False)

if __name__ =='__main__':
    main()