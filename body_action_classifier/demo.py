import os, json
import numpy as np
from PIL import Image
import torch_cropAndResize
import time

def load_json(json_path):
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    return json_file


def main():
    "Predicted the whole image bbox"
    estimator = torch_cropAndResize.TorchCrop()
    # image_bboxes_info is a dict
    image_bboxes_info = load_json('data/bodybbox.json')
    workdir = os.getcwd()
    imgdir = os.path.join(workdir, 'data/images')
    classes = ['listen','read','hand','stand']
    student_behavior = {}
    count = 0
    for image_name in image_bboxes_info:
        count += 1
        print('useimage count %d' % count)
        image_path = os.path.join(imgdir, image_name)
        outimage = image_path.split('/')[-1]
        image = Image.open(image_path)
        bboxes = image_bboxes_info[image_name]
        preds = estimator.predict_behavior(image, bboxes)
        index = np.argmax(preds, axis=1)
        score = preds.max(axis=1)
        behavior_per_image = []
        for i, bbox in enumerate(bboxes):
            index_i = int(index[i])
            cls_i = classes[index_i]
            score_i = float(score[i])
            behavior_per_image.append([bbox[0], bbox[1], bbox[2],bbox[3], cls_i, index_i, score_i])
        student_behavior[outimage] = behavior_per_image
    with open('student_behavior.json', 'w',encoding='utf-8') as f:
        json.dump(student_behavior,f, indent=4, ensure_ascii=False) 


if __name__ =='__main__':
    main()
