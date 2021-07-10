import os
import json

import pandas as pd
import sys
import argparse
import cv2
import torch
# import face_detection as fd

from mmdet.apis import init_detector 
from mmdet.apis import inference_detector

from tqdm import tqdm

# from utils import get_max_bbox
from utils import get_best_bbox, filter_bbox, get_highest_bbox

CLASS_NAME2LABEL_DICT = {
    'no_gesture': 0,
    'stop': 1,
    'victory': 2,
    'mute': 3,
    'ok': 4,
    'like': 5,
    'dislike': 6
}


def main(args: argparse.Namespace) -> None:
    """
    Runs the code for detecting all hands in the images from the data list,
    selecting the highest hand and saving the result to json file. 
    :param args: all parameters necessary for launch
    :return:
    """
    
    config_file = '../../sample_debug_mmdet_full3/configs/retinanet_r50_fpn_1x_coco_hand2.py'
    checkpoint_file = '../../sample_debug_mmdet_full3/exp/hand_fp16_mstr_albu_7ep/epoch_7.pth'
    # config_file = '../../sample_debug_mmdet_full3/configs/retinanet_r50_fpn_1x_coco_arm.py'
    # checkpoint_file = '../../sample_debug_mmdet_full3/exp/arm_fp16_mstr_albu_7ep/epoch_4.pth'
    device = 'cuda:0'
    hand_detector = init_detector(config_file, checkpoint_file, device=device)

    data_df = pd.read_csv(args.data_list)

    data_df = data_df.sample(n=100, random_state=666)

    # result_arr = []
    result_arr = {}
    for idx, frame_path in tqdm(enumerate(data_df.frame_path.values), total=len(data_df)):
        image_path = os.path.join(args.prefix_path, frame_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]

        detections = inference_detector(hand_detector, img)
        detections = filter_bbox(detections[0])
        detections = get_best_bbox(detections)[:6]
        detections = get_highest_bbox(detections)
        # format: x1,y1,x2,y2, s

        if len(detections) > 0:
            det = detections[0]
            # for det in detections:
            #     if (det[2]-det[0]) / W < 0.05 or (det[3]-det[1]) / H < 0.05:
            #         continue
            #     break
            assert len(det) == 5
            x1 = max(0, round(det[0]))
            y1 = max(0, round(det[1]))
            x2 = min(W, round(det[2]))
            y2 = min(H, round(det[3]))
            w = x2 - x1
            h = y2 - y1
            highest_bbox = [x1, y1, w, h]
            class_name = data_df.class_name.iloc[idx]
            item = {
                'frame_path': frame_path,
                'video_name': data_df.video_name.iloc[idx],
                'frame_id': int(data_df.frame_id.iloc[idx]),
                'label': CLASS_NAME2LABEL_DICT.get(class_name),
                'class_name': class_name,
                'bbox': highest_bbox
            }
            # print(item)
            # result_arr.append(item)
            result_arr[idx] = item

    result_arr = list(result_arr.values())
    with open(args.output_json_path, 'w') as fout:
        json.dump(result_arr, fout, indent=4)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_path', type=str, help='Path to directory with data')
    parser.add_argument('--data_list', type=str, default='./train.csv', help='Path to data list file.')
    parser.add_argument('--output_json_path', type=str, default='./bboxes.json', help='Path to output json file.')
    # parser.add_argument('--detector_type', type=str, default="RetinaNetResNet50", help='detector name')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
