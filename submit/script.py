import yaml
from collections import namedtuple
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn
import face_detection as fd
from torchvision import models
from torchvision import transforms as tfs
from tqdm import tqdm
from augmentations import ValidationAugmentations
import os

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

os.system('pip install antialiased_cnns-0.3-py3-none-any.whl')

def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

def get_best_bbox(bboxes):
    bbox_scores = [x[4] for x in bboxes]
    bbox_index = np.argsort(bbox_scores)[::-1]
    return bboxes[bbox_index]

def filter_bbox(bboxes, SCORE_THR=0.20):
    bbox_scores = [x[4] for x in bboxes]
    bbox_index = np.where(np.array(bbox_scores) > SCORE_THR)
    return bboxes[bbox_index]

def get_highest_bbox(bboxes):
    y_centers = [(x[1]+x[3])/2 for x in bboxes]
    bbox_index = np.argsort(y_centers)
    return bboxes[bbox_index]


def load_resnet(path, config):
    model_type = config.model.model_type
    num_classes = config.dataset.num_of_classes  
    device='cuda'
    
    if model_type == 'resnet34' or model_type == 'torchvision_resnet34':
        model = models.resnet34(pretrained=False)
        model.fc = nn.Sequential(
            nn.Dropout(config.model.drop_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )        
    elif model_type == 'torchvision_resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Sequential(
            nn.Dropout(config.model.drop_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_type == 'torchvision_resnet18_antialias':
        import antialiased_cnns
        model = antialiased_cnns.resnet18(pretrained=False) 
        model.fc = nn.Sequential(
            nn.Dropout(config.model.drop_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )

    elif model_type == 'torchvision_resnet34_2fc':
        model = models.resnet34(pretrained=False)
        model.fc = nn.Sequential(
            nn.Dropout(config.model.drop_rate),
            nn.Linear(model.fc.in_features, 64),
            nn.ReLU(True),
            nn.Dropout(config.model.drop_rate),
            nn.Linear(64, num_classes),
        )

    else:
        raise Exception("Unknown model type: {}".format(model_type))

    model.load_state_dict(torch.load(path, map_location='cpu')["state_dict"])
    model.to(device)
    model.eval()
    return model


def read_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError('Failed to read {}'.format(image_file))
    return img


def rescale_bbox(bbox, change_log):
    x, y, w, h = bbox
    x_offset, y_offset, scale_f = change_log
    x = x - x_offset
    y = y - y_offset

    x = round(x / scale_f)
    y = round(y / scale_f)
    w = round(w / scale_f)
    h = round(h / scale_f)

    return [x, y, w, h]

def save_results(scores, frame_pathes, save_path):
    result_df = pd.DataFrame({
        'no_gesture': scores[:, 0],
        'stop': scores[:, 1],
        'victory': scores[:, 2],
        'mute': scores[:, 3],
        'ok': scores[:, 4],
        'like': scores[:, 5],
        'dislike': scores[:, 6],
        'frame_path': frame_pathes
    })

    result_df.to_csv(save_path, index=False)

class TestDataset(object):
    def __init__(self, image_list, max_resolution=640, stride=32):
        self.image_list = image_list
        self.max_resolution = max_resolution

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        cv2.setNumThreads(6)
        # image_fpath = '../../data/train_data/' + self.image_list[idx]
        image_fpath = self.image_list[idx]
        img = read_image(image_fpath)

        orgl_h, orgl_w = img.shape[:-1]
        max_size = max(orgl_h, orgl_w)
        scale_f = self.max_resolution / max_size

        resized_img = cv2.resize(img, (round(orgl_w * scale_f), round(orgl_h * scale_f)))
        # resized_img = np.array(resized_img)

        new_h, new_w = resized_img.shape[:-1]
        border_arr = np.zeros((self.max_resolution, self.max_resolution, 3), dtype=np.uint8)
        x_offset = (self.max_resolution - new_w) // 2
        y_offset = (self.max_resolution - new_h) // 2

        border_arr[y_offset:y_offset + new_h, x_offset: x_offset + new_w] = resized_img
        border_arr = border_arr.transpose(2, 0, 1)
        border_arr = np.ascontiguousarray(border_arr)
        
        return image_fpath, border_arr#, img



CONFIG_PATH = './configs/r18_aug_crop0.7_final.yml'
MODEL_PATH = './checkpoints/r18_aug_crop0.7_final/model_0000.pth'
CONFIG_PATH2 = './configs/r18_aug_crop0.6_final.yml'
MODEL_PATH2 = './checkpoints/r18_aug_crop0.6_final/model_0000.pth'
CONFIG_PATH3 = './configs/r18_aug_crop0.7_final_antialias.yml'
MODEL_PATH3 = './checkpoints/r18_aug_crop0.7_final_antialias/model_0000.pth'

#r18_aug_crop0.7_final_antialias

INPUT_PATH = 'data/test.csv'
# INPUT_PATH = '../../code/EDA/val_df.csv'
OUT_PATH = './answers.csv'

checkpoint_file = './runs/train/yolo5m6_1024_2x_100doh/exp/weights/best.pt'
device = 'cuda:0'

# BATCH_SIZE = 32
BATCH_SIZE = 4
NUM_WORKERS = 8

def main():
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    config = convert_dict_to_tuple(data)
    with open(CONFIG_PATH2) as f:
        data2 = yaml.safe_load(f)
    config2 = convert_dict_to_tuple(data2)
    with open(CONFIG_PATH3) as f:
        data3 = yaml.safe_load(f)
    config3 = convert_dict_to_tuple(data3)

    # YOLO detector
    conf_thres = 0.20  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 16
    half = True
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False
    yolo_model = attempt_load(checkpoint_file, map_location=device)  # load FP32 model
    stride = int(yolo_model.stride.max())  # model stride
    imgsz = 1280 #640
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # names = yolo_model.module.names if hasattr(yolo_model, 'module') else model.names  # get class names
    if half:
        yolo_model.half()  # to FP16

    
    model = load_resnet(MODEL_PATH, config)
    model2 = load_resnet(MODEL_PATH2, config2)
    model3 = load_resnet(MODEL_PATH3, config3)
    print('\nClassification Model1: ', model)
    print('\nClassification Model2: ', model2)
    print('\nClassification Model3: ', model3)
    softmax_func = torch.nn.Softmax(dim=1)
    val_augs = ValidationAugmentations(config)
    val_augs2 = ValidationAugmentations(config2)
    val_augs3 = ValidationAugmentations(config3)
    preproc = tfs.Compose([tfs.ToTensor(), tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    test_df = pd.read_csv(INPUT_PATH)
    scores = np.zeros((len(test_df), 7), dtype=np.float32)
    scores[:, 0] = 1

    test_dataset = TestDataset(test_df.frame_path.values, max_resolution=imgsz, stride=stride)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=True
    )

    for batch_idx, (image_pathes, batch_images) in tqdm(enumerate(test_loader), total=len(test_loader)):
        batch_images_np = list(np.array(batch_images))

        batch_images = batch_images.to('cuda')
        batch_images = batch_images.half() if half else batch_images.float()
        batch_images /= 255.0

        detections = yolo_model(batch_images, augment=False)[0]
        detections = non_max_suppression(detections, conf_thres, iou_thres, 
            classes, agnostic_nms, max_det=max_det)

        crops = []
        crops2 = []
        crops3 = []
        score_using_indexes = []
        det_cnt = 0
        img2detections_idx = {}
        img_with_good_dets_cnt = 0
        for idx, img_detections in enumerate(detections):
            img_detections = img_detections.detach().cpu().numpy() #(16, 6)
            
            detections = filter_bbox(img_detections)
            detections = get_best_bbox(detections)[:8]
            detections = get_highest_bbox(detections) # (6, 6)

            
            if len(detections) > 0:
                img = batch_images_np[idx].transpose(1, 2, 0)
                H, W = img.shape[:2]

                det_cnt_list = []
                # print(idx, 'detections: ', detections)
                for det in detections[:3]:
                    # print('    det:', det)
                    x1 = max(0, round(det[0]))
                    y1 = max(0, round(det[1]))
                    x2 = min(W, round(det[2]))
                    y2 = min(H, round(det[3]))
                    w = x2 - x1
                    h = y2 - y1
                    max_bbox = [x1, y1, w, h]
                    if w/W < 0.01 or h/H < 0.01:
                        continue
                    crop, *_ = val_augs(img, max_bbox, None)
                    crop2, *_ = val_augs2(img, max_bbox, None)
                    crop3, *_ = val_augs3(img, max_bbox, None)
                    crops.append(preproc(crop))
                    crops2.append(preproc(crop2))
                    crops3.append(preproc(crop3))
                    det_cnt_list.append(det_cnt)
                    det_cnt += 1
                
                if len(det_cnt_list) > 0:
                    img2detections_idx[img_with_good_dets_cnt] = np.array(det_cnt_list)
                    img_with_good_dets_cnt += 1
                    score_using_indexes.append(batch_idx * BATCH_SIZE + idx)

        if len(crops) > 0:
            clf_tensor = torch.stack(crops)
            clf_tensor = clf_tensor.to('cuda')
            with torch.no_grad():
                out = model(clf_tensor)
            out = softmax_func(out).squeeze().detach().cpu().numpy()

            if type(out[0]) == np.float32:
                out = np.array([out])

            clf_tensor2 = torch.stack(crops2)
            clf_tensor2 = clf_tensor2.to('cuda')
            with torch.no_grad():
                out2 = model2(clf_tensor2)
            out2 = softmax_func(out2).squeeze().detach().cpu().numpy()

            if type(out2[0]) == np.float32:
                out2 = np.array([out2])

            clf_tensor3 = torch.stack(crops3)
            clf_tensor3 = clf_tensor3.to('cuda')
            with torch.no_grad():
                out3 = model3(clf_tensor3)
            out3 = softmax_func(out3).squeeze().detach().cpu().numpy()

            if type(out3[0]) == np.float32:
                out3 = np.array([out3])



            # print('\n\n',score_using_indexes, len(crops), 
            #     img2detections_idx, out.shape)
            # print(f'out={out}')
            # print(f'out2={out2}')

            # ensembling two model predictions
            out = np.mean( np.array([out, out2, out3]), axis=0 )

            # print(f'out_mean={out}')
            
            # aggregate preds per each image
            agg_out = np.zeros((len(score_using_indexes), 7))
            for img_idx, det_idx in img2detections_idx.items():
                img_pred = out[det_idx, :]

                # print(img_idx, len(img_pred),' :img_pred:',img_pred)
                if len(img_pred) == 1:
                    agg_img_pred = img_pred
                elif len(img_pred) == 2:
                    if np.argmax(img_pred[0]) == 0 and np.argmax(img_pred[1]) != 0:
                        agg_img_pred = img_pred[1]
                    else:
                        agg_img_pred = img_pred[0]
                elif len(img_pred) == 3:
                    if np.argmax(img_pred[0]) == 0 and np.argmax(img_pred[1]) != 0:
                        agg_img_pred = img_pred[1]
                    elif np.argmax(img_pred[0]) == 0 and np.argmax(img_pred[1]) == 0 and np.argmax(img_pred[2]) != 0:
                        agg_img_pred = img_pred[2]
                    else:
                        agg_img_pred = img_pred[0]


                agg_out[img_idx, :] = agg_img_pred
            # print('agg_out: ', agg_out)
            scores[score_using_indexes] = agg_out

        if batch_idx % 100 == 0 and batch_idx > 0:
            save_results(scores, test_df.frame_path.values, OUT_PATH)

    save_results(scores, test_df.frame_path.values, OUT_PATH)

if __name__ == '__main__':
    main()
