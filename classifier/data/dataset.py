import json
from collections import defaultdict

import cv2
from torchvision import transforms as tfs

from . import augmentations


def read_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError('Failed to read {}'.format(image_file))
    return img


class GestureDataset(object):
    def __init__(self, config, is_train):
        if is_train:
            self.annot_main = config.dataset.train_annotation_main
            self.transforms = augmentations.get_train_aug(config)
        else:
            self.annot_main = config.dataset.val_annotation_main
            self.transforms = augmentations.get_val_aug(config)

        self.preproc = tfs.Compose([tfs.ToTensor(),
                                    tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                                    ])
        self.preproc2 = tfs.Compose([tfs.ToTensor(),
                                    tfs.Normalize(mean=[0,0,0],
                                                  std=[1.0,1.0,1.0])
                                    ])
        self.data = defaultdict(list)
        self.read_annotations()
        self.dataset_length = len(self.data)

    def read_annotations(self):
        with open(self.annot_main) as f:
            data_json = json.load(f)

        self.data = data_json

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        cv2.setNumThreads(6)
        sample = self.data[idx]
        # image = read_image(sample['frame_path'])
        image = read_image(sample['frame_path'][3:])
        crop, bbox, sample = self.transforms(image, sample['bbox'], sample)
        
        crop = self.preproc(crop)

        # crop = self.preproc2(crop)

        # savecrop = crop.permute(1, 2, 0).numpy()*255
        # savecrop = cv2.cvtColor(savecrop, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f'./DEBUG_CROPS/{idx}.jpg', savecrop )
        return crop, sample['label']
