import os
from tqdm import tqdm
from shutil import copyfile
import json

print(f'\n\ndownloading may take 1-2 hours')
os.system(f'wget -c http://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/downloads/raw.zip')
# copyfile('../../../data/100DOH/raw.zip', 'raw.zip')

os.system(f'wget -c http://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/downloads/file.zip')

print(f'\n\nunpacking')
os.system(f'unzip -qq raw.zip')
os.system(f'unzip -qq file.zip')

IMG_SRC = './raw'
ANN_SRC = './file/trainval.json'
SAVE_YOLO_ANN_PATH = 'oid6_hand_yolo_100doh/labels/train'
SAVE_IMG_PATH = 'oid6_hand_yolo_100doh/images/train'


with open(ANN_SRC, 'r') as f:
    annotations = json.load(f)
print(f'\n\n{len(annotations)} images with annotations in {ANN_SRC}\n\n')

print(f'\n\nCopy images from individual folders in {IMG_SRC} to single folder:{SAVE_IMG_PATH}...')
for cnt, (name, ann) in tqdm(enumerate(annotations.items())):
	im_path = os.path.join(IMG_SRC, name)
	short_name = im_path.split('/')[-1]
	dst_im_path = os.path.join(SAVE_IMG_PATH, short_name)

	copyfile(im_path, dst_im_path)

	save_ann_name = os.path.join(SAVE_YOLO_ANN_PATH, short_name.split('.jpg')[0] + '.txt')
	with open(save_ann_name, 'w') as f_yolo:
		for bbox in ann:
			x1 = bbox['x1']
			x2 = bbox['x2']
			y1 = bbox['y1']
			y2 = bbox['y2']

			cl = 0
			x_c = (x1 + x2) / 2
			y_c = (y1 + y2) / 2
			w = (x2 - x1)
			h = (y2 - y1)

			f_yolo.write(f'{cl} {x_c} {y_c} {w} {h}\n')
			
print(f'\n\ndeleting files and folders downloaded before')
os.system(f'rm -rf file file.zip raw raw.zip validation-annotations-bbox.csv oidv6-train-annotations-bbox.csv')
