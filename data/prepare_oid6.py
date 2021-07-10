import os
import sys
import pandas as pd
from tqdm import tqdm

#############
SAVE_DIR = 'oid6_hand_yolo_100doh'
target_cl = '/m/0k65p' # human hand
#############

I prepared image list by myself. It contains hands and small amount of not hands images (for background images)
should be less than 1 hour
print('\n\ndownload images given image list')
for split in ['train', 'val']:
	os.makedirs(f'{SAVE_DIR}/images/{split}', exist_ok=True)

	os.system(f"python downloader.py oid_{split}_imagelist.csv --download_folder={SAVE_DIR}/images/{split} --num_processes=4")
	

print('\n\ndownload original annotation')
os.system(f'wget -c https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv')
os.system(f'wget -c https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv')


print('\n\nconvert it to yolo format')
for split in ['train', 'val']:
	os.makedirs(f'{SAVE_DIR}/labels/{split}', exist_ok=True)

	if split == 'train':
		df = pd.read_csv(f'oidv6-train-annotations-bbox.csv')
		target_df = df.loc[df['LabelName'] == target_cl, :]
	else:
		df = pd.read_csv(f'validation-annotations-bbox.csv')
		target_df = df.loc[df['LabelName'] == target_cl, :]

	# print(split, ', df:', df.shape, ', target_df:', target_df.shape)
	# print(target_df.head(5))
	# print(target_df.groupby(by='ImageID'))
	
	for image_id, ann in tqdm(target_df.groupby(by='ImageID')):
	    name = image_id + '.txt'
	    ann_name = f'{SAVE_DIR}/labels/{split}/{name}' 

	    with open(ann_name, 'w') as f:
	        for _, row in ann.iterrows():
	            label = 0
	            x_center = round((row['XMax'] + row['XMin']) / 2, 6)
	            y_center = round((row['YMax'] + row['YMin']) / 2, 6)
	            width = round(row['XMax'] - row['XMin'], 6)
	            height = round(row['YMax'] - row['YMin'], 6)
	            f.write(f'{label} {x_center} {y_center} {width} {height}\n')
	    # break
