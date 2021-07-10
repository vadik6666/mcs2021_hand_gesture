## Start

You need install some libraries like `torch`, `torchvision`, e.t.c.
```
pip install -r requirements.txt
```

## Data

You will do work with dataset inside `data` folder.
```
cd data
```

Download [MCS21 dataset (90GB)](https://boosters.pro/championship/machinescansee2021/data/train.zip) and put to `data/train_data/`
so `data/train_data/` will contain 2853 folders (each folder is unique video_id)

```
├── data/train_data/
│   ├── video_id_1
│   │   │── frame_1.jpg
│   │   │── frame_2.jpg
...
│   ├── video_id_2853
│   │   │── frame_1.jpg
│   │   │── frame_2.jpg
```

Now run below scripts, it will automatically download and preprocess a subset of [OpenImages](https://storage.googleapis.com/openimages/web/download.html) dataset and full [100DOH dataset](http://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/download.html). It may take 2-3 hours.
Your data folder shuld be like this:
```
├── data/
│   ├── train_data
│   ├── oid6_hand_yolo_100doh
...
```


```
python prepare_oid6.py
python 100doh_convert_annotation2yolo.py
```

## Hand detector

Hand detector training take at least 8 hours on 4x1080Ti
```
cd ../yolo_detector
python -m torch.distributed.launch --master_port 5216 --nproc_per_node 4 train.py --img 1024 --batch 24 --epochs 10 --data hand_oid6_100doh.yaml --weights yolov5m6.pt --device 0,1,2,3 --project 'runs/train/yolo5m6_1024_2x_100doh' --workers 16
```

## Classication models

Train three classifiers. Training on 1x1080Ti will take at least 10 mins per model.
```
cd ../classifier
python ./main.py --cfg ./config/r18_aug_crop0.7_final_antialias.yml
python ./main.py --cfg ./config/r18_aug_crop0.7_final.yml
python ./main.py --cfg ./config/r18_aug_crop0.6_final.yml
```

## Now prepare submission

Copy trained yolo hand detector and classification models to `./submit` folder
```
cd ..
rsync -avr ./yolo_detector/runs ./submit/ 
rsync -avr ./classifier/experiments/* ./submit/checkpoints/ 
```

Make an archive of the submission
```
cd submit
zip -r mcs21_submit.zip *
```

Submit now ready to upload to [boosters.pro](https://boosters.pro/championship/machinescansee2021/overview)




