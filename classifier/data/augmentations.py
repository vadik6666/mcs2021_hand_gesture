import cv2
import albumentations as albu


def image_crop(img, bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=(0, 0, 0))
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bbox, annotation):
        for t in self.transforms:
            img, bbox, annotation = t(img, bbox, annotation)
        return img, bbox, annotation


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, bbox, annotation):
        image = cv2.resize(image, (self.size, self.size))
        return image, bbox, annotation


class Crop(object):
    def __init__(self, crop_coeff):
        self.crop_coeff = crop_coeff

    def __call__(self, image, bbox, annotation, extra_y_scale=1.2):
        x, y, w, h = bbox

        max_size = max(w, h)
        x_c = x + w / 2
        y_c = y + h / 2

        x2 = max_size * self.crop_coeff + x_c
        y2 = extra_y_scale * max_size * (self.crop_coeff) + y_c
        x1 = -max_size * self.crop_coeff + x_c
        y1 = -extra_y_scale * max_size * (self.crop_coeff) + y_c

        crop = image_crop(image, [int(x1), int(y1), int(x2), int(y2)])
        return crop, bbox, annotation


class PreparedAug(object):
    def __init__(self):
        augs = [
            albu.HorizontalFlip(p=0.5),
            albu.Rotate(limit=10, p=0.5),
            albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=50, p=0.5),
            albu.GaussianBlur(p=0.4),
            albu.ToGray(p=0.3)
        ]
        self.augs = albu.Compose(augs)

    def __call__(self, image, bbox, annotation):
        image = self.augs(image=image)['image']
        return image, bbox, annotation

class PreparedAug2(object):
    def __init__(self):
        augs = [
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=0.3),
            albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=50, p=0.3),
            
            albu.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
            
            albu.MotionBlur(blur_limit=3, p=0.3),

            albu.GaussianBlur(p=0.3),
            albu.ToGray(p=0.3)
        ]
        self.augs = albu.Compose(augs)

    def __call__(self, image, bbox, annotation):
        image = self.augs(image=image)['image']
        return image, bbox, annotation

class PreparedAug2b(object):
    def __init__(self):
        augs = [
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=10, p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=0.3),
            albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=50, p=0.3),
            
            albu.ImageCompression(quality_lower=40, quality_upper=95, p=0.2),
            
            albu.MotionBlur(blur_limit=11, p=0.3),

            albu.GaussianBlur(p=0.3),
            albu.ToGray(p=0.3)
        ]
        self.augs = albu.Compose(augs)

    def __call__(self, image, bbox, annotation):
        image = self.augs(image=image)['image']
        return image, bbox, annotation

class PreparedAug3(object):
    def __init__(self, config):
        augs = [
            albu.HorizontalFlip(p=0.5),
            albu.ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            albu.GaussNoise(p=0.1),
            albu.OneOf([
                albu.GaussianBlur(blur_limit=[3, 9], p=0.5),
                albu.MotionBlur(blur_limit=[3, 9], p=0.5),
                albu.MedianBlur(blur_limit=[3, 9], p=0.5)], 
                p=0.4),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                albu.FancyPCA(),
                albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=50)], 
                p=0.7),
            albu.ToGray(p=0.3),
            albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=10, 
                border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        self.augs = albu.Compose(augs)

    def __call__(self, image, bbox, annotation):
        image = self.augs(image=image)['image']
        return image, bbox, annotation

class PreparedAug4(object):
    def __init__(self, config):
        augs = [
            albu.HorizontalFlip(p=0.5),
            albu.ImageCompression(quality_lower=20, quality_upper=100, p=0.3),
            albu.GaussNoise(p=0.1),
            albu.OneOf([
                albu.GaussianBlur(blur_limit=9, p=0.5),
                albu.MedianBlur(blur_limit=5, p=0.5)], 
                p=0.4),
            albu.MotionBlur(blur_limit=13, p=0.3),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                albu.FancyPCA(),
                albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=50)], 
                p=0.7),
            albu.ToGray(p=0.3),
            albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=(-0.1,0.3), rotate_limit=10, 
                border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        self.augs = albu.Compose(augs)

    def __call__(self, image, bbox, annotation):
        image = self.augs(image=image)['image']
        return image, bbox, annotation


class DefaultAugmentations(object):
    def __init__(self, config):
        self.augment = Compose([
            Crop(crop_coeff=config.dataset.crop_size),
            Resize(size=config.dataset.input_size),
            # PreparedAug(),
            PreparedAug2(),
        ])

    def __call__(self, image, bbox, annotation):
        return self.augment(image, bbox, annotation)

class DefaultAugmentations_b(object):
    def __init__(self, config):
        self.augment = Compose([
            Crop(crop_coeff=config.dataset.crop_size),
            Resize(size=config.dataset.input_size),
            # PreparedAug(),
            PreparedAug2b(),
        ])

    def __call__(self, image, bbox, annotation):
        return self.augment(image, bbox, annotation)

class DefaultAugmentations3(object):
    def __init__(self, config):
        self.augment = Compose([
            Crop(crop_coeff=config.dataset.crop_size),
            Resize(size=config.dataset.input_size),
            PreparedAug3(config)
        ])

    def __call__(self, image, bbox, annotation):
        return self.augment(image, bbox, annotation)

class DefaultAugmentations4(object):
    def __init__(self, config):
        self.augment = Compose([
            Crop(crop_coeff=config.dataset.crop_size),
            Resize(size=config.dataset.input_size),
            PreparedAug4(config)
        ])

    def __call__(self, image, bbox, annotation):
        return self.augment(image, bbox, annotation)


class ValidationAugmentations(object):
    def __init__(self, config):
        self.augment = Compose([
            Crop(crop_coeff=config.dataset.crop_size),
            Resize(size=config.dataset.input_size),
        ])

    def __call__(self, image, bbox, annotation):
        return self.augment(image, bbox, annotation)


def get_train_aug(config):
    if config.dataset.augmentations == 'default':
        print('DefaultAugmentations')
        train_augs = DefaultAugmentations(config)
    elif config.dataset.augmentations == 'default_b':
        print('DefaultAugmentations #2b')
        train_augs = DefaultAugmentations_b(config)
    elif config.dataset.augmentations == 'default3':
        print('DefaultAugmentations #3')
        train_augs = DefaultAugmentations3(config)
    elif config.dataset.augmentations == 'default4':
        print('DefaultAugmentations #4')
        train_augs = DefaultAugmentations4(config)
    else:
        raise Exception("Unknonw type of augs: {}".format(config.dataset.augmentations))
    return train_augs


def get_val_aug(config):
    if config.dataset.augmentations_valid == 'default':
        val_augs = ValidationAugmentations(config)
    else:
        raise Exception("Unknonw type of augs: {}".format(config.dataset.augmentations))
    return val_augs
