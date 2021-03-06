import cv2


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


# class Crop(object):
#     def __init__(self, crop_coeff):
#         self.crop_coeff = crop_coeff

#     def __call__(self, image, bbox, annotation):
#         x, y, w, h = bbox

#         max_size = max(w, h)
#         x_c = x + w / 2
#         y_c = y + h / 2

#         x2 = max_size * self.crop_coeff + x_c
#         y2 = max_size * (self.crop_coeff) + y_c
#         x1 = -max_size * self.crop_coeff + x_c
#         y1 = -max_size * (self.crop_coeff) + y_c

#         crop = image_crop(image, [int(x1), int(y1), int(x2), int(y2)])
#         return crop, bbox, annotation
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


class ValidationAugmentations(object):
    def __init__(self, config):
        self.augment = Compose([
            Crop(crop_coeff=config.dataset.crop_size),
            Resize(size=config.dataset.input_size),
        ])

    def __call__(self, image, bbox, annotation):
        return self.augment(image, bbox, annotation)
