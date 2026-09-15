import numpy as np
import cv2
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

MIN_SIZE = 128
INITIAL_RANGE = 40
FINE_RADIUS = 2

def get_schedule(h, w):
    schedule = [(h, w)]
    while min(schedule[-1]) > 2 * MIN_SIZE:
        schedule.append((schedule[-1][0] // 2, schedule[-1][1] // 2))
    return schedule[::-1]

def load_image(filename):
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return F.to_tensor(im).float()


def split_image_into_channels(im):
    height = int(np.floor(im.shape[1] / 3.0))
    b = im[:, :height]
    g = im[:, height: 2*height]
    r = im[:, 2*height: 3*height]
    return b, g, r

def shift_image(im, shift_x, shift_y):
    # Shift the image using torchvision.transforms.functional.affine in torchvision
    return F.affine(im, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)  

def normalize_image(im):
    mean = im.mean()
    std = im.std()
    return (im - mean) / std


def crop_border(im, fraction=0.12):
    _, h, w = im.shape
    dh, dw = int(h * fraction), int(w * fraction)
    return im[:, dh:h - dh, dw:w - dw]

def shift_image_batch(im, shift_x_range, shift_y_range):
    shifted_images = []
    for shift_x in shift_x_range:
        for shift_y in shift_y_range:
            shifted_images.append(shift_image(im, shift_x, shift_y))
    return torch.stack(shifted_images, dim=0)

def get_ranges_from_schedule(index, selected_index=None):
    if index == 0:
        return range(-INITIAL_RANGE, INITIAL_RANGE + 1), range(-INITIAL_RANGE, INITIAL_RANGE + 1)
    s_x, s_y = selected_index
    return (range(2 * s_x - FINE_RADIUS, 2 * s_x + FINE_RADIUS + 1),
            range(2 * s_y - FINE_RADIUS, 2 * s_y + FINE_RADIUS + 1))
