import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import utils

import argparse

def align_at_res(image, reference, shift_x_range, shift_y_range):
    # Normalize images
    image = utils.normalize_image(image)
    reference = utils.normalize_image(reference)
    
    # Shift the image in all possible ways and compute the sum of squared differences with the reference
    shifted_images = utils.shift_image_batch(image, shift_x_range, shift_y_range)
    cropped = torch.stack([utils.crop_border(s) for s in shifted_images])
    
    # Euclidean Dist
    dist = torch.sum((utils.crop_border(reference) - cropped) ** 2, dim=(1, 2, 3))
    
    # Find the index of the minimum SSD
    min_index = torch.argmin(dist)
    
    # Convert the index back to shift_x and shift_y
    num_shifts_y = len(shift_y_range)
    shift_x = shift_x_range[min_index // num_shifts_y]
    shift_y = shift_y_range[min_index % num_shifts_y]
    
    return shift_x, shift_y

def align_single_scale(image, reference, radius=15):
    shift = align_at_res(image, reference, range(-radius, radius + 1), range(-radius, radius + 1))
    print(f"Displacement (x, y): {shift}")
    return utils.shift_image(image, shift[0], shift[1])

def align_single_channel(image, reference):
    schedule = utils.get_schedule(image.shape[1], image.shape[2])
    shift = (0, 0)
    for i, size in enumerate(schedule):
        downsampled_image = F.interpolate(image.unsqueeze(0), size=size, mode='bilinear', align_corners=False, antialias=True).squeeze(0)
        downsampled_reference = F.interpolate(reference.unsqueeze(0), size=size, mode='bilinear', align_corners=False, antialias=True).squeeze(0)
        shift_x_range, shift_y_range = utils.get_ranges_from_schedule(i, shift)
        shift = align_at_res(downsampled_image, downsampled_reference, shift_x_range, shift_y_range)
    print(f"Displacement (x, y): {shift}")
    return utils.shift_image(image, shift[0], shift[1])

def main(imname='./cathedral.jpg', single_scale=False):
    # Load the grayscale image
    print(f"Loading image: {imname}")
    im = utils.load_image(imname)

    # Split the image into color channels
    print("Splitting image into color channels...")
    b, g, r = utils.split_image_into_channels(im)

    print("Image split into color channels.")

    # Align the green and red channels to the blue channel
    print("Aligning channels...")
    align = align_single_scale if single_scale else align_single_channel
    ag = align(g, b)
    ar = align(r, b + ag / 2)  # Align red channel to the average of blue and aligned green channels
    
    # Create a color image by stacking the aligned channels
    H, W = b.shape[1], b.shape[2]
    im_out = torch.stack([ar, ag, b], dim=0).reshape(-1, H, W) # stack along first dim 
    print("im out shape:", im_out.shape)
    
    # Convert the image to numpy for visualization
    im_out_np = im_out.permute(1, 2, 0).numpy()
    print("Converting image to numpy array...")

    # Save the image with cv2
    print("Saving the colorized image...")
    out_uint8 = np.clip(im_out_np * 255.0, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
    fname = './out_fname.jpg'
    cv2.imwrite(fname, out_bgr)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align color channels of a grayscale image.')
    parser.add_argument('--imname', type=str, default='./church.tif', help='Path to the input grayscale image.')
    parser.add_argument('--single-scale', action='store_true', help='Exhaustive search at full resolution (no pyramid).')
    args = parser.parse_args()
    main(args.imname, args.single_scale)
