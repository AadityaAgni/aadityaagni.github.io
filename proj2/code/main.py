import utils
from align_image_code import align_images
import numpy as np
import cv2 as cv
import scipy
import os

BASE_PATH = "/Users/aadia/Desktop/Coding/cs180/cs180-proj1/proj2/"

def part_1_1():
    image = cv.imread(BASE_PATH + "data/selfie.png", cv.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    
    # sharpening filter
    filter = np.ones((9, 9), dtype=np.float32) * -1
    filter[4, 4] = 81
    
        
    fourloop_convolved = utils.convolve_4loop(image, filter)
    twoloop_convolved = utils.convolve_2loop(image, filter)
    scipy_convolved = scipy.signal.convolve2d(image, filter, mode='same')
    
    dx = np.array([[1, 0, -1]])
    dy = dx.reshape(3, 1)

    fourloop_dx = utils.convolve_4loop(image, dx)
    twoloop_dx = utils.convolve_2loop(image, dx)
    scipy_dx = scipy.signal.convolve2d(image, dx, mode='same')

    fourloop_dy = utils.convolve_4loop(image, dy)
    twoloop_dy = utils.convolve_2loop(image, dy)
    scipy_dy = scipy.signal.convolve2d(image, dy, mode='same')
    
    os.makedirs(BASE_PATH + "/output/part_1_1/", exist_ok=True)
    
    cv.imwrite(BASE_PATH + "/output/part_1_1/" + "base.png", image)
    
    cv.imwrite(BASE_PATH + "/output/part_1_1/" + "box4loop.png", fourloop_convolved)
    cv.imwrite(BASE_PATH + "/output/part_1_1/" + "box2loop.png", twoloop_convolved)
    cv.imwrite(BASE_PATH + "/output/part_1_1/" + "boxscipy.png", scipy_convolved)
    
    cv.imwrite(BASE_PATH + "/output/part_1_1/" + "dx_fourloop.png", fourloop_dx)
    cv.imwrite(BASE_PATH + "/output/part_1_1/" + "dx_twoloop.png", twoloop_dx)
    cv.imwrite(BASE_PATH + "/output/part_1_1/" + "dx_scipy.png", scipy_dx)
    
    cv.imwrite(BASE_PATH + "/output/part_1_1/" + "dy_fourloop.png", fourloop_dy)
    cv.imwrite(BASE_PATH + "/output/part_1_1/" + "dy_twoloop.png", twoloop_dy)
    cv.imwrite(BASE_PATH + "/output/part_1_1/" + "dy_scipy.png", scipy_dy)

    
def part_1_2():
    image = cv.imread(BASE_PATH + "data/cameraman.png", cv.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    for threshold in [10, 30, 60, 100]:
        dx = np.array([[1, 0, -1]])
        dy = dx.reshape(3, 1)

        convolved_dx = scipy.signal.convolve2d(image, dx, mode='same')
        convolved_dy = scipy.signal.convolve2d(image, dy, mode='same')
        magnitude_image = np.sqrt(convolved_dx**2 + convolved_dy**2)
        magnitude_image[magnitude_image < threshold] = 0
        magnitude_image[magnitude_image >= threshold] = 1


        os.makedirs(BASE_PATH + "/output/part_1_2/", exist_ok=True)
        cv.imwrite(BASE_PATH + "/output/part_1_2/" + f"gradient_magnitude_{threshold}.png", magnitude_image*255)

def part_1_3():
    image = cv.imread(BASE_PATH + "data/cameraman.png", cv.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    threshold = 60

    one_d_gaussian = cv.getGaussianKernel(9, 1)
    gaussian_kernel = one_d_gaussian @ one_d_gaussian.T  
    dx = np.array([[1, 0, -1]])
    dy = dx.reshape(3, 1)
    
    ## Gaussian then Derivative
    convolved_gauss = scipy.signal.convolve2d(image, gaussian_kernel, mode='same')
    convolved_dx = scipy.signal.convolve2d(convolved_gauss, dx, mode='same')
    convolved_dy = scipy.signal.convolve2d(convolved_gauss, dy, mode='same')
    gauss_then_derivative = np.sqrt(convolved_dx**2 + convolved_dy**2)
    
    ## Derivative over Gaussian
    dog_x = scipy.signal.convolve2d(gaussian_kernel, dx, mode='full')
    dog_y = scipy.signal.convolve2d(gaussian_kernel, dy, mode='full')

    dog_dx = scipy.signal.convolve2d(image, dog_x, mode='same')
    dog_dy = scipy.signal.convolve2d(image, dog_y, mode='same')
    derivative_then_gauss = np.sqrt(dog_dx**2 + dog_dy**2)

    # the two routes are the same computation, so this should be ~0.
    difference = np.abs(gauss_then_derivative - derivative_then_gauss)
    print("max difference between the two routes:", difference.max())
    
    os.makedirs(BASE_PATH + "/output/part_1_3/", exist_ok=True)
    cv.imwrite(BASE_PATH + "/output/part_1_3/" + "blurred.png", convolved_gauss)
    cv.imwrite(BASE_PATH + "/output/part_1_3/" + "partial_dx.png", np.abs(convolved_dx))
    cv.imwrite(BASE_PATH + "/output/part_1_3/" + "partial_dy.png", np.abs(convolved_dy))
    cv.imwrite(BASE_PATH + "/output/part_1_3/" + "dog_partial_dx.png", np.abs(dog_dx))
    cv.imwrite(BASE_PATH + "/output/part_1_3/" + "dog_partial_dy.png", np.abs(dog_dy))

    cv.imwrite(BASE_PATH + "/output/part_1_3/" + "gauss_then_derivative.png", gauss_then_derivative)
    cv.imwrite(BASE_PATH + "/output/part_1_3/" + "derivative_then_gauss.png", derivative_then_gauss)
    cv.imwrite(BASE_PATH + "/output/part_1_3/" + "gauss_then_derivative_binarized.png", (gauss_then_derivative >= threshold)*255)
    cv.imwrite(BASE_PATH + "/output/part_1_3/" + "derivative_then_gauss_binarized.png", (derivative_then_gauss >= threshold)*255)

    # upscale with nearest neighbor so each filter cell becomes a solid block, still visibly 9x11
    scaled_dogx = cv.resize(utils.normalize(dog_x) * 255, (0, 0), fx=30, fy=30, interpolation=cv.INTER_NEAREST)
    scaled_dogy = cv.resize(utils.normalize(dog_y) * 255, (0, 0), fx=30, fy=30, interpolation=cv.INTER_NEAREST)

    cv.imwrite(BASE_PATH + "/output/part_1_3/" + f"filter_dogx.png", scaled_dogx)
    cv.imwrite(BASE_PATH + "/output/part_1_3/" + f"filter_dogy.png", scaled_dogy)


def part_2_1():
    one_d_gaussian = cv.getGaussianKernel(9, 1)
    gaussian_kernel = one_d_gaussian @ one_d_gaussian.T

    identity = np.zeros((9, 9))
    identity[4, 4] = 1

    # Run over images
    os.makedirs(BASE_PATH + "/output/part_2_1/", exist_ok=True)

    for name in ["selfie2.jpeg", "taj.jpg", "blurry-chit.png"]:
        image = cv.imread(BASE_PATH + "data/" + name, cv.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)

        stem = name.split(".")[0]
        cv.imwrite(BASE_PATH + "/output/part_2_1/" + f"{stem}_base.png", image)

        for a in [1, 2, 5, 7]:
            sharpen_kernel = (1+a)*identity - (a)*gaussian_kernel
            sharpened = scipy.signal.convolve2d(image, sharpen_kernel, mode='same')
            cv.imwrite(BASE_PATH + "/output/part_2_1/" + f"{stem}_sharpened_{a}.png", sharpened)

    # take a sharp image, blur it ourselves, then try to recover it by sharpening
    image = cv.imread(BASE_PATH + "data/cabo.jpeg", cv.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    blurred = scipy.signal.convolve2d(image, gaussian_kernel, mode='same')

    cv.imwrite(BASE_PATH + "/output/part_2_1/" + "cabo_base.png", image)
    cv.imwrite(BASE_PATH + "/output/part_2_1/" + "cabo_blurred.png", blurred)

    for a in [1, 2, 5, 7]:
        sharpen_kernel = (1+a)*identity - (a)*gaussian_kernel
        unblurred = scipy.signal.convolve2d(blurred, sharpen_kernel, mode='same')
        cv.imwrite(BASE_PATH + "/output/part_2_1/" + f"cabo_unblurred_{a}.png", unblurred)

def part_2_2():
    # (high frequency image, low frequency image, sigma for the high pass, sigma for the low pass)
    pairs = [
        ("selfie.png", "dog.jpeg", 5, 8),
    ]

    os.makedirs(BASE_PATH + "/output/part_2_2/", exist_ok=True)

    for high_name, low_name, high_sigma, low_sigma in pairs:
        high_image = cv.imread(BASE_PATH + "data/" + high_name, cv.IMREAD_GRAYSCALE) / 255.
        low_image = cv.imread(BASE_PATH + "data/" + low_name, cv.IMREAD_GRAYSCALE) / 255.

        high_image, low_image = align_images(high_image, low_image)

        high_gaussian = cv.getGaussianKernel(6*high_sigma+1, high_sigma)
        high_gaussian = high_gaussian @ high_gaussian.T
        low_gaussian = cv.getGaussianKernel(6*low_sigma+1, low_sigma)
        low_gaussian = low_gaussian @ low_gaussian.T

        # high pass is the image minus its own low pass
        high_passed = high_image - scipy.signal.convolve2d(high_image, high_gaussian, mode='same')
        low_passed = scipy.signal.convolve2d(low_image, low_gaussian, mode='same')
        # the selfie is much higher contrast than the dog, so damp it or it wins at every distance
        hybrid = np.clip(0.5*high_passed + low_passed, 0, 1)

        stem = high_name.split(".")[0] + "_" + low_name.split(".")[0]
        cv.imwrite(BASE_PATH + "/output/part_2_2/" + f"{stem}_high_aligned.png", high_image*255)
        cv.imwrite(BASE_PATH + "/output/part_2_2/" + f"{stem}_low_aligned.png", low_image*255)
        cv.imwrite(BASE_PATH + "/output/part_2_2/" + f"{stem}_high_passed.png", (high_passed+0.5)*255)
        cv.imwrite(BASE_PATH + "/output/part_2_2/" + f"{stem}_low_passed.png", low_passed*255)
        cv.imwrite(BASE_PATH + "/output/part_2_2/" + f"{stem}_hybrid.png", hybrid*255)

        for fft_name, fft_image in [("high_aligned", high_image), ("low_aligned", low_image),
                                    ("high_passed", high_passed), ("low_passed", low_passed),
                                    ("hybrid", hybrid)]:
            spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(fft_image))) + 1e-12)
            cv.imwrite(BASE_PATH + "/output/part_2_2/" + f"{stem}_fft_{fft_name}.png", utils.normalize(spectrum)*255)


def main():
    
    print("Running part 1.1")
    # part_1_1()
    
    print("Running part 1.2")
    part_1_2()
    
    print("Running part 1.3")
    part_1_3()

    print("Running part 2.1")
    part_2_1()

    print("Running part 2.2")
    part_2_2()
    
    
    
if __name__ == "__main__":
    main()
    