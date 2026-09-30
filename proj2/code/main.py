import utils
from align_image_code import align_images
import numpy as np
import cv2 as cv
import scipy.signal
import os
import time

# resolve paths relative to this file so the project runs from anywhere
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"

def part_1_1():
    image = cv.imread(BASE_PATH + "data/selfie.png", cv.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    
    filter = np.ones((9, 9), dtype=np.float32) / 81

    dx = np.array([[1, 0, -1]])
    dy = dx.reshape(3, 1)

    # warm up scipy before timing anything, so the first measurement does not absorb
    # the one-off cost of importing and initialising scipy.signal
    scipy.signal.convolve2d(image, filter, mode='same')

    results = {}
    for kernel_name, kernel in [("box 9x9", filter), ("dx", dx), ("dy", dy)]:
        start = time.perf_counter()
        fourloop = utils.convolve_4loop(image, kernel)
        fourloop_time = time.perf_counter() - start

        start = time.perf_counter()
        twoloop = utils.convolve_2loop(image, kernel)
        twoloop_time = time.perf_counter() - start

        start = time.perf_counter()
        scipy_result = scipy.signal.convolve2d(image, kernel, mode='same')
        scipy_time = time.perf_counter() - start

        results[kernel_name] = (fourloop, twoloop, scipy_result)

        print(f"{kernel_name:8s} runtime  4loop {fourloop_time:7.3f}s  2loop {twoloop_time:7.3f}s  scipy {scipy_time:.4f}s"
              f"  ({fourloop_time/scipy_time:.0f}x and {twoloop_time/scipy_time:.0f}x slower)")
        print(f"{kernel_name:8s} mse      4loop vs scipy {np.mean((fourloop - scipy_result)**2):.6e}"
              f"   2loop vs scipy {np.mean((twoloop - scipy_result)**2):.6e}")

    fourloop_convolved, twoloop_convolved, scipy_convolved = results["box 9x9"]
    fourloop_dx, twoloop_dx, scipy_dx = results["dx"]
    fourloop_dy, twoloop_dy, scipy_dy = results["dy"]

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

    dx = np.array([[1, 0, -1]])
    dy = dx.reshape(3, 1)

    convolved_dx = scipy.signal.convolve2d(image, dx, mode='same')
    convolved_dy = scipy.signal.convolve2d(image, dy, mode='same')
    magnitude_image = np.sqrt(convolved_dx**2 + convolved_dy**2)

    os.makedirs(BASE_PATH + "/output/part_1_2/", exist_ok=True)
    cv.imwrite(BASE_PATH + "/output/part_1_2/" + "partial_dx.png", np.abs(convolved_dx))
    cv.imwrite(BASE_PATH + "/output/part_1_2/" + "partial_dy.png", np.abs(convolved_dy))
    cv.imwrite(BASE_PATH + "/output/part_1_2/" + "gradient_magnitude.png", magnitude_image)

    for threshold in [10, 30, 60, 100]:
        cv.imwrite(BASE_PATH + "/output/part_1_2/" + f"gradient_magnitude_{threshold}.png",
                   (magnitude_image >= threshold)*255)

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

    for name in ["selfie2.jpeg", "taj.jpg", "dcmp-chit.jpeg"]:
        image = cv.imread(BASE_PATH + "data/" + name, cv.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)

        stem = name.split(".")[0]
        blurred = scipy.signal.convolve2d(image, gaussian_kernel, mode='same')
        cv.imwrite(BASE_PATH + "/output/part_2_1/" + f"{stem}_base.png", image)
        cv.imwrite(BASE_PATH + "/output/part_2_1/" + f"{stem}_blurred.png", blurred)
        cv.imwrite(BASE_PATH + "/output/part_2_1/" + f"{stem}_highfreq.png", np.abs(image - blurred))

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
    pairs = [
        ("selfie_dog", "selfie.png", "dog.jpeg", 6, 8, 0.8),
        ("selfie_passport", "selfie.png", "passport.JPG", 4, 10, 2.5),
        ("nutmeg_DerekPicture", "nutmeg.jpg", "DerekPicture.jpg", 4, 8, 4.5),
        # the same cat pair with derek's eyes marked up at his hairline, so the alignment
        # translates the cat off his face. kept as a worked example of what goes wrong.
        ("nutmeg_misaligned", "nutmeg.jpg", "DerekPicture.jpg", 4, 8, 4.5),
    ]

    points = {
        "selfie_dog":
            ((222.0, 292.0), (318.0, 292.0),
             (163.0, 272.0), (372.0, 275.0)),
        "selfie_passport":
            ((222.0, 292.0), (318.0, 292.0),
             (592.0, 700.0), (885.0, 700.0)),
        "nutmeg_DerekPicture":
            ((585.0, 290.0), (745.0, 368.0),
             (250.0, 337.0), (437.0, 333.0)),
        "nutmeg_misaligned":
            ((585.0, 290.0), (745.0, 368.0),
             (250.0, 180.0), (437.0, 176.0)),
    }

    os.makedirs(BASE_PATH + "/output/part_2_2/", exist_ok=True)

    for stem, high_name, low_name, high_sigma, low_sigma, high_weight in pairs:
        high_image = cv.imread(BASE_PATH + "data/" + high_name, cv.IMREAD_GRAYSCALE) / 255.
        low_image = cv.imread(BASE_PATH + "data/" + low_name, cv.IMREAD_GRAYSCALE) / 255.

        high_image = (high_image - high_image.mean()) / high_image.std()
        low_image = (low_image - low_image.mean()) / low_image.std()

        high_image, low_image = align_images(high_image, low_image, points.get(stem, None))

        # trim the outermost pixels
        crop = 10
        high_image = high_image[crop:-crop, crop:-crop]
        low_image = low_image[crop:-crop, crop:-crop]

        high_gaussian = cv.getGaussianKernel(6*high_sigma+1, high_sigma)
        high_gaussian = high_gaussian @ high_gaussian.T
        low_gaussian = cv.getGaussianKernel(6*low_sigma+1, low_sigma)
        low_gaussian = low_gaussian @ low_gaussian.T

        # high pass is the image minus its own low pass
        high_passed = high_image - scipy.signal.convolve2d(high_image, high_gaussian, mode='same')
        low_passed = scipy.signal.convolve2d(low_image, low_gaussian, mode='same')

        hybrid = high_weight*high_passed + low_passed

        out = BASE_PATH + "/output/part_2_2/" + stem + "/"
        os.makedirs(out, exist_ok=True)

        # everything is in standard deviations now, so rescale to 0-1 just for writing
        cv.imwrite(out + "high_aligned.png", utils.normalize(high_image)*255)
        cv.imwrite(out + "low_aligned.png", utils.normalize(low_image)*255)
        cv.imwrite(out + "high_passed.png", utils.normalize(high_passed)*255)
        cv.imwrite(out + "low_passed.png", utils.normalize(low_passed)*255)
        cv.imwrite(out + "hybrid.png", utils.normalize(hybrid)*255)

        for fft_name, fft_image in [("high_aligned", high_image), ("low_aligned", low_image),
                                    ("high_passed", high_passed), ("low_passed", low_passed),
                                    ("hybrid", hybrid)]:
            spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(fft_image))) + 1e-12)
            cv.imwrite(out + f"fft_{fft_name}.png", utils.normalize(spectrum)*255)


def build_stacks(image, levels):
    gaussian_stack = [image]
    for i in range(levels - 1):
        sigma = 2 * 2**i
        kernel = cv.getGaussianKernel(6*sigma + 1, sigma)
        kernel = kernel @ kernel.T
        if image.ndim == 3:
            blurred = np.dstack([scipy.signal.convolve2d(image[:, :, c], kernel, mode='same', boundary='symm')
                                 for c in range(3)])
        else:
            blurred = scipy.signal.convolve2d(image, kernel, mode='same', boundary='symm')
        gaussian_stack.append(blurred)

    laplacian_stack = [gaussian_stack[i] - gaussian_stack[i+1] for i in range(levels - 1)]
    laplacian_stack.append(gaussian_stack[-1])
    return gaussian_stack, laplacian_stack


def part_2_3():
    out = BASE_PATH + "/output/part_2_3/"
    os.makedirs(out, exist_ok=True)

    levels = 5

    apple = cv.imread(BASE_PATH + "data/apple.jpeg").astype(np.float32) / 255.
    orange = cv.imread(BASE_PATH + "data/orange.jpeg").astype(np.float32) / 255.

    # vertical seam down the middle. 1 keeps the apple, 0 keeps the orange.
    mask = np.zeros(apple.shape[:2], dtype=np.float32)
    mask[:, :apple.shape[1]//2] = 1.

    gaussians = {}
    laplacians = {}
    for name, image in [("apple", apple), ("orange", orange), ("mask", mask)]:
        gaussian_stack, laplacian_stack = build_stacks(image, levels)
        gaussians[name] = gaussian_stack
        laplacians[name] = laplacian_stack

        for i in range(levels):
            cv.imwrite(out + f"{name}_gaussian_{i}.png", np.clip(gaussian_stack[i], 0, 1) * 255)
            cv.imwrite(out + f"{name}_laplacian_{i}.png", utils.normalize(laplacian_stack[i]) * 255)

    # blend each band separately, using the mask blurred by the same amount as that band.
    # a sharp band gets a sharp seam, a blurry band gets a wide one.
    blended = []
    for i in range(levels):
        m = gaussians["mask"][i][:, :, None]
        blended.append(m * laplacians["apple"][i] + (1 - m) * laplacians["orange"][i])

    # recreating figure
    panels = []
    for i in [0, 2, 4]:
        m = gaussians["mask"][i][:, :, None]
        panels.append((f"apple_level_{i}", m * laplacians["apple"][i], False))
        panels.append((f"orange_level_{i}", (1 - m) * laplacians["orange"][i], False))
        panels.append((f"blend_level_{i}", blended[i], False))

    apple_side = sum(gaussians["mask"][i][:, :, None] * laplacians["apple"][i] for i in range(levels))
    orange_side = sum((1 - gaussians["mask"][i][:, :, None]) * laplacians["orange"][i] for i in range(levels))
    panels.append(("apple_collapsed", apple_side, True))
    panels.append(("orange_collapsed", orange_side, True))
    panels.append(("oraple", sum(blended), True))

    for letter, (name, panel, is_image) in zip("abcdefghijkl", panels):
        # the collapsed row is a real image already, the band rows need stretching
        panel = np.clip(panel, 0, 1) if is_image else utils.normalize(panel)
        cv.imwrite(out + f"{letter}_{name}.png", panel * 255)

    # the stacks should add back up to what we started with
    print("apple stack reconstruction error:", np.abs(sum(laplacians["apple"]) - apple).max())



def part_2_4():
    levels = 5

    # a mask of 1 keeps the first image, 0 keeps the second. the oval is given as
    # centre and radii in fractions of the image, so it follows whatever size we feed it.
    blends = [
        ("oraple", "apple.jpeg", "orange.jpeg", "vertical", None),
        ("dog_me", "selfie_face.jpeg", "dog_face.jpeg", "oval", (0.471, 0.400, 0.240, 0.120)),
        ("day_night", "day_crop.jpeg", "night_crop.jpeg", "vertical", None),
    ]

    for stem, first_name, second_name, mask_shape, oval in blends:
        first = cv.imread(BASE_PATH + "data/" + first_name).astype(np.float32) / 255.
        second = cv.imread(BASE_PATH + "data/" + second_name).astype(np.float32) / 255.
        height, width = first.shape[:2]

        mask = np.zeros((height, width), dtype=np.float32)
        if mask_shape == "vertical":
            mask[:, :width//2] = 1.
        else:
            # an oval cut around my face so it lands inside the dog's head. a straight
            # seam could never cut this shape out.
            cx, cy, ax, ay = oval
            cv.ellipse(mask, (int(cx*width), int(cy*height)),
                       (int(ax*width), int(ay*height)), 0, 0, 360, 1., -1)

        first_gaussian, first_laplacian = build_stacks(first, levels)
        second_gaussian, second_laplacian = build_stacks(second, levels)
        mask_gaussian, mask_laplacian = build_stacks(mask, levels)

        out = BASE_PATH + "/output/part_2_4/" + stem + "/"
        os.makedirs(out, exist_ok=True)

        cv.imwrite(out + "first.png", first * 255)
        cv.imwrite(out + "second.png", second * 255)
        cv.imwrite(out + "mask.png", mask * 255)

        # blend each band with the mask blurred by the same amount as that band, so a sharp
        # band gets a sharp seam and a blurry band gets a wide one. this is the whole trick.
        blended = []
        for i in range(levels):
            m = mask_gaussian[i][:, :, None]
            first_band = m * first_laplacian[i]
            second_band = (1 - m) * second_laplacian[i]
            blended.append(first_band + second_band)

            cv.imwrite(out + f"first_masked_{i}.png", utils.normalize(first_band) * 255)
            cv.imwrite(out + f"second_masked_{i}.png", utils.normalize(second_band) * 255)
            cv.imwrite(out + f"blended_{i}.png", utils.normalize(blended[i]) * 255)
            cv.imwrite(out + f"mask_gaussian_{i}.png", mask_gaussian[i] * 255)

        # collapsing each side's masked stack on its own gives the last row of figure 10
        first_side = sum(mask_gaussian[i][:, :, None] * first_laplacian[i] for i in range(levels))
        second_side = sum((1 - mask_gaussian[i][:, :, None]) * second_laplacian[i] for i in range(levels))
        cv.imwrite(out + "first_collapsed.png", np.clip(first_side, 0, 1) * 255)
        cv.imwrite(out + "second_collapsed.png", np.clip(second_side, 0, 1) * 255)
        cv.imwrite(out + "blend.png", np.clip(sum(blended), 0, 1) * 255)

        # what a hard cut with no blending looks like, for comparison
        m = mask[:, :, None]
        cv.imwrite(out + "hard_seam.png", np.clip(m*first + (1-m)*second, 0, 1) * 255)


def main():
    
    print("Running part 1.1")
    part_1_1()
    
    print("Running part 1.2")
    part_1_2()
    
    print("Running part 1.3")
    part_1_3()

    print("Running part 2.1")
    part_2_1()

    print("Running part 2.2")
    part_2_2()

    print("Running part 2.3")
    part_2_3()

    print("Running part 2.4")
    part_2_4()
    
    
    
if __name__ == "__main__":
    main()
    