import numpy

def convolve_4loop(image: numpy.ndarray, kernel: numpy.ndarray) -> numpy.ndarray:
    kernel_height, kernel_width = kernel.shape
    padded_image = numpy.pad(image, ((kernel_height//2, kernel_height//2), (kernel_width//2, kernel_width//2)), mode='constant')
    output_image = numpy.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    ii = i + ki - kernel_height//2
                    jj = j + kj - kernel_width//2
                    output_image[i, j] += padded_image[i + ki, j + kj] * kernel[kernel_height - 1 - ki, kernel_width - 1 - kj] # flipped kernel
    return output_image

def convolve_2loop(image: numpy.ndarray, kernel: numpy.ndarray) -> numpy.ndarray:
    kernel_height, kernel_width = kernel.shape
    
    kernel = numpy.flip(kernel).reshape(-1)
    
    padded_image = numpy.pad(image, ((kernel_height//2, kernel_height//2), (kernel_width//2, kernel_width//2)), mode='constant')
    output_image = numpy.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_snippet = padded_image[
                i: i + kernel_height,
                j: j + kernel_width
            ].reshape(-1)
            
            output_image[i, j] += numpy.dot(image_snippet, kernel)
    return output_image

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())
