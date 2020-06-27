import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image
from skimage import exposure
from skimage import color, restoration
from scipy.signal import convolve2d as conv2

from keras.preprocessing import image

import scipy.misc
from scipy import misc
from scipy.misc.pilutil import Image

def thresholding(img):
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return img

def ahe(img):
    img = np.asarray(img)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    return img_adapteq

def cs(img):
    p2, p98 = np.percentile(img, (2,98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale


def eh(img):
    img = PIL.ImageEnhance.Contrast(img)
    return img

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

def invert(img):
    im_array = scipy.misc.fromimage(img)
    im_inverse = 255 - im_array
    im_result = scipy.misc.toimage(im_inverse)
    return im_result


def preprocessing(img):
    ## img = rescale_image(img)
    # img = lab_image(img)
    ## img = histogram_equalization(img)
    # img = gaussian_filter(img)
    ## img = wiener_filter(img)
    # img = np.reshape(img, (64,64,3))
    # img = color.rgb2gray(img)
    # img = clahe(img)
    #img = contrastscretching(img)
    #img = contrast_stretching(img)
    img = thresholding(img)
    return img
