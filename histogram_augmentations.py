from __future__ import print_function

import os
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from skimage import data, img_as_float
from skimage import exposure


# Histogram Stretching
# Three types:  Histogram Equalization, Contrast Stretching, Adaptive Equalization

### this is a place holder that gets a random image from file
bg_pics_dir = "/home/iolie/PycharmProjects/keras__practice/number_checker/lfw_mix"
piclist = []
for dirpath, subdirs, files in os.walk(bg_pics_dir):
    for f in files:
        if f.endswith('.jpg'):
            piclist.append(os.path.join(dirpath, f))
im = Image.open(piclist[random.randint(0, len(piclist)-1)]).convert('RGBA')
im.show()

arr = np.array(im)[:, :, :3]  # converts image and drops alpha channel
arr = arr.astype('float32')
arr /= 255  # this is same as arr = arr/255

# Contrast stretching
p2, p98 = np.percentile(arr, (2, 98))
img_rescale = exposure.rescale_intensity(arr, in_range=(p2, p98))
img_rescale_PIL = Image.fromarray(np.uint8(img_rescale*255))
img_rescale_PIL.show()

# Equalization
img_eq = exposure.equalize_hist(arr)
img_eq_PIL= Image.fromarray(np.uint8(img_eq*255))
img_eq_PIL.show()

# Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(arr, clip_limit=0.03)
img_adapteq_PIL = Image.fromarray(np.uint8(img_adapteq*255))
img_adapteq_PIL.show()



def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.
    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 4), dtype=np.object)
axes[0, 0] = fig.add_subplot(2, 4, 1)

for i in range(1, 4):
    axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])

for i in range(0, 4):
    axes[1, i] = fig.add_subplot(2, 4, 5 + i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(im, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')

ax_hist.set_yticks(np.linspace(0, y_max, 5))
ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
ax_img.set_title('Histogram equalization')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
ax_img.set_title('Adaptive equalization')
ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 5))

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()

