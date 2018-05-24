from __future__ import print_function

import os
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


# input image dimensions  --- this is a place holder that gets a random one from file
bg_pics_dir = "/home/iolie/PycharmProjects/keras__practice/number_checker/lfw_mix"
piclist = []
for dirpath, subdirs, files in os.walk(bg_pics_dir):
    for f in files:
        if f.endswith('.jpg'):
            piclist.append(os.path.join(dirpath, f))
sample_im = Image.open(piclist[random.randint(0, len(piclist)-1)]).convert('RGBA')
sample_im.show()

# make a practice array of images
images = range(0,9)
X_data = []
for i in images:
    im = Image.open(piclist[i])

    arr = np.array(im)[:, :, :3]  # converts image and drops alpha channel
    arr = arr.astype('float32')
    arr /= 255  # this is same as arr = arr/255
    X_data.append(arr)

    plt.subplot(330 + 1 + i)
    plt.imshow(im, cmap=plt.get_cmap('gray'))

plt.show()
#plt.clf()
print(np.array(X_data).shape)
X_data = np.array(X_data)
placeholder = [1,2,3,4,5,6,7,8,9]
Y_data = np.array(placeholder)


datagen = ImageDataGenerator(rotation_range=360)
datagen.fit(X_data)
for X_batch, y_batch in datagen.flow(X_data, Y_data, batch_size= 9):
    # Show 9 images
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i])
    # show the plot
    plt.show()
    break

# can do vertical or horizontal shift, this isn't really as useful
# Would shift images vertically or horizontally + fill missing pixels with the color of the nearest pixel
# not good for pixel wise output, I would guess
# better to augment where we crop




