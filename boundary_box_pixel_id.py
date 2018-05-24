import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from itertools import groupby
from matplotlib import pyplot as plt


def crop_image(im,size):
    w,h=im.size
    w_max,h_max = w-size[0],h-size[1]
    l,d = random.randint(0,w_max+1),random.randint(0,h_max+1)
    box =(l,d,l+size[0],d+size[1])
    return im.crop(box), box


im = Image.open("FAKECELLS.jpg").convert("1")
im.show()
print(im.size)

# crop_im, b_box = crop_image(im, (150,150))
# crop_im.show()
# print("bounding box: " + str(b_box))
#
# arr = np.array(im)
# print(arr)
# print(arr.shape)
#
#
# w_range = range(b_box[0], b_box[2])
# h_range = range(b_box[1], b_box[3])
# bb_w1_pixels = [arr[(b_box[1]), i] for i in w_range]
# bb_w2_pixels = [arr[(b_box[3]), i] for i in w_range]
# bb_h1_pixels = [arr[(i, b_box[0])] for i in h_range]
# bb_h2_pixels = [arr[(i, b_box[2])] for i in h_range]

#
# print(bb_w1_pixels)
# print("topline: " + str([len(list(group)) for key, group in groupby(bb_w1_pixels)]))
# print("bottomline: " + str([len(list(group)) for key, group in groupby(bb_w2_pixels)]))
# print("leftline: " + str([len(list(group)) for key, group in groupby(bb_h1_pixels)]))
# print("rightline: " + str([len(list(group)) for key, group in groupby(bb_h2_pixels)]))

# bb_w1_test_pixels = [(i, b_box[1]) for i in w_range]
# bb_w2_test_pixels = [(i, b_box[3]) for i in w_range]
# bb_h1_test_pixels = [(b_box[0], i) for i in h_range]
# bb_h2_test_pixels = [(b_box[2], i) for i in h_range]
# test_bounding_box_pixels = bb_w1_test_pixels + bb_w2_test_pixels + bb_h1_test_pixels + bb_h2_test_pixels
# draw = ImageDraw.Draw(im)
# draw.point(test_bounding_box_pixels, fill=128)   ####### YES!!!! correct
# im.show()

#plt.clf()
i = 0
while i < 9:
    print(i)
    crop_im, b_box = crop_image(im, (150,150))
    arr = np.array(im)
    w_range = range(b_box[0], b_box[2])
    h_range = range(b_box[1], b_box[3])
    try:
        bb_w1_pixels = [arr[(b_box[1]), a] for a in w_range]
        bb_w2_pixels = [arr[(b_box[3]), b] for b in w_range]
        bb_h1_pixels = [arr[(c, b_box[0])] for c in h_range]
        bb_h2_pixels = [arr[(d, b_box[2])] for d in h_range]
        x = [any(bb_w1_pixels), any(bb_w2_pixels), any(bb_h1_pixels), any(bb_h2_pixels)]
        print(x)
        print(all(x))
        if all(x) == True:
            plt.subplot(330 + 1 + i)
            plt.imshow(crop_im, cmap=plt.get_cmap('gray'))
            i += 1
        else:
            print("out of bounds")
    except:
        print("I dunno why this indexes wrong")

plt.show()



# import os
# cwd = os.getcwd()
# print(cwd)