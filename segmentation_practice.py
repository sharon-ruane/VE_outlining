#matplotlib inline
import mahotas as mh
from PIL import Image
import PIL.ImageOps
import numpy as np
from matplotlib import pyplot as plt
from IPython.html.widgets import interact, fixed


#
# dna = mh.demos.load('nuclear')
# print(dna.shape)
# dna = dna.max(axis=2)
# print(dna.shape)
# Image.fromarray(dna).show()


# T_otsu = mh.otsu(dna)
# print(T_otsu)
# Image.fromarray(dna > T_otsu).show()


bw = "/home/iolie/Desktop/THESIS IMAGES/savedmodels_unet_12/titletraining_weightsatloss_0.28/ea2a195a-e341-4738-aa0e-5dc32daff93bground_truth_0.png"
wb = "/home/iolie/Desktop/THESIS IMAGES/savedmodels_unet_12/titletraining_weightsatloss_0.28/ea2a195a-e341-4738-aa0e-5dc32daff93b_predicted0_threshold0.8.png"

im = Image.open(wb).convert("L")
im.show()
# inv = PIL.ImageOps.invert(im)
imm = np.asarray(im)


labeled, nr_objects = mh.label(imm)
print(nr_objects)
Image.fromarray(labeled*10).show()

sizes = mh.labeled.labeled_size(labeled)
min_size = 3
filtered = mh.labeled.remove_regions_where(labeled, sizes < min_size)
labeled2,nr_objects = mh.labeled.relabel(filtered)
print("Number of cells: {}".format(nr_objects))

#Image.fromarray(filtered).show()
Image.fromarray(labeled2*20).show()
print(labeled2)

# @interactlabeled,nr_objects = mh.labeled.relabel(filtered)(sigma=(1., 16))
# def checkprint("Number of cells: {}".format(nr_objects))_sigma(sigma):
#     dnaf = mh.gaussian_filter(imm.astype(np.float32), sigma)
#     maxima = mh.regmax(mh.stretch(dnaf))
#     maxima = mh.dilate(maxima, np.ones((5,5)))
#     print(maxima)
#     Image.fromarray(mh.as_rgb(np.maximum(255*maxima, dnaf), imm, imm > T_mean)).show() # sigma = 3.0

#
# sigma = 8.5
# dnaf = mh.gaussian_filter(imm.astype(float), sigma)
# maxima = mh.regmax(mh.stretch(dnaf))
# maxima, _ = mh.label(maxima)
# Image.fromarray(maxima).show()
# # #





