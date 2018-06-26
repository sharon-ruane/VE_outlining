import os
import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from itertools import groupby
from keras.preprocessing.image import ImageDataGenerator

def crop_image_in_centre(im,size):
    w,h=im.size
    w_max,h_max = w-size[0],h-size[1]
    l,d = random.randint(w_max//3,(w_max//3)*2), random.randint(h_max//3,(h_max//3)*2)
    box =(l,d,l+size[0],d+size[1])
    return im.crop(box), box


data_folder = "/home/iolie/PhD_Thesis_Data/epithelial_cell_border_identification"
emb_list = os.listdir(data_folder)
print(emb_list)
emb_list.remove('.DS_Store')
print(emb_list)

opt_z_stack_dict = {}
opt_z_stack_dict["AAntnew33-47.lsm (cropped)"] = 3
opt_z_stack_dict["ANTERIOR \"EMB 4\" Nov. 28th Emb (2)_L5_Sum.lsm (spliced)"] = 3
opt_z_stack_dict["Anterior = Embryo 5\" Feb 20th"] = 4
opt_z_stack_dict["USEAnt(potential)2_march__t2.lsm (spliced) (cropped)"] = 3
opt_z_stack_dict["LATERAL \"EMB 3\" Oct 2nd Emb (1)_L3_Sum.lsm (spliced)"] = 3
opt_z_stack_dict["LATERAL\"EMB 6\" Nov. 28th Emb (2)_L7_Sum.lsm "] = 3
opt_z_stack_dict["LATERAL \"EMB 9\" Dec 15th Emb (1)_L12_Sum.lsm (spliced)"] = 4
opt_z_stack_dict["LATERAL \"EMB 12\" Nov. 28th Emb (2)_L12_Sum.lsm "] = 4
opt_z_stack_dict["Outline this movie tp 6-22 posterior copy"] = 5
opt_z_stack_dict["POSTERIOR = \"Embryo 2\", Nov. 28th Emb (2)_L3_Sum.lsm (spliced)"] = 3
opt_z_stack_dict["EARLY Posterior = \"Embryo 6\", Feb. 20th Emb (1)_L6_Sum.lsm (spliced)"] = 3
opt_z_stack_dict["LATE Posterior = \"Embryo 6\", Feb. 20th Emb (1)_L6_Sum.lsm (spliced)"] = 4
#print(opt_z_stack_dict)

batch_size = 1

#while True:
image_section_batch = []
pixel_labels_batch = []
counter = 0
while len(image_section_batch) < batch_size:
#while counter < 5:
    emb_choice = random.choice(emb_list)
    #print(emb_choice)
    timepoints = []  #### in fairness dude you should pre-compile this
    for file in os.listdir(os.path.join(data_folder, emb_choice)):
        if file.startswith('T'):
            timepoints.append(os.path.join(data_folder, emb_choice, file))
    #print(timepoints)
    opt_z = opt_z_stack_dict[emb_choice]
    #print(opt_z)
    timepoint = random.choice(timepoints)
    #print(timepoint)
    time_split = timepoint.split("/")[-1]
    #print(time_split)
    image_needed = time_split + "C02" + "Z00" + str(opt_z) + ".tif"
    #print(image_needed)

    emb_raw_image = Image.open(os.path.join(timepoint, image_needed))
    outlines_poss_multiple = []
    for file in os.listdir(timepoint):
        if file.startswith('xCell'):
            outlines_poss_multiple.append(file)
    #print(outlines_poss_multiple)

    #emb_raw_image.show()
    emb_outlines_image = Image.open(os.path.join(timepoint, random.choice(outlines_poss_multiple)))
    #emb_outlines_image.show()

    pixdata = emb_outlines_image.load()
    for y in xrange(emb_outlines_image.size[1]):
        for x in xrange(emb_outlines_image.size[0]):
            [r, g, b] = pixdata[x, y]
            if (max(r,g,b) - min(r,g,b)) < 80: ## trial and error = best fit
                pixdata[x, y] = (0,0,0)
            else:
                pixdata[x, y] = (255, 255, 255)
    #emb_outlines_image.show()

    rotations = random.randint(0, 360)
    emb_raw_image = emb_raw_image.rotate(rotations)
    emb_outlines_image = emb_outlines_image.rotate(rotations)
    #emb_raw_image.show()
    #emb_outlines_image.show()
    #counter += 1

#
    binary_outlines = emb_outlines_image.convert("1")
    binary_outlines_arr = np.asarray(binary_outlines)
    #print(np.asarray(binary_outlines))

    x = random.randint(50, 80)

    crop_im, b_box = crop_image_in_centre(emb_outlines_image, (x,x))
    #crop_im.show()
    #print("bounding box: " + str(b_box))

    w_range = range(b_box[0], b_box[2])
    h_range = range(b_box[1], b_box[3])
    bb_w1_pixels = [binary_outlines_arr[(b_box[1]), i] for i in w_range]
    bb_w2_pixels = [binary_outlines_arr[(b_box[3]), i] for i in w_range]
    bb_h1_pixels = [binary_outlines_arr[(i, b_box[0])] for i in h_range]
    bb_h2_pixels = [binary_outlines_arr[(i, b_box[2])] for i in h_range]

    #print(bb_w1_pixels)
    # print("topline: " + str([len(list(group)) for key, group in groupby(bb_w1_pixels)]))
    # print("bottomline: " + str([len(list(group)) for key, group in groupby(bb_w2_pixels)]))
    # print("leftline: " + str([len(list(group)) for key, group in groupby(bb_h1_pixels)]))
    # print("rightline: " + str([len(list(group)) for key, group in groupby(bb_h2_pixels)]))

    bb_w1_test_pixels = [(i, b_box[1]) for i in w_range]
    bb_w2_test_pixels = [(i, b_box[3]) for i in w_range]
    bb_h1_test_pixels = [(b_box[0], i) for i in h_range]
    bb_h2_test_pixels = [(b_box[2], i) for i in h_range]
    test_bounding_box_pixels = bb_w1_test_pixels + bb_w2_test_pixels + bb_h1_test_pixels + bb_h2_test_pixels
    #draw = ImageDraw.Draw(emb_outlines_image)
    #draw.point(test_bounding_box_pixels, fill=128)   ####### YES!!!! correct
    #emb_outlines_image.show()

    x = [any(bb_w1_pixels), any(bb_w2_pixels), any(bb_h1_pixels), any(bb_h2_pixels)]
    #print(x)
    #print(all(x))
    if all(x) == True:
        print("Successful crop --- Adding to batch!")
        image_section_batch.append(np.asarray(emb_raw_image.crop(box = b_box)))
        pixel_labels_batch.append((np.asarray(emb_outlines_image.crop(box = b_box))))
        emb_outlines_image.show()

        ### resizing step!!

        emb_raw_image.crop(box=b_box).show()
        emb_raw_image.show()
        emb_outlines_image.crop(box=b_box).show()
    else:
        print("Not suitable")
    counter += 1
    print("Currently looking in " + str(emb_choice) + str(time_split))
    print("Attempting crop number: " + str(counter))



# for pic in image_section_batch:
#     img = Image.fromarray(pic)
#     img.show()

for pic in pixel_labels_batch:
    img = Image.fromarray(pic)
    img.show()


# emb_raw_image_arr = np.asarray(emb_raw_image)
# print(emb_raw_image_arr.shape)
# emb_outlines_image_arr = np.asarray(emb_outlines_image)
# print(emb_outlines_image_arr.shape)



# emb_image = Image.open(os.path.join(data_folder, "AAntnew33-47.lsm (cropped)/T00010/xCell outlines copy 3"))
#
# emb_image.show()
# emb_arr = np.asarray(emb_image)
# print(emb_arr.shape)
#
# pixdata = emb_image.load()
#
#
# for y in xrange(emb_image.size[1]):
#     for x in xrange(emb_image.size[0]):
#         [r, g, b] = pixdata[x, y]
#         if (max(r,g,b) - min(r,g,b)) < 90: ## trial and error = best fit
#             pixdata[x, y] =  (0,0,0)
#         else:
#             pixdata[x, y] = (255, 255, 255)
#
# emb_image.save(os.path.join(data_folder, "AAntnew33-47.lsm (cropped)/T00010", "test6.tiff"), dpi=(300, 300))
