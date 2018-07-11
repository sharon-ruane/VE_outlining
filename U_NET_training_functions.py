import logging
import os
import random
import numpy as np
from PIL import Image

log = logging.getLogger("")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

def overlap_box(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    return (x_left, y_top, x_right, y_bottom)

def crop_image_in_special_box(im, special_box, size):
    l,d = random.randint(special_box[0], special_box[2] - size[0]), random.randint(special_box[1], special_box[3] - size[1])
    box =(l,d,l+size[0],d+size[1])
    return im.crop(box), box


def emb_image_batch_generator(data_folder, emb_list, batch_size, size_to_resize_to, opt_z_stack_dict):
    while True:
        log.info("Spawning a new embryo image batch...")
        image_section_batch = []
        pixel_labels_batch = []

        while len(image_section_batch) < batch_size:

            emb_choice = random.choice(emb_list)
            timepoints = []
            for file in os.listdir(os.path.join(data_folder, emb_choice)):
                if file.startswith('T'):
                    timepoints.append(os.path.join(data_folder, emb_choice, file))
            opt_z = opt_z_stack_dict[emb_choice]
            timepoint = random.choice(timepoints)
            time_split = timepoint.split("/")[-1]
            image_needed = time_split + "C02" + "Z00" + str(opt_z) + ".tif"

            emb_raw_image = Image.open(os.path.join(timepoint, image_needed)).convert("L")
            outlines_poss_multiple = []
            for file in os.listdir(timepoint):
                if file.startswith('xCell'):
                    outlines_poss_multiple.append(file)

            emb_outlines_image = Image.open(os.path.join(timepoint, random.choice(outlines_poss_multiple)))

            pixdata = emb_outlines_image.load()
            for y in xrange(emb_outlines_image.size[1]):
                for x in xrange(emb_outlines_image.size[0]):
                    [r, g, b] = pixdata[x, y]
                    if (max(r, g, b) - min(r, g, b)) < 80:  ## trial and error = best fit
                        pixdata[x, y] = (0, 0, 0)
                    else:
                        pixdata[x, y] = (255, 255, 255)

            rotations = random.randint(0, 360)
            emb_raw_image = emb_raw_image.rotate(rotations)
            emb_outlines_image = emb_outlines_image.rotate(rotations)

            binary_outlines = emb_outlines_image.convert("1")
            usable_outlines = binary_outlines.convert("L")
            binary_outlines_arr = np.asarray(binary_outlines)

            auto_bounding_box_pixels = emb_outlines_image.getbbox()
            center_bit = ((emb_outlines_image.size[0] // 4), (emb_outlines_image.size[1] // 4),
                          (emb_outlines_image.size[0] // 4) * 3, (emb_outlines_image.size[1] // 4) * 3)

            counter = 0
            while counter < 5 and len(image_section_batch) < batch_size:
                y = random.randint(50, 64)
                if binary_outlines_arr.any():
                    check_box = overlap_box(auto_bounding_box_pixels, center_bit)
                else:
                    check_box = center_bit
                    log.error("Image is too small")

                if y >= check_box[2] - check_box[0]:
                    check_box = center_bit
                    pass
                if y >= check_box[3] - check_box[1]:
                    check_box = center_bit
                    pass

                crop_im, b_box = crop_image_in_special_box(emb_outlines_image, check_box, (y, y))

                w_range = range(b_box[0], b_box[2])
                h_range = range(b_box[1], b_box[3])
                bb_w1_pixels = [binary_outlines_arr[(b_box[1]), i] for i in w_range]
                bb_w2_pixels = [binary_outlines_arr[(b_box[3]), i] for i in w_range]
                bb_h1_pixels = [binary_outlines_arr[(i, b_box[0])] for i in h_range]
                bb_h2_pixels = [binary_outlines_arr[(i, b_box[2])] for i in h_range]


                x = [any(bb_w1_pixels), any(bb_w2_pixels), any(bb_h1_pixels), any(bb_h2_pixels)]
                if all(x) == True:
                    #log.info("Successful crop --- Adding to batch!")
                    crop_emb_raw_image = emb_raw_image.crop(box=b_box)
                    crop_emb_raw_image_arr = np.asarray(crop_emb_raw_image.resize((64, 64))).reshape(64, 64, 1)
                    #Image.fromarray(crop_emb_raw_image_arr.reshape(64, 64)).show()

                    image_section_batch.append(crop_emb_raw_image_arr/float(255))
                    #crop_emb_outlines_image = binary_outlines.crop(box=b_box)
                    crop_emb_outlines_image = usable_outlines.crop(box=b_box)
                    crop_emb_outlines_image_arr = np.asarray(crop_emb_outlines_image.resize((64, 64))).reshape(64,64, 1)
                     # crop_emb_outlines_image_arr = np.asarray(crop_emb_outlines_image.resize((80, 80))).reshape(80, 80, 1)
                    #Image.fromarray(crop_emb_outlines_image_arr.reshape(64, 64)).show()
                    #crop_emb_outlines_image.show()
                    pixel_labels_batch.append(1-(crop_emb_outlines_image_arr/float(255)))
                counter += 1


        yield np.asarray(image_section_batch), np.asarray(pixel_labels_batch)




#
#






def make_test_batch(test_emb_folder, test_batch_size, opt_z_stack_dict):
    test_batch = []
    ground_truth = []
    test_emb_list = os.listdir(test_emb_folder)

    while len(test_batch) < test_batch_size:
        emb_choice = random.choice(test_emb_list)
        timepoints = []
        for file in os.listdir(os.path.join(test_emb_folder, emb_choice)):
            if file.startswith('T'):
                timepoints.append(os.path.join(test_emb_folder, emb_choice, file))
        opt_z = opt_z_stack_dict[emb_choice]
        timepoint = random.choice(timepoints)
        time_split = timepoint.split("/")[-1]
        image_needed = time_split + "C02" + "Z00" + str(opt_z) + ".tif"

        emb_raw_image = Image.open(os.path.join(timepoint, image_needed)).convert("L")
        outlines_poss_multiple = []
        for file in os.listdir(timepoint):
            if file.startswith('xCell'):
                outlines_poss_multiple.append(file)

        emb_outlines_image = Image.open(os.path.join(timepoint, random.choice(outlines_poss_multiple)))

        pixdata = emb_outlines_image.load()
        for y in xrange(emb_outlines_image.size[1]):
            for x in xrange(emb_outlines_image.size[0]):
                [r, g, b] = pixdata[x, y]
                if (max(r, g, b) - min(r, g, b)) < 80:  ## trial and error = best fit
                    pixdata[x, y] = (0, 0, 0)
                else:
                    pixdata[x, y] = (255, 255, 255)

        rotations = random.randint(0, 360)
        emb_raw_image = emb_raw_image.rotate(rotations)
        emb_outlines_image = emb_outlines_image.rotate(rotations)

        binary_outlines = emb_outlines_image.convert("1")

        binary_outlines_arr = np.asarray(binary_outlines)

        auto_bounding_box_pixels = emb_outlines_image.getbbox()
        center_bit = ((emb_outlines_image.size[0] // 4), (emb_outlines_image.size[1] // 4),
                      (emb_outlines_image.size[0] // 4) * 3, (emb_outlines_image.size[1] // 4) * 3)

        counter = 0
        while counter < 5 and len(test_batch) < test_batch_size:
            y = 64

            if binary_outlines_arr.any():
                check_box = overlap_box(auto_bounding_box_pixels, center_bit)
            else:
                check_box = center_bit
                log.error("Image is too small")

            if y >= check_box[2] - check_box[0]:
                check_box = center_bit
                pass
            if y >= check_box[3] - check_box[1]:
                check_box = center_bit
                pass

            crop_im, b_box = crop_image_in_special_box(emb_outlines_image, check_box, (y, y))

            w_range = range(b_box[0], b_box[2])
            h_range = range(b_box[1], b_box[3])
            bb_w1_pixels = [binary_outlines_arr[(b_box[1]), i] for i in w_range]
            bb_w2_pixels = [binary_outlines_arr[(b_box[3]), i] for i in w_range]
            bb_h1_pixels = [binary_outlines_arr[(i, b_box[0])] for i in h_range]
            bb_h2_pixels = [binary_outlines_arr[(i, b_box[2])] for i in h_range]

            x = [any(bb_w1_pixels), any(bb_w2_pixels), any(bb_h1_pixels), any(bb_h2_pixels)]
            if all(x) == True:
                # log.info("Successful crop --- Adding to batch!")
                crop_emb_raw_image = emb_raw_image.crop(box=b_box)
                crop_emb_raw_image_arr = np.asarray(crop_emb_raw_image.resize((64, 64))).reshape(64, 64, 1)
                test_batch.append(crop_emb_raw_image_arr / float(255)) # this has gotta go in the net
                crop_emb_outlines_image = emb_outlines_image.crop(box=b_box)
                ground_truth.append(np.asarray(crop_emb_outlines_image))  # just want pic here
            counter += 1
    return np.asarray(test_batch), ground_truth

#
#
# model_dir = "/home/iolie/PycharmProjects/THESIS/savedmodels_unet_5/titletraining_weightsatloss_0.69"
# data_folder = "/home/iolie/PhD_Thesis_Data/epithelial_cell_border_identification"
# emb_list = os.listdir(data_folder)
# #print(emb_list)
# emb_list.remove('.DS_Store')
# randoms = random.sample(emb_list, 2)
#
#
# opt_z_stack_dict = {}
# opt_z_stack_dict["AAntnew33-47.lsm (cropped)"] = 3
# opt_z_stack_dict["ANTERIOR \"EMB 4\" Nov. 28th Emb (2)_L5_Sum.lsm (spliced)"] = 3
# opt_z_stack_dict["Anterior = Embryo 5\" Feb 20th"] = 4
# opt_z_stack_dict["USEAnt(potential)2_march__t2.lsm (spliced) (cropped)"] = 3
# opt_z_stack_dict["LATERAL \"EMB 3\" Oct 2nd Emb (1)_L3_Sum.lsm (spliced)"] = 3
# opt_z_stack_dict["LATERAL\"EMB 6\" Nov. 28th Emb (2)_L7_Sum.lsm "] = 3
# opt_z_stack_dict["LATERAL \"EMB 9\" Dec 15th Emb (1)_L12_Sum.lsm (spliced)"] = 4
# opt_z_stack_dict["LATERAL \"EMB 12\" Nov. 28th Emb (2)_L12_Sum.lsm "] = 4
# opt_z_stack_dict["Outline this movie tp 6-22 posterior copy"] = 5
# opt_z_stack_dict["POSTERIOR = \"Embryo 2\", Nov. 28th Emb (2)_L3_Sum.lsm (spliced)"] = 3
# opt_z_stack_dict["EARLY Posterior = \"Embryo 6\", Feb. 20th Emb (1)_L6_Sum.lsm (spliced)"] = 3
# opt_z_stack_dict["LATE Posterior = \"Embryo 6\", Feb. 20th Emb (1)_L6_Sum.lsm (spliced)"] = 4
#
# # x = make_test_batch(data_folder, 3, opt_z_stack_dict)
# # a, b = x.next()
#
# x = emb_image_batch_generator(data_folder, emb_list, 2, (64,64), opt_z_stack_dict)
# a, b = x.next()
#
#
#
# pass
#