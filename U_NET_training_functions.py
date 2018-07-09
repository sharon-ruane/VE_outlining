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
            while counter < 10 and len(image_section_batch) < batch_size:
                y = random.randint(50, 176)
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
                    crop_emb_raw_image_arr = np.asarray(crop_emb_raw_image.resize((176, 176))).reshape(176, 176, 1)
                    #Image.fromarray(crop_emb_raw_image_arr.reshape(128, 128)).show()
                    image_section_batch.append(crop_emb_raw_image_arr/255)
                    # crop_emb_outlines_image = binary_outlines.crop(box=b_box)
                    crop_emb_outlines_image = usable_outlines.crop(box=b_box)
                    crop_emb_outlines_image_arr = np.asarray(crop_emb_outlines_image.resize((176, 176))).reshape(176, 176, 1)
                    #Image.fromarray(crop_emb_outlines_image_arr.reshape(128, 128)).show()
                    #crop_emb_outlines_image.show()
                    pixel_labels_batch.append(crop_emb_outlines_image_arr/255)
                counter += 1


        yield (np.asarray(image_section_batch), np.asarray(pixel_labels_batch))















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
        while counter < 10 and len(test_batch) < test_batch_size:
            y = 96
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
                test_batch.append(np.asarray(crop_emb_raw_image).reshape(176, 176, 1)) # this has gotta go in the net
                crop_emb_outlines_image = emb_outlines_image.crop(box=b_box)
                ground_truth.append(np.asarray(crop_emb_outlines_image))  # just want pic here
            counter += 1
    return test_batch, ground_truth
