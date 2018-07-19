import os
import uuid
import random
import numpy as np
from PIL import Image
from PIL import ImageDraw


def make_256_batch(test_emb_folder, batch_size, opt_z_stack_dict, batch_id, save_dir):
    counter = 0
    test_emb_list = os.listdir(test_emb_folder)

    while counter < batch_size:
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
            print(y)
            for x in xrange(emb_outlines_image.size[0]):
                [r, g, b] = pixdata[x, y]
                if (max(r, g, b) - min(r, g, b)) < 80:  ## trial and error = best fit
                    pixdata[x, y] = (0, 0, 0)
                else:
                    pixdata[x, y] = (255, 255, 255)

        cX, cY = (256, 256)  # Size of Bounding Box for ellipse
        center_bit = (emb_outlines_image.size[0]/ 2 - cX / 2, emb_outlines_image.size[1] / 2 - cY / 2,
                        emb_outlines_image.size[0] / 2 + cX / 2, emb_outlines_image.size[1] / 2 + cY / 2)

        # draw = ImageDraw.Draw(emb_outlines_image)
        # draw.rectangle(center_bit, fill=(0,128, 0))
        # emb_outlines_image.show()

        crop_emb_raw_image = emb_raw_image.crop(box=center_bit)
        crop_emb_outlines_image = emb_outlines_image.crop(box=center_bit)
        crop_emb_raw_image.save(os.path.join(save_dir, "raw", str(counter) + ".png"), "PNG")
        crop_emb_outlines_image.save(os.path.join(save_dir, "outlines", str(counter) + ".png"), "PNG")

        counter += 1


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

batch_id = str(uuid.uuid4())
save_dir = "/home/iolie/PycharmProjects/unet/data/shaz_test"
test_embs = "/home/iolie/PhD_Thesis_Data/epithelial_cell_border_test_embs"
make_256_batch(test_embs, 30, opt_z_stack_dict, batch_id, save_dir)