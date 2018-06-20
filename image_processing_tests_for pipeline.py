import os
import colorsys
import numpy as np
from PIL import Image

data_folder = "/home/iolie/PhD_Thesis_Data/epithelial_cell_border_identification"
#
# emb_image_list = []
# for dirpath, subdirs, files in os.walk(data_folder):
#     for f in files:
#         if f.endswith('.jpg'):
#             emb_image_list.append(os.path.join(dirpath, f))
#
#
# opt_z_stack_dict = {}
# opt_z_stack_dict["AAntnew33-47.lsm (cropped)"] = 3
# opt_z_stack_dict["ANTERIOR \"EMB 4\" Nov. 28th Emb (2)_L5_Sum.lsm (spliced)"] = 3
# opt_z_stack_dict["Anterior = Embryo 5\" Feb 20th"] = 4
# opt_z_stack_dict["USEAnt(potential)2_march__t2.lsm (spliced) (cropped)"] = 3
#
# opt_z_stack_dict["LATERAL \"EMB 3\" Oct 2nd Emb (1)_L3_Sum.lsm (spliced)"] = 3
# opt_z_stack_dict["LATERAL\"EMB 6\" Nov. 28th Emb (2)_L7_Sum.lsm"] = 3
# opt_z_stack_dict["LATERAL \"EMB 9\" Dec 15th Emb (1)_L12_Sum.lsm (spliced)"] = 4
# opt_z_stack_dict["LATERAL \"EMB 12\" Nov. 28th Emb (2)_L12_Sum.lsm"] = 4
#
# opt_z_stack_dict["Outline this movie tp 6-22 posterior copy"] = 5
# opt_z_stack_dict["POSTERIOR = \"Embryo 2\", Nov. 28th Emb (2)_L3_Sum.lsm (spliced)"] = 3
# opt_z_stack_dict["EARLY Posterior = \"Embryo 6\", Feb. 20th Emb (1)_L6_Sum.lsm (spliced)"] = 3
# opt_z_stack_dict["LATE Posterior = \"Embryo 6\", Feb. 20th Emb (1)_L6_Sum.lsm (spliced)"] = 4
#
# print(opt_z_stack_dict)


emb_image = Image.open(os.path.join(data_folder, "AAntnew33-47.lsm (cropped)/T00010/xCell outlines copy 3"))

emb_image.show()
emb_arr = np.asarray(emb_image)
print(emb_arr.shape)

pixdata = emb_image.load()


for y in xrange(emb_image.size[1]):
    for x in xrange(emb_image.size[0]):
        [r, g, b] = pixdata[x, y]
        if (max(r,g,b) - min(r,g,b)) < 90: ## trial and error = best fit
            pixdata[x, y] =  (0,0,0)
        else:
            pixdata[x, y] = (255, 255, 255)

emb_image.save(os.path.join(data_folder, "AAntnew33-47.lsm (cropped)/T00010", "test6.tiff"), dpi=(300, 300))
