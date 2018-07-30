import os
import random
import mahotas as mh
from PIL import Image
import PIL.ImageOps
import numpy as np
from keras.models import model_from_json

test_folder = ""
test_list = []
piclist = []
for dirpath, subdirs, files in os.walk(test_folder):
    for f in files:
        if f.endswith('.jpg'):
            piclist.append(os.path.join(dirpath, f))

def make_test_batch(test_list, batch_size = 10):

    image_section_batch = []
    ground_truth_labels = []

    while len(image_section_batch) < batch_size:

    ## however len is doing rotations ....
        raw_image = Image.open( os.path.join("", random.choice(""))  ## ??? how to link?
        ground_truth = Image.open(os.path.join("", random.choice(""))





    return np.asarray(image_section_batch), np.asarray(ground_truth_labels)



def pixel_and_cell_count_test(predictions_batch, ground_truth_batch, model_dir,  mode, rsz = (80,80), batch_size = 10, min_size = 5):

    json_file = open(os.path.join(model_dir, "model.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(model_dir, "weights.h5"))
    outlines_pred = model.predict(predictions_batch, batch_size= batch_size,
                                  verbose=1, steps=None)


    results = np.zeros()
    for i in range(outlines_pred.shape[0]):
        pred =ground_truth_batch[i]
        gt = ground_truth_batch[i]

        mask = pred.reshape(rsz[0], rsz[1])


        direct_output_mask = Image.fromarray(mask * 255)

        for j in range(1,9):
            mask = np.copy(pred.reshape(rsz[0], rsz[1]))
            mask[mask > j/10] = 255
            mask[mask <= j/10] = 0
            imgr = Image.fromarray(mask)



        #imgx.convert('L')


        inv = PIL.ImageOps.invert(ground_truth)
        imm = np.asarray(inv)
        gt_labeled, gt_nr_objects = mh.label(imm)
        sizes = mh.labeled.labeled_size(gt_labeled)
        gt_filtered = mh.labeled.remove_regions_where(gt_labeled, sizes < min_size)
        gt_final_labeled, gt_final_nr_objects = mh.labeled.relabel(gt_filtered)




        pixel_pred = Image.open(os.path.join("", random.choice(""))
        inv = PIL.ImageOps.invert(ground_truth)
        imm = np.asarray(inv)
        gt_labeled, gt_nr_objects = mh.label(imm)
        sizes = mh.labeled.labeled_size(gt_labeled)
        gt_filtered = mh.labeled.remove_regions_where(gt_labeled, sizes < min_size)
        gt_final_labeled, gt_final_nr_objects = mh.labeled.relabel(gt_filtered)



        if mode == "demo": ###???
            Image.fromarray(gt_labeled * 10).show()


print("Number of cells: {}".format(nr_objects))