import os
import pandas as pd
import random
import mahotas as mh
from PIL import Image
import PIL.ImageOps
import numpy as np
from keras.models import model_from_json
from U_NET_training_functions import emb_image_batch_generator_v2
from sklearn.model_selection import train_test_split

test_folder = ""
test_list = []
piclist = []
for dirpath, subdirs, files in os.walk(test_folder):
    for f in files:
        if f.endswith('.jpg'):
            piclist.append(os.path.join(dirpath, f))




def pixel_and_cell_count_test(predictions_batch, ground_truth_batch, model_dir, save_dir, demo = False, rsz = (80,80), batch_size = 10, min_size = 5):

    json_file = open(os.path.join(model_dir, "model.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(model_dir, "weights.h5"))
    outlines_pred = model.predict(predictions_batch, batch_size= batch_size,
                                  verbose=1, steps=None)

    columns = ['ground_truth_cell_count', 'cell_count_0.1_threshold',
               'cell_count_0.2_threshold', 'cell_count_0.3_threshold',
               'cell_count_0.4_threshold', 'cell_count_0.5_threshold',
               'cell_count_0.6_threshold', 'cell_count_0.7_threshold',
               'cell_count_0.8_threshold', 'cell_count_0.9_threshold',
               'pixel_accuracy_0.1_threshold', 'pixel_accuracy_0.2_threshold',
               'pixel_accuracy_0.3_threshold', 'pixel_accuracy_0.4_threshold',
               'pixel_accuracy_0.5_threshold', 'pixel_accuracy_0.6_threshold',
               'pixel_accuracy_0.7_threshold', 'pixel_accuracy_0.8_threshold',
               'pixel_accuracy_0.9_threshold', 'prediction_rmse']

    df = pd.DataFrame(columns=columns)

    for i in range(outlines_pred.shape[0]):

        raw = predictions_batch[i]
        gt = ground_truth_batch[i]
        prediction = outlines_pred[i]

        raw.convert('RGB').save(os.path.join(save_dir,"raw_image_" + str(
                                                  i) + ".png"), "PNG")

        gt.convert('RGB').save(os.path.join(save_dir, "gt_image" + str(
                                                 i) + ".png"), "PNG")




""""
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


        if demo:
            Image.fromarray(gt_labeled * 10).show()
"""""




if __name__ == '__main__':
    batch_size = 32
    # size_to_resize_to = (144, 144)
    # data_folder = "/home/iolie/PhD_Thesis_Data/epithelial_cell_border_identification"
    # emb_list =
    # model_dir = "/home/iolie/PycharmProjects/THESIS/savedmodels_unet_22"
    # save_dir = ""
    # batch_size = 10
    #
    # raw_image_batch, ground_truth_batch = emb_image_batch_generator_v2(data_folder, emb_list, batch_size, size_to_resize_to)
    # pixel_and_cell_count_test(raw_image_batch, ground_truth_batch,
    #                               model_dir,save_dir, demo=False, rsz=(144, 144),   ## resize as a variable read from the array shape...
    #                               batch_size=10, min_size=5)

