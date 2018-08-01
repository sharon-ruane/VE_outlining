import os
import json
import random
import logging
import numpy as np
import pandas as pd
import mahotas as mh
from keras.optimizers import Adam
from keras import metrics, Model
from PIL import Image, ImageOps, ImageFilter
from keras.models import model_from_json
from sklearn.metrics import accuracy_score as acc
from U_NET_training_functions import get_base_dataset, emb_image_batch_generator_v2


log = logging.getLogger("")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

def count_cells(array, min_size = 3):
    labeled, nr_objects = mh.label(array)
    print(nr_objects)
    #Image.fromarray(array*255).show()
    sizes = mh.labeled.labeled_size(labeled)
    filtered = mh.labeled.remove_regions_where(labeled,
                                               sizes < min_size)
    labeled2, cell_count = mh.labeled.relabel(filtered)
    #Image.fromarray(labeled2 * 10).show()
    return labeled2, cell_count


def pixel_and_cell_count_test(predictions_batch, ground_truth_batch, model_dir, save_dir, demo = False, rsz = (80,80), batch_size = 2):

    json_file = open(os.path.join(model_dir, "model.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(model_dir, "weights.h5"))
    outlines_pred = model.predict(predictions_batch, batch_size= batch_size,
                                  verbose=1, steps=None)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
                  metrics=['accuracy', 'mae', 'mse'])

    evals=model.test_on_batch(predictions_batch, ground_truth_batch, sample_weight=None)

    columns = ['ground_truth_cell_count', 'ground_truth_thick_cell_count', 'cell_count_0.1_threshold',
               'cell_count_0.2_threshold', 'cell_count_0.3_threshold',
               'cell_count_0.4_threshold', 'cell_count_0.5_threshold',
               'cell_count_0.6_threshold', 'cell_count_0.7_threshold',
               'cell_count_0.8_threshold', 'cell_count_0.9_threshold',
               'pixel_accuracy_0.1_threshold', 'pixel_accuracy_0.2_threshold',
               'pixel_accuracy_0.3_threshold', 'pixel_accuracy_0.4_threshold',
               'pixel_accuracy_0.5_threshold', 'pixel_accuracy_0.6_threshold',
               'pixel_accuracy_0.7_threshold', 'pixel_accuracy_0.8_threshold',
               'pixel_accuracy_0.9_threshold', 'prediction_rmse', "accuracy?"]


    df = pd.DataFrame(columns=columns)
    print(df.index)
    print(range(outlines_pred.shape[0]))
    for i in range(outlines_pred.shape[0]):

        raw = predictions_batch[i].reshape(rsz)
        gt = ground_truth_batch[i].reshape(rsz)
        prediction = outlines_pred[i]
        error = np.mean(abs(prediction-gt))
        df.loc[i, "accuracy?"] = 1-error


        Image.fromarray(raw*255).convert('RGB').save(os.path.join(save_dir, str(
                                                  i) + "_raw_image_" + ".png"), "PNG")

        Image.fromarray(gt*255).convert('RGB').save(os.path.join(save_dir,str(
                                            i) + "_gt_image"  + ".png"), "PNG")

        Image.fromarray(gt*255).filter(ImageFilter.MinFilter(3)).convert('RGB').save(os.path.join(save_dir, str(
                                                   i) +"_gt_thick_image" +  ".png"), "PNG")

        gt_cell_count = count_cells(gt)
        gt_thick = np.asarray(Image.fromarray(gt).filter(ImageFilter.MinFilter(3)))
        gt_thick_cell_count = count_cells(gt_thick)

        df.loc[i, 'ground_truth_cell_count'] = gt_cell_count
        df.loc[i, 'ground_truth_thick_cell_count'] = gt_thick_cell_count
        df.loc[i, "prediction_rmse"] = np.sqrt(np.mean((prediction - gt) ** 2))
        print(np.sqrt(np.mean((prediction - gt) ** 2)))

        min_threshold = 0.1
        max_threshold = 1.0
        counter = 1

        for j in (np.arange(min_threshold, max_threshold, 0.1)).astype(
                np.float32):
            mask = np.copy(prediction.reshape(rsz[0], rsz[1]))
            mask[mask > j] = 1
            mask[mask <= j] = 0
            imgr = Image.fromarray(mask*255)#.convert("L")
            imgr.convert('RGB').save(os.path.join(save_dir, str(i) + "_predicted"+ str(j) + "_threshold" + ".png"), "PNG")

            labels, cell_count = count_cells(mask)
            Image.fromarray((labels*20)).convert('RGB').save(os.path.join(save_dir,
                                                                       str(
                                                                           i) + "_LABELS" + str(
                                                                           j) + "_threshold" + ".png"), "PNG")

            df.loc[i, "cell_count_{}_threshold".format(
                float(counter) / 10)] = cell_count
            df.loc[i, "pixel_accuracy_{}_threshold".format(float(counter) / 10)] = acc(np.uint8(gt).ravel(), np.uint8(mask).ravel())
            print(acc(np.uint8(gt).ravel(), np.uint8(mask).ravel()))
            counter += 1





    #print(df)
    df.to_csv(os.path.join(save_dir, "parameters"))
    print("Average pixel_accuracy_0.1_threshold: ", df["pixel_accuracy_0.1_threshold"].mean())
    print("Average pixel_accuracy_0.2_threshold: ", df["pixel_accuracy_0.2_threshold"].mean())
    print("Average pixel_accuracy_0.3_threshold: ", df["pixel_accuracy_0.3_threshold"].mean())
    print("Average pixel_accuracy_0.4_threshold: ", df["pixel_accuracy_0.4_threshold"].mean())
    print("Average pixel_accuracy_0.5_threshold: ", df["pixel_accuracy_0.5_threshold"].mean())
    print("Average pixel_accuracy_0.6_threshold: ", df["pixel_accuracy_0.6_threshold"].mean())
    print("Average pixel_accuracy_0.7_threshold: ", df["pixel_accuracy_0.7_threshold"].mean())
    print("Average pixel_accuracy_0.8_threshold: ", df["pixel_accuracy_0.8_threshold"].mean())
    print("Average pixel_accuracy_0.9_threshold: ", df["pixel_accuracy_0.9_threshold"].mean())


    # print("cell_count_rmse_0.1 ", np.sqrt(np.mean(
    #     df["cell_count_0.1_threshold"] - df['ground_truth_cell_count']) ** 2))
    # print("cell_count_rmse_0.2 ", np.sqrt(np.mean(
    #     df["cell_count_0.2_threshold"] - df['ground_truth_cell_count']) ** 2))
    # print("cell_count_rmse_0.3 ", np.sqrt(np.mean(
    #     df["cell_count_0.3_threshold"] - df['ground_truth_cell_count']) ** 2))
    # print("cell_count_rmse_0.4 ", np.sqrt(np.mean(
    #     df["cell_count_0.4_threshold"] - df['ground_truth_cell_count']) ** 2))
    # print("cell_count_rmse_0.5 ", np.sqrt(np.mean(
    #     df["cell_count_0.5_threshold"] - df['ground_truth_cell_count']) ** 2))
    # print("cell_count_rmse_0.6 ", np.sqrt(np.mean(
    #     df["cell_count_0.6_threshold"] - df['ground_truth_cell_count']) ** 2))
    # print("cell_count_rmse_0.7 ", np.sqrt(np.mean(
    #     df["cell_count_0.7_threshold"] - df['ground_truth_cell_count']) ** 2))
    # print("cell_count_rmse_0.8 ", np.sqrt(np.mean(
    #     df["cell_count_0.8_threshold"] - df['ground_truth_cell_count']) ** 2))
    # print("cell_count_rmse_0.9 ", np.sqrt(np.mean(
    #     df["cell_count_0.9_threshold"] - df['ground_truth_cell_count']) ** 2))

    #
    # print("thick_cell_count_rmse_0.1 :", np.sqrt(np.mean(
    #     df["cell_count_0.1_threshold"] - df['ground_truth_thick_cell_count']) ** 2))
    # print("thick_cell_count_rmse_0.2 :", np.sqrt(np.mean(
    #     df["cell_count_0.2_threshold"] - df['ground_truth_thick_cell_count']) ** 2))
    # print("thick_cell_count_rmse_0.3 :", np.sqrt(np.mean(
    #     df["cell_count_0.3_threshold"] - df['ground_truth_thick_cell_count']) ** 2))
    # print("thick_cell_count_rmse_0.4 :", np.sqrt(np.mean(
    #     df["cell_count_0.4_threshold"] - df['ground_truth_thick_cell_count']) ** 2))
    # print("thick_cell_count_rmse_0.5 :", np.sqrt(np.mean(
    #     df["cell_count_0.5_threshold"] - df['ground_truth_thick_cell_count']) ** 2))
    # print("thick_cell_count_rmse_0.6 :", np.sqrt(np.mean(
    #     df["cell_count_0.6_threshold"] - df['ground_truth_thick_cell_count']) ** 2))
    # print("thick_cell_count_rmse_0.7 :", np.sqrt(np.mean(
    #     df["cell_count_0.7_threshold"] - df['ground_truth_thick_cell_count']) ** 2))
    # print("thick_cell_count_rmse_0.8 :", np.sqrt(np.mean(
    #     df["cell_count_0.8_threshold"] - df['ground_truth_thick_cell_count']) ** 2))
    # print("thick_cell_count_rmse_0.9 :", np.sqrt(np.mean(
    #     df["cell_count_0.9_threshold"] - df['ground_truth_thick_cell_count']) ** 2))

    # print("cell_count_0.1 ", (1-((abs(df["cell_count_0.1_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_cell_count'])).mean())
    # print("cell_count_0.2 ", (1-((abs(df["cell_count_0.2_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_cell_count'])).mean())
    # print("cell_count_0.3 ", (1-((abs(df["cell_count_0.3_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_cell_count'])).mean())
    # print("cell_count_0.4 ", (1-((abs(df["cell_count_0.4_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_cell_count'])).mean())
    # print("cell_count_0.5 ", (1-((abs(df["cell_count_0.5_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_cell_count'])).mean())
    # print("cell_count_0.6 ", (1-((abs(df["cell_count_0.6_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_cell_count'])).mean())
    # print("cell_count_0.7 ", (1-((abs(df["cell_count_0.7_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_cell_count'])).mean())
    # print("cell_count_0.8 ", (1-((abs(df["cell_count_0.8_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_cell_count'])).mean())
    # print("cell_count_0.9 ", (1-((abs(df["cell_count_0.9_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_cell_count'])).mean())
    #
    # print("thick_cell_count_0.1 ", (1-((abs(df["cell_count_0.1_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_thick_cell_count'])).mean())
    # print("thick_cell_count_0.2 ", (1-((abs(df["cell_count_0.2_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_thick_cell_count'])).mean())
    # print("thick_cell_count_0.3 ", (1-((abs(df["cell_count_0.3_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_thick_cell_count'])).mean())
    # print("thick_cell_count_0.4 ", (1-((abs(df["cell_count_0.4_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_thick_cell_count'])).mean())
    # print("thick_cell_count_0.5 ", (1-((abs(df["cell_count_0.5_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_thick_cell_count'])).mean())
    # print("thick_cell_count_0.6 ", (1-((abs(df["cell_count_0.6_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_thick_cell_count'])).mean())
    # print("thick_cell_count_0.7 ", (1-((abs(df["cell_count_0.7_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_thick_cell_count'])).mean())
    # print("thick_cell_count_0.8 ", (1-((abs(df["cell_count_0.8_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_thick_cell_count'])).mean())
    # print("thick_cell_count_0.9 ", (1-((abs(df["cell_count_0.9_threshold"] - df['ground_truth_cell_count']))/ df['ground_truth_thick_cell_count'])).mean())

    print("Average accuracy overall: ",
          df["accuracy?"].mean())

    print("evals: ", evals)




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
    batch_size = 20

    im_size = (96, 96)
    data_folder = "/home/iolie/PhD_Thesis_Data/epithelial_cell_border_identification"

    #dataset = get_base_dataset(data_folder)
    saved_test = 'saved_testset.json'
    test_data = json.load(open(saved_test))

    model_dir = "/home/iolie/PycharmProjects/THESIS/savedmodels_unet_29/titletraining_weightsatloss_0.32"
    main_dir = "/home/iolie/Desktop/THESIS IMAGES/"
    save_dir = os.path.join(main_dir, model_dir.split("/")[-2], model_dir.split("/")[-1])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    batch_gen = emb_image_batch_generator_v2(data_folder, test_data, batch_size, im_size, MIN_SIZE=im_size[0])
    raw_image_batch, ground_truth_batch = next(batch_gen)
    pixel_and_cell_count_test(raw_image_batch, ground_truth_batch, model_dir, save_dir, demo = False, rsz = im_size, batch_size =2)

