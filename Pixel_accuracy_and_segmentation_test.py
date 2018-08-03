import os
import json
import random
import logging
import numpy as np
import pandas as pd
import mahotas as mh
from keras.optimizers import Adam
from keras import metrics, Model
from PIL import Image, ImageOps, ImageFilter, ImageTk, Image
from keras.models import model_from_json
from sklearn.metrics import accuracy_score as acc
from U_NET_training_functions import get_base_dataset, emb_image_batch_generator_v2
import Tkinter as tk

log = logging.getLogger("")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

def accuracy(y_true, y_pred):
    return acc(np.rint(y_true.ravel()), np.rint(y_pred.ravel()))


def precision(y_true, y_pred):
    true_positives = np.sum(np.rint(y_true) * np.rint(y_pred))
    predicted_positives = np.sum(np.rint(y_pred))
    precision = true_positives / predicted_positives
    return precision


def recall(y_true, y_pred):
    true_positives = np.sum(np.rint(y_true) * np.rint(y_pred))
    possible_positives = np.sum(np.rint(y_true))
    recall = true_positives / possible_positives
    return recall

def f_measure(y_true, y_pred, beta=1):
    """
    Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    """
    p = precision(np.rint(y_true), np.rint(y_pred))
    r = recall(np.rint(y_true), np.rint(y_pred))
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r)
    return fbeta_score

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

    columns = [ 'ground_truth_cell_count', 'ground_truth_thick_cell_count',
                'cell_count_0.1_threshold', 'cell_count_0.2_threshold',
                'cell_count_0.3_threshold', 'cell_count_0.4_threshold',
                'cell_count_0.5_threshold', 'cell_count_0.6_threshold',
                'cell_count_0.7_threshold', 'cell_count_0.8_threshold',
                'cell_count_0.9_threshold', 'pixel_accuracy_0.1_threshold',
                'pixel_accuracy_0.2_threshold', 'pixel_accuracy_0.3_threshold',
                'pixel_accuracy_0.4_threshold', 'pixel_accuracy_0.5_threshold',
                'pixel_accuracy_0.6_threshold', 'pixel_accuracy_0.7_threshold',
                'pixel_accuracy_0.8_threshold', 'pixel_accuracy_0.9_threshold',

                'recall_0.2_threshold', 'recall_0.3_threshold',
                'recall_0.4_threshold', 'recall_0.5_threshold',
                'recall_0.6_threshold', 'recall_0.7_threshold',
                'recall_0.8_threshold', 'recall_0.9_threshold',

                'precision_0.2_threshold', 'precision_0.3_threshold',
                'precision_0.4_threshold', 'precision_0.5_threshold',
                'precision_0.6_threshold', 'precision_0.7_threshold',
                'precision_0.8_threshold', 'precision_0.9_threshold',
                'prediction_rmse', "non_cat_acc"]

    df = pd.DataFrame(columns=columns)
    print(df.index)
    print(range(outlines_pred.shape[0]))

    modified_batch = []

    for i in range(outlines_pred.shape[0]):
        raw = predictions_batch[i].reshape(rsz)
        gt = ground_truth_batch[i].reshape(rsz)
        prediction = outlines_pred[i].reshape(rsz)
        df.loc[i, "non_cat_acc"] = accuracy(gt, prediction)

        _raw = Image.fromarray(raw*255).convert('RGB')
        _pred = Image.fromarray(prediction * 255).convert('RGB')
        _gtimg = Image.fromarray(gt*255).convert('RGB')
        _thick = Image.fromarray(gt*255).filter(ImageFilter.MinFilter(3)).convert('RGB')

        # return images for display
        this_batch = [_raw, _pred, _gtimg, _thick]

        # save images for posterity
        _raw.save(os.path.join(save_dir, str(i) + "_raw_image_" + ".png"), "PNG")
        _pred.save(os.path.join(save_dir, str(i) + "_PREDICTION_" + ".png"), "PNG")
        _gtimg.save(os.path.join(save_dir,str(i) + "_gt_image"  + ".png"), "PNG")
        _thick.save(os.path.join(save_dir, str(i) +"_gt_thick_image" +  ".png"), "PNG")

        _, gt_cell_count = count_cells(gt)
        gt_thick = np.asarray(Image.fromarray(gt).filter(ImageFilter.MinFilter(3)))
        _, gt_thick_cell_count = count_cells(gt_thick)

        df.loc[i, 'ground_truth_cell_count'] = gt_cell_count
        df.loc[i, 'ground_truth_thick_cell_count'] = gt_thick_cell_count
        df.loc[i, "prediction_rmse"] = np.sqrt(np.mean((prediction - gt) ** 2))

        # print(np.sqrt(np.mean((prediction - gt) ** 2)))

        min_threshold = 0.1
        max_threshold = 1.0
        counter = 1

        for j in (np.arange(min_threshold, max_threshold, 0.1)).astype(np.float32):
            mask = np.copy(prediction.reshape(rsz[0], rsz[1]))
            mask[mask > j] = 1
            mask[mask <= j] = 0
            imgr = Image.fromarray(mask*255).convert('RGB')
            imgr.save(os.path.join(save_dir, str(i) + "_predicted"+ str(j) + "_threshold" + ".png"), "PNG")
            # return for display
            this_batch.append(imgr)

            labels, cell_count = count_cells(mask)
            lbl_img = Image.fromarray((labels*20)).convert('RGB')
            lbl_img.save(os.path.join(save_dir, str(i) + "_LABELS" + str(j) + "_threshold" + ".png"), "PNG")
            # return for display
            this_batch.append(lbl_img)

            df.loc[i, "recall_{}_threshold".format(float(counter) / 10)] = recall(gt, mask)
            df.loc[i, "precision_{}_threshold".format(float(counter) / 10)] = precision(gt, mask)

            # print("r:", r)
            # print("p:", p)
            df.loc[i, "f_measure"] = f_measure(gt, mask)

            df.loc[i, "cell_count_{}_threshold".format(
                float(counter) / 10)] = cell_count
            df.loc[i, "pixel_accuracy_{}_threshold".format(float(counter) / 10)] = acc(np.uint8(gt).ravel(), np.uint8(mask).ravel())
            print("pixel_accuracy_{}_threshold".format(float(counter) / 10), acc(np.uint8(gt).ravel(), np.uint8(mask).ravel()))
            counter += 1
        modified_batch.append(this_batch)

    df.to_csv(os.path.join(save_dir, "parameters"))

    # debug printing 0.1 to 0.9
    cutoff_range = np.arange(0.1, 1.0, 0.1)
    for c in cutoff_range:
        print("Average pixel_accuracy_{}_threshold: ".format(c),
              df["pixel_accuracy_{}_threshold".format(c)].mean())
    for c in cutoff_range:
        print("cell_count_rmse_{} ".format(c), np.sqrt(np.mean(
            df["cell_count_{}_threshold".format(c)] - df[
                'ground_truth_cell_count']) ** 2))
    for c in cutoff_range:
        print("thick_cell_count_rmse_{} :".format(c), np.sqrt(np.mean(
            df["cell_count_{}_threshold".format(c)] - df[
                'ground_truth_thick_cell_count']) ** 2))
    for c in cutoff_range:
        print("cell_count_{} ".format(c), (1 - ((abs(
            df["cell_count_{}_threshold".format(c)] - df[
                'ground_truth_cell_count'])) / df[
                                    'ground_truth_cell_count'])).mean())
    for c in cutoff_range:
        print("thick_cell_count_{} ".format(c), (1 - ((abs(
            df["cell_count_{}_threshold".format(c)] - df[
                'ground_truth_cell_count'])) / df[
                                    'ground_truth_thick_cell_count'])).mean())
    print("Average accuracy overall: ", df["non_cat_acc"].mean())
    print("F-score overall: ", f_measure(ground_truth_batch, predictions_batch))
    print("F-score average: ", df['f_measure'].mean())
    print("r:", recall(ground_truth_batch, predictions_batch))
    print("p", precision(ground_truth_batch, predictions_batch))
    print("evals: ", evals)

    return modified_batch

if __name__ == '__main__':
    batch_size = 5
    im_size = (96, 96)
    data_folder = "/home/iolie/PhD_Thesis_Data/epithelial_cell_border_identification"
    saved_test = 'saved_testset.json'
    test_data = json.load(open(saved_test))
    model_dir = "/home/iolie/PycharmProjects/THESIS/savedmodels_unet_29/titletraining_weightsatloss_0.30"
    main_dir = "/home/iolie/Desktop/THESIS IMAGES/"
    save_dir = os.path.join(main_dir, model_dir.split("/")[-2], model_dir.split("/")[-1], "X")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    batch_gen = emb_image_batch_generator_v2(data_folder, test_data, batch_size, im_size, MIN_SIZE=im_size[0])
    raw_image_batch, ground_truth_batch = next(batch_gen)

    display_set = pixel_and_cell_count_test(raw_image_batch, ground_truth_batch, model_dir, save_dir, demo = False, rsz = im_size, batch_size =2)

    window = tk.Tk()
    window.title("data")
    window.geometry("1200x800")
    window.configure(background='grey')
    canvas = tk.Canvas(window, width=2000, height=2000)
    canvas.delete("all")
    canvas.pack()

    def close_window():
        window.destroy()

    display_index = 0
    tmp = []  # need to store reference to image outside loops to avoid garbage collection

    def next_window(display_index):
        canvas.delete("all")
        canvas.pack()
        canvas.create_text(25, 25, fill="black", font="Times 20 bold",
                           anchor="nw", text="Test Dataset {}".format(display_index))
        if display_index < batch_size -1:
            state = "normal"
        else:
            state = "disabled"
        button = tk.Button(canvas, text="Next", state=state, command=lambda: next_window(display_index + 1))
        button.configure(width=10, activebackground="#33B5E5", relief='raised')
        button_window = canvas.create_window(500, 500, anchor='nw',
                                             window=button)

        button1 = tk.Button(canvas, text="Close", command=close_window)
        button1.configure(width=10, activebackground="#33B5E5", relief='raised')
        button_window1 = canvas.create_window(620, 500, anchor='nw',
                                             window=button1)

        # display 0 to 3 are main images
        main_row = display_set[display_index][:4]
        # take every second one to get 2 sets of 9
        treshholds = display_set[display_index][4:50:2]
        counts = display_set[display_index][5:50:2]
        padding = 10
        top_offset = 100
        left_offset = 25

        for i, im in enumerate(main_row):
            img = ImageTk.PhotoImage(im)
            tmp.append(img)
            canvas.create_image(
                (left_offset + (i * (im_size[0] + padding)), top_offset), image=img, anchor="nw")

        for i, im in enumerate(treshholds):
            img = ImageTk.PhotoImage(im)
            tmp.append(img)
            canvas.create_image(
                (left_offset + (i * (im_size[0] + padding)), top_offset + im_size[0] + padding), image=img, anchor="nw")

        for i, im in enumerate(counts):
            img = ImageTk.PhotoImage(im)
            tmp.append(img)
            canvas.create_image(
                (left_offset + (i * (im_size[0] + padding)), top_offset + im_size[0] * 2 + padding * 2), image=img, anchor="nw")
        window.update_idletasks()
        window.update()

    # initial call
    next_window(display_index)
    window.mainloop()
