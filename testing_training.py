from keras.models import model_from_json
import numpy as np
import os
from PIL import Image
import uuid
from U_NET import myUnet as myUnet1
from U_NET_2 import myUnet as myUnet2
from U_NET_3 import myUnet as myUnet3
from U_NET_4 import myUnet as myUnet4
from U_NET_5 import myUnet as myUnet5
from U_NET_training_functions import emb_image_batch_generator_v2

def train_me(data_folder, train_emb_list, val_emb_list, model_save_dir,
             batch_size=32, size_to_resize_to=(96,96), unet_to_load=1):

    # do the splitting
    training_generator = emb_image_batch_generator_v2(data_folder, train_emb_list,
                                                   batch_size,
                                                   size_to_resize_to)
    validation_generator = emb_image_batch_generator_v2(data_folder, val_emb_list,
                                                     batch_size,
                                                     size_to_resize_to)
    if unet_to_load == 1:
        unet = myUnet1(model_save_dir, lowest_loss=2,
                       img_rows=size_to_resize_to[0],
                       img_cols=size_to_resize_to[1])
        unet.train(training_generator, validation_generator)
    if unet_to_load == 2:
        unet = myUnet2(model_save_dir, lowest_loss=2,
                       img_rows=size_to_resize_to[0],
                       img_cols=size_to_resize_to[1])
        unet.train(training_generator, validation_generator)
    if unet_to_load == 3:
        unet = myUnet3(model_save_dir, lowest_loss=2,
                       img_rows=size_to_resize_to[0],
                       img_cols=size_to_resize_to[1])
        unet.train(training_generator, validation_generator)
    if unet_to_load == 4:
        unet = myUnet4(model_save_dir, lowest_loss=2,
                       img_rows=size_to_resize_to[0],
                       img_cols=size_to_resize_to[1])
        unet.train(training_generator, validation_generator)
    if unet_to_load == 5:
        unet = myUnet5(model_save_dir, lowest_loss=2,
                        img_rows=size_to_resize_to[0],
                        img_cols=size_to_resize_to[1])
        unet.train(training_generator, validation_generator)

def test_me(image_folder, model_dir, test_data_path, opt_z_stack_dict, size_to_resize_to=(96, 96)):
    pass
    #
    # save_dir = os.path.join(image_folder, model_dir.split("/")[-2], model_dir.split("/")[-1])
    # rsz = size_to_resize_to
    # test_batch_size = 10
    # test_batch, ground_truth = make_test_batch(test_data_path, test_batch_size, size_to_resize_to, opt_z_stack_dict)
    # batch_id = str(uuid.uuid4())
    # min_threshold = 0.2
    # max_threshold = 0.8
    #
    # for i,pic in enumerate(test_batch):
    #     img = Image.fromarray((pic*255).reshape(rsz[0], rsz[1]))
    #     batch_dir = os.path.join(save_dir, str(i))
    #     if not os.path.isdir(batch_dir):
    #         os.makedirs(batch_dir)
    #     img.convert('RGB').save(os.path.join(batch_dir, batch_id + "real_image_" + str(i) + ".png"), "PNG")
    #
    #     gt = Image.fromarray(ground_truth[i])
    #     gt.convert('RGB').save(os.path.join(batch_dir, batch_id + "ground_truth_" + str(i) + ".png"), "PNG")
    #
    # json_file = open(os.path.join(model_dir, "model.json"), 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    # model.load_weights(os.path.join(model_dir, "weights.h5"))
    # outlines_test = model.predict(test_batch, batch_size=test_batch_size , verbose=1, steps=None)
    # for i in range(outlines_test.shape[0]):
    #     pred = outlines_test[i]
    #     mask = (pred).reshape(rsz[0], rsz[1])
    #
    #     batch_dir = os.path.join(save_dir, str(i))
    #
    #     imgx = Image.fromarray(mask*255)
    #     imgx.convert('RGB').save(os.path.join(batch_dir, batch_id + "predicted_x255_image_" + str(i) + ".png"), "PNG")
    #
    #     for j in (np.arange(min_threshold, max_threshold, 0.1)).astype(np.float32):
    #         mask = np.copy(pred.reshape(rsz[0], rsz[1]))
    #         mask[mask > j] = 255
    #         mask[mask <= j] = 0
    #         imgr = Image.fromarray(mask)
    #         imgr.convert('RGB').save(os.path.join(batch_dir, batch_id + "_predicted" + str(i) + "_threshold" + str(j) + ".png"), "PNG")
    """
    weight_dir_list = [x for x in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, x))]
    for d in weight_dir_list:
        save_dir = os.path.join(image_folder, model_dir.split("/")[-1], d)
        rsz = size_to_resize_to
        test_batch_size = 10
        test_batch, ground_truth = make_test_batch(test_data_path, test_batch_size, size_to_resize_to, opt_z_stack_dict)
        batch_id = str(uuid.uuid4())
        min_threshold = 0.2
        max_threshold = 0.8

        for i,pic in enumerate(test_batch):
            img = Image.fromarray((pic*255).reshape(rsz[0], rsz[1]))
            batch_dir = os.path.join(save_dir, str(i))
            if not os.path.isdir(batch_dir):
                os.makedirs(batch_dir)
            img.convert('RGB').save(os.path.join(batch_dir, batch_id + "real_image_" + str(i) + ".png"), "PNG")

            gt = Image.fromarray(ground_truth[i])
            gt.convert('RGB').save(os.path.join(batch_dir, batch_id + "ground_truth_" + str(i) + ".png"), "PNG")

        json_file = open(os.path.join(model_dir, d, "model.json"), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(os.path.join(model_dir, d, "weights.h5"))
        outlines_test = model.predict(test_batch, batch_size=test_batch_size , verbose=1, steps=None)
        for i in range(outlines_test.shape[0]):
            pred = outlines_test[i]
            mask = (pred).reshape(rsz[0], rsz[1])

            batch_dir = os.path.join(save_dir, str(i))

            imgx = Image.fromarray(mask*255)
            imgx.convert('RGB').save(os.path.join(batch_dir, batch_id + "predicted_x255_image_" + str(i) + ".png"), "PNG")

            for j in (np.arange(min_threshold, max_threshold, 0.1)).astype(np.float32):
                mask = np.copy(pred.reshape(rsz[0], rsz[1]))
                mask[mask > j] = 255
                mask[mask <= j] = 0
                imgr = Image.fromarray(mask)
                imgr.convert('RGB').save(os.path.join(batch_dir, batch_id + "_predicted" + str(i) + "_threshold" + str(j) + ".png"), "PNG")

    """
