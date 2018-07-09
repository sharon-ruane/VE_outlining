import os
import numpy as np
from PIL import Image
from U_NET import myUnet
from keras.models import model_from_json
from U_NET_training_functions import emb_image_batch_generator, make_test_batch

model_dir = "/home/iolie/PycharmProjects/THESIS/savedmodels_unet_3/titletraining_weightsatloss_0.69"
data_folder = "/home/iolie/PhD_Thesis_Data/epithelial_cell_border_identification"
emb_list = os.listdir(data_folder)
#print(emb_list)
emb_list.remove('.DS_Store')
#print(emb_list)

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

batch_size = 72
size_to_resize_to = (176,176)

## need to sep out the lists -- allocate 2 for validation
training_generator = emb_image_batch_generator(data_folder, emb_list, batch_size, size_to_resize_to, opt_z_stack_dict)
validation_generator = emb_image_batch_generator(data_folder, emb_list, batch_size, size_to_resize_to, opt_z_stack_dict)
#
sharon_unet = myUnet(lowest_loss=2)
while True:
    sharon_unet.train(training_generator, validation_generator)
#
# test_embs = "/home/iolie/PhD_Thesis_Data/epithelial_cell_border_test_embs"
# test_batch_size = 5
# test_batch, ground_truth = make_test_batch(test_embs, test_batch_size, opt_z_stack_dict)
# # for pic in test_batch:
# #     img = Image.fromarray(pic.reshape(80,80))
# #     img.show()
# #
# # for pic in ground_truth:
# #     img = Image.fromarray(pic)
# #     img.show()
#
#
# json_file = open(os.path.join(model_dir, "model.json"), 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights(os.path.join(model_dir, "weights.h5"))
# print("Loaded model from disk")
#
# print(test_batch[0].shape)
# outlines_test = model.predict(np.asarray(test_batch), batch_size=5, verbose=1, steps=None)
# #
# print(outlines_test.shape)
#
# for i in range(outlines_test.shape[0]):
#     pred = outlines_test[i]
#     mask = pred.reshape(100,100)
#
#     mask = mask
#     mask[mask > 0.5] = 255
#     mask[mask <= 0.5] = 0
#     imgr = Image.fromarray(mask)
#     imgr.show()
# #
#
#
#



"""
while True:
    callback = model.fit_generator(training_generator, validation_data=validation_generator, steps_per_epoch=1000,
                                   epochs=1, max_queue_size=50, validation_steps=50)
    loss = float(callback.history['loss'][0])
    val_loss = float(callback.history['val_loss'][0])
    if loss < lowest_loss - 0.02:
        weightfolder = 'savedmodels_x/titletraining_weightsatloss_{0:.2f}'.format(loss)
        if not os.path.isdir(weightfolder):
            os.makedirs(weightfolder)
        print('Saving {}/weights.h5'.format(weightfolder))
        model.save_weights(weightfolder + '/weights.h5')
        open(weightfolder + '/model.json', 'w').write(model.to_json())
        # picklefile = open(weightfolder + '/indices.pickle', 'wb')
        # pickle.dump((char_to_index, index_to_char, first_char_probs), picklefile)  ## what needs to go here??
        # picklefile.close()
        lowest_loss = loss
"""