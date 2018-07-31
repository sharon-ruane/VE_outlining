import argparse
import json
import random
import os
from sklearn.model_selection import train_test_split

import testing_training as tt
from U_NET_training_functions import get_base_dataset


batch_size = 6
size_to_resize_to = (80, 80)
data_folder = "/home/iolie/PhD_Thesis_Data/epithelial_cell_border_identification"

# WARNING THIS WILL OVERWRITE YOUR SAVE LIST OF INPUTS
regenerate_test_split = False



model_dir = "/home/iolie/PycharmProjects/THESIS/savedmodels_unet_15/"
image_folder = "/home/iolie/Desktop/THESIS IMAGES/"
test_embs = "/home/iolie/PhD_Thesis_Data/epithelial_cell_border_test_embs"

_Z_STACK = {
    "AAntnew33-47.lsm (cropped)": 3,
    "ANTERIOR \"EMB 4\" Nov. 28th Emb (2)_L5_Sum.lsm (spliced)": 3,
    "Anterior = Embryo 5\" Feb 20th": 4,
    "USEAnt(potential)2_march__t2.lsm (spliced) (cropped)": 3,
    "LATERAL \"EMB 3\" Oct 2nd Emb (1)_L3_Sum.lsm (spliced)": 3,
    "LATERAL\"EMB 6\" Nov. 28th Emb (2)_L7_Sum.lsm ": 3,
    "LATERAL \"EMB 9\" Dec 15th Emb (1)_L12_Sum.lsm (spliced)": 4,
    "LATERAL \"EMB 12\" Nov. 28th Emb (2)_L12_Sum.lsm ": 4,
    "Outline this movie tp 6-22 posterior copy": 5,
    "POSTERIOR = \"Embryo 2\", Nov. 28th Emb (2)_L3_Sum.lsm (spliced)": 3,
    "EARLY Posterior = \"Embryo 6\", Feb. 20th Emb (1)_L6_Sum.lsm (spliced)": 3,
    "LATE Posterior = \"Embryo 6\", Feb. 20th Emb (1)_L6_Sum.lsm (spliced)": 4
}
model_save_dir = 'savedmodels_unet_23'


def main(args=None, train=True, test=False):
    dataset = get_base_dataset(data_folder)

    saved_train = 'saved_trainset.json'  # choo choooo
    saved_test = 'saved_testset.json'
    saved_val = 'saved_valset.json'
    if regenerate_test_split:
        train_data, test_data = train_test_split(dataset, test_size=0.2)
        test_data, val_data = train_test_split(test_data, test_size=0.5)
        j_train = json.dumps(train_data)
        j_test = json.dumps(test_data)
        j_val = json.dumps(val_data)
        with open(saved_train, 'w') as jout:
            jout.write(j_train)
        with open(saved_test, 'w') as jout:
            jout.write(j_test)
        with open(saved_val, 'w') as jout:
            jout.write(j_val)
    else:
        train_data = json.load(open(saved_train))
        test_data = json.load(open(saved_test))
        val_data = json.load(open(saved_val))

    #TODO: split the base dataset train/val/test

    if train:
        tt.train_me(data_folder, train_data, val_data,
                    model_save_dir, batch_size=batch_size,
                    size_to_resize_to=size_to_resize_to, unet_to_load=5)
    if test:
        tt.test_me(image_folder, model_dir, test_embs, _Z_STACK,
                   size_to_resize_to=size_to_resize_to)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('input_dir', type=str,
    #                     help='Input directory with unaligned images.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    #main(args)
    main(train=True, test=False)
