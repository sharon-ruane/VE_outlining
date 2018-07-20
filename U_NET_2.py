### citation == https://github.com/zhixuhao/unet/blob/master/unet.py

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as kerasing
from PIL import Image


class myUnet(object):
    def __init__(self, model_save_dir, img_rows=64, img_cols=64, lowest_loss=2):
        print ("THIS IS UNET 2")
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.lowest_loss = lowest_loss
        self.current_loss = None
        self.model_save_dir = model_save_dir

    # def load_data(self):
    #     mydata = dataProcess(self.img_rows, self.img_cols)
    #     imgs_train, imgs_mask_train = mydata.load_train_data()
    #     imgs_test = mydata.load_test_data()
    #     return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))

        conv1 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.25)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.25)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
                      metrics=['accuracy', 'mae', 'mse'])

        return model

    def train(self, training_generator, validation_generator):
        print("loading data")
        # imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")

        while True:
            model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
            print('Fitting model...')
            # model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=10, verbose=1, validation_split=0.2, shuffle=True,
            #           callbacks=[model_checkpoint])
            callback = model.fit_generator(training_generator, validation_data=validation_generator, steps_per_epoch=200,
                                           epochs=1, max_queue_size=50, validation_steps=20)

            self.current_loss = float(callback.history['loss'][0])
            print("current_loss: {}").format(self.current_loss)

            if self.current_loss < self.lowest_loss - 0.02:
                weightfolder = os.path.join(self.model_save_dir,
                                            'titletraining_weightsatloss_{0:.2f}'.format(
                                                self.current_loss))
                if not os.path.isdir(weightfolder):
                    os.makedirs(weightfolder)
                print('Saving {}/weights.h5'.format(weightfolder))
                model.save_weights(weightfolder + '/weights.h5')
                open(weightfolder + '/model.json', 'w').write(model.to_json())
                # picklefile = open(weightfolder + '/indices.pickle', 'wb')
                # pickle.dump((char_to_index, index_to_char, first_char_probs), picklefile)  ## what needs to go here??
                # picklefile.close()
                self.lowest_loss = self.current_loss



            #print('predict test data')
            #imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
            #np.save('../results/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("array to image")
        imgs = np.load('imgs_mask_test.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = Image.fromarray(img)
            img.save("../results/%d.jpg" % (i))


if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    myunet.save_img()

