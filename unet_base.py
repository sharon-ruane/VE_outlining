### citation == https://github.com/zhixuhao/unet/blob/master/unet.py
import abc
import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from PIL import Image

class unet_base(object):
    def __init__(self,
                 model_save_dir='./saved_model',
                 lowest_loss=2,
                 epochs=1,
                 steps_per_epoch=200,
                 validation_steps=20,
                 max_queue_size=50
                 ):
        self.model_save_dir = model_save_dir
        self.lowest_loss = lowest_loss
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.max_queue_size = max_queue_size
        self.current_loss = None

    @abc.abstractmethod
    def get_unet(self):
        raise NotImplementedError

    def train(self, training_generator, validation_generator):
        model = self.get_unet()
        while True:
            model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
            callback = model.fit_generator(training_generator,
                                           validation_data=validation_generator,
                                           epochs=self.epochs,
                                           steps_per_epoch=self.steps_per_epoch,
                                           validation_steps=self.validation_steps,
                                           max_queue_size=self.max_queue_size)

            self.current_loss = float(callback.history['loss'][0])
            print("current_loss: {}".format(self.current_loss))
            if self.current_loss < self.lowest_loss - 0.02:
                weightfolder = os.path.join(self.model_save_dir,
                                'titletraining_weightsatloss_{0:.2f}'.format(
                                    self.current_loss))
                if not os.path.isdir(weightfolder):
                    os.makedirs(weightfolder)
                print('Saving {}/weights.h5'.format(weightfolder))
                model.save_weights(weightfolder + '/weights.h5')
                open(weightfolder + '/model.json', 'w').write(model.to_json())
                self.lowest_loss = self.current_loss

    def save_img(self):
        imgs = np.load('imgs_mask_test.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = Image.fromarray(img)
            img.save("../results/%d.jpg" % (i))
