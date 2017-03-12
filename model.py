'''
Model based on NVIDIA http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
'''

from keras.models import Sequential
from keras.layers import Dense, Cropping2D
from keras.layers import Lambda, Convolution2D, Flatten, Dropout
from keras.models import load_model
from keras.optimizers import Adam


class CNNModel:

    def __init__(self, height = 66, width = 200, channels = 3):
        self.height = height
        self.width = width
        self.channels = channels

    def createModel(self):
        self.model = Sequential()

        # Preprocess incoming data, centered around zero with small standard deviation
        self.model.add(Lambda(lambda x: x / 127.5 - 1.,
                         input_shape=(self.height, self.width, self.channels),
                         output_shape=(self.height, self.width, self.channels)))

        # 5 Convolution layer
        self.model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu', border_mode='valid'))
        self.model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu', border_mode='valid'))
        self.model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu', border_mode='valid'))
        self.model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu', border_mode='valid'))
        self.model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu', border_mode='valid'))

        self.model.add(Flatten())

        self.model.add(Dense(1164, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1))

    def get_model(self):
        return self.model

    def save_model(self, h5file = 'model.h5'):
        # serialize weights to HDF5
        self.model.save("model.h5")
        print("Saved model to disk")

    def load_model(self, h5_file):

        self.model = load_model(h5_file)
        print("Loaded model from disk")

    def summary(self):
        self.model.summary()

    def train(self, train_generator, validation_generator, train_samples, validation_samples, learning_rate = 0.000001, loss = 'mse', nb_epoch = 3):
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = nb_epoch)