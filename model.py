'''
Model based on NVIDIA http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
'''

from keras.models import Sequential
from keras.layers import Dense, Cropping2D
from keras.layers import Lambda, Convolution2D, Flatten, Dropout
from keras.models import model_from_json

class CNNModel:

    def __init__(self, height, width, channels):
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
        self.model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu', border_mode='valid'))
        self.model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu', border_mode='valid'))

        self.model.add(Flatten())

        self.model.add(Dense(1164, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1, activation='tanh'))

    def get_model(self):
        return self.model


    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

    def load_model(self, json_file, h5_file):
        # load json and create model
        json_file = open(json_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(h5_file)
        print("Loaded model from disk")

    def summary(self):
        self.model.summary()