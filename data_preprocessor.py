import cv2
import numpy as np
from sklearn.utils import shuffle

def crop_resize(image):
    image = image[50: 140, 20: 300]
    # Resize to NVIDIA model input size
    return cv2.resize(image, (200, 66))

def flip_image(image, measurement):
    image_flipped = cv2.flip(image, 1)
    measurement_flipped = -measurement
    return image_flipped, measurement_flipped



def generator(samples, batch_size=32, data_folder='./'):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = data_folder + '/IMG/'+batch_sample[0].split('/')[-1]
                center_image_orig = cv2.imread(name)
                if center_image_orig is None:
                    print(name)
                    continue
                center_image = cv2.cvtColor(center_image_orig, cv2.COLOR_BGR2YUV)
                center_image = crop_resize(center_image)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                #Fliiping
                image_flipped, measurement_flipped = flip_image(center_image_orig, center_angle)
                image_flipped = cv2.cvtColor(image_flipped, cv2.COLOR_BGR2YUV)
                image_flipped = crop_resize(image_flipped)
                images.append(image_flipped)
                angles.append(measurement_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


