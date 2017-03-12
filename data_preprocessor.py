import cv2
import numpy as np
import random
from sklearn.utils import shuffle

CORRECTION_ANGLE = 0.2
LEFT_IMAGE_INDEX = 1
RIGHT_IMAGE_INDEX = 2
CENTER_IMAGE_INDEX = 0

#https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.11yb60izi
def trans_image(image, steer, trans_range):
    rows, cols, ch = image.shape
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, steer_ang

def crop_resize(image):
    image = image[50: 140, 20: 300]
    # Resize to NVIDIA model input size
    return cv2.resize(image, (200, 66))

def flip_image(image, measurement):
    image_flipped = cv2.flip(image, 1)
    measurement_flipped = -measurement
    return image_flipped, measurement_flipped

def generator(samples, batch_size=32, data_folder='./', translate = True, center_image = False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            folder_path = './' + data_folder + '/'
            images = []
            angles = []
            for batch_sample in batch_samples:

                image_index = random.randint(0, 2)
                if center_image == True:
                    image_index = CENTER_IMAGE_INDEX

                steering_angle = float(batch_sample[3])
                img_file = folder_path + batch_sample[image_index].strip()
                image = cv2.imread(img_file)

                if random.randint(0, 1) == 0:
                    image, steering_angle = flip_image(image, steering_angle)

                train_image = crop_resize(cv2.cvtColor(image, cv2.COLOR_BGR2YUV))

                if train_image is not None:
                    if translate == True:
                        train_image, steering_angle = trans_image(train_image, steering_angle, 100)

                    images.append(train_image)
                    angles.append(steering_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


