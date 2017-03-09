import cv2
import numpy as np
import random
from sklearn.utils import shuffle

CORRECTION_ANGLE = 0.25
LEFT_IMAGE_INDEX = 1
RIGHT_IMAGE_INDEX = 2
CENTER_IMAGE_INDEX = 0

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
            folder_path = './' + data_folder + '/'
            images = []
            angles = []
            for batch_sample in batch_samples:

                randomness = random.randint(0, 3)

                train_img = None
                steering_angle = None

                angle = float(batch_sample[3])
                if abs(angle) < 0.1 and randomness == 0:
                    continue

                for index in range(0, 3):
                    img_file = folder_path + batch_sample[index].strip()
                    image= cv2.imread(img_file)

                    if index == CENTER_IMAGE_INDEX and randomness == CENTER_IMAGE_INDEX:
                        if random.randint(0, 1) == 0:
                            train_img = crop_resize(cv2.cvtColor(image, cv2.COLOR_BGR2YUV))
                            steering_angle = angle
                        else:
                            image_flipped, steering_angle = flip_image(image, angle)
                            train_img = crop_resize(cv2.cvtColor(image_flipped, cv2.COLOR_BGR2YUV))

                    if index == LEFT_IMAGE_INDEX and randomness == LEFT_IMAGE_INDEX:
                        angle = angle + CORRECTION_ANGLE
                        if random.randint(0, 1) == 0:
                            train_img = crop_resize(cv2.cvtColor(image, cv2.COLOR_BGR2YUV))
                            steering_angle = angle
                        else:
                            image_flipped, steering_angle = flip_image(image, angle)
                            train_img = crop_resize(cv2.cvtColor(image_flipped, cv2.COLOR_BGR2YUV))

                    if index == RIGHT_IMAGE_INDEX and randomness == RIGHT_IMAGE_INDEX:
                        angle = angle - CORRECTION_ANGLE
                        if random.randint(0, 1) == 0:
                            train_img = crop_resize(cv2.cvtColor(image, cv2.COLOR_BGR2YUV))
                            steering_angle = angle
                        else:
                            image_flipped, steering_angle = flip_image(image, angle)
                            train_img = crop_resize(cv2.cvtColor(image_flipped, cv2.COLOR_BGR2YUV))
                if train_img is not None:
                    images.append(train_img)
                    angles.append(steering_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


