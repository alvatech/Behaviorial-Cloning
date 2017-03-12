import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import data_preprocessor as dp
import random

def data_histogram(data, file_name, title):
    plt.figure()
    plt.hist(data, bins=100, histtype="bar")  # plt.hist passes it's arguments to np.histogram
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    file_name = "examples/"+file_name
    plt.savefig(file_name)

def read_log():
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        print("driving log loaded")
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            samples.append(line)
    return samples

def angle_distribution():
    samples = read_log()
    angles = []
    for sample in samples:
        angles.append(float(sample[3]))

    data_histogram(angles, "samples", "Angle distribution in the original data: " + str(len(angles)))

    train_generator = dp.generator(samples, batch_size=64, data_folder="data")


    angles2 = []
    for value in train_generator:
        for angle in value[1]:
            angles2.append(angle)
        if len(angles2) >= len(samples):
                break
    data_histogram(angles2, "training_data", "Angle distribution in the augmented training data" +  str(len(angles)))

def showImage(image, file_name):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(image)
    file_name = "examples/" + file_name
    plt.savefig(file_name)

def test_processing():
    samples = read_log()

    for index in range(0,3):
        random_ind = random.randint(0, len(samples) -1)
        sample = samples[random_ind]
        file_name = "./data/" + sample[0].strip()
        image = cv2.imread(file_name)
        showImage(image, "original" + str(index))

        crop_image = dp.crop_resize(image)
        showImage(crop_image, "crop_resize" + str(index))

        flip_image_original, angle = dp.flip_image(image, 0)
        showImage(flip_image_original, "flip_original" + str(index))

        flip_image_crop, angle = dp.flip_image(crop_image, 0)
        showImage(flip_image_crop, "flip_crop" + str(index))

    print("Images saved")
if __name__ == "__main__":
    angle_distribution()
    test_processing()