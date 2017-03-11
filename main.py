import argparse, csv
from model import CNNModel
import data_preprocessor as dp
import visualize as viz
import numpy as np

def visualize_data(samples,  train_data):

    angles = []
    for line in samples:
        angles.append(float(line[3]))

    viz.data_histogram(angles, "samples", "Angle distribution in the original data")

    viz.data_histogram(train_data, "training_data", "Angle distribution in the preprocessed training data")


def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning')
    parser.add_argument('data', help="Path to the data folder root")
    parser.add_argument('--load', nargs = '*', help="Loads a existing model h5 and json from given path")
    args = parser.parse_args()

    # Initialize CNNModel instance
    cnn = CNNModel()
    print(args.load)
    if args.load:
        model_file = args.load[0]
        model = cnn.load_model(model_file)
    else:
        # Initialize a new model
        cnn.createModel()
        print("Initializing a new model")
        cnn.summary()

    samples = []
    with open(args.data +'/driving_log.csv') as csvfile:
        print("driving log loaded")
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            samples.append(line)

    # Split the data into training and validation set
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print("Training samples count {0} , Validation samples count {1}".format(len(train_samples), len(validation_samples)))
    train_generator = dp.generator(train_samples, batch_size=64, data_folder=args.data)
    validation_generator = dp.generator(validation_samples, batch_size=64, data_folder=args.data)

    angles2 = []
    for value in train_generator:
        for angle in value[1]:
            angles2.append(angle)
        if len(angles2) >= len(train_samples):
                break

    print(len(angles2))
    visualize_data(samples,angles2)

    print("Training data...")
    cnn.train(train_generator=train_generator,
              validation_generator=validation_generator,
              train_samples=train_samples, validation_samples=validation_samples)

    cnn.save_model()

if __name__ == "__main__":
    main()