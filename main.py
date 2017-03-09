import argparse, csv
from model import CNNModel
import data_preprocessor as dp

def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning')
    parser.add_argument('data', help="Path to the data folder root")
    parser.add_argument('--load', nargs = '*', help="Loads a existing model h5 and json from given path")
    args = parser.parse_args()

    # Initialize CNNModel instance
    cnn = CNNModel()

    if args.load:
        model_file = args.load[0]
        json_file = args.load[1]
        model = cnn.load_model('model.json')
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

    print("Training data...")
    cnn.train(train_generator=train_generator,
              validation_generator=validation_generator,
              train_samples=train_samples, validation_samples=validation_samples)

    cnn.save_model()

if __name__ == "__main__":
    main()