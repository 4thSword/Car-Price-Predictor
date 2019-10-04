# Imports:

import pandas as pd


# Global Constants:

TRAIN_PATH = "../input/cars_train.csv"
TEST_PATH = "../input/cars_test.csv"
OUTPUT_TRAIN_PATH = "../input/processed_train.csv"
OUTPUT_TEST_PATH = "../input/processed_test.csv"

# Functions definitions:

def load_data(path):
    return pd.read_csv(path)

if __name__ == "__main__":
    
    #First step: To load our training data and our test data:
    
    
    training = load_data(TRAIN_PATH)
    test = load_data(TEST_PATH)
    print(training.head())
    print(test.head())