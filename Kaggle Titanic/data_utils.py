import pandas as pd


def load_train_data():
    return pd.read_csv('datasets/train.csv', sep=',')


def load_test_data():
    return pd.read_csv('datasets/test.csv', sep=',')


def load_full_data():
    return pd.concat([load_train_data(), load_test_data()], ignore_index=True)

