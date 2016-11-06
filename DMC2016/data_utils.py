import pandas as pd


def get_orders_train():
    return pd.read_csv('datasets/orders_train.txt', sep=';')


def get_train_data():
    return get_orders_train()[680001:860001]


def get_test_data():
    return get_orders_train()[2000001:2200001]


def clean_by_quantity(data):
    return data[data['quantity'] > 0]
