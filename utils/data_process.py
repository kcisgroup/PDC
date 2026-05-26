import pandas as pd
import numpy as np
import torch
from make_datasets.Datasets import Datasets, Seg_Datasets


def get_feature_list(feature_path):
    feature_file = open(feature_path, 'r')
    all_feature_list = []
    for ft in feature_file:
        all_feature_list.append(ft.strip())

    return all_feature_list


def construct_data(data, feature_list, labels=0):
    res = []
    # get every feature data
    for feature in feature_list:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')

    sample_n = len(res[0])
    # add labels
    if type(labels) == int:
        res.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res.append(labels)

    return res


def get_dataset(data_path, feature_path):
    data_orig = pd.read_csv(data_path, sep=',', index_col=None)
    print("data:", data_orig.shape)
    data = data_orig
    # get all feature name
    feature_list = get_feature_list(feature_path)

    train_data = construct_data(data, feature_list, labels=data.label.tolist())
    train_dataset = Datasets(train_data)

    return feature_list, train_dataset


def get_dataset2(data_path, feature_path):
    data_orig = pd.read_csv(data_path, sep=',', index_col=None)
    data = data_orig
    # get all feature name
    feature_list = get_feature_list(feature_path)

    data = construct_data(data, feature_list, labels=data.label.tolist())
    dataset = Seg_Datasets(data)

    return feature_list, dataset