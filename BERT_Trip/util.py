import json
import os
import pickle
import random
import time
import warnings
import numpy as np
from typing import Dict, List, Optional
#import torch
#from transformers import PreTrainedTokenizer
import subprocess
from datetime import datetime
import pandas as pd

dataset_metadata = {
    'Edin': {'USER_NUM':386, 'TIME_NUM': 48},
    'Glas': {'USER_NUM':90, 'TIME_NUM': 48},
    'melb': {'USER_NUM':265, 'TIME_NUM': 48},
    'Osak': {'USER_NUM':40, 'TIME_NUM': 48},
    'Toro': {'USER_NUM':196, 'TIME_NUM': 48},
    #new one
    'weeplaces_poi_25_length_3-15-numtraj_765': {'USER_NUM':400, 'TIME_NUM': 48},
    'weeplaces_poi_50_length_3-15-numtraj_2134': {'USER_NUM':883, 'TIME_NUM': 48},
    'weeplaces_poi_100_length_3-15-numtraj_4497': {'USER_NUM':1555, 'TIME_NUM': 48},
    'weeplaces_poi_200_length_3-15-numtraj_7790': {'USER_NUM':2537, 'TIME_NUM': 48},
    'weeplaces_poi_400_length_3-15-numtraj_12288': {'USER_NUM':3357, 'TIME_NUM': 48},
    'weeplaces_poi_801_length_3-15-numtraj_19385': {'USER_NUM':4230, 'TIME_NUM': 48},
    'weeplaces_poi_1600_length_3-15-numtraj_31545': {'USER_NUM':5276, 'TIME_NUM': 48},
    'weeplaces_poi_3200_length_3-15-numtraj_53901': {'USER_NUM':6531, 'TIME_NUM': 48},
    'weeplaces_poi_6400_length_3-15-numtraj_90248': {'USER_NUM':7819, 'TIME_NUM': 48},
    'weeplaces_poi_12800_length_3-15-numtraj_147287': {'USER_NUM':9326, 'TIME_NUM': 48},
}
file_max_sequence_length = {
    '': 0,
    'weeplaces_poi_800_length_3-15-numtraj_14374':17,
    'weeplaces_poi_25_length_3-15-numtraj_765': 17,
    'weeplaces_poi_50_length_3-15-numtraj_2134': 17,
    'weeplaces_poi_100_length_3-15-numtraj_4497': 17,
    'weeplaces_poi_200_length_3-15-numtraj_7790': 17,
    'weeplaces_poi_400_length_3-15-numtraj_12288': 17,
    'weeplaces_poi_801_length_3-15-numtraj_19385': 17,
    'weeplaces_poi_1600_length_3-15-numtraj_31545': 17,
    'weeplaces_poi_3200_length_3-15-numtraj_53901': 17,
    'weeplaces_poi_6400_length_3-15-numtraj_90248': 17,
    'weeplaces_poi_12800_length_3-15-numtraj_147287': 17,
    'Edin': 15,
    'Glas': 10,
    'melb': 22,
    'Osak': 8,
    'Toro': 15,
}


def read_POIs(data):
    df = pd.read_csv(data.name)
    df = df.rename(columns = {'poiID': 'id', 'poiCat': 'cat', 'poiLat': 'lat', 'poiLon': 'lon'})
    df['id'] = df['id'].astype('str')
    df = df.set_index('id')
    return df.to_dict('index')

# current_directory = os.getcwd()
# # 打印当前工作目录
# print("当前路径:", current_directory)

osak_poi_name = "poi-" + "Osak" + ".csv"
osak_op_tdata = open('./data/origin_data/' + osak_poi_name, 'r')
osak_points = read_POIs(osak_op_tdata)
edin_poi_name = "poi-" + "Edin" + ".csv"
edin_op_tdata = open('./data/origin_data/' + edin_poi_name, 'r')
edin_points = read_POIs(edin_op_tdata)
glas_poi_name = "poi-" + "Glas" + ".csv"
glas_op_tdata = open('./data/origin_data/' + glas_poi_name, 'r')
glas_points = read_POIs(glas_op_tdata)
toro_poi_name = "poi-" + "Toro" + ".csv"
toro_op_tdata = open('./data/origin_data/' + toro_poi_name, 'r')
toro_points = read_POIs(toro_op_tdata)

import pandas as pd


def Haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) **2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    d = R * c
    return round(d, 4)


def log(i, config, result):
    pretrain_result_csv = './pretrain_result.csv'
    log_file = './log.txt'
    detail_data = result['details']
    time = datetime.now()
    seq_length_keys = sorted(detail_data.keys())
    for seq_length in seq_length_keys:
        details = detail_data[seq_length]
        with open(pretrain_result_csv, 'a') as f:
            name = f'Pretrained--{result["strategy"]} (length = {seq_length})'
            epoch = f'(epoch = {i})'
            mln = f'(mln = {config.mlm_probability:.2f} %)'
            w = f'{time},{config.dataset:>10},{config.model_type:>10},{name:>35} {mln:>15}, {epoch:>15},{details["incorrect_f1"]:.3f},{details["incorrect_pairs_f1"]:.3f},{details["true_f1"]:.3f},{details["true_pairs_f1"]:.3f}, {details["bleu"]:.3f}\n'
            f.write(w)
        with open(log_file, 'a') as f:
            w = f'Time: {time}\n'
            w += f'Dataset: {config.dataset} (length = {seq_length})\n'
            w += f'PreTrained Model: {config.model_type} (i = {i})\n'
            w += f'Evaluation:{result["strategy"]}\n'
            w += f'incorrect_f1: {details["incorrect_f1"]:.3f}\nincorrect_pairs_f1: {details["incorrect_pairs_f1"]:.3f}\n'
            w += f'f1: {details["true_f1"]:.3f}\npairs_f1: {details["true_pairs_f1"]:.3f}\nBLEU score: {details["bleu"]:.3f}\n\n'
            f.write(w)

def wccount(filename):
    out = subprocess.Popen(['wc', '-l', filename],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT
                         ).communicate()[0]
    print("=====", out.partition((b' ')))
    return int(out.partition(b' ')[0])
# cpdef float calc_pairsF1(y, y_hat):

def calc_F1(expected0, predict0, noloop=False):
    predict = predict0.copy()
    expected = expected0.copy()
    predict[0] = expected[0]
    predict[-1] = expected[-1]
    '''Compute recall, precision and F1 for recommended trajectories'''
    assert (isinstance(noloop, bool))
    assert (len(expected) > 0)
    assert (len(predict) > 0)

    if noloop == True:
        intersize = len(set(expected) & set(predict))
    else:
        # match_tags = np.zeros(len(expected), dtype=np.bool)
        match_tags = np.zeros(len(expected), dtype=bool)
        for poi in predict:
            for j in range(len(expected)):
                if match_tags[j] == False and poi == expected[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]

    recall = intersize * 1.0 / len(expected)
    precision = intersize * 1.0 / len(predict)
    denominator = recall + precision
    if denominator == 0:
        denominator = 1
    score = 2 * precision * recall * 1.0 / denominator
    return score


def calcu_distance_shift(pred0, true0, dataset):
    pred = pred0.copy()
    true = true0.copy()
    pred[0] = true[0]
    pred[-1] = true[-1]
    distance = 0
    points = {}
    if dataset == 'Osak':
        points = osak_points
    if dataset == 'Edin':
        points = edin_points
    if dataset == 'Glas':
        points = glas_points
    if dataset == 'Toro':
        points = toro_points
    for i in range(len(pred)):
        distance += Haversine(points[pred[i]]['lat'], points[pred[i]]['lon'], points[true[i]]['lat'], points[true[i]]['lon'])
    return distance


def calcu_start_ratio(pred, true):
    if pred[0] == true[0]:
        return 1.0
    else:
        return 0.0


def calcu_end_ratio(pred, true):
    if pred[-1] == true[-1]:
        return 1.0
    else:
        return 0.0


# TODO:self-looping
def count_adjacent_percentage(lst0, true0):
    lst = lst0.copy()
    true = true0.copy()
    lst[0] = true[0]
    lst[-1] = true[-1]
    # count the adjacent trajectory length
    if len(lst) < 2:
        return 0

    # If the list length is less than 2, there are no adjacent elements, and the number of duplicates is 0
    adjacent_items = len(lst) - 1

    # Initialize the previous element as the first element in the list
    count = 0
    prev = lst[0]

    # Start traversing from the second element of the list
    for current in lst[1:]:
        # If the current element is the same as the previous element, increase the repeat count
        if current == prev:
            count += 1
        # Update the previous element to the current element
        prev = current

    loop_ratio = count / adjacent_items

    return loop_ratio


# TODO:repetition
def count_repetition_percentage(input_data0, true0):
    # if list
    input_data = input_data0.copy()
    true = true0.copy()
    input_data[0] = true[0]
    input_data[-1] = true[-1]

    if isinstance(input_data, list):
        unique_items = set(input_data)
    # if tensor
    elif hasattr(input_data, 'numpy'):
        unique_items = set(input_data.numpy().tolist())
    else:
        raise ValueError("Input data must be a list or a tensor.")

    total_items = len(input_data)
    repetition_items_count = total_items - len(unique_items)
    repetition_ratio = repetition_items_count / total_items

    return repetition_ratio


def calc_pairsF1(y0, y_hat0):
    y = y0.copy()
    y_hat = y_hat0.copy()
    y_hat[0] = y[0]
    y_hat[-1] = y[-1]
    assert (len(y) > 0)
    assert (len(y) == len(set(y)))  # no loops in y
    # cdef int n, nr, nc, poi1, poi2, i, j
    # cdef double n0, n0r

    n = len(y)
    nr = len(y_hat)
    #assert (n == nr)
    n0 = n * (n - 1) / 2
    n0r = nr * (nr - 1) / 2
    # y determines the correct visiting order
    order_dict = dict()
    for i in range(n):
        order_dict[y[i]] = i

    nc = 0
    for i in range(nr):
        poi1 = y_hat[i]
        for j in range(i + 1, nr):
            poi2 = y_hat[j]
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]: nc += 1


    precision = (1.0 * nc) / (1.0 * n0r)
    recall = (1.0 * nc) / (1.0 * n0)
    if nc == 0:
        f1 = 0
    else:
        f1 = 2. * precision * recall / (precision + recall)
    return float(f1)


def true_f1(expected0, predict0, noloop=False):
    '''Compute recall, precision and F1 for recommended trajectories'''
    predict = predict0.copy()
    expected = expected0.copy()
    predict[0] = expected[0]
    predict[-1] = expected[-1]
    assert (isinstance(noloop, bool))
    assert (len(expected) > 0)
    assert (len(predict) > 0)
    # expected = expected[1:-1]
    # predict = predict[1:-1]
    predict_size = len(expected)
    if noloop == True:
        intersize = len(set(expected) & set(predict))
    else:
        # match_tags = np.zeros(predict_size, dtype=np.bool)
        match_tags = np.zeros(predict_size, dtype=bool)
        for poi in predict:
            for j in range(len(expected)):
                if match_tags[j] == False and poi == expected[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]
    recall = intersize * 1.0 / len(expected)
    precision = intersize * 1.0 / len(predict)
    denominator = recall + precision
    if denominator == 0:
        denominator = 1
    score = 2 * precision * recall * 1.0 / denominator
    return score


def true_pairs_f1(y0, y_hat0):
    y = y0.copy()
    y_hat = y_hat0.copy()
    y_hat[0] = y[0]
    y_hat[-1] = y[-1]
    #['296142', '3976', '14458', '3976'] ['296142', '3976', '3976', '3976'] 0.75 0.5 0.5 0.0 0.24672975470447223
    assert (len(y) > 2)
    # y = y[1:-1]
    # y_hat = y_hat[1:-1]
    #assert (len(y) == len(set(y)))  # no loops in y
    # cdef int n, nr, nc, poi1, poi2, i, j
    # cdef double n0, n0r
    n = len(y)
    nr = len(y_hat)

    n0 = n * (n - 1) / 2
    n0r = nr * (nr - 1) / 2
    # y determines the correct visiting order
    order_dict = dict()
    for i in range(n):
        order_dict[y[i]] = i

    nc = 0
    for i in range(nr):
        poi1 = y_hat[i]
        #print(f'i:{i}, poi1:{poi1}, nr:{nr}')
        for j in range(i + 1, nr):
            poi2 = y_hat[j]
            #print(f'j:{j}, poi2:{poi2}, nc:{nc}')
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]:
                    nc += 1

    if n0r == 0:
        n0 = 1
        n0r = 1
        if y[0] == y_hat[0]:
            nc = 1

    precision = (1.0 * nc) / (1.0 * n0r)
    recall = (1.0 * nc) / (1.0 * n0r)
    if nc == 0:
        f1 = 0
    else:
        f1 = 2. * precision * recall / (precision + recall)
    """
    if f1 == 0:
        print(order_dict)
        print(y)
        print(y_hat)
        print()
    """
    return float(f1)


def bleu(reference, candidate):
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.bleu_score import SmoothingFunction
    #delete starting and ending points
    refSize = len(reference)
    canSize = len(candidate)
    assert(refSize == canSize)
    reference = [reference[1:-1]]
    candidate = candidate[1:-1]
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference, candidate, auto_reweigh = True, smoothing_function = smoothie)
    return score


def get_model(model_name):
    from model.bert.bert_model import BertForMaskedLM as PureBERT
    from model.bert.bert_siam import SiamBERT
    from model.bert.bert_temporal import TemporalBERT
    from model.bert.bert_trip import TripBERT
    from model.tester import Tester
    config = {
        'add_user_token': False,
        'add_time_token': False,
        'use_data_agumentation': False,
    }
    base_model = None
    if model_name == 'bert':
        base_model = PureBERT
    elif model_name == 'bert_siam':
        base_model = SiamBERT
        config['use_data_agumentation'] = True
    elif model_name == 'bert_temporal':
        base_model = TemporalBERT
        config['add_user_token'] = True
        config['add_time_token'] = True
    elif model_name == 'bert_trip':
        base_model = TripBERT
        config['use_data_agumentation'] = True
        config['add_user_token'] = True
        config['add_time_token'] = True

    evaluator = Tester

    return base_model, evaluator, config

def get_kfold_data(df, shuffle = True, cold_start_user = True, fixed_random_state = None):
    base_random_state = int(time.time())
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import ShuffleSplit
    offset = 0
    while True:
        if offset >= 50:
            break
        if fixed_random_state == None:
            random_state = base_random_state + offset
        else:
            random_state = fixed_random_state
        is_all_pois_in_train_set = True
        train_data, test_data = train_test_split(df, test_size = 0.2, random_state = random_state)
        train_trajectories = train_data.iloc[:, 1].values
        test_trajectories = test_data.iloc[:, 1].values
        train_pois = set()
        for trajectory in train_trajectories:
            pois = trajectory.split(',')
            for poi in pois:
                train_pois.add(poi)
        for trajectory in test_trajectories:
            pois = trajectory.split(',')
            for poi in pois:
                if poi not in train_pois:
                    is_all_pois_in_train_set = False

        offset += 1
        if is_all_pois_in_train_set:
            print("tries", offset)
            break
        else:
            pass
            #assert(fixed_random_state == None)
    return train_data, test_data, random_state

def get_root_dir():
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent.parent
    return root_dir
def get_data_dir():
    return get_root_dir() / 'BERT_Trip' / 'data'
def get_dataset_dir(dataset):
    return get_data_dir() / dataset

def datetime_to_interval(d):
    return int(d.hour * 6 + d.minute / 60)

def evaluate_results(results):
    e_1 = []
    e_2 = []
    e_3 = []
    e_4 = []
    e_5 = []
    e_6 = []
    for result in results:
        expected = result['expected']
        prediction = result['predict']
        e_1.append(calc_F1(expected, prediction))
        e_2.append(calc_pairsF1(expected, prediction))
        e_3.append(true_f1(expected, prediction))
        e_4.append(true_pairs_f1(expected, prediction))
        e_5.append(count_adjacent_percentage(prediction))
        e_6.append(count_repetition_percentage(prediction))
        #add bleu if you like
    return {'f1_include_head_tail': np.mean(e_1), 'pairs_f1_include_head_tail': np.mean(e_2), 'f1': np.mean(e_3), 'pairs_f1': np.mean(e_4), 'self_loop': np.mean(e_5), 'repetition': np.mean(e_6)}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_results(dataset, method, train_size, test_size, fold, seed, results, execution_time):

    json_data = {
        'dataset': dataset,
        'method': method,
        'train_size': train_size,
        'test_size': test_size,
        'fold': fold,
        'seed': seed,
        'results': results,
        'execution_time': execution_time,
    }
    root_dir = get_root_dir()
    result_dir = root_dir / 'results'
    result_dir.mkdir(parents=True, exist_ok=True)
    file_path = result_dir / f'{method}.csv'
    with open(file_path, 'a', encoding ="utf-8") as f:
        f.write(f'{json.dumps(json_data, cls=NpEncoder)}\n')
