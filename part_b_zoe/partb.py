from sklearn.impute import KNNImputer
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
from ast import literal_eval
from scipy import spatial
import operator
from datetime import datetime


def collect_question_meta(ques_norm_num_correctness, ques_properties):
    question_dict = {}
    with open(r'../data/question_meta.csv', encoding="ISO-8859-1") as f:
        next(f)
        for line in f:
            fields = line.rstrip('\n').split('\"')
            subject_lst = np.zeros(388)
            question_id = int(fields[0][:-1])
            subjects = fields[1]
            subjects = literal_eval(subjects)

            for subject_id in subjects:
                subject_lst[subject_id] = 1
            question_dict[question_id] = (question_id, subjects, np.array(subject_lst), ques_norm_num_correctness.loc[question_id].get('mean'), ques_properties.loc[question_id].is_correct.get('mean'))
    return question_dict


def collect_user_meta(user_norm_num_correctness, user_properties):
    user_dict = {}
    with open(r'../data/student_meta.csv', encoding="ISO-8859-1") as f:
        next(f)
        for line in f:
            fields = line.rstrip('\n').split(',')
            user_id = int(fields[0])
            gender = int(fields[1])
            user_dict[user_id] = (user_id, gender, user_norm_num_correctness.loc[user_id].get('mean'), user_properties.loc[user_id].is_correct.get('mean'))
    return user_dict


def compute_ques_distance(a, b):
    subject_a = a[2]
    subject_b = b[2]
    subject_distance = spatial.distance.cosine(subject_a, subject_b)
    correctness_a = a[3]
    correctness_b = b[3]
    correctness_distance = abs(correctness_a - correctness_b)
    return subject_distance + correctness_distance


def get_ques_neighbors(question_id, question_dict, K):
    distances = []
    for question in question_dict:
        if question != question_id:
            dist = compute_ques_distance(question_dict[question_id], question_dict[question])
            distances.append((question, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])

    avg_ques_correctness = 0

    for neighbor in neighbors:
        avg_ques_correctness += question_dict[neighbor][4]
    avg_ques_correctness /= K
    return avg_ques_correctness


def compute_user_distance(a, b):
    gender_a = a[1]
    gender_b = b[1]
    gender_distance = abs(gender_a - gender_b)
    correctness_a = a[2]
    correctness_b = b[2]
    correctness_distance = abs(correctness_a - correctness_b)
    return gender_distance + correctness_distance


def get_user_neighbors(user_id, user_dict, K):
    distances = []
    for user in user_dict:
        if user != user_id:
            dist = compute_user_distance(user_dict[user_id], user_dict[user])
            distances.append((user, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])

    avg_user_correctness = 0

    for neighbor in neighbors:
        avg_user_correctness += user_dict[neighbor][3]
    avg_user_correctness /= K
    return avg_user_correctness


def predict_ques(sparse_matrix, question_dict, K):
    user_nums, question_nums = sparse_matrix.shape
    question_pred_ll = np.zeros(question_nums)
    # question_pred_matrix = np.zeros((user_nums, question_nums))
    for q_id in range(question_nums):
        question_pred = get_ques_neighbors(q_id, question_dict, K)
        question_pred_ll[q_id] = question_pred
    question_pred_matrix = np.repeat(np.array([question_pred_ll]), user_nums, axis=0)
    return question_pred_matrix


def predict_user(sparse_matrix, user_dict, K):
    user_nums, question_nums = sparse_matrix.shape
    user_pred_ll = []
    # user_pred_matrix = np.zeros((user_nums, question_nums))
    for u_id in range(user_nums):
        user_pred = get_user_neighbors(u_id, user_dict, K)
        user_pred_ll.append(user_pred)
    user_pred_matrix = np.repeat(np.array([user_pred_ll]), question_nums, axis=0).T
    return user_pred_matrix


def predict_all(sparse_matrix, val_data, question_dict, user_dict, K):
    question_pred_matrix = predict_ques(sparse_matrix, question_dict, K)
    user_pred_matrix = predict_user(sparse_matrix, user_dict, K)

    final_pred_matrix = (question_pred_matrix + user_pred_matrix)/2

    # print(final_pred_matrix)
    pred_matrix = sparse_matrix.copy()
    pred_matrix[np.isnan(sparse_matrix)] = final_pred_matrix[np.isnan(sparse_matrix)]

    acc = sparse_matrix_evaluate(val_data, pred_matrix)
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    r_cols = ['question_id', 'user_id', 'is_correct']
    correctness = pd.read_csv('../data/train_data.csv', header=0, usecols=range(3))
    print(correctness.head())

    #  Group everything by question ID, and compute the total number of
    #  correctness  and the average correctness for every question
    ques_properties = correctness.groupby('question_id').agg({'is_correct': [np.size, np.mean]})
    user_properties = correctness.groupby('user_id').agg({'is_correct': [np.size, np.mean]})
    print(ques_properties.head())
    print(user_properties.head())

    # Create a new DataFrame that contains the normalized number of correctness.
    # So, a value of 0 means nobody did it correctly, and a value of 1 will
    # mean it's the most easy one to do.
    ques_num_correctness = pd.DataFrame(ques_properties['is_correct']['mean'])
    ques_norm_num_correctness = ques_num_correctness.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    print(ques_norm_num_correctness.head())
    user_num_correctness = pd.DataFrame(user_properties['is_correct']['mean'])
    user_norm_num_correctness = user_num_correctness.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    print(user_norm_num_correctness.head())

    question_dict = collect_question_meta(ques_norm_num_correctness, ques_properties)
    user_dict = collect_user_meta(user_norm_num_correctness, user_properties)
    valid_accu_values = []
    k_values = list(range(1, 10, 2))
    for K in range(1, 10, 2):
        print('---------------')
        print("When k = {}:".format(K))
        valid_accu = predict_all(sparse_matrix, val_data, question_dict, user_dict, K)
        print("Validation Accuracy: {}".format(valid_accu))
        valid_accu_values.append(valid_accu)

    valid_accu_values = np.array(valid_accu_values)
    best_k = k_values[valid_accu_values.argmax()]
    print("When k* = {}, it has the highest performance on validation data.".format(best_k))

    # Report the final test accuracy
    test_accu = predict_all(sparse_matrix, test_data, question_dict, user_dict, best_k)
    print("The final test accuracy is {}.".format(test_accu))


if __name__ == "__main__":
    main()
