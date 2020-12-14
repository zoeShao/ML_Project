from utils import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ast import literal_eval
from scipy import spatial
import operator
from matplotlib.ticker import MaxNLocator


def pre_process_ques_data():
    """
    Pre-process the training dataset to gain the statistical property of
    the question data.

    :return: DataFrame
    """
    correctness = pd.read_csv('../data/train_data.csv', header=0, usecols=range(3))

    # Group by question ID, and compute the total number of
    # correctness and the average correctness for each question
    ques_properties = correctness.groupby('question_id').agg({'is_correct': [np.size, np.mean]})
    # print(ques_properties.head())

    # Create a new DataFrame that contains the normalized correctness.
    ques_correctness = pd.DataFrame(ques_properties['is_correct']['mean'])
    ques_norm_correctness = ques_correctness.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # print(ques_norm_correctness.head())

    return ques_norm_correctness


def pre_process_user_data():
    """
    Pre-process the training dataset to gain the statistical property of
    the student data.

    :return: DataFrame
    """
    correctness = pd.read_csv('../data/train_data.csv', header=0, usecols=range(3))

    user_properties = correctness.groupby('user_id').agg({'is_correct': [np.size, np.mean]})
    # print(user_properties.head())

    user_correctness = pd.DataFrame(user_properties['is_correct']['mean'])
    user_norm_correctness = user_correctness.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # print(user_norm_correctness.head())

    return user_norm_correctness


def collect_question_meta(ques_norm_correctness):
    """
    Collect more information about each question, like subjects. There are 388
    distinct subjects. Convert the subject list into an array that contains 388
    entries. If the question has subject i, then the i-th entry would be 1,
    and 0 otherwise. Return a dictionary where keys are the question ids.

    :param ques_norm_correctness:
    :return: A dictionary {question_id: tuple}
    """
    question_dict = {}
    with open(r'../data/question_meta.csv', encoding="ISO-8859-1") as f:
        next(f)
        for line in f:
            fields = line.rstrip('\n').split('\"')
            subject_lst = np.zeros(388)
            question_id = int(fields[0][:-1])
            subjects = fields[1]
            subjects = literal_eval(subjects)
            # Convert the subject list into an array.
            for subject_id in subjects:
                subject_lst[subject_id] = 1
            question_dict[question_id] = (question_id, subjects, np.array(subject_lst), ques_norm_correctness.loc[question_id].get('mean'))
    return question_dict


def collect_user_meta(user_norm_correctness):
    """
    Collect more information about each student, like gender.
    Return a dictionary where keys are the user ids.

    :param user_norm_correctness:
    :return: A dictionary {question_id: tuple}
    """
    user_dict = {}
    with open(r'../data/student_meta.csv', encoding="ISO-8859-1") as f:
        next(f)
        for line in f:
            fields = line.rstrip('\n').split(',')
            user_id = int(fields[0])
            gender = int(fields[1])
            user_dict[user_id] = (user_id, gender, user_norm_correctness.loc[user_id].get('mean'))
    return user_dict


def compute_ques_distance(ques_a, ques_b):
    """
    Compute the distance between two questions based on how similar
    their subjects are, and how similar their correctness are.

    :param ques_a: tuple
    :param ques_b: tuple
    :return: float
    """
    subject_a = ques_a[2]
    subject_b = ques_b[2]
    subject_distance = spatial.distance.cosine(subject_a, subject_b)
    correctness_a = ques_a[3]
    correctness_b = ques_b[3]
    correctness_distance = abs(correctness_a - correctness_b)
    return subject_distance + correctness_distance


def knn_for_ques(question_id, question_dict, k):
    """ Use KNN model to predict the correctness for a specific
    question.

    :param question_id: int
    :param question_dict: A dictionary {question_id: tuple}
    :param k: int
    :return: float
    """
    distances = []
    for question in question_dict:
        if question != question_id:
            dist = compute_ques_distance(question_dict[question_id], question_dict[question])
            distances.append((question, dist))
    distances.sort(key=operator.itemgetter(1))

    # Find the nearest k neighbours
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])

    avg_ques_correctness = 0
    for neighbor in neighbors:
        avg_ques_correctness += question_dict[neighbor][3]
    avg_ques_correctness /= k
    return avg_ques_correctness


def compute_user_distance(user_a, user_b):
    """
    Compute the distance between two students based on how similar
    their gender are, and how similar their correctness are.

    :param user_a: tuple
    :param user_b: tuple
    :return: float
    """
    gender_a = user_a[1]
    gender_b = user_b[1]
    gender_distance = abs(gender_a - gender_b)
    correctness_a = user_a[2]
    correctness_b = user_b[2]
    correctness_distance = abs(correctness_a - correctness_b)
    return gender_distance + correctness_distance


def knn_for_user(user_id, user_dict, k):
    """ Use KNN model to predict the correctness of a specific
    user.

    :param user_id: int
    :param user_dict: A dictionary {user_id: tuple}
    :param k: int
    :return: float
    """
    distances = []
    for user in user_dict:
        if user != user_id:
            dist = compute_user_distance(user_dict[user_id], user_dict[user])
            distances.append((user, dist))
    distances.sort(key=operator.itemgetter(1))

    # Find the nearest k neighbours
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])

    avg_user_correctness = 0
    for neighbor in neighbors:
        avg_user_correctness += user_dict[neighbor][2]
    avg_user_correctness /= k
    return avg_user_correctness


def predict_all_ques(sparse_matrix, question_dict, k):
    """ Get the prediction matrix (52x1774) for all questions by using knn
    to predict each question correctness.

    :param sparse_matrix: 2D sparse matrix
    :param question_dict: A dictionary {question_id: tuple}
    :param k: int
    :return: 2D matrix
    """
    user_nums, question_nums = sparse_matrix.shape
    question_pred_ll = np.zeros(question_nums)

    for q_id in range(question_nums):
        question_pred = knn_for_ques(q_id, question_dict, k)
        question_pred_ll[q_id] = question_pred
    question_pred_matrix = np.repeat(np.array([question_pred_ll]), user_nums, axis=0)
    return question_pred_matrix


def predict_all_user(sparse_matrix, user_dict, k):
    """
    Get the prediction matrix (52x1774) for all students by using knn
    to predict each student correctness.

    :param sparse_matrix: 2D sparse matrix
    :param user_dict: A dictionary {user_id: tuple}
    :param k: int
    :return: 2D matrix
    """
    user_nums, question_nums = sparse_matrix.shape
    user_pred_ll = []

    for u_id in range(user_nums):
        user_pred = knn_for_user(u_id, user_dict, k)
        user_pred_ll.append(user_pred)
    user_pred_matrix = np.repeat(np.array([user_pred_ll]), question_nums, axis=0).T
    return user_pred_matrix


def predict_all(sparse_matrix, val_data, question_dict, user_dict, k):
    """
    Get the prediction matrix (52x1774) for the sparse matrix by averaging the
    prediction over the user_pred_matrix and question_pred_matrix.
    Return the accuracy on valid_data.

    :param sparse_matrix: 2D sparse matrix
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param question_dict: A dictionary {question_id: tuple}
    :param user_dict: A dictionary {user_id: tuple}
    :param k: int
    :return: float
    """
    question_pred_matrix = predict_all_ques(sparse_matrix, question_dict, k)
    user_pred_matrix = predict_all_user(sparse_matrix, user_dict, k)
    final_pred_matrix = (question_pred_matrix + user_pred_matrix)/2

    pred_matrix = sparse_matrix.copy()
    pred_matrix[np.isnan(sparse_matrix)] = final_pred_matrix[np.isnan(sparse_matrix)]

    acc = sparse_matrix_evaluate(val_data, pred_matrix)
    return acc


def plot_accu(data_accu_values):
    """
    Generate a plot showing the validation accuracy for k ∈ {1,6,11,16,21,26}.

    :param data_accu_values: list of float
    :return: None
    """
    plt.figure(figsize=(10, 6))
    # Plotting the Training error points
    x1, y1 = range(1, 10, 2), data_accu_values

    plt.plot(x1, y1, color='blue', label='Validation', linestyle='dashed',
             marker='o', markerfacecolor='blue', markersize=5)

    plt.xlabel('k - Number of Nerest Neighbors')
    plt.ylabel('Accuracy')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(np.arange(1, 10, 2))
    plt.legend()
    plt.savefig("partb_accuracy.png")
    plt.show()


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Calculating...")
    ques_norm_correctness = pre_process_ques_data()
    user_norm_correctness = pre_process_user_data()

    question_dict = collect_question_meta(ques_norm_correctness)
    user_dict = collect_user_meta(user_norm_correctness)
    # Examples:
    # print(question_dict[1])
    # print(user_dict[1])

    valid_accu_values = []
    k_values = list(range(1, 10, 2))
    for k in range(1, 10, 2):
        print('---------------')
        print("When k = {}:".format(k))
        valid_accu = predict_all(sparse_matrix, val_data, question_dict, user_dict, k)
        print("Validation Accuracy: {}".format(valid_accu))
        valid_accu_values.append(valid_accu)
    plot_accu(valid_accu_values)

    valid_accu_values = np.array(valid_accu_values)
    best_k = k_values[valid_accu_values.argmax()]
    print("When k* = {}, it has the highest performance on validation data.".format(best_k))

    # Report the final test accuracy
    test_accu = predict_all(sparse_matrix, test_data, question_dict, user_dict, best_k)
    print("The final test accuracy is {}.".format(test_accu))


if __name__ == "__main__":
    main()
