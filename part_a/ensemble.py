# TODO: complete this file.
from utils import *
from part_a.item_response import *
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def sample_with_replacement(train_data, sample_times):
    """
    D : train_data, N: number of examples. Generate sample_times new datasets
    with replacement.

    :param sample_times: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param train_data: int
    :return: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    """
    # Get bootstrap set
    N = len(train_data["user_id"])
    X = np.zeros((N, 2), dtype=int)
    X[:, 0], X[:, 1] = train_data["user_id"], train_data["question_id"]
    y = train_data["is_correct"]

    new_datasets = []
    for i in range(sample_times):
        new_X, new_y = resample(X, y, replace=True)
        new_dataset = {"user_id": new_X[:, 0], "question_id": new_X[:, 1], "is_correct": new_y}
        new_datasets.append(new_dataset)
    return new_datasets


def irt_base_model(sample_dataset, lr, iterations, sparse_matrix):
    """

    :param sample_dataset:
    :param lr:
    :param iterations:
    :param sparse_matrix:
    :return:
    """
    theta = np.random.random(sparse_matrix.shape[0])
    beta = np.random.random(sparse_matrix.shape[1])

    for i in range(iterations):
        theta, beta = update_theta_beta(sample_dataset, lr, theta, beta)

    return theta, beta


def predict(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a)
    return np.array(pred)


def evaluate_pred(pred, data):
    """ Evaluate the model given prediction and data and return the accuracy.
    :param pred: Vector
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :return: float
    """
    pred = (pred >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def ensemble(sample_datasets, val_data, test_data, lr, iterations, sparse_matrix):
    """

    :param sample_datasets:
    :param val_data:
    :param lr:
    :param iterations:
    :param sparse_matrix:
    :return:
    """

    val_pred_ll = []
    test_pred_ll = []
    for data_set in sample_datasets:
        theta, beta = irt_base_model(data_set, lr, iterations, sparse_matrix)
        val_prediction = predict(val_data, theta, beta)
        val_pred_ll.append(val_prediction)
        test_prediction = predict(test_data, theta, beta)
        test_pred_ll.append(test_prediction)
    avg_val_prediction = (val_prediction[0] + val_prediction[1] + val_prediction[2])/3
    avg_test_prediction = (test_pred_ll[0] + test_pred_ll[1] + test_pred_ll[2])/3

    return avg_val_prediction, avg_test_prediction


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    sample_datasets = sample_with_replacement(train_data, 3)

    lr = 0.01
    iterations = 20

    theta, beta = irt_base_model(train_data, lr, iterations, sparse_matrix)
    # report the validation and test accuracies on single base model
    val_accuracy = evaluate(val_data, theta, beta)
    print("The final validation accuracy on single base model is: " + str(val_accuracy))
    test_accuracy = evaluate(test_data, theta, beta)
    print("The final test accuracy on single base model is: " + str(test_accuracy))

    avg_val_prediction, avg_test_prediction = ensemble(sample_datasets, val_data, test_data, lr, iterations, sparse_matrix)
    val_accuracy = evaluate_pred(avg_val_prediction, val_data)
    print("The final bagging validation accuracy is: " + str(val_accuracy))
    test_accuracy = evaluate_pred(avg_test_prediction, test_data)
    print("The final bagging validation accuracy is: " + str(test_accuracy))


if __name__ == "__main__":
    main()
