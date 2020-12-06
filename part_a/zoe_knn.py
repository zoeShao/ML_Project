from sklearn.impute import KNNImputer
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = np.transpose(nbrs.fit_transform(matrix.T))
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def plot_accu(data_accu_values, filter_mode):
    """
    Generate a plot showing the validation accuracy for k ∈ {1,6,11,16,21,26}.
    The filter_mode is either user-based or item-based.

    :param data_accu_values: list of float
    :param filter_mode: str
    :return: None
    """
    plt.figure(figsize=(10, 6))
    # Plotting the Training error points
    x1, y1 = range(1, 31, 5), data_accu_values

    plt.plot(x1, y1, color='blue', label='Validation', linestyle='dashed',
             marker='o', markerfacecolor='blue', markersize=5)

    plt.xlabel('k - Number of Nerest Neighbors')
    plt.ylabel('Accuracy ({})'.format(filter_mode))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(np.arange(1, 31, 5))
    plt.legend()
    plt.savefig("{}_accuracy.png".format(filter_mode))
    plt.show()


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # Q1.(a) Plot & report the accuracy on the validation data
    valid_accu_values = []
    k_values = list(range(1, 31, 5))
    for k in range(1, 31, 5):
        print("When k = {}:".format(k))
        valid_accu = knn_impute_by_user(sparse_matrix, val_data, k)
        valid_accu_values.append(valid_accu)
    plot_accu(valid_accu_values, 'user-based')

    # Q1.(b) Choose and Report k∗ that has the highest performance on
    # validation data.
    valid_accu_values = np.array(valid_accu_values)
    best_k = k_values[valid_accu_values.argmax()]
    print("When k* = {}, it has the highest performance on validation data.".format(best_k))

    # Report the final test accuracy.
    nbrs = KNNImputer(n_neighbors=best_k)
    mat = nbrs.fit_transform(sparse_matrix)
    test_accu = sparse_matrix_evaluate(test_data, mat)
    print("The final test accuracy is {}.".format(test_accu))

    # Q1.(c) Plot & report the accuracy on the validation data
    valid_accu_values = []
    k_values = list(range(1, 31, 5))
    for k in range(1, 31, 5):
        print("When k = {}:".format(k))
        valid_accu = knn_impute_by_item(sparse_matrix, val_data, k)
        valid_accu_values.append(valid_accu)
    plot_accu(valid_accu_values, 'item-based')

    # Q1.(c) Choose and Report k∗ that has the highest performance on
    # validation data.
    valid_accu_values = np.array(valid_accu_values)
    best_k = k_values[valid_accu_values.argmax()]
    print("When k* = {}, it has the highest performance on validation data.".format(best_k))

    # Report the final test accuracy.
    nbrs = KNNImputer(n_neighbors=best_k)
    mat = nbrs.fit_transform(sparse_matrix)
    test_accu = sparse_matrix_evaluate(test_data, mat)
    print("The final test accuracy is {}.".format(test_accu))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
