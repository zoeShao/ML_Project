from sklearn.impute import KNNImputer
from utils import *


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
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


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
    k_lst = [1, 6, 11, 16, 21, 26]
    best_k_user = None
    best_k_item = None
    highest_acc_user = 0
    highest_acc_item = 0
    for k in k_lst:
        acc_by_user = knn_impute_by_user(sparse_matrix, val_data, k)
        if acc_by_user > highest_acc_user:
            highest_acc_user = acc_by_user
            best_k_user = k
        acc_by_item = knn_impute_by_item(sparse_matrix, val_data, k)
        if acc_by_item > highest_acc_item:
            highest_acc_item = acc_by_item
            best_k_item = k
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    print("k that has the highest performance on validation data by user-based "
          "collaborative filtering: {}".format(best_k_user))
    print("final test accuracy: {}".format(test_acc_user))
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, best_k_item)
    print("k that has the highest performance on validation data by item-based "
          "collaborative filtering: {}".format(best_k_item))
    print("final test accuracy: {}".format(test_acc_item))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

