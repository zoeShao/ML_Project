from utils import *

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # prob = sigmoid(theta-beta)
    # targets = data['is_correct']
    # log_lklihood = np.dot(targets, np.log(prob)) + np.dot((1 - targets), np.log(1 - prob))
    # log_lklihood = sum(log_lklihood)
    log_lklihood = 0
    for i in range(len(data["question_id"])):
        question = data["question_id"][i]
        user = data["user_id"][i]
        target = data["is_correct"][i]
        # x = np.sum(theta[user] - beta[question])
        # prob = sigmoid(x)
        # log_lklihood += target * np.log(prob) + (1 - target) * np.log(1 - prob)
        x = theta[user] - beta[question]
        log_lklihood += np.sum(target * x - np.log(1 + np.exp(x)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # prob = sigmoid(theta - beta)
    # # theta
    # theta_gradient = data['is_correct'] - prob
    # theta = theta + lr * theta_gradient
    #
    # #beta
    # beta_gradient = prob - data['is_correct']
    # beta = beta + lr * beta_gradient

    for i in range(len(data["question_id"])):
        question = data["question_id"][i]
        user = data["user_id"][i]
        target = data["is_correct"][i]
        x = np.sum((theta[user] - beta[question]))
        prob = sigmoid(x)
        # update theta
        theta_gradient = target - prob
        theta[user] = theta[user] + lr * theta_gradient
        # update beta with the new theta
        x = np.sum((theta[user] - beta[question]))
        prob = sigmoid(x)
        beta_gradient = prob - target
        beta[question] = beta[question] + lr * beta_gradient
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations, sparse_matrix):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.random.random(sparse_matrix.shape[0])
    beta = np.random.random(sparse_matrix.shape[1])

    training_lld = []
    validation_lld = []

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        training_lld.append(-neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        validation_lld.append(-val_neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, training_lld, validation_lld


def evaluate(data, theta, beta):
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
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 20
    theta, beta, val_acc_lst, training_neg_lld, validation_neg_lld = irt(train_data, val_data, lr, iterations, sparse_matrix)

    # plot training log likelihood
    plt.plot(list(range(iterations)), training_neg_lld)
    plt.xlabel("Iterations")
    plt.ylabel("Log-LikeLihood")
    plt.title("Log-Likelihoods of Training Set with IRT")
    plt.show()
    # plot validation log likelihood
    plt.plot(list(range(iterations)), validation_neg_lld)
    plt.xlabel("Iterations")
    plt.ylabel("Log-LikeLihood")
    plt.title("Log-Likelihoods of Validation Set with IRT")
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    # report the final validation and test accuracies
    val_accuracy = evaluate(val_data, theta, beta)
    print("The final validation accuracy is: " + str(val_accuracy))
    test_accuracy = evaluate(test_data, theta, beta)
    print("The final test accuracy is: " + str(test_accuracy))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    # part (d)
    theta.sort()
    plt.plot(theta, sigmoid(theta - beta[2]), label='j1')
    plt.plot(theta, sigmoid(theta - beta[4]), label='j2')
    plt.plot(theta, sigmoid(theta - beta[6]), label='j3')
    plt.plot(theta, sigmoid(theta - beta[8]), label='j4')
    plt.plot(theta, sigmoid(theta - beta[10]), label='j5')
    plt.xlabel("Thetas")
    plt.ylabel("Probability of Correct Response")
    plt.title("Probability of Correct Response Based on Theta")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
