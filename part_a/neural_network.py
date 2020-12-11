from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        s = nn.Sigmoid()
        encoded = s(self.g(inputs))
        out = s(self.h(encoded))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_loss_lst = []
    valid_acc_lst = []
    valid_loss_lst = []
    best_model = None
    best_valid_acc = 0

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss.backward()

            train_loss += loss.item()
            # Q3(e): add the L2 regularization
            if lamb is not None:
                train_loss += (lamb / 2) * (model.get_weight_norm())
            optimizer.step()

        valid_acc, valid_loss = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {} \tValid Loss: {}".format(epoch, train_loss, valid_acc, valid_loss))
        train_loss_lst.append(train_loss)
        valid_acc_lst.append(valid_acc)
        valid_loss_lst.append(valid_loss)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = model
    return train_loss_lst, valid_acc_lst, valid_loss_lst, best_model
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0
    sum_valid_loss = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        target = valid_data["is_correct"][i]
        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        sum_valid_loss += (guess - target) ** 2
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total), sum_valid_loss


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k_vals = [10, 50, 100, 200, 500]
    lr = 0.01
    num_epoch = 50
    lamb = None
    max_valid_acc = 0
    best_k = 0
    print("--------------Q3(c)----------------")
    for k in k_vals:
        print("---------" + str(k) + "---------")
        model = AutoEncoder(len(train_matrix[0]), k)
        train_loss_lst, valid_acc_lst, \
        valid_loss_lst, best_model = train(model, lr, lamb, train_matrix,
                                           zero_train_matrix, valid_data, num_epoch)
        valid_acc, valid_loss = evaluate(model, zero_train_matrix, valid_data)
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            best_k = k

    print("--------------Q3(d)----------------")
    model = AutoEncoder(len(train_matrix[0]), best_k)
    best_train_loss_lst, best_valid_acc_lst, \
    best_valid_loss_lst, final_best_model = train(model, lr, lamb, train_matrix,
                                                  zero_train_matrix, valid_data,
                                                  num_epoch)
    print("best k: {}".format(best_k))
    plt.xlabel("epoch")
    plt.ylabel("Training Cost")
    epoch = range(num_epoch)
    plt.plot(epoch, best_train_loss_lst, label="Training Cost")
    plt.savefig("Training_Cost.png")
    plt.clf()

    plt.xlabel("epoch")
    plt.ylabel("Valid Acc")
    plt.plot(epoch, best_valid_acc_lst, label="Valid Acc")
    plt.savefig("Valid_Acc.png")
    plt.clf()

    plt.xlabel("epoch")
    plt.ylabel("Valid Loss")
    plt.plot(epoch, best_valid_loss_lst, label="Valid Loss")
    plt.savefig("Valid_Loss.png")
    plt.clf()

    test_acc, test_loss = evaluate(final_best_model, zero_train_matrix, test_data)
    print("final test accuracy: {}".format(test_acc))

    print("--------------Q3(e)----------------")
    max_valid_acc = 0
    lamb_lst = [0.001, 0.01, 0.1, 1]
    best_lamb = None
    final_best_model = None
    k = best_k
    for lamb in lamb_lst:
        print("---------" + str(lamb) + "---------")
        model = AutoEncoder(len(train_matrix[0]), k)
        train_loss_lst, valid_acc_lst, \
        best_valid_loss_lst, best_model = train(model, lr, lamb, train_matrix,
                                                zero_train_matrix, valid_data,
                                                num_epoch)
        valid_acc, valid_loss = evaluate(model, zero_train_matrix, valid_data)
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            best_lamb = lamb
            final_best_model = best_model
    print("chosen lambda: {}".format(best_lamb))
    print("final validation accuracy: {}".format(max_valid_acc))
    test_acc, test_loss = evaluate(final_best_model, zero_train_matrix, test_data)
    print("final test accuracy: {}".format(test_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
