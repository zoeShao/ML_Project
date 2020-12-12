import pandas
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    # main()
    with open("merged_training_data.csv", 'r') as csvfile:
        train_data = pandas.read_csv(csvfile)
    with open("merged_test_data.csv", 'r') as csvfile:
        test_data = pandas.read_csv(csvfile)
    x_columns = []
    for i in list(range(388)):
        x_columns.append(str(i))
    x_columns.extend(["gender", "premium_pupil"])
    y_column = ["is_correct"]

    # Create the knn model.
    # Look at the five closest neighbors.
    knn = KNeighborsClassifier(n_neighbors=5)
    # Fit the model on the training data.
    train_targets = [i[0] for i in train_data[y_column].values]
    knn.fit(train_data[x_columns].values, train_targets)
    # Make point predictions on the test set using the fit model.
    predictions = knn.predict(test_data[x_columns])
    # Get the actual values for the test set.
    actual = test_data[y_column]
    actual_arr = actual.to_numpy()
    # Compute the mean squared error of our predictions.
    # mse = (((predictions - actual) ** 2).sum()) / len(predictions)
    # print("------------predictions----------------")
    # print(predictions)
    # print("------------actual----------------")
    # print(actual_arr.T[0])
    # print("------------acc----------------")
    # acc = np.sum(actual_arr.T[0] == predictions) / len(actual_arr)
    # print(acc)
    test_accuracy = knn.score(test_data[x_columns].values, [i[0] for i in test_data[y_column].values])
    print(test_accuracy)
