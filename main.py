import numpy as np
import torch
from syndata import load_data, vec_normalization, normalization, normalization64
from utils import batchify_data, train_model, run_epoch
from mycnn import CNN


def main():
    # Load data
    X_train, Y_train = load_data('synthetic_data_v6')
    X_test, Y_test = load_data('Download_Real_Data/real_data_v3')
    X_test = X_test[:,:,:-33]
    print(X_test.shape)
    for i in range(X_test.shape[0]):
        X_test[i] = normalization(X_test[i])
    # X_test = vec_normalization(X_test)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    # Y_test = vec_normalization(Y_test)
    # print(X_train, Y_train)
    # print(X_train.shape)
    ####### ME FALTA PREGUNTAR ################
    # Split data

    split_index = int(9 * X_train.shape[0] / 10)
    X_dev = X_train[split_index:]
    Y_dev = Y_train[split_index:]
    X_train = X_train[:split_index]
    Y_train = Y_train[:split_index]


    # permutation = np.array([i for i in range(X_train.shape[0])])
    # np.random.shuffle(permutation)
    permutation = torch.randperm(X_train.shape[0])
    X_train = X_train[permutation]
    Y_train = Y_train[permutation]
    # print(X_train, Y_train)

    # Split dataset into batches
    batch_size = 35
    train_batches = batchify_data(X_train, Y_train, batch_size)
    dev_batches = batchify_data(X_dev, Y_dev, batch_size)
    test_batches = batchify_data(X_test, Y_test, batch_size)
    #### Aqui falta test data ######
    model = CNN(1,7)
    train_model(train_batches, dev_batches, model, nesterov=True)

    # Evaluate the model on the test data set
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print("Loss on test set:" + str(loss) + " Accuracy on test set: " + str(accuracy))





if __name__ == '__main__':
    main()
