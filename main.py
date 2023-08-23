import numpy as np
import torch
from syndata import load_data
from utils import batchify_data, train_model, run_epoch
from mycnn import CNN


def main():
    # Load data
    X_train, Y_train = load_data('synthetic_data_v3')
    # print(X_train, Y_train)
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
    batch_size = 30
    train_batches = batchify_data(X_train, Y_train, batch_size)
    dev_batches = batchify_data(X_dev, Y_dev, batch_size)
    #### Aqui falta test data ######
    model = CNN(1,6)
    train_model(train_batches, dev_batches, model, nesterov=True)

    # Evaluate the model on the test data set
    # loss, accuracy = run_epoch(test_batches, model.eval(), None)

    # print("Loss on test set:" + str(loss) + " Accuracy on test set: " + str(accuracy))





if __name__ == '__main__':
    main()
