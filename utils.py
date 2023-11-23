import torch
from syndata import load_data
from torch import Tensor
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


# def data_labeling()

def batchify_data(X_data: Tensor, Y_data: Tensor, batch_size: int):
    n = X_data.shape[0]
    batches = []
    for i in range(0, n, batch_size):
        batches.append({'x': X_data[i:i + batch_size], 'y': Y_data[i:i + batch_size]})
    return batches

def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    # a1 = np.equal(predictions.numpy(), y.numpy())
    # a2 = np.mean(np.equal(predictions.numpy(), y.numpy()))
    return np.mean(np.equal(predictions.numpy(), y.numpy()))

def train_model(train_data, dev_data, model, lr=0.005, momentum=0.9, nesterov=False, n_epochs=4):
    """Train a model for N epochs given data and hyper-params."""
    # We optimize with SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    # print(train_data)

    for epoch in range(1, n_epochs):
        print("-------------\nEpoch {}:\n".format(epoch))


        # Run **training***
        loss, acc = run_epoch(train_data, model.train(), optimizer)
        print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))

        # Run **validation**
        val_acc = 'No calculado'
        val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer)
        print('Val loss:   {:.6f} | Val accuracy:   {:.6f}'.format(val_loss, val_acc))
        # Save model
        torch.save(model, 'neural_network_v2.pt')
    return val_acc

def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, acccuracy"""
    # Gather losses
    losses = []
    batch_accuracies = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for batch in tqdm(data):
        # Grab x and y
        x, y = batch['x'], batch['y']
        # print(x,y)

        # Get output predictions
        out = model(x)
        # print(f'ddddddddddddddddd {out}')

        # Predict and store accuracy
        predictions = torch.argmax(out, dim=1)
        # print(predictions)
        # print(y.reshape(-1))
        batch_accuracies.append(compute_accuracy(predictions, y.reshape(-1)))
        # print(batch_accuracies)

        # Compute loss
        loss = F.cross_entropy(out, y.reshape(-1))
        losses.append(loss.data.item())

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(batch_accuracies)
    return avg_loss, avg_accuracy


if __name__ == '__main__':
    filename2 = 'synthetic_data'
    X_data, Y_data = load_data(filename2)
    # print(inputs)
    output1 = batchify_data(X_data, Y_data, 5)
    # print(output1)
