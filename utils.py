import torch
from syndata import load_data


def data_labeling()

def batchify_data(data, batch_size):
    n = data.shape[0]
    batches = []
    for i in range(0, n, batch_size):
        batches.append({'x': data[i:i + batch_size], 'y': i * torch.ones(batch_size, 1, 1)})
    return batches


if __name__=='__main__':
    inputs = load_data()
    # print(inputs)
    output1 = batchify_data(inputs[0], 3)
    print(output1)

