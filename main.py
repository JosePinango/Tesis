import numpy as np
import torch
from syndata import load_data, vec_normalization, normalization, normalization64, normalization_real, transform_database, normalization_realdata
from utils import batchify_data, train_model, run_epoch
from mycnn import CNN
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
# from Download_Real_Data.download_dataset import recognition_pattern
from download_dataset import recognition_pattern
import time
import time_series

def main():
    # ticker = 'MSFT'
    # data_sp500 = load_data('data_sp500')
    # aapl = data_sp500[ticker]
    # data_aux = aapl[aapl.index <= '2019-09-16']
    # data_aux = data_aux[-31:]
    # data = torch.Tensor(aapl.values)
    # dates = aapl.index

    # Load data
    X_train, Y_train = load_data('data3patterns_v1')
    X_train = X_train.type(torch.float32)
    data_sp500 = load_data('data_sp500')
    symbols_list = load_data('sp500')
    list_random = []
    for symbol in symbols_list[:90]:
        try:
            aapl = data_sp500[symbol]
            data_aux = aapl[aapl.index <= '2023-09-23']

            i=0
            print(data_aux.size)
            while i < data_aux.size - 31:
                data_aux1 = data_aux[i:i+31]
                data = torch.Tensor(data_aux1.values).reshape(1, -1)
                data = normalization(data).reshape(1, 1, -1)
                list_random.append(data)
                i += 31
        except KeyError:
            pass
    X_random = torch.cat(list_random)


    # SP500_data = load_data('data_sp500')



    print(f'Size of X_random: {X_random.shape}')
    # print(X_random)
    # time.sleep(30)


    Y_random = torch.full((X_random.shape[0],1,1), 3)
    X_train = torch.cat([X_train, X_random])
    Y_train = torch.cat([Y_train, Y_random])
    permutation = torch.randperm(X_train.shape[0])
    X_train = X_train[permutation]
    Y_train = Y_train[permutation]
    # print(X_train)
    # print(Y_train)
    # print(X_train.shape)
    # print(Y_train.shape)
    # X_test, Y_test = load_data('real_data_05nov23')
    # print(X_test)
    # print(X_test.shape)
    # print(Y_test)
    # norm_list = []
    # for i in range(X_test.shape[0]):
    #     norm_datatest = normalization_realdata(X_test[i])
    #     norm_list.append(norm_datatest)
    # print(norm_list)
    # X_test = torch.stack(norm_list)
    # X_test = normalization_realdata(X_test)
    # print('after normalization')
    # print(X_test)
    # print(X_test.shape)
    # print(Y_test)
    # X_data = load_data('Download_Real_Data/AAPL')
    # X_train = transform_database(X_train)
    #
    # permutation = torch.randperm(X_test.shape[0])
    # X_test = X_test[permutation]
    # Y_test = Y_test[permutation]

    # X_test = X_test[:,:,:,:-33]
    # X_test = transform_database(X_test)
    # p1 = Y_test == 5
    # p2 = Y_test == 6
    # p3 = Y_test == 9
    # p1 = p1.reshape(-1)
    # p2 = p2.reshape(-1)
    # p3 = p3.reshape(-1)
    # X_train = torch.rand(10000,1,4,31)
    # Y_train = torch.randint(0,10,(10000,1,1,1))
    # X_test = torch.rand(1000,1,4,31)
    # Y_test = torch.randint(0,10,(1000,1,1,1))
    # X_test1 = X_test[p1]
    # Y_test1 = Y_test[p1]
    # Y_test1[:,:,:,:] = 0
    # X_test2 = X_test[p2]
    # Y_test2 = Y_test[p2]
    # Y_test2[:,:,:,:] = 1
    # X_test3 = X_test[p3]
    # Y_test3 = Y_test[p3]
    # Y_test3[:,:,:,:] = 2
    # X_test = torch.cat([X_test1, X_test2, X_test3], dim=0)
    # Y_test = torch.cat([Y_test1, Y_test2, Y_test3], dim=0)
    # X_test = torch.cat([X_test2, X_test3], dim=0)
    # Y_test = torch.cat([Y_test2, Y_test3], dim=0)

    # X_data = X_test[0,0]
    # # X_test = pd.DataFrame(X_data.numpy())

    # for n in range(0,10):
    #     X_data = X_test[n, 0]
    # #     b = Y_test[n]
    # #
    # # #     # x_date = pd.DataFrame(x.numpy())
    #     dates = pd.date_range(start='1/1/2023', end='1/31/2023')
    #     fig = go.Figure(data=[go.Candlestick(x=dates,
    #                                          open=X_data[0],
    #                                          high=X_data[1],
    #                                          low=X_data[2],
    #                                          close=X_data[3])])
    #     fig.show()
    #     x = torch.linspace(0, 30, steps=31)
    #     plt.plot(x, X_test[n, 0, 0].reshape(-1))
    #     plt.plot(x, X_test[n, 0, 1].reshape(-1))
    #     plt.plot(x, X_test[n, 0, 2].reshape(-1))
    #     plt.plot(x, X_test[n, 0, 3].reshape(-1))
    #     plt.show()

    # permutation = torch.randperm(X_test.shape[0])
    # X_test = X_test[permutation]
    # Y_test = Y_test[permutation]
    #
    # print(X_test.shape)
    # for i in range(X_test.shape[0]):
    #     for j in range(X_test.shape[-2]):
    #         X_test[i,-1,j] = normalization64(X_test[i, -1, j])
    # X_test = vec_normalization(X_test)
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # X_test = X_test[:,:,0:2,:]


    # split_index = int(9 * X_test.shape[0] / 10)
    # X_train = X_test[:split_index]
    # Y_train = Y_test[:split_index]
    # X_test = X_test[split_index:]
    # Y_test = Y_test[split_index:]

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
    # test_batches = batchify_data(X_test, Y_test, batch_size)
    #### Aqui falta test data ######
    model = CNN(1, 4,4)
    train_model(train_batches, dev_batches, model, nesterov=True)
    # trained_model = CNN(1, 6,6)
    # trained_model.load_state_dict(torch.load('neural_network_v2.pt'))
    # trained_model.eval()

    # Evaluate the model on the test data set
    # loss, accuracy = run_epoch(test_batches, model.eval(), None)

    # print("Loss on test set:" + str(loss) + " Accuracy on test set: " + str(accuracy))

    # aapl = load_data('AAPL')
    # aapl = torch.tensor(aapl.values)
    # for i in range(1000):
    #     if len(aapl[i:i + 62]) > 61:
    #         print(i)
    #         print(model(aapl[i:i + 62].reshape(1,1,1,-1)))
    # ticker = 'MSFT'
    ticker = 'RCL'

    # data = time_series.read_ts_from_ibdb(ticker, '1 day', None, '2023-08-31', last=2000)
    # data_adj_close = data[0]['adj_close']
    # # data = torch.stack([data_open, data_high, data_low, data_close])
    # with open(ticker + '.pt', 'wb') as f:
    #     torch.save(data_adj_close, f)
    # date_end = '2019-09-16' #MSFT
    # date_end = '2022-09-23' #RCL
    # data_sp500 = load_data('data_sp500')
    # aapl = data_sp500[ticker]
    # data_aux = aapl[aapl.index <= date_end]
    # data_aux = data_aux[-31:]
    # data = torch.Tensor(data_aux.values).reshape(1,1,-1)
    # dates = data_aux.index
    # pred = model(data)
    # print(f'Symbol: {ticker}, End date: {date_end}, Prediction: {pred} >> wedge_rising--head_and_shoulders--cup_with_handle--triangle_ascending--triple_tops--double_bottoms_eve_adam')

    recognition_pattern(ticker, model.eval())
    patterns = load_data(ticker)
    print(patterns)
    labels = patterns[1]
    patterns = patterns[0]
    # pattern_titles = ['wedge_rising', 'head_and_shoulders', 'cup_with_handle', 'triangle_ascending',
    #                      'double_bottoms_eve_adam', 'random']
    pattern_titles = ['head_and_shoulders', 'cup_with_handle', 'triangle_ascending',
                      'random']
    n = patterns.shape[0]
    x = torch.linspace(0, 30, steps=31)
    for i in range(n):
    #     plt.plot(x, X_test[n, 0, 0].reshape(-1))
    #     plt.plot(x, X_test[n, 0, 1].reshape(-1))
    #     plt.plot(x, X_test[n, 0, 2].reshape(-1))
        if labels[i] <3:
            plt.plot(x, patterns[i, 0].reshape(-1))
            plt.title(pattern_titles[labels[i].item()])
            plt.show()
            time.sleep(5)
    # model1 = model.eval()
    # output = model(X_test)
    # predictions = torch.argmax(output, dim=1)
    #
    # confusion_matrix = torch.zeros(3,3)
    # confusion_matrix = torch.zeros(7, 7)

    # print(Y_test.shape)
    # Y_test = Y_test.reshape(-1)
    # print(Y_test)
    # print(predictions)

    # for label_test, label_predicted in zip(Y_train,predictions):
        # print(label_test)
        # for label_predicted in predictions:
            # print(label_predicted)
            # if label_test.item() == label_predicted.item():
        # confusion_matrix[label_test.item(), label_predicted.item()] += 1

    # print(confusion_matrix)
    # plt.figure()
    # plt.imshow(confusion_matrix, alpha=0.8)
    # plt.xticks(np.arange(10))
    # plt.yticks(np.arange(10))
    # plt.xlabel('Real')
    # plt.ylabel('Prediction')
    # plt.title('Confusion Matrix')
    # plt.show()



    # list_dates = ['2019-09-16', '2022-09-23']
    # list_symbols = ['MSFT', 'RCL']
    # for i in range(len(list_symbols)):
    #     ticker = list_symbols[i]
    #     date_end = list_dates[i]
    #     data_sp500 = load_data('data_sp500')
    #     aapl = data_sp500[ticker]
    #     data_aux = aapl[aapl.index <= date_end]
    #     data_aux = data_aux[-31:]
    #     data = torch.Tensor(data_aux.values).reshape( 1, -1)
    #     data = normalization(data).reshape(1,1,-1)
    #     # dates = data_aux.index
    #     pred = model(data)
    #     print(f'Symbol: {ticker}, End date: {date_end}, Prediction: {pred} >> wedge_rising--head_and_shoulders--cup_with_handle--triangle_ascending--triple_tops--double_bottoms_eve_adam')
    #
    #     x = torch.linspace(0, 30, steps=31)
    #     plt.plot(x, data[0, 0].reshape(-1))
    #     plt.title('Head and shoulders tops')
    #     plt.show()
    #     time.sleep(5)

#




if __name__ == '__main__':
    # X_test, Y_test = load_data('Download_Real_Data/real_data_v3')
    # print(Y_test.unique(return_counts=True))
    main()
