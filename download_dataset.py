import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
from torch import nn, Tensor
import numpy as np
from typing import Any, Callable, List, Optional, Tuple
import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta
import torch
import time
# import sys
# sys.path.append("/home/jose/Documents/Tesis/Download_Real_Data")
from Download_Real_Data import time_series
from syndata import load_data
from mycnn import CNN
import matplotlib.pyplot as plt


def downloader(ticker: str, start_date: str, end_date: str) -> Tensor:
    yahoo_financials = YahooFinancials(ticker)
    data = yahoo_financials.get_historical_price_data(start_date=start_date, end_date=end_date, time_interval='daily')
    return data


def normalization(data: Tensor) -> Tensor:
    std, mean = torch.std_mean(data)
    output = (data - mean) / std
    return output


def main():
    filename = 'cup_handle'
    # if __name__ == '__main__':
    # def download_dataset():
    # pattern_templates = ['Pipe bottom', 'Triangle, symmetrical', 'Pipe top', 'Double Bottom, Adam and Adam',
    #                      'Ugly double bottom', 'Double Top, Adam and Adam', 'Head-and-shoulders bottom',
    #                      'Dead-cat bounce', 'Triple bottom', 'Triple top']
    # pattern_templates = ['Head-and-shoulders top', 'Triangle, ascending', 'Triple top', 'Double Bottom, Eve and Adam']
    # pattern_templates = ['Rising wedge', 'Head-and-shoulders top', 'Cup with handle', 'Triangle, ascending', 'Double Bottom, Eve and Adam']
    pattern_templates = ['Head-and-shoulders top', 'Cup with handle', 'Triangle, ascending']
    dataset = []
    label_set = []
    ticker_fail = []
    label_aux_list = []
    with open('Download_Real_Data/' + filename + '.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # line_count = 0
        for value, row in enumerate(csv_reader):
            if value == 0:
                # line_count += 1
                # print(row)
                pass
            else:
                # try:
                # line_count +=1
                print(value)
                print(row)
                # start_date = datetime.datetime.strptime(row[3], '%m/%d/%Y').strftime('%Y-%m-%d')

                # start_date = (datetime.strptime(row[4], '%m/%d/%Y') - relativedelta(months=4)).strftime('%Y-%m-%d')
                # end_date = datetime.strptime(row[4], '%m/%d/%Y').strftime('%Y-%m-%d')
                end_date = (datetime.strptime(row[-1], '%m/%d/%Y') + relativedelta(days=7)).strftime('%Y-%m-%d')
                ticker = row[0]
                # print(f'ticker {ticker}')
                # data = downloader(row[0], start_date=start_date, end_date=end_date)
                try:
                    data = time_series.read_ts_from_ibdb(row[0], '1 day', None, end_date, last=70)
                    # print(data)
                    # print(type(data))
                    # print(type(data[0]))
                    # print(data[0]['adj_close'])
                    # data_open = data[0]['open']
                    # data_open = torch.tensor(data_open.values)
                    # data_high = data[0]['high']
                    # data_high = torch.tensor(data_high.values)
                    # data_low = data[0]['low']
                    # data_low = torch.tensor(data_low.values)
                    # data_close = data[0]['close']
                    # data_close = torch.tensor(data_close.values)
                    # data = torch.stack([data_open, data_high, data_low, data_close])
                    # time.sleep(60)

                    # data_df = pd.DataFrame(data)
                    # print(data_df)
                    # data_df = pd.DataFrame(data[row[0]]['prices'])
                    # data_df = data_df.drop('date', axis=1) #.set_index('formatted_date')
                    # data_df = data_df['adj_close']
                    # data_df.head()
                    # data = data_df.values[:64]
                    # time.sleep(60)
                    # print(data.values)
                    # data = torch.tensor(data.values)
                    data = data[0]['adj_close']
                    data = torch.Tensor(data.values)
                    print(data)

                    # data = normalization(data)
                    try:
                        # label_aux = row[1]
                        # label_aux_list.append(label_aux)
                        label = torch.tensor(pattern_templates.index(row[1]))
                    except ValueError:
                        label = torch.tensor(4)
                    print(f'Pattern: {filename}, label: {label}')
                    if data.shape[-1] == 70:
                        dataset.append(data.reshape(1, -1))

                        label_set.append(label.reshape(1, -1))

                    x = torch.linspace(0, 30, steps=70)
                    plt.plot(x, data.reshape(-1))
                    plt.title(filename + ' Pattern ')
                    plt.show()
                    time.sleep(15)

                except (KeyError, TypeError):
                    ticker_fail.append(row)
                # if value in np.arange(0,730, 25):
                #     time.sleep(30)

                # print(torch.tensor(data))
                # print(label)
                # print(f'sssss {row}')
            # except (TypeError, KeyError):
            #     ticker_fail.append(row[0])
            #     ticker_fail.append(value)

    # print(ticker_fail)
    # list = []
    # try:
    real_data = torch.stack(dataset, dim=0)
    print(real_data.shape)

    real_label = torch.stack(label_set, dim=0)
    print(real_label.shape)
    # except RuntimeError:
    #     pass

    # with open('all_patterns.pt', 'wb') as f:
    #     torch.save(label_aux_list, f)

    with open(filename + '_v2.pt', 'wb') as f:
        torch.save((real_data, real_label), f)

    with open('symbols_not_found_v2' + filename + '.csv', 'w') as f:
        writer = csv.writer(f)
        for row in ticker_fail:
            writer.writerow(row)

def cut_head_shoulders(filename: str) -> None:
    #(cut_s, cut_f, patt_s, patt_f, good_pattern=1)
    indexes = [(33,64,8,31,1), (33, 64,5,31,1),(34,65,16,31,1), (33,64,8,31,0),(33,64,5,31,0), (33,64,12,31,1), (32,63,11,31,1),(27,58,7,31,1)]
    with open(filename + '_v2.pt', 'rb') as f:
        data = torch.load(f)
        for i in range(data[0].shape[0]):
            x = torch.linspace(0, 30, steps=31)
            data_aux = data[0][i].reshape(-1)
            plt.plot(x, data_aux[27:58])
            plt.title(filename + ' Pattern ' + str(i))
            plt.show()
            time.sleep(10)

def recognition_pattern(ticker, model):
    # data = time_series.read_ts_from_ibdb(ticker, '1 day', None, '2023-08-31', last=2000)
    # data_adj_close = data[0]['adj_close']
    # # data = torch.stack([data_open, data_high, data_low, data_close])
    # with open(ticker + '.pt', 'wb') as f:
    #     torch.save(data_adj_close, f)
    data_sp500 = load_data('data_sp500')
    aapl = data_sp500[ticker]
    data = torch.Tensor(aapl.values)
    dates = aapl.index
    # aapl = normalization(data)
    aapl = data
    # aapl= aapl.type(torch.float64)
    # aapl = torch.rand(10000)
    # i = 0
    i = aapl.shape[-1]-1
    list_patterns = []
    list_labels = []
    # while i < aapl.shape[-1]:
    while i>30:
        # subsequence = aapl[i:i + 31]
        subsequence = aapl[i-31:i]
        subsequence = normalization(subsequence)
        dates_aux = dates[i]
        if len(subsequence) > 30:

            # print(i)

            output = model(subsequence.reshape(1, 1, -1))
            # print(f'Output: {output}')
            # print(f'First element: {output[0, 0]}')
            prediction = torch.argmax(output, dim=-1)
            label = prediction.item()
            print(f'Label: {prediction}')
            if label != 5 and output[0, label] > 0.99:
                print(f'Probability: {output[0, label].item()}')
                print(f'End date: {dates_aux}')
                i = i - 31

                # print(f'Nuevo índice: {i}')
                list_patterns.append(subsequence.reshape(1, -1))
                list_labels.append(torch.Tensor(prediction).reshape( 1, -1))
        i = i - 1
    patterns = torch.stack(list_patterns, dim=0)
    labels = torch.stack(list_labels, dim=0)
    with open(ticker + '.pt', 'wb') as f:
        torch.save((patterns, labels), f)
    print(patterns.shape)
    print(labels.shape)

def list_symbols_sp500(filename: str = 'sp500'):
    with open(filename + '.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # line_count = 0
        symbols_sp500 = {}
        for value, row in enumerate(csv_reader):
            print(value)
            print(row[0])
            symbols_sp500.append(row[0])
        symbols_sp500 = symbols_sp500[1:]
        print(symbols_sp500)
    with open('sp500.pt', 'wb') as f:
        torch.save(symbols_sp500, f)


def download_data_sp500(number_bar: int = 2000, end_date: str = '2023-10-27', filename: str = 'sp500'):
    with open(filename + '.pt', 'rb') as f:
        # data = torch.cat(torch.load(f), dim=0)
        sp500 = torch.load(f)
    dict_data_sp500 = {}
    not_founded_sp500 = []
    for symbol in sp500:
        try:
            data = time_series.read_ts_from_ibdb(symbol, '1 day', None, end_date, last=number_bar)
            data_adj_close = data[0]['adj_close']
            dict_data_sp500[symbol] = data_adj_close
        except (KeyError, TypeError):
            not_founded_sp500.append(symbol)
    with open('data_sp500.pt', 'wb') as f:
        torch.save(dict_data_sp500, f)
    print(dict_data_sp500)
    print(not_founded_sp500)


if __name__ == '__main__':
    main()
    # cut_head_shoulders('head_shoulders')
    # ticker = 'AMZN'
    # data = time_series.read_ts_from_ibdb(ticker, '1 day', None, '2023-08-31', last=2000)
    # # data_open = data[0]['open']
    # # data_open = torch.tensor(data_open.values)
    # # data_high = data[0]['high']
    # # data_high = torch.tensor(data_high.values)
    # # data_low = data[0]['low']
    # # data_low = torch.tensor(data_low.values)
    # # data_close = data[0]['close']
    # # data_close = torch.tensor(data_close.values)
    # data_adj_close = data[0]['adj_close']
    # # data = torch.stack([data_open, data_high, data_low, data_close])
    # with open(ticker + '.pt', 'wb') as f:
    #     torch.save(data_adj_close, f)
    # #
    # # with open('AAPL.csv', 'w') as f:
    # #     writer = csv.writer(f)
    # #     # for row in ticker_fail:
    # #     writer.writerow(data_adj_close)
    # model = CNN(1, 7, 7)
    # recognition_pattern(ticker, model.eval())
    # list_symbols_sp500()
    # download_data_sp500()
    # data = load_data('data_sp500')
    # print(data['AAPL'])


