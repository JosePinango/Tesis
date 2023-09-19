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

def downloader(ticker: str, start_date: str, end_date: str) -> Tensor:
    yahoo_financials = YahooFinancials(ticker)
    data = yahoo_financials.get_historical_price_data(start_date=start_date, end_date=end_date, time_interval='daily')
    return data

def normalization(data: Tensor) -> Tensor:
    std, mean = torch.std_mean(data)
    output = (data - mean) / std
    return output

if __name__ == '__main__':
# def download_dataset():
    pattern_templates = ['Rising wedge', 'Head-and-shoulders top', 'Cup with handle', 'Triangle, ascending', 'Triple top', 'Double Bottom, Adam and Eve']
    dataset = []
    label_set = []
    ticker_fail = []
    with open('bulkowski_data_v1.csv', mode='r') as csv_file:
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
                end_date = datetime.strptime(row[4], '%m/%d/%Y').strftime('%Y-%m-%d')
                ticker = row[0]
                # print(f'ticker {ticker}')
                # data = downloader(row[0], start_date=start_date, end_date=end_date)
                try:
                    data = time_series.read_ts_from_ibdb(row[0], '1 day', None, end_date, last=64)
                    # print(data)
                    # print(type(data))
                    # print(type(data[0]))
                    # print(data[0]['adj_close'])
                    data = data[0]['adj_close']
                    # time.sleep(60)

                    # data_df = pd.DataFrame(data)
                    # print(data_df)
                    # data_df = pd.DataFrame(data[row[0]]['prices'])
                    # data_df = data_df.drop('date', axis=1) #.set_index('formatted_date')
                    # data_df = data_df['adjclose']
                    # data_df.head()
                    # data = data_df.values[:64]
                    # time.sleep(60)
                    # print(data.values)
                    data = torch.tensor(data.values)
                    print(data)

                    # data = normalization(data)
                    try:
                        label = torch.tensor(pattern_templates.index(row[1]))
                    except ValueError:
                        label = torch.tensor(6)
                    print(label)
                    if data.shape[-1] == 64:
                        dataset.append(data.reshape(1,-1))

                        label_set.append(label.reshape(1,-1))
                except KeyError:
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


    with open('real_data_v3.pt', 'wb') as f:
        torch.save((real_data, real_label), f)

    with open('symbols_not_found_v1.csv', 'w') as f:
        writer = csv.writer(f)
        for row in ticker_fail:
            writer.writerow(row)

