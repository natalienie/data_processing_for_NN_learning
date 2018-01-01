import quandl
import pandas as pd
import json
import urllib.request
import numpy as np
import time
import pickle


def JsonToDf(exchange, contract_name):


    url = 'https://www.quandl.com/api/v3/datasets/' + exchange + '/' + contract_name + '.json?api_key=5DJz3EfSFxJq2RYxe1ZA'
    opem1 = urllib.request.urlopen(url)
    r = opem1.read()
    opem1.close()
    dic = json.loads(r.decode())
    column_names = dic['dataset']['column_names']
    data = dic['dataset']['data']
    df = pd.DataFrame(data, columns = column_names)
    df2 = df.sort_values(by='Date', ascending=1)
    df2.set_index('Date')

    return df2

def create_feature_set_keras(exchange, contract_name, step):
    df = JsonToDf(exchange, contract_name)
    df_price = df['Settle']
    doc_X, doc_y = [], []
    for i in range(df_price.shape[0]/step - 1):
        doc_X.append(df_price[i*step : (i+1)*step])
        doc_y.append(df_price[i*step + 1 : (i+1)*step + 1])
    return doc_X, doc_y

def creat_feature_set_RNN(exchange, contract_name, N, K):
    '''
    N = sample_size
    K = the number predicted into the future
    batch_size = N
    chunk_size = 1
    n_chunks = N
    '''
    df = JsonToDf(exchange, contract_name)
    df_price = df['Settle']
    doc_X, doc_y = [], []

    leng = len(df_price)

    L = np.arange(N+K) + np.arange(leng - (N+K))
    for l in L:

        M = df_price[l]
        l_X = M[:N]
        l_y = M[N:]
        l_X_reshaped = l_X.reshape(-1, 1)
        l_y_reshaped = l_y.reshape(-1, 1)
        doc_X.append(l_X_reshaped)
        doc_y.append(l_y_reshaped)

    return doc_X, doc_y
