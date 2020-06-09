import pandas as pd
import numpy as np


#TODO
# [x]read data
# [x]separate input from labels
# [x]one-hot enconding of labels
# [x]save values in csv

def prepare_data(path, train_p):
    #read file, pandas is the faster way
    df = pd.read_csv(path, header = None)
    #shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    df_train = df.iloc[:int(train_p*len(df))]
    df_test = df.iloc[int(train_p*len(df)):]

    #separate input from lables
    df_input_train = df_train.iloc[:,:-1]
    df_label_train = df_train.iloc[:,-1:]
    df_input_test = df_test.iloc[:,:-1]
    df_label_test = df_test.iloc[:,-1:]
    #one-hot encoding
    df_label_train = pd.get_dummies(df_label_train.iloc[:,0])
    df_label_test = pd.get_dummies(df_label_test.iloc[:,0])

    df_input_train.to_csv('train_input.csv', header=False, index=False)
    df_label_train.to_csv('train_label.csv', header=False, index=False)
    df_input_test.to_csv('test_input.csv', header=False, index=False)
    df_label_test.to_csv('test_label.csv', header=False, index=False)


if __name__ == '__main__':
    with open('param_sae.csv', 'r') as f:
        params = f.readlines()
    params = [float(x.strip()) for x in params]
    p_train = params[0]

    prepare_data('Data.csv', p_train)
