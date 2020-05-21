import pandas as pd
import numpy as np


#TODO
# read data
# separate input from labels
# one-hot enconding of labels
# save values in csv

def prepare_data(path):
    #read file, pandas is the faster way
    df = pd.read_csv(path, header = None)
    #separate input from lables
    df_input = df.iloc[:,:-1]
    df_label = df.iloc[:,-1:]
    #one-hot encoding
    df_label = pd.get_dummies(df_label.iloc[:,0])
    df_input.to_csv('data_input.csv', header=False, index=False)
    df_label.to_csv('data_label.csv', header=False, index=False)


if __name__ == '__main__':
    prepare_data('./Data.csv')
