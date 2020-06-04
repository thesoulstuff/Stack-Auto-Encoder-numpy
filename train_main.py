import numpy as np
import pandas as pd


def sigmoid(x):
    return 1/(1+np.exp(-x))

def glorot_uniform(fan_in, fan_out, size):
    '''Generate tensor uniform xavier glorot for the weights'''
    r = np.sqrt(6/(fan_in + fan_out))
    w = abs(fan_out-fan_in)*np.random.random_sample(size) + min(fan_in, fan_out)
    w = w*2*r
    w = w - r
    return w



def stack_autoencoder(X, node_list, C):
    '''
    X : numpy.Array - Input data to select features
    node_list: list - features

    Return list of autoencoder weights
    '''
    Xnew = X.copy()
    input_size = Xnew.shape[1]
    samples = Xnew.shape[0]
    weight_list = []
    
    for nodes in node_list:
        # I'm using the sigmoid function but I still believe is useless
        print(Xnew.shape)
        print(samples, nodes)
        w = glorot_uniform(input_size, nodes, (input_size, nodes))
        w2 = glorot_uniform(nodes, input_size, (nodes, input_size))

        syn0 = sigmoid(Xnew.dot(w))
        syn1 = syn0.dot(w2)
        newW = output_learning(syn0, C, Xnew)
        weight_list.append(newW)
        Xnew = Xnew.dot(newW.T)
        
        input_size = Xnew.shape[1]
        
    return Xnew, weight_list



def output_learning(H, C, target_output):
    U = H.T.dot(H)
    U = U + np.identity(U.shape[0])/C
    U = np.linalg.pinv(U)
    U = U.dot(H.T)
    U = U.dot(target_output)
    return U


if __name__ == '__main__':
    df_output = pd.read_csv("data_label.csv", header=None)
    df_input = pd.read_csv("data_input.csv", header=None)
    nodes = 50
    C = 10**6
    samples = df_input.shape[1]
    w = np.random.random((samples, nodes))*2 - 1
    w2 = np.random.random((nodes, samples))*2 -1
    X = df_input.copy()
    syn0 = X.dot(w)
    syn1 = syn0.dot(w2)
    new = output_learning(syn0, C, X)
    x, _ = stack_autoencoder(X, [64, 32, 16, 8], C)
    print(X)
    print(x)

    
    

