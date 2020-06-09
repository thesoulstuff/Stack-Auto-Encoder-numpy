from train_main import *
from pre_proceso import *
import numpy as np
import pickle


if __name__ == '__main__':
    with open('param_sae.csv', 'r') as f:
        params = f.readlines()
    params = [float(x.strip()) for x in params]
    p_train = params[0]
    C = int(params[1])
    ae_nodes = []
    for i in range(2,len(params)):
        ae_nodes.append(int(params[i]))

    #params softmax
    with open('param_softmax.csv', 'r') as f:
        params = f.readlines()

    params = [float(x.strip()) for x in params]
    
    
    epochs = int(params[0])
    l= params[1]
    lamb= params[2]

    

    x_t = pd.read_csv("test_input.csv", header=None).iloc[0:,:].values
    y_t = pd.read_csv("test_label.csv", header=None).iloc[0:,:].values

    with open("w_ae.pkl", "rb") as f:
        w_ae = pickle.load(f)

    with open("deepl_pesos.npy", "rb") as f:
        w = np.load(f)

    predict(x_t, y_t, ae_nodes, w, C, lamb, w_ae)
    
