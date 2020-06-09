import numpy as np
import pandas as pd
import csv
import pickle


def sigmoid(x):
    return 1/(1+np.exp(-x))

def glorot_uniform(fan_in, fan_out, size):
    '''Generate tensor uniform xavier glorot for the weights'''
    r = np.sqrt(6/(fan_in + fan_out))
    w = np.random.uniform(low=-r, high=r, size=size)
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
        w = glorot_uniform(nodes, input_size, (input_size, nodes))

        syn0 = sigmoid(Xnew.dot(w))
        #syn1 = syn0.dot(w2)
        #syn1 = w2.dot(syn0.T)
        newW = output_learning(syn0, C, Xnew)
        weight_list.append(newW.T)
        
        Xnew = sigmoid(Xnew.dot(newW.T))
        #jprint(newW)
        
        input_size = Xnew.shape[1]
        
    return Xnew, weight_list
        



def output_learning(H, C, target_output):
    U = H.T.dot(H)
    U = U + np.identity(U.shape[0])/C
    U = np.linalg.pinv(U)
    U = U.dot(H.T)
    U = U.dot(target_output)
    return U

def softmax(x):
    # max as a fix for the infinity result from numpy
    e_x = np.exp(x - np.max(x))
    s_e_x = e_x.sum(axis=0, keepdims = True)
    #print("SOFMAX DIVIDER: ", s_e_x, len(s_e_x))
    return e_x/s_e_x

def cross_entropy_loss(Y, A, l, w):
    return -np.sum(Y* np.log(np.nan_to_num(A.T, nan=1e-9)+ 1e-8), axis=1) #+ (l/2)*(np.linalg.norm(w))**2

def classifier(x, Y, l, lamb, epochs):
    #weight creation
    input_features = x.shape[1]
    labels = Y.shape[1]
    w = glorot_uniform(labels, input_features, (labels, input_features))
    b = glorot_uniform(labels, input_features, (1, labels))
    x = x.T
    debug = 0
    full_scores = []
    full_cost = []
    #foward propagation
    for i in range(epochs):

        
        Z = w.dot(x) #+ b
        A = softmax(Z)
        cost = np.mean(cross_entropy_loss(Y, A, l, w))
        full_cost.append(cost)
        e = A - Y.T

        y_true = np.apply_along_axis(np.argmax, axis=0, arr=Y.T)
        y_pred = np.apply_along_axis(np.argmax, axis=0, arr=A)
        e = np.nan_to_num(e, nan=1e-9)
        dw = (e.dot(x.T))

        scores = []
        for i in range(labels):
           scores.append(compute_f1_score(y_true, y_pred, i))
        full_scores.append(scores)
        


        w = w - l*dw + lamb*w
    np.savetxt("deepl_pesos.csv", w, delimiter=",")
    with open("deepl_pesos.npy", 'wb') as f:
        np.save(f, w)
    with open('deepl_costos.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in full_cost:
            writer.writerow([row])
    print("F1 Score final de entrenamiento: \n", scores)
    return w, full_cost 



def compute_f1_score(y_true, y_pred, label):
    # calculates the F1 score
    tp, tn, fp, fn = compute_tp_tn_fn_fp(y_true, y_pred, label)
    precision = compute_precision(tp, fp)/100
    recall = compute_recall(tp, fn)/100
    f1_score = (2*precision*recall)/ (precision + recall)
    f1_score = 0 if np.isnan(f1_score) else f1_score
    return f1_score




def compute_recall(tp, fn):
    '''	
    Recall = TP /FN + TP 

    '''
    return (tp  * 100)/ float( tp + fn)

def compute_precision(tp, fp):
    '''
    Precision = TP  / FP + TP 

    '''
    return (tp  * 100)/ float( tp + fp)



def compute_accuracy(tp, tn, fn, fp, samples):
    '''
    Accuracy = TP + TN / FP + FN + TP + TN

    '''
    return ((tp + tn) * 100)/ float( tp + tn + fn + fp)



def compute_tp_tn_fn_fp(y_act, y_pred, label):

    '''
    True positive - actual = 1, predicted = 1
    False positive - actual = 1, predicted = 0
    False negative - actual = 0, predicted = 1
    True negative - actual = 0, predicted = 0
    '''
    tp = sum((y_act == label) & (y_pred == label))
    tn = sum((y_act != label) & (y_pred != label))
    fn = sum((y_act == label) & (y_pred != label))
    fp = sum((y_act != label) & (y_pred == label))
    return tp, tn, fp, fn


    #predict
def predict(x_t, y_t, ae_nodes, w, C, lamb, w_ae):
    #x, weights = stack_autoencoder(x_t, ae_nodes, C)
    x = x_t.copy()
    for weight in w_ae:
        x = sigmoid(x.dot(weight))
    x = x.T
    Z = w.dot(x)
    A = softmax(Z)
    cost = np.mean(cross_entropy_loss(y_t, A, lamb, w))
    y_true = np.apply_along_axis(np.argmax, axis=0, arr=y_t.T)
    y_pred = np.apply_along_axis(np.argmax, axis=0, arr=A)
    print(y_true)
    print(y_pred)
    scores = []
    y_features = y_t.shape[1]
    for i in range(y_features):
        scores.append(compute_f1_score(y_true, y_pred, i))
    with open('fscore.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(scores)




if __name__ == '__main__':
    df_output = pd.read_csv("train_label.csv", header=None)
    df_input = pd.read_csv("train_input.csv", header=None)
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


    X = df_input.copy().iloc[0:,:].values
    df_output = df_output.iloc[0:,:].values
    
    x, weights = stack_autoencoder(X,  ae_nodes, C)
    w, _ = classifier(x, df_output, l, lamb, epochs)
    
    


        
    

