import numpy as np


#TODO
#[ ]Create Layer class (maybe abstraction for input, dense and output)
##[ ]Set activation function (callback?)
##[ ]Set topology (input and nodes)
##[ ]Set foward pass
##[ ]Set Backpropagation (or whatever the teacher decided to use as learning)
#[ ]OPTIONAL: Create model function sequential-esque
#all the after things later... maybe use the model for the whole learning process?



class Layer:
    def foward_pass(self,):
        pass
    def learn(self, callback):
        pass

class Input(Layer):
    def __init__(self, input_values):
       self.nodes = input_values
       self.samples = input_values.shape[0]
       self.input_nodes = input_values[0].size
    def foward_pass(self, ):
        return self.nodes

class Dense(Layer):
    def __init__(self, size, weights=None, activation='linear'):
        self.size = size
        if weights is None:
            self.w = np.random.random(size)*2 -1
        else:
            self.w = weights
        self.activation = activation
    def foward_pass(self, other):
        output = np.dot(other, self.w)
        if self.activation == 'linear':
            return output
        elif self.activation == 'tanh':
            return np.tanh(output)
        elif self.activation == 'sigmoid':
            return 1/np.exp(output)
        
class Output(Layer):
    "maybe i could use the dense layer"
    def __init__(self, size):
        self.w = np.random.random(size)*2 - 1
        self.size = size
    def foward_pass(self,other):
        output = np.dot(other, self.w)
        return output




if __name__ == '__main__':
    #maybe test an autoencoder i dont know
    import pandas as pd
    df = pd.read_csv('data_input.csv', header = None).values
        print(df.shape)
    l1 = Input(df)
    nodes = 100
    in_size = l1.input_nodes
    l2 = Dense((in_size, nodes))
    nodes = 4
    #ol = Output()
    in_syn = l1.foward_pass()
    syn0 = l2.foward_pass(in_syn)
    #syn1 =
    print(syn0.size)

