'''
    Deep neural network training using backpropagation.
    Inspired by this tutorial:
    https://mlfromscratch.com/neural-network-tutorial/#/

    Create a deep neaural network and train it to detect handwritten digits.

    Author: Kasper Skov Hansen
'''
print('Importing libraries...')
import numpy as np                                      # Matrix handling
from datetime import datetime                           # Track time
from sklearn.datasets import fetch_openml               # Load mnist dataset
from keras.utils.np_utils import to_categorical         # Convert to one-hot encoded labels
from sklearn.model_selection import train_test_split    # Split dataset into training and test
from os import path                                     # Filesystem

def load_mnist_data():
    '''
        Load mnist labeled handwritten digits and split into training data and test data.
    '''
    # Load data and labels from mnist dataset
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # Normalize to values between 0 and 1
    x = (x/255).astype('float32')
    # Convert labels to one-hot encoded labels
    y = to_categorical(y)
    
    # Split dataset into training data and test data
    # [x_train, x_val, y_train, y_val]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)    
    
    return [x_train, x_val, y_train, y_val]


class DeepNeuralNetwork():    
    def __init__(self, sizes, epochs=10, learning_rate=0.01):   
        '''
            Create and train a deep neural network
            sizes: Sizes of each layer in the network
        '''
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # All network parameters are saved here
        self.params = self.initializer(sizes)


    def initializer(self, sizes):
        '''
            Setup the dimensions of the network
            sizes: array containing number of nodes in each layer
        '''        

        input_layer = sizes[0]
        layer_1 = sizes[1]
        layer_2 = sizes[2]
        output_layer = sizes[3]
                  
        params = {
            'W1': np.random.randn(layer_1, input_layer) * 0.1,
            'W2': np.random.randn(layer_2, layer_1) * 0.1,
            'W3': np.random.randn(output_layer, layer_2) * 0.1
        }    
        return params

    def sigmoid(self, x, derivative=False):
        '''
            Activation function with the caractiristic S-shape. The values are treated separately and will have a value between 0 and 1.
            x: Vector with input values.
            derivative: Whether the derivative should be returned instead.
        '''
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1+np.exp(-x))

    def softmax(self, x, derivative=False):
        '''
            Activation function often used in the final layer. The probabilities will add up to 1.
            x: Vector with input values.
            derivative: Whether the derivative should be returned instead.
        '''
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)
    
    def forward_pass(self, x_data):
        '''
            Calculate the output from the network for a given set of input values.
            x_data: Input values for the network.
        '''
        params = self.params
               
        params['A0'] = x_data

        # Matrix vector multiplication
        params['Z1'] = np.dot(params['W1'], params['A0'])  
        params['A1'] = self.sigmoid(params['Z1'])

        params['Z2'] = np.dot(params['W2'], params['A1'])        
        params['A2'] = self.sigmoid(params['Z2'])

        params['Z3'] = np.dot(params['W3'], params['A2'])        
        params['A3'] = self.sigmoid(params['Z3'])

        return params['A3']

    def backward_pass(self, output, y_data):
        '''
            Using the backpropagation algorithm to calculate the updates for the neural network parameters.
            output: Result from forward pass.
            y_data: Correct labels for given example

             Note: There is a stability issue that causes warnings. This is 
                  caused  by the dot and multiply operations on the huge arrays.
                  
                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        new_weights = {}
                
        # Calculate W3 update
        error = 2 * (output - y_data) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        new_weights['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        new_weights['W2'] = np.outer(error, self.params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        new_weights['W1'] = np.outer(error, params['A0'])
        
        return new_weights

    def compute_accuracy(self, x_val, y_val):
        '''
            Test the networc accuracy by comparing the network output with correct labels for a number of test examples. 
            The accuracy is returned as a number between 0 and 1.
            x_val: List of input vectors.
            y_val: List of vectors containing output labels.
        '''
        predictions = []
        for x, y in zip(x_val, y_val):
            pred = self.forward_pass(x)
            predictions.append(pred.argmax() == y.argmax())
        return np.mean(predictions)

    def update_network_params(self, new_params):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.
            new_params: Dictionary containing the gradients ∇J(x, y).

            θ = θ - η * ∇J(x, y), 
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''        
        for key, value in new_params.items():
            self.params[key] -= self.learning_rate * value     

    def train_network(self, x_train, x_val, y_train, y_val):
        '''
            Train the network using the specified training and test data.
            x_train: List of input vectors for the network to use in training.
            x_val: List of input vectors for the network to use in testing.
            y_train: Labels corresponding to x_train.
            y_data: Labels corresponding to x_val.
        '''
        start_time = datetime.now()        
        
        # Iterate over each epoch and compute accuracy of network for each iteration.      
        for epoch in range(self.epochs):            
            # Iterate over all training examples in dataset and tune network parameters.                                  
            for x, y in zip(x_train, y_train):                
                output = self.forward_pass(x)                
                new_weights = self.backward_pass(output, y)                
                self.update_network_params(new_weights)

            # Test how well the network performs
            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch {} - Elapsed time: {} - Accuracy: {}'.format(epoch + 1, datetime.now() -  start_time, accuracy))
            print('Saving...')
            np.savez('network_params.npz', **self.params)
            print('Done saving.')


np.random.seed(0)

print('Loading data...')
dataset = load_mnist_data()
print('Creating network...')
myNet = DeepNeuralNetwork([784, 128, 64, 10])


# Load trained network from .npz file
network_path = 'network_params.npz'
if path.exists(network_path):
    print('Loading params from external file...')
    params = np.load(network_path)
    for key in params.files:
        myNet.params[key] = params[key]
    
print('Training network...')
myNet.train_network(dataset[0], dataset[1], dataset[2], dataset[3])