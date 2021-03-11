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
from tqdm import tqdm                                   # Progress bar
from matplotlib import pyplot as plt                    # Used for creating plots
from matplotlib import style                            # Graph style
style.use('fivethirtyeight')


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


def draw_images(images, guesses=None, labels=None):
    '''
        Draw a grid of misguessed images. Titles contain guesses and labels        
    '''
    fig2, axes = plt.subplots(5,5, figsize=(16,10), constrained_layout=True)    
    for i,ax in enumerate(axes.flat):
        if i < len(images):        
            ax.imshow(np.reshape(images[i], (28,28)))
            ax.axis('off')            
            ax.set_title('Guess: {}. Label: {}'.format(guesses[i], labels[i]))
        else: break
    fig2.suptitle('Digits from MNIST dataset', fontsize=16)
    plt.draw()


class DeepNeuralNetwork():    
    def __init__(self, sizes, epochs=10, learning_rate=0.01):   
        '''
            Create and train a deep neural network
            sizes: Sizes of each layer in the network
        '''
        
        self.sizes = sizes
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.ReLU = np.vectorize(self.relu_not_vect)
        
        # All network parameters are saved here
        self.params = self.initializer(sizes)


    def initializer(self, sizes):
        '''
            Setup the dimensions of the network including weights and biases.
            sizes: array containing number of nodes in each layer
        '''        
        params = {}

        for i in range(len(sizes) - 1):
            # Setup weights
            params['W' + str(i + 1)] = np.random.randn(sizes[i + 1], sizes[i]) * 0.1
            # Setup biases
            params['B' + str(i + 1)] = np.zeros(sizes[i + 1])

        # List for tested accuracies
        params['ACC'] = []
        # Time training for each epoch
        params['ACC_DT'] = []

        return params

    def sigmoid_not_vect(self, x, derivative=False):
        '''
            Activation function with the caractiristic S-shape. The values are treated separately and will have a value between 0 and 1.
            x: Vector with input values.
            derivative: Whether the derivative should be returned instead.
        '''
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1+np.exp(-x))

    def relu_not_vect(self, x, derivative=False):    
        if x > 0:
            if derivative:
                return 1
            else: return x
        else:
            return 0          

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

        # Iterate over all layers in order to calculate the final output
        for i in range(1, len(self.sizes)):            
            params['Z' + str(i)] = np.dot(params['W' + str(i)], params['A' + str(i - 1)]) + params['B' + str(i)]
            params['A' + str(i)] = self.ReLU(params['Z' + str(i)])

        # Return the activations from the last layer
        return params['A' + str(len(self.sizes) - 1)]

    def backward_pass(self, output, y_data):
        '''
            Using the backpropagation algorithm to calculate the updates for the neural network parameters.
            output: Result from forward pass.
            y_data: Correct labels for given example.
        '''
        params = self.params
        new_params = {}

        
        last_layer = len(self.sizes) - 1

        # Calculate last layer updates
        error = 2 * (output - y_data) / output.shape[0] * self.softmax(params['Z' + str(last_layer)], derivative=True)
        new_params['W' + str(last_layer)] = np.outer(error, params['A' + str(last_layer - 1)])        
        new_params['B' + str(last_layer)] = error

        # Propagate backwards and calculate other layer updates
        for i in range(1, last_layer):            
            # Calculate layer (last_layer - i) updates
            error = np.dot(params['W' + str(last_layer - i + 1)].T, error) * self.ReLU(params['Z' + str(last_layer - i)], derivative=True)
            new_params['W' + str(last_layer - i)] = np.outer(error, self.params['A' + str(last_layer - i - 1)])
            new_params['B' + str(last_layer - i)] = error
        
        return new_params

    def compute_accuracy(self, x_val, y_val):
        '''
            Test the networc accuracy by comparing the network output with correct labels for a number of test examples. 
            The accuracy is returned as a number between 0 and 1.
            x_val: List of input vectors.
            y_val: List of vectors containing output labels.
        '''
        predictions = []
        wrong = [[],[]]
        for i, (x, y) in enumerate(zip(x_val, y_val)):
            pred = self.forward_pass(x)
            if (pred.argmax() == y.argmax()):
                predictions.append(True)
            else:
                predictions.append(False)
                wrong[0].append(i)
                wrong[1].append(pred.argmax())

        return (np.mean(predictions), wrong)

    def sgd_update_network_params(self, new_params):
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


    def mini_b_update_network_params(self, new_params, divisor):
        for key, value in new_params.items():
            self.params[key] -= value / divisor


    def iterate_minibatches(self, inputs, targets, batch_size, shuffle=True):
        # Make sure the number of examples matches the number of targets
        assert inputs.shape[0] == targets.shape[0]

        indices = np.arange(inputs.shape[0])        # Create list of all indices            
        if shuffle: np.random.shuffle(indices)      # Shuffle indices

        # Remove indices from the end of indices array so that (indices.shape[0] % batch_size == 0)
        mod = indices.shape[0] % batch_size
        if not mod == 0: indices = indices[0: - int(indices.shape[0] % batch_size)]

        # Iterate over every start index for mini batches in indices array
        for start_idx in range(0, indices.shape[0] - batch_size + 1, batch_size):
            end_idx = start_idx + batch_size
            if shuffle: 
                excerpt = indices[start_idx : end_idx]
            else:
                excerpt = slice(start_idx, end_idx)
            # yield is used instead of return, as it is faster, and we only need to iterate over the mini-batches once
            yield [inputs[excerpt], targets[excerpt]]        


    def train_network(self, x_train, x_val, y_train, y_val, optimizer='MINI_B'):
        '''
            Train the network using the specified training and test data.
            x_train: List of input vectors for the network to use in training.
            x_val: List of input vectors for the network to use in testing.
            y_train: Labels corresponding to x_train.
            y_data: Labels corresponding to x_val.
        '''
        max_epoch = len(self.params['ACC']) + self.epochs
        # Iterate over each epoch and compute accuracy of network for each iteration
        for epoch in range(len(self.params['ACC']) + 1, max_epoch):
            print('Training epoch {} of {}...'.format(epoch, max_epoch))
            
            start_time = datetime.now()        

            if optimizer == 'MINI_B':
                # Mini-batching
                # Tune parameters with average from every example in a mini-batch at a time
                mini_batch_size = 4
                mini_batches = self.iterate_minibatches(x_train, y_train, mini_batch_size, shuffle=True)            
                for mini_batch in tqdm(mini_batches, total=(x_train.shape[0] - (x_train.shape[0] % mini_batch_size)) / mini_batch_size):
                    batch_x, batch_y = mini_batch
                    avg_new_weights = None
                    
                    for x, y in zip(batch_x, batch_y):
                        output = self.forward_pass(x)                    
                        new_weights = self.backward_pass(output, y)                
                        if not avg_new_weights: avg_new_weights = new_weights
                        else: 
                            for key, value in new_weights.items():
                                avg_new_weights[key] -= self.learning_rate * value
                    self.mini_b_update_network_params(new_weights, batch_x.shape[0])

            elif optimizer == 'SGD':
                # Stochastic gradient descent
                # Iterate over all training examples in dataset and tune network parameters for each example
                for x, y in zip(tqdm(x_train), y_train):
                    output = self.forward_pass(x)                
                    new_weights = self.backward_pass(output, y)                
                    self.sgd_update_network_params(new_weights)


            # Test how well the network performs
            accuracy, wrong = self.compute_accuracy(x_val, y_val)
            print('Epoch {} - Elapsed time: {} - Accuracy: {}'.format(epoch, datetime.now() -  start_time, accuracy * 100))
            print('Saving...')
            
            # Save the tested accuracy
            self.params['ACC'] = np.append(self.params['ACC'], accuracy)
            self.params['ACC_DT'] = np.append(self.params['ACC_DT'], (datetime.now() - start_time).total_seconds())            

            # Save the network params to disk
            np.savez('network_params.npz', **self.params)
            print('Done saving.')            
                    
            # Draw the training statistics to an interactive diagram
            acc = self.params['ACC']
            ax.plot(range(0, len(acc)), acc * 100)
            for i in range(len(acc)):
                ax.annotate('{} = {}'.format(i + 1, np.round(acc[i] * 100, 1)), (i, acc[i] * 100 + 0.4), ha='center', va='center')            

            # Draw diagram with 100 wrong guesses
            wrong_imgs = [[x_train[i]] for i in wrong[0]]             
            labels = [y_train[i].argmax() for i in wrong[0]]
            draw_images(wrong_imgs, labels=labels, guesses=wrong[1])

            plt.draw()
            plt.pause(.001)


# np.random.seed(0)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig.set_size_inches(14,7)

print('Loading data...')
dataset = load_mnist_data()

# print('Exporting digits...')
# # Export n first digits
# n = 100
# digits = dataset[0][slice(100)]
# digits_export = {}
# digits_export['digits'] = digits
# np.savez(str(n) + '_digits.npz', **digits_export)

# print('Drawing digits...')
# draw_images(dataset[0][slice(100)])

print('Creating network...')
myNet = DeepNeuralNetwork([784, 128, 64, 10], epochs=20, learning_rate=0.005)


# Load trained network from .npz file
network_path = 'network_params.npz'
if path.exists(network_path):
    print('Loading params from external file...')
    params = np.load(network_path)
    for key in params.files:
        myNet.params[key] = params[key]

    # Draw the training statistics to a diagram
    acc = params['ACC']
    ax.plot(range(0, len(acc)), acc * 100)
    # Add annotations
    for i in range(len(acc)):
        ax.annotate('{} = {}'.format(i + 1, np.round(acc[i] * 100, 1)), (i, acc[i] * 100 + 0.4), ha='center', va='center')

    plt.draw()
    plt.pause(.001)

else:
    print('No external file named \'{}\' was found.'.format(network_path))
    
print('Training network...')
myNet.train_network(dataset[0], dataset[1], dataset[2], dataset[3])