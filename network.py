# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes: int):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        # list[list[List[float]]] , second layer has 30 elements (neurons), third has 10
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # y is num of rows, x is num of cols
        # weights is a list layers (1 & 2), each holding a list of # neurons in their respective layer, 
        # each neuron holds a list of weights to all neurons in preious layer
        # ex: vector[30, 10] * vector[784, 30] 
        # list[list[list[list[float]]]]
        # if you wanted to access the weight from the w connecting the 5 neuron in the first layer
        # the third in the second I believe it would be indexed like so: weights[0][3][6]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data: tuple[list, list[float]], epochs: int, mini_batch_size: int, eta: int,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)


        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            # shuffle the training data to make mini-batch size more random
            random.shuffle(training_data)
            # create list[list[data]], with len of each 
            # element being mini_batch_size (10 in this example)
            # bunch of lists of size 10, each containing 10
            # (image, accuracy) tuples
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            # updates the weight and biases of each mini-batch
            # eta is the learning rate
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch: list[tuple[list, list[float]]], eta: int):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # creates a clone of bias and wigthts matrices except populated with zero's
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # x is the training inputs, y is the desired output
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # return the updated bias and weight layer
            # in numpy adding two matrices add the corresponding numbers in each spot together
            # This is just transfering the updated b and w arrays into this method
            # adds delta of w & b to 0 & 0 respectively
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # this is gradient descent
        # taking the old w and subtracting the avg BP4 over every mini batch input, x, for each specified layer
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y) -> tuple[list, list]:
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        This function returns the updated weight and bias layer."""
        # initializing memory for the updated weights and bias's
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        # list of 784 neurons
        activation = x
        # list of a list holding 784 neurons
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        # is the weight result before the activation function (sigmoid) is applied
        # in the first iteration,
            # b = the bias's for each neuron in the second layer
            # w = the list of weights connecting all 784 neurons from layer 1 to each neuron in layer 2
        for b, w in zip(self.biases, self.weights):
            # dot product vector multiplication, z is now expected vector
            # I would expect this to be (w1*a1 + w2*a1 +  ... wn*a1) + b
            # updates the activatins from layer 30 and layer 10
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        # this is where we find the error of the last layer, L
        # sigmoid prime is the derivative of the sigmoid function, dC/da * dsigma/dz
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) 
        # updating new weights and bias's
        nabla_b[-1] = delta # b = the errors of each layer
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # BP4
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        # starts at 2 bc we already did the last layer above (nabla_b[-1], nabla_w[-1])
        # This is where we use the last layer L, to find the error in every layer prior to it
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # BP2, transpose reverse the weigth matrix jxk -> kxj
            # updating every bias and weight layer
            nabla_b[-l] = delta # BP3
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) # BP4
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
        Remember the derivative of the cost function is just ouptut - expected"""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def main():
    # Import mnist_loader module
    import mnist_loader

    # Load the data using mnist_loader.load_data_wrapper()
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # Import network module
    import network

    # Create a Network instance
    net = network.Network([784, 30, 10])

    # Call the SGD (Stochastic Gradient Descent) method on the network instance
    net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)


main()