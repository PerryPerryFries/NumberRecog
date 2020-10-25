import numpy as np
import scipy.special
import matplotlib.pyplot as plt


class neuralNetworks:

    def __init__(self, inputNodes, hiddenNodes, outputNodes,
                 learningRate):
        # Set the number of nodes in each layer
        self.outputNodes = outputNodes
        self.hiddenNodes = hiddenNodes
        self.inputNodes = inputNodes

        # Learning rate
        self.learningRate = learningRate

        # Weights
        self.wih = np.random.normal(0.0, pow(self.hiddenNodes, -0.5),
                                    (self.hiddenNodes, self.inputNodes))
        self.who = np.random.normal(0.0, pow(self.outputNodes, -0.5),
                                    (self.outputNodes, self.hiddenNodes))

        # Activation Function Sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):

        # Convert input and target lists to 2D array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layers
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate signals emerging from hidden layers
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into output layers
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate signals emerging from output layers
        final_outputs = self.activation_function(final_inputs)

        # Error = target - actual at output Layer
        output_errors = targets - final_outputs

        # Error at hidden layer = W(hidden_out).T - errors_output using back propagation
        hidden_errors = np.dot(self.who.T, output_errors)

        # Updating weights for the links between the hidden and output layers
        self.who += self.learningRate * np.dot((output_errors * final_outputs *
                                                (1.0 - final_outputs)), np.transpose(hidden_outputs))

        # Updating weights for the links between the input and hidden layers
        self.wih += self.learningRate * np.dot((hidden_errors * hidden_outputs *
                                                (1.0 - hidden_outputs)), np.transpose(inputs))

        pass

    def query(self, inputs_list):

        # Convert input list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # Calculate signals into hidden layers
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate signals emerging from hidden layers
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into output layers
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate signals emerging from output layers
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    pass
