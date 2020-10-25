import numpy as np
import scipy.misc
import neuralNetworkClass


file_folder = "/home/ankur/Desktop/ML/ML Number recognition/Number Recognition/"

# Loading training data csv file into list
training_data_file = open(file_folder + "mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Loading testing data csv file into list
testing_data_file = open(file_folder + "mnist_test.csv", 'r')
testing_data_list = testing_data_file.readlines()
testing_data_file.close()

# Image pixel values divided into 28*28
input_nodes = 784
# Randomly chosen between input and output nodes (Can change to improve the network)
hidden_nodes = 100
# 10 as targets only vary from 0-9
output_nodes = 10

# Learning rate (Can change to improve the network)
learning_rate = 0.1

# Creating instance of a neural network
n = neuralNetworkClass.neuralNetworks(input_nodes, hidden_nodes,
                                      output_nodes, learning_rate)

# Training neural network with the training set

# epochs is the number of times the training set will repeat to improve the network
# (Can change to improve the network)
epochs = 4

for e in range(epochs):
    for record in training_data_list:
        # splitting the record with ','
        all_values = record.split(',')
        # Scaling and shifting Inputs 0.01-1
        inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
        # Creating target values (all 0.01 except the desired value 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target value
        targets[int(all_values[0])] = 0.99

        # Training for each value
        n.train(inputs, targets)

        pass
    pass

score = []

# Testing the trained network
for record in testing_data_list:
    # Splitting record with ','
    all_values = record.split(',')
    # Scaling values
    inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01

    # Correct label is at all_values[0]
    correct_label = int(all_values[0])
    # print("Correct label : ", correct_label)

    # Query the network : this will give an array with 10 values
    outputs = n.query(inputs)

    # Label predicted is the one having the largest value from the output array
    label = np.argmax(outputs)
    # print("Network's Prediction : ", label)

    # Providing Score to the Network
    if (correct_label == label):
        score.append(1)
    else:
        score.append(0)
        pass

    pass

score_array = np.asarray(score)
print("Performance : ", score_array.sum() / score_array.size)
