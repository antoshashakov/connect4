import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

# Python program to print
# colored text and background
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

class Network:
    # Constructor with parameter sizes is a list of number of nodes for each layer,
    # parameter weights is a list of numpy arrays for weight matrices,
    # parameter biases is a list of numpy arrays for bias matrices,
    def __init__(self, sizes, weights = None, biases = None):
        # Set sizes variable according to the given list in parameter sizes
        self.sizes = sizes
        # If given parameter weights and biases, then set weights and biases variables accordingly
        # Otherwise, fill weights and biases variables with random numbers in correct numpy arrays
        if biases and weights:
            self.biases = biases
            self.weights = weights
        else:
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # Train model with training_data is a list of tuples
    # (x, y) representing the training inputs and the desired outputs,
    # epochs is the number of epochs, and lr is the learning rate
    def train_model(self, training_data, epochs, lr):
        # Create a 'losses' list to store loss value after each epoch,
        # which is then returned by the function in order to plot loss function
        losses = []
        # Iterate through all the epochs
        for j in range(epochs):
            # Initialize variable for loss value of current epoch
            loss = 0
            # Lists that store the derivatives of full cost function with respect to biases and weights respectively
            delta_b = [np.zeros(b.shape) for b in self.biases]
            delta_w = [np.zeros(w.shape) for w in self.weights]
            # Iterate through each training example
            for x, y in training_data:
                # Get the derivative of the example cost function with respect to bias, weight,
                # and activation respectively after applying back propagation to the training example
                d_b, d_w, d_a = self.back_propagation(x, y)
                # Update loss value
                loss += (self.get_predicted_output(x)-y)**2
                # Add the derivatives of cost function for the current training example
                # to the list of the derivatives of full cost function,
                # which are then later divided by the number of training examples when updating weights and biases
                # (because derivative of full cost function is equal to the average of all training examples)
                delta_b = [db+d_b for db, d_b in zip(delta_b, d_b)]
                delta_w = [dw+d_w for dw, d_w in zip(delta_w, d_w)]
            # Add the loss value of current epoch to the losses list
            losses.append(loss)
            # Update weights and biases of model
            self.weights = [w - (lr/len(training_data)) * dw
                            for w, dw in zip(self.weights, delta_w)]
            self.biases = [b - (lr/len(training_data)) * db
                           for b, db in zip(self.biases, delta_b)]
        return losses

    # Train the model with added clone node and
    # training_data is a list of tuples (x, y) representing the training inputs and the desired outputs,
    # epochs is the number of epochs, lr is the learning rate,
    # clone_list is a list of tuples where each tuple is of form (clone_layer, original_pos, new_pos) where
    # clone_layer is the layer of cloning, and original_pos and new_pos are the original and new nodes
    def train_model_with_clone(self, training_data, epochs, lr, clone_list):
        # Create a 'losses' list to store loss value after each epoch,
        # which is then returned by the function in order to plot loss function
        losses = []
        # Iterate through all the epochs
        for j in range(epochs):
            loss = 0
            # Derivative of full cost function
            delta_b = [np.zeros(b.shape) for b in self.biases]
            delta_w = [np.zeros(w.shape) for w in self.weights]

            # Iterate through each training example
            for x, y in training_data:
                # Get the derivative of the example cost function with respect to bias, weight,
                # and activation respectively after applying back propagation to the training example
                d_b, d_w, d_a = self.back_propagation(x, y)
                # Update loss value
                loss += (self.get_predicted_output(x) - y) ** 2

                # Loop through each pair of copied nodes to update the nodes accordingly
                for clone_layer, original_pos, new_pos in clone_list:
                    # Set partial derivatives with respect to weights and bias of the original node to be zero
                    # when its auxiliary partial derivative is less than zero
                    if d_a[clone_layer - 2][original_pos - 1] < 0:
                        # Set bias of the original node to zero
                        d_b[clone_layer - 2][original_pos - 1] = 0
                        # Set weights of the original node in input weight matrix to zero
                        for c in range(d_w[clone_layer - 2].shape[1]):
                            d_w[clone_layer - 2][original_pos - 1][c] = 0
                        # Set weights of the original node in output weight matrix to zero
                        for r in range(d_w[clone_layer - 1].shape[0]):
                            d_w[clone_layer - 1][r][original_pos - 1] = 0
                    # Set partial derivatives with respect to weights and bias of the new node to be zero
                    # when its auxiliary partial derivative is more than zero
                    if d_a[clone_layer - 2][new_pos-1] > 0:
                        # Set bias of the new node to zero
                        d_b[clone_layer - 2][new_pos-1] = 0
                        # Set weights of the new node in input weight matrix to zero
                        for c in range(d_w[clone_layer - 2].shape[1]):
                            d_w[clone_layer - 2][new_pos-1][c] = 0
                        # Set weights of the new node in output weight matrix to zero
                        for r in range(d_w[clone_layer - 1].shape[0]):
                            d_w[clone_layer - 1][r][new_pos-1] = 0
                # Add the derivatives of cost function for the current training example
                # to the list of the derivatives of full cost function,
                # which are then later divided by the number of training examples when updating weights and biases
                # (because derivative of full cost function is equal to the average of all training examples)
                delta_b = [db + d_b for db, d_b in zip(delta_b, d_b)]
                delta_w = [dw + d_w for dw, d_w in zip(delta_w, d_w)]
            # Add the loss value of current epoch to the losses list
            losses.append(loss)
            # Update weights and biases of model
            self.weights = [w - (lr / len(training_data)) * dw
                            for w, dw in zip(self.weights, delta_w)]
            self.biases = [b - (lr / len(training_data)) * db
                           for b, db in zip(self.biases, delta_b)]
        return losses

    def back_propagation(self, x, y):
        # Lists that store derivatives of the cost function with respect to
        # bias, weight, activation respectively of a training example
        d_b = [np.zeros(b.shape) for b in self.biases]
        d_w = [np.zeros(w.shape) for w in self.weights]
        d_a = [np.zeros(b.shape) for b in self.biases[:-1]]

        # Feed forward
        # Modify input shape
        activation = np.array(x).transpose()
        # List to store all the activations
        activations = [activation]
        # List to store all z values
        z_values = []
        # Iterate through each bias and weight matrix to calculate z value and
        # activation after applying sigmoid
        for b, w in zip(self.biases, self.weights):
            # Calculate z value
            z_value = np.dot(w, activation)+b
            # Add z value to the list of z values
            z_values.append(z_value)
            # Apply sigmoid function to z value to get activation
            activation = sigmoid(z_value)
            # Add activation to the list of activations
            activations.append(activation)

        # Backpropagation
        # Calculate derivative of cost function with respect to bias, weight and activation respectively
        # for the output layer
        common = 2*(activations[-1] - y) * sigmoid_prime(z_values[-1])
        d_b[-1] = common
        d_w[-1] = np.dot(common, np.array(activations[-2]).transpose())
        d_a[-1] = np.dot(np.array(self.weights[-1]).transpose(), common)

        # Calculate derivative of cost function with respect to bias, weight and activation respectively
        # for the hidden layers
        for i in range(2, len(self.sizes)):
            z_value = z_values[-i]
            common = np.dot(np.array(self.weights[-i+1]).transpose(), common) * sigmoid_prime(z_value)
            d_b[-i] = common
            d_w[-i] = np.dot(common, np.array(activations[-i-1]).transpose())
            if i > 2:
                d_a[-i+1] = np.dot(np.array(self.weights[-i+1]).transpose(), d_b[-i+1])
        return d_b, d_w, d_a

    # Make a new clone node from a given node at a layer, and return position of new node
    def make_clone_node(self, layer, node_pos):
        # Update size of the model
        self.sizes[layer-1] += 1
        # Update the input weight matrix of the changed layer
        # Add a new row to the bottom of the matrix
        self.weights[layer-2].resize((1+ self.weights[layer-2].shape[0],  self.weights[layer-2].shape[1]))
        # Fill the bottom row with values same as the values of the original node's row
        self.weights[layer - 2][-1] = [w for w in self.weights[layer - 2][node_pos-1]]

        # Update the output weight matrix of the changed layer
        # Create a temp matrix with a column of zeroes, and same number of rows as the original weight matrix
        temp = np.zeros((self.weights[layer-1].shape[0], 1), dtype= self.weights[layer-1].dtype)
        # Concatenate the temp matrix with the original weight matrix to essentially add a new column in the right
        # of the original weight matrix
        self.weights[layer-1] = np.concatenate((self.weights[layer-1],temp), axis=1)
        # Update the values of the original node's column, which means divide the numbers by 2
        self.weights[layer-1][:, [node_pos - 1]] = self.weights[layer-1][:, [node_pos - 1]]/2
        # Fill the right-most column values same as the values of the original node's column
        self.weights[layer-1][:, [-1]] = self.weights[layer-1][:, [node_pos - 1]]

        # Update bias matrix
        # Add a new row to the bottom of the matrix
        self.biases[layer-2].resize((1+ self.biases[layer-2].shape[0],  self.biases[layer-2].shape[1]))
        # Fill the bottom row with value same as the value of the original node's row
        self.biases[layer - 2][-1] = self.biases[layer - 2][node_pos-1]
        # Return new node's position
        return self.sizes[layer-1]

    # Return a node in a layer that is required the largest difference in training data's auxiliary partial derivative
    def get_most_change(self, training_data):
        # A list that stores auxiliary partial derivative for all training examples
        temp_a = []
        # A  list that stores the difference in auxiliary partial derivatives of all nodes
        nodes_da = []
        # Initialize returned layer
        layer = 0
        # Fill the list with auxiliary partial derivatives for all training examples
        for x, y in training_data:
            d_b, d_w, d_a = self.back_propagation(x, y)
            temp_a.append(d_a)
        # Calculate the difference
        for i in range(len(temp_a[0])):
            for r in range(temp_a[0][i].shape[0]):
                nodes_da.append(find_max_dif([temp_a[n][i][r][0] for n in range(len(temp_a))]))

        # Find layer and node position
        node_pos = nodes_da.index(max(nodes_da)) + 1
        for i, num_nodes in enumerate(self.sizes[1:-1]):
            if node_pos - num_nodes > 0:
                node_pos -= num_nodes
            layer = i +2
        return layer, node_pos

    # Return sizes, weights and biases
    def get_info(self):
        return self.sizes, self.weights, self.biases

    # Return model's prediction for the given input
    def get_predicted_output(self, input_data):
        # Modify input shape
        input_data = np.array(input_data).transpose()
        # Feed forward input data to get model output
        for b, w in zip(self.biases, self.weights):
            input_data = sigmoid(np.dot(w, input_data) + b)
        return np.squeeze(input_data)

    # Return a list of activation arrays with the given input data
    def get_activation_values(self, input_data):
        # Modify input shape
        activation = np.array(input_data).transpose()
        # List to store all the activations
        activations = [activation]
        # Iterate through each bias and weight matrix to calculate z value and
        # activation after applying sigmoid
        for b, w in zip(self.biases, self.weights):
            # Calculate activation
            activation = sigmoid(np.dot(w, activation) + b)
            # Add activation to the list of activations
            activations.append(activation)
        return activations

    # Return a list of auxiliary partial derivatives of the given input data
    def get_da(self, input_data):
        # A list to stores all auxiliary partial derivatives of the given input data
        da = []
        # Iterate through each training example
        for x, y in input_data:
            # Get the derivative of the example cost function with respect to bias, weight,
            # and activation respectively after applying back propagation to the training example
            d_b, d_w, d_a = self.back_propagation(x, y)
            da.append(d_a)
        return da

    # Return a list of partial derivatives of cost function w.r.t biases of the given input data
    def get_db(self, input_data):
        # A list to stores all partial derivatives of cost function w.r.t biases of the given input data
        db = []
        # Iterate through each training example
        for x, y in input_data:
            # Get the derivative of the example cost function with respect to bias, weight,
            # and activation respectively after applying back propagation to the training example
            d_b, d_w, d_a = self.back_propagation(x, y)
            db.append(d_b)
        return db

    # Return a list of partial derivatives of cost function w.r.t weights of the given input data
    def get_dw(self, input_data):
        # A list to stores all partial derivatives of cost function w.r.t weights of the given input data
        dw = []
        # Iterate through each training example
        for x, y in input_data:
            # Get the derivative of the example cost function with respect to bias, weight,
            # and activation respectively after applying back propagation to the training example
            d_b, d_w, d_a = self.back_propagation(x, y)
            dw.append(d_w)
        return dw

#### Helper functions

# Sigmoid function
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
# Find max difference given a list of numbers
def find_max_dif(nums):
    return max(nums) - min(nums)

# Plot loss function with respect to epochs
def plot_loss_function(loss_list):
    plt.plot(loss_list)
    plt.xlabel("EPOCHS")
    plt.ylabel("Loss value")
    plt.show()

training_input = [([[0,0]], 0), ([[0,1]], 1), ([[1, 0]], 1), ([[1, 1]], 0)]
# model = Network([2, 2, 1])
model = Network([2, 2, 1], [np.array([[ 4, -6 ],
       [ -4,  -5]]), np.array([[4, -4]])],[np.array([[-3],
       [ 0.5]]), np.array([[0]])])
prRed("Initial model")
print("Model's prediction for [0,0]:")
prGreen("{:.4f}".format(model.get_predicted_output([[0,0]])))
print("Model's prediction for [0,1]:")
prGreen("{:.4f}".format(model.get_predicted_output([[0,1]])))
print("Model's prediction for [1,0]:")
prGreen("{:.4f}".format(model.get_predicted_output([[1,0]])))
print("Model's prediction for [1,1]:")
prGreen("{:.4f}".format(model.get_predicted_output([[1,1]])))
# print("Weights")
# pprint(model.weights)
# print("Biases")
# pprint(model.biases)

# losses1 = model.train_model(training_input,50000,1)


# prRed("\nAfter training")
# print("Model's prediction for [0,0]:")
# prGreen("{:.4f}".format(model.get_predicted_output([[0,0]])))
# print("Model's prediction for [0,1]:")
# prGreen("{:.4f}".format(model.get_predicted_output([[0,1]])))
# print("Model's prediction for [1,0]:")
# prGreen("{:.4f}".format(model.get_predicted_output([[1,0]])))
# print("Model's prediction for [1,1]:")
# prGreen("{:.4f}".format(model.get_predicted_output([[1,1]])))
# plot_loss_function(losses1)
# print("Weights")
# pprint(model.weights)
# print("Biases")
# pprint(model.biases)



# node_layer, node_position = model.get_most_change(training_input)
clone = model.make_clone_node(2,2)
# # clone1 = model.make_clone_node(node_layer,node_position)
# # clone2 = model.make_clone_node(2,1)
# prRed("\nAfter make clone")
# print("Model's prediction for [0,0]:")
# prGreen("{:.4f}".format(model.get_predicted_output([[0,0]])))
# print("Model's prediction for [0,1]:")
# prGreen("{:.4f}".format(model.get_predicted_output([[0,1]])))
# print("Model's prediction for [1,0]:")
# prGreen("{:.4f}".format(model.get_predicted_output([[1,0]])))
# print("Model's prediction for [1,1]:")
# prGreen("{:.4f}".format(model.get_predicted_output([[1,1]])))
# print("Weights")
# pprint(model.weights)
# print("Biases")
# pprint(model.biases)
# print("Sizes")
# print(model.sizes)
# print("\n Auxiliary partial derivatives")
# pprint(model.get_da(training_input))
# print("\n Partial derivatives w.r.t. biases")
# pprint(model.get_db(training_input))
# print("\n Partial derivatives w.r.t. weights")
# pprint(model.get_dw(training_input))


losses3 = model.train_model_with_clone(training_input, 400, 1, [(2,2,3)])
prRed("\nAfter training")
print("Model's prediction for [0,0]:")
prGreen("{:.4f}".format(model.get_predicted_output([[0,0]])))
print("Model's prediction for [0,1]:")
prGreen("{:.4f}".format(model.get_predicted_output([[0,1]])))
print("Model's prediction for [1,0]:")
prGreen("{:.4f}".format(model.get_predicted_output([[1,0]])))
print("Model's prediction for [1,1]:")
prGreen("{:.4f}".format(model.get_predicted_output([[1,1]])))
print("\nWeights")
pprint(model.weights)
print("\nBiases")
pprint(model.biases)
print("\n Auxiliary partial derivatives")
pprint(model.get_da(training_input))
print("\n Partial derivatives w.r.t. biases")
pprint(model.get_db(training_input))
print("\n Partial derivatives w.r.t. weights")
pprint(model.get_dw(training_input))
plot_loss_function(losses3)

# losses4 = model.train_model(training_input, 1, 1)
# print("\nAfter training")
# print("Model's prediction for [0,0]:")
# print("{:.4f}".format(model.get_predicted_output([[0,0]])))
# print("Model's prediction for [0,1]:")
# print("{:.4f}".format(model.get_predicted_output([[0,1]])))
# print("Model's prediction for [1,0]:")
# print("{:.4f}".format(model.get_predicted_output([[1,0]])))
# print("Model's prediction for [1,1]:")
# print("{:.4f}".format(model.get_predicted_output([[1,1]])))
# print("Weights")
# pprint(model.weights)
# print("Biases")
# pprint(model.biases)
# print("\n Auxiliary partial derivatives")
# pprint(model.get_da(training_input))
# print("\n Partial derivatives w.r.t. biases")
# pprint(model.get_db(training_input))
# print("\n Partial derivatives w.r.t. weights")
# pprint(model.get_dw(training_input))
# plot_loss_function(losses4)
