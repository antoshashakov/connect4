import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import GameBoard as gameBoard
import gc

# load the old data from the shallow network for transfer learning
old_model = tf.keras.models.load_model('my_model')
old_weights = old_model.layers[0].get_weights()[0]
old_biases = old_model.layers[0].get_weights()[1]
old_weights_2 = old_model.layers[1].get_weights()[0]
old_biases_2 = old_model.layers[1].get_weights()[1]

# create a new network with another hidden layer and extra nodes on the first to serve as "pass-through" nodes
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1126, input_dim=126, activation='sigmoid'))
model.add(tf.keras.layers.Dense(2000, activation='relu'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

# create the new weight lists
weights = []
for i in range(126):
    # copy the old weights, after which point place 100 at index i and 0 elsewhere
    new_weights = [(old_weights[i][j] if j < 1000 else (100 if j - 1000 == i else 0)) for j in range(1126)]
    weights.append(new_weights)
# create the new biases list
biases = [(old_biases[i] if i < 1000 else -50) for i in range(1126)]

# join these together
layer_data = [np.array(weights), np.array(biases)]

# set the weights and biases for the first layer of the new model
model.layers[0].set_weights(layer_data)
model.layers[0].trainable = False

# set the outgoing weights from the first hidden layer
weights_2 = [[(1 if j == 2 * i else (-1 if j == (2 * i) + 1 else 0)) for j in range(2000)] for i in range(1126)]
biases_2 = [0 for _ in range(2000)]
layer_data_2 = [np.array(weights_2), np.array(biases_2)]
model.layers[1].set_weights(layer_data_2)

# set the outgoing weights from the second hidden layer
weights_3 = [old_weights_2[i // 2] for i in range(2000)]
biases_3 = old_biases_2
layer_data_3 = [np.array(weights_3), np.array(biases_3)]
model.layers[2].set_weights(layer_data_3)

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam')

# test that the behaviours of the old and new networks align
# b = gameBoard.get_samples(1, model)
# b_data = gameBoard.get_training_data(b)
# v1 = old_model(b_data)[0]
# v2 = model(b_data)[0]
# print(gameBoard.vector_string(v1))
# print(gameBoard.vector_string(v2))

# flag for whether in testing mode
testing = True

# parameters for training
set_size = 500
training_cycles = 100
b_size = 100
eps = 100
trials = 10

# alternate parameters for testing mode
if testing:
    set_size = 500
    training_cycles = 100
    b_size = 50
    eps = 50
    trials = 10

# keep track of average euclidean distances
avg_dist = []

# amount of detail the evaluation method should give; levels of depth explained above the function
evaluation_depth = 0

# set the number of trials for the GameBoard class
gameBoard.set_trials(trials)

# the set that will be used to evaluate the euclidean distance
eval_set = gameBoard.get_samples(10, model)
# display the evaluation set
print("Set of boards used for evaluation:")
for b in eval_set:
    print(b)

# avg_dist.append(gameBoard.evaluate(eval_set, model, evaluation_depth))  # pre-training evaluation

begin_time = datetime.datetime.now()

# training loop
for i in range(training_cycles):
    # loop information
    print("Outer loop:", i + 1, "/", training_cycles)
    # get a new set of boards for testing
    boards = gameBoard.get_samples(set_size, model)
    # get the training data for the new boards and the evaluation set
    training = gameBoard.get_training_data(np.append(boards, eval_set))
    # get the target data for the new boards and the evaluation set
    target = gameBoard.get_target_data(np.append(boards, eval_set), model)
    # evaluate the set and store the average distance
    distance = gameBoard.evaluate(model(np.stack(training[set_size:])), target[set_size:], eval_set, evaluation_depth)
    avg_dist.append(distance)
    # train the model
    model.fit(training[:set_size], target[:set_size], verbose=0, batch_size=b_size, epochs=eps)
    gc.collect()

print("Final training time: " + str(datetime.datetime.now() - begin_time))
print("Current parameters:")
print("\tset_size = " + str(set_size))
print("\ttraining_cycles  = " + str(training_cycles))
print("\tb_size = " + str(b_size))
print("\ttrials = " + str(trials))

model.save('my_model_2')

x = np.array([])
for j in range(training_cycles):
    x = np.append(x, [j])

print(avg_dist)
plt.plot(x, avg_dist)
plt.xlabel('training cycles')
plt.ylabel('average euclidean distance')
plt.title('performance')
plt.show()
