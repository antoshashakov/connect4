import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import GameBoard as gameBoard

# parameters for network architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1000, input_dim=126, activation='sigmoid'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

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
    training_cycles = 50
    b_size = 50
    eps = 100
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
    # distance = gameBoard.evaluate(model(np.stack(training[set_size:])), target[set_size:], eval_set, evaluation_depth)
    # avg_dist.append(distance)
    # train the model
    history = model.fit(training[:set_size], target[:set_size], verbose=0, batch_size=b_size, epochs=eps)

print("Final training time: " + str(datetime.datetime.now() - begin_time))
print("Current parameters:")
print("\tset_size = " + str(set_size))
print("\ttraining_cycles  = " + str(training_cycles))
print("\tb_size = " + str(b_size))
print("\ttrials = " + str(trials))

model.save('my_model')

quit()

x = np.array([])
for j in range(training_cycles):
    x = np.append(x, [j])

print(avg_dist)
plt.plot(x, avg_dist)
plt.xlabel('training cycles')
plt.ylabel('average euclidean distance')
plt.title('performance')
plt.show()
