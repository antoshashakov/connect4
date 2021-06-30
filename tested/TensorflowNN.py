import datetime
import importlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import GameBoard as gb
import random as rand

begin_time = datetime.datetime.now()
print(datetime.datetime.now())

# parameters for network architecture
g_board = importlib.import_module('GameBoard')  # Karl, Thinh
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1000, input_dim=126, activation='sigmoid'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

testing = True

set_size = 500
training_cycles = 100
b_size = 100
eps = 100

if testing:
    set_size = 3
    training_cycles = 5
    b_size = 3
    eps = 3

avg_dist = []

# amount of detail the evaluation method should give; levels of depth explained above the function
evaluation_depth = 1

eval_set = gb.get_samples(10)
avg_dist.append(g_board.evaluate(eval_set, model, evaluation_depth))  # pre-training evaluation

# training loop
for i in range(training_cycles):
    boards = g_board.get_samples(set_size)
    training = g_board.get_training_data(boards)
    target = g_board.get_target_data(boards, model)
    history = model.fit(training, target, verbose=2, batch_size=b_size, epochs=eps)
    avg_dist.append(g_board.evaluate(eval_set, model, evaluation_depth))

print("Final training time: " + str(datetime.datetime.now() - begin_time))
print("Current parameters:")
print("set_size = " + str(set_size))
print("training_cycles  = " + str(training_cycles))
print("b_size = " + str(b_size))
print("trials = 10" + "\n")  # change this manually for now

x = np.array([])
for j in range(training_cycles + 1):
    x = np.append(x, [j])

print(avg_dist)
plt.plot(x, avg_dist)
plt.xlabel('training cycles')
plt.ylabel('average euclidean distance')
plt.title('performance')
plt.show()
