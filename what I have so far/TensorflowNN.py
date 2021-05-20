import tensorflow as tf
import importlib


g_board = importlib.import_module('GameBoard') #Karl, Thinh
# d_prob = importlib.import_module('desired_probabilities') #Ian, Uvernes

#rough

negloglik = lambda y, p_y: -p_y.log_prob(y) #log_likelihood custom

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1000, input_dim=126, activation='sigmoid'))
model.add(tf.keras.layers.Dense(7, input_dim=1000, activation='softmax'))

model.compile(loss='negloglik', metrics='accuracy')

g = g_board.GameBoard(-1)
training_data = tf.constant(g.getBoards()[0] + g.getBoards()[1] + g.getBoards()[2])

print(training_data)

target_data = g.desired_probabilities(model)

history = model.fit(training_data, target_data, epochs=10, verbose=2)

print(model.predict(training_data))

