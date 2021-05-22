import tensorflow as tf
import importlib

g_board = importlib.import_module('GameBoard')  # Karl, Thinh

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1000, input_dim=126, activation='sigmoid'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def init_data():
    training_data = []
    target_data = []
    for i in range(0, 3):  # number of boards to be generated in our database
        g = g_board.GameBoard(-1)
        training_data = training_data + [g.getBoards()[0] + g.getBoards()[1] + g.getBoards()[2]]
        target_data = target_data + [g.desired_probabilities(model)]
    return tf.constant(training_data), tf.constant(target_data)


history = model.fit(init_data()[0], init_data()[1], epochs=1, verbose=2)



