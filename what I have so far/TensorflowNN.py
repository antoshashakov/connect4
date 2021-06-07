import datetime
import importlib
import tensorflow as tf

begin_time = datetime.datetime.now()
print(datetime.datetime.now())
g_board = importlib.import_module('GameBoard')  # Karl, Thinh
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(126, input_dim=126, activation='relu'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

set_size = 3

for i in range(0, 2):
    training = g_board.get_training_data(set_size, model)
    target = g_board.get_target_data(g_board.tr_array, model)
    history = model.fit(training, target, verbose=2, batch_size=set_size, epochs=100)
print(datetime.datetime.now() - begin_time)


