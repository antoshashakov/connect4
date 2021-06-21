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

set_size = 10

eval_set = g_board.get_samples(10)
g_board.evaluate(eval_set, model, True)


for i in range(10):
    boards = g_board.get_samples(set_size)
    training = g_board.get_training_data(boards)
    target = g_board.get_target_data(boards, model)
    history = model.fit(training, target, verbose=0, batch_size=set_size, epochs=100)
    g_board.evaluate(eval_set, model, True)
print(datetime.datetime.now() - begin_time)
