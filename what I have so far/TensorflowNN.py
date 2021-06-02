import importlib
import tensorflow as tf

g_board = importlib.import_module('GameBoard')  # Karl, Thinh

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(126, input_dim=126, activation='sigmoid'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

set_size = 2


def get_training_data(g):
    training_data = []
    print(g)
    train = [g.getBoards()[0] + g.getBoards()[1] + g.getBoards()[2]]
    training_data = training_data + train
    print("predicted_probabilities")
    print(model.predict(train))
    return tf.constant(training_data)


def get_target_data(g):
    target_data = []
    des = g.desired_probabilities(model)
    print("desired_probabilities")
    print(des)
    target_data = target_data + [des]
    return tf.constant(target_data)


for i in range(0, set_size):  # number of boards to be generated in our database
    g = g_board.GameBoard(-1)
    trainer = get_training_data(g)
    target = get_target_data(g)

for i in range(0, 1):
    history = model.fit(trainer, target, verbose=2, batch_size=set_size, epochs=100)
    print(trainer)
    print(model.predict(trainer))


