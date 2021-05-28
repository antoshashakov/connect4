import importlib
import tensorflow as tf

g_board = importlib.import_module('GameBoard')  # Karl, Thinh

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(126, input_dim=126, activation='sigmoid'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')


def init_data():
    training_data = []
    target_data = []
    for i in range(0, 5):  # number of boards to be generated in our database
        g = g_board.GameBoard(-1)
        print(g)
        train = [g.getBoards()[0] + g.getBoards()[1] + g.getBoards()[2]]
        training_data = training_data + train
        des = g.desired_probabilities(model)
        print("desired_probabilities")
        print(des)
        target_data = target_data + [des]
        print("predicted probs")
        print(model.predict(train))
    return tf.constant(training_data), tf.constant(target_data)


for i in range(0, 3):
    history = model.fit(init_data()[0], init_data()[1], verbose=2)
    print(model.predict(init_data[0]))




