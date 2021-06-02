import datetime
import importlib
import tensorflow as tf

begin_time = datetime.datetime.now()
print(datetime.datetime.now())
g_board = importlib.import_module('GameBoard')  # Karl, Thinh
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(126, input_dim=126, activation='sigmoid'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

set_size = 10

def init_data():
    training_data = []
    target_data = []
    for i in range(0, set_size):  # number of boards to be generated in our database
        g = g_board.GameBoard(-1)
        print(g)
        train = [g.getBoards()[0] + g.getBoards()[1] + g.getBoards()[2]]
        training_data = training_data + train
        des = g.desired_probabilities(model)
        print("desired_probabilities")
        print(des)
        target_data = target_data + [des]
        print("predicted_probabilities")
        print(model.predict(train))
    return tf.constant(training_data), tf.constant(target_data)


for i in range(0, 1):
    train = init_data()[0]
    history = model.fit(train, init_data()[1], verbose=2, batch_size=set_size, epochs=100)
    print(train)
    print(model.predict(train))
print(datetime.datetime.now() - begin_time)


