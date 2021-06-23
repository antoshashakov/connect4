import datetime
import importlib
import tensorflow as tf

begin_time = datetime.datetime.now()
print(datetime.datetime.now())
g_board = importlib.import_module('GameBoard')  # Karl, Thinh
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1000, input_dim=126, activation='sigmoid'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

testing = True

set_size = 1000
training_cycles = 10
b_size = 100

if testing:
    set_size = 3
    training_cycles = 5
    b_size = 3

eval_set = g_board.get_samples(10)
g_board.evaluate(eval_set, model, True) #pre-training evaluation


for i in range(training_cycles):
    boards = g_board.get_samples(set_size)
    training = g_board.get_training_data(boards)
    target = g_board.get_target_data(boards, model)
    history = model.fit(training, target, verbose=2, batch_size=b_size, epochs=100)
    g_board.evaluate(eval_set, model, True)
print(datetime.datetime.now() - begin_time)

x = np.array([])
for j in range(training_cycles + 1):
    x = np.append(x, [j])

print(g_board.avg_dist)
plt.plot(x, g_board.avg_dist)
plt.xlabel('training cycles')
plt.ylabel('average euclidean distance')
plt.title('performance')
plt.show()
