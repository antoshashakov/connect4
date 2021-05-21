import tensorflow as tf #import tensorflow library

training_data = tf.constant(([[0,0],[0,1],[1,0],[1,1]])) #all possible combinations of 2 bits
target_data = tf.constant(([[0],[1],[1],[0]])) #desired outputs for each combo

model = tf.keras.models.Sequential() #model initialization

model.add(tf.keras.layers.Dense(2, input_dim=2, activation='sigmoid')) 
#"hidden layer" with 2 neurons; sigmoid activation function
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) #output layer


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy']) #binary_accuracy metric tells us how good NN is at a given pt. in time at plotting data

model.summary() #Summary of model's architecture (layers, neurons, etc.)

print("trainable variables:", model.trainable_variables)

history=model.fit(training_data,target_data,epochs=5000, verbose=0) #NN trains with training data to produce target data; epochs is the number of "training sessions"


print(model.predict(training_data))
print()
print(model.predict(training_data).round()) #after training complete, we assess if NN is likely to give the correct output with some binary input

# Printing all trainable variables (weights and biases). With weight matrices, each column corresponds to a nod
# on the next layer, and each row correspond to a nod on a previous layer
print("trainable variables:", model.trainable_variables)


print("training data: [0,0]")
training_data = tf.constant(([[0,0]])) #additional training on a single input
target_data = tf.constant(([[0]])) #desired output
history=model.fit(training_data,target_data,epochs=50, verbose=0)
print("trainable variables:", model.trainable_variables)

print("training data: [0,1]")
training_data = tf.constant(([[0,1]])) #additional training on a single input
target_data = tf.constant(([[1]])) #desired outputs for each combo
history=model.fit(training_data,target_data,epochs=50, verbose=0)
print("trainable variables:", model.trainable_variables)

print("training data: [1,0]")
training_data = tf.constant(([[1,0]])) #additional training on a single input
target_data = tf.constant(([[1]])) #desired outputs for each combo
history=model.fit(training_data,target_data,epochs=50, verbose=0)
print("trainable variables:", model.trainable_variables)

print("training data: [1,1]")
training_data = tf.constant(([[1,1]])) #additional training on a single input
target_data = tf.constant(([[0]])) #desired outputs for each combo
history=model.fit(training_data,target_data,epochs=50, verbose=0)
print("trainable variables:", model.trainable_variables)
