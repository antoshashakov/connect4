import tensorflow as tf #import tensorflow library

training_data = tf.constant(([[0,0],[0,1],[1,0],[1,1]])) #all possible combinations of 2 bits
target_data = tf.constant(([[0],[1],[1],[0]])) #desired outputs for each combo

model = tf.keras.models.Sequential() #model initialization

model.add(tf.keras.layers.Dense(2, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(2, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(2, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(2, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) 


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy']) #binary_accuracy metric tells us how good NN is at a given pt. in time at plotting data

model.summary() #Summary of model's architecture (layers, neurons, etc.)

history=model.fit(training_data,target_data,epochs=500,verbose=2) #NN trains with training data to produce target data; epochs is the number of "training sessions"


print(model.predict(training_data)) 
print()
print(model.predict(training_data).round()) #after training complete, we assess if NN is likely to give the correct output with some binary input


