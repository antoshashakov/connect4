import random as rand
import math

training_size = 50
batch_size = 5
learn_rate = 0.1
entry_size = 2
weights = [0 for i in range(entry_size + 1)]
training_data = [[0 for i in range(entry_size)] for j in range(training_size)]
training_targets = [0 for i in range(training_size)]
training_results = [0 for i in range(training_size)]
testing_data = [[0 for i in range(entry_size)] for j in range(10)]


def go(epochs):
    generate_testing_data()
    generate_training_data()
    generate_weights()
    train(epochs)
    test()


def train(epochs):
    for e in range(epochs):
        for b in range(0, training_size, batch_size):
            derivs = [0 for i in range(entry_size + 1)]
            for i in range(b, b + batch_size):
                training_results[i] = sigma(weighted_sum(training_data[i]))
                for j in range(entry_size):
                    derivs[j] += 2.0 * (training_results[i] - training_targets[i]) * sigma_prime(weighted_sum(training_data[i])) * sigma(training_data[i][j])
                derivs[entry_size] += 2.0 * (training_results[i] - training_targets[i]) * sigma_prime(weighted_sum(training_data[i]))
            for i in range(entry_size):
                weights[i] -= learn_rate * derivs[i] / batch_size


def test():
    for sequence in testing_data:
        print(">", end="")
        for entry in sequence:
            print("{:.0f}".format(entry), end="")
        print(":", "{:.0f}".format(sigma(weighted_sum(sequence))))


def generate_training_data():
    for i in range(training_size):
        for j in range(entry_size):
            training_data[i][j] = rand.randint(0, 1)
        training_targets[i] = 0.0
        if training_data[i][0] == 1:
            training_targets[i] += 1.0
        if training_data[i][entry_size - 1] == 1:
            training_targets[i] -= 1.0


def generate_testing_data():
    for i in range(10):
        for j in range(entry_size):
            testing_data[i][j] = rand.randint(0, 1)


def generate_weights():
    for i in range(entry_size + 1):
        weights[i] = rand.random()


def weighted_sum(input_data):
    total = 0.0
    for i in range(entry_size):
        total += sigma(input_data[i]) * weights[i]
    total += weights[entry_size]
    return total


def sigma(input_data):
    return (2.0 / (1.0 + math.exp(-input_data))) - 1.0


def sigma_prime(input_data):
    return 0.5 * (1.0 - math.pow(sigma(input_data), 2))


go(500)
