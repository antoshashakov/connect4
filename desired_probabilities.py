import math
import numpy as np
import random


# VERY EARLY version of the desired probabilities code; there are likely many ways in which this can be optimized,
# and I haven't yet had the chance to test it

def soft_max(arr1):
    # total to later divide each value by
    total = 0
    # list of probabilities to later be made into an array
    probs = []
    # add exponential values to the probabilities, and determine the total
    for i in arr1:
        probs.append(math.exp(i))
        total += math.exp(i)
    # divide each value by the total
    for j in range(len(probs)):
        probs[j] = probs[j] / total
    return np.array(probs)


# we need to be given the two boards and the model
def desired_probabilities(board1, board2, model, num_moves):
    # number of test games we will play
    trials = 100
    # the list storing the win/loss/draw stats
    stats = [0, 0, 0]
    # play the given number of games
    for i in range(trials):
        # track the current number of moves to know when the game is a draw
        curr_moves = num_moves
        # play until the game is finished
        while True:
            # get the probability distribution from the model
            prob_distribution = soft_max(model.predict(board1, board2))
            # pick a column at random using the helper function
            column = pick_probability(prob_distribution)
            # make a move
            board1, board2, result = make_move(board1, board2, column)
            curr_moves += 1
            # check if the game is over
            if result == 1:
                stats[0] += 1
                break
            if result == -1:
                stats[1] += 1
                break
            if result == 0 and curr_moves >= 42:
                stats[2] += 1
                break
            # if the game is still going, we now play as the other player (we swap the order of several things)
            # get the probability distribution from the model
            prob_distribution = soft_max(model.predict(board2, board1))
            # pick a column at random using the helper function
            column = pick_probability(prob_distribution)
            # make a move
            board2, board1, result = make_move(board2, board1, column)
            curr_moves += 1
            # check if the game is over
            if result == 1:
                stats[1] += 1
                break
            if result == -1:
                stats[0] += 1
                break
            if result == 0 and curr_moves >= 42:
                stats[2] += 1
                break
    stats_arr = np.array(stats)
    return soft_max(stats_arr)


# assumes that arr1 represents percentage chances
def pick_probability(arr1):
    # generate a random number
    r = random.random()
    # total to keep track of the range we will check
    total = 0
    for i in range(len(arr1)):
        total += arr1[i]
        if r <= total:
            return i + 1
    return len(arr1) + 1
