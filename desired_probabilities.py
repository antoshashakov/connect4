import math
import numpy as np
import random


# TODO: update method to work with newer version of board class
# we need to be given the board, the model, and the number of moves
def desired_probabilities(base_board, model, num_moves):
    # number of test games we will play
    trials = 100
    # the list storing the win/loss/draw stats
    stats = [0, 0, 0]
    # play the given number of games
    for i in range(trials):
        # make a copy of the board that we will modify
        board = base_board.copy()
        # track the current number of moves to know when the game is a draw
        curr_moves = num_moves
        # play until the game is finished
        while True:
            # get the probability distribution from the model
            prob_distribution = soft_max(model.predict(board))
            # pick a column at random using the helper function
            column = pick_probability(prob_distribution)
            # make a move
            board1, board2, result = board.make_move(column)
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


# HELPER FUNCTIONS:

# expects an array of real numbers
def soft_max(arr1):
    # create a list and fill it with exponentiated values from the original array
    exponentials = [math.exp(i) for i in arr1]
    # sum the values in the list to divide
    total = 0
    [total := total + i for i in exponentials]
    # get percentages by dividing each number in percentages by the total so they all sum to 1
    percentages = [i / total for i in exponentials]
    # returns percentages in array form
    return np.array(percentages)


# expects arr1 to represent a set of percentage chances (it should be composed of positive real numbers with a sum of 1)
def pick_probability(arr1):
    # generate a random number from 0 to 1
    r = random.random()
    # total to keep track of the range we will check
    total = 0
    # we check if our number is in the range from 0 to the first probability, then between the first and second,
    # and so on, until we have done so for all values in the array
    for i in range(len(arr1)):
        total += arr1[i]
        if r <= total:
            return i + 1
    # if the above fails, we will choose the last result
    # we return the index of the probability array that was chosen, plus one so that we start counting from 1
    return len(arr1) + 1
