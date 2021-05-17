import math
import numpy as np
import random


# create seven bins, one for each column, and compute the desired probabilities for each based on their likelihood
def desired_probabilities(base_board, model):
    # keep track of the results from each column to later form percentage odds
    stats = [0, 0, 0, 0, 0, 0, 0]
    # for each number 1 to 7, we play a move in that column and determine how useful that move is
    for i in range(1, 8):
        # copy the board to not harm the original
        board = base_board.copy()
        # make a move in the column
        _, _, result = board.make_move(i)
        # adjust the stats based on the result of that move, either the immediate result or the value for future games
        if result == 1:
            # if the game was immediately won, we have a 100% win rate in that column
            stats[i - 1] = 1
        elif result == -1:
            # if the game was immediately lost, we have a 100% loss rate in that column
            stats[i - 1] = -1
        elif board.pieces >= 42:
            # if that move brought us to 42 moves, we have a draw in that column
            stats[i - 1] = 0
        elif result == 0:
            # if the game did not end after the last move, get the board position value
            stats[i - 1] = board_value(board, model)
    # return soft max of the results
    return soft_max(np.array(stats))


# returns the "value" of a given board position (corresponds to the likelihood of winning from that position)
# we need to be given the board, and the model
def board_value(base_board, model):
    # number of test games we will play
    trials = 100
    # the list storing the win/loss/draw stats
    stats = [0, 0, 0]
    # index of player we want the stats for
    player = base_board.playerTurn

    # play the given number of games
    for i in range(trials):
        # make a copy of the board that we will modify
        board = base_board.copy()
        # play until the game is finished - continually alternating b/w the two players
        result = 0
        while result == 0 and board.pieces < 42:
            result = play_next_move(board, model)
        # Update stats after a given game is completed
        update_stats(player, stats, result, board.playerTurn)

    # At this point, we have all our stats, and we return the "value" of the move
    return (stats[0] - stats[1]) / trials


# HELPER FUNCTIONS:

# Plays next move given the current board, using the trained model. Returns flag from make_move
def play_next_move(board, model):
    # get the probability distribution from the model
    prob_distribution = soft_max(model.predict(board))
    # pick a column at random using the helper function
    column = pick_probability(prob_distribution)
    # make a move
    _, _, result = board.make_move(column)
    return result


# For a given game played to completion, we update stats relative to our starting player
def update_stats(player, stats, result, player_turn):
    # Various checks to see whether our player won, lost, or draw
    if result == 0:
        stats[2] += 1
    elif (result == 1 and player == player_turn) or (result == -1 and player != player_turn):
        stats[0] += 1
    else:
        stats[1] += 1


# expects an array of real numbers
def soft_max(arr1):
    # create a list and fill it with exponentiated values from the original array
    exponentials = [math.exp(i) for i in arr1]
    # sum the values in the list to divide
    total = sum(exponentials)
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
