import random as rand
import copy
import tensorflow as tf
import numpy as np


class GameBoard:
    # class-wide variable dictating whether debug print statements are called
    debug_messages = False

    # generates a board based on the given information
    # expects number of pieces (optional; default is random), board lists (optional),
    # and whether we want to be one move away from a win (optional)
    def __init__(self, pieces=-1, game_boards=None, near_win=False):
        # track whether we have found a win, for near-win games
        self.found_win = False
        # case for when we pass a board
        if game_boards is not None:
            self.game_boards = game_boards
            self.pieces = count_pieces(self.game_boards)
            self.player_turn = self.pieces % 2
            return
        # boards for player 1, player 2, and placeable positions respectively
        self.game_boards = [[0] * 42, [0] * 42, [(1 if i < 7 else 0) for i in range(42)]]

        # case for creating a board that is one move away from a win
        if near_win:
            # plays a random game until we fail to place a piece
            for i in range(42):
                self.place_random(i % 2, near_win)
                # if we are one move away from a win, stop placing pieces.
                if self.found_win:
                    self.pieces = i
                    self.player_turn = (self.pieces - 1) % 2
                    return
        # creating a fully random board
        else:
            # sets pieces to a random amount if -1 is entered
            if pieces == -1:
                self.pieces = rand.randrange(42)
            # otherwise sets pieces to pieces
            else:
                self.pieces = pieces
            # generate the board based on the number of pieces:
            if GameBoard.debug_messages:
                print("Attempting to place " + str(self.pieces) + " pieces.")
            # simulating a random game where some amount of moves are played and nobody has won yet
            for i in range(self.pieces):
                self.place_random(i % 2)
            # instance variable for which player will place a piece next
            self.player_turn = self.pieces % 2
            if GameBoard.debug_messages:
                print("Managed to place " + str(self.pieces) + " pieces.")

    # simulates a random move
    # never returns a game with four in a row
    # expects the number of the board (1 or 2), depending on which player is making a move,
    # and whether we want to be one move away from a win (optional)
    def place_random(self, board_num, near_win=False):
        # repeatedly attempt to place a piece
        # 15 is arbitrary here, a high value will ensure less "missed" pieces
        for i in range(15):
            # pick a random column
            col = rand.randrange(7)
            # ensure the column is a valid move
            if self.valid_move(col):
                # find the row where the piece will fall
                row = self.find_row(col)
                # temporary board to use is.win with without altering the original board
                temp_board = self.game_boards[board_num].copy()
                temp_board[7 * row + col] = 1
                # check to see if the move in question would win the game (which we don't want)
                if not (is_win(temp_board)):
                    # adjust the real board
                    self.game_boards[board_num][7 * row + col] = 1
                    self.game_boards[2][7 * row + col] = 0
                    # if we are not on the top row, adjust the board representing available moves
                    if not (row == 5):
                        self.game_boards[2][7 * (row + 1) + col] = 1
                    return
                # if we are generating a GameBoard that we want to be one move away from a move, we are done
                if near_win:
                    self.found_win = True
                    return
        if GameBoard.debug_messages:
            print(" > Could not place a piece.")
        # if we failed to place a piece, decrease the number of pieces by 1
        self.pieces -= 1
        return

    # checks to see if there is an open slot in the given column of the board
    # expects the number of the column (0 to 6)
    def valid_move(self, col):
        # check each row in the board representing open slots
        for i in range(6):
            if self.game_boards[2][7 * i + col] == 1:
                return True
        return False

    # returns the height at which a piece would land if placed in that column
    # expects a integer 0 <= col <= 6 which represents a column
    def find_row(self, col):
        # ensure that the column is within the bounds (0 to 6)
        if (col < 0) or (col > 6):
            return None
        # if this value does not change, we output -1 which indicates the given column is full
        row = -1
        # iterate through the rows
        for i in range(6):
            # check if the position is marked in game_boards[2]
            if self.game_boards[2][7 * i + col] == 1:
                row = i
        return row

    # returns a basic graphical representation of a GameBoard
    def __str__(self):
        # string representation
        s = ""
        # iterate through the rows in reverse order
        for row in range(5, -1, -1):
            # list the row number
            s += "[" + str(row + 1) + "]"
            # iterate through the columns
            for col in range(7):
                # index within the board lists
                index = 7 * row + col
                # get the character based on the current player board and other player board
                s += " " + piece_character(self.game_boards[self.player_turn][index] +
                                           (2 * self.game_boards[(self.player_turn + 1) % 2][index]))
            s += "\n"
        # general information
        s += "Number of pieces: " + str(self.pieces) + "\n"
        s += "Next move: " + piece_character(self.player_turn + 1) + "\n"
        # return the string we have constructed
        return s

    # "getters" for boards, pieces, and turn
    # returns the three lists representing different aspects of the board
    def get_boards(self):
        return [self.game_boards[0], self.game_boards[1], self.game_boards[2]]

    # returns the number of pieces
    def get_pieces(self):
        return self.pieces

    # returns the player turn
    def get_player_turn(self):
        return self.player_turn

    # expects a column where a move is made, returns a flag representing the result
    # flag: 1 -> win, -1 -> loss / invalid move, 0 -> no result
    def make_move(self, col):
        # make sure the column is within the bounds
        if 1 <= col <= 7:
            # loss if the move is not valid
            if not self.valid_move(col - 1):
                return -1
            else:
                # find the row
                row = self.find_row(col - 1)
                # adjust the boards
                self.game_boards[0][7 * row + col - 1] = 1
                self.game_boards[2][7 * row + col - 1] = 0
                # adjust piece count and player turn
                self.pieces += 1
                self.player_turn = self.pieces % 2
                # if the piece was not dropped in the top row, update the board of placeable positions
                if not (row == 5):
                    self.game_boards[2][7 * (row + 1) + col - 1] = 1
                # switch which board is "active"
                self.game_boards[0], self.game_boards[1] = self.game_boards[1], self.game_boards[0]
                # check for a win and return win or no result
                if is_win(self.game_boards[1]):
                    return 1
                else:
                    return 0
        # invalid move / loss if column not within bounds
        return -1

    # returns a set of probabilities corresponding to the probability of wins and losses if playing move in each column
    # expects a tensorflow neural network model with the correct input and output size and format
    def desired_probabilities(self, model):
        # number of games we will test
        trials = 10
        # the maximum number of moves (42 <=> no limit)
        move_limit = 42 - self.pieces
        # all of the many boards that we will work with
        board_list = []
        # keep track of starting player
        player = self.player_turn
        # the overall stats for each of the seven initial moves
        results = [float(0)] * 7
        # keep track of which of the seven initial moves immediately ended the game
        immediate_end = [False] * 7
        # the stats of the random games; 2D, to track wins and losses for each
        stats = [[0, 0]] * 7
        # keep track of which boards no longer need to be played on
        finished = [False] * (trials * 7)
        # for each number 1 to 7, we play a move in that column and determine how useful that move is
        for i in range(7):
            # copy the board to not harm the original
            board = copy.deepcopy(self)
            # make a move in the column
            result = board.make_move(i + 1)
            # check if the game will continue
            if result == 0 and board.pieces < 42:
                pass  # do nothing
            # otherwise, mark the now finished boards
            else:
                # mark this initial move as finished
                immediate_end[i] = True
                # mark each of the trials for this bin as finished
                for j in range(trials):
                    finished[i * trials + j] = True
                # adjust the stats based on the result of the move if applicable
                if result == 1:
                    # if the game was immediately won, we have a 100% win rate in that column
                    results[i] = 10
                elif result == -1:
                    # if the game was immediately lost, we have a 100% loss rate in that column
                    results[i] = -10
            # copy this board into the main boards list a number of times equal to trials
            for k in range(trials):
                board_list.append(copy.deepcopy(board))
        # repeat as many moves as we are allowed
        for i in range(move_limit):
            # store all of the raw board outputs to be fed to the model
            board_data = []
            # concatenate all of these as one large tensor
            for b in board_list:
                b_output = b.get_boards()
                board_data.append(np.array(b_output[0] + b_output[1] + b_output[2]))
            # feed the data to the model and get our output
            move_data = model(np.stack(board_data), training=False)
            # for each board...
            for k in range(7 * trials):
                # check if the board is already done
                if finished[k]:
                    continue  # this board is finished and we skip it
                # pick a column to play based on model output
                col = pick_probability(move_data[k])
                # store the result of playing in the column
                result = board_list[k].make_move(col)
                # track the result and mark a finished board as finished
                if (result == 1 and board_list[k].player_turn == player) or (
                        result == -1 and board_list[k].player_turn != player):
                    stats[k // trials][0] += 1
                    finished[k] = True
                if (result == 1 and board_list[k].player_turn != player) or (
                        result == -1 and board_list[k].player_turn == player):
                    stats[k // trials][1] += 1
                    finished[k] = True
                if board_list[k].get_pieces() >= 42:
                    finished[k] = True
            # check to see if all boards are now finished, in which case we end
            if all_true(finished):
                break
        # for the boards that did not immediately finish, calculate their score
        for i in range(7):
            if not immediate_end[i]:
                results[i] = (stats[i][0] - stats[i][1]) / trials
        # softmax the score and return
        return soft_max(np.array(results))


# returns the desired_probabilities (see above) for each board in a list of boards
# expects a list of boards and a tensorflow neural network model
def desired_probabilities_batch(start_boards, model):
    # the length of the input
    length = len(start_boards)
    # number of games we will test
    trials = 10
    # all of the many boards that we will work with
    board_list = []
    # keep track of starting player for each board
    player = [b.player_turn for b in start_boards]
    # the overall stats for each of the seven initial moves for each of the initial boards
    # results = [[0 for i in range(7)] for b in start_boards]
    results = [[float(0)] * 7] * len(start_boards)
    # keep track of which of the seven initial moves immediately ended the game
    # immediate_end = [[False for i in range(7)] for b in start_boards]
    immediate_end = [[False] * 7] * len(start_boards)
    # the stats of the random games; 2D, to track wins and losses for each
    # stats = [[[0, 0] for i in range(7)] for b in start_boards]
    stats = [[[0, 0]] * 7] * len(start_boards)
    # keep track of which boards no longer need to be played on
    # finished = [False for i in range(trials * 7 * length)]
    finished = [False] * (trials * 7 * length)
    # for each of the starting boards...
    board_index = 0
    for b in start_boards:
        # for each number 1 to 7, we play a move in that column and determine how useful that move is
        for i in range(7):
            # copy the board to not harm the original
            board = copy.deepcopy(b)
            # make a move in the column
            result = board.make_move(i + 1)
            # check if the game will continue
            if result == 0 and board.pieces < 42:
                pass  # do nothing
            # otherwise, mark the now finished boards
            else:
                # mark this initial move as finished
                immediate_end[board_index][i] = True
                # mark each of the trials for this bin as finished
                for j in range(trials):
                    finished[(board_index * 7 * trials) + (i * trials) + j] = True
                # adjust the stats based on the result of the move if applicable
                if result == 1:
                    # if the game was immediately won, we have a 100% win rate in that column
                    results[board_index][i] = 10
                elif result == -1:
                    # if the game was immediately lost, we have a 100% loss rate in that column
                    results[board_index][i] = -10
            # copy this board into the main boards list a number of times equal to trials
            for k in range(trials):
                board_list.append(copy.deepcopy(board))
        board_index += 1
    # repeat as many moves as we are allowed
    for i in range(42):
        # store all of the raw board outputs to be fed to the model
        board_data = []
        # concatenate all of these as one large tensor
        for b in board_list:
            b_output = b.get_boards()
            board_data.append(np.array(b_output[0] + b_output[1] + b_output[2]))
        # feed the data to the model and get our output
        move_data = model(np.stack(board_data), training=False)
        # for each starting board...
        for j in range(length):
            # for each of its games...
            for k in range(7 * trials):
                # check if the board is already done
                if finished[(j * trials * 7) + k]:
                    continue  # this board is finished and we skip it
                # pick a column to play based on model output
                col = pick_probability(move_data[(j * trials * 7) + k])
                # store the result of playing in the column
                result = board_list[(j * trials * 7) + k].make_move(col)
                # track the result and mark a finished board as finished
                if (result == 1 and board_list[(j * trials * 7) + k].player_turn == player[j]) or (
                        result == -1 and board_list[(j * trials * 7) + k].player_turn != player[j]):
                    stats[j][k // trials][0] += 1
                    finished[(j * trials * 7) + k] = True
                if (result == 1 and board_list[(j * trials * 7) + k].player_turn != player[j]) or (
                        result == -1 and board_list[(j * trials * 7) + k].player_turn == player[j]):
                    stats[j][k // trials][1] += 1
                    finished[(j * trials * 7) + k] = True
                if board_list[(j * trials * 7) + k].get_pieces() >= 42:
                    finished[(j * trials * 7) + k] = True
        # check to see if all boards are now finished, in which case we end
        if all_true(finished):
            break
    # for the boards that did not immediately finish, calculate their score
    for j in range(length):
        for i in range(7):
            if not immediate_end[j][i]:
                results[j][i] = (stats[j][i][0] - stats[j][i][1]) / trials
    # softmax the score and return
    return [soft_max(results[i]) for i in range(length)]


# returns a version of the array where each value has been translated into a "probability"
# expects an array / list of numbers
def soft_max(arr1):
    # exponentiate the array
    e = np.exp(arr1)
    # divide the array by the sum of its values
    return e / np.sum(e)


# returns a number from 1 to n, where n is the size of the list, picked at random based on the percentages represented
# expects arr1 to represent a set of percentage chances (it should be composed of positive real numbers with a sum of 1)
def pick_probability(arr1):
    # generate a random number from 0 to 1
    # rand.seed(2.718281828459045)  # used for testing purposes
    r = rand.random()
    rand.seed()
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


# expects a list / array of boolean values, checks if entirely true values
# used to help determine if all of the games are finished in desired_probabilities
def all_true(arr):
    # checks if each element (e) of the array is true
    for e in arr:
        if not e:
            # if any is not true, we return false
            return False
    # if none are false, return true
    return True


# expects a list of three lists of length 42, representing the boards of a connect4 game
# returns the number of pieces by counting digit 1 in first two boards
def count_pieces(boards):
    # keep track of piece count
    piece_count = 0
    # check all elements of board 0
    for i in boards[0]:
        if i == 1:
            piece_count += 1
    # check all elements of board 1
    for i in boards[1]:
        if i == 1:
            piece_count += 1
    return piece_count


# gives an array of random boards with the given size
# expects a natural number for set_size, which represents the number of boards generated
def get_samples(set_size):
    # store the boards
    boards = []
    # generate a number of random boards equal to the set size
    for i in range(set_size):
        boards.append(GameBoard())
        # print the board if applicable
        if GameBoard.debug_messages:
            print(boards[i])
    # change the list to a numpy array
    return np.array(boards)


# gets the training data (the raw board lists) for a given set of sample boards
# expects a list / array of boards
def get_training_data(samples):
    # the list
    training_data = []
    # iterate through the samples
    for s in samples:
        # add the boards
        training_data += [s.get_boards()[0] + s.get_boards()[1] + s.get_boards()[2]]
    # template and return
    return tf.constant(training_data)


# returns the target data (desired probabilities) for each of the samples in the list
# expects a list of boards serving as the samples to get data from, and a tensorflow neural network model
def get_target_data(samples, model):
    # store the data
    target_data = []
    # iterate through the samples, and get the desired probabilities for each
    for s in samples:
        target_data += [s.desired_probabilities(model)]
    # convert to a tensorflow tensor, which works better with the model
    return tf.constant(target_data)


# by default, returns the average euclidean distance between desired_probabilities and model output between the elements
# different values of report_type cause different things to be printed IN ADDITION to returning the distance
# report_type = 0 -> nothing is printed
# report_type = 1 -> average distance is printed
# report_type = 2 -> the distances for individual boards are printed (as well as average)
# report_type = 3 -> the boards themselves and their vectors are printed (as well as all of the above)
def evaluate(samples, model, report_type=0):
    # print the title bar if appropriate
    if report_type >= 1:
        print("=" * 28, "SAMPLE EVALUATION", "=" * 28)
        print()
    # keep track of the distances to find the average
    total_distance = 0
    for i in range(len(samples)):
        # get a the next board
        s = samples[i]
        # get the raw data; call the model
        board_data = [np.array(s.get_boards()[0] + s.get_boards()[1] + s.get_boards()[2])]
        x = model(np.stack(board_data), training=False)
        # get the desired probabilities
        y = s.desired_probabilities(model)
        # determine the Euclidean distance
        distance = np.linalg.norm(x[0] - y)
        # add it to the total (will be used for the average
        total_distance += distance
        # print the appropriate data
        if report_type >= 2:
            print("-" * 33, "Board", i + 1, "-" * 33)
            print()
        if report_type >= 3:
            print(samples[i])
            print("Actual data:", vector_string(x[0]))
            print("Target data:", vector_string(y))
            print()
        if report_type >= 2:
            print("Euclidean distance between target and actual:", "{:.4f}".format(distance))
            print()
    # divide the total by the number of samples to get the average distance
    average_distance = total_distance / len(samples)
    # print the average if applicable
    if report_type >= 1:
        print("*" * 20, "Average Euclidean distance:", "{:.4f}".format(average_distance), "*" * 20)
        print()
    # return the average
    return average_distance


# converts an iterable object containing doubles into a slightly cleaner form
# expects a list / array
def vector_string(vector):
    # print the first element, to four decimal places
    s = "[ " + "{:.4f}".format(vector[0])
    # for each other element, print it to four decimal places
    for i in range(1, len(vector)):
        s += ", " + "{:.4f}".format(vector[i])
    # close the brackets
    s += " ]"
    # return the string we have built
    return s


# enables or disables the debug messages printed throughout the GameBoard class
# expects a boolean (True -> debug messages will appear)
def allow_debug(boolean):
    GameBoard.debug_messages = boolean


# gets a character representing one of the pieces on the board, coloured for visibility
# expects a number (calculated in the __str__ method)
def piece_character(num):
    # dash for empty space
    if num == 0:
        return "-"
    # "R" coloured red
    if num == 1:
        return "\33[91mR\33[0m"
    # "B" coloured blue
    if num == 2:
        return "\33[94mB\33[0m"


# returns the entry at a particular position within a particular board
# expects a list representing the board,
# and a tuple (k, s) representing the coordinates, where k is the level and s is the column
def board_entry(board, position):
    # ensure that the position is within the bounds
    if 1 <= position[0] <= 6 and 1 <= position[1] <= 7:
        # return the element of the array at that position, by the dimensions of the connect4 board
        return board[7 * (position[0] - 1) + position[1] - 1]
    # return -1 if invalid move
    return -1


# determines whether a list has four consecutive digits '1'
# expects a list representing a line
def line_of_four(line):
    # iterate through the list, leaving space for the line
    for i in range(len(line) - 3):
        # check the current position and the three after
        if line[i] & line[i + 1] & line[i + 2] & line[i + 3] == 1:
            return True
    # false if no line was found
    return False


# checks if a given board contains a winning combination of pieces
# expects a list representing a board
def is_win(board):
    # check each row
    for k in range(1, 7):
        if line_of_four([board_entry(board, (k, s)) for s in range(1, 8)]):
            return True
    # check each column
    for s in range(1, 8):
        if line_of_four([board_entry(board, (k, s)) for k in range(1, 7)]):
            return True
    # check the diagonals
    for i in range(1, 4):
        if line_of_four([board_entry(board, (i + j, 1 + j)) for j in range(7 - i)]):
            return True
        if line_of_four([board_entry(board, (1 + j, 1 + i + j)) for j in range(7 - i)]):
            return True
        if line_of_four([board_entry(board, (i + j, 7 - j)) for j in range(7 - i)]):
            return True
        if line_of_four([board_entry(board, (1 + j, 7 - i - j)) for j in range(7 - i)]):
            return True
    # return false if no win was found
    return False
