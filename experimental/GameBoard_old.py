import random as rand
import copy
import tensorflow as tf
import numpy as np


avg_dist = np.array([])


class GameBoard:

    # initialization of a game board
    # if a random amount of moves should be done, pass -1 as an argument
    # def __init__(self, pieces=-1):
    #     self.pieces = pieces  # instance variable for how many pieces are on the board
    #     if pieces == -1:
    #         self.pieces = 2 * rand.randrange(1, 21)
    #
    #     #                  row 1           row 2          row 3          row 4          row 5          row 6
    #     self.gameBoards = [
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0],  # board for player 1
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0],  # board for player 2
    #         [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0]]  # board of currently placeable positions
    #
    #     self.playerTurn = self.pieces % 2  # instance variable for which player will place a piece next
    #
    #     print("Placing " + str(self.pieces) + " pieces!")
    #     for i in range(0, self.pieces):  # simulating each player taking turns placing pieces at random
    #         self.placeRandPieces(i % 2)
    #     self.playerTurn = self.pieces % 2
    #     print("Managed to place " + str(self.pieces) + " pieces")

    def __init__(self, pieces=-1, game_boards=None, oneAwayFromWin=False):
        self.foundWin = False

        # Case for when we pass a board
        if game_boards is not None:
            self.gameBoards = game_boards
            self.pieces = count_pieces(self.gameBoards)
            self.playerTurn = self.pieces % 2
            return

        self.gameBoards = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],  # board for player 1 (Red)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0],  # board for player 2 (Blue)
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0]]  # board of currently placeable positions

        # case for creating a GameBoard that is one move away from a win
        if oneAwayFromWin:

            # plays a random game until we fail to place a piece
            for i in range(42):
                self.placeRandPieces(i % 2, oneAwayFromWin)

                # if we are one move away from a win, stop placing pieces.
                if self.foundWin:
                    self.pieces = i
                    self.playerTurn = (self.pieces - 1) % 2  # we subtract 1 from self.pieces because in the last round through this loop we didn't place a piece
                    return

        # creating a random GameBoard
        else:
            # sets pieces to a random amount if -1 is entered
            if pieces == -1:
                self.pieces = rand.randrange(0, 42)
            # otherwise sets pieces to pieces
            else:
                self.pieces = pieces

            # simulating a random game where some amount of moves are played and now one has won yet
            print("Placing " + str(self.pieces) + " pieces!")
            for i in range(self.pieces):
                self.placeRandPieces(i % 2)

            self.playerTurn = self.pieces % 2  # instance variable for which player will place a piece next

            print("Managed to place " + str(self.pieces) + " pieces")

    # simulates a random move
    # never returns a game with 4 in a row already
    def placeRandPieces(self, boardNum, oneAwayFromWin=False):
        for i in range(0, 15):  # 15 is arbitrary here, a high value will ensure less "missed" pieces
            col = rand.randrange(0, 7)

            if self.validMove(col):
                row = self.rowFinder(col)

                if row is not None:
                    tempBoard = self.gameBoards[boardNum].copy()  # temporary board to use is.win with without altering the original board
                    tempBoard[7 * row + col] = 1

                    # check to see if the move in question would win the game (which we don't want)
                    if not (self.is_win(tempBoard)):
                        self.gameBoards[boardNum][7 * row + col] = 1  # we place a piece here
                        self.gameBoards[2][7 * row + col] = 0  # updates where we can place a piece in the given column

                        # if we are not on the top row, do this
                        if not (row == 5):
                            self.gameBoards[2][7 * (row + 1) + col] = 1  # updates where we can place a piece in the given column
                        return

                    # if we are generating a GameBoard that we want to be one move away from a move, we are done
                    if oneAwayFromWin:
                        self.foundWin = True
                        return

        print("could not place a piece")
        self.pieces -= 1
        return

    # checks to see if there is an open slot in the given row of a connect4 board
    def validMove(self, col):
        for i in range(0, 6):
            if self.gameBoards[2][7 * i + col] == 1:
                return True

        return False

    # expects a integer 0 <= col <= 6 which represents a column
    # returns the height at which a piece would land if placed in that column
    def rowFinder(self, col):
        row = None  # if this value does not change, we return None which indicates the given column is full
        if (col < 0) or (col > 6):
            return row

        for i in range(0, 6):
            if self.gameBoards[2][7 * i + col] == 1:
                row = i

        return row

    # returns a basic graphical representation of a GameBoard
    def __str__(self):
        s = ""
        for row in range(5, -1, -1):
            s += "Row" + str(row + 1)
            if self.playerTurn == 0:
                for col in range(0, 7):
                    s += " " + self.numToLetter(self.gameBoards[0][7 * row + col] + 2 * self.gameBoards[1][7 * row + col])
                s += "\n"
            if self.playerTurn == 1:
                for col in range(0, 7):
                    s += " " + self.numToLetter(self.gameBoards[1][7 * row + col] + 2 * self.gameBoards[0][7 * row + col])
                s += "\n"
        s += "There are " + str(self.pieces) + " pieces on this board" + "\n"
        s += "The " + self.numToLetter(self.playerTurn + 1) + " will place the next piece" + "\n"

        return s

    def numToLetter(self, num):
        if num == 0:
            return "-"
        if num == 1:
            return "R"
        if num == 2:
            return "B"

    # function for getting the boards of player 1 and player 2
    def getBoards(self):
        return [self.gameBoards[0], self.gameBoards[1], self.gameBoards[2]]

    def getPieces(self):
        return self.pieces

    def getPlayerTurn(self):
        return self.playerTurn

    # make_move function below here
    def make_move(self, col):
        # Thinh's "make_move" function goes here
        # Take in 2 boards, and the column player 1 wants to place the disc,
        # and return 2 new boards, and a flag (1 for win move, -1 for invalid move, 0 otherwise)
        if 1 <= col <= 7:
            if not (self.validMove(col - 1)):
                return -1
            else:
                row = self.rowFinder(col - 1)
                self.gameBoards[0][7 * row + col - 1] = 1
                self.gameBoards[2][7 * row + col - 1] = 0
                self.pieces += 1
                self.playerTurn = self.pieces % 2
                if not (row == 5):
                    self.gameBoards[2][7 * (row + 1) + col - 1] = 1
                self.gameBoards[0], self.gameBoards[1] = self.gameBoards[1], self.gameBoards[0]
                if self.is_win(self.gameBoards[1]):
                    return 1
                else:
                    return 0
        return -1

    def board_value(self, board, position):
        # Take in a board(list) and a tuple (k,s) where k is the level and s is the column,
        # and return the value of the board at that position (-1 if given invalid position)
        if 1 <= position[0] <= 6 and 1 <= position[1] <= 7:
            return board[7 * (position[0] - 1) + position[1] - 1]
        return -1

    def has_4_in_a_line(self, line):
        # Check if there are four consecutive 1's on a line (list)
        for i in range(len(line) - 3):
            if line[i] & line[i + 1] & line[i + 2] & line[i + 3] == 1:
                return True
        return False

    def is_win(self, board):
        # Check if it is a winning board

        # Check each level
        for k in range(1, 7):
            if self.has_4_in_a_line([self.board_value(board, (k, s)) for s in range(1, 8)]):
                return True
        # Check each column
        for s in range(1, 8):
            if self.has_4_in_a_line([self.board_value(board, (k, s)) for k in range(1, 7)]):
                return True
        # Check diagonals
        for i in range(1, 4):
            if self.has_4_in_a_line([self.board_value(board, (i + j, 1 + j)) for j in range(7 - i)]):
                return True
            if self.has_4_in_a_line([self.board_value(board, (1 + j, 1 + i + j)) for j in range(7 - i)]):
                return True
            if self.has_4_in_a_line([self.board_value(board, (i + j, 7 - j)) for j in range(7 - i)]):
                return True
            if self.has_4_in_a_line([self.board_value(board, (1 + j, 7 - i - j)) for j in range(7 - i)]):
                return True
        return False

    # create seven bins, one for each column, and compute the desired probabilities for each based on their likelihood
    def desired_probabilities(self, model):
        # keep track of the results from each column to later form percentage odds
        stats = [0, 0, 0, 0, 0, 0, 0]
        # for each number 1 to 7, we play a move in that column and determine how useful that move is
        for i in range(1, 8):
            # copy the board to not harm the original
            board = copy.deepcopy(self)
            # make a move in the column
            result = board.make_move(i)
            # adjust the stats based on the result of the move, either the immediate result or the score of future games
            if result == 1:
                # if the game was immediately won, we have a 100% win rate in that column
                stats[i - 1] = 3
            elif result == -1:
                # if the game was immediately lost, we have a 100% loss rate in that column
                stats[i - 1] = -3
            elif board.pieces >= 42:
                # if that move brought us to 42 moves, we have a draw in that column
                stats[i - 1] = 0
            elif result == 0:
                # if the game did not end after the last move, get the board position value
                stats[i - 1] = board.board_score(model)
        # return soft max of the results
        return soft_max(np.array(stats))

    # returns the "value" of a given board position (corresponds to the likelihood of winning from that position)
    # we need to be given the board, and the model
    def board_score(self, model):
        # number of test games we will play
        trials = 10
        # the maximum number of moves we will try (used to reduce runtime)
        # a limit of 42 is effectively equivalent to no limit
        move_limit = 42 - self.pieces
        # keep track of the number of pieces we will be allowed to see placed before giving up
        piece_limit = self.pieces + move_limit
        # if this number manages to be greater than 42, cap it at 42
        if piece_limit > 42:
            piece_limit = 42
        # the list storing the win/loss stats
        stats = [0, 0]
        # index of player we want the stats for
        player = self.playerTurn

        # play the given number of games
        for i in range(trials):
            # make a copy of the board that we will modify
            board = copy.deepcopy(self)  # TODO: consider coding custom version with faster runtime
            # play until the game is finished, either by win, loss, or overstepping the piece limit
            result = 0
            while result == 0 and board.pieces < piece_limit:
                result = board.play_next_move(model)
            # update stats after a given game is completed
            board.update_stats(player, stats, result, board.playerTurn, piece_limit)

        # at this point, we have all our stats, and we return the "score" of the move
        return (stats[0] - stats[1]) / trials

    # HELPER FUNCTIONS:

    # plays next move given the current board, using the trained model. Returns flag from make_move
    def play_next_move(self, model):
        # get the probability distribution from the model
        trainer = tf.constant([(self.getBoards()[0]) + (self.getBoards()[1]) + (self.getBoards()[2])])
        prob_distribution = model.predict(trainer)[0]
        # pick a column at random using the helper function
        column = pick_probability(prob_distribution)
        # make a move
        result = self.make_move(column)
        return result

    def copy(self):
        return GameBoard(self.getPieces(), self.gameBoards.copy())

    def desired_probabilities_2(self, model):
        # number of games we will test
        trials = 10
        # the maximum number of moves (42 <=> no limit)
        move_limit = 42 - self.pieces  # TODO
        # all of the many boards that we will work with
        board_list = []
        # keep track of starting player
        player = self.playerTurn
        # the overall stats for each of the seven initial moves
        results = [0 for i in range(7)]
        # keep track of which of the seven initial moves immediately ended the game
        immediate_end = [False for i in range(7)]
        # the stats of the random games; 2D, to track wins and losses for each
        stats = [[0, 0] for i in range(7)]
        # keep track of which boards no longer need to be played on
        finished = [False for i in range(trials * 7)]
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
                    results[i] = 3
                elif result == -1:
                    # if the game was immediately lost, we have a 100% loss rate in that column
                    results[i] = -3
            # copy this board into the main boards list a number of times equal to trials
            for k in range(trials):
                board_list.append(copy.deepcopy(board))
        # repeat as many moves as we are allowed
        for i in range(move_limit):
            # store all of the raw board outputs to be fed to the model
            board_data = []
            # concatenate all of these as one large tensor
            for b in board_list:
                b_output = b.getBoards()
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
                if (result == 1 and board_list[k].playerTurn == player) or (
                        result == -1 and board_list[k].playerTurn != player):
                    stats[k // trials][0] += 1
                    finished[k] = True
                if (result == 1 and board_list[k].playerTurn != player) or (
                        result == -1 and board_list[k].playerTurn == player):
                    stats[k // trials][1] += 1
                    finished[k] = True
                if board_list[k].getPieces() >= 42:
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

    # For a given game played to completion, we update stats relative to our starting player
    def update_stats(self, player, stats, result, player_turn, piece_limit):
        # Various checks to see whether our player won, lost, or draw
        if result == 0 and (self.pieces >= 42 or self.pieces >= piece_limit):
            pass  # do nothing
        elif (result == 1 and player == player_turn) or (result == -1 and player != player_turn):
            stats[1] += 1
        else:
            stats[0] += 1


def desired_probabilities_3(start_boards, model):
    # the length of the input
    length = len(start_boards)
    # number of games we will test
    trials = 10
    # # the maximum number of moves (42 <=> no limit)
    # move_limit = 41 - self.pieces  # TODO
    # all of the many boards that we will work with
    board_list = []
    # keep track of starting player for each board
    player = [b.playerTurn for b in start_boards]
    # the overall stats for each of the seven initial moves for each of the initial boards
    results = [[0 for i in range(7)] for b in start_boards]
    # keep track of which of the seven initial moves immediately ended the game
    immediate_end = [[False for i in range(7)] for b in start_boards]
    # the stats of the random games; 2D, to track wins and losses for each
    stats = [[[0, 0] for i in range(7)] for b in start_boards]
    # keep track of which boards no longer need to be played on
    finished = [False for i in range(trials * 7 * length)]
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
                    results[board_index][i] = 3
                elif result == -1:
                    # if the game was immediately lost, we have a 100% loss rate in that column
                    results[board_index][i] = -3
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
            b_output = b.getBoards()
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
                if (result == 1 and board_list[(j * trials * 7) + k].playerTurn == player[j]) or (
                        result == -1 and board_list[(j * trials * 7) + k].playerTurn != player[j]):
                    stats[j][k // trials][0] += 1
                    finished[(j * trials * 7) + k] = True
                if (result == 1 and board_list[(j * trials * 7) + k].playerTurn != player[j]) or (
                        result == -1 and board_list[(j * trials * 7) + k].playerTurn == player[j]):
                    stats[j][k // trials][1] += 1
                    finished[(j * trials * 7) + k] = True
                if board_list[(j * trials * 7) + k].getPieces() >= 42:
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


def soft_max(arr1):
    # exponentiate the array
    e = np.exp(arr1)
    # divide the array by the sum of its values
    return e / np.sum(e)


# expects arr1 to represent a set of percentage chances (it should be composed of positive real numbers with a sum of 1)
def pick_probability(arr1):
    # generate a random number from 0 to 1
    r = rand.random()
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
    for e in arr:
        if not e:
            return False
    return True


# expects a list of three lists of length 42, representing the boards of a connect4 game
# returns the number of pieces by counting digit 1 in first two boards
def count_pieces(boards):
    piece_count = 0
    for i in boards[0]:
        if i == 1:
            piece_count += 1
    for i in boards[1]:
        if i == 1:
            piece_count += 1
    return piece_count


# gives an array of random boards with the given size
def get_samples(set_size):
    boards = []
    for i in range(set_size):
        boards.append(GameBoard())
        print(boards[i])
        # print(GameBoard())
    return np.array(boards)


# gets the training data (the raw board lists) for a given set of sample boards
def get_training_data(samples):
    # the list
    training_data = []
    # iterate through the samples
    for s in samples:
        # add the boards
        training_data += [s.getBoards()[0] + s.getBoards()[1] + s.getBoards()[2]]
    # template and return
    return tf.constant(training_data)


# TODO: consider using desired_probabilities_3 on the whole list at once
def get_target_data(samples, model):
    target_data = []
    for s in samples:
        target_data += [s.desired_probabilities_2(model)]
    return tf.constant(target_data)


def evaluate(samples, model, full_report=False):
    global avg_dist
    print("=" * 28, "SAMPLE EVALUATION", "=" * 28)
    average_distance = 0
    i = 0
    for s in samples:
        i += 1
        board_data = [np.array(s.getBoards()[0] + s.getBoards()[1] + s.getBoards()[2])]
        x = model(np.stack(board_data), training=False)
        y = s.desired_probabilities_2(model)
        distance = np.linalg.norm(x[0] - y)
        average_distance += distance
        if full_report:
            print("-" * 33, "Board", i, "-" * 33)
            print("Actual data:", x[0])
            print("Target data:", y)
            print("Euclidean distance between target and actual:", distance)
    average_distance /= len(samples)
    avg_dist = np.append(avg_dist, [average_distance])
    print(avg_dist)
    print(">>>>> Average euclidean distance:", average_distance, "<<<<<")
    print()
    print()


f = GameBoard(0)
g = GameBoard(1)

print(f)
print(f.getBoards())
f.make_move(2)
print(f)
print(f.getBoards())

x = GameBoard(oneAwayFromWin=True)
y = GameBoard(oneAwayFromWin=True)
z = GameBoard(oneAwayFromWin=True)
w = GameBoard(oneAwayFromWin=True)

print(x)
print(y)
print(z)
print(w)
