import random as rand


class GameBoard:

    # initialization of a game board
    # if a random amount of moves should be done, pass -1 as an argument
    def __init__(self, pieces):
        self.pieces = pieces  # instance variable for how many pieces are on the board
        if pieces == -1:
            self.pieces = 2*rand.randrange(1, 21)

        #                  row 1           row 2          row 3          row 4          row 5          row 6
        self.gameBoards = [[0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0],  # board for player 1
                           [0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0],  # board for player 2
                           [1,1,1,1,1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]]  # board of currently placeable positions

        self.playerTurn = self.pieces % 2  # instance variable for which player will place a piece next

        print("Placing " + str(self.pieces) + " pieces!")
        for i in range(0, self.pieces):  # simulating each player taking turns placing pieces at random
            self.placeRandPieces(i % 2)

    # functions for piece placement
    def placeRandPieces(self, boardNum):
        positionsTried = []
        while True:
            col = rand.randrange(0, 7)

            if col in positionsTried:
                continue  # if we have already tried this position, we don't want to waste time trying it again
            else:
                positionsTried.append(col)

            if self.validMove(col) and not (self.winningMove(col, boardNum)):
                row = self.positionFinder(col)

                self.gameBoards[boardNum][7*row + col] = 1
                self.gameBoards[2][7*row + col] = 0

                if not(row == 5):
                    self.gameBoards[2][7*(row+1) + col] = 1

                return

    # checks to see if there is an open slot in the given row of a connect4 board
    def validMove(self, col):
        for i in range(0, 6):
            if self.gameBoards[2][7*i + col] == 1:
                return True

        return False

    def winningMove(self, col, boardNum):

        # check for horizontal win

        # check for vertical win

        # check for diagonal win

        return False

    # recovers the position of the next empty slot in a column
    def positionFinder(self, col):
        row = 0
        for i in range(0, 6):
            if self.gameBoards[2][7*i + col] == 1:
                row = i

        return row

    #  function for printing the current game board
    def __str__(self):
        s = ""
        for row in range(5, -1, -1):
            s += "Row" + str(row + 1)
            if self.playerTurn == 0:
                for col in range(0, 7):
                    s += " " + self.numToLetter(self.gameBoards[0][7*row + col] + 2*self.gameBoards[1][7*row + col])
                s += "\n"
            if self.playerTurn == 1:
                for col in range(0, 7):
                    s += " " + self.numToLetter(self.gameBoards[1][7*row + col] + 2*self.gameBoards[0][7*row + col])
                s += "\n"
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
        return self.gameBoards[0], self.gameBoards[1], self.gameBoards[2]

    def getPieces(self):
        return self.pieces

    def getPlayerTurn(self):
        return self.playerTurn

    def make_move(self, column):
        # Thinh's "make_move" function goes here
        # Take in 2 boards, and the column player 1 wants to place the disc,
        # and return 2 new boards, and a flag (1 for win move, -1 for invalid move, 0 otherwise)
        if 1 <= column <= 7:
            if self.level_check(self.gameBoards[0], self.gameBoards[1], column) == 6:
                return self.gameBoards[0], self.gameBoards[1], -1

            if self.level_check(self.gameBoards[0], self.gameBoards[1], column) < 6:
                self.board_change(self.gameBoards[0], (self.level_check(self.gameBoards[0], self.gameBoards[1], column) + 1, column))
                self.pieces += 1
                self.playerTurn = self.pieces % 2

            if self.is_win(self.gameBoards[0]):
                return self.gameBoards[0], self.gameBoards[1], 1

            else:
                self.gameBoards[0], self.gameBoards[1] = self.gameBoards[1], self.gameBoards[0]
                return self.gameBoards[0], self.gameBoards[1], 0

        else:
            return self.gameBoards[0], self.gameBoards[1], -1

    def board_change(self, board, position):
        # Place a dic to a position with given board input and tuple for position
        if 1 <= position[0] <= 6 and 1 <= position[1] <= 7:
            board[7 * (position[0] - 1) + position[1] - 1] = 1

    def board_value(self, board, position):
        # Take in a board(list) and a tuple (k,s) where k is the level and s is the column,
        # and return the value of the board at that position (-1 if given invalid position)
        if 1 <= position[0] <= 6 and 1 <= position[1] <= 7:
            return board[7 * (position[0] - 1) + position[1] - 1]
        else:
            return -1

    def level_check(self, board1, board2, column):
        # Take in 2 boards and a column, return the current level of the column (-1 if given invalid column)
        if 1 <= column <= 7:
            b1 = [self.board_value(board1, (k, column)) for k in range(6, 0, -1)]
            b2 = [self.board_value(board2, (k, column)) for k in range(6, 0, -1)]
            try:
                pos1 = 6 - b1.index(1)
            except ValueError:
                pos1 = 0
            try:
                pos2 = 6 - b2.index(1)
            except ValueError:
                pos2 = 0
            if pos1 == 6 or pos2 == 6:
                return 6
            if pos1 >= pos2:
                return pos1
            else:
                return pos2
        else:
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


# the following lines are examples of how to use a GameBoard object
g = GameBoard(-1)
print(g)  # returns a graphical version of the current GameBoard object
print(g.getBoards()[0])  # returns player1's board
g.make_move(6)  # this is how our neural networks makes a move (specifically in this case placing a piece in column 6)
print(g)
