import random as rand


class GameBoard:

    # initialization of a game board
    # if a random amount of moves should be done, pass -1 as an argument
    def __init__(self, pieces):
        self.pieces = pieces

        #                  row 1           row 2          row 3          row 4          row 5          row 6
        self.gameBoards = [[0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0],  # board for player 1
                           [0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0],  # board for player 2
                           [1,1,1,1,1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]]  # board of currently placeable positions
        if pieces == -1:
            self.pieces = rand.randrange(1, 42)

        print("Placing " + str(self.pieces) + " pieces!")
        for i in range(0, self.pieces):  # simulating each player taking turns placing pieces at random
            self.placeRandPieces(i % 2)

    # functions for piece placement
    def placeRandPieces(self, boardNum):
        positionsTried = []
        while True:
            row = rand.randrange(0, 6)
            col = rand.randrange(0, 7)

            if [row, col] in positionsTried:
                continue  # if we have already tried this position, we don't want to waste time trying it again
            else:
                positionsTried.append([row, col])

            if self.validMove(row, col) and not (self.winningMove(row, col, boardNum)):
                self.gameBoards[boardNum][7*row + col] = 1
                self.gameBoards[2][7*row + col] = 0

                if not(row == 5):
                    self.gameBoards[2][7*(row+1) + col] = 1

                return

    def validMove(self, row, col):
        if self.gameBoards[2][7*row + col] == 0:
            return False

        return True

    def winningMove(self, row, col, boardNum):

        # check for horizontal win

        # check for vertical win

        # check for diagonal win

        return False

    #  function for printing the current game board
    def __str__(self):
        s = ""
        for row in range(5, -1, -1):
            s += "Row" + str(row + 1)
            for col in range(0, 7):
                s += " " + self.numToLetter(self.gameBoards[0][7*row + col] + 2*self.gameBoards[1][7*row + col])
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
        return self.gameBoards[0], self.gameBoards[1]

    def make_move(self, boardNum, Col):
        # Thinh's "make_move" function goes here
        pass


# the following lines are examples of how to use a GameBoard object
g = GameBoard(-1)
print(g)  # returns a graphical version of the current GameBoard object
print(g.getBoards()[0])  # returns player1's board
# g.make_move((PlayerNum-1), col) This will the the way that both a our neural network, and a player make moves, make_move already has access to both boards through the self indicator here
