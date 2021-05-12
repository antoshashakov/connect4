import random as rand

# board1 = [0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]  # player1 board
# board2 = [0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]  # player2 board
# board3 = [1,1,1,1,1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]  # board of placeable slots

#              row 1          row 2          row 3          row 4          row 5          row 6
gameBoards = [[0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0],  # player1 board (Red pieces)
              [0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0],  # player2 board (Blue pieces)
              [1,1,1,1,1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]]  # board of placeable slots


def placePiece(boardNum):  # places a piece of either the red or blue player's board
    positionsTried = []  # create a list of positions already tried
    while True:
        row = rand.randrange(0, 6)
        col = rand.randrange(0, 7)

        if [row, col] in positionsTried:
            continue
        else:
            positionsTried.append([row, col])

        if validMove(col, row) and not(winningMove(col, row, gameBoards[boardNum])):  # create winningMove(col, row gameBoards[boardNum])
            gameBoards[boardNum][7*row + col] = 1
            gameBoards[2][7*row + col] = 0

            if not(row == 5):  # allows a piece to be placed on the slot above where we just placed a piece
                gameBoards[2][7*(row+1) + col] = 1
            return


def validMove(col, row):
    if gameBoards[2][7*row + col] == 0:  # checks to see if a slot defined by (row, col) is empty
        return False
    return True


def winningMove(col, row, board):

    # check for horizontal win

    # check for vertical win

    # check for diagonal win

    return False


def main():
    piecesToPlace = rand.randrange(1, 42)
    print("Placing " + str(piecesToPlace) + " pieces")
    for i in range(0, piecesToPlace):
        placePiece(i % 2)  # alternating between player1 and player2

    toString()
    print(" ")
    print("Printing board for red player")
    redToString()

    print(" ")
    print("Printing board for blue player")
    blueToString()

    return gameBoards[0], gameBoards[1], gameBoards[2]


# The rest of these functions are printing out board
def toString():
    for row in range(5, -1, -1):
        print("Row" + str(row+1), end=" ")
        for col in range(0, 7):
            print(" " + numToLetter(gameBoards[0][7*row + col] + 2*gameBoards[1][7*row + col]), end=" ")
        print(" ")


def redToString():
    for row in range(5, -1, -1):
        print("Row" + str(row+1), end=" ")
        for col in range(0, 7):
            print(" " + numToLetter(gameBoards[0][7*row + col]), end=" ")
        print(" ")


def blueToString():
    for row in range(5, -1, -1):
        print("Row" + str(row+1), end=" ")
        for col in range(0, 7):
            print(" " + numToLetter(2*gameBoards[1][7*row + col]), end=" ")
        print(" ")


def numToLetter(num):
    if num == 0:
        return "-"
    if num == 1:
        return "R"
    if num == 2:
        return "B"


main()
