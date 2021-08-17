from tensorflow import keras
import GameBoard as gb
import time
import numpy as np
model = keras.models.load_model('my_model_2')

b = gb.GameBoard(0)
b.make_move(1)
b.make_move(7)
b.make_move(2)
b.make_move(7)
b.make_move(3)
print(b)
print(b.desired_probabilities(model))

board_arr = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # board for player 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # board for player 2
           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0]]  # board of currently placeable positions

board = gb.GameBoard(game_boards=board_arr)

player = int(input("RED (1) or BLUE (2)?: "))

if player == 1:
    print(board)

if player == 2:
    result = gb.pick_max(
        model.predict([board.get_boards()[0] + board.get_boards()[1] + board.get_boards()[2]])[0])

    print(model.predict([board.get_boards()[0] + board.get_boards()[1] + board.get_boards()[2]])[0])

    board.make_move(result)

    print(board)


while True:

    MOVE_COL = int(input("COL: "))

    board.make_move(MOVE_COL)

    print(board)

    time.sleep(1)

    if gb.is_win(board.get_boards()[1]):
        print("WIN")
        break

    result = gb.pick_max(model.predict([board.get_boards()[0] + board.get_boards()[1] + board.get_boards()[2]])[0])

    print(model.predict([board.get_boards()[0] + board.get_boards()[1] + board.get_boards()[2]])[0])

    board.make_move(result)

    print(board)

    if gb.is_win(board.get_boards()[1]):
        print("LOSS")
        break