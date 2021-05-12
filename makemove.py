def make_move(board1, board2, column):
    # Take in 2 boards, and the column player 1 wants to place the disc,
    # and return 2 new boards, and a flag (1 for win move, -1 for invalid move, 0 otherwise)
    if 1<=column<=7:
        if level_check(board1, board2, column) == 6:
            return board1, board2, -1
        if level_check(board1, board2, column) < 6:
            board_change(board1,(level_check(board1, board2, column)+1,column))
        if is_win(board1):
            return board1, board2, 1
        else:
            return board1, board2, 0
    else:
        return board1, board2, -1

def board_change(board, position):
    # Place a dic to a position with given board input and tuple for position
    if 1<=position[0]<=6 and 1<=position[1]<=7:
        board[7 * (position[0] - 1) + position[1] - 1] = 1

def board_value(board, position):
    # Take in a board(list) and a tuple (k,s) where k is the level and s is the column,
    # and return the value of the board at that position (-1 if given invalid position)
    if 1 <= position[0] <= 6 and 1 <= position[1] <= 7:
        return board[7 * (position[0] - 1) + position[1] - 1]
    else: return -1

def level_check(board1, board2, column):
    # Take in 2 boards and a column, return the current level of the column (-1 if given invalid column)
    if 1<=column<=7:
        b1 = [board_value(board1, (k, column)) for k in range(6, 0,-1)]
        b2 = [board_value(board2, (k, column)) for k in range(6, 0,-1)]
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
    else: return -1

def has_4_in_a_line(line):
    # Check if there are four consecutive 1's on a line (list)
    for i in range(len(line)-3):
        if line[i] & line[i+1] & line[i+2] & line[i+3]==1:
            return True
    return False

def is_win(board):
    # Check if it is a winning board

    # Check each level
    for k in range(1,7):
        if has_4_in_a_line([board_value(board,(k,s)) for s in range(1,8)]):
            return True
    # Check each column
    for s in range(1,8):
        if has_4_in_a_line([board_value(board, (k,s)) for k in range(1,7)]):
            return True
    # Check diagonals
    for i in range(1,4):
        if has_4_in_a_line([board_value(board, (i+j,1+j)) for j in range(7-i)]):
            return True
        if has_4_in_a_line([board_value(board, (1+j,1+i+j)) for j in range(7-i)]):
            return True
        if has_4_in_a_line([board_value(board, (i+j,7-j)) for j in range(7-i)]):
            return True
        if has_4_in_a_line([board_value(board, (1+j,7-i-j)) for j in range(7-i)]):
            return True
    return False

def main():
    board2 = [0, 1, 0, 1, 1, 0, 0,
              1, 0, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 0, 0, 0,
              0, 0, 1, 0, 0, 0, 0,
              0, 0, 0, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0]
    board1 = [0, 0, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 0, 0, 0,
              1, 0, 0, 0, 0, 0, 0,
              0, 1, 1, 0, 0, 0, 0,
              0, 0, 1, 0, 0, 0, 0,
              0, 0, 0, 0, 1, 0, 0]
    print(make_move(board1,board2,8))

if __name__ == "__main__":
    main()