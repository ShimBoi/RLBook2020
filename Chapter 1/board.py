import numpy as np

class Board:
    def __init__(self, board=None):
        if board is not None:
            self.board = board
        else:
            self.board = np.array([[" " for _ in range(3)] for _ in range(3)])

    def check_winner(self):
        for player in ["X", "O"]:
            for i in range(3):
                if all(self.board[i, j] == player for j in range(3)): return player
                if all(self.board[j, i] == player for j in range(3)): return player

            if all(self.board[i, i] == player for i in range(3)): return player
            if all(self.board[i, 2-i] == player for i in range(3)): return player

        # check for draws
        draw = all(cell != " " for cell in self.board.flatten())
        if draw:
            return "DRAW"

        return None

    def reset(self):
        self.board = np.array([[" " for _ in range(3)] for _ in range(3)])

    def copy(self):
        return Board(board=self.board.copy())

    def get(self, row, col):
        return self.board[row, col]

    def set(self, row, col, value):
        self.board[row, col] = value

    def tobytes(self):
        return self.board.tobytes()

    def __str__(self):
        lines = []
        for row in self.board:
            line = " | ".join(row)
            lines.append(line)
        board = "\n---------\n".join(lines)
        board += "\n##########\n"
        return board