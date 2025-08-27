from board import Board

class State:
    def __init__(self, board=None, value=0.5, children=None, player="X"):
        self.board = board if board is not None else Board()
        self.children = children if children is not None else {}
        self.value = value
        self.player = player

    def is_terminal(self):
        return self.board.check_winner() is not None

    def get_board(self):
        return self.board

    def hash(self):
        return str(self.board.tobytes()) + self.player

    def __str__(self):
        return f"State(Value: {self.value}, Player: {self.player})\n{self.board}"