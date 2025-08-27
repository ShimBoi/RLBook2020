from board import Board
from policy import Policy

class TicTacToe:
    def __init__(self, policy):
        self.board = Board()
        self.player = "X" # X always starts first
        self.policy = policy

    def switch_player(self, player):
        return "O" if player == "X" else "X"

    def run(self):
        while True:
            print(self.board)
            state = self.board.check_winner()

            if state == "X" or state == "O":
                print(f"Player {state} wins!")
                self.board.reset()
                self.policy.reset()
                self.player = "X"
                continue
            elif state == "DRAW":
                print("It's a draw!")
                self.board.reset()
                self.policy.reset()
                self.player = "X"
                continue

            if self.player == "X":
                row, col = self.policy.get_best_move(self.board, "X")
                self.board.set(row, col, "X")
            else:
                try:
                    parts = input("Enter your move (row and column): ").split()
                    if len(parts) != 2:
                        print("Please enter two numbers for row and column.")
                        continue
                    row, col = map(int, parts)
                except ValueError:
                    print("Invalid input! Please try again.")
                    continue

                if row >= 3 or row < 0 or col >= 3 or col < 0 or self.board.get(row, col) != " ":
                    print("Invalid move! Try again.")
                    continue
                prev_board = self.board.copy()
                self.board.set(row, col, "O")

                self.policy.update_current_state(prev_board, self.board, "O")

            self.player = self.switch_player(self.player)

if __name__ == "__main__":
    policy = Policy(alpha=0.8, epsilon=0.3)
    game = TicTacToe(policy)
    game.run()