from board import Board
from policy import Policy
from tictactoe import TicTacToe

class TicTacToeSelfPlay:
    def __init__(self, policy):
        self.board = Board()
        self.player = "X" # X always starts first
        self.policy = policy

    def switch_player(self, player):
        return "O" if player == "X" else "X"

    def run(self, games=100000):
        while games > 0:
            state = self.board.check_winner()

            if state == "X" or state == "O":
                print(f"Player {state} wins!")
                self.board.reset()
                self.policy.reset()
                self.player = "X"
                games -= 1
                continue
            elif state == "DRAW":
                print("It's a draw!")
                self.board.reset()
                self.policy.reset()
                self.player = "X"
                games -= 1
                continue
            
            row, col = self.policy.get_best_move(self.board, self.player)
            self.board.set(row, col, self.player)
            self.player = self.switch_player(self.player)
        print("Self-play training completed.")

if __name__ == "__main__":
    policy = Policy(alpha=0.9, epsilon=0.2, decay=0.9995, min_epsilon=0.01, self_play=True)
    train = TicTacToeSelfPlay(policy)
    train.run()

    game = TicTacToe(policy)
    game.run()

