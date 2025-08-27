import numpy as np
from state import State
from board import Board

class Policy:
    def __init__(self, alpha=0.5, epsilon=1.0, decay=0.9, min_epsilon=0.01, self_play=False):
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.self_play = self_play
        self.visited = {}
        self.tree = self.generate_tree(State(), "X")
        self.curr = self.tree

        print("Game tree generated with", len(self.visited), "unique states.")

    def debug_print(self, node, depth=0):
        indent = "  " * depth
        print(f"{indent}Value: {node.value}, Children: {len(node.children)}")
        for move, child in node.children.items():
            print(f"{indent}Move: {move}")
            self.debug_print(child, depth + 1)
    
    def generate_tree(self, node, player):
        key = node.hash()
        if key in self.visited:
            return self.visited[key]

        self.visited[key] = node
        node.player = player

        board = node.get_board()
        state = board.check_winner()
        if state == "X":
            node.value = 1.0
            return node 
        elif state == "DRAW":
            node.value = 0.0
            return node
        elif state == "O": 
            node.value = -1.0 if self.self_play else 0.0
            return node

        children = {}
        for i in range(3):
            for j in range(3):
                if board.get(i, j) == " ":
                    new_board = board.copy()
                    new_board.set(i, j, player)
                    next_player = "O" if player == "X" else "X"
                    child_state = State(board=new_board, player=next_player)
                    child_tree = self.generate_tree(child_state, next_player)
                    children[(i, j)] = child_tree
        node.children = children

        return node

    def td_update(self, curr_state_value, next_state_value):
        updated_value = curr_state_value + self.alpha * (next_state_value - curr_state_value)
        return updated_value

    def find_current_state(self, board, player):
        temp_state = State(board=board, player=player)
        target_hash = temp_state.hash()
        
        if target_hash in self.visited:
            return self.visited[target_hash]
        
        raise Exception(f"Current board state not found in tree! Player: {player}\nBoard:\n{board}")

    def update_current_state(self, board, next_board, player):
        state = self.find_current_state(board, player)
        old_value = state.value
        new_state = self.find_current_state(next_board, "O" if player == "X" else "X")
        state.value = self.td_update(state.value, new_state.value)
        self.curr = new_state

        print(f"📈 TD UPDATE: {old_value:.3f} -> {state.value:.3f}")

    def get_best_move(self, board, player):
        state = self.find_current_state(board, player)
        self.curr = state

        if len(state.children) == 0:
            raise Exception("No available moves from current state.", str(state), len(state.children))

        # self.print_move_analysis(state, player)

        if np.random.rand() < self.epsilon:
            random_move = list(state.children.keys())[np.random.randint(len(state.children))]
            print(f"🎲 RANDOM MOVE SELECTED: {random_move} (ε={self.epsilon:.3f})")
            self.curr = state.children[random_move]
            return random_move
        else:
            if player == "X":
                best_value = -1
                best_move = None
                for move, child in state.children.items():
                    if child.value > best_value:
                        best_value = child.value
                        best_move = move
            elif player == "O":
                best_value = float('inf')
                best_move = None
                for move, child in state.children.items():
                    if child.value < best_value:
                        best_value = child.value
                        best_move = move

            print(f"🎯 BEST MOVE SELECTED PLAYER {player}: {best_move} (value: {best_value:.3f})")

            # perform TD update
            next_state = state.children[best_move]
            old_value = state.value
            
            self.update_current_state(board, next_state.board, player)
            self.curr = next_state
            return best_move

    def reset(self):
        self.curr = self.tree
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)