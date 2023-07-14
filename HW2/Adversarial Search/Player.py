from Board import BoardUtility
import random

class Player:
    def __init__(self, player_piece):
        self.piece = player_piece

    def play(self, board):
        return 0


class RandomPlayer(Player):
    def play(self, board):
        return random.choice(BoardUtility.get_valid_locations(board))


class HumanPlayer(Player):
    def play(self, board):
        move = int(input("input the next column index 0 to 8:"))
        return move


class MiniMaxPlayer(Player):
    def __init__(self, player_piece, depth=5):
        super().__init__(player_piece)
        self.depth = depth

    def play(self, board):
        valid_locations = BoardUtility.get_valid_locations(board)
        best_score = -1000000
        move = random.choice(valid_locations)
        for c in valid_locations:
            new_board = board.copy()
            BoardUtility.make_move(new_board, c, self.piece)
            score = self.minimax(new_board, self.depth, -1000000, 1000000, False)
            if score > best_score:
                best_score = score
                move = c
        return move

    def minimax(self, board, depth, alpha, beta, is_max_node):
        if depth == 0 or BoardUtility.is_terminal_state(board):
            return BoardUtility.score_position(board, self.piece)
        score = self.maximize(board, depth, alpha, beta) if is_max_node else self.minimize(board, depth, alpha, beta)
        return score
    
    def maximize(self, board, depth, alpha, beta):
        valid_locations = BoardUtility.get_valid_locations(board)
        best_score = -1000000
        for col in valid_locations:
            b_copy = board.copy()
            BoardUtility.make_move(b_copy, col, self.piece)
            score = self.minimax(b_copy, depth - 1, alpha, beta, False)
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score

    def minimize(self, board, depth, alpha, beta):
        valid_locations = BoardUtility.get_valid_locations(board)
        best_score = 1000000
        for col in valid_locations:
            b_copy = board.copy()
            BoardUtility.make_move(b_copy, col, 3 - self.piece)
            score = self.minimax(b_copy, depth - 1, alpha, beta, True)
            best_score = min(best_score, score)
            beta = min(beta, score)
            if alpha >= beta:
                break
        return best_score

    
class MiniMaxProbPlayer(Player):
    def __init__(self, player_piece, depth=5, prob_stochastic=0.1):
        super().__init__(player_piece)
        self.depth = depth
        self.prob_stochastic = prob_stochastic

    def play(self, board):
        player = MiniMaxPlayer(self.piece, self.depth)
        move = player.play(board)
        if random.random() < self.prob_stochastic:
            valid_locations = BoardUtility.get_valid_locations(board)
            move = random.choice(valid_locations)
        return move