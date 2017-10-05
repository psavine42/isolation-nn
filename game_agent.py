"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
#import sample_players
import numpy as np



##############################################################################
#       Code for Custom Score Functions
##############################################################################


p_inf = float("inf")
n_inf = float("-inf")


def get_locations_util(game, player):
    "return self and player positions"
    return game.get_player_location(player), game.get_player_location(game.get_opponent(player))


def get_moves_util(game, player):
    "return self and player positions"
    return game.get_legal_moves(player), game.get_legal_moves(game.get_opponent(player))

def remaining_moves(game):
    return game.width * game.height - game.move_count

def num_in_corner(moves):
    pass

def edge_and_corner(game, moves):
    "utility function to get edge and corner moves"
    counter = 0
    for r, c in moves:
        if r == 0: counter += 1
        if c == 0: counter += 1
        if r == game.height: counter += 1
        if c == game.width: counter += 1
    return counter

def valid_moves_2n(game, player):
    """a fast way to approximate a game two moves deep
        not precise because a cell may be blocked but close enough
    """
    r, c = game.get_player_location(player)

    dirs_base = [(-1, -1), (-1, 1), (1, -1), (1, 1),
                 (3, 3), (3, -3), (-3, 3), (-3, -3),
                 (0, 2), (0, -2), (2, 0), (-2, 0),
                 (4, 0), (-4, 0), (0, 4), (0, -4),
                 (3, 1), (-3, 1), (1, 3), (1, -3),
                 (-1, -3), (-1, 3), (3, -1), (-3, -1),
                 (4, 2), (-4, 2), (2, 4), (2, -4),
                 (-4, -2), (4, -2), (-2, 4), (-2, -4)]

    valid_moves = [(r + dr, c + dc) for dr, dc in dirs_base if game.move_is_legal((r + dr, c + dc))]
    random.shuffle(valid_moves)
    return valid_moves


def difference_in_moves2d(game, player, m_weight, o_weight):

    own_moves = len(valid_moves_2n(game, player))
    opp_moves = len(valid_moves_2n(game, game.get_opponent(player)))
    if opp_moves == 0:
        return n_inf
    if own_moves == 0:
        return p_inf
    return float(m_weight * own_moves - o_weight * opp_moves)


def difference_in_moves1d(game, player, self_weight=1, opp_weight=1):
    "simple player moves vs opponent moves with some weights "
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(self_weight * own_moves - opp_weight * opp_moves)

def m_distance_helper(cell1, cell2):
    "manhattan distance between two cells"
    return float(abs(cell1[0]-cell2[0]) +  abs(cell1[1]-cell2[1]))

def num_common_moves(moves1, moves2):
    "intersection of any two sets of moves"
    return len(set(moves1).intersection(set(moves2)))


def normalized_move_count(game, player):
    return float(difference_in_moves1d(game, player, 1, 1) / max(remaining_moves(game), 1))


def diff_in_moves_normby_distance(game, player, self_weight=1, opp_weight=1):
    """ difference in own moves and opponent's moves as weighted and 
        porporitonal to distance of self and opponent.
         (this is toggled to inverse proportional in main )
    """

    own_moves, opp_moves = get_moves_util(game, player)
    own_pos, opp_pos = get_locations_util(game, player)
    distance = m_distance_helper(own_pos, opp_pos)
    diff_in_moves = self_weight *len(own_moves) - opp_weight * len(opp_moves)
    return diff_in_moves * distance


def diff_in_moves_with_penalty(game, player, self_weight=1, opp_weight=1):
    " "

    own_moves, opp_moves = get_moves_util(game, player)
    own_pos, opp_pos = get_locations_util(game, player)

    own_penalty = edge_and_corner(game, own_moves)
    opp_penalty = edge_and_corner(game, own_moves)
    distance = m_distance_helper(own_pos, opp_pos)

    own_final = self_weight * (len(own_moves) - 0.5 * own_penalty)
    opp_final = opp_weight * (len(opp_moves) - 0.5 * opp_penalty)

    return (own_final - opp_final) * distance



##############################################################################
#       Base Lesson Code
##############################################################################

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
        of the given player.

        This should be the best heuristic function for your project submission.

        Note: this function should be called from within a Player instance as
        `self.score()` -- you should not need to call this function directly.

        Parameters
            ----------
            game : `isolation.Board`
                An instance of `isolation.Board` encoding the current state of the
                game (e.g., player locations and blocked cells).
            player : object
                A player instance in the current game (i.e., an object corresponding to
                one of the player objects `game.__player_1__` or `game.__player_2__`.)
        Returns
        -------
        float
        The heuristic value of the current game state to the specified player.
     """
    if game.is_loser(player):
        return n_inf
    if game.is_winner(player):
        return p_inf
    return diff_in_moves_normby_distance(game, player, 2, 1)



def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
        of the given player.

        Note: this function should be called from within a Player instance as
        `self.score()` -- you should not need to call this function directly.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        player : object
            A player instance in the current game (i.e., an object corresponding to
            one of the player objects `game.__player_1__` or `game.__player_2__`.)

        Returns
        -------
        float
            The heuristic value of the current game state to the specified player.
     """
    if game.is_loser(player):
        return n_inf
    if game.is_winner(player):
        return p_inf
    return diff_in_moves_with_penalty(game, player, 2, 1)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
        of the given player.

        Note: this function should be called from within a Player instance as
        `self.score()` -- you should not need to call this function directly.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        player : object
            A player instance in the current game (i.e., an object corresponding to
            one of the player objects `game.__player_1__` or `game.__player_2__`.)

        Returns
        -------
        float
            The heuristic value of the current game state to the specified player.
     """
    if game.is_loser(player):
        return n_inf
    if game.is_winner(player):
        return p_inf
    return  difference_in_moves2d(game, player, 2, 1)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
        constructed or tested directly.

        ********************  DO NOT MODIFY THIS CLASS  ********************

            Parameters
            ----------
            search_depth : int (optional)
                A strictly positive integer (i.e., 1, 2, 3,...) for the number of
                layers in the game tree to explore for fixed-depth search. (i.e., a
                depth of one (1) would only explore the immediate sucessors of the
                current state.)

            score_fn : callable (optional)
                A function to use for heuristic evaluation of game states.

            timeout : float (optional)
                Time remaining (in milliseconds) when search is aborted. Should be a
                positive value large enough to allow the function to return before the
                timer expires.
        """
    def __init__(self, search_depth=3, score_fn=custom_score, random_start=False, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.random_start = random_start
        self.num_calcs = 0
        self.TIMER_THRESHOLD = timeout


##############################################################################
#       Problems code
##############################################################################
unroll_time = 30

class MinimaxPlayer(IsolationPlayer):

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
            result before the time limit expires.

            **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

            For fixed-depth search, this function simply wraps the call to the
            minimax method, but this method provides a common interface for all
            Isolation agents, and you will replace it in the AlphaBetaPlayer with
            iterative deepening search.

            Parameters
                ----------
                game : `isolation.Board`
                    An instance of `isolation.Board` encoding the current state of the
                    game (e.g., player locations and blocked cells).

                time_left : callable
                    A function that returns the number of milliseconds left in the
                    current turn. Returning with any less than 0 ms remaining forfeits
                    the game.

            Returns
                -------
                (int, int)
                    Board coordinates corresponding to a legal move; may return
                    (-1, -1) if there are no available legal moves.
            """
        self.time_left = time_left
        if self.random_start:
            return random.choice(game.get_legal_moves())
        best_move = (-1, -1)
        try:
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass
        return best_move


    def min_v(self, game, depth):
        """
            Doc
            """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout(depth)

        if depth == 0:
            self.num_calcs += 1
            return self.score(game, self)

        moves = game.get_legal_moves()
        score = p_inf

        if not moves:
            return score

        for move in moves:
            score = min(score, self.max_v(game.forecast_move(move), depth - 1))
        return score

    def max_v(self, game, depth):
        """
            Doc
            """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            self.num_calcs += 1
            return self.score(game, self)
        moves = game.get_legal_moves()
        score = n_inf
        if not moves:
            return score

        for move in moves:
            score = max(score, self.min_v(game.forecast_move(move), depth - 1))
        return score


    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
            the lectures.

            This should be a modified version of MINIMAX-DECISION in the AIMA text.
            https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

            **********************************************************************
                You MAY add additional methods to this class, or define helper
                    functions to implement the required functionality.
            **********************************************************************

            Parameters
            ----------
            game : isolation.Board
                An instance of the Isolation game `Board` class representing the
                current game state

            depth : int
                Depth is an integer representing the maximum number of plies to
                search in the game tree before aborting

            Returns
            -------
            (int, int)
                The board coordinates of the best move found in the current search;
                (-1, -1) if there are no legal moves

            Notes
            -----
                (1) You MUST use the `self.score()` method for board evaluation
                    to pass the project tests; you cannot call any other evaluation
                    function directly.

                (2) If you use any helper functions (e.g., as shown in the AIMA
                    pseudocode) then you must copy the timer check into the top of
                    each helper function or else your agent will timeout during
                    testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = n_inf
        best_move = None
        for move in game.get_legal_moves():
            score = self.min_v(game.forecast_move(move), depth - 1)
            if self.time_left() < unroll_time:
                if not best_move:
                    return move
                return best_move
            if score >= best_score:
                best_score = score
                best_move = move
        return best_move



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
        search with alpha-beta pruning. You must finish and test this player to
        make sure it returns a good move before the search time limit expires.
        for depth = 0 to ∞ do
        result ← DEPTH-LIMITED-SEARCH(problem,depth)
            if result ≠ cutoff then return result
        """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
            result before the time limit expires.

            Modify the get_move() method from the MinimaxPlayer class to implement
            iterative deepening search instead of fixed-depth search.

            **********************************************************************
            NOTE: If time_left() < 0 when this function returns, the agent will
                forfeit the game due to timeout. You must return _before_ the
                timer reaches 0.
            **********************************************************************

            Parameters
            ----------
            game : `isolation.Board`
                An instance of `isolation.Board` encoding the current state of the
                game (e.g., player locations and blocked cells).

            time_left : callable
                A function that returns the number of milliseconds left in the
                current turn. Returning with any less than 0 ms remaining forfeits
                the game.

            Returns
            -------
            (int, int)
                Board coordinates corresponding to a legal move; may return
                (-1, -1) if there are no available legal moves.
            """
        self.time_left = time_left
        if self.random_start:
            return random.choice(game.get_legal_moves())
        self.best_move = (-1, -1)
        self.best_score = n_inf

        try:
            for depth in range(1, game.width * game.height):
                move = self.alphabeta(game.copy(), depth)
                if move == (-1, -1):
                    # at depth, it is found the best move is losing.
                    # return best thing so far and hope he doesnt the win
                    return self.best_move
                self.best_move = move

        except SearchTimeout:
            #print("AB timewarning")
            return self.best_move
        return self.best_move


    def terminal(self, moves, depth):
        """terminal search function
            """
        return any(((depth <= 0), (not moves)))

    def max_v(self, game, depth, alpha, beta):
        """
            max value function
            """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        moves = game.get_legal_moves()
        if self.terminal(moves, depth):
            self.num_calcs += 1
            return self.score(game, self)

        v = n_inf
        for move in moves:
            v = max(v, self.min_v(game.forecast_move(move), depth - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_v(self, game, depth, alpha, beta):
        """
            min value function
            """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves()
        if self.terminal(moves, depth):
            self.num_calcs += 1
            return self.score(game, self)

        v = p_inf
        for move in moves:
            v = min(v, self.max_v(game.forecast_move(move), depth - 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def alphabeta(self, game, depth, alpha=n_inf, beta=p_inf):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        best_move = (-1, -1)
        best_score = n_inf
        moves = game.get_legal_moves()

        for move in moves:
            score = self.min_v(game.forecast_move(move), depth - 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        return best_move
        # return best_move #, best_score



##############################################################################
#   TODO - Finish neural net - shamelessly lifted from  
#   https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
#   All this needs to do is load a trained network and run forward passes
##############################################################################

class NPNet():
    def __init__(self, location):
        self.location = location
        self.weight_dict = {}
        pass

    def load_pkl(self):
        pass


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def conv2d(X, W, b, stride=1, padding=1):
    #cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dim')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = W_col @ X_col + b
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    #cache = (X, W, b, stride, padding, X_col)

    return out

def fully_connected(X, W, b):
    out = X @ W + b
    return out


def activation_relu(X):
    return np.maximum(X, 0)