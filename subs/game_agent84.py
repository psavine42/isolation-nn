"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
#import sample_players
#import numpy

##############################################################################
#       UTILS
##############################################################################
"""
#https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
def conv2d(X, W, b, stride=1, padding=1):
    cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dim')

    h_out, w_out = int(h_out), int(w_out)

    X_col = [] #im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = W_col @ X_col + b
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, b, stride, padding, X_col)

    return out, cache


def activation_relu():
    pass

"""
##############################################################################
#       Code for Custom Score Functions
##############################################################################


p_inf = float("inf")
n_inf = float("-inf")


def normalized_move_count(game, player):
    """
    Idea is:
    (moves I have - moves Opp has) /
    """
    if game.is_loser(player):
        return n_inf

    if game.is_winner(player):
        return p_inf

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    blank = len(game.get_blank_spaces())
    #full = game.
    return 1

##############################################################################
#       Base Lesson Code
##############################################################################

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    #def __init__(self):
        #self.move = move
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
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
 
    return float(own_moves - opp_moves) 
    #return sample_players.open_move_score(game, player)

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
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
 
    return float(own_moves - opp_moves) 


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
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    
    return float((h - y)**2 + (w - x)**2)
    #return sample_players.open_move_score(game, player)

#lambda

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
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


##############################################################################
#       Problems code
##############################################################################
unroll_time = 30

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
        search. You must finish and test this player to make sure it properly uses
        minimax to return a good move before the search time limit expires.
        """

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
        #self.unroll_time = 30
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass
            #print("search time out")

        # Return the best move from the last completed search iteration
        return best_move


    def min_(self, game, depth):
        """
            Doc
            """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout(depth)

        if depth == 0:
            return self.score(game, self)

        moves = game.get_legal_moves()
        score = p_inf

        if not moves:
            return score

        for move in moves:
            score = min(score, self.max_(game.forecast_move(move), depth - 1))
        return score

    def max_(self, game, depth):
        """
            Doc
            """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self)

        moves = game.get_legal_moves()
        score = n_inf

        if not moves:
            return score

        for move in moves:
            score = max(score, self.min_(game.forecast_move(move), depth - 1))
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
            score = self.min_(game.forecast_move(move), depth - 1)

            #print("s ", best_score, score, move, best_move, self.time_left())

            if self.time_left() < unroll_time:
                print("timewarning!")
                if not best_move:
                    return move
                return best_move

            if score >= best_score:
                best_score = score
                best_move = move
        #print("--------------------", best_move, score)
        return best_move



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
        search with alpha-beta pruning. You must finish and test this player to
        make sure it returns a good move before the search time limit expires.
        for depth = 0 to ∞ do
        result ← DEPTH-LIMITED-SEARCH(problem,depth)
        if result ≠ cutoff then return result
        """
    # unroll_time = 40
    #self.best_move = (-1, -1)
    #self.best_score = n_inf

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

        #
        self.time_left = time_left
        best_move = (-1, -1)
        best_score = n_inf
        alpha = n_inf
        beta = p_inf
        moves = game.get_legal_moves()
        # print(game.to_string())
        try:
            # return self.alphabeta(game, self.search_depth)

            for depth in range(1, game.width * game.height):
                # move = self.alphabeta(game.copy(), depth)

                #if self.best_score == p_inf:
                    #best_move = move
                move, score = self.__alphabeta(game.copy(), depth, moves)
                print(depth, "-------", move, score, "-", best_move, best_score, "-", self.time_left())
                # print("           ", moves)
                if score == p_inf:
                    return move

                if score == n_inf and move in moves:
                    moves.remove(move)

                if score > best_score:
                    best_score = score
                    best_move = move

        except SearchTimeout:
            #print("search time out", self.best_move)
            return best_move

        print("FF", best_move)
        return best_move


    def terminal(self, moves, depth):
        """Doc
            """
        return any(((depth <= 0), (not moves)))
        # (self.time_left() < self.unroll_time),

    def max_v(self, game, depth, alpha, beta):
        """
            Doc
            """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves()
        if self.terminal(moves, depth):
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
            Doc
            """
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves()
        if self.terminal(moves, depth):
            return self.score(game, self)

        v = p_inf
        for move in moves:
            v = min(v, self.max_v(game.forecast_move(move), depth - 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v


    def alphabeta(self, game, fi, alpha=n_inf, beta=p_inf):
        """#Alphabeta interface"""
        move = (-1,-1)
        moves = game.get_legal_moves()
        try:
            if callable(fi):
                self.time_left = fi
                depth = self.search_depth
                move, _ = self.__alphabeta(game, depth, moves)
            else:
                move, _ = self.__alphabeta(game, fi, moves)
        except SearchTimeout:
            print("abs - alpha-beta-TO")
            return random.random(moves)
        return move


    def __alphabeta(self, game, depth, moves, alpha=n_inf, beta=p_inf):
        """Alphabeta implementation"""
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        best_move = (-1, -1)
        best_score = n_inf
        # moves = game.get_legal_moves()

        for move in moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                return best_move, best_score
            # print(move, depth)
            score = self.min_v(game.forecast_move(move), depth - 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        return best_move, best_score


