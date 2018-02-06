"""
This file contains the `Board` class, which implements the rules for the
game Isolation as described in lecture, modified so that the players move
like knights in chess rather than queens.

You MAY use and modify this class, however ALL function signatures must
remain compatible with the defaults provided, and none of your changes will
be available to project reviewers.
"""
import random
import timeit
from copy import copy
import numpy as np
TIME_LIMIT_MILLIS = 150
from zlib import adler32


class Board(object):
    """Implement a model for the game Isolation assuming each player moves like
    a knight in chess.

    Parameters
    ----------
    player_1 : object
        An object with a get_move() function. This is the only function
        directly called by the Board class for each player.

    player_2 : object
        An object with a get_move() function. This is the only function
        directly called by the Board class for each player.

    width : int (optional)
        The number of columns that the board should have.

    height : int (optional)
        The number of rows that the board should have.
    """
    BLANK = 0
    NOT_MOVED = None

    def __init__(self, player_1, player_2, width=7, height=7,
                 data_dir='/home/psavine/data/isolation/games2/'):
        self.width = width
        self.height = height
        self.move_count = 0
        self._player_1 = player_1
        self.history = []
        self._player_2 = player_2
        self._active_player = player_1
        self._inactive_player = player_2
        self.datadir = data_dir

        # The last 3 entries of the board state includes initiative (0 for
        # player 1, 1 for player 2) player 2 last move, and player 1 last move
        self._board_state = [Board.BLANK] * (width * height + 3)
        self._board_state[-1] = Board.NOT_MOVED
        self._board_state[-2] = Board.NOT_MOVED

    def hash(self):
        return str(self._board_state).__hash__()

    def get_player_N(self, n):
        if n == 1:
            return self._player_1
        else:
            return self._player_2

    @property
    def active_player(self):
        """The object registered as the player holding initiative in the
            current game state.
            """
        return self._active_player

    @property
    def inactive_player(self):
        """The object registered as the player in waiting for the current
            game state.
            """
        return self._inactive_player

    @property
    def terminal(self):
        if len(self.get_legal_moves()) > 0:
            return True
        else:
            return False

    def setstate_NQA(self, state):
        new_board = self.copy()
        new_board._board_state = state
        return new_board

    def get_opponent(self, player):
        """Return the opponent of the supplied player.

            Parameters
            ----------
            player : object
                An object registered as a player in the current game. Raises an
                error if the supplied object is not registered as a player in
                this game.

            Returns
            -------
            object
                The opponent of the input player object.
            """
        if player == self._active_player:
            return self._inactive_player
        elif player == self._inactive_player:
            return self._active_player
        raise RuntimeError("`player` must be an object registered as a player in the current game.")

    def copy(self):
        """ Return a deep copy of the current board. """
        new_board = Board(self._player_1, self._player_2, width=self.width, height=self.height, data_dir=self.datadir)
        new_board.move_count = self.move_count
        new_board.history = copy(self.history) 
        new_board._active_player = self._active_player
        new_board._inactive_player = self._inactive_player
        new_board._board_state = copy(self._board_state)
        return new_board

    def forecast_move(self, move):
        """Return a deep copy of the current game with an input move applied to
            advance the game one ply.

            Parameters
            ----------
            move : (int, int)
                A coordinate pair (row, column) indicating the next position for
                the active player on the board.

            Returns
            -------
            isolation.Board
                A deep copy of the board with the input move applied.
            """
        new_board = self.copy()
        new_board.apply_move(move)
        return new_board

    def move_is_legal(self, move):
        """Test whether a move is legal in the current game state.

            Parameters
            ----------
            move : (int, int)
                A coordinate pair (row, column) indicating the next position for
                the active player on the board.

            Returns
            -------
            bool
                Returns True if the move is legal, False otherwise
        """
        idx = move[0] + move[1] * self.height
        return (0 <= move[0] < self.height and 0 <= move[1] < self.width and
                self._board_state[idx] == Board.BLANK)

    def get_blank_spaces(self):
        """Return a list of the locations that are still available on the board.
        """
        return [(i, j) for j in range(self.width) for i in range(self.height)
                if self._board_state[i + j * self.height] == Board.BLANK]

    def get_player_location(self, player):
        """Find the current location of the specified player on the board.

            Parameters
            ----------
            player : object
                An object registered as a player in the current game.

            Returns
            -------
            (int, int) or None
                The coordinate pair (row, column) of the input player, or None
                if the player has not moved.
            """
        if player == self._player_1:
            if self._board_state[-1] == Board.NOT_MOVED:
                return Board.NOT_MOVED
            idx = self._board_state[-1]
        elif player == self._player_2:
            if self._board_state[-2] == Board.NOT_MOVED:
                return Board.NOT_MOVED
            idx = self._board_state[-2]
        else:
            raise RuntimeError(
                "Invalid player in get_player_location: {}".format(player))
        w = idx // self.height
        h = idx % self.height
        return (h, w)

    def get_legal_moves(self, player=None):
        """Return the list of all legal moves for the specified player.

            Parameters
            ----------
            player : object (optional)
                An object registered as a player in the current game. If None,
                return the legal moves for the active player on the board.

            Returns
            -------
            list<(int, int)>
                The list of coordinate pairs (row, column) of all legal moves
                for the player constrained by the current game state.
        """
        if player is None:
            player = self.active_player
        return self.__get_moves(self.get_player_location(player))

    def apply_move(self, move):
        """Move the active player to a specified location.

            Parameters
            ----------
            move : (int, int)
                A coordinate pair (row, column) indicating the next position for
                the active player on the board.
        """
        idx = move[0] + move[1] * self.height
        last_move_idx = int(self.active_player == self._player_2) + 1
        self._board_state[-last_move_idx] = idx
        self._board_state[idx] = 1
        self._board_state[-3] ^= 1
        self._active_player, self._inactive_player = self._inactive_player, self._active_player
        self.history.append(move)
        self.move_count += 1

    def is_winner(self, player):
        """ Test whether the specified player has won the game. """
        return player == self._inactive_player and not self.get_legal_moves(self._active_player)

    def is_loser(self, player):
        """ Test whether the specified player has lost the game. """
        return player == self._active_player and not self.get_legal_moves(self._active_player)

    def utility(self, player):
        """Returns the utility of the current game state from the perspective
            of the specified player.

                        /  +infinity,   "player" wins
            utility =  |   -infinity,   "player" loses
                        \          0,    otherwise

            Parameters
            ----------
            player : object (optional)
                An object registered as a player in the current game. If None,
                return the utility for the active player on the board.

            Returns
            ----------
            float
                The utility value of the current game state for the specified
                player. The game has a utility of +inf if the player has won,
                a value of -inf if the player has lost, and a value of 0
                otherwise.
            """
        if not self.get_legal_moves(self._active_player):

            if player == self._inactive_player:
                return float("inf")

            if player == self._active_player:
                return float("-inf")

        return 0.

    def __get_moves(self, loc):
        """Generate the list of possible moves for an L-shaped motion (like a
        knight in chess).
        """
        if loc == Board.NOT_MOVED:
            return self.get_blank_spaces()

        r, c = loc
        directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                      (1, -2), (1, 2), (2, -1), (2, 1)]
        valid_moves = [(r + dr, c + dc) for dr, dc in directions
                       if self.move_is_legal((r + dr, c + dc))]
        random.shuffle(valid_moves)
        return valid_moves

    def to_string(self, symbols=['1', '2']):
        """Generate a string representation of the current game state, marking
            the location of each player and indicating which cells have been
            blocked, and which remain open.
        """
        p1_loc = self._board_state[-1]
        p2_loc = self._board_state[-2]

        col_margin = len(str(self.height - 1)) + 1
        prefix = "{:<" + "{}".format(col_margin) + "}"
        offset = " " * (col_margin + 3)
        out = offset + '   '.join(map(str, range(self.width))) + '\n\r'
        for i in range(self.height):
            out += prefix.format(i) + ' | '
            for j in range(self.width):
                idx = i + j * self.height
                if not self._board_state[idx]:
                    out += ' '
                elif p1_loc == idx:
                    out += symbols[0]
                elif p2_loc == idx:
                    out += symbols[1]
                else:
                    out += '-'
                out += ' | '
            out += '\n\r'

        return out

    def save_self(self, start, history, res_type):
        end = timeit.timeit()
        duration = end - start
        np.savez_compressed(self.datadir + str(start) + '.npz',
                            size=[self.width, self.height],
                            result = res_type,
                            state = self._board_state,
                            hist=history,
                            duration=duration,
                            winner=str(self._inactive_player))

    def play(self, time_limit=TIME_LIMIT_MILLIS, name='', sdir=None):
        """Execute a match between the players by alternately soliciting them
            to select a move and applying it in the game.

            Parameters
            ----------
            time_limit : numeric (optional)
                The maximum number of milliseconds to allow before timeout
                during each turn.

            Returns
            ----------
            (player, list<[(int, int),]>, str)
                Return multiple including the winning player, the complete game
                move history, and a string indicating the reason for losing
                (e.g., timeout or invalid move).
        """
        move_history = []
        start = timeit.timeit()

        time_millis = lambda: 1000 * timeit.default_timer()

        while True:

            legal_player_moves = self.get_legal_moves()
            game_copy = self.copy()

            move_start = time_millis()
            time_left = lambda: time_limit - (time_millis() - move_start)
            curr_move = self._active_player.get_move(game_copy, time_left)
            move_end = time_left()

            if curr_move is None:
                curr_move = Board.NOT_MOVED

            if move_end < 0:
                #print("timeout", len(move_history))
                self.save_self(start, move_history, "timeout")
                return self._inactive_player, move_history, "timeout"

            if curr_move not in legal_player_moves:
                if len(legal_player_moves) > 0:
                    self.save_self(start, move_history, "forfeit")
                    #print("forfiet", len(move_history))
                    return self._inactive_player, move_history, "forfeit"
                #print("illegal move", len(move_history))
                self.save_self(start, move_history, "illegal move")
                return self._inactive_player, move_history, "illegal move"

            move_history.append(list(curr_move))

            self.apply_move(curr_move)

    def play_sl(self, time_limit=TIME_LIMIT_MILLIS, sl_player=2, sl_mult=10):
        """Execute a match between the players by alternately soliciting them
            to select a move and applying it in the game.

        """
        move_history = []
        start = timeit.timeit()

        time_millis = lambda: 1000 * timeit.default_timer()
        player = -1
        while True:
            #swithc to 1
            player =~ player
            legal_player_moves = self.get_legal_moves()
            game_copy = self.copy()
            move_start = time_millis()

            if player == 0:
                time_left = lambda : time_limit - (time_millis() - move_start)
            else:
                time_left = lambda : sl_mult * time_limit - (time_millis() - move_start)

            #print("player", self._active_player, player)
            curr_move = self._active_player.get_move(self, time_left)
            # print("moved:", curr_move)
            move_end = time_left()

            if curr_move is None:
                curr_move = Board.NOT_MOVED

            if move_end < 0:
                self.save_self(start, move_history, "timeout")
                return self._inactive_player, move_history, "timeout"

            if curr_move not in legal_player_moves:
                if len(legal_player_moves) > 0:
                    self.save_self(start, move_history, "forfeit")
                    return self._inactive_player, move_history, "forfeit"
                self.save_self(start, move_history, "illegal move")
                return self._inactive_player, move_history, "illegal move"

            move_history.append(list(curr_move))

            self.apply_move(curr_move)


class LTBoard:
    """
    state is represented as [0, 0, 1, 2, ... 1]
    state[-1]             = active_player E{1, 2}
    available squares     = 0
    not available squares = -1
    player_loc            = 1 || 2

    """
    BLANK = 0
    NOT_MOVED = None
    MOVED = -1

    def __init__(self, width=7, height=7, hist=None, state=None):
        self.width, self.height = width, height
        self.history = [] if hist is None else hist
        self._player_1, self._player_2 = 1, 2
        self._board_state = [0] * (self.width * self.height) + [1] if state is None else state

    def copy_move(self, idx):
        new = self.copy()
        return new.apply_move(idx)

    def apply_move(self, idx):
        """
            [0, 0, 0, 0, 1]
            [1, 0, 0, 0, 2]
            [1, 0, 2, 0, 1]  928 7791919
            [-1, 1, 2, 0, 1]
            """
        # self.history.append(idx)
        # new_state = self.smove(self._board_state, idx)
        new_history = self.history.copy()
        new_state = self._board_state.copy()

        # game_state = current_state.copy()
        active = self.state[-1]
        inactive = 1 if active == 2 else 2
        state = self.state[:-1]
        if active in state:
            last_move_idx = state.index(active)
            new_state[last_move_idx] = -1
        new_state[idx] = active
        new_state[-1] = inactive
        new_history.append(idx)

        return LTBoard(width=self.width, height=self.height,
                       hist=new_history, state=new_state)

    @staticmethod
    def smove(current_state: list, action_idx: int) -> list:
        """
        static version of 'move'

        :param current_state:
        :param action_idx:
        :return: new state (list)
        """
        game_state = current_state.copy()
        active = game_state[-1]
        inactive = 1 if active == 2 else 2
        state = game_state[:-1]
        if active in state:
            last_move_idx = state.index(active)
            game_state[last_move_idx] = -1
        game_state[action_idx] = active
        game_state[-1] = inactive
        return game_state

    def forward(self, idx):
        return self.apply_move(idx)

    def backward(self):
        if len(self.history) == 0:
            return self

        new_history = self.history.copy()
        game_state = self._board_state.copy()

        inactive= game_state[-1]
        active = 1 if inactive == 2 else 2
        last_move_idx = self.history[-1]

        game_state[last_move_idx] = 0
        game_state[-1] = active
        return LTBoard(state=game_state, hist=new_history[:-1])

    def to_feature(self):
        """
            Initially my gpu was taking lik 20 secs to do this,
            so I made updates to stack as fast as i could...
            turns out this was pointless.
            update self fast index to create game inputs:

        :return
            0 L_pos_self [0 -> 1], prev [1 -> 0]
            1 L_pos_opp  [0 -> 1], prev [1 -> 0]
            2 L_open     [1 -> 0], [1 -> 0]
            3 L_ones
            4 L_legal    clear->0, get->1
            5 L_closed   [0->1],  [0->1]
            6 L_zeros
            """
        x, y = self.height, self.width
        L_pos_self, L_zeros, L_pos_opp, L_legal, L_open, = \
            [np.zeros((x, y), dtype=int) for _ in range(5)]
        L_ones, L_closed = [np.ones((x, y), dtype=int) for _ in range(2)]

        for blank_idx in self.get_legal_moves():
            row, col = self.idx_to_2d(blank_idx)
            L_legal[row][col] = 1

        for blank_idx in self.get_blank_spaces():
            row, col = self.idx_to_2d(blank_idx)
            L_open[row][col] = 1

        L_pos_self = self.player_plane(self.active_player)
        L_pos_opp = self.player_plane(self.inactive_player)

        L_closed = L_closed - L_open - L_pos_opp - L_pos_self
        return np.stack([L_pos_self, L_pos_opp, L_open, L_ones, L_legal, L_closed, L_zeros], axis=0)

    def player_plane(self, player, plane=None):
        if plane is None:
            plane = np.zeros((self.height, self.width), dtype=int)
        loc = self.get_player_location(player)
        if loc is not None:
            s_r, s_c = self.idx_to_2d(loc)
            plane[s_r][s_c] = 1
        return plane

    def to_feat_v2(self, n_hist=2):
        """
        :return
            0 [0 L_pos_self [0 -> 1], prev [1 -> 0]
               5 L_closed   [0->1],  [0->1] ]
            1 [0 L_pos_self [0 -> 1], prev [1 -> 0]
               5 L_closed   [0->1],  [0->1] ]
            2 L_pos_opp  [0 -> 1], prev [1 -> 0]
        """
        x, y = self.height, self.width
        active_plane = np.ones((x, y)) if self.active_player == 2 else np.zeros((x,y))
        own_hist = []
        opp_hist = []

        for blank_idx in self.get_legal_moves():
            pass

        for _ in range(n_hist):
            # prev_state = self.backward()
            pass

        pass

    @property
    def state(self):
        return self._board_state

    @property
    def move_count(self):
        return len([x for x in self._board_state[:-1] if x != 0])

    @property
    def active_player(self):
        return self._board_state[-1]

    @property
    def inactive_player(self):
        return 1 if self._board_state[-1] == 2 else 2

    @property
    def terminal(self):
        if len(self.get_legal_moves()) == 0:
            return True
        else:
            return False

    @property
    def hash(self):
        return self.__hash__()

    def get_opponent(self, player):
        """ Done """
        return 2 if player == 1 else 1

    def copy(self):
        """ Return a deep copy of the current board. """
        new_board = LTBoard(width=self.width, height=self.height)
        new_board.history = copy(self.history)
        new_board._board_state = copy(self._board_state)
        return new_board

    def move_is_legal(self, move):
        r, c = move
        idx = r * self.height + c
        if idx >= self.width * self.height:
            return False
        current = self._board_state[idx]
        return (0 <= r < self.height and 0 <= c < self.width and current == 0)

    def get_blank_spaces(self):
        """ Done """
        return [i for i, x in enumerate(self._board_state[:-1]) if x == 0]

    def get_player_location(self, player_idx):
        state = self._board_state[:-1]
        if player_idx in state:
            return state.index(player_idx)
        return None

    def get_legal_moves(self, player=None):
        """Return the list of all legal moves for the specified player."""
        # todo
        if player is None:
            player = self.active_player
        return self.__get_moves(self.get_player_location(player))

    def is_winner(self, player):
        """ Test whether the specified player has won the game. """
        return player == self.inactive_player and not self.get_legal_moves(self.active_player)

    def is_loser(self, player):
        """ Test whether the specified player has lost the game. """
        return player == self.active_player and not self.get_legal_moves(self.active_player)

    def __get_moves(self, idx):
        if idx is None:
            return self.get_blank_spaces()
        r, c = self.idx_to_2d(idx)
        dirs = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        valid_moves = [self.move_to_idx((r + dr, c + dc)) for dr, dc in dirs if self.move_is_legal((r + dr, c + dc))]
        random.shuffle(valid_moves)
        return valid_moves

    def to_string(self, symbols=['1', '2']):
        p1_loc = self._board_state[:-1].index(1) if 1 in self._board_state[:-1] else None
        p2_loc = self._board_state[:-1].index(2) if 2 in self._board_state[:-1] else None

        col_margin = len(str(self.height - 1)) + 1
        prefix = "{:<" + "{}".format(col_margin) + "}"
        offset = " " * (col_margin + 3)
        out = offset + '   '.join(map(str, range(self.width))) + '\n\r'
        for i in range(self.height):
            out += prefix.format(i) + ' | '
            for j in range(self.width):
                idx = i + j * self.height
                if not self._board_state[idx]:
                    out += ' '
                elif p1_loc == idx:
                    out += symbols[0]
                elif p2_loc == idx:
                    out += symbols[1]
                else:
                    out += '-'
                out += ' | '
            out += '\n\r'
        return out

    def idx_to_2d(self, idx):
        r = idx // self.height
        c = idx % self.width
        return r, c

    def move_to_idx(self, move):
        r, c = move
        return r * self.height + c

    def __repr__(self):
        return str(self._board_state)

    def __hash__(self):
        return adler32(str(self._board_state).encode('utf-8'))

    def __eq__(self, other):
        return self.state == other.state


