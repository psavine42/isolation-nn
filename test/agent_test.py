"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest
from collections import namedtuple
import isolation as iso
from agents import game_agent as ga
import sample_players as sp
from importlib import reload

Player = namedtuple("Player", ["player", "name"])


def print_game(g, w, h, o):
    print("\nWinner: {}\nOutcome: {}".format(w, o))
    print(g.to_string())
    print("Move history:\n{!s}".format(h))
    print(g.move_count)


def play_print(game):
    w, h, o = game.play()
    print_game(game, w, h, o)


def run_agents():
    player1 = ga.AlphaBetaPlayer(score_fn=sp.improved_score)
    player2 = ga.AlphaBetaPlayer(score_fn=sp.improved_score)
    for n in range(1000):
        game = iso.Board(player1, player2)


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(ga)
        self.player1 = "Player1"
        self.player2 = "Player2"
        self.game = iso.Board(self.player1, self.player2)


class MiniMaxTEST(unittest.TestCase):
    def setUp(self):
        reload(ga)
        reload(sp)

    def test1(self):
        player1 = ga.AlphaBetaPlayer(score_fn=sp.improved_score, search_depth=1)
        player2 = ga.MinimaxPlayer(score_fn=sp.center_score)
        game = iso.Board(player1, player2, width=9, height=9)
        game = game.setstate_NQA([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 42])
        time_left = lambda : 1000.0
        print()
        print(game.to_string())
        mv = player1.alphabeta(game, time_left)
        print(mv)
        assert(mv == (7, 2))

    def test2(self):
        player1 = ga.AlphaBetaPlayer(score_fn=sp.open_move_score, search_depth=1)
        player2 = ga.MinimaxPlayer(score_fn=sp.open_move_score, search_depth=1)
        game = iso.Board(player1, player2, width=9, height=9)
        game = game.setstate_NQA([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59, 40])
        time_left = lambda : 10000.0
        mv = player1.alphabeta(game, time_left)
        print("-----------")
        print(mv)
        assert(mv == (7, 2))


class AlphaBetaPlayerTEST(unittest.TestCase):
    def setUp(self):
        reload(ga)
        reload(sp)
        self.num = 10
        self.player2 = ga.MinimaxPlayer(score_fn=sp.improved_score)
        self.player1 = ga.AlphaBetaPlayer(score_fn=sp.improved_score)
        self.game = iso.Board(self.player1, self.player2)

    def test_move(self):
        play_print(self.game)
        #pass

    def test_moves(self):
        ws = []
        for _ in range(self.num):
            game = iso.Board(self.player1, self.player2)


if __name__ == '__main__':
    unittest.main()
