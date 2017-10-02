
import random
from game_agent import IsolationPlayer, SearchTimeout

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import aindnn.slnn as inn
import numpy as np

p_inf = float("inf")
n_inf = float("-inf")



def game_to_input(player, game):
    #L_zeros, L_ones, L_pos_self, L_pos_opp, L_legal, L_open, L_closed
    # L_open, L_closed
    x = game.height
    y = game.width

    L_pos_self, L_zeros, L_pos_opp, L_legal, L_open, = [np.zeros((x, y), dtype=int) for _ in range(5)]
    L_ones, L_closed = [np.ones((x, y), dtype=int) for _ in range(2)]

    for row, col in game.get_legal_moves():
        L_legal[row][col] = 1

    for row, col in game.get_blank_spaces():
        L_open[row][col] = 1

    s_r, s_c = game.get_player_location()
    L_pos_self[s_r][s_c] = 1

    o_r, o_c = game.get_player_location(game.get_opponent(player))
    L_pos_opp[o_r][o_c] = 1

    L_closed = L_open - L_pos_opp - L_pos_self

    return np.stack([[L_pos_self, L_pos_opp, L_open, L_ones, L_legal, L_closed, L_zeros]], axis=0)

def fresh_stack(x, y):
    #x = game.height
    #y = game.width
    L_pos_self, L_zeros, L_pos_opp, L_legal, L_closed, = [np.zeros((x, y), dtype=int) for _ in range(5)]
    L_ones, L_open  = [np.ones((x, y), dtype=int) for _ in range(2)]
    return np.stack([L_pos_self, L_pos_opp, L_open, L_ones, L_legal, L_closed, L_zeros], axis=0)

def fresh_stack_gpu(x, y):
    stack = fresh_stack(x, y)
    return torch.from_numpy(stack).float().unsqueeze(0).cuda(0)


def one_hot_move_to_index(game, move_index):
    #neural net returns a one hot vector.
    #transform to (row, col) move
    r = move_index // game.width
    c = move_index - (r * game.width)
    return r, c

class NNPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
        search. You must finish and test this player to make sure it properly uses
        minimax to return a good move before the search time limit expires.
        """
    def __init__(self, net, name='def',replay=False, gpu=True, h=7, w=7):
        super(self.__class__, self).__init__()
        self.model = net
        self.name = name
        # this is an efficienter index of position_stack
        self.positions_stack = None
        # these hold previous moves to add update
        self.last_opp_pos = None
        self.last_self_pos = None
        # for RL training
        self.saved_actions = []
        self.rewards = []
        self.replay = replay
        # to do implement numpy version for tournament
        self.gpu = gpu
        self.illegal_hist = 0

        #initialize empty game stack
        self.positions_stack = fresh_stack_gpu(h, w)
        #warm up the gpu, or else it times out on first move lol
        #print("stack built")
        self.__run_nn()
        #print("warmup complete")


    def get_move(self, game, time_left):
        """ Parameters
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
        #for randomness in training rl training
        if game.move_count < 3:
            return game.get_legal_moves()[0]

        best_move = (-1, -1)
        try:

            return self.get_nn_move_simple(game)

        except SearchTimeout:
            pass
        return best_move

    def update_stack(self, game, moves):
        """
            Initially my gpu was taking lik 20 secs to do this,
            so I made updates to stack as fast as i could...
            turns out this was pointless.
            update self fast index to create game inputs:
                0 L_pos_self [0 -> 1], prev [1 -> 0] xx
                1 L_pos_opp  [0 -> 1], prev [1 -> 0] xx
                2 L_open     [1 -> 0], [1 -> 0] xx
                3 L_ones
                4 L_legal    clear->0, get->1 xx
                5 L_closed   [0->1],  [0->1] xx
                6 L_zeros

        """
        self.positions_stack[0][4][::] = 0           #L_legal
        for row, col in moves:
            self.positions_stack[0][4][row][col] = 1 #L_legal

        pos_self = game.get_player_location(self)
        if pos_self:
            s_r, s_c = pos_self
            self.positions_stack[0][0][::] = 0       #L_pos_self
            self.positions_stack[0][0][s_r][s_c] = 1 #L_pos_self

        if self.last_opp_pos:
            lor, loc = self.last_opp_pos
            self.positions_stack[0][5][lor][loc] = 1 # L_closed
            self.positions_stack[0][2][lor][loc] = 0 # L_open
            self.positions_stack[0][1][lor][loc] = 0 # L_pos_opp

        if self.last_self_pos:
            ro, co = self.last_self_pos
            self.positions_stack[0][5][ro][co] = 1 # L_closed
            self.positions_stack[0][2][ro][co] = 0 #L_open

        pos_op = game.get_player_location(game.get_opponent(self))
        if pos_op:
            self.positions_stack[0][1][pos_op[0]][pos_op[1]] = 1 #L_pos_opp

        self.last_opp_pos = pos_op
        self.last_self_pos = pos_self


    def __run_nn(self):
        "interface to nn - can error handle here when no torch"
        return self.model(Variable(self.positions_stack))


    def get_nn_move_simple(self, game):
        "test method to return a move from NN prediction - no guardrails."
        #save stack for latter
        moves = game.get_legal_moves()
        self.update_stack(game, moves)

        #run neural net
        logits = self.__run_nn()

        if self.replay:
            # if rl training, not using softmax, but 
            # obtaining a stocastic sample ... 
            index = logits.squeeze().multinomial()
        else:
            _, index = logits.squeeze().max(-1)

        #converd to coords
        move = one_hot_move_to_index(game, index.data[0])

        #this is product of trying to figure out RL todo cleanup
        if move in moves:
            self.saved_actions.append(index)
            self.rewards.append(logits)
            return move
        elif not moves:
            return (-1, -1)
        else:
            if not self.replay:
                self.saved_actions.append(index)
                self.rewards.append(logits)
                # we can just ignore the choice
                # This was not used for SL training.
                # was used in RL to speed up stuff.
                # if there is a win, I get some loss from it.
                self.illegal_hist += 1
                return moves[0]
            else:
                # if sampling from multinomial dist,
                # it should return the failed move
                # and let reinforce handle it
                self.saved_actions.append(index)
                self.rewards.append(logits)
                return move


"""
Namespace(act='train', batch_size=16, desc='nn_test', env='home', epochs=10, lr=0.0005, momentum=0.9, valid_size=100)
positions: 1642346 
positions: 33518 
Epoch: 1/10, step: 19, training_loss: 3.85930
Epoch: 1/10, step: 39, training_loss: 3.84488
Epoch: 1/10, step: 59, training_loss: 3.92691
Epoch: 1/10, step: 79, training_loss: 3.89353
Epoch: 1/10, step: 99, training_loss: 3.93251
Epoch: 1/10, step: 119, training_loss: 3.88021
Epoch: 1/10, step: 139, training_loss: 3.89227
Epoch: 1/10, step: 159, training_loss: 3.85857
Epoch: 1/10, step: 179, training_loss: 3.89479
Epoch: 1/10, step: 199, training_loss: 3.91684
Epoch: 1/10, step: 219, training_loss: 3.81673
Epoch: 1/10, step: 239, training_loss: 3.84902
Epoch: 1/10, step: 259, training_loss: 3.89781
Epoch: 1/10, step: 279, training_loss: 3.88619

"""























