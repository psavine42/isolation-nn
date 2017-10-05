
import random
from game_agent import IsolationPlayer, SearchTimeout
import timeit
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import loader as serial_util
import aindnn.slnn as inn
import numpy as np
from functools import partial
import operator

p_inf = float("inf")
n_inf = float("-inf")



def game_to_input(game):
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

    s_r, s_c = game.get_player_location(game.active_player)
    L_pos_self[s_r][s_c] = 1

    o_r, o_c = game.get_player_location(game.inactive_player)
    L_pos_opp[o_r][o_c] = 1

    L_closed = L_closed - L_open - L_pos_opp - L_pos_self
    return np.stack([L_pos_self, L_pos_opp, L_open, L_ones, L_legal, L_closed, L_zeros], axis=0)



def fresh_stack(x, y):
    #x = game.height
    #y = game.width
    L_pos_self, L_zeros, L_pos_opp, L_legal, L_closed, = [np.zeros((x, y), dtype=int) for _ in range(5)]
    L_ones, L_open  = [np.ones((x, y), dtype=int) for _ in range(2)]
    return np.stack([L_pos_self, L_pos_opp, L_open, L_ones, L_legal, L_closed, L_zeros], axis=0)

def fresh_stack_gpu(x, y):
    stack = fresh_stack(x, y)
    return torch.from_numpy(stack).float().unsqueeze(0).cuda(0)

def fresh_stack_flat(x, y):
    stack = fresh_stack(x, y)
    return torch.from_numpy(stack).float().cuda(0)


def one_hot_move_to_index(game, move_index):
    #neural net returns a one hot vector.
    #transform to (row, col) move
    r = move_index // game.width
    c = move_index - (r * game.width)
    return r, c

#def nn_score_(value_net, game, history, player):
    #move, _ = serial_util.process_one_file(history, player)



class NNPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
        search. You must finish and test this player to make sure it properly uses
        minimax to return a good move before the search time limit expires.
        """
    def __init__(self, policy, value_net=None,
                 mode='policy_only',
                 name='def',
                 silent = True,
                 replay=False,
                 gpu=True,
                 h=7, w=7):
        super(self.__class__, self).__init__()

        #nn_player = NNPlayer(policy_NET, value_net=value_NET, mode=mode)
        if policy is None:
            self.model = inn.SLNet(k=64, chkpt='./outputx/policy.pkl').cuda(0)
        else:
            self.model = policy

        if value_net is None:
            self.value_net = torch.load('./outputx/checkpoints/200-Final.pkl')
        elif value_net is False:
            self.value_net = None
        else:
            self.value_net = value_net

        self.name = name
        self.mode = mode
        # this is an efficienter index of position_stack
        #self.positions_stack = None
        # these hold previous moves to add update
        self.last_opp_pos = None
        self.last_self_pos = None
        self.unroll_time = 50
        # for RL training
        self.saved_actions = []
        self.rewards = []
        self.replay = replay
        self.silent = silent
        # to do implement numpy version for tournament
        self.gpu = gpu
        self.illegal_hist = 0
        self.game_history = []


        #initialize empty game stack
        self.positions_stack = fresh_stack_gpu(h, w) #fresh_stack_gpu(h, w)
        #warm up the gpu, or else it times out on first move lol
        #print("stack built")
        self.__run_policy(self.positions_stack)
        if value_net:
            self.__run_value(self.positions_stack)
        #print("warmup complete")


    def get_move(self, game, time_left):
        self.time_left = time_left
        #
        opp_last_move = game.get_player_location(game.get_opponent(self))
        if opp_last_move:
            self.game_history.append(opp_last_move)

        if self.random_start:
            return random.choice(game.get_legal_moves())

        best_move = (-1, -1)
        try:
            if self.mode == 'policy_only':
                return self.get_nn_move_simple(game)

            elif self.mode == 'minimax':
                best_move = self.minimax(game, 2)
                self.game_history.append(best_move)
                #print(best_move)
                return best_move

            elif self.mode == 'alphabeta':
                fresh_game = game.copy()
                #print("opponent:", opp_last_move)
                for depth in range(1, game.width * game.height):
                    move = self.alphabeta(fresh_game, depth)
                    if move == (-1, -1):
                        # at depth, it is found the best move is losing.
                        # return best thing so far and hope he doesnt the win
                        return best_move
                    best_move = move

            elif self.mode == 'test_mm':
                move = self.not_alphabeta(game, 2)
                self.game_history.append(move)
                return move
            elif self.mode == 'minimax':
                move = self.minimax(game, 3)
                return move
            else:
                return "mode error"
        except SearchTimeout:
            #print("NN timewarning")
            return best_move
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

    def __update_stack(self, game, stack):

        pass


    def __run_policy(self, positions_stack):
        "interface to nn - can error handle here when no torch"
        return self.model(Variable(positions_stack))

    def __run_value(self, positions_stack):
        "interface to nn - can error handle here when no torch"
        return self.value_net(Variable(positions_stack))

    def get_nn_move_simple(self, game):
        "test method to return a move from NN prediction - no guardrails."
        #save stack for latter
        moves = game.get_legal_moves()
        self.update_stack(game, moves)

        #run neural net
        logits = self.__run_policy(self.positions_stack)

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


    def terminal(self, moves, depth):
        """terminal search function
            """
        return any(((depth <= 0), (not moves)))

    def to_cuda(self, eval_stacks):
        choices = self.value_net(Variable(torch.from_numpy(np.asarray(eval_stacks)).float().cuda(0)))
        return choices.squeeze().data.cpu().numpy().tolist()

    def nn_score(self, game, ref):
        self.num_calcs += 1
        output = self.to_cuda([game_to_input(game)])
        self.saved_actions.append({output[0]: game.history})
        return output[0]

    def get_move_proposals(self, game):
        pass

    ##############################################################################
    #   MINIMAX
    ##############################################################################
    def min_v(self, game, depth):
        """Doc """
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout(depth)
        if depth == 0: return self.nn_score(game, "min")
        moves = game.get_legal_moves()
        score = 1
        if not moves:
            return score
        #
        for move in moves:
            score = min(score, self.max_v(game.forecast_move(move), depth - 1))
        return score

    def max_v(self, game, depth):
        """Doc """
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()
        if depth == 0: return self.nn_score(game, "max")
        moves = game.get_legal_moves()
        score = -1
        if not moves:
            return score
        #
        for move in moves:
            score = max(score, self.min_v(game.forecast_move(move), depth - 1))
        return score

    def minimax(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()

        best_score = -1
        best_move = None
        for move in game.get_legal_moves():
            score = self.min_v(game.forecast_move(move), depth - 1)
            if self.time_left() < self.unroll_time:
                if not best_move:
                    return move
                return best_move
            if score >= best_score:
                best_score = score
                best_move = move
        return best_move

    ##############################################################################
    #   ALPHABETA
    ##############################################################################
    def stat_active(self, pr, game, mv=''):
        if not self.silent:
            print("{}, active:{}, loc:{}, mv:{}".format(
                pr, game.active_player, game.get_player_location(game.active_player), mv))

    def max_ab(self, game, depth, alpha, beta):
        """max value function"""
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()
        if depth == 0: return self.nn_score(game, "max_ab")
        moves = game.get_legal_moves()
        v = -1
        if not moves: return v
        self.stat_active("start max_ab", game)

        for move in moves:
            self.stat_active("start max_ab", game, mv=move)
            v = max(v, self.min_ab(game.forecast_move(move), depth - 1, alpha, beta))
            if v >= beta: return v
            alpha = max(alpha, v)
        return v

    def min_ab(self, game, depth, alpha, beta):
        """min value function"""
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()
        moves = game.get_legal_moves()
        if self.terminal(moves, depth): return self.nn_score(game, "min_ab")
        v = 1
        self.stat_active("start min_ab", game)
        for move in moves:
            self.stat_active("min_ab", game, mv=move)
            v = min(v, self.max_ab(game.forecast_move(move), depth - 1, alpha, beta))
            if v <= alpha: return v
            beta = min(beta, v)
        return v

    def alphabeta(self, game, depth, alpha=-1, beta=1):
        "Alpha beta"
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()
        best_move, best_score = (-1, -1), -1
        moves = game.get_legal_moves()

        self.stat_active("start main:", game)
        for move in moves:
            self.stat_active("main:", game, mv=move)
            score = self.min_ab(game.forecast_move(move), depth - 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        return best_move


    def not_alphabeta(self, game, depth, alpha=-1, beta=1):
        best_move = (-1, -1)
        best_score = n_inf
        moves = game.get_legal_moves()
        #print(moves)
        #self.update_stack(game, moves)
        eval_stacks = []
        move_choices = []
        positions_at_1 = {}
        #idx = 0
        start = timeit.timeit()
        for move1 in moves:
            copygame = game.forecast_move(move1)
            legal_opp_moves = copygame.get_legal_moves()
            ems = []
            for move2 in legal_opp_moves:
                fgame = copygame.forecast_move(move2)
                ems.append(game_to_input(fgame))
                #idx += 1
                #move_choices.append([move1, move1])
            if ems:
                choices = self.to_cuda(ems)
                positions_at_1[move1] = 1 * sum(choices) / float(len(choices))

        print(positions_at_1)

        if len(positions_at_1) > 0:
            best_move = max(positions_at_1, key=positions_at_1.get)

        print(best_move)
        return best_move

        #score = self.min_v(game.forecast_move(move), depth - 1, alpha, beta)
        #if score > best_score:
            #best_score = score
            #best_move = move
        #alpha = max(alpha, score)











