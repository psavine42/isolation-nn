import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
from collections import namedtuple
from .MCTS import mcts
"""
Feature Planes -> encode a move history

"""

"""
Modules

Conv Block
    conv 256 filters, 3 x 3, stride 1
    Batch Norm
    Relu

Residual Block
1 layer of resnt 

"""


Thetas = namedtuple("Theta", ("MCTS", "NN", "timestamp", "desc"))


class Trainer(object):


class IntialTrain(Trainer):
    """
    Fan
        -Train policy supervised
        -Value predict winner of policy self play
    """

class FanPlayer(Trainer):
    """
    Fan
        -Train policy supervised
        -Value predict winner of policy self play
    """


""" 
Components of the training pipeline 
    Optimizer - thetas 
    evaluator - theta_star <- theta_star vs theta_alpha  
    self-play - games, thetas <-  


 |---> OPTIMIZE------|
 |                   |
 |                  \|/
 |                Evaluate
 |                   |
 |-----SelfPlay<-----|
 
 
"""


class SelfPlay(Trainer):
    """

    lookahead search is inside the training loop

    Attributes
        game (Game Object):
            must implement .score(), .terminal(),
        num_sims (int) :
        num_games (int) :

    """
    def __init__(self,
                 game,
                 network=None,
                 num_sims=1600,
                 num_games=25000):
        self.theta = network

        self.active = 0
        self.resign_threshold = -0.9
        self.num_games = num_games
        self.num_sims = num_sims
        self.game = game

    def look_ahead(self):
        "call network.MCTS (if neeeded) "
        pass

    def mcts_move(self, state):
        """
        FIG 2.
        a. s <- Select edge with maximum Q.
        b. [E...] <- expand leaf node s, and evaluate position with nn
            (P(s, . ), V(s)) = f_theta(s)
            store P values in respective edges.
            this becomes the prior for mcts
        c.

        :return:
        """
        pass

    def play_game(self, challenger=None):
        """
        call network.forward until game is complete.
        when game is complete, return result z

        Need a place to put the mcts and states

        :return: z - game result
        """
        turn, time_steps = 0, []
        # play until terminal is reached
        while not self.game.terminal:
            # get a move
            if challenger is None:
                state, prob, move = self.theta(self.game)
            elif challenger is not None and turn == 1:
                state, prob, move = challenger(self.game)
            else:
                state, prob, move = self.theta(self.game)
            time_steps.append([state, prob])
            # swap active
            turn ^= 1

        even_won = 1 if turn == 0 else -1
        for idx, step in enumerate(time_steps):
            res = even_won if idx % 2 == 0 else -even_won
            step.append(res)

        return time_steps

    def train(self):
        for _ in range(self.num_games):
            time_steps = self.play_game()
        self.backward()
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class Evaluator(SelfPlay):
    """

    Attributes
        threshold (float):
        tao (float): temperature for exploration
    """
    def __init__(self,
                 tao = 10e-5,
                 win_threshold = 0.55,
                 num_games=400,
                 num_sims=1600):
        super(Evaluator, self).__init__()
        self._theta_star = None
        self.num_games = num_games
        self.tao = tao
        self.threshold = win_threshold
        self.num_sims = num_sims

    @property
    def theta_star(self):
        return self._theta_star

    def set_theta(self, thetas):
        self._theta_star = thetas

    def evaluate_new(self, challenger):
        wins = 0
        for _ in range(self.num_games):
            z = self.play_game(challenger)
            # todo something about winning
        if wins / self.num_games > self.threshold:
            self.set_theta(challenger)

    def train(self):
        raise NotImplemented


class Agent(nn.Module):
    """
    container for network and mcts
    """
    def __init__(self, game, num_sims=1600):
        super(Agent, self).__init__()
        self.nn = Network()
        self.mcts = mstc.MCTS()
        self.game = game
        self.num_sims = num_sims
        self.optimizer = torch.optim.SGD(self.nn.parameters())
        self.tao = 1
        self.training = True


    def _backward(self, nn_outputs, z, pi, tao):
        """ loss function and grads
        l = (z - v)^2 - pi^t * log(p)  +  c || theta ||^2
             [ MSE ]   [ Crossentropy ]    [ L2 Loss ]
        """
        if self.training:
            self.optimizer.zero_grad()
        p, v = nn_outputs

        mse = F.mse_loss(z, v)
        cross_ent = F.cross_entropy(pi ** tao, p)
        l2_norm = nn.NLLLoss( )
        loss = mse - cross_ent + l2_norm

        if self.training:
            loss.backward()
            self.optimizer.step()

        return loss.data[0]

    def run_mstc_search(self, state):
        """
        Fig 2 a-c from paper - Search Algorithm
        the child node corresponding to the final action
        becomes the root of the new search tree.
        :return:
        """
        # fig 2a 'Forward'
        unevaluated_state_reached = False
        #while unevaluated_state_reached is False:
        for i in range(self.num_sims):
            state, needs_eval = self.mcts.forward(state)

            if needs_eval:
                # fig 2b 'Expand and Evaluate'
                # send state to nn for evaluation
                value = self.nn(state)
                # fig 2c 'Backup'
                # send back to MCTS to update its values
                self.mcts.update_value(state, value)

            #iter += 0
        # lock thread


        # fig 2d 'Play Move'
        # some function to execute action
        # result of move is used to compute loss of nn_theta



    def forward(self, input_state):
        """

        :param input: a state (position on a board)
        :return:
        """
        self.run_mstc_search(input_state)
        
        return move

################################################

################################################


class Network(nn.Module):
    """
    (p, v) = f_theta(s)
    policy, value = network theta of state

    p [policy]
        -is a vector of move porbabilites for selecting each move (p_a)
        p_a = Pr ( a | s ) probability of action given stat (s)


    """
    def execute_mstc(self, position):
        """
        returns mstc probablities (pi) of playing a move
            (dm-note: these return stronger moves than the raw
                probabilitis 'p' returned by f_theta(s)
                MSTC can be seen as POLICY IMPROVEMENT OPERATOR)

        :recieves grad = False (?)

        :param position:
        :return: pi
        """
        pass

    def PSEUDO_backward(self):
        """
        update parameters theta to make move probabilities (p, v) = f_theta(S)
        more closely match the improved search probabilities and
        self play closer to self play winner (pi, z).
        parameters f_theta^t+1 will theoretically be stronger
        :return:
        """
        pass


    def forward(self, input):
        """one network, no Monte Carlo roll out at train time

        :recieves grad = True (?)
        """



        pass




























