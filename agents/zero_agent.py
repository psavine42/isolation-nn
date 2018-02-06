import torch
import torch.nn as nn
from torch.autograd import Variable
from aindnn.MCTS.mcts import MCTS
from collections import namedtuple
from aindnn.MCTS.utils import to_mem
from time import time
import os

Thetas = namedtuple("Theta", ("mcts", "nn", "timestamp"))


class Agent(nn.Module):
    """
    container for network and mcts
    """
    def __init__(self,
                 nn=None,
                 mc=None,
                 tao=1,
                 chkpt_dir=None,
                 num_sims=1600,
                 verbose=False):
        super(Agent, self).__init__()
        tm = time()
        self.time = '{0:.0f}'.format(tm)
        self.nn = nn

        self.num_sims = num_sims
        self.verbose = verbose
        self.chkpt_dir = chkpt_dir
        # tao is 1 for first ~10% of moves
        self.tao = 1
        self.training = True
        self.mcts = MCTS(self.nn, tao=tao, verbose=verbose) \
            if mc is None else mc

    @property
    def terminal(self):
        return self.env.terminal

    @property
    def is_cuda(self):
        return next(self.nn.parameters()).is_cuda

    def get_latest_version(self):
        return Thetas(nn=self.nn, mcts=self.mcts, timestamp='')

    def run_search(self, init_state):
        """
        Fig 2 a-c from paper - Search Algorithm
        the child node corresponding to the final action
        becomes the root of the new search tree.
        :return: None -
        """
        i = 0
        # if the node is not in the tree, it is
        # if init_state.hash in self.mcts.
        s_node = init_state
        self.mcts.initial = init_state

        # fig 2a 'Forward'
        # while unevaluated_state_reached is False:
        while i < self.num_sims:
            new_s_node = self.mcts.forward(s_node)
            if new_s_node.hash == s_node.hash:

                # fig 2b 'Expand and Evaluate'
                # send state to nn for evaluation
                nn_input = self.to_input(s_node)
                nn_out = self.nn(nn_input)
                p_and_v = to_mem(nn_out)
                expanded = s_node.expand(p_and_v)
                self.mcts.add_state(expanded)

                # fig 2c 'Backup'
                # send back to MCTS to update its values
                self.mcts.backward(expanded)
                s_node = self.mcts.initial

            elif new_s_node.terminal is True:
                s_node = self.mcts.initial
                i += 1
            else: # continue search
                s_node = new_s_node
                i += 1
        return

    def log(self, *msg):
        if self.verbose is True:
            print(*msg)

    def to_input(self, s_node):
        if self.is_cuda is True:
            return Variable(s_node.to_input().cuda())
        else:
            return Variable(s_node.to_input())

    def save(self, iter=0):
        if self.chkpt_dir is not None:
            nn_path = os.path.join(self.chkpt_dir, self.time, str(iter))
            if not os.path.exists(nn_path):
                os.makedirs(nn_path)
            torch.save(self, nn_path + '/model.pkl')

    def show_tree(self):
        for k, v in self.mcts._tree.items():
            print(k, v)

    def reset(self):
        """
        propogate reset signal to children
        :return:
        """
        self.mcts.reset()

    def backward(self):
        pass

    def forward(self, s_node, tao=None):
        """

        :param init_state: a state (position on a board)
        :return:
        """
        if tao is None:
            tao = self.tao
        if self.training is True:
            self.run_search(s_node)
            # fig 2d 'Play Move'
            # some function to execute action
            # result of move is used to compute loss of nn_theta
            result = self.mcts.get_best_action(s_node, tao)
            # todo discard remainder of tree
            return result

