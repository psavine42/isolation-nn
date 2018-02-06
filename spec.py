import unittest
import torch
import torch.nn as nn
from torch.autograd import Variable
from envs.isolation.isolation import LTBoard
from aindnn.MCTS.mcts import MCTS
from aindnn.MCTS.nodes import Node
from aindnn.MCTS.utils import hash_fn, to_input, to_mem
import os, sys, math
from visdom import Visdom
from aindnn.Trainers.trainers import MiniBatch, Trainer, SelfPlay
from multiprocessing import Queue, Process, JoinableQueue
from agents.zero_agent import Agent
import aindnn.slnn as models
import aindnn.zero as zero
import argparse


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        inputs2 = Variable(torch.rand([49]))
        p = self.softmax(inputs2)
        v = torch.rand([1])
        return p.data, v


class Testmcts(unittest.TestCase):
    def setUp(self):
        self.game = LTBoard()
        self.nn = Dummy()
        # self.Agent()
        self.MC = MCTS(self.nn, self.game, tao=1)
        print('init')

    def test_cache(self):
        pass

    def path_is_good(self, s_node):
        root = False
        init_node = s_node
        while root == False:
            t_node = self.MC._tree[init_node.hash]
            if t_node.parent is None:
                assert self.MC._path[0] == t_node.hash
                root = True
                break
            action_node = t_node.parent
            print(action_node)
            assert action_node.parent is not None
            init_node = action_node.parent

    def test_mcts2(self):
        s_node0 = Node(self.game.state)
        assert s_node0.expanded is False
        num_turns = 2
        self.MC.initial = s_node0.hash
        s_node1 = self.MC.forward(s_node0)

        s_node2 = self.MC.forward(s_node1)
        s_node3 = self.MC.forward(s_node2)

        s_node4 = self.MC.forward(s_node3)
        s_node5 = self.MC.forward(s_node4)

        s_node6 = self.MC.forward(s_node5)

        print('---------------------\n')
        print(self.MC)
        assert len(self.MC.tree) == num_turns
        assert self.MC.path_len() == num_turns
        assert isinstance(s_node2, Node)
        self.path_is_good(s_node2)
        print(s_node0.state)
        print(s_node1.state)
        print(s_node2.state)

    def test_base_move(self):
        """

        :return:
        """

    def test_mcts_self_play(self):
        pass


class testLT(unittest.TestCase):
    def setUp(self):
        self.gm = LTBoard()

    def test_statics(self):
        gm = LTBoard()
        new = gm.apply_move(2)
        board = new.apply_move(13)
        assert board.state[-1] == board.active_player == 1
        cloned = LTBoard(state=LTBoard.smove(board.state, 7))
        # print(cloned)
        assert cloned.state[7] == 1
        assert cloned.state[2] == -1
        # assert cloned.state[-1] == cloned.active_player == 2

    def test_wtv(self):
        state = self.gm.state
        assert len(state) == 50
        assert state[-1] == 1

    def test_moves(self):
        """apply some moves to a board"""
        # move 1
        new = self.gm.apply_move(2)
        assert new.state[2] == 1
        assert new.state[-1] == 2
        # move 2
        new2 = self.gm.apply_move(13)
        assert new2.state[13] == 2
        assert new2.inactive_player == 2
        assert new2.state[-1] == new2.active_player == 1
        assert new2.move_count == 2

    def test_back(self):
        gm0 = LTBoard()
        gm1 = gm0.copy_move(2)
        gm2 = gm1.copy_move(5)
        gmb1 = gm2.backward()
        gmb0 = gmb1.backward()
        gmbn0 = gmb0.backward()
        assert gmb1 == gm1
        assert gmb0 == gm0
        assert gmbn0 == gm0

    def test_feature(self):
        gm0 = LTBoard()
        gm1 = gm0.copy_move(2)
        gm2 = gm1.copy_move(5)
        inputs = torch.from_numpy(gm2.to_feature()).float()

        assert type(inputs) == torch.FloatTensor
        print(inputs)


class testNN0(unittest.TestCase):
    """
    Basic run-through of making moves in a board, and sending to NN for eval
    """
    def setUp(self):
        self.NN_0 = models.ZeroNet(channels=7, n_blocks=2, verbose=False)

    def test_res_tower(self):
        """Test that inputs go through the network with no error,"""
        gm0 = LTBoard()
        gm1 = gm0.copy_move(2)
        gm2 = gm1.copy_move(5)
        inputs = torch.from_numpy(gm2.to_feature()).float().unsqueeze(0)
        p, v = self.NN_0(Variable(inputs))
        assert list(v.data.size()) == [1, 1]
        assert list(p.data.size()) == [1, 50]

    def test_res_batches(self):
        gm0 = LTBoard()
        gm1 = gm0.forward(2)
        gm2 = gm1.forward(5)
        inputs1 = torch.from_numpy(gm0.to_feature()).float().unsqueeze(0)
        inputs2 = torch.from_numpy(gm1.to_feature()).float().unsqueeze(0)
        inputs3 = torch.from_numpy(gm2.to_feature()).float().unsqueeze(0)
        inputs = torch.cat([inputs1, inputs2, inputs3], 0)
        p, v = self.NN_0(Variable(inputs))
        assert list(v.data.size()) == [3, 1]
        assert list(p.data.size()) == [3, 50]

    def test_add_noise(self):
        gm0 = LTBoard()
        gm1 = gm0.forward(2)
        inputs1 = torch.from_numpy(gm1.to_feature()).float().unsqueeze(0)
        outs = self.NN_0(Variable(inputs1))
        p, v = to_mem(outs)
        noised = Node.dirichlet_noise(p)
        assert 10e-5 >= math.fabs(1 - sum(p))
        assert 10e-5 >= math.fabs(1 - sum(noised))


class PlayAgent(unittest.TestCase):
    def setUp(self):
        n = 5
        parser = argparse.ArgumentParser()
        # parser.add_argument('--epochs', type=int, default=2000, help='[]')
        self.args = parser.parse_args([])

        print(self.args)
        self._visdom = Visdom()
        test_loc = './test/chkpt/'
        self.nn1 = models.ZeroNet(channels=7, n_blocks=2)
        self.nn2 = models.ZeroNet(channels=7, n_blocks=2)
        self.nn3 = models.ZeroNet(channels=7, n_blocks=2)
        self.agent1 = Agent(nn=self.nn1, num_sims=n, verbose=False, chkpt_dir=test_loc)
        self.agent2 = Agent(nn=self.nn2, num_sims=n, verbose=False)
        self.agent3 = Agent(nn=self.nn3, num_sims=n, verbose=False)
        self.tasks = JoinableQueue()
        self.best = JoinableQueue()
        self.challengers = JoinableQueue()

    def make_moves(self, agent):
        init_state = Node(LTBoard(), None)
        agent.run_search(init_state)
        agent.show_tree()

        print('\n-------------------')
        # print('init_state', init_state)

        assert init_state == agent.mcts.initial
        assert init_state.hash == agent.mcts.initial.hash
        final_action, pi = agent.mcts.get_best_action(init_state)
        assert pi > 0

        new_state = final_action.child
        agent.run_search(new_state)
        assert new_state == agent.mcts.initial
        assert new_state.hash == agent.mcts.initial.hash
        final_action2, pi = agent.mcts.get_best_action(new_state)
        new_state2 = final_action2.child
        print(init_state)
        print(new_state)
        print(new_state2)
        # print('final_action', final_action)

    def test_move_cpu(self):
        self.make_moves(self.agent1)

    def test_move_cuda(self):
        nn = models.ZeroNet(channels=7, n_blocks=2).cuda()
        agent3 = Agent(nn=nn, num_sims=10, verbose=True)
        self.make_moves(agent3)

    def test_problem_terminal(self):
        """a problem can play itself"""
        game = LTBoard()
        while game.terminal is False:
            moves = game.get_legal_moves()
            game = game.forward(moves[0])
        assert game.terminal is True
        assert game.state[-1] == 1 or game.state[-1] == 2

    def test_play_self(self):
        sp = zero.SelfPlay(LTBoard, num_games=1, num_sims=100)
        data, batches = sp.forward(self.agent1)
        [print(d) for d in data]
        assert len(data[-1][0]) == 50
        assert len(data[-1]) == 3
        assert type(data[-1][1][0]) == float
        assert data[-1][2][0] == 1. or data[-1][2][0] == -1.

    def test_optim_int(self):
        self.args.verbose = True
        self.args.show_every = 1
        self.args.viz = self._visdom
        self.args.batch_size = 8
        self.args.best_model = ''

        spl = zero.SelfPlay(LTBoard,
                            self.args,
                            in_queue=self.best,
                            out_queue=self.tasks,
                            num_games=20,
                            num_sims=202)
        opt = zero.OptimizeProc(self.args,
                                in_queue=self.tasks,
                                out_queue=self.challengers)
        evl = zero.Evaluator(LTBoard,
                             self.args,
                             in_queue=self.challengers,
                             out_queue=self.best,
                             num_sims=202)

        for n in [spl, opt, evl]:
            n.daemon = True
            n.start()

        mb = MiniBatch('agent', self.agent2, None)
        mb2 = MiniBatch('agent', self.agent1, None)
        mb3 = MiniBatch('agent', self.agent3, None)

        self.tasks.put(mb2)
        self.challengers.put(mb3)
        self.best.put(mb)

        spl.join()
        evl.join()
        opt.join()

    def test_evaluator(self):
        pass

    def test_save_load_agent(self):
        self.make_moves(self.agent1)
        self.agent1.save(iter=1)
        nn_path = os.path.join(self.agent1.chkpt_dir, self.agent1.time, str(1), 'model.pkl')
        assert os.path.exists(nn_path)
        loaded = torch.load(nn_path)
        self.agent1.show_tree()
        loaded.show_tree()
        # print('\n+++++++')
        assert self.agent1.mcts._tree == loaded.mcts._tree


if __name__ == '__main__':
    unittest.main()