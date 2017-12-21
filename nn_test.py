"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.

todo - cleanup paths. finish move selection
"""

import unittest
import controller as cntr
import torch.nn as nn
from isolation import *
from problem.dataloaders import *
from problem import loader as serial_util
from sample_players import (RandomPlayer, improved_score)
from agents.game_agent import (MinimaxPlayer, AlphaBetaPlayer)
from agents.nn_players import *
import glob, argparse

test_npz = './testdata/best_in-r-lossm13p2.npz'
model_old = './outputx/checkpoints/model_nn_test_0.008578838998801075_2.pkl'
epsilon = 1e-6

def dotimed(net, args):
   pass


def print_game(g, w, h, o):
    print("\nWinner: {}\nOutcome: {}".format(w, o))
    print(g.to_string())
    print("Move history:\n{!s}".format(h))
    print("num moves", g.move_count)


def log_loss(y_true, y_pred):
    """
        Keras 'loss' function for the REINFORCE algorithm, where y_true is the action that was
        taken, and updates with the negative gradient will make that action more likely. We use the
        negative gradient because keras expects training data to minimize a loss function.
        """
    return -y_true * torch.log(torch.clip_gradient(y_pred, epsilon, 1.0 - epsilon))


class NNBuild(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(NNBuild, self).__init__(*args, **kwargs)
        self.value_loc = './outputx/checkpoints/200-Final.pkl'
        self.policy_loc = './outputx/checkpoints/model_nn_test_0.008578838998801075_2.pkl'
        self.game1 = '/home/psavine/data/isolation/games/0.006683826970402151.npz'

    def setup_nn(self):
        model = torch.load(model_old)
        policy = inn.PolicyNet(model).cuda(0)
        pos = load_npz(test_npz, ['position', 'move', 'result'])
        print(policy)

        criterion = nn.MSELoss().cuda(0)
        target = Variable(torch.Tensor([[-1]]).float().cuda(0))

        inputs = torch.from_numpy(pos['position']).float().unsqueeze(0).cuda(0)
        inputs = Variable(inputs)
        print(inputs.size())
        pred = policy(inputs)

        print("pred", pred)
        print("target", target)
        loss = criterion(pred, target)
        print("loss", loss)

    def load_game(self, game, p=1):
        hist = load_npz(game, ['hist', 'size', 'duration'])
        win_defs, _ = serial_util.process_one_file(hist, p) # player2
        return win_defs, hist['hist']

    def value_net(self):
        """Testing for no error on basic load and run of game"""
        policy = torch.load(self.value_loc)
        # hist1 = load_npz(test_npz, ['position', 'move', 'result'])['hist']
        hist_b = load_npz(self.game1, ['hist', 'size', 'duration'])
        hist2 = hist_b['hist']
        player1 = RandomPlayer()
        player2 = RandomPlayer()

        nn_player = NNPlayer(policy) # set as p2 for win, 1 for loss
        game = Board(player1, player2)

        game_at = play_forward(game, hist2, to=10)
        game_f = play_forward(game, hist2)

        print(game_f.to_string())
        print("game length", len(hist2))
        winner = len(hist2) % 2 == 0
        print("p2 wins", winner)
        print("p1 - win", game_f.is_winner(player1))
        print("p2 - win", game_f.is_winner(player2)) # winner
        start = timeit.timeit()
        win_defs, _ = serial_util.process_one_file(hist_b, 1) # player2
        end = timeit.timeit()
        print("time", 1000 * (end - start))
        loss_defs, _ = serial_util.process_one_file(hist_b, 0)

        in_win = Variable(torch.from_numpy(win_defs).float().cuda(0))
        in_loss = Variable(torch.from_numpy(loss_defs).float().cuda(0))
        eval_win = policy(in_win)
        eval_loss = policy(in_loss)

        print("evals for winner @ move")
        print(eval_win)
        print("evals for loser @ move")
        print(eval_loss)
        """Testing for no error on basic load and run of game"""

    def setup_NNplayer(self, mode='test_mm'):
        value_NET = torch.load('./outputx/checkpoints/200-Final.pkl')
        policy_NET = inn.SLNet(k=64, chkpt='./outputx/policy.pkl').cuda(0)
        nn_player = NNPlayer(policy_NET, value_net=value_NET, mode=mode)
        return nn_player

    def test_ab_value_net(self):
        """testing history and get move stuff"""
        move_idx = 5

        # load neural networks
        nn_player = self.setup_NNplayer()
        _, hist = self.load_game(self.game1, p=1)
        print("game length", len(hist))

        # create game
        opponent = AlphaBetaPlayer(score_fn=improved_score)
        game = Board(opponent, nn_player)
        game_at = play_forward(game, hist, to=move_idx)
        print("active player", game_at.active_player)

        print(game_at.to_string())
        start = timeit.timeit()
        move = nn_player.alphabeta(game_at, 2)
        end = timeit.timeit()
        print("time for 1 run", 1000 * (end - start))
        print("final-move", move)

    def game_to_input(self):
        move_idx = 11
        win_defs, hist = self.load_game(self.game1, p=1)
        opponent = AlphaBetaPlayer(score_fn=improved_score)
        #nn
        value_NET = torch.load('./outputx/checkpoints/200-Final.pkl')
        policy_NET = inn.SLNet(k=64, chkpt='./outputx/policy.pkl').cuda(0)
        nn_player = NNPlayer(policy_NET,
                                value_net=value_NET,
                                mode='alphabeta', move_strategy='nn')

        # setup board
        game = Board(opponent, nn_player)
        game_at = play_forward(game, hist, to=move_idx)
        # sanity checks
        print(game_at.to_string())
        print(game_at.history)
        assert nn_player == game_at.active_player
        assert len(game_at.history) == move_idx
        ##
        time_millis = lambda: 1000 * timeit.default_timer()
        move_start = time_millis()
        time_left = lambda: 1000 - (time_millis() - move_start)

        legals = game_at.get_legal_moves()
        moves = nn_player.get_move(game_at, time_left)
        print(legals)
        print(moves)

    ###################################################################

    def play_value_net(self):
        nn_player = self.setup_NNplayer(mode='test_mm')
        opponent = MinimaxPlayer(score_fn=improved_score)
        game = Board(opponent, nn_player)
        w, h, o = game.play()
        print_game(game, w, h, o)

    def minimax(self):
        nn_player = self.setup_NNplayer(mode='minimax')
        opponent = MinimaxPlayer(score_fn=improved_score)
        game = Board(opponent, nn_player)
        w, h, o = game.play_sl()
        print_game(game, w, h, o)
        print("nn_saved:", nn_player.game_history)
        print("nn_calcs:", nn_player.num_calcs)
        print("mm_calcs:", opponent.num_calcs)

    def alphabeta(self):
        self.minimax()
        print("-----------------------------------")
        nn_player = self.setup_NNplayer(mode='alphabeta')
        opponent = AlphaBetaPlayer(score_fn=improved_score)
        game = Board(opponent, nn_player)
        w, h, o = game.play_sl()
        print("winner", w)
        print("nn_calcs:", nn_player.num_calcs)
        print("mm_calcs:", opponent.num_calcs)

    def alphabeta_avg(self):
        self.minimax()
        print("-----------------------------------")
        nn_player = self.setup_NNplayer(mode='alphabeta')
        opponent = AlphaBetaPlayer(score_fn=improved_score)
        game = Board(opponent, nn_player)
        w, h, o = game.play_sl()
        print_game(game, w, h, o)
        print("nn_saved:", nn_player.game_history)
        print("nn_calcs:", nn_player.num_calcs)
        print("mm_calcs:", opponent.num_calcs)

    ###################################################################

    def test_dict(self):
        """test loading network to different class and run some metrics"""
        policy_NET = inn.SLNet(k=64, chkpt='./outputx/policy.pkl').cuda(0)
        value_NET = torch.load(self.value_loc)

        win_defs, hist = self.load_game(self.game1)
        l_defs, hist = self.load_game(self.game1, p=0)
        # warmup
        position = Variable(torch.from_numpy(win_defs[6]).unsqueeze(0).float().cuda(0))
        x = policy_NET(position)
        n = value_NET(position)
        print("even move =", n)
        n = policy_NET(position)
        n = value_NET(position)

        # single run time
        start = timeit.timeit()
        position = Variable(torch.from_numpy(win_defs[5]).unsqueeze(0).float().cuda(0))
        pl = policy_NET(position)
        value = value_NET(position)
        print("even move =", value)
        end = timeit.timeit()
        print("time for 1 run", 1000 * (end - start))

        # n runs time
        start1 = timeit.timeit()
        position = Variable(torch.from_numpy(win_defs).float().cuda(0))
        pl = policy_NET(position)
        value = value_NET(position)
        end1 = timeit.timeit()
        print("time for n runs", 1000 * (end1 - start1))

    def test_conv(self):
        conv1 = nn.Conv2d(7, 49, 1, stride=1, bias=True)
        lin = nn.Linear(49, 256)
        input = Variable(torch.randn(128, 20))
        m = nn.Linear(20, 30)
        print(input.size())
        output = m(input)
        print(output.size())


class NNTEST(unittest.TestCase):
    def test_move(self):
        # setup networks
        policy = './outputx/checkpoints/model_nn_small_0.006865953999977137_2.pkl'

        model_old = './outputx/checkpoints/model_nn_small_0.006865953999977137_0.pkl'
        model = torch.load(policy)

        parser = argparse.ArgumentParser(description='Hyperparams')
        parser.add_argument('--dd', nargs='?', type=str, default='train', help='[]')
        args = parser.parse_args()

        player2 = RandomPlayer()
        ##
        args.epochs = 100
        args.lr = 0.0005
        args.self_play_dir = '/home/psavine/data/isolation/selfplay/'
        args.pool_dir = './outputx/pool/'
        args.out_pool_dir = './outputx/out_pool/'
        args.start_model = 'model_nn_small_0.006865953999977137_2.pkl'
        # cant fugure out this reinforce with pytorch stuff.
        # why am i adding a stochastic function when i can just do gradients?
        new_model, win_pct = cntr.play_game_rl(model, player2, 'random', args, save_mode=True)

    def test_value_loader(self):
        pass

    def test_load_winner(self):
        game = './testdata/0.006985372980125248.npz'
        pos = './testdata/0.006985372980125248m16p1.npz'

        gamez = load_npz(game, ['hist'])['hist']
        poz = load_npz(pos, ['position', 'move'])

        r, c = serial_util.one_hot_move_to_index(7, poz['move'][0])
        print(poz['move'], " idx -> location:", r, c)
        print("total_moves", len(gamez))
        #even, player 1 is winner

        def mover_is_winner(game_hist, move):
            game_hist = game_hist.tolist()
            if move not in game_hist:
                print("error")
                return None
            move_idx = game_hist.index(move)
            winner = len(game_hist) % 2 == 0
            mover = move_idx % 2 == 0
            print("move_num", move_idx)
            print("is_win", mover == winner)
            return mover == winner

        # assert mover_is_winner(null_history, [5, 3]) is None
        assert mover_is_winner(gamez, [4, 2]) is True
        assert mover_is_winner(gamez[:4], [4, 2]) is True
        assert mover_is_winner(gamez[5:], [4, 2]) is None
        assert mover_is_winner(gamez[:5], [4, 2]) is False

    def test_load_res(self):
        positions = '/home/psavine/data/isolation/positions/'
        games = '/home/psavine/data/isolation/games/'
        sample = '/home/psavine/data/isolation/positions/0.006985372980125248m16p1.npz'
        name = os.path.splitext(os.path.basename(sample))[0]

        snm = name.split('m')
        mv, pl = snm[1].split('p')
        print(pl)
        nm = snm[0]
        print(nm)
        #valid_data = CoupledLoader('/home/psavine/data/isolation/positions/',
                                   #train=False,
                                   #vl = True,
                                   #base_root='/home/psavine/data/isolation/games/')
        #print(glob.glob(positions + nm + '*.npz' ))
        files_root = glob.glob(games + nm + '*.npz')[0]
        print(files_root)

        ret = load_npz(sample, ['position', 'move'])
        game_hist = load_npz(files_root, ['hist'])
        hist_l = len(game_hist['hist'])

        #move = np.where(hist == ret['move'])
        print("-----------------")
        r = ret['move'][0] // 7
        c = ret['move'][0] - (r * 7)

        move_id = np.where(game_hist['hist'] == [r, c])
        print("total_moves", len(game_hist['hist']))
        print(move_id)
        #0 -> player 1
        bl = hist_l % 2 == 0 #even, player 1 is winner


if __name__ == '__main__':
    unittest.main()
