"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest
import os
import numpy as np
import torch
import controller as cntr
from importlib import reload
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import aindnn.slnn as inn
import glob
from torch.utils import data
import timeit
from isolation import *
from dataloaders import CoupledLoader
import loader as serial_util
import torch.nn.functional as F
from sample_players import (RandomPlayer, open_move_score,
                            improved_score, center_score)
from game_agent import (MinimaxPlayer, AlphaBetaPlayer, custom_score,
                        custom_score_2, custom_score_3)
from nn_players import *
import glob
import argparse






epsilon = 1e-6

def log_loss(y_true, y_pred):
    '''Keras 'loss' function for the REINFORCE algorithm, where y_true is the action that was
    taken, and updates with the negative gradient will make that action more likely. We use the
    negative gradient because keras expects training data to minimize a loss function.
    '''
    return -y_true * torch.log(torch.clip_gradient(y_pred, epsilon, 1.0 - epsilon))


class NNTEST(unittest.TestCase):

    def test_move(self):
        #setup networks
        policy = './outputx/checkpoints/model_nn_small_0.006865953999977137_2.pkl'
        

        model_old = './outputx/checkpoints/model_nn_small_0.006865953999977137_0.pkl'
        model = torch.load(policy)
        #print("loaded", model)
        #model = inn.PolicyNet(model)

        parser = argparse.ArgumentParser(description='Hyperparams')
        parser.add_argument('--dd', nargs='?', type=str, default='train', help='[]')
        args = parser.parse_args()
        #player2 = cntr.loadNNPlayer(model_old)
        player2 = RandomPlayer()
        ##
        args.epochs = 100
        args.lr = 0.0005
        args.self_play_dir = '/home/psavine/data/isolation/selfplay/'
        args.pool_dir = './outputx/pool/'
        args.out_pool_dir = './outputx/out_pool/'
        args.start_model = 'model_nn_small_0.006865953999977137_2.pkl'
        #cant fugure out this reinforce with pytorch stuff.
        #why am i adding a stochastic function when i can just do gradients?


        #new_model, win_pct = cntr.play_game_questionable(policy, player2, args, save_mode=False)
        new_model, win_pct = cntr.play_game_rl(model, player2, 'random', args, save_mode=True)
        #print(new_model, win_pct)

        args.matches = 3
        #controller.play_game(policy, player_old_nn, criterion)
        #cntr.tournament_schedule(args)
        #


        #print(optimizer.)
        # print("ACTIONS", len(saved_actions), saved_actions)
        #print("Illegal moves {} out of {}".format(
            #game.get_player_N(2).illegal_hist, game.move_count//2))


        #rewards = plc.rewards
        #print(rewards[0])
        #rewards = rewards.data
        #print(rewards[0])
        #frwards = [r  for r in rewards]
        #rewards = []
        #for r in policy.rewards:
            #R = r + args.gamma * R
            #rewards.append(r.data )
        #

        #print(rewards)
        #https://github.com/Rochester-NRT/RocAlphaGo/blob/develop/AlphaGo/training/reinforcement_policy_trainer.py

        #rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        #rewards = torch.Tensor(rewards)
        #rewards = (rewards - rewards.mean()) / rewards.std()

        #for action, logits in zip(policy.saved_actions, policy.rewards):
            #print("action", action)
            #print("reward", r )
            #logits.
            #lg = logits.Softmax()
            #print(logits.data)
            #print(lg)
            #action.reinforce(lg)

        #optimizer.zero_grad()
        #autograd.backward(saved_actions, [None for _ in saved_actions])
        #optimizer.step()
        #print(rewards)
        #assert(z_T == 1)

    def test_load_res(self):
        positions = '/home/psavine/data/isolation/positions/'
        games =      '/home/psavine/data/isolation/games/'
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
        files_root = glob.glob(games + nm + '*.npz' )[0]
        print(files_root)
        npz = np.load(sample)
        ret = dict((name, npz[name].copy()) for name in ['position', 'move'])
        npz.close()
        #print(ret)
        npz = np.load(files_root)
        game_hist = dict((name, npz[name].copy()) for name in ['hist'])
        npz.close()
        hist_l = len(game_hist['hist'])
      
        #move = np.where(hist == ret['move'])
        print("-----------------")
        r = ret['move'][0] // 7
        c = ret['move'][0] - (r * 7)

        move_id = np.where(game_hist['hist'] == [r,c])
        print("total_moves", len(game_hist['hist']))
        print(move_id)
        #0 -> player 1
        bl = hist_l % 2 == 0 #even, player 1 is winner
        #print(nm)








if __name__ == '__main__':
    unittest.main()
