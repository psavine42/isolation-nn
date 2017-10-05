import numpy as np
import torch
import random
from torch.utils import data
import glob
import os
import loader as serial_util
random.seed()


def mover_is_winner(game_hist, move):
    game_hist = game_hist.tolist()
    if move not in game_hist:
        return None
    move_idx = game_hist.index(move)
    winner = len(game_hist) % 2 == 0
    mover = move_idx % 2 == 0
    return mover == winner


def load_npz(fname, kys):
    if not os.path.exists(fname):
        return None
    npz = np.load(fname)
    ret = dict((name, npz[name].copy()) for name in kys)
    npz.close()
    return ret

def play_forward(game, file, to=None):
    if to:
        hist = file[:to]
    else:
        hist = file
    g = game.copy()
    for move in hist:
        g = g.forecast_move(move)
    return g


class CoupledLoader(data.Dataset):
    def __init__(self, root, ext=".npz", pct_train=0.95, vl=False, train=True):
        self.root = root
        self.ext = ext
        self.files = {}
        # self.game_root = base_root
        self.base_files = {}
        self.is_value_training = vl
        files_root = sorted(glob.glob(self.root + '*' + self.ext))
        #theyre in order, so this is how i role
        total = int(len(files_root) * pct_train)
        if train:
            self.files = files_root[:total]
        else:
            self.files = files_root[total:]
        if not self.files:
            raise Exception("No files in" +  self.root)

        print("positions: %d " % (len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        if not self.is_value_training:
            #so i did not save the frickin win index in my position files
            #cuz i hadnt read that far in alphago paper
            ret = load_npz(self.files[index], ['position', 'move'])
            position = torch.from_numpy(ret['position']).float()
            moves = torch.from_numpy(ret['move']).long().squeeze()
            return position, moves
        else:
            ret = load_npz(self.files[index], ['hist', 'size', 'duration'])
            length = len(ret['hist'])
            if length < 8:
                return self.__getitem__(index + 1)
            move_idx = random.randint(5, length)
            player = random.randint(0, 1)

            file_def, moves = serial_util.process_one_file(ret, player, stop_at=move_idx)
            position = torch.from_numpy(file_def[-1]).float()
            print(self.files[index])
            winner = length % 2 == 0
            if player == winner:
                return position, torch.Tensor([1]).float()
            else:
                return position, torch.Tensor([-1]).float()

        """
        else:
            # to retrieve it, I have to find the origional game..
            name = os.path.splitext(os.path.basename(self.files[index]))[0]
            game_hist = load_npz(self.game_root + name.split('m')[0] + '.npz', ['hist'])
            game_hist = game_hist['hist']
            move_loc = serial_util.one_hot_move_to_index(7, ret['move'][0])
            mover_wins = mover_is_winner(game_hist, move_loc)
            print(move_loc)
            if mover_wins is False:
                return position, torch.Tensor([-1]).long()
            elif mover_wins is True:
                return position, torch.Tensor([1]).long()
            else:
                print("WARNING: position error")
                return self.__getitem__(index + 1)
        """

