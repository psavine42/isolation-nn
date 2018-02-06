import numpy as np
import torch
import random
from torch.utils import data
import glob
import os
# import envs.isolation.loader as serial_util

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
    hist = file[:to] if to else file
    g = game.copy()
    for move in hist:
        g = g.forecast_move(move)
    return g


class CoupledLoader(data.Dataset):
    def __init__(self, root, ext=".npz", pct_train=0.95, vl=False, train=True):
        self.root = root
        self.ext = ext
        self.files = {}
        self.base_files = {}
        self.is_value_training = vl
        files_root = sorted(glob.glob(self.root + '*' + self.ext))

        # theyre in order, so this is how i role
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
            print('sl')
            # so i did not save the frickin win index in my position files
            # cuz i hadnt read that far in alphago paper
            ret = load_npz(self.files[index], ['position', 'move'])
            position = torch.from_numpy(ret['position']).float()
            moves = torch.from_numpy(ret['move']).long().squeeze()
            return position, moves
        else:
            print('rl')
            ret = load_npz(self.files[index], ['hist', 'size', 'duration'])
            length = len(ret['hist'])
            if length < 8:
                return self.__getitem__(index + 1)
            move_idx = random.randint(5, length)
            player = random.randint(0, 1)

            file_def, moves = [], [] #serial_util.process_one_file(ret, player, stop_at=move_idx)
            position = torch.from_numpy(file_def[-1]).float()
            print(self.files[index])
            winner = length % 2 == 0
            if player == winner:
                return position, torch.Tensor([1]).float()
            else:
                return position, torch.Tensor([-1]).float()


if __name__ == '__main__':
    homedir = os.path.expanduser('~')
    data_dir = homedir + '/data/isolation/positions/'
    print(data_dir)

    dataset1 = CoupledLoader(data_dir, vl=False, train=False)
    dataset2 = CoupledLoader(data_dir, vl=False, train=False)
    itm = dataset1.__getitem__(0)
    print(itm)









